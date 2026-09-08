"""Whole runs on declared ticks, for the tests that assert timing.

Every link, every controller stage and every decoder here carries a
declared tick value instead of a measured one, so a test's expected tick
is arithmetic over this table and never host time. The three builders
are the three shapes a run takes: weak only, strong primary, and
weak-primary switching. The QEC cycle is one microsecond, and packet
assembly is declared free (its own per-round charge is priced in
tests/controller/test_round_assembly.py).
"""

import dataclasses

import decsim.controller.policies as boundary_policies
import decsim.controller.settings as controller_settings
import decsim.decoders.decoder_memory as decoder_memory
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.escalation.policies as escalation_policies
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings
import decsim.machine as machine_module
import decsim.observe.settings as observe_settings
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.records.program as program_records
import decsim.windows.settings as window_settings
import decsim.windows.windowing_schemes as windowing_schemes
from decsim.config import microseconds_to_ticks

# every declared stage of the fabric, in microseconds
DECLARED_MICROSECONDS = {
    "qpu_to_controller": 2.0,
    "readout_to_bits": 3.0,
    "packing": 0.0,
    "controller_to_weak_buffer": 4.0,
    "controller_to_strong_buffer": 7.0,
    "weak_buffer_to_weak_decoder": 5.0,
    "weak_decoder_to_strong_decoder": 3.0,
    "strong_buffer_to_strong_decoder": 6.0,
    "weak": 10.0,
    "strong": 30.0,
    "weak_decoder_to_frame": 2.0,
    "decoder_to_decoder": 0.5,
    "strong_decoder_to_frame": 4.0,
    "frame": 1.0,
    "frame_to_controller": 2.0,
    "controller_to_qpu": 2.0,
}
ROUND_MICROSECONDS = 1.0
# every path whose latency the card carries, in the order the reference
# card declares them; controller_to_strong_buffer is set per run
DECLARED_EDGE_NAMES = (
    "qpu_to_controller",
    "controller_to_weak_buffer",
    "weak_buffer_to_weak_decoder",
    "weak_decoder_to_strong_decoder",
    "strong_buffer_to_strong_decoder",
    "weak_decoder_to_frame",
    "decoder_to_decoder",
    "strong_decoder_to_frame",
    "frame_to_controller",
    "controller_to_qpu",
)
ESCALATION_THRESHOLD = 0.5


def sliding_scheme():
    """The sliding windows with the lookahead tail every run here uses."""
    lookahead = windowing_schemes.SlidingTerminalPolicy.REGULAR_STRIDE_LOOKAHEAD
    return windowing_schemes.SlidingWindowScheme(terminal_policy=lookahead)


def declared_edge(base_edge, latency_microseconds):
    """One path of the card, at a declared latency and no rate bound."""
    latency_ticks = microseconds_to_ticks(latency_microseconds)
    channel = link_settings.ChannelSettings(
        base_edge.channel.name,
        latency_ticks,
        None,
        "declared tick",
    )
    return link_settings.PathSettings(
        channel, base_edge.default_payload, base_edge.actual_payload_source
    )


def declared_profile(*, strong_buffer_microseconds=None):
    """The reference card with every latency replaced by a declared tick."""
    base = link_profiles.logical_reference_profile()
    declared_edges = {}
    for name in DECLARED_EDGE_NAMES:
        base_edge = getattr(base, name)
        latency = DECLARED_MICROSECONDS[name]
        declared_edges[name] = declared_edge(base_edge, latency)
    strong_latency = strong_buffer_microseconds
    if strong_latency is None:
        strong_latency = DECLARED_MICROSECONDS["controller_to_strong_buffer"]
    strong_store = declared_edge(
        base.controller_to_strong_buffer, strong_latency
    )
    declared_edges["controller_to_strong_buffer"] = strong_store
    return dataclasses.replace(
        base,
        # the declared qpu tick is wire time only; readout classification
        # prices the controller processing separately
        is_controller_processing_outside_qpu_to_controller=True,
        **declared_edges,
    )


def declared_controller(**changes):
    """The controller's declared readout and packing stages."""
    return controller_settings.ControllerSettings(
        readout_to_bits_microseconds=DECLARED_MICROSECONDS["readout_to_bits"],
        packing_microseconds_per_round=DECLARED_MICROSECONDS["packing"],
        **changes,
    )


def declared_qpu(round_microseconds=ROUND_MICROSECONDS):
    """A distance-three patch at the declared cycle."""
    return qpu_settings.QpuSettings(
        distance=3, round_period_microseconds=round_microseconds
    )


def declared_frame():
    """The frame's declared write cost."""
    commit = DECLARED_MICROSECONDS["frame"]
    return pauli_frame_module.PauliFrameConfig(commit_microseconds=commit)


def memory_operation(operation_id=1, **changes):
    """One patch's memory operation, on its own qubit and patch."""
    name = f"mem{operation_id}"
    return program_records.Operation(
        id=operation_id,
        name=name,
        qubits=(operation_id,),
        patches=(operation_id,),
        **changes,
    )


def declared_workload(operations, rounds):
    """The workload of those operations at a fixed round count."""
    listed = operations_or_one(operations)
    rounds_policy = round_policies.FixedRounds(rounds)
    return workload_settings.WorkloadSettings(
        operations=listed, rounds_policy=rounds_policy
    )


def run_machine(settings, seed=0, probes=()):
    """Build the machine, connect the test's probes, run it."""
    machine = machine_module.Machine.build(settings, seed)
    for probe in probes:
        machine.engine.action_done.connect(probe.observe)
    machine.run()
    return machine


def operations_or_one(operations):
    """The operations given, or one memory operation on patch 1."""
    if operations is not None:
        return operations
    return [memory_operation(1)]


def weak_only_run(
    *,
    rounds=6,
    operations=None,
    seed=0,
    io_trace=False,
    round_store=None,
    controller=None,
    observation=None,
    probes=(),
):
    """The weak-only baseline: one tier, readiness on Buffer 0."""
    workload = declared_workload(operations, rounds)
    decoder = decoders.PresetLatencyDecoder(DECLARED_MICROSECONDS["weak"])
    weak_decoder = decoder_settings.DecoderSettings(decoder=decoder)
    links = declared_profile()
    if controller is None:
        controller = declared_controller()
    if observation is None:
        observation = observe_settings.ObservationSettings(
            log_component_io=io_trace
        )
    qpu = declared_qpu()
    frame = declared_frame()
    settings = machine_module.MachineSettings(
        workload=workload,
        qpu=qpu,
        weak_decoder=weak_decoder,
        links=links,
        controller=controller,
        pauli_frame=frame,
        observation=observation,
    )
    if round_store is not None:
        settings = dataclasses.replace(settings, round_store=round_store)
    return run_machine(settings, seed, probes)


def strong_only_run(
    *, rounds=6, operations=None, seed=0, io_trace=False, record=False
):
    """The strong-primary baseline: readiness listens to Buffer 1."""
    workload = declared_workload(operations, rounds)
    decoder = decoders.PresetLatencyDecoder(DECLARED_MICROSECONDS["strong"])
    strong_decoder = decoder_settings.DecoderSettings(decoder=decoder)
    policy = escalation_policies.StrongOnly()
    escalation = escalation_settings.EscalationSettings(policy=policy)
    observation = observe_settings.ObservationSettings(
        log_component_io=io_trace, record_switching_windows=record
    )
    qpu = declared_qpu()
    links = declared_profile()
    controller = declared_controller()
    frame = declared_frame()
    settings = machine_module.MachineSettings(
        workload=workload,
        qpu=qpu,
        strong_decoder=strong_decoder,
        escalation=escalation,
        links=links,
        controller=controller,
        pauli_frame=frame,
        observation=observation,
    )
    return run_machine(settings, seed)


def switching_decoder(escalation_probability, probability_for):
    """The weak tier that reports a sampled confidence, and its strong."""
    latency = decoders.PresetLatencyDecoder(DECLARED_MICROSECONDS["weak"])
    weak = decoders.SampledConfidenceDecoder(
        latency, escalation_probability, probability_for=probability_for
    )
    strong = decoders.PresetLatencyDecoder(DECLARED_MICROSECONDS["strong"])
    return decoders.SwitchingRouter(weak=weak, strong=strong)


def switching_run(
    *,
    rounds=6,
    escalation_probability=0.0,
    operations=None,
    run_both_at_once=False,
    strong_window="two_sided_context",
    unit_pools=None,
    seed=0,
    io_trace=False,
    probability_for=None,
    record=False,
    strong_buffer_microseconds=None,
    weak_memory_rounds=None,
    round_microseconds=ROUND_MICROSECONDS,
    bulk_strong=False,
    probes=(),
):
    """Weak-primary switching on the declared fabric.

    A probability of 0.0 or 1.0, or a per-job probability_for returning
    one of the two, keeps the run deterministic: the sampled gap is 1.0
    (keep the weak result) or 0.0 (escalate) against the threshold.
    """
    router = switching_decoder(escalation_probability, probability_for)
    threshold = threshold_sources.FixedThreshold(ESCALATION_THRESHOLD)
    policy = escalation_policies.Switching(
        threshold,
        decoders.SAMPLED_CONFIDENCE_SOURCE,
        run_both_at_once=run_both_at_once,
    )
    workload = declared_workload(operations, rounds)
    # serial switching needs Held boundaries; the forward window refuses
    # them (escalation.policies.Switching.check_plan)
    boundary_policy = boundary_policies.Held()
    if strong_window == "forward":
        boundary_policy = None
    scheme = sliding_scheme()
    windows = window_settings.WindowSettings(
        scheme=scheme, boundary_policy=boundary_policy
    )
    pools = unit_pools
    if pools is None:
        pools = {"default": 1, "strong": 1}
    memory = None
    if weak_memory_rounds is not None:
        memory = decoder_memory.DecoderMemoryConfig(
            {"default": weak_memory_rounds}
        )
    decoder_manager = decoder_settings.DecoderManagerSettings(
        router=router,
        unit_pools=pools,
        decoder_memory=memory,
        bulk_strong=bulk_strong,
    )
    escalation = escalation_settings.EscalationSettings(
        policy=policy, strong_window=strong_window
    )
    links = declared_profile(
        strong_buffer_microseconds=strong_buffer_microseconds
    )
    observation = observe_settings.ObservationSettings(
        log_component_io=io_trace, record_switching_windows=record
    )
    qpu = declared_qpu(round_microseconds)
    controller = declared_controller()
    frame = declared_frame()
    settings = machine_module.MachineSettings(
        workload=workload,
        qpu=qpu,
        windows=windows,
        decoder_manager=decoder_manager,
        escalation=escalation,
        links=links,
        controller=controller,
        pauli_frame=frame,
        observation=observation,
    )
    return run_machine(settings, seed, probes)


def escalate_only(window_ids):
    """A per-job probability of 1.0 for those windows and 0.0 elsewhere."""

    def probability(job):
        if job.window_id in window_ids:
            return 1.0
        return 0.0

    return probability


def log_tick(log_lines, needle):
    """The tick of the first log line containing the needle."""
    for line in log_lines:
        if needle not in line:
            continue
        parts = line.split("]")
        head = parts[0]
        opened = head.lstrip("[")
        stamp = opened.strip()
        assert stamp.endswith("us"), line
        digits = stamp[:-2]
        microseconds = float(digits)
        return microseconds_to_ticks(microseconds)
    raise AssertionError(f"no log line contains {needle!r}")


def log_index(log_lines, needle):
    """The position of the first log line containing the needle."""
    for index, line in enumerate(log_lines):
        if needle in line:
            return index
    raise AssertionError(f"no log line contains {needle!r}")


def log_lines_containing(machine, needle):
    """Every log line of the run that contains the needle."""
    found = []
    for line in machine.observation.log.lines:
        if needle in line:
            found.append(line)
    return found


def frame_tiers(machine):
    """(window key, tier) of every frame record, in commit order."""
    snapshot = machine.pauli_frame.snapshot()
    tiers = []
    for record in snapshot.records:
        tiers.append((record.window_key, record.tier))
    return tiers


class OccupancyProbe:
    """Each store's live-round timeline, sampled after every action."""

    name = "occupancy_probe"

    def __init__(self, window_manager):
        self.window_manager = window_manager
        self.weak_timeline = []
        self.strong_timeline = []

    def observe(self, tick):
        """Note each store's occupancy whenever it changed."""
        weak_store = self.window_manager.retention.weak_store
        self._note(self.weak_timeline, tick, weak_store)
        strong_store = self.window_manager.retention.strong_store
        self._note(self.strong_timeline, tick, strong_store)

    def _note(self, timeline, tick, store):
        """Append the occupancy when this store has a new one."""
        if store is None:
            return
        occupancy = store.occupancy
        if not timeline:
            timeline.append((tick, occupancy))
            return
        last_occupancy = timeline[-1][1]
        if last_occupancy != occupancy:
            timeline.append((tick, occupancy))
