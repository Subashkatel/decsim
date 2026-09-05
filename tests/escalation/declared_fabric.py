"""What the escalation tests share: a switching machine on declared ticks.

Every link and stage has a declared tick value and every decoder a
preset latency, so what a test asserts follows from arithmetic over the
declared ticks, never from measured host time. The sampled confidence
decoder with probability 0.0 or 1.0 per window keeps a run
deterministic: the sampled gap is 1.0 (keep) or 0.0 (escalate) against
the threshold 0.5.
"""

import dataclasses

import decsim.config as config
import decsim.controller.policies as boundary_policies
import decsim.controller.settings as controller_settings
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.decoders.weak_strong_switching as weak_strong_switching
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings
import decsim.machine as machine_module
import decsim.message as message
import decsim.observe.settings as observe_settings
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.windows.settings as window_settings
import decsim.windows.windowing_schemes as windowing_schemes

DECLARED_MICROSECONDS = {
    "qpu_to_controller": 2.0,
    "readout_to_bits": 3.0,
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
DECLARED_EDGE_NAMES = (
    "qpu_to_controller",
    "weak_buffer_to_weak_decoder",
    "weak_decoder_to_strong_decoder",
    "strong_buffer_to_strong_decoder",
    "weak_decoder_to_frame",
    "decoder_to_decoder",
    "strong_decoder_to_frame",
    "frame_to_controller",
    "controller_to_qpu",
)


def escalate_only(window_ids) -> callable:
    """A per-job probability: 1.0 for the listed windows, 0.0 elsewhere."""

    def probability(job) -> float:
        if job.window_id in window_ids:
            return 1.0
        return 0.0

    return probability


def switching_machine(
    *,
    rounds: int,
    escalated_windows,
    double_window: bool = False,
    run_both_at_once: bool = False,
    round_microseconds: float = 1.0,
    strong_buffer_microseconds: float = 7.0,
    record: bool = False,
) -> machine_module.Machine:
    """One d=3 memory operation, weak-primary switching on declared ticks."""
    probability = escalate_only(escalated_windows)
    weak_latency = decoders.PresetLatencyDecoder(DECLARED_MICROSECONDS["weak"])
    weak = decoders.SampledConfidenceDecoder(
        weak_latency, 0.0, probability_for=probability
    )
    strong = decoders.PresetLatencyDecoder(DECLARED_MICROSECONDS["strong"])
    router = decoders.SwitchingRouter(weak=weak, strong=strong)
    policy = weak_strong_switching.Switching(
        0.5,
        decoders.SAMPLED_CONFIDENCE_SOURCE,
        run_both_at_once=run_both_at_once,
    )
    boundary_policy = boundary_policies.Held()
    if double_window:
        boundary_policy = None
    operation = message.Operation(id=1, name="mem1", qubits=(1,), patches=(1,))
    rounds_policy = round_policies.FixedRounds(rounds)
    workload = workload_settings.WorkloadSettings(
        operations=[operation], rounds_policy=rounds_policy
    )
    qpu = qpu_settings.QpuSettings(
        distance=3, round_period_microseconds=round_microseconds
    )
    lookahead = windowing_schemes.SlidingTerminalPolicy.REGULAR_STRIDE_LOOKAHEAD
    scheme = windowing_schemes.SlidingWindowScheme(terminal_policy=lookahead)
    windows = window_settings.WindowSettings(
        scheme=scheme, boundary_policy=boundary_policy
    )
    decoder_manager = decoder_settings.DecoderManagerSettings(
        router=router, unit_pools={"default": 1, "strong": 1}
    )
    escalation = decoder_settings.EscalationSettings(
        policy=policy, double_window=double_window
    )
    links = declared_profile(strong_buffer_microseconds)
    controller = controller_settings.ControllerSettings(
        readout_to_bits_microseconds=DECLARED_MICROSECONDS["readout_to_bits"],
        packing_microseconds_per_round=0.0,
    )
    pauli_frame = pauli_frame_module.PauliFrameConfig(
        commit_microseconds=DECLARED_MICROSECONDS["frame"]
    )
    observation = observe_settings.ObservationSettings(
        log_component_io=True, record_switching_windows=record
    )
    settings = machine_module.MachineSettings(
        workload=workload,
        qpu=qpu,
        windows=windows,
        decoder_manager=decoder_manager,
        escalation=escalation,
        links=links,
        controller=controller,
        pauli_frame=pauli_frame,
        observation=observation,
    )
    return machine_module.Machine.build(settings, 0)


def declared_profile(strong_buffer_microseconds: float):
    """The reference card with every latency replaced by a declared tick."""
    base = link_profiles.logical_reference_profile()
    declared_edges = {}
    for name in DECLARED_EDGE_NAMES:
        base_edge = getattr(base, name)
        declared_edges[name] = _declared_edge(base_edge, name)
    profile = dataclasses.replace(
        base,
        # the declared qpu tick is wire time only; readout classification
        # prices the controller processing separately
        is_controller_processing_outside_qpu_to_controller=True,
        **declared_edges,
    )
    profile = link_profiles.with_controller_to_weak_buffer_path(
        profile,
        latency_microseconds=DECLARED_MICROSECONDS["controller_to_weak_buffer"],
        aggregate_bits_per_microsecond=None,
        source="escalation test declared tick",
    )
    return link_profiles.with_controller_to_strong_buffer_path(
        profile,
        latency_microseconds=strong_buffer_microseconds,
        aggregate_bits_per_microsecond=None,
        source="escalation test declared tick",
    )


def frame_tiers(machine) -> list:
    """(window key, tier) of every frame record, in commit order."""
    snapshot = machine.pauli_frame.snapshot()
    tiers = []
    for record in snapshot.records:
        tiers.append((record.window_key, record.tier))
    return tiers


def log_lines_containing(machine, needle: str) -> list:
    """Every log line that contains the needle."""
    lines = []
    for line in machine.engine.log_lines:
        if needle in line:
            lines.append(line)
    return lines


def _declared_edge(base_edge, name: str) -> link_settings.PathSettings:
    latency_ticks = config.microseconds_to_ticks(DECLARED_MICROSECONDS[name])
    channel = link_settings.ChannelSettings(
        base_edge.channel.name,
        latency_ticks,
        None,
        "escalation test declared tick",
    )
    return link_settings.PathSettings(
        channel, base_edge.default_payload, base_edge.actual_payload_source
    )
