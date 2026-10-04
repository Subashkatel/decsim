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
import functools
from collections.abc import Callable
from typing import Optional

import decsim.config as config
import decsim.controller.settings as controller_settings
import decsim.decoders.decoder as decoder_module
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings
import decsim.machine as machine_module
import decsim.observe.settings as observe_settings
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.settings as machine_settings
import decsim.windows.settings as window_settings

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
# the controller and the frame price their cycles on a 2 MHz clock, so
# every declared microsecond above is a whole number of its cycles and
# every declared instant lands on one of its edges
DECLARED_CLOCK = config.Clock(500_000)


# the unit's fetch and release are free, so a decode holds its unit for
# its declared latency alone
DECLARED_ENGINE = decoder_settings.EngineSettings(
    clock=DECLARED_CLOCK,
    fetch_cycles_per_round=0,
    fetch_cycles_per_job=0,
    release_cycles_per_job=0,
    release_cycles_per_round=0,
)


@dataclasses.dataclass(frozen=True)
class OneDecoder:
    """A test's own decoder instance, as the record a tier builds it from.

    It hands back the same instance, so it builds one machine only.
    """

    decoder: object
    name = "test_decoder"

    def build(self):
        """The test's instance."""
        return self.decoder


def declared_cycles(name):
    """One declared stage's microseconds, as cycles of the declared clock."""
    ticks = config.microseconds_to_ticks(DECLARED_MICROSECONDS[name])
    return ticks // DECLARED_CLOCK.period_ticks


ROUND_MICROSECONDS = 1.0
# every path whose latency the card carries, in the order the reference
# card declares them
DECLARED_EDGE_NAMES = (
    "qpu_to_controller",
    "controller_to_weak_buffer",
    "controller_to_strong_buffer",
    "weak_buffer_to_weak_decoder",
    "weak_decoder_to_strong_decoder",
    "strong_buffer_to_strong_decoder",
    "weak_decoder_to_frame",
    "decoder_to_decoder",
    "strong_decoder_to_frame",
    "frame_to_controller",
    "controller_to_qpu",
)
# 0.5 nats, which decibels_to_nats gives back exactly
ESCALATION_THRESHOLD_DECIBELS = threshold_sources.nats_to_decibels(0.5)
DECLARED_CONFIDENCE_SOURCE = decoding_records.SoftOutputSource(
    method="declared_confidence"
)


class DeclaredConfidence:
    """The confidence of a weak decoder that declares its own gap.

    The window's one decode carries its soft output already, so the
    signal hands it on at no cost and needs no evidence of the decode.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The declared signal, as the switching slot names it."""

        name = "declared_confidence"

        def build(self, weak_algorithm, threshold_nats):
            """The signal; it reads neither the decoder nor the threshold."""
            del weak_algorithm
            del threshold_nats
            return DeclaredConfidence()

    source = DECLARED_CONFIDENCE_SOURCE
    fault_model_requirement = None
    decoder_evidence_requirement = frozenset()
    evidence_refusal = "the declared confidence needs no evidence"
    forced_logical_classes = ()

    def compute(self, solves):
        """The one solve's declared soft output, computed in no time."""
        soft_output = solves[0].soft_output
        return decoding_records.SoftOutputComputation(soft_output)


def declared_switching(**changes):
    """The switching slot over the declared confidence and threshold."""
    confidence = DeclaredConfidence.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=ESCALATION_THRESHOLD_DECIBELS
    )
    switching = escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold
    )
    return dataclasses.replace(switching, **changes)


# the two strong window rows decsim ships, at their own settings
REDO_WINDOW = strong_window_shapes.RedoWindow.Settings()
DOUBLE_WINDOW = strong_window_shapes.DoubleWindow.Settings()


def windows_on(windows, scheme_row=None, **sizes):
    """The windows with another scheme row, other sizes, or both.

    scheme_row is a windowing scheme class; None keeps the windows' own
    row. A size left out keeps the scheme's own.
    """
    scheme = windows.scheme
    if scheme_row is not None:
        scheme = scheme_row.Settings(
            commit_rounds=scheme.commit_rounds,
            buffer_rounds=scheme.buffer_rounds,
        )
    scheme = dataclasses.replace(scheme, **sizes)
    return dataclasses.replace(windows, scheme=scheme)


def declared_edge(base_edge, latency_microseconds):
    """One path of the card, at a declared latency and no rate bound."""
    latency_ticks = config.microseconds_to_ticks(latency_microseconds)
    channel = link_settings.ChannelSettings(
        base_edge.channel.name,
        latency_ticks,
        None,
        "declared tick",
    )
    return link_settings.PathSettings(
        channel,
        base_edge.default_payload,
        base_edge.actual_payload_source,
        excludes_receiver_processing=base_edge.excludes_receiver_processing,
    )


def declared_profile():
    """The reference card with every latency replaced by a declared tick."""
    base = link_profiles.logical_reference_profile()
    declared_edges = {}
    for name in DECLARED_EDGE_NAMES:
        base_edge = getattr(base, name)
        latency = DECLARED_MICROSECONDS[name]
        declared_edges[name] = declared_edge(base_edge, latency)
    # the declared qpu tick is wire time only; readout classification
    # prices the controller processing separately
    declared_edges["qpu_to_controller"] = dataclasses.replace(
        declared_edges["qpu_to_controller"], excludes_receiver_processing=True
    )
    return dataclasses.replace(base, **declared_edges)


def declared_controller(**changes):
    """The controller's declared readout and packing stages."""
    readout_cycles = declared_cycles("readout_to_bits")
    packing_cycles = declared_cycles("packing")
    return controller_settings.ControllerSettings(
        clock=DECLARED_CLOCK,
        readout_to_bits_cycles=readout_cycles,
        packing_cycles_per_round=packing_cycles,
        **changes,
    )


def declared_qpu(round_microseconds=ROUND_MICROSECONDS):
    """A distance-three patch at the declared cycle."""
    return qpu_settings.QpuSettings(
        distance=3, round_period_microseconds=round_microseconds
    )


@dataclasses.dataclass(frozen=True)
class GivenSource:
    """A source record whose build returns the source the test made."""

    source: object

    def build(self, code, circuit_arguments):
        del code, circuit_arguments
        return self.source


@dataclasses.dataclass(frozen=True)
class GivenCard:
    """A code card record whose build returns the card the test made."""

    card: object

    def build(self, distance, commit_rounds_override, buffer_rounds_override):
        del distance, commit_rounds_override, buffer_rounds_override
        return self.card


@dataclasses.dataclass(frozen=True)
class GivenLayout:
    """A layout record whose build returns the layout the test made."""

    layout: object

    def build(self, code):
        del code
        return self.layout


def declared_frame():
    """The frame's declared write cost."""
    write_cycles = declared_cycles("frame")
    return pauli_frame_module.PauliFrameConfig(
        write_cycles=write_cycles, clock=DECLARED_CLOCK
    )


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
    """Build the machine, attach the test's probes to it, run it."""
    machine = machine_module.Machine.build(settings, seed)
    for probe in probes:
        probe.attach(machine)
    machine.run()
    return machine


@dataclasses.dataclass(frozen=True)
class EndedRequest:
    """One decode request at its end: its job, its result, its outcome."""

    job: decoding_records.DecodeJob
    result: Optional[decoding_records.DecodeResult]
    outcome: decoding_records.RequestProcessingOutcome


class EndedRequests:
    """A probe that hears every decode request end, on both sides.

    It listens on the decode outcomes' request_ended, the source the
    package's record ledgers hear, so a test reads a request's window,
    ticks and outcome off the job itself. Attach it before the run.
    """

    def __init__(self) -> None:
        self.ended: list = []

    def attach(self, machine) -> None:
        """Hear the requests of both decoder managers the run has."""
        part = machine.decoders
        managers = (part.decoder_manager, part.strong_decoder_manager)
        for manager in managers:
            if manager is None:
                continue
            manager.outcomes.trace.request_ended.connect(self.request_ended)

    def request_ended(self, job, result, outcome, decode_output_ticks) -> None:
        """One request reached its terminal outcome."""
        del decode_output_ticks
        ended = EndedRequest(job, result, outcome)
        self.ended.append(ended)

    def of_tier(self, tier) -> list:
        """The requests of one tier, in the order they ended."""
        requests = []
        for ended in self.ended:
            if ended.job.request_key.tier is tier:
                requests.append(ended)
        return requests


@dataclasses.dataclass(frozen=True)
class FinishedDecode:
    """One decode whose unit gave its compute back, with its two ticks."""

    job: decoding_records.DecodeJob
    dispatch_ticks: int
    finish_ticks: int


class FinishedDecodes:
    """A probe that hears every decode give its unit back, on both sides.

    It listens on each decode service's job_finished, which fires when
    the unit's compute goes back, so a decode's span runs from its
    dispatch to the end of any confidence walk charged on its unit.
    Attach it before the run.
    """

    def __init__(self) -> None:
        self.finished: list = []

    def attach(self, machine) -> None:
        """Hear the decode services of both decoder managers the run has."""
        part = machine.decoders
        managers = (part.decoder_manager, part.strong_decoder_manager)
        listener = functools.partial(self.job_finished, machine.engine)
        for manager in managers:
            if manager is None:
                continue
            manager.service.trace.job_finished.connect(listener)

    def job_finished(self, engine, job, unit) -> None:
        """One decode's unit took its compute back at this tick."""
        del unit
        finished = FinishedDecode(job, job.service_dispatch_ticks, engine.now)
        self.finished.append(finished)

    def of_pool(self, pool: str) -> list:
        """The decodes of one pool, in the order they finished."""
        decodes = []
        for finished in self.finished:
            if finished.job.pool == pool:
                decodes.append(finished)
        return decodes


class UnitMemoryDeposits:
    """A probe that hears every unit memory's deposits, on both sides.

    It listens on each unit memory's deposited source and reads the
    memory's own held bits there, which already count the input that
    landed, so the most a memory ever held is the most it read here.
    Attach it before the run.
    """

    def __init__(self) -> None:
        self.held_bits_by_unit: dict = {}

    def attach(self, machine) -> None:
        """Hear the unit memories of both decoder managers the run has."""
        part = machine.decoders
        managers = (part.decoder_manager, part.strong_decoder_manager)
        for manager in managers:
            if manager is None:
                continue
            for unit in manager.pool.units:
                self._hear(unit)

    def deposit_count(self, unit) -> int:
        """How many inputs landed in the unit's memory."""
        held_bits = self.held_bits_by_unit[unit.name]
        return len(held_bits)

    def peak_held_bits(self, unit) -> int:
        """The most bits the unit's memory held at once."""
        held_bits = self.held_bits_by_unit[unit.name]
        return max(held_bits)

    def _hear(self, unit) -> None:
        """Start the unit's list and connect its memory's deposits."""
        self.held_bits_by_unit[unit.name] = []
        listener = functools.partial(self._deposited, unit)
        unit.memory.trace.deposited.connect(listener)

    def _deposited(self, unit, _job, _decoder_input) -> None:
        """The memory's held bits right after one input landed."""
        held_bits = self.held_bits_by_unit[unit.name]
        held_bits.append(unit.memory.occupied_bits)


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
    weak_syndrome_buffer=None,
    copies_input=True,
    windows=None,
    controller=None,
    observation=None,
    probes=(),
    clock=None,
):
    """The weak-only baseline: one tier, readiness on the weak syndrome buffer.

    One complete machine keeps every path fixed while a test replaces
    only the component card whose reaction-time shift it measures.
    """
    workload = declared_workload(operations, rounds)
    weak_microseconds = DECLARED_MICROSECONDS["weak"]
    algorithm = decoders.PresetLatencyDecoder.Settings(weak_microseconds)
    weak_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=algorithm, copies_input=copies_input, engine=DECLARED_ENGINE
    )
    links = declared_profile()
    if controller is None:
        controller = declared_controller()
    if observation is None:
        observation = observe_settings.ObservationSettings(
            log_component_io=io_trace
        )
    qpu = declared_qpu()
    frame = declared_frame()
    settings = machine_settings.MachineSettings(
        clock=clock,
        workload=workload,
        qpu=qpu,
        weak_decoder=weak_decoder,
        links=links,
        controller=controller,
        pauli_frame=frame,
        observation=observation,
    )
    if windows is not None:
        settings = dataclasses.replace(settings, windows=windows)
    if weak_syndrome_buffer is not None:
        settings = dataclasses.replace(
            settings, weak_syndrome_buffer=weak_syndrome_buffer
        )
    return run_machine(settings, seed, probes)


def strong_only_run(
    *, rounds=6, operations=None, seed=0, io_trace=False, record=False
):
    """The strong-primary baseline listens to the strong syndrome buffer."""
    workload = declared_workload(operations, rounds)
    strong_microseconds = DECLARED_MICROSECONDS["strong"]
    algorithm = decoders.PresetLatencyDecoder.Settings(strong_microseconds)
    strong_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=algorithm, engine=DECLARED_ENGINE
    )
    observation = observe_settings.ObservationSettings(
        log_component_io=io_trace, record_switching_windows=record
    )
    qpu = declared_qpu()
    links = declared_profile()
    controller = declared_controller()
    frame = declared_frame()
    settings = machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        strong_decoder=strong_decoder,
        links=links,
        controller=controller,
        pauli_frame=frame,
        observation=observation,
    )
    return run_machine(settings, seed)


class DeclaredConfidenceDecoder(decoder_module.DecoderBase):
    """A preset latency whose confidence gap each window declares.

    A timing-only decoder has no syndrome to compute a confidence from,
    so the test declares it: gap 0.0, under every threshold these runs
    set, for a window is_escalated(job) names, so Switching escalates
    it; gap 1.0 elsewhere, so the weak result is kept.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The latency and which windows declare a gap under the threshold."""

        latency_microseconds: float
        is_escalated: Callable
        name = "declared_confidence"

        def build(self):
            """A fresh decoder of these settings."""
            return DeclaredConfidenceDecoder(
                self.latency_microseconds, self.is_escalated
            )

    def __init__(self, latency_microseconds, is_escalated):
        self.latency_microseconds = latency_microseconds
        self.is_escalated = is_escalated

    def latency(self, job):
        del job
        return config.microseconds_to_ticks(self.latency_microseconds)

    def decode(self, job):
        confidence_gap = 1.0
        if self.is_escalated(job):
            confidence_gap = 0.0
        soft_output = decoding_records.SoftOutput(
            gap=confidence_gap, source=DECLARED_CONFIDENCE_SOURCE
        )
        return decoding_records.DecodeResult(
            job.operation_id, job.window_id, soft_output=soft_output
        )


def switching_decoders(escalates):
    """The weak tier that declares its confidence, and its strong."""

    def is_escalated(job):
        del job
        return escalates

    weak_microseconds = DECLARED_MICROSECONDS["weak"]
    weak = DeclaredConfidenceDecoder.Settings(weak_microseconds, is_escalated)
    strong_microseconds = DECLARED_MICROSECONDS["strong"]
    strong = decoders.PresetLatencyDecoder.Settings(strong_microseconds)
    return weak, strong


def switching_run(
    *,
    rounds=6,
    escalates=False,
    operations=None,
    run_both_at_once=False,
    strong_window=REDO_WINDOW,
    weak_units=1,
    seed=0,
    io_trace=False,
    record=False,
    weak_memory_bits=None,
    round_microseconds=ROUND_MICROSECONDS,
    bulk_strong=False,
    probes=(),
    clock=None,
    threshold_cycles=0,
    switch_cycles=0,
    strong_copies_input=True,
):
    """Weak-primary switching on the declared fabric.

    escalates declares every window's confidence gap 0.0 (escalate)
    rather than 1.0 (keep the weak result) against the threshold.
    """
    weak, strong = switching_decoders(escalates)
    weak_memory = decoder_settings.UnitMemorySettings(bits=weak_memory_bits)
    weak_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=weak,
        unit_count=weak_units,
        unit_memory=weak_memory,
        engine=DECLARED_ENGINE,
    )
    strong_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=strong,
        copies_input=strong_copies_input,
        engine=DECLARED_ENGINE,
    )
    workload = declared_workload(operations, rounds)
    plain_windows = window_settings.WindowSettings()
    windows = window_settings.switching_windows(plain_windows, strong_window)
    decoder_manager = decoder_settings.DecoderManagerSettings(
        bulk_strong=bulk_strong
    )
    switching = declared_switching(
        run_both_at_once=run_both_at_once,
        strong_window=strong_window,
        clock=clock,
        threshold_cycles=threshold_cycles,
        switch_cycles=switch_cycles,
    )
    links = declared_profile()
    observation = observe_settings.ObservationSettings(
        log_component_io=io_trace,
        record_switching_windows=record,
    )
    qpu = declared_qpu(round_microseconds)
    controller = declared_controller()
    frame = declared_frame()
    settings = machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        windows=windows,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        decoder_manager=decoder_manager,
        switching=switching,
        links=links,
        controller=controller,
        pauli_frame=frame,
        observation=observation,
    )
    return run_machine(settings, seed, probes)


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
        return config.microseconds_to_ticks(microseconds)
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
    snapshot = machine.control.pauli_frame.snapshot()
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


def reaction_ticks(machine) -> tuple:
    """One window's readiness, queue, dispatch, decode and frame ticks."""
    windows = machine.windows.window_manager.planner.windows_by_key.values()
    (window,) = windows
    snapshot = machine.control.pauli_frame.snapshot()
    (record,) = snapshot.records
    return (
        window.t_data_complete,
        window.t_queued,
        window.t_dispatch,
        window.t_done,
        record.accepted_ticks,
        record.committed_ticks,
    )
