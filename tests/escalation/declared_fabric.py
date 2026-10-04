"""What the escalation tests share: a switching machine on declared ticks.

Every link and stage has a declared tick value and every decoder a
preset latency, so what a test asserts follows from arithmetic over the
declared ticks, never from measured host time. The weak tier declares
each window's confidence gap, 1.0 (keep) or 0.0 (escalate) against the
threshold 0.5, so a run is deterministic.
"""

import dataclasses
from typing import Optional

import decsim.config as config
import decsim.controller.settings as controller_settings
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.frontends.settings as workload_settings
import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings
import decsim.machine as machine_module
import decsim.observe.settings as observe_settings
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.records.program as program_records
import decsim.settings as machine_settings
import decsim.windows.boundary_policies as boundary_policies
import decsim.windows.schemes.sliding as sliding_scheme
import decsim.windows.settings as window_settings
import tests.declared_run as declared_run

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
# the controller and the frame price their cycles on a 2 MHz clock, so
# every declared microsecond above is a whole number of its cycles and
# every declared instant lands on one of its edges
DECLARED_CLOCK = config.Clock(500_000)


def declared_cycles(name):
    """One declared stage's microseconds, as cycles of the declared clock."""
    ticks = config.microseconds_to_ticks(DECLARED_MICROSECONDS[name])
    return ticks // DECLARED_CLOCK.period_ticks


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


def escalate_only(window_ids) -> callable:
    """Whether a job's window is one of the listed windows."""

    def is_escalated(job) -> bool:
        return job.window_id in window_ids

    return is_escalated


def switching_machine(
    *,
    rounds: int,
    escalated_windows,
    strong_window=declared_run.REDO_WINDOW,
    run_both_at_once: bool = False,
    round_microseconds: float = 1.0,
    escalation_microseconds: Optional[float] = None,
    record: bool = False,
    switching=None,
    trace_path=None,
    scheme=None,
    weak_syndrome_buffer=None,
    decision_cycles: int = 0,
    online_threshold=None,
) -> machine_module.Machine:
    """One d=3 memory operation, weak-primary switching on declared ticks.

    switching replaces the declared switching slot when given (a table
    row of its own, say), and online_threshold is the calibrator an
    online threshold row decides on; scheme replaces the lookahead
    sliding windows with a caller's own windowing scheme; decision_cycles
    is the window side's decision, on the declared clock.
    """
    is_escalated = escalate_only(escalated_windows)
    weak_microseconds = DECLARED_MICROSECONDS["weak"]
    weak = declared_run.DeclaredConfidenceDecoder.Settings(
        weak_microseconds, is_escalated
    )
    strong_microseconds = DECLARED_MICROSECONDS["strong"]
    strong = decoders.PresetLatencyDecoder.Settings(strong_microseconds)
    weak_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=weak, engine=declared_run.DECLARED_ENGINE
    )
    strong_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=strong, engine=declared_run.DECLARED_ENGINE
    )
    boundary_policy = boundary_policies.Held.Settings()
    if strong_window.absorbs_weak_windows:
        boundary_policy = boundary_policies.Eager.Settings()
    operation = program_records.Operation(
        id=1, name="mem1", qubits=(1,), patches=(1,)
    )
    rounds_policy = round_policies.FixedRounds(rounds)
    workload = workload_settings.WorkloadSettings(
        operations=[operation], rounds_policy=rounds_policy
    )
    qpu = qpu_settings.QpuSettings(
        distance=3, round_period_microseconds=round_microseconds
    )
    if scheme is None:
        scheme = sliding_scheme.SlidingWindowScheme.Settings()
    windows = window_settings.WindowSettings(
        clock=DECLARED_CLOCK,
        decision_cycles=decision_cycles,
        scheme=scheme,
        terminal_policy="lookahead",
        boundary_policy=boundary_policy,
    )
    if switching is None:
        switching = declared_run.declared_switching(
            run_both_at_once=run_both_at_once,
            strong_window=strong_window,
        )
    links = declared_profile(escalation_microseconds)
    readout_cycles = declared_cycles("readout_to_bits")
    controller = controller_settings.ControllerSettings(
        clock=DECLARED_CLOCK,
        readout_to_bits_cycles=readout_cycles,
        packing_cycles_per_round=0,
    )
    write_cycles = declared_cycles("frame")
    pauli_frame = pauli_frame_module.PauliFrameConfig(
        write_cycles=write_cycles, clock=DECLARED_CLOCK
    )
    trace = "off"
    if trace_path is not None:
        trace = str(trace_path)
    observation = observe_settings.ObservationSettings(
        log_component_io=True,
        record_switching_windows=record,
        trace=trace,
    )
    settings = machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        windows=windows,
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        switching=switching,
        links=links,
        controller=controller,
        pauli_frame=pauli_frame,
        observation=observation,
    )
    if weak_syndrome_buffer is not None:
        settings = dataclasses.replace(
            settings, weak_syndrome_buffer=weak_syndrome_buffer
        )
    return machine_module.Machine.build(
        settings, 0, online_threshold=online_threshold
    )


def declared_profile(escalation_microseconds: Optional[float] = None):
    """The reference card with every latency replaced by a declared tick.

    escalation_microseconds replaces the declared 3 us of
    weak_decoder_to_strong_decoder, the hop an escalated window's
    rounds ride up on. A switching run never takes the controller's hop
    into the strong syndrome buffer, so that latency is the declared one
    and no knob.
    """
    base = link_profiles.logical_reference_profile()
    declared_edges = {}
    for name in DECLARED_EDGE_NAMES:
        base_edge = getattr(base, name)
        latency = DECLARED_MICROSECONDS[name]
        declared_edges[name] = _declared_edge(base_edge, latency)
    if escalation_microseconds is not None:
        declared_edges["weak_decoder_to_strong_decoder"] = _declared_edge(
            base.weak_decoder_to_strong_decoder, escalation_microseconds
        )
    # the declared qpu tick is wire time only; readout classification
    # prices the controller processing separately
    declared_edges["qpu_to_controller"] = dataclasses.replace(
        declared_edges["qpu_to_controller"], excludes_receiver_processing=True
    )
    return dataclasses.replace(base, **declared_edges)


def frame_tiers(machine) -> list:
    """(window key, tier) of every frame record, in commit order."""
    snapshot = machine.control.pauli_frame.snapshot()
    tiers = []
    for record in snapshot.records:
        tiers.append((record.window_key, record.tier))
    return tiers


def log_lines_containing(machine, needle: str) -> list:
    """Every log line that contains the needle."""
    lines = []
    for line in machine.observation.log.lines:
        if needle in line:
            lines.append(line)
    return lines


def _declared_edge(
    base_edge, latency_microseconds: float
) -> link_settings.PathSettings:
    latency_ticks = config.microseconds_to_ticks(latency_microseconds)
    channel = link_settings.ChannelSettings(
        base_edge.channel.name,
        latency_ticks,
        None,
        "escalation test declared tick",
    )
    return link_settings.PathSettings(
        channel,
        base_edge.default_payload,
        base_edge.actual_payload_source,
        excludes_receiver_processing=base_edge.excludes_receiver_processing,
    )
