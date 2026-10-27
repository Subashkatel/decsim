"""What the escalation tests share: a switching machine on declared ticks.

Every hop and stage has its declared tick value (tests/declared_run.py)
and every decoder a preset latency, so what a test asserts follows from
arithmetic over the declared ticks, never from measured host time. The
weak tier declares each window's confidence gap, 1.0 (keep) or 0.0
(escalate) against the threshold 0.5, so a run is deterministic.
"""

from typing import Optional

import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.machine as machine_module
import decsim.observe.settings as observe_settings
import decsim.presets.machines as machines
import decsim.windows.schemes.sliding as sliding_scheme
import decsim.windows.settings as window_settings
import tests.declared_run as declared_run


def escalate_only(window_ids) -> callable:
    """Whether a job's window is one of the listed windows."""

    def is_escalated(job) -> bool:
        return job.window_id in window_ids

    return is_escalated


def switching_machine(
    *, online_threshold=None, **graph_changes
) -> machine_module.Machine:
    """One d=3 memory operation, weak-primary switching on declared ticks.

    online_threshold is the calibrator an online threshold row decides
    on; graph_changes are switching_graph's.
    """
    graph = switching_graph(**graph_changes)
    return machine_module.Machine.build(
        graph, 0, online_threshold=online_threshold
    )


def switching_graph(
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
):
    """The graph of switching_machine.

    switching replaces the declared switching part when given (a table
    row of its own, say); scheme replaces the lookahead sliding windows
    with a caller's own windowing scheme; decision_cycles is the window
    side's decision, on the declared clock; escalation_microseconds
    replaces the declared latency of the hop an escalated window's
    rounds ride up on.
    """
    is_escalated = escalate_only(escalated_windows)
    weak_microseconds = declared_run.DECLARED_MICROSECONDS["weak"]
    weak = declared_run.DeclaredConfidenceDecoder.Settings(
        weak_microseconds, is_escalated
    )
    strong_microseconds = declared_run.DECLARED_MICROSECONDS["strong"]
    strong = decoders.PresetLatencyDecoder.Settings(strong_microseconds)
    weak_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=weak, engine=declared_run.DECLARED_ENGINE
    )
    strong_decoder = decoder_settings.DecoderPoolSettings(
        algorithm=strong, engine=declared_run.DECLARED_ENGINE
    )
    # the windows take the strong window's boundary row, as
    # window_settings.switching_windows gives it
    boundary_policy = strong_window.boundary_policy
    workload = declared_run.declared_workload(None, rounds)
    qpu = declared_run.declared_qpu(round_microseconds)
    if scheme is None:
        scheme = sliding_scheme.SlidingWindowScheme.Settings()
    windows = window_settings.WindowSettings(
        clock=declared_run.DECLARED_CLOCK,
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
    fabric = declared_run.declared_fabric()
    if escalation_microseconds is not None:
        fabric = declared_run.declared_fabric(forward=escalation_microseconds)
    trace = "off"
    if trace_path is not None:
        trace = str(trace_path)
    observation = observe_settings.ObservationSettings(
        log_component_io=True,
        record_switching_windows=record,
        trace=trace,
    )
    controller = declared_run.declared_controller()
    frame = declared_run.declared_frame()
    machine = machines.base(
        workload,
        qpu,
        windows,
        controller,
        frame,
        observation=observation,
        fabric=fabric,
    )
    with_weak = machines.with_pool(
        machine, weak_decoder, store=weak_syndrome_buffer
    )
    # the windows stay as written: with_escalation leaves them alone
    return machines.with_escalation(
        with_weak,
        machines.WEAK_POOL,
        machines.STRONG_POOL,
        strong_decoder,
        switching,
        switching_name=machines.SWITCHING,
    )
