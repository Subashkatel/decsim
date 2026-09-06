"""The issuer: an admitted operation becomes one command at the next boundary.

A preloaded command (a program root, an ordinary successor) starts at
the QPU's next cycle boundary and the runtime hears that boundary through
on_started (SimPy's callback on the event, simpy/core.py step()); a
feedback-blocked command pays the decision-to-pulse cost (17 ticks here)
and the controller_to_qpu crossing first. The idle rounds claimed at the
issue are prepended for the windows.
"""

import types

import decsim.controller.feedback_streams as feedback_streams
import decsim.controller.instruction_output as instruction_output
import decsim.controller.operation_issue as operation_issue
import decsim.engine as engine_module
import decsim.message as message
import decsim.observe.log_writers as log_writers
import decsim.observe.round_events as round_events

BOUNDARY_TICK = 3000
PULSE_TICKS = 17


class RecordingQpu:
    def __init__(self):
        self.issued = []
        self.finished = False

    def issue(self, command):
        self.issued.append(command)

    def next_boundary(self):
        return BOUNDARY_TICK

    def finish(self):
        self.finished = True


class RecordingIdleRounds:
    def __init__(self, claimed=0):
        self.claimed = claimed
        self.ended = []

    def end_idle_period(self, operation, patch):
        self.ended.append((operation.id, patch))

    def claim(self, operation):
        del operation
        return self.claimed


class RecordingWindows:
    def __init__(self):
        self.prepended = []

    def prepend_idle_rounds(self, operation_id, count):
        self.prepended.append((operation_id, count))


def resolved(operation_id, round_ticks=1000, round_count=6):
    return types.SimpleNamespace(
        operation_id=operation_id,
        round_ticks=round_ticks,
        round_count=round_count,
    )


def issuer_with(engine, qpu, idle_rounds, windows, recorder, link=None):
    output = instruction_output.InstructionOutput(
        engine, link, qpu, PULSE_TICKS
    )
    if recorder is not None:
        output.output_event.connect(recorder.output)
    streams = feedback_streams.NoFeedbackStreams()
    resolved_operations = (resolved(1), resolved(2))
    return operation_issue.OperationIssuer(
        engine, streams, idle_rounds, windows, resolved_operations, output
    )


def ignore_boundary(boundary) -> None:
    del boundary


def test_a_preloaded_operation_starts_at_the_next_boundary_and_says_so():
    engine = engine_module.Engine()
    log = log_writers.LogWriter()
    engine.line.connect(log.write)
    qpu = RecordingQpu()
    idle_rounds = RecordingIdleRounds()
    windows = RecordingWindows()
    recorder = round_events.RoundEventRecorder(engine)
    issuer = issuer_with(engine, qpu, idle_rounds, windows, recorder)
    operation = message.Operation(1, "memory", (0,), patches=(0,))
    started = []

    issuer.issue_operation(operation, started.append)
    engine.run()

    assert started == [BOUNDARY_TICK]
    (command,) = qpu.issued
    assert command.operation is operation
    assert command.round_count == 6
    assert command.round_ticks == 1000
    kinds = [event.kind for event in recorder.output_events]
    assert kinds == ["PRELOADED_COMMAND"]
    assert idle_rounds.ended == [(1, 0)]
    assert windows.prepended == []
    (line,) = log.lines
    assert line.endswith("Controller: START memory  (Clifford, qubits (0,))")


def test_the_idle_rounds_claimed_at_the_issue_are_prepended_for_the_windows():
    engine = engine_module.Engine()
    qpu = RecordingQpu()
    idle_rounds = RecordingIdleRounds(claimed=4)
    windows = RecordingWindows()
    recorder = None
    issuer = issuer_with(engine, qpu, idle_rounds, windows, recorder)
    operation = message.Operation(1, "memory", (0,), patches=(0,))

    issuer.issue_operation(operation, ignore_boundary)

    assert windows.prepended == [(1, 4)]


def test_a_feedback_blocked_operation_pays_the_pulse_cost_before_it_starts():
    engine = engine_module.Engine()
    qpu = RecordingQpu()
    idle_rounds = RecordingIdleRounds()
    windows = RecordingWindows()
    recorder = round_events.RoundEventRecorder(engine)
    issuer = issuer_with(engine, qpu, idle_rounds, windows, recorder)
    operation = message.Operation(
        2, "corrected", (0,), patches=(0,), blocked_by=1
    )
    started = []

    def on_started(boundary):
        started.append((engine.now, boundary))

    issuer.issue_operation(operation, on_started)
    issued_before_the_pulse = list(qpu.issued)
    engine.run()

    assert issued_before_the_pulse == []
    assert started == [(PULSE_TICKS, BOUNDARY_TICK)]
    kinds_and_ticks = [
        (event.kind, event.tick) for event in recorder.output_events
    ]
    assert kinds_and_ticks == [("CONTROL_PULSE_COMMAND_ISSUED", PULSE_TICKS)]


def test_the_last_release_stops_the_qpu():
    engine = engine_module.Engine()
    qpu = RecordingQpu()
    idle_rounds = RecordingIdleRounds()
    windows = RecordingWindows()
    recorder = None
    issuer = issuer_with(engine, qpu, idle_rounds, windows, recorder)
    operation = message.Operation(1, "memory", (0,), patches=(0,))

    issuer.after_successor_release(operation, False, False)
    running = qpu.finished
    issuer.after_successor_release(operation, False, True)

    assert running is False
    assert qpu.finished is True
