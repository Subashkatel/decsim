"""Behavior tests for decoded-result orchestration."""

from types import SimpleNamespace

import pytest

from decsim.program.orchestrators import ExecutionOrchestrator


class RecordingEngine:
    """Provide a controllable clock and capture log calls."""

    def __init__(self, now=0, events=None):
        self.now = now
        self.events = [] if events is None else events

    def log(self, component, message):
        self.events.append(("log", component, message))


class RecordingController:
    """Capture relayed decisions without interpreting them."""

    def __init__(self, events):
        self.events = events

    def relay_instruction(self, decision, sink):
        self.events.append(("relay", decision, sink))


def operation(op_id, name="decode", requires_return=False):
    """Build the operation fields consumed by the orchestrator."""
    return SimpleNamespace(
        id=op_id,
        name=name,
        requires_result_return_to_qpu=requires_return,
    )


def result(logical_observables=None, **fields):
    """Build a decoded result with deliberately irrelevant extra fields."""
    return SimpleNamespace(logical_observables=logical_observables, **fields)


def test_construction_initializes_history_archive_stats_and_collaborators():
    """Construction stores its engine and initializes all owned state."""
    engine = RecordingEngine()

    orchestrator = ExecutionOrchestrator(engine, history_size=3, retain_all=True)
    unarchived = ExecutionOrchestrator(engine, retain_all=False)

    assert orchestrator.engine is engine
    assert orchestrator.blocked_by_index == {}
    assert list(orchestrator.history) == []
    assert orchestrator.history.maxlen == 3
    assert orchestrator.stats == {
        "outcomes": 0,
        "decisions": 0,
        "result_returns": 0,
    }
    assert orchestrator.archive == {}
    assert orchestrator.controller is None
    assert orchestrator.decision_sink is None
    assert unarchived.archive is None
    assert unarchived.history.maxlen == 512


def test_history_size_uses_deque_natural_validation_and_accepts_zero():
    """History capacity accepts zero and otherwise follows deque validation."""
    engine = RecordingEngine()

    assert ExecutionOrchestrator(engine, history_size=0).history.maxlen == 0
    with pytest.raises(ValueError):
        ExecutionOrchestrator(engine, history_size=-1)
    with pytest.raises(TypeError):
        ExecutionOrchestrator(engine, history_size=1.5)


def test_connect_overwrites_collaborators_without_calling_them():
    """Connecting is passive and later connections replace earlier ones."""
    orchestrator = ExecutionOrchestrator(RecordingEngine())
    calls = []

    def first_sink(decision):
        calls.append(("first", decision))

    def second_sink(decision):
        calls.append(("second", decision))

    first_controller = RecordingController(calls)
    second_controller = RecordingController(calls)

    assert orchestrator.connect(None, None) is None
    assert orchestrator.controller is None
    assert orchestrator.decision_sink is None
    assert orchestrator.connect(first_controller, first_sink) is None
    assert calls == []
    assert orchestrator.connect(second_controller, second_sink) is None
    assert calls == []
    assert orchestrator.controller is second_controller
    assert orchestrator.decision_sink is second_sink


def test_registration_keeps_order_duplicates_and_has_no_unrelated_effects():
    """Blocking registration preserves every target without other mutations."""
    engine = RecordingEngine(now=4)
    orchestrator = ExecutionOrchestrator(engine, retain_all=True)

    assert orchestrator.register_blocked_operation(7, 5) is None
    orchestrator.register_blocked_operation(7, 5)
    orchestrator.register_blocked_operation(5, 5)
    orchestrator.register_blocked_operation(5, 7)
    orchestrator.register_blocked_operation(8, 7)

    assert orchestrator.blocked_by_index == {5: [7, 7, 5], 7: [5, 8]}
    assert list(orchestrator.history) == []
    assert orchestrator.archive == {}
    assert orchestrator.stats == {
        "outcomes": 0,
        "decisions": 0,
        "result_returns": 0,
    }
    assert engine.events == []


def test_blocked_results_pop_once_preserve_order_duplicates_and_take_priority():
    """Blocked dependents route first, in registration order, and consume their bucket."""
    engine = RecordingEngine(now=11)
    orchestrator = ExecutionOrchestrator(engine)
    payload = tuple([1, 0, 1])
    source = operation(4, name="measure", requires_return=True)
    decoded = result(
        payload,
        op_id=999,
        window_id=12,
        correction="ignored",
        soft_output="ignored",
        boundary_data="ignored",
    )
    orchestrator.register_blocked_operation(9, 4)
    orchestrator.register_blocked_operation(3, 4)
    orchestrator.register_blocked_operation(9, 4)

    decisions = orchestrator.on_result(source, decoded)

    assert [decision.target_operation_id for decision in decisions] == [9, 3, 9]
    assert all(decision.releases_operation for decision in decisions)
    assert 4 not in orchestrator.blocked_by_index
    assert orchestrator.stats == {
        "outcomes": 0,
        "decisions": 1,
        "result_returns": 0,
    }
    assert len(orchestrator.history) == 1
    assert orchestrator.history[0]["kind"] == "decision"
    assert orchestrator.history[0]["logical_observables"] is payload

    following = orchestrator.on_result(source, result(None))

    assert len(following) == 1
    assert following[0].target_operation_id == 4
    assert following[0].releases_operation is False
    assert orchestrator.stats["result_returns"] == 1
    assert orchestrator.history[-1]["logical_observables"] is None


def test_result_return_and_outcome_routes_record_complete_observables():
    """Unblocked results choose result return or outcome and preserve predictions."""
    engine = RecordingEngine(now=6)
    orchestrator = ExecutionOrchestrator(engine)
    payload = (0, 1, 1, 0)

    returned = orchestrator.on_result(
        operation(21, requires_return=True),
        result(payload, op_id=-1, correction=object(), window_id=-2),
    )
    outcome = orchestrator.on_result(
        operation(22, name="ordinary"),
        result(None, op_id=-1, boundary_data=object()),
    )

    assert len(returned) == 1
    assert returned[0].target_operation_id == 21
    assert returned[0].releases_operation is False
    assert outcome == []
    assert [record["kind"] for record in orchestrator.history] == [
        "result_return",
        "outcome",
    ]
    assert orchestrator.history[0]["logical_observables"] is payload
    assert orchestrator.history[1]["logical_observables"] is None
    assert orchestrator.stats == {
        "outcomes": 1,
        "decisions": 0,
        "result_returns": 1,
    }


def test_record_uses_current_clock_exact_fields_and_shared_archive_identity():
    """A record captures exact fields and is shared by history and archive."""
    engine = RecordingEngine(now=17)
    orchestrator = ExecutionOrchestrator(engine, retain_all=True)
    payload = (1,)

    assert orchestrator._record(operation(8, name="logical M"), "outcome", payload) is None

    record = orchestrator.history[0]
    assert record == {
        "t": 17,
        "op_id": 8,
        "name": "logical M",
        "kind": "outcome",
        "logical_observables": payload,
    }
    assert set(record) == {
        "t",
        "op_id",
        "name",
        "kind",
        "logical_observables",
    }
    assert orchestrator.archive[8] is record
    assert orchestrator.stats == {
        "outcomes": 1,
        "decisions": 0,
        "result_returns": 0,
    }


def test_bounded_history_evicts_but_archive_keeps_latest_record_per_id():
    """Bounded history evicts old entries while archive and counters keep accounting."""
    engine = RecordingEngine(now=1)
    orchestrator = ExecutionOrchestrator(engine, history_size=2, retain_all=True)

    orchestrator.on_result(operation(1, name="first"), result((1,)))
    first_archive_record = orchestrator.archive[1]
    engine.now = 2
    orchestrator.on_result(operation(2, name="second", requires_return=True), result((0,)))
    engine.now = 3
    orchestrator.register_blocked_operation(30, 1)
    orchestrator.register_blocked_operation(31, 1)
    orchestrator.on_result(operation(1, name="replacement"), result((1, 1)))

    assert [(record["op_id"], record["t"]) for record in orchestrator.history] == [
        (2, 2),
        (1, 3),
    ]
    assert orchestrator.archive[1] is orchestrator.history[-1]
    assert orchestrator.archive[1] is not first_archive_record
    assert orchestrator.archive[1]["name"] == "replacement"
    assert set(orchestrator.archive) == {1, 2}
    assert orchestrator.stats == {
        "outcomes": 1,
        "decisions": 1,
        "result_returns": 1,
    }


def test_zero_history_still_counts_and_archives_attempted_records():
    """Zero-capacity history drops records without dropping accounting or archive state."""
    orchestrator = ExecutionOrchestrator(
        RecordingEngine(now=3), history_size=0, retain_all=True
    )
    orchestrator.register_blocked_operation(2, 1)
    orchestrator.register_blocked_operation(3, 1)

    decisions = orchestrator.on_result(operation(1), result((1, 0)))

    assert len(decisions) == 2
    assert list(orchestrator.history) == []
    assert orchestrator.stats["decisions"] == 1
    assert orchestrator.archive[1] == {
        "t": 3,
        "op_id": 1,
        "name": "decode",
        "kind": "decision",
        "logical_observables": (1, 0),
    }


def test_unknown_record_kind_commits_history_and_archive_before_failing():
    """An unknown kind fails only after its record has been appended and archived."""
    orchestrator = ExecutionOrchestrator(RecordingEngine(now=9), retain_all=True)

    with pytest.raises(KeyError, match="mystery"):
        orchestrator._record(operation(5), "mystery", (1,))

    assert len(orchestrator.history) == 1
    assert orchestrator.history[0]["kind"] == "mystery"
    assert orchestrator.archive[5] is orchestrator.history[0]
    assert orchestrator.stats == {
        "outcomes": 0,
        "decisions": 0,
        "result_returns": 0,
    }


def test_connected_integration_logs_then_relays_in_decision_order_with_labels():
    """Connected dispatch logs route labels before relaying each decision in order."""
    events = []
    engine = RecordingEngine(now=13, events=events)
    controller = RecordingController(events)
    sink = object()
    orchestrator = ExecutionOrchestrator(engine)
    orchestrator.connect(controller, sink)
    orchestrator.register_blocked_operation(12, 7)
    orchestrator.register_blocked_operation(4, 7)

    assert orchestrator.integrate(operation(7), result((1,))) is None
    assert [event[0] for event in events] == ["log", "relay", "log", "relay"]
    assert events[0] == (
        "log",
        "Orchestrator",
        "DISPATCH conditional release for op#12 -> controller -> controller sequencer",
    )
    assert events[2] == (
        "log",
        "Orchestrator",
        "DISPATCH conditional release for op#4 -> controller -> controller sequencer",
    )
    assert [events[index][1].target_operation_id for index in (1, 3)] == [12, 4]
    assert events[1][2] is sink
    assert events[3][2] is sink

    events.clear()
    orchestrator.integrate(operation(9, requires_return=True), result((0,)))

    assert events[0] == (
        "log",
        "Orchestrator",
        "DISPATCH result return for op#9 -> controller -> controller sequencer",
    )
    assert events[1][0] == "relay"
    assert events[1][1].releases_operation is False


def test_disconnected_outcome_still_records_without_dispatch():
    """An outcome needing no decision records successfully without a connection."""
    engine = RecordingEngine(now=2)
    orchestrator = ExecutionOrchestrator(engine)

    assert orchestrator.integrate(operation(1), result((0,))) is None

    assert orchestrator.stats["outcomes"] == 1
    assert orchestrator.history[0]["logical_observables"] == (0,)
    assert engine.events == []


def test_missing_controller_fails_after_dependency_pop_record_and_log():
    """A generated decision with no controller fails after route state is committed."""
    engine = RecordingEngine(now=5)
    orchestrator = ExecutionOrchestrator(engine)
    orchestrator.register_blocked_operation(10, 4)

    with pytest.raises(AttributeError):
        orchestrator.integrate(operation(4), result((1,)))

    assert 4 not in orchestrator.blocked_by_index
    assert orchestrator.stats["decisions"] == 1
    assert orchestrator.history[0]["kind"] == "decision"
    assert engine.events == [
        (
            "log",
            "Orchestrator",
            "DISPATCH conditional release for op#10 -> controller -> controller sequencer",
        )
    ]


def test_missing_sink_fails_through_controller_after_route_commit():
    """A controller that calls a missing sink exposes failure after recording."""
    events = []

    class CallingController:
        def relay_instruction(self, decision, sink):
            events.append(("relay", decision.target_operation_id, sink))
            sink(decision)

    engine = RecordingEngine(now=8, events=events)
    orchestrator = ExecutionOrchestrator(engine)
    orchestrator.connect(CallingController(), None)

    with pytest.raises(TypeError):
        orchestrator.integrate(operation(6, requires_return=True), result((0, 1)))

    assert orchestrator.stats["result_returns"] == 1
    assert orchestrator.history[0]["logical_observables"] == (0, 1)
    assert [event[0] for event in events] == ["log", "relay"]
    assert events[1] == ("relay", 6, None)


def test_logger_failure_propagates_after_recording_and_before_relay():
    """Logger failure leaves route accounting committed and prevents relay."""
    relays = []

    class FailingEngine:
        now = 14

        def log(self, component, message):
            raise RuntimeError("log failed")

    orchestrator = ExecutionOrchestrator(FailingEngine())
    orchestrator.connect(RecordingController(relays), object())

    with pytest.raises(RuntimeError, match="log failed"):
        orchestrator.integrate(operation(2, requires_return=True), result((1,)))

    assert orchestrator.stats["result_returns"] == 1
    assert orchestrator.history[0]["t"] == 14
    assert relays == []


def test_relay_failure_propagates_after_prior_targets_and_single_source_commit():
    """Relay failure preserves earlier dispatches and the single source record."""
    events = []

    class FailingSecondController:
        def __init__(self):
            self.calls = 0

        def relay_instruction(self, decision, sink):
            self.calls += 1
            events.append(("relay", decision.target_operation_id))
            if self.calls == 2:
                raise RuntimeError("relay failed")

    engine = RecordingEngine(now=20, events=events)
    orchestrator = ExecutionOrchestrator(engine)
    orchestrator.connect(FailingSecondController(), object())
    orchestrator.register_blocked_operation(30, 3)
    orchestrator.register_blocked_operation(31, 3)
    orchestrator.register_blocked_operation(32, 3)

    with pytest.raises(RuntimeError, match="relay failed"):
        orchestrator.integrate(operation(3), result((1, 1)))

    assert [event[0] for event in events] == ["log", "relay", "log", "relay"]
    assert [event[1] for event in events if event[0] == "relay"] == [30, 31]
    assert 3 not in orchestrator.blocked_by_index
    assert len(orchestrator.history) == 1
    assert orchestrator.stats["decisions"] == 1


def test_mismatched_result_ids_duplicate_integration_and_late_registration_are_unchecked():
    """Routing does not validate result identity, repeated integration, or late dependency state."""
    orchestrator = ExecutionOrchestrator(RecordingEngine(now=1), retain_all=True)
    source = operation(40, name="source")

    orchestrator.on_result(
        source,
        result(
            (1,),
            op_id=999,
            window_id=-5,
            correction="unused",
            soft_output="unused",
            boundary_data="unused",
        ),
    )
    first_record = orchestrator.archive[40]
    orchestrator.engine.now = 2
    orchestrator.on_result(source, result((0,), op_id=-1))
    orchestrator.register_blocked_operation(40, 40)
    orchestrator.register_blocked_operation(88, 40)
    orchestrator.register_blocked_operation(88, 40)

    assert orchestrator.stats["outcomes"] == 2
    assert len(orchestrator.history) == 2
    assert orchestrator.archive[40] is orchestrator.history[-1]
    assert orchestrator.archive[40] is not first_record
    assert [record["op_id"] for record in orchestrator.history] == [40, 40]
    assert orchestrator.blocked_by_index[40] == [40, 88, 88]
