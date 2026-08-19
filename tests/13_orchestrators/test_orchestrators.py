"""Behavior tests for conditional release: a final result releases the
operations conditioned on it, over the controller."""

from types import SimpleNamespace

from decsim.pauli_frame.conditional_release import ConditionalRelease


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
    return SimpleNamespace(id=op_id, name=name, requires_result_return_to_qpu=requires_return)


def result(logical_observables=None, **fields):
    return SimpleNamespace(logical_observables=logical_observables, **fields)


def test_registration_keeps_order_and_duplicates():
    unit = ConditionalRelease(RecordingEngine())
    unit.register_blocked_operation(9, 4)
    unit.register_blocked_operation(3, 4)
    unit.register_blocked_operation(9, 4)
    assert unit.blocked_by_index == {4: [9, 3, 9]}


def test_blocked_results_release_once_in_order_and_take_priority_over_return():
    """Blocked dependents are released first, in registration order, and the
    bucket is consumed; a later result of the same source falls back to a
    result return when the operation asks for one."""
    unit = ConditionalRelease(RecordingEngine(now=11))
    source = operation(4, name="measure", requires_return=True)
    unit.register_blocked_operation(9, 4)
    unit.register_blocked_operation(3, 4)
    unit.register_blocked_operation(9, 4)

    decisions = unit.on_result(source, result((1, 0, 1), correction="ignored"))

    assert [decision.target_operation_id for decision in decisions] == [9, 3, 9]
    assert all(decision.releases_operation for decision in decisions)
    assert 4 not in unit.blocked_by_index

    following = unit.on_result(source, result(None))
    assert [(d.target_operation_id, d.releases_operation) for d in following] == [(4, False)]

    assert unit.on_result(operation(5), result((0,))) == []


def test_connected_integration_logs_then_relays_in_decision_order():
    events = []
    engine = RecordingEngine(now=13, events=events)
    controller = RecordingController(events)
    sink = object()
    unit = ConditionalRelease(engine)
    unit.connect(controller, sink)
    unit.register_blocked_operation(12, 7)
    unit.register_blocked_operation(4, 7)

    assert unit.integrate(operation(7), result((1,))) is None
    assert [event[0] for event in events] == ["log", "relay", "log", "relay"]
    assert events[0] == (
        "log", "PauliFrame",
        "DISPATCH conditional release for op#12 -> controller -> controller sequencer")
    assert events[2] == (
        "log", "PauliFrame",
        "DISPATCH conditional release for op#4 -> controller -> controller sequencer")
    assert [events[index][1].target_operation_id for index in (1, 3)] == [12, 4]
    assert events[1][2] is sink and events[3][2] is sink

    events.clear()
    unit.integrate(operation(9, requires_return=True), result((0,)))
    assert events[0] == (
        "log", "PauliFrame",
        "DISPATCH result return for op#9 -> controller -> controller sequencer")
    assert events[1][0] == "relay" and events[1][1].releases_operation is False

    events.clear()
    unit.integrate(operation(2), result((0,)))
    assert events == []
