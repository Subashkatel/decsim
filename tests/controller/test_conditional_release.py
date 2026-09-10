"""Conditional release: which operations one final result lets go.

When an operation's final result is in, every operation blocked on it
may start, one decision each; when nothing waits but the QPU itself
needs the outcome, the one decision is a result return. The value of the
outcome never enters: a conditional instruction starts once its
dependency is fully decoded, whatever it decoded to (Caune et al.
2410.05202 lines 364-367 and 1255-1262: the program stalls on the
decoder's status register, then the conditional gate costs the same
whichever way the result came out), so nothing here reads the result's
bits.

Every decision travels to the QPU through the controller, never straight
from the frame, which is why the unit holds the controller it relays
over and logs one line per decision before handing it on.
"""

import types

import decsim.controller.conditional_release as conditional_release


class RecordingEngine:
    """A clock a test sets, and the lines the unit narrated."""

    def __init__(self, now=0):
        self.now = now
        self.lines = []

    def log(self, component, message):
        self.lines.append((component, message))


class RecordingController:
    """The relay: which decisions reached the controller, and over what."""

    def __init__(self):
        self.relayed = []

    def relay_instruction(self, decision, deliver_decision):
        self.relayed.append((decision.target_operation_id, deliver_decision))


def operation(operation_id, name="decode", requires_return=False):
    return types.SimpleNamespace(
        id=operation_id,
        name=name,
        requires_result_return_to_qpu=requires_return,
    )


def bare_release():
    """The unit with no controller wired: only its decisions are read."""
    engine = RecordingEngine()
    return conditional_release.ConditionalRelease(engine)


def connected_release():
    """The unit wired to a recording controller and one delivery path."""
    engine = RecordingEngine(now=13)
    controller = RecordingController()
    deliver_decision = object()
    unit = conditional_release.ConditionalRelease(engine)
    unit.connect(controller, deliver_decision)
    return unit, engine, controller, deliver_decision


def test_every_waiting_operation_is_registered_once_per_edge():
    """Two operations may wait on one result, and one may wait twice."""
    unit = bare_release()
    unit.register_blocked_operation(9, 4)
    unit.register_blocked_operation(3, 4)
    unit.register_blocked_operation(9, 4)
    assert unit.waiting_by_blocker == {4: [9, 3, 9]}


def test_a_result_releases_its_waiters_in_registration_order():
    unit = bare_release()
    unit.register_blocked_operation(9, 4)
    unit.register_blocked_operation(3, 4)

    source = operation(4)
    decisions = unit.decisions_for(source)

    released = [decision.target_operation_id for decision in decisions]
    assert released == [9, 3]
    assert all(decision.releases_operation for decision in decisions)


def test_a_released_waiter_is_not_released_again():
    """The waiters are consumed, so a second result releases nobody."""
    unit = bare_release()
    unit.register_blocked_operation(9, 4)
    source = operation(4)

    unit.decisions_for(source)

    assert 4 not in unit.waiting_by_blocker
    assert unit.decisions_for(source) == []


def test_a_waiter_comes_before_the_qpus_own_result_return():
    """A release carries the outcome onward; a return only reports it."""
    unit = bare_release()
    source = operation(4, requires_return=True)
    unit.register_blocked_operation(9, 4)

    releases = unit.decisions_for(source)
    returns = unit.decisions_for(source)

    assert [decision.target_operation_id for decision in releases] == [9]
    assert [decision.target_operation_id for decision in returns] == [4]
    assert releases[0].releases_operation is True
    assert returns[0].releases_operation is False


def test_a_result_nobody_waits_for_and_the_qpu_ignores_releases_nothing():
    unit = bare_release()
    unheard_of = operation(5)
    assert unit.decisions_for(unheard_of) == []


def test_every_decision_reaches_the_qpu_through_the_controller():
    unit, engine, controller, deliver_decision = connected_release()
    unit.register_blocked_operation(12, 7)
    unit.register_blocked_operation(4, 7)
    source = operation(7)

    unit.release_waiters(source)

    relayed = [operation_id for operation_id, _sink in controller.relayed]
    sinks = [sink for _operation_id, sink in controller.relayed]
    assert relayed == [12, 4]
    assert sinks == [deliver_decision, deliver_decision]


def test_each_decision_is_narrated_before_it_is_relayed():
    unit, engine, controller, _deliver = connected_release()
    unit.register_blocked_operation(12, 7)
    source = operation(7)

    unit.release_waiters(source)

    assert engine.lines == [
        (
            "PauliFrame",
            "DISPATCH conditional release for op#12 -> controller "
            "-> controller sequencer",
        )
    ]


def test_a_result_return_is_narrated_as_a_return():
    unit, engine, controller, _deliver = connected_release()

    source = operation(9, requires_return=True)
    unit.release_waiters(source)

    assert engine.lines == [
        (
            "PauliFrame",
            "DISPATCH result return for op#9 -> controller "
            "-> controller sequencer",
        )
    ]
    assert controller.relayed[0][0] == 9
