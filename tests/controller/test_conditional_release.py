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
from the frame, and it leaves by the frame's end of
frame_to_controller: the unit holds that dispatch and hands each
decision to it, with the delivery the runtime waits on
(tests/pauli_frame/test_decision_dispatch.py carries the send itself).
"""

import types

import decsim.controller.conditional_release as conditional_release


class RecordingEngine:
    """A clock a test sets."""

    def __init__(self, now=0):
        self.now = now


class RecordingDispatch:
    """The frame's end: which decisions were sent, and with what delivery."""

    def __init__(self):
        self.dispatched = []

    def dispatch_decision(self, decision, deliver_decision):
        self.dispatched.append((decision.target_operation_id, deliver_decision))


def operation(operation_id, name="decode", requires_return=False):
    return types.SimpleNamespace(
        id=operation_id,
        name=name,
        requires_result_return_to_qpu=requires_return,
    )


def bare_release():
    """The unit with no dispatch wired: only its decisions are read."""
    engine = RecordingEngine()
    return conditional_release.ConditionalRelease(engine)


def connected_release():
    """The unit wired to a recording dispatch and one delivery path."""
    engine = RecordingEngine(now=13)
    dispatch = RecordingDispatch()
    deliver_decision = object()
    unit = conditional_release.ConditionalRelease(engine)
    unit.connect(dispatch, deliver_decision)
    return unit, dispatch, deliver_decision


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


def test_every_decision_leaves_by_the_frames_end_with_its_delivery():
    unit, dispatch, deliver_decision = connected_release()
    unit.register_blocked_operation(12, 7)
    unit.register_blocked_operation(4, 7)
    source = operation(7)

    unit.release_waiters(source)

    sent = [operation_id for operation_id, _sink in dispatch.dispatched]
    sinks = [sink for _operation_id, sink in dispatch.dispatched]
    assert sent == [12, 4]
    assert sinks == [deliver_decision, deliver_decision]


def test_a_result_return_leaves_by_the_same_end():
    """Nothing waits, but the QPU needs the outcome: one return goes out."""
    unit, dispatch, _deliver = connected_release()

    source = operation(9, requires_return=True)
    unit.release_waiters(source)

    assert dispatch.dispatched[0][0] == 9
