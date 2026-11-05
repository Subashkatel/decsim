"""The final-result gate: a decision starts once earlier ones are final.

A decision is an operation whose outcome is needed: one another operation
is blocked_by, or one whose result the QPU needs back. Under a provisional
release (windows/operation_results.py) a decision lets its waiters go on
the weak answers of the decodes it awaits, and the strong answers follow.
A T gate applied on a wrong frame leaves "a logical Clifford gate" as its
correction, learned "right before the next logical two-qubit gate or T
gate" (Chamberland, Iyer and Poulin 1704.06662 Sec. 2); in a Clifford
frame that correction turns the axis of the next non-Clifford rotation,
which is fixed when the rotation starts (Litinski 1808.02892 p. 5). So
the gate holds each decision's start until every decision it descends
from has its final result, and the frame is repaired before then.

The decisions a decision descends from are the nearest ones along its
parents: its predecessors, completion and patch order as the lowered
program wires them (frontends/circuit_frontend.py), and its blocker.
Each counts as not commuting with it, so the gate holds at least as long
as one that read the program's commutation would. A final notice crosses
frame_to_controller as a decision does (pauli_frame/decision_dispatch.py)
and drives no pulse; the operations it frees start on the runtime's next
try.

A bounded gate holds a decision for at most its limit, timed from the
first time it is asked. Then the decision starts on the weak answers,
and a strong answer that lands later repairs nothing. This is the
fallback that bounds the worst case: when the strong tier falls behind,
the program keeps its pace and loses only the answers that came late.

StartHold is the controller package's own seam: the issuer asks it.
"""

import functools
from typing import Optional, Protocol

import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.log_sources as log_sources
import decsim.records.program as program_records


class StartHold(Protocol):
    """The gate, as the issuer asks it before an operation starts."""

    def is_open(self, operation: program_records.Operation) -> bool:
        """Whether every decision the operation descends from is final."""


class FinalResultGate:
    """Holds a decision until the decisions it descends from are final."""

    dispatch = ports.Port(ports.DecisionDispatch)
    runtime = ports.Port(ports.OperationRuntime)

    def __init__(
        self,
        engine: engine_module.Engine,
        operations: tuple,
        hold_limit_ticks: Optional[int] = None,
    ) -> None:
        self.engine = engine
        self.earlier_by_decision = earlier_decisions(operations)
        # each decision whose final result is at the controller, to the tick
        # it landed
        self.final_ticks: dict = {}
        # None holds a decision until the finals are in
        self.hold_limit_ticks = hold_limit_ticks
        # the decisions whose bounded hold has begun, and those it ended
        self.bounded_ids: set = set()
        self.expired_ids: set = set()

    def note_result_final(self, operation: program_records.Operation) -> None:
        """A decision's awaited decodes are final: the notice leaves."""
        landed = functools.partial(self.open_after, operation.id)
        self.dispatch.dispatch_final_result(operation.id, landed)

    def is_open(self, operation: program_records.Operation) -> bool:
        """Whether every decision the operation descends from is final."""
        earlier = self.earlier_by_decision.get(operation.id, ())
        for decision_id in earlier:
            if decision_id not in self.final_ticks:
                return self._has_reached_its_limit(operation.id)
        return True

    def expire(self, operation_id: int) -> None:
        """The hold reached its limit: the decision starts on weak answers."""
        self.expired_ids.add(operation_id)
        self.engine.log(
            log_sources.CONTROLLER,
            f"op#{operation_id} held to its limit: starts on weak answers",
        )
        self.runtime.retry_ready_operations()

    def _has_reached_its_limit(self, operation_id: int) -> bool:
        """Whether a bounded hold ended; the first refusal starts its clock."""
        if operation_id in self.expired_ids:
            return True
        if self.hold_limit_ticks is None:
            return False
        if operation_id in self.bounded_ids:
            return False
        self.bounded_ids.add(operation_id)
        expire = functools.partial(self.expire, operation_id)
        self.engine.schedule(self.hold_limit_ticks, expire, label="hold limit")
        return False

    def open_after(self, operation_id: int) -> None:
        """The decision's final result is at the controller: retry starts."""
        self.final_ticks[operation_id] = self.engine.now
        self.engine.log(
            log_sources.CONTROLLER,
            f"final result of op#{operation_id} at the controller",
        )
        self.runtime.retry_ready_operations()


def earlier_decisions(operations: tuple) -> dict:
    """Each decision's id, to the nearest decisions it descends from.

    The operations come in the lowered program's order, each after its
    parents, so one pass fills every operation's nearest decisions.
    """
    ordered_ids = program_records.decision_ids(operations)
    decision_ids = set(ordered_ids)
    nearest_by_operation = {}
    for operation in operations:
        nearest = set()
        for parent_id in _parents(operation):
            assert parent_id in nearest_by_operation, (
                f"{operation.name} comes before its parent op#{parent_id}"
            )
            reached = _reached_through(
                parent_id, decision_ids, nearest_by_operation
            )
            nearest.update(reached)
        nearest_by_operation[operation.id] = frozenset(nearest)
    earlier_by_decision = {}
    for decision_id in ordered_ids:
        nearest = nearest_by_operation[decision_id]
        earlier_by_decision[decision_id] = tuple(sorted(nearest))
    return earlier_by_decision


def _parents(operation: program_records.Operation) -> list:
    """The operations it follows: its predecessors and its blocker."""
    parents = list(operation.predecessors)
    if operation.blocked_by is not None:
        parents.append(operation.blocked_by)
    return parents


def _reached_through(
    parent_id: int, decision_ids: set, nearest_by_operation: dict
) -> frozenset:
    """The parent itself if it decides, else the decisions it descends from."""
    if parent_id in decision_ids:
        return frozenset((parent_id,))
    return nearest_by_operation[parent_id]
