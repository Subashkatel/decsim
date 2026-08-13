"""Default orchestrator for decoded predictions and feedback timing."""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional

from .message import Decision, DecodeResult, Operation


class ExecutionOrchestrator:
    """Record final predictions and release operations that depend on them."""

    def __init__(self, engine, history_size: int = 512, retain_all: bool = False):
        self.engine = engine
        self.blocked_by_index: dict[int, list[int]] = {}
        self.history: deque = deque(maxlen=history_size)
        self.stats: dict[str, int] = {
            "outcomes": 0,
            "decisions": 0,
            "result_returns": 0,
        }
        self.archive: Optional[dict] = {} if retain_all else None
        self.controller = None
        self.decision_sink: Optional[Callable] = None

    def connect(self, controller, decision_sink: Callable) -> None:
        """Wire the decision return path through the controller."""
        self.controller = controller
        self.decision_sink = decision_sink

    def register_blocked_operation(
        self,
        blocked_op_id: int,
        blocking_op_id: int,
    ) -> None:
        """Record that one operation waits on another decoded result."""
        self.blocked_by_index.setdefault(blocking_op_id, []).append(blocked_op_id)

    def integrate(self, operation: Operation, result: DecodeResult) -> None:
        """Record a final result and dispatch its timing decisions."""
        for decision in self.on_result(operation, result):
            if self.controller is None or self.decision_sink is None:
                continue
            instruction = (
                "conditional release"
                if decision.releases_operation
                else "result return"
            )
            self.engine.log(
                "Orchestrator",
                f"DISPATCH {instruction} for "
                f"op#{decision.target_operation_id} -> controller -> controller sequencer",
            )
            self.controller.relay_instruction(decision, self.decision_sink)

    def on_result(
        self,
        operation: Operation,
        result: DecodeResult,
    ) -> list[Decision]:
        """Preserve the complete prediction and choose only its timing route."""
        blocked_operations = self.blocked_by_index.pop(operation.id, [])
        if blocked_operations:
            decisions = [
                Decision(target_operation_id)
                for target_operation_id in blocked_operations
            ]
            self._record(operation, "decision", result.logical_observables)
            return decisions
        if operation.requires_result_return_to_qpu:
            self._record(operation, "result_return", result.logical_observables)
            return [Decision(operation.id, releases_operation=False)]
        self._record(operation, "outcome", result.logical_observables)
        return []

    def _record(
        self,
        operation: Operation,
        kind: str,
        logical_observables: Optional[tuple[int, ...]],
    ) -> None:
        record = {
            "t": self.engine.now,
            "op_id": operation.id,
            "name": operation.name,
            "kind": kind,
            "logical_observables": logical_observables,
        }
        self.history.append(record)
        if self.archive is not None:
            self.archive[operation.id] = record
        stat_key = {
            "decision": "decisions",
            "result_return": "result_returns",
            "outcome": "outcomes",
        }[kind]
        self.stats[stat_key] += 1
