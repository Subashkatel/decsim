"""The final result of an operation is its logical measurement. This unit
receives it and releases the operations conditioned on it: a "conditional
release" decision for each operation blocked by this one, or a "result
return" when the operation's outcome must travel back to the QPU. Each
decision goes over OC to the controller and CQ to the QPU, which is the
feedback part of the reaction time. The frame itself (what the outcome is)
is the Pauli frame's; the timing model never branches on the value, so this
unit reads none. XQsim builds the same split in hardware: a Pauli frame unit
that accumulates, and a logical measurement unit that interprets and feeds
forward."""

from __future__ import annotations

from typing import Callable, Optional

from ..message import Decision, DecodeResult, Operation


class LogicalMeasurements:
    def __init__(self, engine):
        self.engine = engine
        self.blocked_by_index: dict[int, list[int]] = {}
        self.controller = None
        self.decision_sink: Optional[Callable] = None

    def connect(self, controller, decision_sink: Callable) -> None:
        """Wire the decision return path: through the controller, into the sink."""
        self.controller = controller
        self.decision_sink = decision_sink

    def register_blocked_operation(self, blocked_op_id: int, blocking_op_id: int) -> None:
        """Record that one operation waits on another's logical measurement."""
        self.blocked_by_index.setdefault(blocking_op_id, []).append(blocked_op_id)

    def integrate(self, operation: Operation, result: DecodeResult) -> None:
        """A final result arrived: dispatch the decisions it releases."""
        for decision in self.on_result(operation, result):
            instruction = ("conditional release" if decision.releases_operation
                           else "result return")
            self.engine.log(
                "Orchestrator",
                f"DISPATCH {instruction} for "
                f"op#{decision.target_operation_id} -> controller -> controller sequencer",
            )
            self.controller.relay_instruction(decision, self.decision_sink)

    def on_result(self, operation: Operation, result: DecodeResult) -> list[Decision]:
        """The decisions one final result releases: one per blocked operation,
        else a result return when the QPU needs the outcome, else none."""
        blocked_operations = self.blocked_by_index.pop(operation.id, [])
        if blocked_operations:
            return [Decision(target_operation_id)
                    for target_operation_id in blocked_operations]
        if operation.requires_result_return_to_qpu:
            return [Decision(operation.id, releases_operation=False)]
        return []
