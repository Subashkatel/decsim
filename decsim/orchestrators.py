"""Orchestrators integrate decoded results and return feedback instructions."""

from __future__ import annotations
 
from collections import deque
from typing import Callable, Optional, TYPE_CHECKING
 
from .engine import Engine
from .message import Operation, DecodeResult, Decision, WindowPlan
 
if TYPE_CHECKING:
    from .protocols import Controller, ExecutionPlanner, WorkloadManager


class ExecutionOrchestrator:
    """Prepare execution, integrate decode results, and return feedback."""

    def __init__(self, engine: Engine, history_size: int = 512, retain_all: bool = False):
        """Hold result state and the feedback-blocking map."""

        self.engine = engine
        self.pauli_frame: dict = {}
        self.outcomes: dict[int, int] = {}
        self.blocked_by_index: dict[int, list[int]] = {}
        self.history: deque = deque(maxlen=history_size)
        self.stats: dict[str, int] = {
            "frame_updates": 0,
            "outcomes": 0,
            "decisions": 0,
            "result_returns": 0,
        }
        self.retain_all = retain_all
        self.archive: Optional[dict] = {} if retain_all else None
        self.controller: Optional["Controller"] = None
        self.decision_sink: Optional[Callable] = None

    def connect(self, controller: "Controller", decision_sink: Callable) -> None:
        """Wire the decision return path: this orchestrator's decisions are relayed
        through `controller` (paying t_oc + t_cq) and delivered to `decision_sink`
        (the chip's on_decision callback in the default wiring)."""
        self.controller = controller
        self.decision_sink = decision_sink

    def register_blocked_operation(self, blocked_op_id: int, blocking_op_id: int) -> None:
        """Record that one operation waits on another operation's decode result."""
        self.blocked_by_index.setdefault(blocking_op_id, []).append(blocked_op_id)
 
    def prepare_execution(self, *, operations: list[Operation],
                          cluster: "WorkloadManager",
                          planner: "ExecutionPlanner",
                          decode_operations: Optional[list[Operation]] = None) -> WindowPlan:
        """Prepare the decode workload before runtime starts.

        The orchestrator owns this handoff in the DecLat execution model. It
        delegates window construction to the supplied planner, then loads the
        resulting job plan into the decoder cluster. This costs zero simulated
        ticks because it happens before syndrome data is produced.
        """
        planned_operations = decode_operations if decode_operations is not None else operations
        for operation in planned_operations:
            cluster.register_op(operation)

        plan = planner.plan(planned_operations)
        self._log_prepared_execution(plan)
        cluster.load_execution_plan(plan)
        return plan

    def _log_prepared_execution(self, plan: "WindowPlan") -> None:
        """Log the off-path plan handoff to the decoder cluster."""
        self.engine.log("Orchestrator",
                        f"compiled execution plan off the reaction path (0 ticks); sending "
                        f"{plan.total_windows} decode job(s) across {len(plan.window_count)} operation(s) "
                        f"to the decoder cluster ahead of time")
 
    def _record_and_gc(self, op: Operation, kind: str, outcome: int,
                       basis: Optional[str] = None) -> None:
        """Save a compact record of this result (for debugging / audit), then free the live
        per-op frame/outcome state. The live dicts otherwise grow with the circuit an
        unbounded leak at utility scale."""
        
        # This records the result in a compact form for debugging and audit.
        rec = {"t": self.engine.now, "op_id": op.id, "name": op.name,
               "kind": kind, "outcome": outcome, "basis": basis}
        self.history.append(rec)
        if self.archive is not None:
            self.archive[op.id] = rec
        stat_key = {
            "frame_update": "frame_updates",
            "decision": "decisions",
            "result_return": "result_returns",
            "outcome": "outcomes",
        }[kind]
        self.stats[stat_key] += 1
        self.outcomes.pop(op.id, None)
        self.pauli_frame.pop(op.id, None)
 
    def integrate(self, op: Operation, result: DecodeResult) -> None:
        """Integrate a result and send any released feedback decisions."""
        for decision in self.on_result(op, result):
            if self.controller is None or self.decision_sink is None:
                continue
            instruction = "conditional release" if decision.releases_operation \
                else "result return"
            self.engine.log("Orchestrator",
                            f"DISPATCH {instruction} for "
                            f"op#{decision.target_operation_id}: "
                            f"basis '{decision.basis}' -> controller -> chip")
            self.controller.relay_instruction(decision, self.decision_sink)
 
    def on_result(self, op: Operation, result: DecodeResult) -> list[Decision]:
        """Save an outcome and return decisions for operations it unblocks."""
        
        outcome = result.logical_value if result.logical_value is not None else 1
        self.outcomes[op.id] = outcome
        if op.clifford:
            self.pauli_frame[op.id] = "frame-updated"
            self.engine.log("Orchestrator",
                            f"result for {op.name}: Pauli-frame update (Clifford); "
                            f"stays here, no instruction returns to the QPU")
            self._record_and_gc(op, "frame_update", outcome)
            return []

        self.engine.log("Orchestrator",
                        f"result for {op.name}: non-Clifford outcome decoded")
        blocked_ops = self.blocked_by_index.pop(op.id, [])
        if blocked_ops:
            basis = "X" if outcome else "Z"
            targets = ", ".join(f"op#{g}" for g in blocked_ops)
            self.engine.log("Orchestrator",
                            f"decides basis '{basis}' for {targets} and releases the chip")
            self._record_and_gc(op, "decision", outcome, basis)
            return [Decision(target_operation_id=g, basis=basis)
                    for g in blocked_ops]
        if op.requires_result_return_to_chip:
            basis = "X" if outcome else "Z"
            self.engine.log("Orchestrator",
                            f"result for {op.name} must return to the chip; "
                            f"sending basis '{basis}'")
            self._record_and_gc(op, "result_return", outcome, basis)
            return [Decision(
                target_operation_id=op.id,
                basis=basis,
                releases_operation=False,
            )]
        self._record_and_gc(op, "outcome", outcome)
        return []
