"""Orchestrators integrate decoded results and return feedback instructions."""

from __future__ import annotations
 
from collections import deque
from typing import Callable, Optional, TYPE_CHECKING
 
from .engine import Engine
from .message import Operation, DecodeResult, Decision, WindowPlan
from .pauli_frame import PauliFrame
 
if TYPE_CHECKING:
    from .protocols import Controller, ExecutionPlanner, WorkloadManager


class ExecutionOrchestrator:
    """Prepare execution, integrate decode results, and return feedback."""

    def __init__(self, engine: Engine, history_size: int = 512, retain_all: bool = False):
        """Hold result state and the feedback-blocking map."""

        self.engine = engine
        # Per-qubit Pauli frame: decoded corrections accumulate here and the chip
        # folds feed-forward byproducts in.
        self.frame = PauliFrame()
        #: op_id -> intrinsic destructive magic-state measurement bit (0/1); empty => 0.
        self.magic_measurements: dict[int, int] = {}
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
 
    @staticmethod
    def _byproduct(op: Operation, outcome: int) -> tuple[str, bool]:
        """The (Pauli byproduct, apply_s) a teleported gate feeds forward: a decoded
        measurement of 1 leaves ``op.byproduct_pauli`` + corrective S, a 0 the identity
        (gate-teleportation algebra, Litinski arXiv:1808.02892)."""
        if outcome:
            return op.byproduct_pauli, True
        return "I", False

    def on_result(self, op: Operation, result: DecodeResult) -> list[Decision]:
        """Save an outcome and return decisions for operations it unblocks."""

        outcome = result.logical_value if result.logical_value is not None else 1
        self.outcomes[op.id] = outcome
        qubit = op.qubits[0] if op.qubits else 0
        if op.clifford:
            # Per-qubit frame update: XOR the decoded correction in.
            self.frame.accumulate(qubit, x=outcome)
            self.engine.log("Orchestrator",
                            f"result for {op.name}: Pauli-frame update (Clifford); "
                            f"frame[q{qubit}] X^={outcome}, no instruction to the QPU")
            self._record_and_gc(op, "frame_update", outcome)
            return []

        # Feed-forward on the believed measurement: decoded value XOR the intrinsic
        # magic-state bit (no intrinsic bit -> just the decoded value).
        m_raw = self.magic_measurements.get(op.id, 0) if op.needs_magic_state else 0
        measurement = (outcome ^ m_raw) & 1
        self.engine.log("Orchestrator",
                        f"result for {op.name}: non-Clifford measurement decoded "
                        f"(decoded={outcome}, intrinsic={m_raw}, believed={measurement})")
        pauli, apply_s = self._byproduct(op, measurement)
        blocked_ops = self.blocked_by_index.pop(op.id, [])
        if blocked_ops:
            basis = "X" if measurement else "Z"
            targets = ", ".join(f"op#{g}" for g in blocked_ops)
            self.engine.log("Orchestrator",
                            f"decides basis '{basis}', byproduct '{pauli}'"
                            f"{' + S' if apply_s else ''} for {targets} and "
                            f"releases the chip")
            self._record_and_gc(op, "decision", outcome, basis)
            return [Decision(target_operation_id=g, basis=basis, pauli=pauli,
                             apply_s=apply_s, correction_value=measurement,
                             strong_committed=bool(op.requires_strong_commit))
                    for g in blocked_ops]
        if op.requires_result_return_to_chip:
            basis = "X" if measurement else "Z"
            self.engine.log("Orchestrator",
                            f"result for {op.name} must return to the chip; "
                            f"sending basis '{basis}'")
            self._record_and_gc(op, "result_return", outcome, basis)
            return [Decision(
                target_operation_id=op.id,
                basis=basis,
                releases_operation=False,
                pauli=pauli,
                apply_s=apply_s,
                correction_value=measurement,
            )]
        self._record_and_gc(op, "outcome", outcome)
        return []
