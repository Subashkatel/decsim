"""Default orchestrator (port 15): resolve/Decision algebra.

Part module: faithful port of orchestrators.ExecutionOrchestrator's result
integration (the compile-time prepare_execution half moves to loop.py).

Reviewed behavior (Contract 3 rule 3 — do not "fix"): a None
logical value publishes as outcome 1 — stream-segment deliveries always carry
logical_value=None and therefore publish 1 ("assume correction needed").
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional

from .message import Decision, DecodeResult, Operation
from .pauli_frame import PauliFrame


class ExecutionOrchestrator:
    """Integrate decode results; emit Decisions for operations they unblock."""

    def __init__(self, engine, history_size: int = 512, retain_all: bool = False):
        self.engine = engine
        # Per-qubit Pauli frame: decoded corrections accumulate here and the
        # chip folds feed-forward byproducts in (shared object in the wiring).
        self.frame = PauliFrame()
        #: op_id -> intrinsic destructive magic-state measurement bit; empty => 0.
        self.magic_measurements: dict[int, int] = {}
        self.blocked_by_index: dict[int, list[int]] = {}
        self.history: deque = deque(maxlen=history_size)
        self.stats: dict[str, int] = {
            "frame_updates": 0, "outcomes": 0,
            "decisions": 0, "result_returns": 0}
        self.retain_all = retain_all
        self.archive: Optional[dict] = {} if retain_all else None
        self.controller = None
        self.decision_sink: Optional[Callable] = None

    def connect(self, controller, decision_sink: Callable) -> None:
        """Wire the decision return path (pays t_oc + t_cq via the controller)."""
        self.controller = controller
        self.decision_sink = decision_sink

    def register_blocked_operation(self, blocked_op_id: int,
                                   blocking_op_id: int) -> None:
        """Record that one operation waits on another's decode result."""
        self.blocked_by_index.setdefault(blocking_op_id, []).append(blocked_op_id)

    # ------------------------------------------------------------ integration

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
    def _byproduct(op: Operation, outcome: int) -> tuple:
        """(Pauli byproduct, apply_s) of a teleported gate: measurement 1 ->
        op.byproduct_pauli + corrective S; 0 -> identity (Litinski 1808.02892)."""
        if outcome:
            return op.byproduct_pauli, True
        return "I", False

    def on_result(self, op: Operation, result: DecodeResult) -> list:
        """Save an outcome and return decisions for operations it unblocks."""
        outcome = result.logical_value if result.logical_value is not None else 1
        qubit = op.qubits[0] if op.qubits else 0
        if op.clifford:
            self.frame.accumulate(qubit, x=outcome)
            self.engine.log("Orchestrator",
                            f"result for {op.name}: Pauli-frame update (Clifford); "
                            f"frame[q{qubit}] X^={outcome}, no instruction to the QPU")
            self._record_and_gc(op, "frame_update", outcome)
            return []

        m_raw = self.magic_measurements.get(op.id, 0) if op.needs_magic_state else 0
        measurement = (outcome ^ m_raw) & 1
        self.engine.log("Orchestrator",
                        f"result for {op.name}: non-Clifford measurement decoded "
                        f"(decoded={outcome}, intrinsic={m_raw}, "
                        f"believed={measurement})")
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
            return [Decision(target_operation_id=op.id, basis=basis,
                             releases_operation=False, pauli=pauli,
                             apply_s=apply_s, correction_value=measurement)]
        self._record_and_gc(op, "outcome", outcome)
        return []

    def _record_and_gc(self, op: Operation, kind: str, outcome: int,
                       basis: Optional[str] = None) -> None:
        """Compact audit record, then free live per-op state (leak guard)."""
        rec = {"t": self.engine.now, "op_id": op.id, "name": op.name,
               "kind": kind, "outcome": outcome, "basis": basis}
        self.history.append(rec)
        if self.archive is not None:
            self.archive[op.id] = rec
        stat_key = {"frame_update": "frame_updates", "decision": "decisions",
                    "result_return": "result_returns", "outcome": "outcomes"}[kind]
        self.stats[stat_key] += 1
