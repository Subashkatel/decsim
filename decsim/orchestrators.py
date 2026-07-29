"""Default orchestrator: publish predictions and form feedback effects."""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional

from .message import (
    Decision,
    DecodeResult,
    FeedbackEffect,
    Operation,
    RunSeedChild,
    RunSeedPathSegment,
    same_stable_identity,
)
from .pauli_frame import PauliFrame


class ExecutionOrchestrator:
    """Integrate decode results; emit Decisions for operations they unblock."""

    def __init__(self, engine, history_size: int = 512, retain_all: bool = False):
        self.engine = engine
        # Per-qubit Pauli frame: decoded corrections accumulate here and the
        # chip folds feed-forward byproducts in (shared object in the wiring).
        self.frame = PauliFrame()
        self.blocked_by_index: dict[int, list[int]] = {}
        self.history: deque = deque(maxlen=history_size)
        self.stats: dict[str, int] = {
            "frame_updates": 0, "outcomes": 0,
            "decisions": 0, "result_returns": 0}
        self.retain_all = retain_all
        self.archive: Optional[dict] = {} if retain_all else None
        self.controller = None
        self.decision_sink: Optional[Callable] = None

    def run_seed_children(self):
        """Expose the retained frame that determines feedback behavior."""
        return (
            RunSeedChild(
                (RunSeedPathSegment("field", "frame"),),
                self.frame,
            ),
        )

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
            effect = decision.effect
            detail = "timing-only" if effect is None \
                else (
                    f"observable {effect.logical_observable_index}, "
                    f"basis '{effect.basis}'"
                )
            self.engine.log(
                "Orchestrator",
                f"DISPATCH {instruction} for "
                f"op#{decision.target_operation_id}: {detail} "
                f"-> controller -> chip",
            )
            self.controller.relay_instruction(decision, self.decision_sink)

    @staticmethod
    def _byproduct(op: Operation, outcome: int) -> tuple:
        """(Pauli byproduct, apply_s) of a teleported gate: measurement 1 ->
        op.byproduct_pauli + corrective S; 0 -> identity (Litinski 1808.02892)."""
        if outcome:
            return op.byproduct_pauli, True
        return "I", False

    def on_result(self, op: Operation, result: DecodeResult) -> list:
        """Publish one complete prediction and form effects only when needed."""
        logical_observables = result.logical_observables
        blocked_operations = self.blocked_by_index.get(op.id, [])
        if logical_observables is None:
            return self._handle_timing_only_result(
                op,
                blocked_operations,
            )

        has_scalar_consumer = bool(
            blocked_operations
            or op.clifford
            or op.requires_result_return_to_chip
        )
        if not has_scalar_consumer:
            self._record(
                op,
                "outcome",
                logical_observables,
                selected_observable_index=None,
                effect=None,
            )
            return []

        observable_index, decoded_value = self._select_observable(
            op,
            logical_observables,
        )
        if blocked_operations:
            effect = self._feedback_effect(
                op,
                observable_index,
                decoded_value,
            )
            targets = ", ".join(
                f"op#{operation_id}"
                for operation_id in blocked_operations
            )
            self.engine.log(
                "Orchestrator",
                f"result for {op.name}: observable {observable_index} "
                f"decoded as {decoded_value}; basis '{effect.basis}', "
                f"byproduct '{effect.pauli}'"
                f"{' + S' if effect.apply_s else ''} for {targets}",
            )
            self.blocked_by_index.pop(op.id)
            self._record(
                op,
                "decision",
                logical_observables,
                selected_observable_index=observable_index,
                effect=effect,
            )
            return [
                Decision(
                    target_operation_id=operation_id,
                    effect=effect,
                    strong_committed=bool(op.requires_strong_commit),
                )
                for operation_id in blocked_operations
            ]

        qubit = op.qubits[0] if op.qubits else 0
        if op.clifford:
            self.frame.accumulate(qubit, x=decoded_value)
            self.engine.log(
                "Orchestrator",
                f"result for {op.name}: observable {observable_index} "
                f"updates the Pauli frame (Clifford); "
                f"frame[q{qubit}] X^={decoded_value}, no QPU instruction",
            )
            self._record(
                op,
                "frame_update",
                logical_observables,
                selected_observable_index=observable_index,
                effect=None,
            )
            return []

        if op.requires_result_return_to_chip:
            effect = self._feedback_effect(
                op,
                observable_index,
                decoded_value,
            )
            self.engine.log(
                "Orchestrator",
                f"result for {op.name} must return to the chip; "
                f"sending observable {observable_index}, "
                f"basis '{effect.basis}'",
            )
            self._record(
                op,
                "result_return",
                logical_observables,
                selected_observable_index=observable_index,
                effect=effect,
            )
            return [
                Decision(
                    target_operation_id=op.id,
                    effect=effect,
                    releases_operation=False,
                )
            ]
        raise RuntimeError(
            f"operation {op.id} selected a logical observable without a "
            "scalar consumer")

    def _handle_timing_only_result(
        self,
        op: Operation,
        blocked_operations: list[int],
    ) -> list[Decision]:
        if blocked_operations:
            self.blocked_by_index.pop(op.id)
            self.engine.log(
                "Orchestrator",
                f"timing-only result for {op.name} releases "
                f"{len(blocked_operations)} blocked operation(s) without "
                "a functional effect",
            )
            self._record(
                op,
                "decision",
                None,
                selected_observable_index=None,
                effect=None,
            )
            return [
                Decision(
                    target_operation_id=operation_id,
                    effect=None,
                    strong_committed=bool(op.requires_strong_commit),
                )
                for operation_id in blocked_operations
            ]
        if op.requires_result_return_to_chip:
            self.engine.log(
                "Orchestrator",
                f"timing-only result for {op.name} returns without a "
                "functional effect",
            )
            self._record(
                op,
                "result_return",
                None,
                selected_observable_index=None,
                effect=None,
            )
            return [
                Decision(
                    target_operation_id=op.id,
                    effect=None,
                    releases_operation=False,
                )
            ]
        self._record(
            op,
            "outcome",
            None,
            selected_observable_index=None,
            effect=None,
        )
        return []

    @staticmethod
    def _select_observable(
        op: Operation,
        logical_observables: tuple[int, ...],
    ) -> tuple[int, int]:
        observable_index = op.logical_observable_index
        if observable_index is None:
            if len(logical_observables) != 1:
                raise ValueError(
                    f"operation {op.id} needs logical_observable_index for "
                    f"a scalar consumer with {len(logical_observables)} "
                    "predicted observables")
            observable_index = 0
        if type(observable_index) is not int:
            raise TypeError(
                f"operation {op.id} logical_observable_index must be an "
                "exact int")
        if (
            observable_index < 0
            or observable_index >= len(logical_observables)
        ):
            raise ValueError(
                f"operation {op.id} logical_observable_index "
                f"{observable_index} is outside a prediction of length "
                f"{len(logical_observables)}")
        return observable_index, logical_observables[observable_index]

    def _feedback_effect(
        self,
        op: Operation,
        observable_index: int,
        decoded_value: int,
    ) -> FeedbackEffect:
        intrinsic_measurement = None
        intrinsic_value = 0
        if op.needs_magic_state:
            intrinsic_measurement = op.intrinsic_measurement
            if intrinsic_measurement is None:
                raise ValueError(
                    f"operation {op.id} needs intrinsic_measurement "
                    "provenance for functional magic-state feedback")
            expected_trajectory = (
                op.stream_id if op.stream_id is not None else op.id
            )
            if not same_stable_identity(
                intrinsic_measurement.operation_id,
                op.id,
            ):
                raise ValueError(
                    f"operation {op.id} intrinsic_measurement operation "
                    "identity does not match")
            if not same_stable_identity(
                intrinsic_measurement.trajectory_id,
                expected_trajectory,
            ):
                raise ValueError(
                    f"operation {op.id} intrinsic_measurement trajectory "
                    "identity does not match")
            intrinsic_value = intrinsic_measurement.value
        correction_value = decoded_value ^ intrinsic_value
        basis = "X" if correction_value else "Z"
        pauli, apply_s = self._byproduct(op, correction_value)
        return FeedbackEffect(
            logical_observable_index=observable_index,
            decoded_value=decoded_value,
            intrinsic_measurement=intrinsic_measurement,
            correction_value=correction_value,
            basis=basis,
            pauli=pauli,
            apply_s=apply_s,
        )

    def _record(
        self,
        op: Operation,
        kind: str,
        logical_observables: Optional[tuple[int, ...]],
        *,
        selected_observable_index: Optional[int],
        effect: Optional[FeedbackEffect],
    ) -> None:
        """Retain the complete prediction and any explicitly selected effect."""
        rec = {
            "t": self.engine.now,
            "op_id": op.id,
            "name": op.name,
            "kind": kind,
            "logical_observables": logical_observables,
            "selected_observable_index": selected_observable_index,
            "feedback_effect": effect,
        }
        self.history.append(rec)
        if self.archive is not None:
            self.archive[op.id] = rec
        stat_key = {"frame_update": "frame_updates", "decision": "decisions",
                    "result_return": "result_returns", "outcome": "outcomes"}[kind]
        self.stats[stat_key] += 1
