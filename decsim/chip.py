"""Chip-side operation scheduler and syndrome source driver."""


from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from .config import IDLE_ROUND_MODES, us
from .engine import Engine
from .message import Decision, Operation, SyndromePayload
from .pauli_frame import PauliFrame

if TYPE_CHECKING:
    from .protocols import (Controller, MagicStateFactory, SyndromeSource,
                            WorkloadManager)


class Chip:
    """Run operations, emit syndrome rounds, and unblock feedback-dependent work."""

    def __init__(self, engine: Engine, device: "SyndromeSource", controller: Controller,
                 cluster: WorkloadManager, factory: MagicStateFactory,
                 round_ticks: int, code_distance: int,
                 idle_round_mode: str = "ignore",
                 max_idle_rounds: Optional[int] = None,
                 gates_start_on_round_boundaries: bool = False):
        """Wire the chip to the device, controller, decoder manager, and factory.

        Operation length comes from cluster.rounds_for(op), not from the chip.
        That keeps the physical round stream aligned with the planner.
        """
        self.engine = engine
        self.device = device
        self.controller = controller
        self.cluster = cluster
        self.factory = factory
        self.round_ticks = round_ticks
        self.code_distance = code_distance
        if idle_round_mode not in IDLE_ROUND_MODES:
            raise ValueError(
                f"idle_round_mode must be one of {IDLE_ROUND_MODES} "
                f"(got {idle_round_mode!r})")
        self.idle_round_mode = idle_round_mode
 
        self.ops: dict[int, Operation] = {}
        self._deps_remaining: dict[int, int] = {}
        self._op_successors: dict[int, list[int]] = {}
        self.busy_qubits: dict[int, int] = {}
        self.requested: set[int] = set()
        self.state_ready: set[int] = set()
        self.started: set[int] = set()
        self.done_bodies: set[int] = set()
        self.decode_released: set[int] = set()
        self.body_done_time: dict[int, int] = {}
        self.decode_release_time: dict[int, int] = {}
        self.result_return_time_by_operation: dict[int, int] = {}
        # Pauli frame for feed-forward corrections; shared with the orchestrator by
        # the wiring (private one when the chip is built standalone).
        self.frame = PauliFrame()
        self.applied_basis: dict[int, str] = {}      # op_id -> steered meas. basis
        self.applied_pauli: dict[int, str] = {}      # op_id -> corrective byproduct
        self.applied_s: dict[int, bool] = {}         # op_id -> corrective Clifford S
        # per-successor frame delta the decision applied (x, z), for audit / fold
        self.applied_frame_delta: dict[int, tuple] = {}
        self.frame_applied_time: dict[int, int] = {}
        self.op_start_time: dict[int, int] = {}
        self.idle_rounds_by_patch: dict = {}
        self.idle_cap_hits: list[dict] = []
        self.gates_start_on_round_boundaries = gates_start_on_round_boundaries
        self._patches_emitting: set = set()
        self._stream_next_round: dict = {}
        self.last_finish_time = 0
        self.max_idle_rounds = max_idle_rounds if max_idle_rounds is not None \
            else 100 * code_distance

    def _round_ticks_for(self, operation: Operation) -> int:
        """Syndrome-round time for one operation."""
        round_us = getattr(self.cluster.layout.code_for_op(operation), "round_us", None)
        return us(round_us) if round_us is not None else self.round_ticks

    def _round_ticks_for_patch(self, patch) -> int:
        """Syndrome-round time for one idling patch."""
        round_us = getattr(self.cluster.layout.code_for_patch(patch), "round_us", None)
        return us(round_us) if round_us is not None else self.round_ticks

    def load(self, ops: list[Operation]) -> None:
        """Register operations, build dependencies, then start dependency roots."""
        self._register_operations(ops)
        self._build_dependency_graph(ops)
        self.cluster.build_windows()
        self._start_dependency_roots(ops)

    def _register_operations(self, ops: list[Operation]) -> None:
        """Give every operation to the chip and decoder manager."""
        for operation in ops:
            self.ops[operation.id] = operation
            self.cluster.register_op(operation)

    def _build_dependency_graph(self, ops: list[Operation]) -> None:
        """Build body-done dependency counters from operation predecessors."""
        for operation in ops:
            self._deps_remaining[operation.id] = len(operation.predecessors)
            self._op_successors[operation.id] = []
        for operation in ops:
            for predecessor_id in operation.predecessors:
                self._op_successors[predecessor_id].append(operation.id)

    def _start_dependency_roots(self, ops: list[Operation]) -> None:
        """Start operations that have no body dependency."""
        for operation in ops:
            if self._deps_remaining[operation.id] == 0:
                self._attempt_start(operation)

    def _release_successors(self, operation: Operation) -> None:
        """Decrement successors' body-dependency counters; magic-state/feedback waits overlap in _maybe_begin."""  # ref: arXiv:2411.04270
        for successor_id in self._op_successors[operation.id]:
            self._deps_remaining[successor_id] -= 1
            if self._deps_remaining[successor_id] == 0:
                self._attempt_start(self.ops[successor_id])

    def _attempt_start(self, operation: Operation) -> None:
        """Reserve qubits and fetch the magic state while feedback may still be pending."""
        self._mark_qubits_busy(operation)
        self.requested.add(operation.id)
        if operation.needs_magic_state:
            self.engine.log("Chip", f"{operation.name} needs a magic state; asking the factory")
            self.factory.request(
                operation.id,
                lambda ready_operation=operation: self._on_state_ready(ready_operation),
            )
        else:
            self._on_state_ready(operation)

    def _mark_qubits_busy(self, operation: Operation) -> None:
        """Mark this operation's qubits busy until its body is done.

        A conflict means two operations share a qubit with no ordering edge. That is
        an invalid operation list, so the simulator fails instead of choosing an order.
        """
        for qubit in operation.qubits:
            if qubit in self.busy_qubits:
                if self.busy_qubits[qubit] == operation.id:
                    raise RuntimeError(
                        f"{operation.name} lists qubit {qubit} more than once: "
                        f"{operation.qubits}")
                holder = self.ops[self.busy_qubits[qubit]].name
                raise RuntimeError(
                    f"{operation.name} and {holder} share qubit {qubit} but have no dependency "
                    f"edge. The operation list is missing program-order wiring "
                    f"(run it through _wire_circuit / a frontend)")
            self.busy_qubits[qubit] = operation.id

    def _on_state_ready(self, operation: Operation) -> None:
        """The magic state (if any) is in hand; begin if the reaction dependency is met too."""
        self.state_ready.add(operation.id)
        self._maybe_begin(operation)

    def _maybe_begin(self, operation: Operation) -> None:
        """Begin once magic-state and feedback waits are both clear."""
        if operation.id in self.started or operation.id not in self.state_ready:
            return
        if operation.blocked_by is not None and operation.id not in self.decode_released:
            return
        if self._must_wait_for_round_boundary(operation):
            return
        self._begin(operation)

    def _must_wait_for_round_boundary(self, operation: Operation) -> bool:
        """True if the gate must defer to the next round boundary because its patch is mid-round."""
        if not self.gates_start_on_round_boundaries:
            return False
        patch = self._patch_for_operation(operation)
        return patch in self._patches_emitting
 
    def _begin(self, operation: Operation) -> None:
        """Run an operation's syndrome rounds, then mark its body done."""
        self.started.add(operation.id)
        self.op_start_time[operation.id] = self.engine.now
        idle_rounds = self._consume_idle_rounds(operation)
        prepend = getattr(self.cluster, "prepend_idle_rounds", None)
        if idle_rounds and prepend is not None:
            prepend(operation.id, idle_rounds)
        self._reserve_stream_rounds(operation)
        self.device.begin_operation(operation)
        kind = "Clifford" if operation.clifford else "non-Clifford"
        release_note = "" if operation.blocked_by is None \
            else f" [unblocked by op#{operation.blocked_by}]"
        self.engine.log(
            "Chip",
            f"START {operation.name}  ({kind}, qubits {operation.qubits}){release_note}",
        )
        self.engine.schedule(
            self._round_ticks_for(operation),
            lambda: self._round(operation, 1),
            label=f"round1({operation.name})",
        )

    def _consume_idle_rounds(self, operation: Operation) -> int:
        """Return idle rounds that accumulated on this operation's patch or qubits."""
        patch_ids = operation.patches if operation.patches else operation.qubits
        return sum(self.idle_rounds_by_patch.pop(patch, 0)
                   for patch in patch_ids)

    def _reserve_stream_rounds(self, operation: Operation) -> None:
        """Assign this operation's range in its live syndrome stream."""
        if operation.stream_id is None:
            return

        stream_id = operation.stream_id
        next_round = self._stream_next_round.get(stream_id, 0)
        if operation.stream_offset is None:
            operation.stream_offset = next_round
        elif operation.stream_offset < next_round:
            raise RuntimeError(
                f"{operation.name} starts at stream round {operation.stream_offset + 1}, "
                f"but stream {stream_id!r} has already reserved through round {next_round}")

        operation_end = operation.stream_offset + self.cluster.rounds_for(operation)
        self._stream_next_round[stream_id] = max(next_round, operation_end)

    def _patch_for_operation(self, operation: Operation):
        """Return the patch used for idle emission."""
        if operation.patches:
            return operation.patches[0]
        if operation.qubits:
            return operation.qubits[0]
        return 0

    def _round(self, operation: Operation, round_index: int) -> None:
        """Emit one syndrome round through the controller to the decoder cluster."""
        total_rounds = self.cluster.rounds_for(operation)
        payloads = self.device.round_payloads(operation, round_index)

        self.engine.log("Chip", f"{operation.name} fires round {round_index}/{total_rounds}")
        self._relay_syndrome_payloads(payloads)
        if round_index < total_rounds:
            self.engine.schedule(
                self._round_ticks_for(operation),
                lambda: self._round(operation, round_index + 1),
                label=f"round{round_index + 1}({operation.name})",
            )
        else:
            self._body_done(operation)

    def _body_done(self, operation: Operation) -> None:
        """Finish the operation body and start any successors that are now clear."""
        self._record_body_done(operation)
        self._free_qubits(operation)
        self._release_successors(operation)
        self._log_qpu_finished_if_needed()
        self._close_feedback_boundary_if_needed(operation)
        self._start_idle_stream_if_needed(operation)
        self._seal_finished_streams_if_needed()

    def _record_body_done(self, operation: Operation) -> None:
        """Record physical completion for this operation body."""
        self.done_bodies.add(operation.id)
        self.body_done_time[operation.id] = self.engine.now
        self.last_finish_time = max(self.last_finish_time, self.engine.now)
        self.engine.log("Chip", f"{operation.name} body done")

    def _free_qubits(self, operation: Operation) -> None:
        """Release qubits before any successor tries to reserve them."""
        for qubit in operation.qubits:
            del self.busy_qubits[qubit]

    def _log_qpu_finished_if_needed(self) -> None:
        """Log the moment all physical operation bodies are complete."""
        if len(self.done_bodies) == len(self.ops):
            self.engine.log("Chip",
                            f"QPU finished. All {len(self.ops)} operations are physically "
                            f"complete; decoder may still be draining.")

    def _close_feedback_boundary_if_needed(self, operation: Operation) -> None:
        """Close a live stream boundary when the operation's final measurement does it."""
        if operation.feedback_boundary_mode != "measurement_closed":
            return
        if not self._has_waiting_blocked_successor(operation.id):
            return
        if operation.stream_id is None:
            return

        close_stream_boundary = getattr(self.cluster, "close_stream_boundary", None)
        if close_stream_boundary is None:
            return

        stream_round_count = operation.stream_offset + self.cluster.rounds_for(operation)
        close_stream_boundary(operation.stream_id, stream_round_count)

    def _start_idle_stream_if_needed(self, operation: Operation) -> None:
        """Start memory rounds while a feedback-blocked successor waits."""
        if not operation.has_successor:
            return
        if not self._has_waiting_blocked_successor(operation.id):
            return

        self.engine.log("Chip",
                        f"{operation.name} patch idles (successor blocked on a decode); "
                        f"emitting memory rounds every round until the correction returns")
        patch = self._patch_for_operation(operation)
        self._patches_emitting.add(patch)
        self.engine.schedule(
            self._round_ticks_for_patch(patch),
            lambda operation_id=operation.id, patch_id=patch:
            self._emit_idle_round(operation_id, patch_id, 1),
            label=f"idle-tick({operation.name},1)")

    def _has_waiting_blocked_successor(self, op_id: int) -> bool:
        """Return True while this operation has a successor still waiting on decode."""
        for successor_id in self._op_successors[op_id]:
            successor = self.ops[successor_id]
            if successor.blocked_by is None or successor.id in self.started:
                continue
            if successor.id not in self.decode_released:
                return True
            if self.gates_start_on_round_boundaries:
                return True
        return False

    def _emit_idle_round(self, op_id: int, patch, round_index: int) -> None:
        """Emit one idle memory round, then schedule the next one if still needed."""
        if not self._has_waiting_blocked_successor(op_id):
            self._stop_idle_stream(patch)
            return

        if self._idle_round_cap_reached(op_id, patch, round_index):
            return

        self._relay_idle_round(op_id, patch, round_index)
        self._record_idle_round(patch)
        self._start_released_successors_on_boundary(op_id, patch)
        self._submit_idle_decode_if_due(op_id, patch, round_index)
        self._schedule_next_idle_round(op_id, patch, round_index)

    def _stop_idle_stream(self, patch) -> None:
        """Mark that this patch no longer has an active idle emitter."""
        self._patches_emitting.discard(patch)

    def _idle_round_cap_reached(self, op_id: int, patch, round_index: int) -> bool:
        """Stop idle emission if the safety cap has fired."""
        if round_index <= self.max_idle_rounds:
            return False

        self._stop_idle_stream(patch)
        self.idle_cap_hits.append({
            "time": self.engine.now,
            "op_id": op_id,
            "patch": patch,
            "round_index": round_index,
            "max_idle_rounds": self.max_idle_rounds,
        })
        self.engine.log("Chip",
                        f"WARNING: {self.ops[op_id].name} hit the idle-round cap "
                        f"(max_idle_rounds={self.max_idle_rounds}) with its blocked "
                        f"successor still waiting. No more memory rounds will be "
                        f"emitted, so decoder load and backlog past this point are understated. "
                        f"Raise max_idle_rounds for long-reaction studies.")
        return True

    def _relay_idle_round(self, op_id: int, patch, round_index: int) -> None:
        """Send one idle syndrome round through the same controller path."""
        if self._relay_idle_round_to_live_stream(op_id, patch):
            return

        payload = SyndromePayload(op_id, patch, round_index)
        self.controller.relay_syndrome(
            payload,
            lambda p, source_op_id=op_id: self.cluster.on_memory_round(source_op_id))

    def _relay_idle_round_to_live_stream(self, op_id: int, patch) -> bool:
        """Send an idle round into the source operation's live stream when enabled."""
        if self.idle_round_mode != "extend_stream":
            return False

        operation = self.ops[op_id]
        stream_id = operation.stream_id
        if stream_id is None:
            return False

        has_dynamic_stream = getattr(self.cluster, "has_dynamic_stream", None)
        if has_dynamic_stream is None or not has_dynamic_stream(stream_id):
            return False

        global_round = self._stream_next_round.get(stream_id, 0) + 1
        self._stream_next_round[stream_id] = global_round
        payloads = self.device.idle_round_payloads(
            operation, stream_id, global_round, patch)
        self._relay_syndrome_payloads(payloads)
        return True

    def _relay_syndrome_payloads(self, payloads: list[SyndromePayload]) -> None:
        """Send all fragments from one syndrome round through the controller."""
        for payload in payloads:
            payload.n_fragments = len(payloads)
            self.controller.relay_syndrome(payload, self.cluster.on_syndrome_arrival)

    def _record_idle_round(self, patch) -> None:
        """Count one measured idle round on the patch."""
        self.idle_rounds_by_patch[patch] = self.idle_rounds_by_patch.get(patch, 0) + 1

    def _start_released_successors_on_boundary(self, op_id: int, patch) -> None:
        """Start successors that became ready while the idle stream was mid-round."""
        if self.gates_start_on_round_boundaries:
            for successor_id in self._op_successors[op_id]:
                successor = self.ops[successor_id]
                if (successor.blocked_by is not None and successor.id not in self.started
                        and successor.id in self.decode_released
                        and successor.id in self.state_ready):
                    self._stop_idle_stream(patch)
                    self._begin(successor)

    def _submit_idle_decode_if_due(self, op_id: int, patch, round_index: int) -> None:
        """Submit one memory-window decode after each commit region of idle rounds."""
        if self.idle_round_mode != "separate_decode_jobs":
            return

        code = self.cluster.layout.code_for_patch(patch)
        if round_index % code.commit_rounds() == 0:
            self.cluster.submit_decode(
                code.commit_rounds() + code.buffer_rounds(),
                on_done=lambda: None, code=code.name,
                spatial_nodes=code.spatial_nodes(1),
                label=f"mem({self.ops[op_id].name},r{round_index})")

    def _schedule_next_idle_round(self, op_id: int, patch, round_index: int) -> None:
        """Schedule the next idle tick."""
        self.engine.schedule(
            self._round_ticks_for_patch(patch),
            lambda operation_id=op_id, patch_id=patch, completed_round=round_index:
            self._emit_idle_round(operation_id, patch_id, completed_round + 1),
            label=f"idle-tick({self.ops[op_id].name},{round_index + 1})",
        )

    def _seal_finished_streams_if_needed(self) -> None:
        """Seal live dynamic streams once all scheduled bodies have finished."""
        if len(self.done_bodies) != len(self.ops):
            return

        seal_stream = getattr(self.cluster, "seal_stream", None)
        if seal_stream is None:
            return

        has_dynamic_stream = getattr(self.cluster, "has_dynamic_stream", None)
        if has_dynamic_stream is None:
            return
        for stream_id, total_rounds in list(self._stream_next_round.items()):
            if not has_dynamic_stream(stream_id):
                continue
            seal_stream(stream_id, total_rounds)

    def on_decision(self, decision: Decision) -> None:
        """Receive feedback from the controller."""
        if decision.releases_operation:
            self._release_blocked_operation(decision)
            return
        self._record_result_return(decision)

    def _release_blocked_operation(self, decision: Decision) -> None:
        """Consume a feedback decision (steer basis + fold correction), then release."""
        operation_id = decision.target_operation_id
        target = self.ops[operation_id]
        self._consume_decision(target, decision)
        self.decode_released.add(operation_id)
        self.decode_release_time[operation_id] = self.engine.now
        self.engine.log(
            "Chip",
            f"CONSUMED decision for {target.name}: measure basis "
            f"'{decision.basis}', frame byproduct '{decision.pauli}'"
            f"{' + S' if decision.apply_s else ''} on qubits {target.qubits}"
            f"{' [strong-commit marker]' if decision.strong_committed else ''}; "
            f"successor steered, now trying to start")
        self._maybe_begin(target)

    def _consume_decision(self, target: Operation, decision: Decision) -> None:
        """Steer the successor and fold the correction into its Pauli frame."""
        op_id = target.id
        self.applied_basis[op_id] = decision.basis
        self.applied_pauli[op_id] = decision.pauli
        self.applied_s[op_id] = bool(decision.apply_s)
        primary = target.qubits[0] if target.qubits else None
        x_before = self.frame.x_of(primary) if primary is not None else 0
        z_before = self.frame.z_of(primary) if primary is not None else 0
        for qubit in target.qubits:
            self.frame.apply_pauli(qubit, decision.pauli)
            if decision.apply_s:
                self.frame.apply_s(qubit)
        delta = (0, 0) if primary is None else (
            self.frame.x_of(primary) ^ x_before,
            self.frame.z_of(primary) ^ z_before)
        self.applied_frame_delta[op_id] = delta
        self.frame_applied_time[op_id] = self.engine.now

    def _record_result_return(self, decision: Decision) -> None:
        """Record a decoded result that returns to the chip without starting another op."""
        operation_id = decision.target_operation_id
        self.result_return_time_by_operation[operation_id] = self.engine.now
        target = self.ops[operation_id]
        self.engine.log("Chip",
                        f"received result return for {target.name}: "
                        f"basis '{decision.basis}'")
