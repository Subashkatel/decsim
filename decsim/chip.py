"""Chip-side operation scheduler and syndrome source driver."""


from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from .config import us
from .engine import Engine
from .message import Decision, Operation, SyndromePayload

if TYPE_CHECKING:
    from .protocols import (Controller, DeviceModel, MagicStateFactory,
                            WorkloadManager)


class Chip:
    """Run operations, emit syndrome rounds, and unblock feedback-dependent work."""

    def __init__(self, engine: Engine, device: DeviceModel, controller: Controller,
                 cluster: WorkloadManager, factory: MagicStateFactory,
                 round_ticks: int, code_distance: int,
                 decode_idle_rounds: bool = False,
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
        # DecLat counts idle stabilization rounds during feedback waits. This flag
        # decides whether those rounds also create decoder jobs.
        self.decode_idle_rounds = decode_idle_rounds
 
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
        self.idle_rounds_by_patch: dict = {}
        self.gates_start_on_round_boundaries = gates_start_on_round_boundaries
        self._patches_emitting: set = set()
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
        """This operation's body finished, so successors wait on one fewer body.

        Only data dependencies live here. Magic-state waits and feedback waits are handled later
        in _maybe_begin, so those waits overlap rather than stack (arXiv:2411.04270)."""
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
        """Under gates_start_on_round_boundaries, a gate whose patch is mid-round (its
        idle emitter still ticking) begins at the next round boundary, so the patch's
        syndrome stream stays on one grid."""
        if not self.gates_start_on_round_boundaries:
            return False
        patch = self._patch_for_operation(operation)
        return patch in self._patches_emitting
 
    def _begin(self, operation: Operation) -> None:
        """Run an operation's syndrome rounds, then mark its body done."""
        self.started.add(operation.id)
        idle_rounds = self._consume_idle_rounds(operation)
        prepend = getattr(self.cluster, "prepend_idle_rounds", None)
        if idle_rounds and prepend is not None:
            prepend(operation.id, idle_rounds)
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
        emit = getattr(self.device, "round_payloads", None)
        if emit is not None:
            payloads = emit(operation, round_index)
        else:
            payloads = [self.device.round_payload(operation, round_index)]

        self.engine.log("Chip", f"{operation.name} fires round {round_index}/{total_rounds}")
        for payload in payloads:
            payload.n_fragments = len(payloads)
            self.controller.relay_syndrome(payload, self.cluster.on_syndrome_arrival)
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
        self._start_idle_stream_if_needed(operation)

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
        self.engine.log("Chip",
                        f"WARNING: {self.ops[op_id].name} hit the idle-round cap "
                        f"(max_idle_rounds={self.max_idle_rounds}) with its blocked "
                        f"successor still waiting. No more memory rounds will be "
                        f"emitted, so decoder load and backlog past this point are understated. "
                        f"Raise max_idle_rounds for long-reaction studies.")
        return True

    def _relay_idle_round(self, op_id: int, patch, round_index: int) -> None:
        """Send one idle syndrome round through the same controller path."""
        payload = SyndromePayload(op_id, patch, round_index)
        self.controller.relay_syndrome(
            payload,
            lambda p, source_op_id=op_id: self.cluster.on_memory_round(source_op_id))

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
        if self.decode_idle_rounds:
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

    def on_decision(self, decision: Decision) -> None:
        """A correction came back: release the blocked gate. It begins immediately if its magic
        state has already arrived (fetched in parallel during the reaction); otherwise it begins
        when the state lands. _maybe_begin enforces the AND of the two conditions."""
        self.decode_released.add(decision.gadget_id)
        self.decode_release_time[decision.gadget_id] = self.engine.now
        target = self.ops[decision.gadget_id]
        self.engine.log("Chip",
                        f"received basis '{decision.basis}' for {target.name}; trying to start")
        self._maybe_begin(target)
