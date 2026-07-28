"""The chip-side gate: starts ops, consumes decisions, injects idle rounds.

This is the control half of the QPU seam. The other half — the clocked
per-op syndrome stream — is the Source (devices.py): the gate calls
``source.start(op)`` and hears ``on_body_done(op)`` back in the same event
as the op's final round. Everything pluggable arrives injected: idle
policy, controller relay, cluster facade, factory, source.

Behavior frozen by the timing goldens:
  - any Decision(releases_operation=True) releases the waiting op,
    unconditionally; requires_strong_commit marks the op but gates nothing.
  - while an op waits for feedback, idle rounds are emitted on the round
    clock (capped, 100*d by default), counted per qubit set, and at the
    next op's begin folded into its first window via
    cluster.prepend_idle_rounds.
  - the magic-state wait overlaps the feedback wait: an op starts once both
    are clear; the two waits never add.
  - resource reservation uses typed ResourceClaims; the default layout
    claims qubits only, and duplicate claims on a qubit raise.
"""

from __future__ import annotations

from typing import Optional

from .message import Decision, FeedbackEffect, Operation, SyndromePayload
from .pauli_frame import PauliFrame


class Chip:
    """Gate operation starts on body deps + magic states + feedback finality."""

    def __init__(self, engine, *, source, controller, cluster, factory,
                 round_ticks: int, code_geometry, resolved_operations,
                 resolved_patches, idle_policy,
                 resource_claims_by_operation_id,
                 max_idle_rounds: Optional[int] = None,
                 gates_start_on_round_boundaries: bool = False,
                 frame: Optional[PauliFrame] = None):
        self.engine = engine
        self.source = source
        self.controller = controller
        self.cluster = cluster
        self.factory = factory
        self.round_ticks = round_ticks
        self._code_geometry = code_geometry
        self.idle_policy = idle_policy
        self._resolved_operations = {
            operation.operation_id: operation
            for operation in resolved_operations
        }
        self._resolved_patches = {
            patch.patch_identity: patch
            for patch in resolved_patches
        }
        self._resource_claims_by_operation_id = dict(
            resource_claims_by_operation_id
        )
        self.gates_start_on_round_boundaries = gates_start_on_round_boundaries
        self.frame = frame if frame is not None else PauliFrame()
        self.max_idle_rounds = max_idle_rounds if max_idle_rounds is not None \
            else 100 * code_geometry.distance

        self._ops: dict[int, Operation] = {}
        self._deps_remaining: dict[int, int] = {}
        self._op_successors: dict[int, list[int]] = {}
        self.busy_claims: dict[tuple, int] = {}       # (kind, id) -> op_id
        self.requested: set[int] = set()
        self.state_ready: set[int] = set()
        self.started: set[int] = set()
        self.done_bodies: set[int] = set()
        self.decode_released: set[int] = set()
        self.body_done_time: dict[int, int] = {}
        self.decode_release_time: dict[int, int] = {}
        self.result_return_time_by_operation: dict[int, int] = {}
        self.applied_basis: dict[int, str] = {}
        self.applied_pauli: dict[int, str] = {}
        self.applied_s: dict[int, bool] = {}
        self.applied_frame_delta: dict[int, tuple] = {}
        self.frame_applied_time: dict[int, int] = {}
        self.op_start_time: dict[int, int] = {}
        self.idle_rounds_by_patch: dict = {}
        self.idle_rounds_emitted = 0
        self.idle_cap_hits: list[dict] = []
        self._patches_emitting: set = set()
        self.stream_next_round: dict = {}
        self.last_finish_time = 0

    # -------------------------------------------------------------- loading

    @property
    def workload_complete(self) -> bool:
        """Whether every loaded operation body reached physical completion."""
        return set(self._ops) == self.done_bodies

    def _round_ticks_for(self, operation: Operation) -> int:
        try:
            return self._resolved_operations[operation.id].round_ticks
        except KeyError as error:
            raise ValueError(
                f"operation {operation.id} has no resolved round cadence"
            ) from error

    def _round_ticks_for_patch(self, patch) -> int:
        try:
            return self._resolved_patches[patch].round_ticks
        except KeyError as error:
            raise ValueError(
                f"patch {patch!r} has no resolved round cadence"
            ) from error

    def _round_count_for(self, operation: Operation) -> int:
        try:
            return self._resolved_operations[operation.id].round_count
        except KeyError as error:
            raise ValueError(
                f"operation {operation.id} has no resolved round count"
            ) from error

    def _load(self, ops: list[Operation]) -> None:
        """Register operations, build dependencies, then start dependency roots."""
        for operation in ops:
            self._ops[operation.id] = operation
            self.cluster.register_op(operation)
        for operation in ops:
            self._deps_remaining[operation.id] = len(operation.predecessors)
            self._op_successors[operation.id] = []
        for operation in ops:
            for predecessor_id in operation.predecessors:
                self._op_successors[predecessor_id].append(operation.id)
        for operation in ops:
            if self._deps_remaining[operation.id] == 0:
                self._attempt_start(operation)

    # ---------------------------------------------------------------- start

    def _release_successors(self, operation: Operation) -> None:
        """Magic-state/feedback waits overlap in _maybe_begin (arXiv:2411.04270)."""
        for successor_id in self._op_successors[operation.id]:
            self._deps_remaining[successor_id] -= 1
            if self._deps_remaining[successor_id] == 0:
                self._attempt_start(self._ops[successor_id])

    def _attempt_start(self, operation: Operation) -> None:
        """Reserve resources and fetch the magic state while feedback may pend."""
        self._claim_resources(operation)
        self.requested.add(operation.id)
        if operation.needs_magic_state:
            self.engine.log("Chip", f"{operation.name} needs a magic state; "
                                    f"asking the factory")
            self.factory.request(
                operation.id,
                lambda ready_operation=operation:
                    self._on_state_ready(ready_operation))
        else:
            self._on_state_ready(operation)

    def _claim_resources(self, operation: Operation) -> None:
        """Reserve this op's typed ResourceClaims until its body is done.

        A conflict means two operations share a resource with no ordering edge
        — an invalid operation list, so the simulator fails instead of choosing."""
        seen = set()
        for qubit in operation.qubits:      # malformed op: X(q0,q0)
            if qubit in seen:
                raise RuntimeError(
                    f"{operation.name} lists qubit {qubit} more than "
                    f"once: {operation.qubits}")
            seen.add(qubit)
        for claim in self._resource_claims(operation):
            for rid in sorted(claim.ids, key=repr):
                key = (claim.kind, rid)
                if key in self.busy_claims:
                    holder = self._ops[self.busy_claims[key]].name
                    raise RuntimeError(
                        f"{operation.name} and {holder} share qubit {rid} but "
                        f"have no dependency edge. The operation list is missing "
                        f"program-order wiring (run it through _wire_circuit / "
                        f"a frontend)")
                self.busy_claims[key] = operation.id

    def _free_resources(self, operation: Operation) -> None:
        for claim in self._resource_claims(operation):
            for rid in claim.ids:
                self.busy_claims.pop((claim.kind, rid), None)

    def _resource_claims(self, operation: Operation):
        try:
            return self._resource_claims_by_operation_id[operation.id]
        except KeyError as error:
            raise ValueError(
                f"operation {operation.id} has no resolved resource claims"
            ) from error

    def _on_state_ready(self, operation: Operation) -> None:
        self.state_ready.add(operation.id)
        self._maybe_begin(operation)

    def _maybe_begin(self, operation: Operation) -> None:
        """Begin once magic-state and feedback waits are both clear."""
        if operation.id in self.started or operation.id not in self.state_ready:
            return
        if (operation.blocked_by is not None
                and operation.id not in self.decode_released):
            return
        if self._must_wait_for_round_boundary(operation):
            return
        self._begin(operation)

    def _must_wait_for_round_boundary(self, operation: Operation) -> bool:
        if not self.gates_start_on_round_boundaries:
            return False
        return self._patch_for_operation(operation) in self._patches_emitting

    def _begin(self, operation: Operation) -> None:
        """Consume idle rounds, reserve stream range, hand off to the Source."""
        self.started.add(operation.id)
        self.op_start_time[operation.id] = self.engine.now
        idle_rounds = self._consume_idle_rounds(operation)
        if idle_rounds:
            self.cluster.prepend_idle_rounds(operation.id, idle_rounds)
        self._reserve_stream_rounds(operation)
        kind = "Clifford" if operation.clifford else "non-Clifford"
        release_note = "" if operation.blocked_by is None \
            else f" [unblocked by op#{operation.blocked_by}]"
        self.engine.log("Chip", f"START {operation.name}  ({kind}, qubits "
                                f"{operation.qubits}){release_note}")
        self.source.start(operation, self._round_ticks_for(operation),
                          on_body_done=self._body_done)

    def _consume_idle_rounds(self, operation: Operation) -> int:
        """Sum-and-pop the idle rounds counted on this op's patches/qubits."""
        patch_ids = operation.patches if operation.patches else operation.qubits
        return sum(self.idle_rounds_by_patch.pop(patch, 0)
                   for patch in patch_ids)

    def _reserve_stream_rounds(self, operation: Operation) -> None:
        if operation.stream_id is None:
            return
        stream_id = operation.stream_id
        next_round = self.stream_next_round.get(stream_id, 0)
        if operation.stream_offset is None:
            operation.stream_offset = next_round
        elif operation.stream_offset < next_round:
            raise RuntimeError(
                f"{operation.name} starts at stream round "
                f"{operation.stream_offset + 1}, but stream {stream_id!r} has "
                f"already reserved through round {next_round}")
        operation_end = operation.stream_offset + self._round_count_for(
            operation
        )
        self.stream_next_round[stream_id] = max(next_round, operation_end)

    def _patch_for_operation(self, operation: Operation):
        if operation.patches:
            return operation.patches[0]
        if operation.qubits:
            return operation.qubits[0]
        return 0

    # ------------------------------------------------------------ body done

    def _body_done(self, operation: Operation) -> None:
        """The op's last body round arrived. Step order is frozen by the
        timing goldens: record -> free -> release -> log -> close boundary
        -> idle stream -> seal."""
        self.done_bodies.add(operation.id)
        self.body_done_time[operation.id] = self.engine.now
        self.last_finish_time = max(self.last_finish_time, self.engine.now)
        self.engine.log("Chip", f"{operation.name} body done")
        self._free_resources(operation)
        self._release_successors(operation)
        if len(self.done_bodies) == len(self._ops):
            self.engine.log("Chip",
                            f"QPU finished. All {len(self._ops)} operations are "
                            f"physically complete; decoder may still be draining.")
        self._close_feedback_boundary_if_needed(operation)
        self._start_idle_stream_if_needed(operation)
        self._seal_finished_streams_if_needed()

    def _close_feedback_boundary_if_needed(self, operation: Operation) -> None:
        if operation.feedback_boundary_mode != "measurement_closed":
            return
        if not self._has_waiting_blocked_successor(operation.id):
            return
        if operation.stream_id is None:
            return
        stream_round_count = operation.stream_offset \
            + self._round_count_for(operation)
        self.cluster.close_stream_boundary(operation.stream_id,
                                           stream_round_count)

    def _seal_finished_streams_if_needed(self) -> None:
        if len(self.done_bodies) != len(self._ops):
            return
        for stream_id, total_rounds in list(self.stream_next_round.items()):
            if not self.cluster.has_dynamic_stream(stream_id):
                continue
            self.cluster.seal_stream(stream_id, total_rounds)

    # --------------------------------------------------------- idle emission

    def _start_idle_stream_if_needed(self, operation: Operation) -> None:
        if not operation.has_successor:
            return
        if not self._has_waiting_blocked_successor(operation.id):
            return
        self.engine.log("Chip",
                        f"{operation.name} patch idles (successor blocked on a "
                        f"decode); emitting memory rounds every round until the "
                        f"correction returns")
        patch = self._patch_for_operation(operation)
        self._patches_emitting.add(patch)
        self.engine.schedule(
            self._round_ticks_for_patch(patch),
            lambda operation_id=operation.id, patch_id=patch:
                self._emit_idle_round(operation_id, patch_id, 1),
            label=f"idle-tick({operation.name},1)")

    def _has_waiting_blocked_successor(self, op_id: int) -> bool:
        for successor_id in self._op_successors[op_id]:
            successor = self._ops[successor_id]
            if successor.blocked_by is None or successor.id in self.started:
                continue
            if successor.id not in self.decode_released:
                return True
            if self.gates_start_on_round_boundaries:
                return True
        return False

    def _emit_idle_round(self, op_id: int, patch, round_index: int) -> None:
        """Emit one idle round. Step order is frozen by the timing goldens:
        guard -> cap -> relay -> account -> boundary-start -> separate-decode
        -> reschedule."""
        if not self._has_waiting_blocked_successor(op_id):
            self._patches_emitting.discard(patch)
            return
        if self._idle_round_cap_reached(op_id, patch, round_index):
            return
        self._relay_idle_round(op_id, patch, round_index)
        self.idle_rounds_by_patch[patch] = \
            self.idle_rounds_by_patch.get(patch, 0) + 1
        self.idle_rounds_emitted += 1
        self.idle_policy.account(1, self._ops[op_id])
        self._start_released_successors_on_boundary(op_id, patch)
        self._submit_idle_decode_if_due(op_id, patch, round_index)
        self.engine.schedule(
            self._round_ticks_for_patch(patch),
            lambda operation_id=op_id, patch_id=patch, done=round_index:
                self._emit_idle_round(operation_id, patch_id, done + 1),
            label=f"idle-tick({self._ops[op_id].name},{round_index + 1})")

    def _idle_round_cap_reached(self, op_id: int, patch,
                                round_index: int) -> bool:
        if round_index <= self.max_idle_rounds:
            return False
        self._patches_emitting.discard(patch)
        self.idle_cap_hits.append({
            "time": self.engine.now, "op_id": op_id, "patch": patch,
            "round_index": round_index,
            "max_idle_rounds": self.max_idle_rounds})
        self.engine.log("Chip",
                        f"WARNING: {self._ops[op_id].name} hit the idle-round cap "
                        f"(max_idle_rounds={self.max_idle_rounds}) with its "
                        f"blocked successor still waiting. No more memory rounds "
                        f"will be emitted, so decoder load and backlog past this "
                        f"point are understated. Raise max_idle_rounds for "
                        f"long-reaction studies.")
        return True

    def _relay_idle_round(self, op_id: int, patch, round_index: int) -> None:
        if self._relay_idle_round_to_live_stream(op_id, patch):
            return
        payload = SyndromePayload(op_id, patch, round_index)
        self.controller.relay_syndrome(
            payload,
            lambda p, source_op_id=op_id:
                self.cluster.on_memory_round(source_op_id))

    def _relay_idle_round_to_live_stream(self, op_id: int, patch) -> bool:
        if self.idle_policy.mode != "extend_stream":
            return False
        operation = self._ops[op_id]
        stream_id = operation.stream_id
        if stream_id is None or not self.cluster.has_dynamic_stream(stream_id):
            return False
        global_round = self.stream_next_round.get(stream_id, 0) + 1
        self.stream_next_round[stream_id] = global_round
        payloads = self.source.idle_round_payloads(
            operation, stream_id, global_round, patch)
        self.relay_syndrome_payloads(payloads)
        return True

    def relay_syndrome_payloads(self, payloads) -> None:
        """Send all fragments from one syndrome round through the controller."""
        for payload in payloads:
            payload.n_fragments = len(payloads)
            self.controller.relay_syndrome(payload,
                                           self.cluster.on_syndrome_arrival)

    def _start_released_successors_on_boundary(self, op_id: int, patch) -> None:
        if self.gates_start_on_round_boundaries:
            for successor_id in self._op_successors[op_id]:
                successor = self._ops[successor_id]
                if (successor.blocked_by is not None
                        and successor.id not in self.started
                        and successor.id in self.decode_released
                        and successor.id in self.state_ready):
                    self._patches_emitting.discard(patch)
                    self._begin(successor)

    def _submit_idle_decode_if_due(self, op_id: int, patch,
                                   round_index: int) -> None:
        if self.idle_policy.mode != "separate_decode_jobs":
            return
        patch_record = self._resolved_patches[patch]
        geometry = patch_record.code_geometry
        if round_index % geometry.commit_round_count == 0:
            self.cluster.submit_decode(
                geometry.commit_round_count + geometry.buffer_round_count,
                on_done=lambda: None,
                code=geometry.code_name,
                spatial_nodes=patch_record.spatial_node_count,
                label=f"mem({self._ops[op_id].name},r{round_index})")

    # ------------------------------------------------------------- decisions

    def on_decision(self, decision: Decision) -> None:
        """Receive feedback from the controller. Release is unconditional
        on releases_operation=True."""
        effect = decision.effect
        if effect is not None and type(effect) is not FeedbackEffect:
            raise TypeError(
                f"decision for operation {decision.target_operation_id} "
                "effect must be FeedbackEffect or None")
        if decision.releases_operation:
            self._release_blocked_operation(decision)
            return
        operation_id = decision.target_operation_id
        self.result_return_time_by_operation[operation_id] = self.engine.now
        target = self._ops[operation_id]
        detail = "timing-only" if effect is None \
            else (
                f"observable {effect.logical_observable_index}, "
                f"basis '{effect.basis}'"
            )
        self.engine.log(
            "Chip",
            f"received result return for {target.name}: {detail}",
        )

    def _release_blocked_operation(self, decision: Decision) -> None:
        operation_id = decision.target_operation_id
        target = self._ops[operation_id]
        effect = decision.effect
        if effect is not None:
            self._consume_effect(target, effect)
        self.decode_released.add(operation_id)
        self.decode_release_time[operation_id] = self.engine.now
        if effect is None:
            self.engine.log(
                "Chip",
                f"CONSUMED timing-only release for {target.name}"
                f"{' [strong-commit marker]' if decision.strong_committed else ''}; "
                "no basis or frame effect, now trying to start",
            )
        else:
            self.engine.log(
                "Chip",
                f"CONSUMED decision for {target.name}: observable "
                f"{effect.logical_observable_index}, measure basis "
                f"'{effect.basis}', frame byproduct '{effect.pauli}'"
                f"{' + S' if effect.apply_s else ''} on qubits "
                f"{target.qubits}"
                f"{' [strong-commit marker]' if decision.strong_committed else ''}; "
                "successor steered, now trying to start",
            )
        self._maybe_begin(target)

    def _consume_effect(
        self,
        target: Operation,
        effect: FeedbackEffect,
    ) -> None:
        op_id = target.id
        self.applied_basis[op_id] = effect.basis
        self.applied_pauli[op_id] = effect.pauli
        self.applied_s[op_id] = effect.apply_s
        primary = target.qubits[0] if target.qubits else None
        x_before = self.frame.x_of(primary) if primary is not None else 0
        z_before = self.frame.z_of(primary) if primary is not None else 0
        for qubit in target.qubits:
            self.frame.apply_pauli(qubit, effect.pauli)
            if effect.apply_s:
                self.frame.apply_s(qubit)
        delta = (0, 0) if primary is None else (
            self.frame.x_of(primary) ^ x_before,
            self.frame.z_of(primary) ^ z_before)
        self.applied_frame_delta[op_id] = delta
        self.frame_applied_time[op_id] = self.engine.now
