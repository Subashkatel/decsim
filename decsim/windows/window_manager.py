"""Owns the window life cycle of every operation: which rounds each window
needs, when it is ready, when its result commits and reaches the
Pauli frame and its conditional release. Boundaries between windows are the
BoundaryCourier's, ownership of committed rounds is the LogicalLedger's, the
strong tier is StrongEscalation's (NoStrongTier when the policy never
escalates), dynamic streams are DynamicWindows', replays are the
SpeculativeRecovery's; all of them are built here and work on this manager's
tables. Reading path for one round: on_syndrome_arrival, check_window,
_submit_window_decode, on_decode_done, _commit_window."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum, auto
from types import MappingProxyType
from typing import Callable, Optional

from ..message import (DecodeJob,
                      DecodeResult, DecoderRequestKey, DecoderTier, LogicalContribution,
                      Operation,
                      SeamFaultOwner, StrongDecodeCompletion, StrongRegionPlan,
                      SuccessorReadiness, SyndromeRoundPacket, Window, WindowInfo,
                      WindowPlan, WindowProtocol, WindowReadiness,
                      stable_identity_order_key)
from ..decoders.strong_escalation import NoStrongTier, StrongEscalation
from ..links.links import LinkPath, RequestTransferRelation, TrafficAttribution
from ..syndrome_buffer.syndrome_buffer import (CsdInput, DecoderInputHold, PendingStrong,
                                               PotentialStrong, RephaseGuard, SyndromeBuffer)
from .dynamic_windows import DynamicWindows
from .speculative_recovery import SpeculativeRecovery
from .committed_rounds import LogicalLedger
from .window_boundaries import BoundaryCourier, HeldBoundary



class WindowManager:

    def __init__(self, engine, *, scheme, code_geometry,
                 resolved_operations, resolved_patches,
                 links, conditional_release, boundary_policy,
                 window_interaction,
                 planning_view_by_operation_id,
                 fault_model_requirement_for,
                 feedback_boundary_mode: str = "trailing_buffer",
                 error_model_provider=None, retain_strong_context: bool,
                 double_window: bool,
                 syndrome_buffer: Optional[SyndromeBuffer] = None,
                 pauli_frame=None,
                 capture_enabled: bool = False,
                 escalation_policy, submit_fn: Callable, check_strong_route: Callable,
                 on_workload_complete: Callable[[], None]):
        self.engine = engine
        self.scheme = scheme
        self._code_geometry = code_geometry
        self._resolved_operations = MappingProxyType({
            operation.operation_id: operation
            for operation in resolved_operations
        })
        self._resolved_patches = MappingProxyType({
            patch.patch_identity: patch
            for patch in resolved_patches
        })
        self.links = links
        self.conditional_release = conditional_release
        self.pauli_frame = pauli_frame
        self.boundary_policy = boundary_policy
        self.window_interaction = window_interaction
        self._planning_view_by_operation_id = MappingProxyType(
            dict(planning_view_by_operation_id)
        )
        self._fault_model_requirement_for_code = fault_model_requirement_for
        self.feedback_boundary_mode = feedback_boundary_mode
        self.error_model_provider = error_model_provider
        self.retain_strong_context = retain_strong_context
        self.double_window = double_window
        self._next_decoder_request_sequence = 0
        self._selected_request_keys = {} if capture_enabled else None

        self.escalation_policy = escalation_policy
        self._idle_decode_demand_receiver = None
        self.submit_fn = submit_fn                   # (job, reserve_transfer) -> None
        self.escalation = (StrongEscalation(self, check_strong_route)
                           if retain_strong_context else NoStrongTier())
        self.on_workload_complete = on_workload_complete

        self.syndrome_buffer = syndrome_buffer if syndrome_buffer is not None else SyndromeBuffer()
        self.lifecycle = DynamicWindows(self)

        self._ops: dict[int, Operation] = {}
        self.rounds_arrived: dict[int, int] = {}
        self.memory_rounds: dict[int, int] = {}
        self.memory_rounds_total = 0

        self.windows: dict[tuple, Window] = {}
        self.op_windows: dict[int, list] = {}
        self.window_count: dict[int, int] = {}
        self.successors: dict[int, list] = {}
        self.committed_windows: set = set()
        self._committed_per_op: dict[int, int] = {}
        self.blocking_ops: set[int] = set()
        self.op_results: dict[int, tuple[int, ...]] = {}
        self.segment_results_sent: set = set()
        self._required_stream_end_by_operation_id: dict[int, int] = {}
        self._stream_binding_by_operation_id: dict[int, tuple[object, int]] = {}
        self.ledger = LogicalLedger()
        self.courier = BoundaryCourier(self)
        self._pending_strong_windows: set[tuple] = set()
        self._pending_strong_per_op: dict[int, int] = {}
        self.absorbed_windows: set[tuple] = set()        # skipped by the weak chain
        self.op_strong_commit_time: dict[int, int] = {}
        self._finished_ops: set[int] = set()
        self._workload_complete_sent = False
        self.speculative_recovery = SpeculativeRecovery(self, double_window)
        self.window_models: dict = {}
        self.total_windows = 0
        self._windows_built = False
        self._windowed_by_operation = {}
        self._batch_preceding_idle_rounds_by_operation = {}

    def _fault_model_requirement(self, operation: Operation):
        """Resolve the decoder views for this operation's frozen code."""
        resolved = self._resolved_operations[operation.id]
        return self._fault_model_requirement_for_code(
            resolved.code_geometry.code_name
        )

    def register_op(self, op: Operation) -> None:
        """Track an operation's rounds, payload RAM, and feedback role."""
        if op.id not in self._ops:
            self.rounds_arrived[op.id] = 0
            self.memory_rounds[op.id] = 0
            self.syndrome_buffer.open_operation(op.id)
        self._ops[op.id] = op
        if op.blocked_by is not None:
            self.blocking_ops.add(op.blocked_by)

    def _register_dynamic_stream(
        self,
        stream_op: Operation,
        resolved_operation,
    ) -> None:
        """Register a stream whose windows are created at runtime."""
        resolved_feedback_mode = (
            stream_op.feedback_boundary_mode
            if stream_op.feedback_boundary_mode is not None
            else self.feedback_boundary_mode
        )
        stream_op = replace(
            stream_op,
            feedback_boundary_mode=resolved_feedback_mode,
        )
        stream_id = stream_op.id
        self._ops[stream_id] = stream_op
        self.rounds_arrived.setdefault(stream_id, 0)
        self.memory_rounds.setdefault(stream_id, 0)
        self.syndrome_buffer.open_operation(stream_id)
        self.window_count[stream_id] = 0
        self.op_windows[stream_id] = []
        self.successors.setdefault(stream_id, [])
        self._windowed_by_operation[stream_id] = True
        self._batch_preceding_idle_rounds_by_operation[stream_id] = False
        source_round_limit = None
        if self.error_model_provider is not None:
            source_round_limit = self.error_model_provider.register_dynamic_stream(
                stream_op, self.rounds_for(stream_op),
                fault_model_requirement=self._fault_model_requirement(stream_op))
        finite_geometries = None
        if source_round_limit is not None:
            finite_geometries = self.scheme.plan_operation(
                stream_id,
                source_round_limit,
                commit_round_count=(
                    resolved_operation.code_geometry.commit_round_count
                ),
                buffer_round_count=(
                    resolved_operation.code_geometry.buffer_round_count
                ),
            ).windows
        self.lifecycle.register(
            stream_op,
            commit_round_count=(
                resolved_operation.code_geometry.commit_round_count
            ),
            buffer_round_count=(
                resolved_operation.code_geometry.buffer_round_count
            ),
            source_round_limit=source_round_limit,
            finite_geometries=finite_geometries,
        )

    def rounds_for(self, op: Operation) -> int:
        """Return the root-resolved operation duration."""
        try:
            return self._resolved_operations[op.id].round_count
        except KeyError as error:
            raise ValueError(
                f"operation {op.id} has no resolved planning record"
            ) from error

    def load_execution_plan(self, plan: WindowPlan, buffering_plan) -> None:
        """Install the pre-computed compile-time window plan."""
        if self._windows_built:
            return
        self._windows_built = True
        self.windows = plan.windows
        for window in self.windows.values():
            window.boundary_in = \
                self.window_interaction.initial_boundary_state(
                    WindowInfo.from_window(window))
        self.window_count = plan.window_count
        self.op_windows = plan.op_windows
        self.successors = plan.successors
        self._windowed_by_operation = plan.windowed_by_operation
        self._batch_preceding_idle_rounds_by_operation = (
            plan.batch_preceding_idle_rounds_by_operation
        )
        self._protocol_by_operation = {
            operation_id: plan.protocol_by_operation.get(
                operation_id, WindowProtocol.GENERIC)
            for operation_id in plan.op_windows
        }
        self.total_windows = plan.total_windows
        self._build_window_error_models()
        self.buffering_capacity_rows = {
            "upstream": (
                buffering_plan.minimum_live_rounds,
                buffering_plan.sufficient_live_rounds,
            )
        }
        capacity = self.syndrome_buffer.capacity
        minimum = buffering_plan.minimum_live_rounds
        if capacity is not None and capacity < len(minimum):
            raise ValueError(
                f"upstream syndrome buffer needs {len(minimum)} packet slots, "
                f"got {capacity}"
            )
        self._register_planned_holds(buffering_plan)

    def _register_planned_holds(self, plan) -> None:
        for owner, identities in plan.weak_holds:
            self.syndrome_buffer.register_hold(owner, identities)
        for owner, identities in plan.potential_holds:
            self.syndrome_buffer.register_hold(owner, identities)

    def _transfer_retention_hold(self, previous, replacement) -> tuple:
        keys = self.syndrome_buffer.hold_round_identities(previous)
        self.syndrome_buffer.transfer_hold(previous, replacement)
        return keys

    def _release_hold_if_live(self, owner) -> None:
        if self.syndrome_buffer.has_hold(owner):
            self.syndrome_buffer.release_hold(owner)

    def _transfer_potential_to_pending(
        self, window_key, request_key,
    ) -> tuple:
        return self._transfer_retention_hold(
            PotentialStrong(window_key), PendingStrong(request_key))

    def _build_window_error_models(self) -> None:
        """Ask the syndrome source for per-window detector error models."""
        if self.error_model_provider is None:
            return
        for op_id, op in self._ops.items():
            keys = [(op_id, k) for k in self.op_windows.get(op_id, [])]
            wins = [self.windows[key] for key in keys]
            if not wins:
                continue
            models = self.error_model_provider.window_models_for_operation(
                op, wins, self.rounds_for(op),
                fault_model_requirement=self._fault_model_requirement(op),
                fault_exclusion_ranges=(),
                window_protocol=self._protocol_by_operation[op.id],
            )
            if not models:
                continue
            for key, model in zip(keys, models):
                self.window_models[key] = model

    def _add_window_read_refs(self, key: tuple, window: Window) -> None:
        """Register typed weak and possible-strong owners for a new window."""
        weak = self._read_keys_for_bounds(
            window.op_id, window.start_round, window.buffer_hi, window)
        strong = self._strong_context_read_keys(window, weak)
        self.syndrome_buffer.register_hold(key, weak)
        if self.retain_strong_context:
            self.syndrome_buffer.register_hold(PotentialStrong(key), weak + strong)

    def _read_keys_for_bounds(self, op_id, start_round: int, buffer_hi: int,
                              window: Optional[Window] = None) -> list:
        """Retained payload round keys for a possibly cross-operation range."""
        operation_rounds = self._effective_round_count_for_window(op_id, window)
        reads = [(op_id, r)
                 for r in range(start_round, min(buffer_hi, operation_rounds) + 1)]
        overflow = buffer_hi - operation_rounds
        if overflow <= 0:
            return reads
        for successor_id in self.successors.get(op_id, []):
            reads += [(successor_id, r) for r in range(1, overflow + 1)]
        return reads

    def _strong_context_read_keys(self, window: Window, weak_reads: list) -> list:
        """Rounds retained until we know whether the strong decoder needs them."""
        if not self.retain_strong_context:
            return []
        buffer_lo, _cl, _ch, buffer_hi = self._strong_context_bounds(window)
        weak = set(weak_reads)
        strong = self._read_keys_for_bounds(
            window.op_id, buffer_lo, buffer_hi, window)
        return [rk for rk in strong if rk not in weak]

    def _replace_window_read_refs(self, key: tuple, window: Window) -> None:
        """Move shrinking weak reads into strong retention before release."""
        weak = self._read_keys_for_bounds(
            window.op_id, window.start_round, window.buffer_hi, window)
        strong = self._strong_context_read_keys(window, weak)
        potential = PotentialStrong(key)
        if self.syndrome_buffer.has_hold(potential):
            self.syndrome_buffer.replace_hold(
                potential, weak + strong)
        self.syndrome_buffer.replace_hold(key, weak)

    def _require_retained_payloads(
        self, round_keys: list, purpose: str,
    ) -> None:
        """Reject a new consumer if any already-arrived input was released."""
        missing = [
            round_key for round_key in round_keys
            if (not self.syndrome_buffer.has_operation(round_key[0])
                or (round_key[1] <= self.rounds_arrived.get(round_key[0], 0)
                    and self.syndrome_buffer.retained_fragments(round_key) is None))
        ]
        if missing:
            raise RuntimeError(
                f"{purpose} requires retained payload rounds that are no "
                f"longer available: {missing}")

    def refresh_unqueued_stream_windows(self, stream_id) -> None:
        """Refresh retained reads after a stream boundary closes future context."""
        for window_index in self.op_windows.get(stream_id, []):
            key = (stream_id, window_index)
            window = self.windows[key]
            if window.queued or window.committed:
                continue
            self._replace_window_read_refs(key, window)

    def reset_dynamic_window_reads(self, stream_id, window_index: int,
                                   window: Window) -> None:
        """After clipping a live tail, retain only the weak commit range."""
        key = (stream_id, window_index)
        new_reads = sorted(
            (stream_id, r)
            for r in range(window.start_round, window.commit_hi + 1))
        self.syndrome_buffer.replace_hold(
            key, new_reads)

    def trim_dynamic_window_tail(self, stream_id, stream_round_count: int,
                                 buffer_rounds) -> None:
        for window_index in self.op_windows.get(stream_id, []):
            window = self.windows[(stream_id, window_index)]
            if window.commit_lo <= stream_round_count <= window.commit_hi:
                window.commit_hi = stream_round_count
                window.buffer_hi = stream_round_count + buffer_rounds
                window.n_rounds = window.buffer_hi - window.start_round + 1
                self.reset_dynamic_window_reads(stream_id, window_index, window)
                return

    def create_dynamic_window(self, stream_id, window_index, commit_lo,
                              commit_hi, buffer_hi, *, is_last) -> None:
        """Create one window and connect it to the live stream plan.

        If the previous boundary already arrived, apply it immediately.
        """
        buffer_lo = commit_lo
        window = Window(op_id=stream_id, k=window_index, commit_lo=commit_lo,
                        commit_hi=commit_hi, buffer_hi=buffer_hi,
                        n_rounds=buffer_hi - buffer_lo + 1, buffer_lo=buffer_lo)
        window.boundary_in = self.window_interaction.initial_boundary_state(
            WindowInfo.from_window(window))
        if window_index > 0:
            previous_key = (stream_id, window_index - 1)
            if self.courier.has_committed(previous_key):
                # boundary already shipped (a held one is NOT available yet)
                self.courier.merge_available(
                    previous_key, window, self.courier.committed(previous_key))
            else:
                window.deps.append(previous_key)
                window.deps_remaining = 1
                self.windows[previous_key].dependents.append(
                    (stream_id, window_index))
        self.windows[(stream_id, window_index)] = window
        self.op_windows[stream_id].append(window_index)
        self.window_count[stream_id] += 1
        self.total_windows += 1
        if self.error_model_provider is not None:
            model = self.error_model_provider.window_model_for_stream(
                stream_id, window, is_last=is_last)
            if model is not None:
                self.window_models[(stream_id, window_index)] = model
        self._add_window_read_refs((stream_id, window_index), window)

    def validate_stream_length(self, stream_id, stream_round_count: int) -> None:
        if self.error_model_provider is None:
            return
        self.error_model_provider.validate_stream_length(
            self._ops[stream_id], stream_round_count)

    def on_syndrome_arrival(self, packet: SyndromeRoundPacket) -> None:
        """Retain one complete syndrome round and re-check affected windows."""
        try:
            op = self._ops[packet.operation_id]
        except KeyError as error:
            raise ValueError(
                f"unknown syndrome operation {packet.operation_id!r}"
            ) from error
        self._store_payload(packet, op)
        self.lifecycle.maybe_update(op.id)
        self.escalation.after_arrival(op.id)
        self.check_windows_for_operation(op.id)
        for predecessor_id in op.decoder_boundary_predecessors:
            self.check_windows_for_operation(predecessor_id)

    def accept_window_input(self, packet: SyndromeRoundPacket) -> bool:
        """Publish one already-retained upstream round to window readiness.

        Assembly-to-retention is a state transition on the same allocation;
        this method performs no decoder-input transfer. Window input transfer
        begins only when a decode request is admitted.

        """
        self.on_syndrome_arrival(packet)
        return True

    def accept_feedback_memory_round(self, source_operation_id) -> None:
        """Accept one standalone feedback-memory notification after CWD."""
        self.on_memory_round(source_operation_id)

    def _store_payload(
        self,
        packet: SyndromeRoundPacket,
        op: Operation,
    ) -> None:
        if not self.syndrome_buffer.has_operation(op.id):
            raise RuntimeError(
                f"round {packet.round_index} of {op.name} arrived after the op's "
                f"last window committed and its syndrome RAM was freed. The device "
                f"emitted more rounds than the execution plan expects.")
        round_limit = self.lifecycle.arrival_round_limit(
            op.id,
            fallback_rounds=self.rounds_for(op),
        )
        if round_limit is not None and packet.round_index > round_limit:
            raise ValueError(
                f"round {packet.round_index} of {op.name} exceeds the device "
                f"round limit {round_limit}"
            )
        if (
            self.syndrome_buffer.retained_fragments((op.id, packet.round_index)) != packet.fragments
            or self.syndrome_buffer.publication_tick((op.id, packet.round_index))
            != self.engine.now
        ):
            raise RuntimeError(
                "syndrome packet was not published from the retained "
                "syndrome buffer round")
        self.rounds_arrived[op.id] = max(
            self.rounds_arrived[op.id],
            packet.round_index,
        )
        self.engine.log("DecoderCluster",
                        f"round {packet.round_index} of {op.name} arrived "
                        f"(op now has rounds 1..{self.rounds_arrived[op.id]})")

    def on_memory_round(self, op_id: int) -> None:
        """Record an idle/memory round and re-check waiting windows."""
        self.memory_rounds[op_id] += 1
        self.memory_rounds_total += 1
        self.engine.log("DecoderCluster",
                        f"memory round for {self._ops[op_id].name} "
                        f"(idle buffer rounds: {self.memory_rounds[op_id]})")
        for k in range(self.window_count[op_id]):
            self.check_window((op_id, k))

    def prepend_idle_rounds(self, op_id: int, round_count: int) -> None:
        """Fold pre-gate idle rounds into a batch-style op when the scheme asks."""
        if (
            round_count <= 0
            or not self._batch_preceding_idle_rounds_by_operation.get(
                op_id,
                False,
            )
        ):
            return
        w = self.windows[(op_id, 0)]
        w.batched_preceding_idle_round_count += round_count

    def check_windows_for_operation(self, op_id: int) -> None:
        for window_index in range(self.window_count[op_id]):
            self.check_window((op_id, window_index))

    def check_window(self, key: tuple) -> None:
        """If a window has its data and dependencies, submit via the escalation policy."""
        window = self.windows[key]
        if window.queued or window.committed:
            return
        if (window.t_first_round is None
                and self.rounds_arrived[window.op_id] >= window.start_round):
            window.t_first_round = self.syndrome_buffer.publication_tick(
                (window.op_id, window.start_round)
            )
        if not self._window_data_complete(window):
            return
        if window.t_data_complete is None:
            window.t_data_complete = self.engine.now
        op = self._ops[window.op_id]
        if window.deps_remaining > 0:
            if not window.blocked_logged:
                window.blocked_logged = True
            return
        self._submit_window_decode(key, window, op)

    def _window_data_complete(self, w: Window) -> bool:
        op = self._ops[w.op_id]
        return self.scheme.data_complete(
            w, readiness=self._window_readiness(w),
            operation=self._planning_view_by_operation_id[op.id])

    def _job_desc(self, w: Window, op: Operation) -> str:
        """Human decode-job label."""
        if self._windowed_by_operation[w.op_id]:
            return f"{op.name} W{w.k} [commit {w.commit_lo}-{w.commit_hi}]"
        body_rounds = self._round_count_for_window(op.id, w)
        idle_rounds = w.batched_preceding_idle_round_count
        if idle_rounds:
            effective_rounds = w.n_rounds + idle_rounds
            return (f"{op.name} [whole op, {effective_rounds} rounds: "
                    f"{idle_rounds} idle + {body_rounds} body]")
        return f"{op.name} [whole op, {w.n_rounds} rounds]"

    def _round_count_for_window(self, op_id, window: Optional[Window] = None) -> int:
        return self.lifecycle.round_count_for_window(
            op_id, window,
            fallback_rounds=self.rounds_for(self._ops[op_id])
            if not self.lifecycle.has(op_id) else 0)

    def _effective_round_count_for_window(self, op_id,
                                          window: Optional[Window]) -> int:
        round_count = self._round_count_for_window(op_id, window)
        if window is None:
            return round_count
        closed = self._closed_boundary_round_for_window(window)
        if closed is None:
            return round_count
        return min(round_count, closed)

    def _closed_boundary_round_for_window(self, window: Window) -> Optional[int]:
        stream_boundary = self.lifecycle.closed_boundary_for_window(window)
        if stream_boundary is not None:
            return stream_boundary
        op = self._ops[window.op_id]
        if op.feedback_boundary_mode != "measurement_closed":
            return None
        if op.id not in self.blocking_ops:
            return None
        round_count = self._round_count_for_window(window.op_id, window)
        if window.commit_hi <= round_count < window.buffer_hi:
            return round_count
        return None

    def _window_readiness(self, window: Window) -> WindowReadiness:
        successor_ids = sorted(
            self.successors[window.op_id], key=stable_identity_order_key)
        successors = tuple(
            SuccessorReadiness(
                successor_id,
                self.rounds_arrived[successor_id],
                self._round_count_for_window(successor_id),
            )
            for successor_id in successor_ids
        )
        return WindowReadiness(
            local_rounds_arrived=self.rounds_arrived[window.op_id],
            local_round_count=self._effective_round_count_for_window(
                window.op_id, window),
            successors=successors,
            memory_rounds_arrived=self.memory_rounds[window.op_id],
            tail_closed=self._closed_boundary_round_for_window(window) is not None,
        )

    def _stamp_first_round_tick(self, window: Window) -> None:
        """Retain arrival provenance for latency accounting."""
        if window.t_first_round is None:
            window.t_first_round = self.syndrome_buffer.publication_tick((window.op_id, window.start_round))

    def _new_request_key(
        self, operation_id, window_id: int, tier: DecoderTier,
    ) -> DecoderRequestKey:
        request_key = DecoderRequestKey(
            operation_id, window_id, tier, self._next_decoder_request_sequence)
        self._next_decoder_request_sequence += 1
        return request_key

    def _bind_decoder_input_hold(self, job: DecodeJob, previous_owner) -> None:
        """Atomically transfer upstream retention to an admitted input request.

        The callback is invoked only after decoder memory materialization, so
        overlapping rounds remain upstream until their last consumer transfer.

        """
        owner = DecoderInputHold(job.request_key)
        if previous_owner != owner:
            if self.syndrome_buffer.has_hold(previous_owner):
                self.syndrome_buffer.transfer_hold(previous_owner, owner)
            else:
                identities = tuple(dict.fromkeys(
                    (fragment.operation_id, fragment.round_index)
                    for fragment in job.payloads
                ))
                self.syndrome_buffer.register_hold(owner, identities)
        job.input_hold = lambda token=owner: self.syndrome_buffer.release_hold(token)

    def _submit_window_decode(self, key: tuple, window: Window,
                              op: Operation) -> None:
        """Build the weak job, ask the escalation_policy, and enqueue its submissions."""
        self._stamp_first_round_tick(window)
        window.t_queued = self.engine.now
        request_key = self._new_request_key(
            window.op_id, window.k, DecoderTier.WEAK)
        job = DecodeJob(
                        op_id=window.op_id, window_id=window.k,
                        n_rounds=(
                            window.n_rounds
                            + window.batched_preceding_idle_round_count
                        ),
                        ready_time=self.engine.now,
                        spatial_nodes=self._resolved_operations[op.id].spatial_node_count,
                        payloads=self._assemble_payloads(window),
                        dem=self.window_models.get(key),
                        code=(
                            self._resolved_operations[
                                op.id
                            ].code_geometry.code_name
                        ),
                        window=window, label=self._job_desc(window, op),
                        strong_label=f"strong({op.name} W{window.k})",
                        request_key=request_key,
                        request_created_ticks=self.engine.now)
        window.queued = True
        for submission in self.escalation_policy.on_window_ready(window, job, self.escalation):
            if submission.job.strong_decode_for is None:
                if submission.job.submitted or (
                        submission.job.request_key is not None
                        and self.syndrome_buffer.has_hold(
                            DecoderInputHold(submission.job.request_key))):
                    self.submit_fn(submission.job,
                                   lambda delay=submission.delay_ticks: delay)
                    continue
                self._bind_decoder_input_hold(submission.job, key)
                weak_job = submission.job
                payload_bits = self._job_payload_bits(weak_job)
                extra_delay = submission.delay_ticks

                def reserve_transfer(job=weak_job, bits=payload_bits, extra=extra_delay) -> int:
                    # Unit assigned: move the window from Buffer 0 into its memory (CWD).
                    return (self._link_arrival(LinkPath.CWD, job, payload_bits=bits)
                            - self.engine.now + extra)

                self.submit_fn(weak_job, reserve_transfer)
            else:
                if submission.delay_ticks != 0:
                    raise ValueError(
                        "strong transport delay is owned by the link fabric"
                    )
                self.escalation.submit_strong(submission.job)

    def _assemble_payloads(self, w: Window) -> list:
        """Collect this window's payloads, including successor overflow rounds."""
        operation_rounds = self._effective_round_count_for_window(w.op_id, w)
        end_round = min(w.buffer_hi, operation_rounds)
        payloads = []
        window_info = WindowInfo.from_window(w)
        for round_index in range(w.start_round, end_round + 1):
            frags = self.syndrome_buffer.retained_fragments((w.op_id, round_index))
            if frags is not None:
                payloads += [
                    self.window_interaction.apply_boundary(
                        w.boundary_in, window_info, fragment,
                        round_index)
                    for fragment in sorted(
                        frags,
                        key=lambda item: stable_identity_order_key(
                            item.patch_id
                        ),
                    )
                ]
        overflow = w.buffer_hi - operation_rounds
        if overflow > 0:
            for successor_id in self.successors.get(w.op_id, []):
                for round_index in range(1, overflow + 1):
                    frags = self.syndrome_buffer.retained_fragments((successor_id, round_index))
                    if frags is not None:
                        payloads += [
                            self.window_interaction.apply_boundary(
                                w.boundary_in, window_info, fragment,
                                operation_rounds + round_index)
                            for fragment in sorted(
                                frags,
                                key=lambda item: stable_identity_order_key(
                                    item.patch_id
                                ),
                            )]
        return payloads

    @staticmethod
    def _job_payload_bits(job: DecodeJob) -> Optional[int]:
        sizes = [payload.size_bits for payload in (job.payloads or ())]
        if not all(size is not None for size in sizes):
            return None
        return sum(sizes)

    @staticmethod
    def _job_attribution(job: DecodeJob,
                         request_key: DecoderRequestKey) -> TrafficAttribution:
        patches = {}
        for payload in job.payloads or ():
            patch_id = payload.patch_id
            patches[stable_identity_order_key(patch_id)] = patch_id
        patch_ids = tuple(patches[key] for key in sorted(patches))
        window = job.window
        if window is None:
            raise RuntimeError("window-scoped transport requires a DecodeJob window")
        return TrafficAttribution(
            operation_id=job.op_id,
            patch_ids=patch_ids,
            window_id=job.window_id,
            round_lo=(
                window.commit_lo
                if window.buffer_lo is None
                else window.buffer_lo
            ),
            round_hi=(
                window.commit_hi
                if window.buffer_hi is None
                else window.buffer_hi
            ),
            relation=RequestTransferRelation(request_key),
        )

    @staticmethod
    def _window_attribution(
        window: Window,
        op: Operation,
        request_key: DecoderRequestKey,
    ) -> TrafficAttribution:
        return TrafficAttribution(
            operation_id=op.id,
            patch_ids=tuple(sorted(
                op.patches,
                key=stable_identity_order_key,
            )),
            window_id=window.k,
            round_lo=(
                window.commit_lo
                if window.buffer_lo is None
                else window.buffer_lo
            ),
            round_hi=(
                window.commit_hi
                if window.buffer_hi is None
                else window.buffer_hi
            ),
            relation=RequestTransferRelation(request_key),
        )

    def _window_link_arrival(
        self,
        path: LinkPath,
        window: Window,
        op: Operation,
        request_key: DecoderRequestKey,
    ) -> int:
        reservation = self.links.reserve(
            path,
            payload_bits=None,
            now_ticks=self.engine.now,
            attribution=self._window_attribution(window, op, request_key),
        )
        return self.engine.now + reservation.total_delay_ticks

    def _link_arrival(
        self,
        path: LinkPath,
        job: DecodeJob,
        *,
        payload_bits: Optional[int],
        request_key: Optional[DecoderRequestKey] = None,
    ) -> int:
        relation_key = job.request_key if request_key is None else request_key
        reservation = self.links.reserve(
            path,
            payload_bits=payload_bits,
            now_ticks=self.engine.now,
            attribution=self._job_attribution(job, relation_key),
        )
        return self.engine.now + reservation.total_delay_ticks


    # ---- the EscalationServices seam: what an escalation policy may ask of the run


    @staticmethod
    def _strong_context_bounds(window: Window) -> tuple:
        buffer_rounds = max(0, window.buffer_hi - window.commit_hi)
        buffer_lo = max(1, window.commit_lo - buffer_rounds)
        buffer_hi = window.commit_hi + buffer_rounds
        return buffer_lo, window.commit_lo, window.commit_hi, buffer_hi


    def on_decode_done(self, job: DecodeJob, res: DecodeResult) -> None:
        """Hand the boundary on as the decoder finishes; publish the accepted
        weak result only after its WDO transfer.

        The boundary is decoder state (residual defects at the commit edge), so
        it leaves for the dependent windows at decode done, the way Skoric's
        blocks, LILLIPUT's state register and qLDPC's net_error do; the frame
        commit downstream never gates the next window.
        """
        window = self.windows[(job.op_id, job.window_id)]
        window.t_done = self.engine.now
        if job.awaiting_strong_result:
            self._commit_decode_done(job, res)
            return
        op = self._ops[job.op_id]
        self._hand_on_boundary(job, res, window, op)
        delivery_ticks = self._window_link_arrival(
            LinkPath.WDO,
            window,
            op,
            job.request_key,
        )
        self.engine.schedule(
            delivery_ticks - self.engine.now,
            lambda: self._sink_weak_correction(job, res),
            label=f"weak result {op.name}W{window.k}->pauli frame",
        )

    def _sink_weak_correction(self, job: DecodeJob, res: DecodeResult) -> None:
        """Charge and install one final weak correction before committing it."""
        if self.pauli_frame is None:
            self._commit_decode_done(job, res)
            return
        self.pauli_frame.commit_weak_correction(
            window_key=(job.op_id, job.window_id),
            logical_observables=res.logical_observables,
            request_key=job.request_key,
            on_committed=lambda: self._commit_decode_done(job, res),
        )

    def _commit_decode_done(self, job: DecodeJob, res: DecodeResult) -> None:
        """Commit after weak result transport, or provisionally on escalation."""
        key = (job.op_id, job.window_id)
        window = self.windows[key]
        op = self._ops[job.op_id]
        self._commit_window(job, res, key, window, op)
        if not job.awaiting_strong_result and self._selected_request_keys is not None:
            self._selected_request_keys[key] = job.request_key
        self.lifecycle.update_committed_round_count(op.id)
        if job.awaiting_strong_result:       # provisional: boundary leaves with the commit
            self._hand_on_boundary(job, res, window, op)
        if not job.awaiting_strong_result:
            self._release_hold_if_live(
                PotentialStrong(key))
        self.escalation.after_weak_commit(key)
        self.speculative_recovery.after_commit()
        self._finish_operation_if_ready(op)
        self.finish_workload_if_ready()

    def _hand_on_boundary(self, job: DecodeJob, res: DecodeResult,
                          window: Window, op: Operation) -> None:
        """Ship the decoder's boundary to dependent windows, or hold it when the
        policy waits for a final result."""
        boundary = self.window_interaction.boundary_from_result(res, None)
        if job.awaiting_strong_result:
            self.speculative_recovery.begin(job, boundary)
        final = not job.awaiting_strong_result
        if self.boundary_policy.on_commit(window, final=final):
            self.courier.send(
                window, op, boundary, source_request_key=job.request_key)
        else:
            self.courier.hold((job.op_id, job.window_id),
                              HeldBoundary(job.request_key, op.id, boundary))

    def rounds_backlog(self) -> tuple:
        """Per operation, in stable order: (op_id, patch, rounds arrived but
        not yet decoded in an unbroken prefix from round 1)."""
        rows = []
        for op_id in sorted(self._ops, key=stable_identity_order_key):
            op = self._ops[op_id]
            committed_ranges = sorted(
                (self.windows[key].commit_lo, self.windows[key].commit_hi)
                for key in self.committed_windows if key[0] == op_id)
            decoded = 0
            for start_round, end_round in committed_ranges:
                if start_round <= decoded + 1:
                    decoded = max(decoded, end_round)
                else:
                    break
            waiting = max(0, self.rounds_arrived.get(op_id, 0) - decoded)
            patch = (op.patches[0] if op.patches else op.qubits[0] if op.qubits else op_id)
            rows.append((op_id, patch, waiting))
        return tuple(rows)

    def selected_request_key(self, key: tuple):
        """The request whose result a window finally published, when the run
        captures switching records; None otherwise."""
        if self._selected_request_keys is None:
            return None
        return self._selected_request_keys.get(key)

    def uncommit_window(self, window: Window) -> None:
        """A committed window is about to be replayed: it leaves the committed set."""
        self.committed_windows.discard(window.key)
        remaining = self._committed_per_op.get(window.op_id, 0) - 1
        if remaining > 0:
            self._committed_per_op[window.op_id] = remaining
        else:
            self._committed_per_op.pop(window.op_id, None)
        window.committed = False

    def _commit_window(self, job: DecodeJob, res: DecodeResult, key: tuple,
                       window: Window, op: Operation) -> None:
        window.committed = True
        if window.t_done is None:
            window.t_done = self.engine.now
        self.committed_windows.add(key)
        self._committed_per_op[op.id] = self._committed_per_op.get(op.id, 0) + 1
        self.engine.log("DecoderCluster",
                        f"DECODE DONE {op.name} W{window.k} "
                        f"[commit {window.commit_lo}-{window.commit_hi}]")
        existing_contribution = self.ledger.get(key)
        if (
            existing_contribution is None
            or existing_contribution.ownership_kind != "strong_slab"
        ):
            self.ledger.install(
                LogicalContribution(
                    owner_key=key,
                    commit_lo=window.commit_lo,
                    commit_hi=window.commit_hi,
                    ownership_kind="ordinary_window",
                    logical_observables=res.logical_observables,
                )
            )
        if job.awaiting_strong_result:
            self._mark_window_waiting_for_strong(key, op.id)


    def _mark_window_waiting_for_strong(self, key: tuple, op_id: int) -> None:
        if key in self._pending_strong_windows:
            return
        self._pending_strong_windows.add(key)
        self._pending_strong_per_op[op_id] = \
            self._pending_strong_per_op.get(op_id, 0) + 1

    def on_strong_decode_done(self, completion: StrongDecodeCompletion) -> None:
        """Publish an accepted strong result only after its DO transfer."""
        key = (completion.request_key.operation_id,
               completion.request_key.window_id)
        window = self.windows[key]
        op = self._ops[window.op_id]
        delivery_ticks = self._window_link_arrival(
            LinkPath.DO,
            window,
            op,
            completion.request_key,
        )
        self.engine.schedule(
            delivery_ticks - self.engine.now,
            lambda: self._commit_strong_decode_done(completion),
            label=f"strong result {op.name}W{window.k}->pauli frame",
        )

    def _commit_strong_decode_done(
        self,
        completion: StrongDecodeCompletion,
    ) -> None:
        """Finalize a weak-committed window with the delivered strong result.

        Held ships the strong boundary now. Eager delegates a boundary change
        to SpeculativeRecovery, which replays the affected static descendants.
        """
        key = (completion.request_key.operation_id,
               completion.request_key.window_id)
        result = completion.result
        window = self.windows[key]
        op = self._ops[window.op_id]
        self.op_strong_commit_time[op.id] = max(
            self.op_strong_commit_time.get(op.id, 0), self.engine.now)
        if self._selected_request_keys is not None:
            self._selected_request_keys[key] = completion.request_key
        if self.speculative_recovery.complete(completion):
            return
        if result.logical_observables is not None:
            self.ledger.replace_prediction(
                key,
                result.logical_observables,
            )
        self._resolve_strong_wait(key, op.id)
        held = self.courier.take_held(key)                    # Held: ship now
        if held is not None:
            self.courier.send(
                window, self._ops[held.operation_id],
                self.window_interaction.boundary_from_result(
                    result, held.boundary),
                source_request_key=completion.request_key)
        self.speculative_recovery.after_commit()
        self.release_stream_segments_at_commit(
            op.id, self.lifecycle.committed_round_count(op.id))
        self._finish_operation_if_ready(op)
        self.finish_workload_if_ready()

    def _resolve_strong_wait(self, key: tuple, op_id: int) -> None:
        if key not in self._pending_strong_windows:
            return
        self._pending_strong_windows.remove(key)
        remaining = self._pending_strong_per_op.get(op_id, 0) - 1
        if remaining > 0:
            self._pending_strong_per_op[op_id] = remaining
        else:
            self._pending_strong_per_op.pop(op_id, None)


    def _window_infos(self):
        return MappingProxyType({
            key: WindowInfo.from_window(window)
            for key, window in self.windows.items()
        })

    def _finish_operation_if_ready(self, op: Operation) -> None:
        """Deliver an op result once every window is committed, no strong
        redo or speculative ancestor is pending, and the stream is sealed."""
        if op.id in self._finished_ops:
            return
        if self._pending_strong_per_op.get(op.id, 0) > 0:
            return
        if self.speculative_recovery.blocks_finality(op.id):
            return
        if self.syndrome_buffer.has_live_operation_reference(op.id):
            return
        if (self._committed_per_op.get(op.id, 0) == self.window_count[op.id]
                and self.lifecycle.sealed(op.id)):
            self._finished_ops.add(op.id)
            self._deliver_result(op)
            self.syndrome_buffer.close_operation(op.id)

    def finish_workload_if_ready(self) -> None:
        if self._workload_complete_sent:
            return
        if (len(self.committed_windows) == self.total_windows
                and not self._pending_strong_windows
                and not self.speculative_recovery.has_finality_blockers
                and not self.lifecycle.has_unsealed_streams()
                and self.on_workload_complete is not None):
            self._workload_complete_sent = True
            self.on_workload_complete()

    def _deliver_result(self, op: Operation) -> None:
        window_keys = [
            (op.id, window_index)
            for window_index in self.op_windows[op.id]
        ]
        commit_lo = min(
            self.windows[key].commit_lo
            for key in window_keys
        )
        commit_hi = max(
            self.windows[key].commit_hi
            for key in window_keys
        )
        logical_observables = self.ledger.observables_for_interval(
            op.id,
            commit_lo,
            commit_hi,
            boundary_policy="strict",
        )
        if logical_observables is None:
            self.op_results.pop(op.id, None)
        else:
            self.op_results[op.id] = logical_observables
        result = DecodeResult(
            op.id,
            self.window_count[op.id] - 1,
            logical_observables=logical_observables,
        )
        self.conditional_release.integrate(op, result)

    def release_stream_segments_at_commit(self, stream_id,
                                          committed_round_count: int) -> None:
        """Deliver segment results whose full round range has committed
        (gated the same way as ops: no pending strong may still change it)."""
        for operation in list(self._ops.values()):
            binding = self._stream_binding_by_operation_id.get(operation.id)
            operation_stream_id = operation.stream_id if binding is None else binding[0]
            if operation_stream_id != stream_id:
                continue
            if operation.id not in self.blocking_ops:
                continue
            if operation.id in self.segment_results_sent:
                continue
            segment_end = self._stream_segment_end(operation)
            if segment_end is None or segment_end > committed_round_count:
                continue
            if (self.speculative_recovery.blocks_finality(operation.id)
                    or self.speculative_recovery.blocks_stream_segment(
                        stream_id, segment_end)):
                continue
            if self._segment_waits_for_strong(stream_id, segment_end):
                continue
            operation_stream_offset = operation.stream_offset if binding is None else binding[1]
            segment_start = operation_stream_offset + 1
            logical_observables = self.ledger.observables_for_interval(
                stream_id,
                segment_start,
                segment_end,
                boundary_policy="stream_segment",
            )
            if logical_observables is None:
                self.op_results.pop(operation.id, None)
            else:
                self.op_results[operation.id] = logical_observables
            self.segment_results_sent.add(operation.id)
            self.conditional_release.integrate(
                operation,
                DecodeResult(
                    operation.id,
                    -1,
                    logical_observables=logical_observables,
                ),
            )

    def _segment_waits_for_strong(self, stream_id, segment_end: int) -> bool:
        for key in self._pending_strong_windows:
            if key[0] != stream_id:
                continue
            if self.windows[key].commit_lo <= segment_end:
                return True
        return False

    def connect_idle_decode_demand_receiver(self, receiver) -> None:
        """Connect the optional synthetic idle-load model to decode service."""
        self._idle_decode_demand_receiver = receiver

    def accept_idle_decode_demand(self, *, rounds, code, spatial_nodes,
                                  label) -> None:
        """Submit modeled idle-memory work without exposing a decoder to control."""
        self._idle_decode_demand_receiver(
            rounds, on_done=lambda: None, code=code,
            spatial_nodes=spatial_nodes, label=label)

    def bind_stream_operation(self, operation_id: int, stream_id,
                              stream_offset: int) -> None:
        self._stream_binding_by_operation_id[operation_id] = (stream_id, stream_offset)

    def bind_required_stream_end(self, operation_id: int,
                                 required_stream_end: int) -> None:
        self._required_stream_end_by_operation_id[operation_id] = required_stream_end

    def _stream_segment_end(self, operation: Operation) -> Optional[int]:
        required_end = self._required_stream_end_by_operation_id.get(operation.id)
        if required_end is not None:
            return required_end
        binding = self._stream_binding_by_operation_id.get(operation.id)
        stream_offset = operation.stream_offset if binding is None else binding[1]
        if stream_offset is None:
            return None
        return stream_offset + self.rounds_for(operation)

    def close_stream_boundary(self, stream_id, stream_round_count: int) -> None:
        self.lifecycle.close_boundary(stream_id, stream_round_count)

    def seal_stream(self, stream_id, stream_round_count: int) -> None:
        self.lifecycle.seal(stream_id, stream_round_count)

    def has_dynamic_stream(self, stream_id) -> bool:
        return self.lifecycle.has(stream_id)

    def committed_stream_round_count(self, stream_id) -> int:
        return self.lifecycle.committed_round_count(stream_id)

