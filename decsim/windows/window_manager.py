"""The windows facade: the window life cycle of every operation.

Which windows exist is the WindowPlanner's, which rounds arrived and
whether a window has its data is the RoundTracker's, which rounds each
window holds is the RoundRetention's, a decode is asked for by the
DecodeRequester from a job the DecodeRequestBuilder builds, and every
send rides the WindowTransfers; the facade receives rounds, commits
results and delivers each operation's result, the shape of gem5's cache
(BaseCache owns its MSHR queue, write buffer and tags, each one job, and
implements the ports: src/mem/cache/base.hh). Boundaries between windows are the
BoundaryCourier's, ownership of committed rounds is the LogicalLedger's,
the strong tier is StrongEscalation's (NoStrongTier when the policy
never escalates). One round reads as accept_window_input, check_window,
_submit_window_decode, on_decode_done, _commit_window.

Wide state recorded for the structural commits that split the rest into
committer and results (slice 5 design note, section 3).
"""

import dataclasses
import functools
import types
from typing import Callable, Optional, Protocol, runtime_checkable

import decsim.decoders.strong_escalation as strong_escalation
import decsim.message as message
import decsim.windows.committed_rounds as committed_rounds
import decsim.windows.window_boundaries as window_boundaries
import decsim.windows.window_transfers as window_transfers


@runtime_checkable
class BoundaryPolicy(Protocol):
    """When a committed window may ship its boundary defects to dependents.

    Eager ships at every weak commit (the default); Held ships only once
    the result is final. A provisional boundary that ships is never
    revised, so serial switching (which can revise a result) needs Held.
    """

    def on_commit(self, window: message.Window, *, final: bool) -> bool:
        """Whether to ship the boundary now."""


@runtime_checkable
class ErrorModelProvider(Protocol):
    """Builds the decoder-facing models of windows and streams."""

    def register_dynamic_stream(
        self, stream_operation, round_count: int, *, fault_model_requirement
    ):
        """Note a dynamic stream; its window models come per window."""

    def validate_stream_length(
        self, stream_operation, stream_round_count: int
    ) -> None:
        """Refuse a stream longer than the source can supply."""

    def window_models_for_operation(
        self,
        operation,
        windows: list,
        round_count: int,
        *,
        fault_model_requirement,
        fault_exclusion_ranges: tuple,
        window_protocol,
    ) -> list:
        """One model per window of the operation."""

    def window_model_for_stream(self, stream_id, window):
        """The model of one window of a dynamic stream."""

    def strong_window_model_for_operation(
        self,
        operation,
        window,
        round_count: int,
        *,
        fault_model_requirement,
        exclude_faults_touching=None,
    ):
        """The model of one strong window of the operation."""


class WindowManager:
    """The windows facade: receives rounds, closes windows, commits results."""

    def __init__(
        self,
        engine,
        *,
        planner,
        tracker,
        retention,
        transfers,
        builder,
        requester,
        conditional_release,
        boundary_policy,
        window_interaction,
        feedback_boundary_mode: str = "trailing_buffer",
        pauli_frame=None,
        on_workload_complete: Callable[[], None],
    ):
        self.engine = engine
        self.planner = planner
        self.tracker = tracker
        self.retention = retention
        self.transfers = transfers
        self.builder = builder
        self.requester = requester
        self.conditional_release = conditional_release
        self.pauli_frame = pauli_frame
        self.boundary_policy = boundary_policy
        self.window_interaction = window_interaction
        self.feedback_boundary_mode = feedback_boundary_mode
        self.escalation = strong_escalation.NoStrongTier()
        if retention.is_strong_context_retained:
            self.escalation = strong_escalation.StrongEscalation(self)
        self.on_workload_complete = on_workload_complete
        self.result_by_operation: dict[int, tuple[int, ...]] = {}
        self.segment_results_sent: set = set()
        self._required_stream_end_by_operation_id: dict[int, int] = {}
        self._stream_binding_by_operation_id: dict[int, tuple] = {}
        self.ledger = committed_rounds.LogicalLedger()
        self.courier = window_boundaries.BoundaryCourier(self)
        self._finished_operation_ids: set[int] = set()
        self._workload_complete_sent = False
        # the committed prefix of each stream, so segments release once
        self._committed_round_count_by_stream: dict = {}
        for window in self.planner.windows_by_key.values():
            window_info = message.WindowInfo.from_window(window)
            window.boundary_in = self.window_interaction.initial_boundary_state(
                window_info
            )

    @property
    def windows(self) -> dict:
        """Every window by key; the planner's table."""
        return self.planner.windows_by_key

    # ---- the plan: operations, streams, windows and their holds

    def register_operation(self, operation: message.Operation) -> None:
        """Track an operation's rounds, payload RAM, and feedback role."""
        is_new = self.tracker.register_operation(operation)
        if is_new:
            self.retention.weak_store.open_operation(operation.id)

    def register_stream(self, stream_operation: message.Operation) -> None:
        """Register a stream whose windows are created at runtime."""
        resolved_feedback_mode = stream_operation.feedback_boundary_mode
        if resolved_feedback_mode is None:
            resolved_feedback_mode = self.feedback_boundary_mode
        stream_operation = dataclasses.replace(
            stream_operation, feedback_boundary_mode=resolved_feedback_mode
        )
        self.retention.weak_store.open_operation(stream_operation.id)
        source_round_limit = self.planner.register_stream(stream_operation)
        self.tracker.register_stream(stream_operation, source_round_limit)

    def install_planned_holds(self, buffering_plan) -> None:
        """Check the stores against the plan and place its holds."""
        self.retention.install_planned_holds(buffering_plan)

    # ---- dynamic streams: their windows are planned as rounds arrive

    def _update_stream(self, operation_id) -> None:
        """Grow (and maybe seal) the stream as one more round arrives."""
        if not self.planner.has_stream(operation_id):
            return
        highest_known_round = self.tracker.rounds_arrived(operation_id)
        self._grow_stream(operation_id, highest_known_round, None)
        if self.tracker.is_sealed(operation_id):
            return
        if self.tracker.reaches_source_limit(operation_id):
            limit = self.tracker.source_round_limit(operation_id)
            self.seal_stream(operation_id, limit)

    def _grow_stream(
        self, stream_id, highest_known_round: int, round_cap: Optional[int]
    ) -> None:
        """Plan every window whose commit region has begun, and admit it."""
        if self.tracker.is_sealed(stream_id):
            return
        created = self.planner.grow_stream(
            stream_id, highest_known_round, round_cap
        )
        for window in created:
            self._admit_stream_window(window)

    def _admit_stream_window(self, window: message.Window) -> None:
        """Connect one new stream window: its boundary, its model, its holds.

        If the previous boundary already arrived, apply it immediately.
        """
        window_info = message.WindowInfo.from_window(window)
        window.boundary_in = self.window_interaction.initial_boundary_state(
            window_info
        )
        if window.k > 0:
            previous_key = (window.op_id, window.k - 1)
            self._link_to_previous_window(previous_key, window.key, window)
        self.planner.attach_stream_model(window)
        self.retention.register_window(window.key, window)

    def _link_to_previous_window(
        self, previous_key: tuple, key: tuple, window: message.Window
    ) -> None:
        """A shipped boundary merges now; a held one is not available yet."""
        if self.courier.has_committed(previous_key):
            boundary = self.courier.committed(previous_key)
            self.courier.merge_available(previous_key, window, boundary)
            return
        window.deps.append(previous_key)
        window.deps_remaining = 1
        previous = self.planner.windows_by_key[previous_key]
        previous.dependents.append(key)

    def seal_stream(self, stream_id, stream_round_count: int) -> None:
        """Close a dynamic stream once its full length has arrived."""
        if self.tracker.is_sealed(stream_id):
            return
        self.tracker.check_stream_length(stream_id, stream_round_count)
        self._grow_stream(stream_id, stream_round_count, stream_round_count)
        if not self.planner.is_finite_stream(stream_id):
            clipped = self.planner.trim_stream_tail(
                stream_id, stream_round_count
            )
            if clipped is not None:
                self.retention.reset_clipped_window_reads(clipped)
        self.tracker.seal(stream_id, stream_round_count)
        self.check_windows_for_operation(stream_id)
        self.finish_workload_if_ready()

    def close_stream_boundary(self, stream_id, stream_round_count: int) -> None:
        """Mark a live stream round as a measurement-closed boundary."""
        if not self.planner.has_stream(stream_id):
            return
        self.tracker.close_boundary(stream_id, stream_round_count)
        self._grow_stream(stream_id, stream_round_count, None)
        self._refresh_unqueued_stream_windows(stream_id)
        self.check_windows_for_operation(stream_id)

    def _refresh_unqueued_stream_windows(self, stream_id) -> None:
        """Refresh retained reads once a stream boundary closes the tail."""
        for window in self.planner.windows_of(stream_id):
            if window.queued or window.committed:
                continue
            self.retention.replace_window_reads(window.key, window)

    def has_dynamic_stream(self, stream_id) -> bool:
        """True for a stream whose windows are planned at runtime."""
        return self.planner.has_stream(stream_id)

    # ---- arrivals: the WindowInput port and the room-side store's signal

    def accept_window_input(self, packet: message.SyndromeRoundPacket) -> None:
        """Publish one stored upstream round to window readiness.

        Assembly-to-retention is a state transition on the same
        allocation; no decoder input moves here. Window input transfer
        begins only when a decode request is admitted.
        """
        operation = self.tracker.operation_by_id[packet.operation_id]
        self._refuse_unplanned_round(packet, operation)
        if self.retention.primary_tier is not message.DecoderTier.STRONG:
            # Buffer 0 publication is the readiness authority for the weak lane
            self._count_arrival(operation, packet.round_index)
            self._update_stream(operation.id)
            self.escalation.after_arrival(operation.id)
            self._wake_windows(operation)
        # a round whose every consumer already resolved (an absorbed window's
        # tail, or every round of a strong-primary plan) frees its Buffer 0
        # slot on arrival, the same drop-on-arrival rule syndrome buffer 1
        # applies
        round_key = (packet.operation_id, packet.round_index)
        self.retention.release_round_if_unheld(round_key)

    def _refuse_unplanned_round(
        self, packet: message.SyndromeRoundPacket, operation: message.Operation
    ) -> None:
        """A round past the plan is the device's mistake (it is a plug-in)."""
        if not self.retention.weak_store.has_operation(operation.id):
            raise RuntimeError(
                f"round {packet.round_index} of {operation.name} arrived "
                f"after the op's last window committed and its syndrome RAM "
                f"was freed. The device emitted more rounds than the "
                f"execution plan expects."
            )
        round_limit = self.tracker.arrival_round_limit(operation.id)
        if round_limit is not None and packet.round_index > round_limit:
            raise ValueError(
                f"round {packet.round_index} of {operation.name} exceeds the "
                f"device round limit {round_limit}"
            )

    def accept_feedback_memory_round(self, source_operation_id) -> None:
        """Record one idle or memory round and re-check waiting windows."""
        memory_rounds = self.tracker.note_memory_round(source_operation_id)
        operation = self.tracker.operation_by_id[source_operation_id]
        self.engine.log(
            "DecoderCluster",
            f"memory round for {operation.name} "
            f"(idle buffer rounds: {memory_rounds})",
        )
        self.check_windows_for_operation(source_operation_id)

    def accept_room_round(self, operation_id, round_index: int) -> None:
        """Syndrome buffer 1 stored a round.

        Wake the strong tier's listeners, and drive readiness when the
        strong tier is primary.
        """
        self.tracker.note_room_round(operation_id, round_index)
        self.escalation.after_arrival(operation_id)
        if self.retention.primary_tier is not message.DecoderTier.STRONG:
            return
        operation = self.tracker.operation_by_id[operation_id]
        stored_through = self.tracker.strong_rounds_arrived(operation_id)
        self._count_arrival(operation, stored_through)
        self._update_stream(operation_id)
        self._wake_windows(operation)

    def _count_arrival(
        self, operation: message.Operation, round_index: int
    ) -> None:
        """Advance the readiness arrival counter; the authority calls this."""
        arrived_now = self.tracker.note_arrival(operation.id, round_index)
        self.engine.log(
            "DecoderCluster",
            f"round {round_index} of {operation.name} arrived "
            f"(op now has rounds 1..{arrived_now})",
        )

    def _wake_windows(self, operation: message.Operation) -> None:
        self.check_windows_for_operation(operation.id)
        for predecessor_id in operation.decoder_boundary_predecessors:
            self.check_windows_for_operation(predecessor_id)

    def prepend_idle_rounds(self, operation_id: int, round_count: int) -> None:
        """Fold pre-gate idle rounds into a batch-style operation."""
        self.planner.prepend_idle_rounds(operation_id, round_count)

    # ---- readiness: a window with its data and no owed boundary is requested

    def check_windows_for_operation(self, operation_id: int) -> None:
        """Request every complete window of the operation, in index order."""
        windows = self.planner.windows_of(operation_id)
        self.requester.request_ready_windows(windows, self.escalation)

    def check_window(self, key: tuple) -> None:
        """Request the window if it has its data."""
        window = self.planner.windows_by_key[key]
        self.requester.request_if_ready(window, self.escalation)

    # ---- the result: the boundary leaves at decode done, the frame commits

    def on_decode_done(
        self, job: message.DecodeJob, result: message.DecodeResult
    ) -> None:
        """Hand the boundary on at decode done; publish after the output link.

        The boundary is decoder state (residual defects at the commit
        edge), so it leaves for the dependent windows at decode done, the
        way Skoric's blocks, LILLIPUT's state register and qLDPC's
        net_error do; the frame commit downstream never gates the next
        window.
        """
        window = self.planner.windows_by_key[(job.op_id, job.window_id)]
        window.t_done = self.engine.now
        if job.awaiting_strong_result:
            self._commit_decode_done(job, result)
            return
        operation = self.tracker.operation_by_id[job.op_id]
        self._hand_on_boundary(job, result, window, operation)
        # the result rides its tier's output link home: WDO for the weak
        # tier, DO for a strong-primary decode
        output_path = message.LinkPath.WEAK_DECODER_TO_FRAME
        if job.request_key.tier is not message.DecoderTier.WEAK:
            output_path = message.LinkPath.STRONG_DECODER_TO_FRAME
        payload_bits = window_transfers.result_payload_bits(result, operation)
        sink = functools.partial(self._sink_weak_correction, job, result)
        self.transfers.send_for_window(
            output_path, window, operation, job.request_key, payload_bits, sink
        )

    def _sink_weak_correction(
        self, job: message.DecodeJob, result: message.DecodeResult
    ) -> None:
        """Charge and install one final correction before committing it."""
        if self.pauli_frame is None:
            self._commit_decode_done(job, result)
            return
        commit = functools.partial(self._commit_decode_done, job, result)
        self.pauli_frame.commit_correction(
            window_key=(job.op_id, job.window_id),
            logical_observables=result.logical_observables,
            request_key=job.request_key,
            on_committed=commit,
        )

    def _commit_decode_done(
        self, job: message.DecodeJob, result: message.DecodeResult
    ) -> None:
        """Commit after the weak result's transport, or provisionally."""
        key = (job.op_id, job.window_id)
        window = self.planner.windows_by_key[key]
        operation = self.tracker.operation_by_id[job.op_id]
        self._commit_window(result, key, window, operation)
        is_final = not job.awaiting_strong_result
        if is_final:
            window.published_request_key = job.request_key
        self._update_committed_round_count(operation.id)
        if job.awaiting_strong_result:
            # provisional: the boundary leaves with the commit
            self._hand_on_boundary(job, result, window, operation)
        if is_final and self.retention.strong_store is not None:
            potential = message.PotentialStrong(key)
            self.retention.release_hold_if_live(
                potential, self.retention.strong_store
            )
        self.escalation.after_weak_commit(key)
        self._finish_operation_if_ready(operation)
        self.finish_workload_if_ready()

    def _hand_on_boundary(
        self,
        job: message.DecodeJob,
        result: message.DecodeResult,
        window: message.Window,
        operation: message.Operation,
    ) -> None:
        """Ship the boundary to dependent windows, or hold it until final."""
        boundary = self.window_interaction.boundary_from_result(result, None)
        is_final = not job.awaiting_strong_result
        if self.boundary_policy.on_commit(window, final=is_final):
            self.courier.send(
                window, operation, boundary, source_request_key=job.request_key
            )
            return
        held = window_boundaries.HeldBoundary(
            job.request_key, operation.id, boundary
        )
        self.courier.hold((job.op_id, job.window_id), held)

    def rounds_backlog(self) -> tuple:
        """Rounds arrived but not decoded, per operation in stable order.

        A row is (operation id, patch, rounds arrived past the unbroken
        decoded prefix from round 1).
        """
        rows = []
        ordered_ids = sorted(
            self.tracker.operation_by_id, key=message.stable_identity_order_key
        )
        for operation_id in ordered_ids:
            operation = self.tracker.operation_by_id[operation_id]
            decoded = self._committed_prefix_round_count(operation_id)
            arrived = self.tracker.rounds_arrived(operation_id)
            undecoded = arrived - decoded
            waiting = max(0, undecoded)
            patch = _representative_patch(operation)
            rows.append((operation_id, patch, waiting))
        return tuple(rows)

    def _committed_prefix_round_count(self, operation_id) -> int:
        """Rounds decoded in an unbroken prefix from round 1."""
        committed_ranges = []
        for window in self._committed_windows_of(operation_id):
            committed_ranges.append((window.commit_lo, window.commit_hi))
        committed_ranges.sort()
        decoded = 0
        for start_round, end_round in committed_ranges:
            if start_round > decoded + 1:
                break
            decoded = max(decoded, end_round)
        return decoded

    def _commit_window(
        self,
        result: message.DecodeResult,
        key: tuple,
        window: message.Window,
        operation: message.Operation,
    ) -> None:
        window.committed = True
        if window.t_done is None:
            window.t_done = self.engine.now
        status = result.decode_status
        window.decode_status = None
        status_note = ""
        if status is not None:
            window.decode_status = status.value
            status_note = f" best effort: {status.value}"
        self.engine.log(
            "DecoderCluster",
            f"DECODE DONE {operation.name} W{window.k} "
            f"[commit {window.commit_lo}-{window.commit_hi}]{status_note}",
        )
        existing = self.ledger.get(key)
        if existing is None or existing.ownership_kind != "strong_window":
            contribution = message.LogicalContribution(
                owner_key=key,
                commit_lo=window.commit_lo,
                commit_hi=window.commit_hi,
                ownership_kind="ordinary_window",
                logical_observables=result.logical_observables,
            )
            self.ledger.install(contribution)

    def on_strong_decode_done(
        self, completion: message.StrongDecodeCompletion
    ) -> None:
        """Publish an accepted strong result only after its DO transfer."""
        key = (
            completion.request_key.operation_id,
            completion.request_key.window_id,
        )
        window = self.planner.windows_by_key[key]
        operation = self.tracker.operation_by_id[window.op_id]
        payload_bits = window_transfers.result_payload_bits(
            completion.result, operation
        )
        commit = functools.partial(self._commit_strong_decode_done, completion)
        self.transfers.send_for_window(
            message.LinkPath.STRONG_DECODER_TO_FRAME,
            window,
            operation,
            completion.request_key,
            payload_bits,
            commit,
        )

    def _commit_strong_decode_done(
        self, completion: message.StrongDecodeCompletion
    ) -> None:
        """Finalize a weak-committed window with the delivered strong result.

        A held boundary ships now that the result is final.
        """
        key = (
            completion.request_key.operation_id,
            completion.request_key.window_id,
        )
        result = completion.result
        window = self.planner.windows_by_key[key]
        operation = self.tracker.operation_by_id[window.op_id]
        if self.pauli_frame is None:
            self._finish_strong_commit(
                completion, key, result, window, operation
            )
            return
        # the strong result is this window's FINAL correction: it folds into
        # the frame like any final result (the provisional weak bypassed it),
        # and the priced frame write gates the rest of the commit
        finish = functools.partial(
            self._finish_strong_commit,
            completion,
            key,
            result,
            window,
            operation,
        )
        self.pauli_frame.commit_correction(
            window_key=key,
            logical_observables=result.logical_observables,
            request_key=completion.request_key,
            on_committed=finish,
        )

    def _finish_strong_commit(
        self,
        completion: message.StrongDecodeCompletion,
        key: tuple,
        result: message.DecodeResult,
        window: message.Window,
        operation: message.Operation,
    ) -> None:
        if result.logical_observables is not None:
            self.ledger.replace_prediction(key, result.logical_observables)
        # the strong result is the window's final one: its status is now
        # published, and nothing of the operation waits on it any more
        window.published_request_key = completion.request_key
        held = self.courier.take_held(key)  # Held: ship now
        if held is not None:
            boundary = self.window_interaction.boundary_from_result(
                result, held.boundary
            )
            held_operation = self.tracker.operation_by_id[held.operation_id]
            self.courier.send(
                window,
                held_operation,
                boundary,
                source_request_key=completion.request_key,
            )
        committed_round_count = self._committed_round_count_by_stream.get(
            operation.id, 0
        )
        self.release_stream_segments_at_commit(
            operation.id, committed_round_count
        )
        self._finish_operation_if_ready(operation)
        self.finish_workload_if_ready()

    def _window_infos(self):
        infos = {}
        for key, window in self.planner.windows_by_key.items():
            infos[key] = message.WindowInfo.from_window(window)
        return types.MappingProxyType(infos)

    # ---- the operation's result: delivered once every window is final

    def _finish_operation_if_ready(self, operation: message.Operation) -> None:
        """Deliver the result once every window is final.

        Every window committed, no strong redo pending, the stream sealed.
        """
        if operation.id in self._finished_operation_ids:
            return
        if self._has_window_awaiting_strong(operation.id):
            return
        weak_store = self.retention.weak_store
        if weak_store.has_live_operation_reference(operation.id):
            return
        if self._strong_store_references(operation.id):
            return
        committed = self._committed_windows_of(operation.id)
        if len(committed) != self.planner.window_count_of(operation.id):
            return
        if not self.tracker.is_sealed(operation.id):
            return
        self._finished_operation_ids.add(operation.id)
        self._deliver_result(operation)
        self.retention.weak_store.close_operation(operation.id)
        self._close_strong_store_operation(operation.id)

    def _strong_store_references(self, operation_id) -> bool:
        strong_store = self.retention.strong_store
        if strong_store is None:
            return False
        return strong_store.has_live_operation_reference(operation_id)

    def _close_strong_store_operation(self, operation_id) -> None:
        strong_store = self.retention.strong_store
        if strong_store is None:
            return
        if strong_store.has_operation(operation_id):
            strong_store.close_operation(operation_id)

    def finish_workload_if_ready(self) -> None:
        """Tell the root once every window of the workload is final."""
        if self._workload_complete_sent:
            return
        if self._committed_window_count() != self.planner.total_windows:
            return
        if self._has_window_awaiting_strong(None):
            return
        if self.tracker.has_unsealed_streams():
            return
        if self.on_workload_complete is None:
            return
        self._workload_complete_sent = True
        self.on_workload_complete()

    def _deliver_result(self, operation: message.Operation) -> None:
        windows = self.planner.windows_of(operation.id)
        commit_los = [window.commit_lo for window in windows]
        commit_his = [window.commit_hi for window in windows]
        commit_lo = min(commit_los)
        commit_hi = max(commit_his)
        logical_observables = self.ledger.observables_for_interval(
            operation.id, commit_lo, commit_hi, boundary_policy="strict"
        )
        self._record_result(operation.id, logical_observables)
        self.conditional_release.release_waiters(operation)

    def _record_result(self, operation_id, logical_observables) -> None:
        if logical_observables is None:
            self.result_by_operation.pop(operation_id, None)
            return
        self.result_by_operation[operation_id] = logical_observables

    def release_stream_segments_at_commit(
        self, stream_id, committed_round_count: int
    ) -> None:
        """Deliver the segment results whose full round range committed.

        Gated as operations are: no pending strong may still change it.
        """
        operations = self.tracker.operation_by_id.values()
        operations = list(operations)
        for operation in operations:
            self._release_segment_if_committed(
                operation, stream_id, committed_round_count
            )

    def _release_segment_if_committed(
        self,
        operation: message.Operation,
        stream_id,
        committed_round_count: int,
    ) -> None:
        binding = self._stream_binding_by_operation_id.get(operation.id)
        operation_stream_id = operation.stream_id
        stream_offset = operation.stream_offset
        if binding is not None:
            operation_stream_id, stream_offset = binding
        if operation_stream_id != stream_id:
            return
        if operation.id not in self.tracker.blocking_operation_ids:
            return
        if operation.id in self.segment_results_sent:
            return
        segment_end = self._stream_segment_end(operation)
        if segment_end is None or segment_end > committed_round_count:
            return
        if self._segment_waits_for_strong(stream_id, segment_end):
            return
        segment_start = stream_offset + 1
        logical_observables = self.ledger.observables_for_interval(
            stream_id,
            segment_start,
            segment_end,
            boundary_policy="stream_segment",
        )
        self._record_result(operation.id, logical_observables)
        self.segment_results_sent.add(operation.id)
        self.conditional_release.release_waiters(operation)

    def _update_committed_round_count(self, stream_id) -> None:
        """Advance the stream's committed prefix; release what it covers."""
        committed = self._committed_prefix_round_count(stream_id)
        cached = self._committed_round_count_by_stream.get(stream_id, 0)
        if committed <= cached:
            return
        self._committed_round_count_by_stream[stream_id] = committed
        self.release_stream_segments_at_commit(stream_id, committed)

    def _segment_waits_for_strong(self, stream_id, segment_end: int) -> bool:
        for key, window in self.planner.windows_by_key.items():
            if key[0] != stream_id:
                continue
            if not _is_awaiting_strong(window):
                continue
            if window.commit_lo <= segment_end:
                return True
        return False

    def _committed_windows_of(self, operation_id) -> list:
        """The operation's committed windows, absorbed ones included."""
        committed = []
        for key, window in self.planner.windows_by_key.items():
            if key[0] != operation_id:
                continue
            if window.committed:
                committed.append(window)
        return committed

    def _committed_window_count(self) -> int:
        count = 0
        for window in self.planner.windows_by_key.values():
            if window.committed:
                count += 1
        return count

    def _has_window_awaiting_strong(self, operation_id) -> bool:
        """Whether a window of the operation (or of any) awaits its redo."""
        for key, window in self.planner.windows_by_key.items():
            if operation_id is not None and key[0] != operation_id:
                continue
            if _is_awaiting_strong(window):
                return True
        return False

    def _stream_segment_end(
        self, operation: message.Operation
    ) -> Optional[int]:
        required_end = self._required_stream_end_by_operation_id.get(
            operation.id
        )
        if required_end is not None:
            return required_end
        binding = self._stream_binding_by_operation_id.get(operation.id)
        stream_offset = operation.stream_offset
        if binding is not None:
            stream_offset = binding[1]
        if stream_offset is None:
            return None
        round_count = self.planner.round_count_of(operation.id)
        return stream_offset + round_count

    # ---- what the controller and the feedback streams ask

    def bind_stream_operation(
        self, operation_id: int, stream_id, stream_offset: int
    ) -> None:
        """Note which stream and offset a segment's rounds fold into."""
        self._stream_binding_by_operation_id[operation_id] = (
            stream_id,
            stream_offset,
        )

    def bind_required_stream_end(
        self, operation_id: int, required_stream_end: int
    ) -> None:
        """Note the stream round a protected segment's result waits for."""
        self._required_stream_end_by_operation_id[operation_id] = (
            required_stream_end
        )


def _is_awaiting_strong(window: message.Window) -> bool:
    """Committed provisionally: the strong redo has not published yet."""
    if not window.committed:
        return False
    if window.is_absorbed:
        return False
    return window.published_request_key is None


def _representative_patch(operation: message.Operation):
    """The patch a backlog row names: the first patch, qubit, or the id."""
    if operation.patches:
        return operation.patches[0]
    if operation.qubits:
        return operation.qubits[0]
    return operation.id
