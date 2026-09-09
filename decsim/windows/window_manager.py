"""The windows facade: the window life cycle of every operation.

Which windows exist is the WindowPlanner's, which rounds arrived and
whether a window has its data is the RoundTracker's, which rounds each
window holds is the RoundRetention's, a decode is asked for by the
DecodeRequester from a job the DecodeRequestBuilder builds, its input
is sent by the store's own RoundStoreOutput, a result commits its
window through the WindowCommitter and leaves for the frame through the
decoder side's DecoderOutput, boundaries between windows are the
BoundaryCourier's, ownership of
committed rounds is the LogicalLedger's, and each operation's result is
the OperationResults'; the facade receives rounds and wires them, the
shape of gem5's cache (BaseCache owns its MSHR queue, write buffer and
tags, each one job, and implements the ports: src/mem/cache/base.hh).
The strong tier is the StrongRedecode's (decsim/escalation), on the
same components; a run that never escalates has none. One round reads
as accept_window_input, requester.request_if_ready, job.on_decoded
(verdict.accept_result), results.deliver_if_final.

Wide state recorded: ten attributes, the seven components a round
crosses, the strong redecode the arrivals wake, the interaction that
gives a new window its first boundary and the workload's feedback mode.
"""

import dataclasses
from typing import Optional, Protocol, runtime_checkable

import decsim.records.identity as identity_records
import decsim.records.log_sources as log_sources
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records


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
        requester,
        courier,
        results,
        strong_redecode,
        window_interaction,
        feedback_boundary_mode: str = "trailing_buffer",
    ):
        self.engine = engine
        self.planner = planner
        self.tracker = tracker
        self.retention = retention
        self.requester = requester
        self.courier = courier
        self.results = results
        # the strong tier's window side; None when the run never escalates
        self.strong_redecode = strong_redecode
        self.window_interaction = window_interaction
        self.feedback_boundary_mode = feedback_boundary_mode
        for window in self.planner.windows_by_key.values():
            window_info = window_records.WindowInfo.from_window(window)
            window.boundary_in = self.window_interaction.initial_boundary_state(
                window_info
            )

    @property
    def ledger(self):
        """The logical ledger: which owner committed which rounds."""
        return self.results.ledger

    @property
    def windows(self) -> dict:
        """Every window by key; the planner's table."""
        return self.planner.windows_by_key

    # ---- the plan: operations, streams, windows and their holds

    def register_operation(self, operation: program_records.Operation) -> None:
        """Track an operation's rounds, payload RAM, and feedback role."""
        is_new = self.tracker.register_operation(operation)
        if is_new:
            self.retention.open_operation_store(operation.id)

    def register_stream(
        self, stream_operation: program_records.Operation
    ) -> None:
        """Register a stream whose windows are created at runtime."""
        resolved_feedback_mode = stream_operation.feedback_boundary_mode
        if resolved_feedback_mode is None:
            resolved_feedback_mode = self.feedback_boundary_mode
        stream_operation = dataclasses.replace(
            stream_operation, feedback_boundary_mode=resolved_feedback_mode
        )
        self.retention.open_operation_store(stream_operation.id)
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

    def _admit_stream_window(self, window: window_records.Window) -> None:
        """Connect one new stream window: its boundary, its model, its holds.

        If the previous boundary already arrived, apply it immediately.
        """
        window_info = window_records.WindowInfo.from_window(window)
        window.boundary_in = self.window_interaction.initial_boundary_state(
            window_info
        )
        if window.window_index > 0:
            previous_key = (window.operation_id, window.window_index - 1)
            self._link_to_previous_window(previous_key, window.key, window)
        self.planner.attach_stream_model(window)
        self.retention.register_window(window.key, window)

    def _link_to_previous_window(
        self, previous_key: tuple, key: tuple, window: window_records.Window
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
        self.results.finish_workload_if_ready()

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

    def accept_window_input(
        self, packet: round_records.SyndromeRoundPacket
    ) -> None:
        """Publish one stored upstream round to window readiness.

        Assembly-to-retention is a state transition on the same
        allocation; no decoder input moves here. Window input transfer
        begins only when a decode request is admitted.
        """
        operation = self.tracker.operation_by_id[packet.operation_id]
        self._refuse_unplanned_round(packet, operation)
        if self.retention.primary_tier is not window_records.DecoderTier.STRONG:
            # Buffer 0 publication is the readiness authority for the weak lane
            self._count_arrival(operation, packet.round_index)
            self._update_stream(operation.id)
            self._wake_strong_tier(operation.id)
            self._wake_windows(operation)
        # a round whose every consumer already resolved (an absorbed window's
        # tail, or every round of a strong-primary plan) frees its Buffer 0
        # slot on arrival, the same drop-on-arrival rule syndrome buffer 1
        # applies
        round_key = (packet.operation_id, packet.round_index)
        self.retention.release_round_if_unheld(round_key)

    def _refuse_unplanned_round(
        self,
        packet: round_records.SyndromeRoundPacket,
        operation: program_records.Operation,
    ) -> None:
        """A round past the plan is the device's mistake (it is a plug-in)."""
        if not self.retention.has_operation_store(operation.id):
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
            log_sources.DECODER_MANAGER,
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
        self._wake_strong_tier(operation_id)
        if self.retention.primary_tier is not window_records.DecoderTier.STRONG:
            return
        operation = self.tracker.operation_by_id[operation_id]
        stored_through = self.tracker.strong_rounds_arrived(operation_id)
        self._count_arrival(operation, stored_through)
        self._update_stream(operation_id)
        self._wake_windows(operation)

    def _count_arrival(
        self, operation: program_records.Operation, round_index: int
    ) -> None:
        """Advance the readiness arrival counter; the authority calls this."""
        arrived_now = self.tracker.note_arrival(operation.id, round_index)
        self.engine.log(
            log_sources.DECODER_MANAGER,
            f"round {round_index} of {operation.name} arrived "
            f"(op now has rounds 1..{arrived_now})",
        )

    def _wake_strong_tier(self, operation_id) -> None:
        """A round is stored: a terminal strong window may have its tail."""
        if self.strong_redecode is not None:
            self.strong_redecode.submit_if_stored_data_releases(operation_id)

    def _wake_windows(self, operation: program_records.Operation) -> None:
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
        self.requester.request_ready_windows(windows, self.strong_redecode)

    def check_window(self, key: tuple) -> None:
        """Request the window if it has its data."""
        window = self.planner.windows_by_key[key]
        self.requester.request_if_ready(window, self.strong_redecode)

    def accept_boundary(self, key: tuple, is_unblocked: bool) -> None:
        """A boundary landed: wake the parked decode, or request the window."""
        if is_unblocked:
            self.requester.release_parked(key)
        self.check_window(key)

    def check_settled(self) -> None:
        """At the end of a run every window's solves have been joined."""
        self.requester.check_settled()

    # ---- observation

    def rounds_backlog(self) -> tuple:
        """Rounds arrived but not decoded, per operation in stable order.

        A row is (operation id, patch, rounds arrived past the unbroken
        decoded prefix from round 1).
        """
        rows = []
        ordered_ids = sorted(
            self.tracker.operation_by_id,
            key=identity_records.stable_identity_order_key,
        )
        for operation_id in ordered_ids:
            operation = self.tracker.operation_by_id[operation_id]
            decoded = self.results.committed_prefix_round_count(operation_id)
            arrived = self.tracker.rounds_arrived(operation_id)
            undecoded = arrived - decoded
            waiting = max(0, undecoded)
            patch = _representative_patch(operation)
            rows.append((operation_id, patch, waiting))
        return tuple(rows)

    # ---- what the feedback streams ask

    def bind_stream_operation(
        self, operation_id: int, stream_id, stream_offset: int
    ) -> None:
        """Note which stream and offset a segment's rounds fold into."""
        self.results.bind_stream_segment(operation_id, stream_id, stream_offset)

    def bind_required_stream_end(
        self, operation_id: int, required_stream_end: int
    ) -> None:
        """Note the stream round a protected segment's result waits for."""
        self.results.bind_required_stream_end(operation_id, required_stream_end)


def _representative_patch(operation: program_records.Operation):
    """The patch a backlog row names: the first patch, qubit, or the id."""
    if operation.patches:
        return operation.patches[0]
    if operation.qubits:
        return operation.qubits[0]
    return operation.id
