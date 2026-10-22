"""The windows facade: the window life cycle of every operation.

The facade receives rounds and hands them to components with one job
each: the WindowPlanner (which windows exist), the RoundTracker (which
rounds arrived), the RoundRetention (which rounds each window holds),
the DecodeRequester and its DecodeRequestBuilder (a decode per complete
window), the WindowCommitter (a result commits its window), the
BoundaryCourier (boundaries between windows), the LogicalLedger (who
committed which rounds) and the OperationResults (each operation's
result). This is the shape of gem5's cache, whose BaseCache owns its
MSHR queue, write buffer and tags and implements the ports
(src/mem/cache/base.hh). Each escalate edge's target is that edge's
StrongRedecode's (decsim/escalation), on the same components.
"""

import dataclasses
from typing import Any, Optional, Union

import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.identity as identity_records
import decsim.records.log_sources as log_sources
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
import decsim.trace_source as trace_source
import decsim.windows.decode_requests as decode_requests
import decsim.windows.escalate_edges as escalate_edges
import decsim.windows.operation_results as operation_results
import decsim.windows.round_retention as round_retention_module
import decsim.windows.round_tracker as round_tracker_module
import decsim.windows.window_boundaries as window_boundaries
import decsim.windows.window_interactions as window_interactions
import decsim.windows.window_planner as window_planner


@dataclasses.dataclass(frozen=True)
class WindowTraceSources:
    """One window's life as the observers hear it, named once by the facade.

    window_planned when the plan or a stream lays a window,
    window_data_complete when its rounds are all there,
    solve_held when one of a window's several solves waits for the
    others, window_committed when its correction is the frame's,
    window_final once its record is final, and window_absorbed when a
    strong window takes it over instead.
    edge_traces holds each escalate edge's own sources, in graph order:
    a strong window held for its conditions, and its leaving.
    """

    window_planned: trace_source.TraceSource
    window_data_complete: trace_source.TraceSource
    window_committed: trace_source.TraceSource
    window_final: trace_source.TraceSource
    # the silent source on a run that escalates nothing
    window_absorbed: Union[trace_source.TraceSource, trace_source.SilentSource]
    # the silent source on a run whose confidence reads one solve
    solve_held: Union[trace_source.TraceSource, trace_source.SilentSource]
    edge_traces: tuple = ()


class WindowManager:
    """The windows facade: receives rounds, closes windows, commits results."""

    planner = ports.Port(window_planner.WindowPlanner)
    tracker = ports.Port(round_tracker_module.RoundTracker)
    retention = ports.Port(round_retention_module.RoundRetention)
    requester = ports.Port(decode_requests.DecodeRequester)
    courier = ports.Port(window_boundaries.BoundaryCourier)
    results = ports.Port(operation_results.OperationResults)
    # the run's escalate edges; none when the run never escalates
    edges = ports.Port(escalate_edges.EscalateEdges)
    window_interaction = ports.Port(window_interactions.WindowInteraction)

    def __init__(
        self,
        engine: engine_module.Engine,
        feedback_boundary_mode: str = "trailing_buffer",
    ):
        self.engine = engine
        self.feedback_boundary_mode = feedback_boundary_mode
        # operation id -> position of its first window neither queued nor
        # committed; both hold until a withdrawal moves it back
        self.first_open_by_stream: dict = {}

    def start(self) -> None:
        """Give every window the plan laid its first boundary.

        It runs once the ports are bound, as gem5's startup does
        (src/sim/sim_object.hh lines 194 and 280).
        """
        for window in self.planner.windows_by_key.values():
            window_info = window_records.WindowInfo.from_window(window)
            window.boundary_in = self.window_interaction.initial_boundary_state(
                window_info
            )

    # ---- the plan: operations, streams, windows and their holds

    def register_operation(self, operation: program_records.Operation) -> None:
        """Track an operation's rounds, payload RAM, and feedback role."""
        is_new = self.tracker.register_operation(operation)
        if is_new:
            self.retention.open_stream_store(operation.id)

    def register_stream(
        self,
        stream_operation: program_records.Operation,
        physical_round_limit: Optional[int],
    ) -> None:
        """Register a stream whose windows are created at runtime."""
        resolved_feedback_mode = stream_operation.feedback_boundary_mode
        if resolved_feedback_mode is None:
            resolved_feedback_mode = self.feedback_boundary_mode
        stream_operation = dataclasses.replace(
            stream_operation, feedback_boundary_mode=resolved_feedback_mode
        )
        self.retention.open_stream_store(stream_operation.id)
        source_round_limit = self.planner.register_stream(
            stream_operation, physical_round_limit
        )
        self.tracker.register_stream(stream_operation, source_round_limit)

    def install_planned_holds(
        self, buffering_plan: decoding_records.SyndromeBufferingPlan
    ) -> None:
        """Check the stores against the plan and place its holds."""
        self.retention.install_planned_holds(buffering_plan)

    # ---- dynamic streams: their windows are planned as rounds arrive

    def _update_stream(self, stream_id) -> None:
        """Grow (and maybe seal) the stream as one more round arrives."""
        if not self.planner.has_stream(stream_id):
            return
        highest_known_round = self.tracker.rounds_arrived(stream_id)
        self._grow_stream(stream_id, highest_known_round, None)
        if self.tracker.is_sealed(stream_id):
            return
        if self.tracker.reaches_source_limit(stream_id):
            # the source declared this limit, so it needs no attesting
            limit = self.tracker.source_round_limit(stream_id)
            self.seal_stream(stream_id, limit)

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
        """Connect one new stream window: its boundary, its model, its holds."""
        window_info = window_records.WindowInfo.from_window(window)
        window.boundary_in = self.window_interaction.initial_boundary_state(
            window_info
        )
        if window.window_index > 0:
            previous_key = (window.stream_id, window.window_index - 1)
            self._link_to_previous_window(previous_key, window.key, window)
        self._link_across_couplings(window)
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
        window.deps_remaining += 1
        previous = self.planner.windows_by_key[previous_key]
        previous.dependents.append(key)

    def _link_across_couplings(self, window: window_records.Window) -> None:
        """The window waits for each region reading it, and is waited for.

        The window laid second makes the link: a window counts every
        reader it waits for when it is laid, laid or not, and a reader
        laid later links itself to each window it reads that is laid. A
        reader needs every round of the region it reads, and the window
        on that region is laid at its first, so no reader has committed
        when the window it reads is laid.
        """
        for reader_key in self.planner.reader_keys_of(window):
            window.deps_remaining += 1
            if reader_key is not None:
                self._link_reader(reader_key, window.key)
        for read_key in self.planner.begun_read_keys_of(window):
            self._link_reader(window.key, read_key)

    def _link_reader(self, reader_key: tuple, read_key: tuple) -> None:
        assert not self.courier.has_committed(reader_key), (
            f"{reader_key} committed before the window it reads was laid"
        )
        reader = self.planner.windows_by_key[reader_key]
        read = self.planner.windows_by_key[read_key]
        reader.dependents.append(read_key)
        read.deps.append(reader_key)

    def seal_stream(
        self,
        stream_id: Any,  # an opaque identity
        stream_round_count: int,
    ) -> None:
        """Close a dynamic stream once its full length has arrived."""
        if self.tracker.is_sealed(stream_id):
            return
        stream_operation = self.tracker.operation(stream_id)
        models_changed = self.planner.models.finalize_stream_models(
            stream_operation, stream_round_count
        )
        self._grow_stream(stream_id, stream_round_count, stream_round_count)
        self._close_stream_tail(stream_id, stream_round_count)
        if models_changed:
            self.planner.refresh_stream_models(stream_id)
        self.tracker.seal(stream_id, stream_round_count)
        self.retention.release_later_stream_reads(stream_id)
        self.check_windows_of_stream(stream_id)
        self.results.finish_workload_if_ready()

    def _close_stream_tail(self, stream_id, stream_round_count: int) -> None:
        """The windows the seal moved read as far as they now reach.

        A clipped window reads no round past the seal; a window laid
        again reads its whole new extent (window_records.TailChange).
        """
        change = self.planner.close_stream_tail(stream_id, stream_round_count)
        for window in change.clipped:
            self.retention.reset_clipped_window_reads(window)
        self._replace_reads_of(change.relaid)

    def _replace_reads_of(self, windows) -> None:
        """Re-point the retained reads of each window laid again."""
        for window in windows:
            self.retention.replace_window_reads(window.key, window)

    def close_stream_boundary(
        self,
        stream_id: Any,  # an opaque identity
        stream_round_count: int,
    ) -> None:
        """Mark a live stream round as a measurement-closed boundary.

        The window whose commit region holds the boundary commits through
        it, and the next window starts after it. A finite source's
        windows are its own, and it closes only at its end.
        """
        if not self.planner.has_stream(stream_id):
            return
        self.tracker.close_boundary(stream_id, stream_round_count)
        self._grow_stream(stream_id, stream_round_count, None)
        self.planner.close_stream_tail(stream_id, stream_round_count)
        self._refresh_unqueued_stream_windows(stream_id)
        self.check_windows_of_stream(stream_id)

    def _refresh_unqueued_stream_windows(self, stream_id) -> None:
        """Refresh retained reads once a stream boundary closes the tail."""
        windows = self.planner.windows_of(stream_id)
        unqueued = [
            window
            for window in windows
            if not window.queued and not window.committed
        ]
        self._replace_reads_of(unqueued)

    def _cut_stream_after(self, stream_id, last_round: int) -> None:
        """End a window of the stream on the round; refresh those moved."""
        moved = self.planner.cut_stream_after(stream_id, last_round)
        if not moved:
            return
        self.planner.refresh_stream_models(stream_id)
        self._refresh_unqueued_stream_windows(stream_id)

    def let_go_before(
        self,
        stream_id: Any,  # an opaque identity
        window_index: int,
    ) -> None:
        """The first open window is past the stream's windows let go.

        They are committed (window_release.py), so the walk from the
        first open window passes them anyway.
        """
        first_open = self.first_open_by_stream.get(stream_id, 0)
        first_open = max(first_open, window_index)
        self.first_open_by_stream[stream_id] = first_open

    def has_dynamic_stream(self, stream_id: Any) -> bool:  # an opaque identity
        """True for a stream whose windows are planned at runtime."""
        return self.planner.has_stream(stream_id)

    # ---- arrivals: the WindowInput port and the strong syndrome buffer's
    # signal

    def accept_window_input(
        self, packet: round_records.SyndromeRoundPacket
    ) -> None:
        """Publish one stored upstream round to window readiness.

        No decoder input moves here: a window's input leaves its store
        when its decode request is admitted.
        """
        operation = self.tracker.operation_by_id[packet.stream_id]
        self._refuse_unplanned_round(packet, operation)
        # The weak syndrome buffer publication is the readiness authority
        # for the weak lane; a strong-primary run's rounds never come here
        # (controller/syndrome_round_sender.py writes them to the strong
        # store alone)
        self._count_arrival(operation, packet.round_index)
        self._update_stream(operation.id)
        self._wake_strong_tier(operation.id)
        self._wake_windows(operation)
        self._wake_readers(packet.stream_id, packet.round_index)
        # a round whose every consumer already resolved (an absorbed window's
        # tail) frees its weak syndrome buffer slot on arrival, the same
        # drop-on-arrival rule the strong syndrome buffer applies
        round_key = (packet.stream_id, packet.round_index)
        self.retention.release_round_if_unheld(round_key)

    def _refuse_unplanned_round(
        self,
        packet: round_records.SyndromeRoundPacket,
        operation: program_records.Operation,
    ) -> None:
        """A round past the plan is the device's mistake (it is a plug-in)."""
        if not self.retention.has_stream_store(operation.id):
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

    def accept_feedback_memory_round(
        self,
        source_stream_id: Any,  # an opaque identity
    ) -> None:
        """Record one idle or memory round and re-check waiting windows."""
        self.tracker.note_memory_round(source_stream_id)
        self.check_windows_of_stream(source_stream_id)

    def accept_room_round(
        self,
        store: ports.SyndromeBuffer,
        stream_id: Any,  # an opaque identity
        round_index: int,
    ) -> None:
        """A store a region lands in stored a round.

        Wake every edge's listeners, and drive readiness when the store
        is the one the primary tier reads.
        """
        self.tracker.note_room_round(store, stream_id, round_index)
        self._wake_strong_tier(stream_id)
        if self.retention.primary_store is not store:
            return
        operation = self.tracker.operation_by_id[stream_id]
        stored_through = self.tracker.strong_rounds_arrived(stream_id, store)
        self._count_arrival(operation, stored_through)
        self._update_stream(stream_id)
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

    def _wake_strong_tier(self, stream_id) -> None:
        """A round is stored: a terminal strong window may have its tail."""
        for edge in self.edges.edges:
            edge.submit_if_stored_data_releases(stream_id)

    def _wake_windows(self, operation: program_records.Operation) -> None:
        self.check_windows_of_stream(operation.id)
        for predecessor_id in operation.decoder_boundary_predecessors:
            self.check_windows_of_stream(predecessor_id)

    def _wake_readers(self, stream_id, round_index: int) -> None:
        """A round of a region read across a coupling: check its readers."""
        reader_keys = self.planner.reader_keys_at(stream_id, round_index)
        for reader_key in reader_keys:
            self.check_window(reader_key)

    def prepend_idle_rounds(self, stream_id: int, round_count: int) -> None:
        """Fold pre-gate idle rounds into a batch-style operation's stream."""
        self.planner.prepend_idle_rounds(stream_id, round_count)

    # ---- readiness: a window with its data and no owed boundary is requested

    def check_windows_of_stream(self, stream_id: int) -> None:
        """Request every complete window of the stream, in index order.

        The windows before its first open one are passed by, as a request
        leaves a queued or committed window alone, and the walk ends where
        no window from there on has its first round, as a request neither
        stamps nor submits a window before its first round. Windows need
        not start in index order (a Tan seam starts before the core ahead
        of it), so the end is read off the planner's start floors. A long
        run costs its open windows a round rather than all of them.
        """
        windows = self._windows_from_the_first_open(stream_id)
        self.requester.request_ready_windows(windows)

    def check_window(self, key: tuple) -> None:
        """Request the window if it has its data."""
        window = self.planner.windows_by_key[key]
        self.requester.request_if_ready(window)

    def accept_boundary(self, key: tuple, is_unblocked: bool) -> None:
        """A boundary landed: release the waiting decode, or request it."""
        if is_unblocked:
            self.requester.release_window(key)
        self.check_window(key)

    def check_settled(self) -> None:
        """At the end of a run every window's solves have been joined."""
        self.requester.check_settled()

    # ---- observation

    def backlog_round_counts(self, rounds_read_out_by_stream: dict) -> tuple:
        """The rounds read out whole and not yet final, per stream.

        A row is (stream id, patch, rounds), per stream in stable
        order: the unbroken prefix of windows whose rounds are all read
        out, less the unbroken prefix of final windows. Terhal's backlog
        is syndrome data generated and not yet processed (quoted in
        Skoric et al. 2209.08552 lines 46-55), and the Pauli frame may lag
        by a constant (Terhal 1302.3428 Sec. II.G.3); the rounds read
        out past the first prefix are that constant, the window shape's.
        Toshio et al. 2510.25222 keep r_op, the rounds the circuit spends,
        apart from the rounds the decoding adds (eq. (5), lines 1010-1012
        of the local text); taking the window shape's lag as that baseline
        is decsim's reading, not the paper's. A window waiting for its
        strong answer holds a provisional commit only, so it ends the
        final prefix.
        """
        rows = []
        ordered_ids = sorted(
            self.tracker.operation_by_id,
            key=identity_records.stable_identity_bytes,
        )
        for stream_id in ordered_ids:
            operation = self.tracker.operation_by_id[stream_id]
            whole = self.results.read_out_prefix_round_count(
                stream_id, rounds_read_out_by_stream
            )
            final = self.results.final_prefix_round_count(stream_id)
            waiting = whole - final
            patch = _representative_patch(operation)
            rows.append((stream_id, patch, waiting))
        return tuple(rows)

    def window_sources(self) -> WindowTraceSources:
        """The window life cycle's trace sources, in the order they fire.

        A run that never escalates absorbs no window and a run whose
        confidence reads one solve holds none, so each reports the
        silent source instead. Only the edge from the window pool may
        absorb windows (escalation/strong_window_shapes.py DoubleWindow).
        """
        absorbed = trace_source.SILENT
        primary_tier = self.retention.primary_tier
        edge = self.edges.leaving(primary_tier)
        if edge is not None:
            absorbed = edge.shape.window_absorbed
        held = trace_source.SILENT
        gap_join = self.requester.gap_join
        if gap_join is not None:
            held = gap_join.trace.solve_held
        builder = self.requester.builder
        committer = self.requester.verdict.committer
        edge_traces = []
        for each_edge in self.edges.edges:
            edge_traces.append(each_edge.trace)
        return WindowTraceSources(
            window_planned=self.planner.trace.window_planned,
            window_data_complete=builder.trace.window_data_complete,
            window_committed=committer.trace.window_committed,
            window_final=committer.trace.window_final,
            window_absorbed=absorbed,
            solve_held=held,
            edge_traces=tuple(edge_traces),
        )

    def planned_windows(self) -> dict:
        """Every window the plan laid, by key, for a ledger to load once."""
        return self.planner.windows_by_key

    def copy_sources(self) -> list:
        """The copy_made sources of this side's own hops: none.

        Every copy on the window side's hops lands in a structure
        another package owns, which books it.
        """
        return []

    def reads_windows_from(self, store: Optional[ports.SyndromeBuffer]) -> bool:
        """Whether the primary tier's window reads come from this store."""
        return self.retention.primary_store is store

    # ---- what the feedback streams ask

    def bind_stream_operation(
        self,
        operation_id: int,
        stream_id: Any,  # an opaque identity
        stream_offset: int,
    ) -> None:
        """Note which stream and offset a segment's rounds fold into.

        The segment's first round starts a window of its stream and its
        last round ends one, so its result (operation_results.py) is a
        sum over whole windows.
        """
        self.results.bind_stream_segment(operation_id, stream_id, stream_offset)
        if not self.planner.has_stream(stream_id):
            return
        round_count = self.planner.round_count_of(operation_id)
        segment_end = stream_offset + round_count
        self._cut_stream_after(stream_id, stream_offset)
        self._cut_stream_after(stream_id, segment_end)

    def couple_segments(self, spans_by_block: dict, couplings: tuple) -> None:
        """A joint measurement's segments start and end regions, read across.

        A cut can end the last region begun early, and with it the
        buffer of the window before, so its stream's windows are checked.
        """
        self.planner.couple_segments(spans_by_block, couplings)
        for span in spans_by_block.values():
            moved = self.planner.relay_stream(span.stream_id)
            self._replace_reads_of(moved)
            self.check_windows_of_stream(span.stream_id)

    def bind_required_stream_end(
        self, operation_id: int, required_stream_end: int
    ) -> None:
        """Note the stream round a protected segment's result waits for."""
        self.results.bind_required_stream_end(operation_id, required_stream_end)

    def _windows_from_the_first_open(self, stream_id: int) -> list:
        """The stream's windows from its first open one to the last begun."""
        indices = self.planner.window_indices_of(stream_id)
        first_open = self._first_open_after_withdrawals(stream_id, indices)
        while first_open < len(indices):
            window_index = indices[first_open]
            window = self.planner.window_at((stream_id, window_index))
            if not window.queued and not window.committed:
                break
            first_open += 1
        self.first_open_by_stream[stream_id] = first_open
        floors = self.planner.start_floors_of(stream_id)
        arrived = self.tracker.rounds_arrived(stream_id)
        windows = []
        for position in range(first_open, len(indices)):
            if floors[position] > arrived:
                break
            window_index = indices[position]
            window = self.planner.window_at((stream_id, window_index))
            windows.append(window)
        return windows

    def _first_open_after_withdrawals(
        self, stream_id: int, indices: list
    ) -> int:
        """The cursor, back at the earliest window withdrawn since."""
        first_open = self.first_open_by_stream.get(stream_id, 0)
        withdrawn = self.requester.withdrawn_keys
        for key in list(withdrawn):
            if key[0] != stream_id:
                continue
            withdrawn.discard(key)
            position = indices.index(key[1])
            first_open = min(first_open, position)
        return first_open


def _representative_patch(operation: program_records.Operation):
    """The patch a backlog row names: the first patch, qubit, or the id."""
    if operation.patches:
        return operation.patches[0]
    if operation.qubits:
        return operation.qubits[0]
    return operation.id
