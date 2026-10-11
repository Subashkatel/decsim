"""The operation results: delivered once every covering window is final.

The result is the ledger's XOR over the operation's commit range, or a
stream segment's, delivered to the conditional release once every
window is committed, no strong redo is pending, and the stream is
sealed; a segment releases when the stream's committed prefix covers
it (Skoric et al. 2209.08552: the logical correction of a stream is the
sum over its committed windows). The workload is complete once every
window is final.

An operation's windows commit ranges that follow one another in window
order (the scheme plans them so; a cut or a closed tail clips one in
place), so each prefix is kept as a cursor over the windows in order:
how many pass, and the last round they cover. A test that holds of a
window holds for good (committed, final, read out whole), so the cursor
only moves on, and a stream of any length costs the windows still open,
not every window.
"""

import dataclasses
import functools
from typing import Any, Optional

import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.trace_source as trace_source
import decsim.windows.committed_rounds as committed_rounds
import decsim.windows.round_retention as round_retention_module
import decsim.windows.round_tracker as round_tracker_module
import decsim.windows.window_planner as window_planner


class OperationResults:
    """Delivers each result once it is final.

    A result is an operation's or a stream segment's. Trace sources:
    operation_result_delivered(operation_id, logical_observables) at
    every delivery, and with None when a delivery is withdrawn; the
    result ledger listens. contributions_delivered(operation_id,
    contributions) at an operation's delivery, the contributions whose
    XOR it is, in extent order; read only when someone listens.
    """

    planner = ports.Port(window_planner.WindowPlanner)
    tracker = ports.Port(round_tracker_module.RoundTracker)
    retention = ports.Port(round_retention_module.RoundRetention)
    ledger = ports.Port(committed_rounds.LogicalLedger)
    conditional_release = ports.Port(ports.ReleaseReceiver)
    # the last result of the workload stops the factory producing
    factory = ports.Port(ports.MagicStateFactory)

    def __init__(self) -> None:
        self.deliveries = _Deliveries()
        self.trace = _TraceSources()

    # ---- what the committer tells

    def install_window_contribution(
        self,
        window: window_records.Window,
        logical_observables: Optional[tuple[int, ...]],
    ) -> decoding_records.LogicalContribution:
        """The window owns its commit range, unless a strong window does.

        Returns the contribution that owns the window's rounds.
        """
        existing = self.ledger.get(window.key)
        if existing is not None and existing.ownership_kind == "strong_window":
            return existing
        contribution = decoding_records.LogicalContribution(
            owner_key=window.key,
            commit_lo=window.commit_lo,
            commit_hi=window.commit_hi,
            ownership_kind="ordinary_window",
            logical_observables=logical_observables,
        )
        self.ledger.install(contribution)
        return contribution

    def replace_prediction(
        self, key: tuple, logical_observables: Optional[tuple[int, ...]]
    ) -> None:
        """A strong result replaces the owner's prediction."""
        self.ledger.replace_prediction(key, logical_observables)

    def note_window_committed(
        self, window: window_records.Window, is_final: bool
    ) -> None:
        """A window committed: advance the stream's prefix; free its context.

        No earlier escalation can re-slice its dependents now, so their
        potential restart reads end.
        """
        self._update_committed_round_count(window.operation_id)
        for dependent_key in window.dependents:
            self.retention.release_restart_reads(dependent_key)
        if is_final:
            self.retention.release_potential_strong(window.key)

    def note_window_final(self, window: window_records.Window) -> None:
        """A later revision finalized the window: each edge's read ends.

        A chain keeps the window's rounds in each pool's store until the
        pool's own verdict; the final revision ends what is left.
        """
        self.retention.release_potential_strong(window.key)

    # ---- delivery

    def deliver_if_final(self, operation: program_records.Operation) -> None:
        """Deliver the result once every window is final and the stream sealed.

        Every window committed, no strong redo pending, no store still
        referring to the operation, the stream sealed.
        """
        if operation.id in self.deliveries.finished_operation_ids:
            return
        if not self._is_final(operation.id):
            return
        self.deliveries.finished_operation_ids.add(operation.id)
        self._deliver_result(operation)
        weak_store = self.retention.weak_store
        if weak_store is not None:
            weak_store.close_operation(operation.id)
        self._close_strong_store_operation(operation.id)
        self.finish_workload_if_ready()

    def finish_workload_if_ready(self) -> None:
        """Tell the root once every window of the workload is final."""
        if self.deliveries.is_workload_done:
            return
        if not self._is_every_window_final():
            return
        if self.tracker.has_unsealed_streams():
            return
        self.deliveries.is_workload_done = True
        self.factory.shutdown()

    def release_committed_segments(
        self,
        stream_id: Any,  # an opaque identity
    ) -> None:
        """Deliver the segments the stream's committed prefix covers."""
        committed = self.deliveries.committed_round_count_by_stream.get(
            stream_id, 0
        )
        self.release_stream_segments_at_commit(stream_id, committed)

    def release_stream_segments_at_commit(
        self,
        stream_id: Any,  # an opaque identity
        committed_round_count: int,
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

    # ---- the segment binds

    def bind_stream_segment(
        self,
        operation_id: int,
        stream_id: Any,  # an opaque identity
        stream_offset: int,
    ) -> None:
        """Note which stream and offset a segment's rounds fold into."""
        segment = self._segment(operation_id)
        segment.stream_id = stream_id
        segment.stream_offset = stream_offset

    def bind_required_stream_end(
        self, operation_id: int, required_stream_end: int
    ) -> None:
        """Note the stream round a protected segment's result waits for."""
        segment = self._segment(operation_id)
        segment.required_stream_end = required_stream_end

    # ---- the committed prefix

    def committed_prefix_round_count(
        self,
        operation_id: Any,  # an opaque identity
    ) -> int:
        """Rounds committed in an unbroken prefix from round 1."""
        cursor = self._cursor(_COMMITTED, operation_id)
        return self._joined_prefix(cursor, operation_id, _is_committed)

    def read_out_prefix_round_count(
        self,
        operation_id: Any,  # an opaque identity
        rounds_read_out_by_operation: dict,
    ) -> int:
        """Rounds of the windows read out whole, unbroken from round 1.

        It is the final prefix an instant decoder on instant links would
        hold: a window is decoded only once its last round is measured
        (Skoric et al. 2209.08552 lines 416-418), so the rounds read out
        past it, a trailing buffer and the next window's rounds so far,
        wait on the window shape and on no decoder.
        """
        is_whole = functools.partial(
            _is_read_out_whole, self.tracker, rounds_read_out_by_operation
        )
        cursor = self._cursor(_READ_OUT, operation_id)
        return self._joined_prefix(cursor, operation_id, is_whole)

    def final_prefix_round_count(
        self,
        operation_id: Any,  # an opaque identity
    ) -> int:
        """Rounds whose final correction is in place, unbroken from round 1.

        A window that escalated holds only its provisional weak commit
        until the strong answer lands, so its rounds are not final yet;
        a window it absorbed lies after it and waits with it.
        """
        cursor = self._cursor(_FINAL, operation_id)
        return self._joined_prefix(cursor, operation_id, is_final_window)

    # ---- private

    def _segment(self, operation_id) -> "_Segment":
        segment = self.deliveries.segment_by_operation.get(operation_id)
        if segment is None:
            segment = _Segment()
            self.deliveries.segment_by_operation[operation_id] = segment
        return segment

    def _update_committed_round_count(self, stream_id) -> None:
        committed = self.committed_prefix_round_count(stream_id)
        by_stream = self.deliveries.committed_round_count_by_stream
        cached = by_stream.get(stream_id, 0)
        if committed <= cached:
            return
        by_stream[stream_id] = committed
        self.release_stream_segments_at_commit(stream_id, committed)

    def _deliver_result(self, operation: program_records.Operation) -> None:
        windows = self.planner.windows_of(operation.id)
        commit_los = [window.commit_lo for window in windows]
        commit_his = [window.commit_hi for window in windows]
        commit_lo = min(commit_los)
        commit_hi = max(commit_his)
        logical_observables = self.ledger.observables_for_interval(
            operation.id, commit_lo, commit_hi, boundary_policy="strict"
        )
        self._record_result(operation.id, logical_observables)
        self._report_contributions(operation.id, commit_lo, commit_hi)
        self.conditional_release.release_waiters(operation)

    def _report_contributions(
        self, operation_id, commit_lo: int, commit_hi: int
    ) -> None:
        if not self.trace.contributions_delivered.has_listeners:
            return
        contributions = self.ledger.contributions_for_interval(
            operation_id, commit_lo, commit_hi, boundary_policy="strict"
        )
        self.trace.contributions_delivered.fire(
            operation_id, tuple(contributions)
        )

    def _is_final(self, operation_id) -> bool:
        """Every window committed and final, no store holding, sealed."""
        if not self._are_windows_final(operation_id):
            return False
        weak_store = self.retention.weak_store
        if weak_store is not None and weak_store.has_live_operation_reference(
            operation_id
        ):
            return False
        if self._strong_store_references(operation_id):
            return False
        return self.tracker.is_sealed(operation_id)

    def _record_result(self, operation_id, logical_observables) -> None:
        self.trace.operation_result_delivered.fire(
            operation_id, logical_observables
        )

    def _release_segment_if_committed(
        self,
        operation: program_records.Operation,
        stream_id,
        committed_round_count: int,
    ) -> None:
        segment = self.deliveries.segment_by_operation.get(operation.id)
        operation_stream_id, stream_offset = _stream_place(operation, segment)
        if operation_stream_id != stream_id:
            return
        if operation.id not in self.tracker.blocking_operation_ids:
            return
        if operation.id in self.deliveries.segment_results_sent:
            return
        segment_end = self._stream_segment_end(
            operation, segment, stream_offset
        )
        is_final = self._is_segment_final(
            stream_id, segment_end, committed_round_count
        )
        if not is_final:
            return
        segment_start = stream_offset + 1
        logical_observables = self.ledger.observables_for_interval(
            stream_id,
            segment_start,
            segment_end,
            boundary_policy="stream_segment",
        )
        self._record_result(operation.id, logical_observables)
        self.deliveries.segment_results_sent.add(operation.id)
        self.conditional_release.release_waiters(operation)

    def _stream_segment_end(
        self, operation: program_records.Operation, segment, stream_offset
    ) -> Optional[int]:
        if segment is not None and segment.required_stream_end is not None:
            return segment.required_stream_end
        if stream_offset is None:
            return None
        round_count = self.planner.round_count_of(operation.id)
        return stream_offset + round_count

    def _is_segment_final(
        self,
        stream_id,
        segment_end: Optional[int],
        committed_round_count: int,
    ) -> bool:
        """Its end is known and committed, and no strong redo may change it."""
        if segment_end is None or segment_end > committed_round_count:
            return False
        is_waiting = self._segment_waits_for_strong(stream_id, segment_end)
        return not is_waiting

    def _segment_waits_for_strong(self, stream_id, segment_end: int) -> bool:
        """Whether a window awaiting its redo commits from the segment on.

        The windows before the final cursor are final, so only the open
        ones after it are read.
        """
        for window in self._open_windows(stream_id):
            if not is_awaiting_strong(window):
                continue
            if window.commit_lo <= segment_end:
                return True
        return False

    def _are_windows_final(self, operation_id) -> bool:
        """Whether every window of the operation is committed and final."""
        cursor = self._cursor(_EVERY_FINAL, operation_id)
        self._passed_in_order(cursor, operation_id)
        window_count = self.planner.window_count_of(operation_id)
        return cursor.passed == window_count

    def _is_every_window_final(self) -> bool:
        """Whether every window of every operation is committed and final."""
        for operation_id in self.planner.windowed_operation_ids():
            if not self._are_windows_final(operation_id):
                return False
        return True

    def _open_windows(self, operation_id) -> list:
        """The operation's windows from the first one not final on."""
        cursor = self._cursor(_EVERY_FINAL, operation_id)
        self._passed_in_order(cursor, operation_id)
        indices = self.planner.window_indices_of(operation_id)
        open_indices = indices[cursor.passed :]
        windows = []
        for window_index in open_indices:
            window = self.planner.window_at((operation_id, window_index))
            windows.append(window)
        return windows

    def _cursor(self, kind: str, operation_id) -> "_Cursor":
        cursors = self.deliveries.cursors
        key = (kind, operation_id)
        cursor = cursors.get(key)
        if cursor is None:
            cursor = _Cursor(_CURSOR_TESTS[kind])
            cursors[key] = cursor
        return cursor

    def _joined_prefix(self, cursor: "_Cursor", operation_id, passes) -> int:
        """The rounds the passing windows join from round 1, in window order.

        A window joins while it passes and starts no later than the round
        after the prefix, _unbroken_prefix_round_count's rule.
        """
        indices = self.planner.window_indices_of(operation_id)
        while cursor.passed < len(indices):
            window_index = indices[cursor.passed]
            window = self.planner.window_at((operation_id, window_index))
            if window.commit_lo > cursor.prefix_end + 1:
                break
            if not passes(window):
                break
            cursor.prefix_end = max(cursor.prefix_end, window.commit_hi)
            cursor.passed += 1
        return cursor.prefix_end

    def _passed_in_order(self, cursor: "_Cursor", operation_id) -> None:
        """The cursor past every window, in order, its test holds of."""
        indices = self.planner.window_indices_of(operation_id)
        while cursor.passed < len(indices):
            window_index = indices[cursor.passed]
            window = self.planner.window_at((operation_id, window_index))
            if not cursor.passes(window):
                return
            cursor.passed += 1

    def _strong_store_references(self, operation_id) -> bool:
        """Whether a live hold in a store a region lands in names it."""
        for store in self.retention.landing_stores():
            if store.has_live_operation_reference(operation_id):
                return True
        return False

    def _close_strong_store_operation(self, operation_id) -> None:
        """Retire the operation in every store a region lands in."""
        for store in self.retention.landing_stores():
            if store.has_operation(operation_id):
                store.close_operation(operation_id)


def is_final_window(window: window_records.Window) -> bool:
    """Committed, and not waiting for a strong redo to publish."""
    if not window.committed:
        return False
    return not is_awaiting_strong(window)


def is_awaiting_strong(window: window_records.Window) -> bool:
    """Committed provisionally: the strong redo has not published yet."""
    if not window.committed:
        return False
    if window.is_absorbed:
        return False
    return window.published_request_key is None


def _is_committed(window: window_records.Window) -> bool:
    return window.committed


def _is_read_out_whole(
    tracker, rounds_read_out_by_operation: dict, window
) -> bool:
    return tracker.is_read_out_whole(window, rounds_read_out_by_operation)


def _stream_place(operation: program_records.Operation, segment) -> tuple:
    """The stream and offset the operation sits at; a bound segment's first."""
    if segment is None or segment.stream_id is None:
        return operation.stream_id, operation.stream_offset
    return segment.stream_id, segment.stream_offset


class _Deliveries:
    """The progress of the deliveries so far."""

    def __init__(self) -> None:
        self.is_workload_done = False
        self.finished_operation_ids: set = set()
        self.segment_results_sent: set = set()
        self.committed_round_count_by_stream: dict = {}
        self.segment_by_operation: dict = {}
        # (test, operation id) -> _Cursor
        self.cursors: dict = {}


class _Cursor:
    """How many of an operation's windows, in order, pass one test.

    prefix_end is the last round their commit ranges join from round 1,
    for a cursor that joins them; passes is the test, for one that only
    counts.
    """

    def __init__(self, passes) -> None:
        self.passes = passes
        self.passed = 0
        self.prefix_end = 0


class _Segment:
    """Where one operation's rounds fold into a stream.

    required_stream_end is the stream round its result waits for.
    """

    def __init__(self) -> None:
        self.stream_id = None
        self.stream_offset = None
        self.required_stream_end = None


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the operation results reports, as one member."""

    operation_result_delivered: trace_source.TraceSource = (
        trace_source.new_source()
    )
    contributions_delivered: trace_source.TraceSource = (
        trace_source.new_source()
    )


_COMMITTED = "committed"
_READ_OUT = "read out whole"
_FINAL = "final"
_EVERY_FINAL = "every final"
# the test a counting cursor applies; a joining one is given its own
_CURSOR_TESTS = {
    _COMMITTED: _is_committed,
    _READ_OUT: None,
    _FINAL: is_final_window,
    _EVERY_FINAL: is_final_window,
}
