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
import decsim.windows.release_rules as release_rules
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
    # which segments a blocker's release waits for
    release_rule = ports.Port(release_rules.ReleaseRule)

    def __init__(self) -> None:
        self.deliveries = _Deliveries()
        self.bindings = _Bindings()
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
        self._update_committed_round_count(window.stream_id)
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

        Every window committed, no strong redo pending, the stream
        sealed: no decode can change the result after that. A decode
        whose answer no longer counts, a strong one let run out
        (revision_rules.NoCancel), may still read the stream's rounds;
        each store retires the stream once its last hold ends.
        """
        if operation.id in self.deliveries.finished_operation_ids:
            return
        if not self._is_final(operation.id):
            return
        self.deliveries.finished_operation_ids.add(operation.id)
        self._deliver_result(operation)
        for store in self.retention.entry_stores():
            store.close_stream(operation.id)
        self._close_strong_store_stream(operation.id)
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
        """Deliver the segment results whose full round range committed.

        Gated as operations are: no pending strong may still change it.
        A commit may release any blocker the release rule makes wait for
        an operation bound to the stream.
        """
        bound = self.bindings.operation_ids_by_stream.get(stream_id, ())
        for operation_id in list(bound):
            blocker_ids = self.release_rule.awaiting(operation_id)
            self._release_if_committed(blocker_ids)

    # ---- the segment binds

    def bind_stream_segment(
        self,
        operation_id: int,
        stream_id: Any,  # an opaque identity
        stream_offset: int,
    ) -> None:
        """Note a stream and offset the operation's rounds fold into.

        An operation on several patch streams is a segment of each.
        """
        segment = _Segment(stream_id, stream_offset)
        segments = self.bindings.segments_by_operation.setdefault(
            operation_id, []
        )
        segments.append(segment)
        bound = self.bindings.operation_ids_by_stream.setdefault(stream_id, [])
        bound.append(operation_id)

    def bind_required_stream_end(
        self, operation_id: int, required_stream_end: int
    ) -> None:
        """Note the stream round a protected segment's result waits for."""
        segments = self.bindings.segments_by_operation[operation_id]
        segments[-1].required_stream_end = required_stream_end

    # ---- the committed prefix

    def committed_prefix_round_count(
        self,
        stream_id: Any,  # an opaque identity
    ) -> int:
        """Rounds committed in an unbroken prefix from round 1."""
        cursor = self._cursor(_COMMITTED, stream_id)
        return self._joined_prefix(cursor, stream_id, _is_committed)

    def read_out_prefix_round_count(
        self,
        stream_id: Any,  # an opaque identity
        rounds_read_out_by_stream: dict,
    ) -> int:
        """Rounds of the windows read out whole, unbroken from round 1.

        It is the final prefix an instant decoder on instant links would
        hold: a window is decoded only once its last round is measured
        (Skoric et al. 2209.08552 lines 416-418), so the rounds read out
        past it, a trailing buffer and the next window's rounds so far,
        wait on the window shape and on no decoder.
        """
        is_whole = functools.partial(
            _is_read_out_whole, self.tracker, rounds_read_out_by_stream
        )
        cursor = self._cursor(_READ_OUT, stream_id)
        return self._joined_prefix(cursor, stream_id, is_whole)

    def final_prefix_round_count(
        self,
        stream_id: Any,  # an opaque identity
    ) -> int:
        """Rounds whose final correction is in place, unbroken from round 1.

        A window that escalated holds only its provisional weak commit
        until the strong answer lands, so its rounds are not final yet;
        a window it absorbed lies after it and waits with it.
        """
        cursor = self._cursor(_FINAL, stream_id)
        return self._joined_prefix(cursor, stream_id, is_final_window)

    def final_window_count(
        self,
        stream_id: Any,  # an opaque identity
    ) -> int:
        """How many of the stream's windows, in order, are final."""
        cursor = self._cursor(_EVERY_FINAL, stream_id)
        self._passed_in_order(cursor, stream_id)
        return cursor.passed

    # ---- a stream lets its final windows go

    def let_go_before(
        self,
        stream_id: Any,  # an opaque identity
        window_index: int,
    ) -> None:
        """Every cursor past the windows before the index; the ledger folded.

        The windows are final (window_release.py), so each cursor's test
        holds of them for good: committed, read out whole, final. The
        cursors stand past them, joined to the last one's commit end, as
        a walk over them would leave them, and the ledger folds the
        rounds they commit.
        """
        last_key = (stream_id, window_index - 1)
        last_window = self.planner.window_at(last_key)
        for kind in _CURSOR_TESTS:
            cursor = self._cursor(kind, stream_id)
            _move_past(cursor, window_index, last_window.commit_hi)
        following = self.planner.window_at((stream_id, window_index))
        cut_rounds = self.planner.cut_rounds_of(stream_id)
        self.ledger.fold_before(stream_id, following.commit_lo, cut_rounds)

    # ---- private

    def _update_committed_round_count(self, stream_id) -> None:
        committed = self.committed_prefix_round_count(stream_id)
        by_stream = self.deliveries.committed_round_count_by_stream
        cached = by_stream.get(stream_id, 0)
        if committed <= cached:
            return
        by_stream[stream_id] = committed
        self.release_committed_segments(stream_id)

    def _deliver_result(self, operation: program_records.Operation) -> None:
        commit_lo, commit_hi = self.planner.commit_extent_of(operation.id)
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

    def _is_final(self, stream_id) -> bool:
        """Every window committed and final, and the stream sealed."""
        if not self._are_windows_final(stream_id):
            return False
        return self.tracker.is_sealed(stream_id)

    def _record_result(self, operation_id, logical_observables) -> None:
        self.trace.operation_result_delivered.fire(
            operation_id, logical_observables
        )

    def _release_segments_if_committed(
        self, operation: program_records.Operation
    ) -> None:
        """Deliver once every segment the release rule awaits is final.

        Its own segments always, and under SWIPER's rule those of its
        decoded ancestors too (release_rules.py). Only a needed outcome is
        delivered: one an operation waits on, or one the QPU needs back.
        """
        is_awaited = operation.id in self.tracker.blocking_operation_ids
        is_needed = is_awaited or operation.requires_result_return_to_qpu
        if not is_needed:
            return
        if operation.id in self.deliveries.segment_results_sent:
            return
        awaited_ids = self.release_rule.awaited_by(operation.id)
        for awaited_id in awaited_ids:
            if not self._are_segments_final(awaited_id):
                return
        segments = self.bindings.segments_by_operation[operation.id]
        logical_observables = self._segments_observables(operation, segments)
        self._record_result(operation.id, logical_observables)
        self.deliveries.segment_results_sent.add(operation.id)
        self.conditional_release.release_waiters(operation)

    def _release_if_committed(self, operation_ids: tuple) -> None:
        for operation_id in operation_ids:
            operation = self.tracker.operation_by_id[operation_id]
            self._release_segments_if_committed(operation)

    def _are_segments_final(self, operation_id) -> bool:
        """Whether the operation has begun and every segment of it is final."""
        segments = self.bindings.segments_by_operation.get(operation_id)
        if segments is None:
            return False
        operation = self.tracker.operation_by_id[operation_id]
        for segment in segments:
            if not self._is_segment_final(operation, segment):
                return False
        return True

    def _segments_observables(self, operation, segments: list):
        """The one stream's committed observables from round 1 to its end.

        A measured outcome depends on every error since the patch was
        prepared, so its frame is the XOR of every committed correction
        before it, not of the segment's own rounds (Bombín et al.
        2303.04846 Eq. 4). None for several streams, as patch streams
        run timing only (build/plan.py refuses a source with values).
        """
        if len(segments) > 1:
            return None
        (segment,) = segments
        last_round = self._stream_segment_end(operation, segment)
        return self.ledger.observables_for_interval(
            segment.stream_id,
            1,
            last_round,
            boundary_policy="stream_segment",
        )

    def _stream_segment_end(
        self, operation: program_records.Operation, segment: "_Segment"
    ) -> int:
        if segment.required_stream_end is not None:
            return segment.required_stream_end
        round_count = self.planner.round_count_of(operation.id)
        return segment.stream_offset + round_count

    def _is_segment_final(
        self, operation: program_records.Operation, segment: "_Segment"
    ) -> bool:
        """Its end is committed, and no strong redo may change it."""
        committed_by_stream = self.deliveries.committed_round_count_by_stream
        committed = committed_by_stream.get(segment.stream_id, 0)
        segment_end = self._stream_segment_end(operation, segment)
        if segment_end > committed:
            return False
        is_waiting = self._segment_waits_for_strong(
            segment.stream_id, segment_end
        )
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

    def _are_windows_final(self, stream_id) -> bool:
        """Whether every window of the stream is committed and final."""
        cursor = self._cursor(_EVERY_FINAL, stream_id)
        self._passed_in_order(cursor, stream_id)
        window_count = self.planner.window_count_of(stream_id)
        return cursor.passed == window_count

    def _is_every_window_final(self) -> bool:
        """Whether every window of every stream is committed and final."""
        for stream_id in self.planner.windowed_stream_ids():
            if not self._are_windows_final(stream_id):
                return False
        return True

    def _open_windows(self, stream_id) -> list:
        """The stream's windows from the first one not final on."""
        cursor = self._cursor(_EVERY_FINAL, stream_id)
        self._passed_in_order(cursor, stream_id)
        indices = self.planner.window_indices_of(stream_id)
        windows = []
        for position in range(cursor.passed, len(indices)):
            window_index = indices[position]
            window = self.planner.window_at((stream_id, window_index))
            windows.append(window)
        return windows

    def _cursor(self, kind: str, stream_id) -> "_Cursor":
        cursors = self.deliveries.cursors
        key = (kind, stream_id)
        cursor = cursors.get(key)
        if cursor is None:
            cursor = _Cursor(_CURSOR_TESTS[kind])
            cursors[key] = cursor
        return cursor

    def _joined_prefix(self, cursor: "_Cursor", stream_id, passes) -> int:
        """The rounds the passing windows join from round 1, in window order.

        A window joins while it passes and starts no later than the round
        after the prefix, _unbroken_prefix_round_count's rule.
        """
        indices = self.planner.window_indices_of(stream_id)
        while cursor.passed < len(indices):
            window_index = indices[cursor.passed]
            window = self.planner.window_at((stream_id, window_index))
            if window.commit_lo > cursor.prefix_end + 1:
                break
            if not passes(window):
                break
            cursor.prefix_end = max(cursor.prefix_end, window.commit_hi)
            cursor.passed += 1
        return cursor.prefix_end

    def _passed_in_order(self, cursor: "_Cursor", stream_id) -> None:
        """The cursor past every window, in order, its test holds of."""
        indices = self.planner.window_indices_of(stream_id)
        while cursor.passed < len(indices):
            window_index = indices[cursor.passed]
            window = self.planner.window_at((stream_id, window_index))
            if not cursor.passes(window):
                return
            cursor.passed += 1

    def _close_strong_store_stream(self, stream_id) -> None:
        """Retire the stream in every store a region lands in."""
        for store in self.retention.landing_stores():
            if store.has_stream(stream_id):
                store.close_stream(stream_id)


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
    tracker, rounds_read_out_by_stream: dict, window
) -> bool:
    return tracker.is_read_out_whole(window, rounds_read_out_by_stream)


def _move_past(cursor: "_Cursor", window_count: int, prefix_end: int) -> None:
    """The cursor past the first windows, joined through their last round."""
    if cursor.passed >= window_count:
        return
    cursor.passed = window_count
    cursor.prefix_end = max(cursor.prefix_end, prefix_end)


class _Deliveries:
    """The progress of the deliveries so far."""

    def __init__(self) -> None:
        self.is_workload_done = False
        self.finished_operation_ids: set = set()
        self.segment_results_sent: set = set()
        self.committed_round_count_by_stream: dict = {}
        # (test, stream id) -> _Cursor
        self.cursors: dict = {}


class _Bindings:
    """Where each segment's rounds fold into its streams."""

    def __init__(self) -> None:
        # operation id -> [_Segment], one per stream it folds into
        self.segments_by_operation: dict = {}
        # stream id -> [the operations bound to it, in bind order]
        self.operation_ids_by_stream: dict = {}


class _Cursor:
    """How many of a stream's windows, in order, pass one test.

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

    def __init__(self, stream_id, stream_offset: int) -> None:
        self.stream_id = stream_id
        self.stream_offset = stream_offset
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
