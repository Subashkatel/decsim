"""The operation results: delivered once every covering window is final.

The result is the ledger's XOR over the operation's commit range, or a
stream segment's, delivered to the conditional release once every
window is committed, no strong redo is pending, and the stream is
sealed; a segment releases when the stream's committed prefix covers
it (Skoric et al. 2209.08552: the logical correction of a stream is the
sum over its committed windows). The workload is complete once every
window is final.
"""

from typing import Callable, Optional

import decsim.message as message


class OperationResults:
    """Delivers each operation's and segment's result once it is final."""

    def __init__(
        self,
        planner,
        tracker,
        retention,
        ledger,
        conditional_release,
        on_workload_complete: Optional[Callable[[], None]],
    ) -> None:
        self.planner = planner
        self.tracker = tracker
        self.retention = retention
        self.ledger = ledger
        self.conditional_release = conditional_release
        self.deliveries = _Deliveries(on_workload_complete)

    @property
    def result_by_operation(self) -> dict:
        """Every delivered result by operation id."""
        return self.deliveries.result_by_operation

    # ---- what the committer tells

    def install_window_contribution(
        self, window: message.Window, logical_observables
    ) -> None:
        """The window owns its commit range, unless a strong window does."""
        existing = self.ledger.get(window.key)
        if existing is not None and existing.ownership_kind == "strong_window":
            return
        contribution = message.LogicalContribution(
            owner_key=window.key,
            commit_lo=window.commit_lo,
            commit_hi=window.commit_hi,
            ownership_kind="ordinary_window",
            logical_observables=logical_observables,
        )
        self.ledger.install(contribution)

    def replace_prediction(self, key: tuple, logical_observables) -> None:
        """A strong result replaces the owner's prediction."""
        self.ledger.replace_prediction(key, logical_observables)

    def note_window_committed(
        self, window: message.Window, is_final: bool
    ) -> None:
        """A window committed: advance the stream's prefix; free its context.

        No earlier escalation can re-slice its dependents now, so their
        potential restart reads end.
        """
        self._update_committed_round_count(window.op_id)
        for dependent_key in window.dependents:
            self.retention.release_restart_reads(dependent_key)
        if is_final and self.retention.strong_store is not None:
            potential = message.PotentialStrong(window.key)
            self.retention.release_hold_if_live(
                potential, self.retention.strong_store
            )

    # ---- delivery

    def deliver_if_final(self, operation: message.Operation) -> None:
        """Deliver the result once every window is final and the stream sealed.

        Every window committed, no strong redo pending, no store still
        referring to the operation, the stream sealed.
        """
        if operation.id in self.deliveries.finished_operation_ids:
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
        self.deliveries.finished_operation_ids.add(operation.id)
        self._deliver_result(operation)
        weak_store.close_operation(operation.id)
        self._close_strong_store_operation(operation.id)
        self.finish_workload_if_ready()

    def finish_workload_if_ready(self) -> None:
        """Tell the root once every window of the workload is final."""
        if self.deliveries.workload_done.is_done:
            return
        if self._committed_window_count() != self.planner.total_windows:
            return
        if self._has_window_awaiting_strong(None):
            return
        if self.tracker.has_unsealed_streams():
            return
        self.deliveries.workload_done.run()

    def release_committed_segments(self, stream_id) -> None:
        """Deliver the segments the stream's committed prefix covers."""
        committed = self.deliveries.committed_round_count_by_stream.get(
            stream_id, 0
        )
        self.release_stream_segments_at_commit(stream_id, committed)

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

    # ---- the segment binds

    def bind_stream_segment(
        self, operation_id: int, stream_id, stream_offset: int
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

    # ---- the rounds backlog

    def committed_prefix_round_count(self, operation_id) -> int:
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
            self.deliveries.result_by_operation.pop(operation_id, None)
            return
        self.deliveries.result_by_operation[operation_id] = logical_observables

    def _release_segment_if_committed(
        self,
        operation: message.Operation,
        stream_id,
        committed_round_count: int,
    ) -> None:
        segment = self.deliveries.segment_by_operation.get(operation.id)
        operation_stream_id = operation.stream_id
        stream_offset = operation.stream_offset
        if segment is not None and segment.stream_id is not None:
            operation_stream_id = segment.stream_id
            stream_offset = segment.stream_offset
        if operation_stream_id != stream_id:
            return
        if operation.id not in self.tracker.blocking_operation_ids:
            return
        if operation.id in self.deliveries.segment_results_sent:
            return
        segment_end = self._stream_segment_end(
            operation, segment, stream_offset
        )
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
        self.deliveries.segment_results_sent.add(operation.id)
        self.conditional_release.release_waiters(operation)

    def _stream_segment_end(
        self, operation: message.Operation, segment, stream_offset
    ) -> Optional[int]:
        if segment is not None and segment.required_stream_end is not None:
            return segment.required_stream_end
        if stream_offset is None:
            return None
        round_count = self.planner.round_count_of(operation.id)
        return stream_offset + round_count

    def _segment_waits_for_strong(self, stream_id, segment_end: int) -> bool:
        for key, window in self.planner.windows_by_key.items():
            if key[0] != stream_id:
                continue
            if not is_awaiting_strong(window):
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
            if is_awaiting_strong(window):
                return True
        return False

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


def is_awaiting_strong(window: message.Window) -> bool:
    """Committed provisionally: the strong redo has not published yet."""
    if not window.committed:
        return False
    if window.is_absorbed:
        return False
    return window.published_request_key is None


class _Deliveries:
    """What has been delivered so far, and the one workload callback."""

    def __init__(self, on_workload_complete) -> None:
        self.workload_done = _Once(on_workload_complete)
        self.result_by_operation: dict = {}
        self.finished_operation_ids: set = set()
        self.segment_results_sent: set = set()
        self.committed_round_count_by_stream: dict = {}
        self.segment_by_operation: dict = {}


class _Once:
    """A callback that runs at most once; None runs nothing."""

    def __init__(self, callback) -> None:
        self.callback = callback
        self.is_done = False

    def run(self) -> None:
        if self.callback is None:
            return
        self.is_done = True
        self.callback()


class _Segment:
    """Where one operation's rounds fold into a stream, and its result's end."""

    def __init__(self) -> None:
        self.stream_id = None
        self.stream_offset = None
        self.required_stream_end = None
