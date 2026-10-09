"""The operation results: delivered once every covering window is final.

The result is the ledger's XOR over the operation's commit range, or a
stream segment's, delivered to the conditional release once every
window is committed, no strong redo is pending, and the stream is
sealed; a segment releases when the stream's committed prefix covers
it (Skoric et al. 2209.08552: the logical correction of a stream is the
sum over its committed windows). The workload is complete once every
window is final.
"""

import dataclasses
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
        if is_final and self.retention.strong_store is not None:
            potential = decoding_records.PotentialStrong(window.key)
            self.retention.release_strong_hold_if_live(potential)

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
        if self._committed_window_count() != self.planner.total_windows:
            return
        if self._has_window_awaiting_strong(None):
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
        committed = self._committed_windows_of(operation_id)
        return _unbroken_prefix_round_count(committed)

    def final_prefix_round_count(
        self,
        operation_id: Any,  # an opaque identity
    ) -> int:
        """Rounds whose final correction is in place, unbroken from round 1.

        A window that escalated holds only its provisional weak commit
        until the strong answer lands, so its rounds are not final yet;
        a window it absorbed lies after it and waits with it.
        """
        final = []
        for window in self._committed_windows_of(operation_id):
            if not is_awaiting_strong(window):
                final.append(window)
        return _unbroken_prefix_round_count(final)

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
        if self._has_window_awaiting_strong(operation_id):
            return False
        weak_store = self.retention.weak_store
        if weak_store is not None and weak_store.has_live_operation_reference(
            operation_id
        ):
            return False
        if self._strong_store_references(operation_id):
            return False
        committed = self._committed_windows_of(operation_id)
        if len(committed) != self.planner.window_count_of(operation_id):
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


def is_awaiting_strong(window: window_records.Window) -> bool:
    """Committed provisionally: the strong redo has not published yet."""
    if not window.committed:
        return False
    if window.is_absorbed:
        return False
    return window.published_request_key is None


def _unbroken_prefix_round_count(windows: list) -> int:
    """The last round of the windows' commit ranges joined from round 1."""
    commit_ranges = []
    for window in windows:
        commit_ranges.append((window.commit_lo, window.commit_hi))
    commit_ranges.sort()
    prefix_end = 0
    for start_round, end_round in commit_ranges:
        if start_round > prefix_end + 1:
            break
        prefix_end = max(prefix_end, end_round)
    return prefix_end


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
