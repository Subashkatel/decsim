"""Runtime window layout for streams whose length is not known up front.

The runtime twin of the static plan: the root lays out every window up
front for an operation of known length; this lays them out as a dynamic
stream's rounds arrive (operation segments fold their rounds into it by
Operation.stream_id). Windows are created as rounds arrive (grow), the
tail window is clipped once the stream's true round count is known
(seal), feedback boundaries are closed, and the committed prefix
advances so upstream segments can be released.

Holds a back-reference to the WindowManager rather than owning windows:
window creation, read-ref bookkeeping and readiness checks stay there,
so both sides mutate the manager's one window table. The slice 5
structural commits dissolve this class into the planner and the tracker
(design note, section 3).
"""

from typing import Optional

import decsim.message as message


class DynamicWindows:
    """Per-stream growth state; the window manager owns the windows."""

    def __init__(self, window_manager) -> None:
        self.window_manager = window_manager
        self._streams: dict = {}
        self.closed_boundaries: dict = {}
        self.committed_round_counts: dict = {}

    # ---- register

    def register(
        self,
        stream_operation: message.Operation,
        *,
        commit_round_count,
        buffer_round_count,
        source_round_limit,
        finite_geometries=None,
    ) -> None:
        """Track a stream whose windows are created from arriving rounds."""
        self._streams[stream_operation.id] = {
            "commit_rounds": commit_round_count,
            "buffer_rounds": buffer_round_count,
            "next_window": 0,
            "sealed": False,
            "source_round_limit": source_round_limit,
            "sealed_round_count": None,
            "finite_geometries": finite_geometries,
        }
        self.closed_boundaries.setdefault(stream_operation.id, set())

    def has(self, stream_id) -> bool:
        """True for a registered dynamic stream."""
        return stream_id in self._streams

    def has_unsealed_streams(self) -> bool:
        """True while any registered stream has not sealed."""
        for stream_state in self._streams.values():
            if not stream_state["sealed"]:
                return True
        return False

    def sealed(self, operation_id) -> bool:
        """True for non-streams and sealed streams."""
        stream_state = self._streams.get(operation_id)
        if stream_state is None:
            return True
        return stream_state["sealed"]

    def arrival_round_limit(
        self, operation_id, fallback_rounds: int
    ) -> Optional[int]:
        """Maximum legal device round, or None for an open unbounded stream."""
        stream_state = self._streams.get(operation_id)
        if stream_state is None:
            return fallback_rounds
        if stream_state["sealed"]:
            return stream_state["sealed_round_count"]
        return stream_state["source_round_limit"]

    # ---- grow

    def grow(
        self,
        stream_id,
        rounds_to_plan: Optional[int] = None,
        sealed_round_cap: Optional[int] = None,
    ) -> None:
        """Create every window whose commit region has begun."""
        stream_state = self._streams[stream_id]
        if stream_state["sealed"]:
            return
        highest_known_round = rounds_to_plan
        if highest_known_round is None:
            arrived = self.window_manager.rounds_arrived_by_operation
            highest_known_round = arrived[stream_id]
        finite_geometries = stream_state["finite_geometries"]
        if finite_geometries is not None:
            self._grow_finite(
                stream_id, stream_state, finite_geometries, highest_known_round
            )
            return
        self._grow_arithmetic(
            stream_id, stream_state, highest_known_round, sealed_round_cap
        )

    def _grow_finite(
        self,
        stream_id,
        stream_state: dict,
        finite_geometries,
        highest_known_round: int,
    ) -> None:
        """The source's finite plan: one window per geometry, in order."""
        geometry_count = len(finite_geometries)
        while stream_state["next_window"] < geometry_count:
            window_index = stream_state["next_window"]
            geometry = finite_geometries[window_index]
            if geometry.commit_lo > highest_known_round:
                return
            self.window_manager.create_dynamic_window(
                stream_id,
                window_index,
                geometry.commit_lo,
                geometry.commit_hi,
                geometry.buffer_hi,
            )
            stream_state["next_window"] += 1

    def _grow_arithmetic(
        self,
        stream_id,
        stream_state: dict,
        highest_known_round: int,
        sealed_round_cap: Optional[int],
    ) -> None:
        """An open stream: window k commits [k*ncom+1, (k+1)*ncom]."""
        commit_rounds = stream_state["commit_rounds"]
        buffer_rounds = stream_state["buffer_rounds"]
        while True:
            window_index = stream_state["next_window"]
            commit_lo = window_index * commit_rounds + 1
            if commit_lo > highest_known_round:
                return
            commit_hi = self._commit_hi(
                stream_state, window_index, sealed_round_cap
            )
            buffer_hi = commit_hi + buffer_rounds
            self.window_manager.create_dynamic_window(
                stream_id, window_index, commit_lo, commit_hi, buffer_hi
            )
            stream_state["next_window"] += 1

    @staticmethod
    def _commit_hi(
        stream_state: dict,
        window_index: int,
        sealed_round_cap: Optional[int] = None,
    ) -> int:
        """Commit end for one dynamic window, clipped when a cap is known."""
        commit_rounds = stream_state["commit_rounds"]
        commit_hi = (window_index + 1) * commit_rounds
        known_round_count = sealed_round_cap
        if known_round_count is None:
            known_round_count = stream_state["sealed_round_count"]
        if known_round_count is None:
            known_round_count = stream_state["source_round_limit"]
        if known_round_count is None:
            return commit_hi
        return min(commit_hi, known_round_count)

    def maybe_update(self, operation_id) -> None:
        """Grow (and maybe seal) the stream as one more round arrives."""
        if operation_id not in self._streams:
            return
        self.grow(operation_id)
        stream_state = self._streams[operation_id]
        if stream_state["sealed"]:
            return
        source_round_limit = stream_state["source_round_limit"]
        if source_round_limit is None:
            return
        arrived = self.window_manager.rounds_arrived_by_operation[operation_id]
        if arrived >= source_round_limit:
            self.seal(operation_id, source_round_limit)

    # ---- seal

    def seal(self, stream_id, stream_round_count: int) -> None:
        """Close a dynamic stream once its full length has arrived."""
        stream_state = self._streams[stream_id]
        if stream_state["sealed"]:
            return
        self.window_manager.validate_stream_length(
            stream_id, stream_round_count
        )
        self.grow(
            stream_id,
            rounds_to_plan=stream_round_count,
            sealed_round_cap=stream_round_count,
        )
        if stream_state["finite_geometries"] is None:
            self._trim_tail(stream_id, stream_round_count)
        stream_state["sealed_round_count"] = stream_round_count
        stream_state["sealed"] = True
        self.window_manager.check_windows_for_operation(stream_id)
        self.window_manager.finish_workload_if_ready()

    def _trim_tail(self, stream_id, stream_round_count: int) -> None:
        """Clip the final open-stream commit region to the sealed length."""
        stream_state = self._streams[stream_id]
        self.window_manager.trim_dynamic_window_tail(
            stream_id, stream_round_count, stream_state["buffer_rounds"]
        )

    # ---- feedback boundary

    def close_boundary(self, stream_id, stream_round_count: int) -> None:
        """Mark a live stream round as a measurement-closed boundary."""
        if stream_id not in self._streams:
            return
        if stream_round_count < 1:
            raise ValueError(
                f"stream_round_count must be >= 1 (got {stream_round_count})"
            )
        self._reject_unsupported_boundary(stream_id, stream_round_count)
        closed_rounds = self.closed_boundaries[stream_id]
        if stream_round_count in closed_rounds:
            raise RuntimeError(
                f"stream {stream_id!r} already closed a feedback boundary at "
                f"round {stream_round_count}"
            )
        closed_rounds.add(stream_round_count)
        self.grow(stream_id, rounds_to_plan=stream_round_count)
        self.window_manager.refresh_unqueued_stream_windows(stream_id)
        self.window_manager.check_windows_for_operation(stream_id)

    def _reject_unsupported_boundary(
        self, stream_id, stream_round_count: int
    ) -> None:
        """Reject internal closed boundaries in finite real-syndrome streams."""
        stream_state = self._streams[stream_id]
        source_round_limit = stream_state["source_round_limit"]
        if source_round_limit is None:
            return
        if stream_round_count >= source_round_limit:
            return
        raise RuntimeError(
            "measurement_closed live-stream boundaries inside a finite "
            "real-syndrome stream need a source circuit with that destructive "
            "boundary. A continuous stream model was registered for "
            f"{source_round_limit} rounds, but the feedback boundary closes "
            f"at round {stream_round_count}. Use timing-only payloads for "
            "this timing study, split the workload into finite operation "
            "circuits, or provide a boundary-aware syndrome source."
        )

    def closed_boundary_for_window(
        self, window: message.Window
    ) -> Optional[int]:
        """The earliest closed boundary in the window's trailing buffer."""
        boundaries = self.closed_boundaries.get(window.op_id, ())
        covered = []
        for boundary in boundaries:
            if window.commit_hi <= boundary < window.buffer_hi:
                covered.append(boundary)
        if not covered:
            return None
        return min(covered)

    # ---- committed prefix

    def round_count_for_window(
        self,
        operation_id,
        window: Optional[message.Window],
        fallback_rounds: int,
    ) -> int:
        """The round count to check or read one window against.

        A non-stream operation has its planned rounds; a sealed stream its
        sealed length; a capped stream its source limit; an open stream
        the window's own read end, or the rounds arrived so far.
        """
        stream_state = self._streams.get(operation_id)
        if stream_state is None:
            return fallback_rounds
        if stream_state["sealed"]:
            return stream_state["sealed_round_count"]
        if stream_state["source_round_limit"] is not None:
            return stream_state["source_round_limit"]
        if window is not None:
            return window.buffer_hi
        arrived = self.window_manager.rounds_arrived_by_operation
        return arrived.get(operation_id, 0)

    def committed_round_count(self, stream_id) -> int:
        """Committed-prefix round count cached for a stream, or 0."""
        return self.committed_round_counts.get(stream_id, 0)

    def update_committed_round_count(self, stream_id) -> None:
        """Advance the committed prefix and release the segments it covers."""
        committed = self._committed_prefix_round_count(stream_id)
        cached = self.committed_round_count(stream_id)
        if committed <= cached:
            return
        self.committed_round_counts[stream_id] = committed
        self.window_manager.release_stream_segments_at_commit(
            stream_id, committed
        )

    def _committed_prefix_round_count(self, stream_id) -> int:
        """How many initial rounds are covered by committed windows."""
        committed_ranges = []
        for key, window in self.window_manager.windows.items():
            if key[0] != stream_id:
                continue
            if not window.committed:
                continue
            committed_ranges.append((window.commit_lo, window.commit_hi))
        committed_ranges.sort()
        committed_round_count = 0
        for start_round, end_round in committed_ranges:
            if start_round > committed_round_count + 1:
                break
            committed_round_count = max(committed_round_count, end_round)
        return committed_round_count
