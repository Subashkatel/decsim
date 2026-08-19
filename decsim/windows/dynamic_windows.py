"""Runtime window layout for streams whose length is not known up front.

The runtime twin of static root materialization: the root lays out all windows
up front for ops of known length; this lays them out on the fly for a
dynamic stream, whose total length only emerges as the run proceeds (op
segments fold their rounds into it via Operation.stream_id). Windows are
created as rounds arrive (grow), the tail window is clipped once the
stream's true round count is known (seal), feedback boundaries are closed,
and the committed-prefix accounting advances so upstream segments can be
released.

Holds a back-reference to WindowManager rather than owning windows itself:
window creation, read-ref bookkeeping, and readiness checks stay there so
both sides mutate the window manager's one authoritative window mapping.
"""

from __future__ import annotations

from typing import Optional

from ..message import Operation, Window


class DynamicWindows:
    """Owns per-stream state; the window_manager owns windows and payloads."""

    def __init__(self, window_manager) -> None:
        self.window_manager = window_manager
        self._streams: dict = {}
        self.closed_boundaries: dict = {}
        self.committed_round_counts: dict = {}

    # ------------------------------------------------------------ register

    def register(
        self,
        stream_op: Operation,
        *,
        commit_round_count,
        buffer_round_count,
        source_round_limit,
        finite_geometries=None,
    ) -> None:
        """Track a stream whose windows are created from arriving rounds."""
        self._streams[stream_op.id] = {
            "commit_rounds": commit_round_count,
            "buffer_rounds": buffer_round_count,
            "next_window": 0,
            "sealed": False,
            "source_round_limit": source_round_limit,
            "sealed_round_count": None,
            "finite_geometries": finite_geometries,
        }
        self.closed_boundaries.setdefault(stream_op.id, set())

    def has(self, stream_id) -> bool:
        return stream_id in self._streams

    def has_unsealed_streams(self) -> bool:
        return any(not stream_state["sealed"]
                   for stream_state in self._streams.values())

    def sealed(self, op_id) -> bool:
        """True for non-streams and sealed streams."""
        stream_state = self._streams.get(op_id)
        return stream_state is None or stream_state["sealed"]

    def arrival_round_limit(
        self,
        op_id,
        fallback_rounds: int,
    ) -> Optional[int]:
        """Maximum legal device round, or None for an open unbounded stream."""
        stream_state = self._streams.get(op_id)
        if stream_state is None:
            return fallback_rounds
        if stream_state["sealed"]:
            return stream_state["sealed_round_count"]
        return stream_state["source_round_limit"]

    # ---------------------------------------------------------------- grow

    def grow(
        self,
        stream_id,
        rounds_to_plan: Optional[int] = None,
        sealed_round_cap: Optional[int] = None,
    ) -> None:
        """Create every window whose commit region has begun."""
        wm = self.window_manager
        stream_state = self._streams[stream_id]
        if stream_state["sealed"]:
            return
        commit_rounds = stream_state["commit_rounds"]
        buffer_rounds = stream_state["buffer_rounds"]
        highest_known_round = wm.rounds_arrived[stream_id] \
            if rounds_to_plan is None else rounds_to_plan
        finite_geometries = stream_state["finite_geometries"]
        if finite_geometries is not None:
            while stream_state["next_window"] < len(finite_geometries):
                window_index = stream_state["next_window"]
                geometry = finite_geometries[window_index]
                if geometry.commit_lo > highest_known_round:
                    break
                wm.create_dynamic_window(
                    stream_id,
                    window_index,
                    geometry.commit_lo,
                    geometry.commit_hi,
                    geometry.buffer_hi,
                    is_last=(window_index == len(finite_geometries) - 1),
                )
                stream_state["next_window"] += 1
            return

        while stream_state["next_window"] * commit_rounds + 1 <= highest_known_round:
            window_index = stream_state["next_window"]
            commit_lo = window_index * commit_rounds + 1
            commit_hi = self._commit_hi(
                stream_state, window_index, sealed_round_cap)
            buffer_hi = commit_hi + buffer_rounds
            wm.create_dynamic_window(stream_id, window_index, commit_lo,
                                     commit_hi, buffer_hi, is_last=False)
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

    def maybe_update(self, op_id) -> None:
        """Grow (and maybe seal) the stream as one more round arrives."""
        if op_id not in self._streams:
            return
        self.grow(op_id)
        stream_state = self._streams[op_id]
        if (not stream_state["sealed"]
                and stream_state["source_round_limit"] is not None
                and self.window_manager.rounds_arrived[op_id]
                >= stream_state["source_round_limit"]):
            self.seal(op_id, stream_state["source_round_limit"])

    # ---------------------------------------------------------------- seal

    def seal(self, stream_id, stream_round_count: int) -> None:
        """Close a dynamic stream once its full length has arrived."""
        wm = self.window_manager
        stream_state = self._streams[stream_id]
        if stream_state["sealed"]:
            return
        wm.validate_stream_length(stream_id, stream_round_count)
        self.grow(
            stream_id,
            rounds_to_plan=stream_round_count,
            sealed_round_cap=stream_round_count,
        )
        if stream_state["finite_geometries"] is None:
            self._trim_tail(stream_id, stream_round_count)
        stream_state["sealed_round_count"] = stream_round_count
        stream_state["sealed"] = True
        wm.check_windows_for_operation(stream_id)
        wm.finish_workload_if_ready()

    def _trim_tail(self, stream_id, stream_round_count: int) -> None:
        """Clip the final open-stream commit region to the sealed length."""
        stream_state = self._streams[stream_id]
        self.window_manager.trim_dynamic_window_tail(
            stream_id, stream_round_count, stream_state["buffer_rounds"])

    # ---------------------------------------------------- feedback boundary

    def close_boundary(self, stream_id, stream_round_count: int) -> None:
        """Mark a live stream round as a measurement-closed feedback boundary."""
        wm = self.window_manager
        if stream_id not in self._streams:
            return
        if stream_round_count < 1:
            raise ValueError(
                f"stream_round_count must be >= 1 (got {stream_round_count})")
        self._reject_unsupported_boundary(stream_id, stream_round_count)
        closed_rounds = self.closed_boundaries[stream_id]
        if stream_round_count in closed_rounds:
            raise RuntimeError(
                f"stream {stream_id!r} already closed a feedback boundary at "
                f"round {stream_round_count}")
        closed_rounds.add(stream_round_count)
        self.grow(stream_id, rounds_to_plan=stream_round_count)
        wm.refresh_unqueued_stream_windows(stream_id)
        wm.check_windows_for_operation(stream_id)

    def _reject_unsupported_boundary(self, stream_id,
                                     stream_round_count: int) -> None:
        """Reject internal closed boundaries in finite real-syndrome streams."""
        stream_state = self._streams[stream_id]
        source_round_limit = stream_state["source_round_limit"]
        if source_round_limit is None:
            return
        if stream_round_count >= source_round_limit:
            return
        raise RuntimeError(
            "measurement_closed live-stream boundaries inside a finite real-syndrome "
            "stream need a source circuit with that destructive boundary. A continuous "
            f"stream model was registered for {source_round_limit} rounds, but the "
            f"feedback boundary closes at round {stream_round_count}. Use timing-only "
            "payloads for this timing study, split the workload into finite operation "
            "circuits, or provide a boundary-aware syndrome source.")

    def closed_boundary_for_window(self, window: Window) -> Optional[int]:
        """Return a closed live-stream boundary covered by this window."""
        boundaries = self.closed_boundaries.get(window.op_id, ())
        covered = [b for b in boundaries
                   if window.commit_hi <= b < window.buffer_hi]
        return min(covered) if covered else None

    # ---------------------------------------------------- committed prefix

    def round_count_for_window(self, op_id, window: Optional[Window],
                               fallback_rounds: int) -> int:
        """Round count to use when checking/reading one window.
        For non-streams that is just fallback_rounds (the op's planned rounds)."""
        stream_state = self._streams.get(op_id)
        if stream_state is None:
            return fallback_rounds
        if stream_state["sealed"]:
            return stream_state["sealed_round_count"]
        if stream_state["source_round_limit"] is not None:
            return stream_state["source_round_limit"]
        if window is not None:
            return window.buffer_hi
        return self.window_manager.rounds_arrived.get(op_id, 0)

    def committed_round_count(self, stream_id) -> int:
        """Committed-prefix round count cached for a stream, or 0."""
        return self.committed_round_counts.get(stream_id, 0)

    def update_committed_round_count(self, stream_id) -> None:
        """Advance the committed prefix and release the segments it covers."""
        committed = self._committed_prefix_round_count(stream_id)
        if committed <= self.committed_round_count(stream_id):
            return
        self.committed_round_counts[stream_id] = committed
        self.window_manager.release_stream_segments_at_commit(stream_id, committed)

    def recompute_committed_round_count(self, stream_id) -> int:
        """Reset a speculative stream prefix from its actual committed windows."""
        committed = self._committed_prefix_round_count(stream_id)
        if committed:
            self.committed_round_counts[stream_id] = committed
        else:
            self.committed_round_counts.pop(stream_id, None)
        return committed

    def _committed_prefix_round_count(self, stream_id) -> int:
        """How many initial rounds are covered by committed windows."""
        wm = self.window_manager
        committed_ranges = sorted(
            (wm.windows[key].commit_lo, wm.windows[key].commit_hi)
            for key in wm.committed_windows if key[0] == stream_id)
        committed_round_count = 0
        for start_round, end_round in committed_ranges:
            if start_round > committed_round_count + 1:
                break
            committed_round_count = max(committed_round_count, end_round)
        return committed_round_count
