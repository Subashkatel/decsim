"""The sliding row's windows on a stream that grows as its rounds arrive.

An open stream is laid by the region law (windows/region_streams.py):
each window commits at most F rounds after the last one's and reads B
more, the lookahead law, since the end it would drain toward is not
known yet. A stream whose source declares its length is laid by the
row's finite construction (sliding.py plan_operation) instead, so its
last window drains as the terminal policy says.

A segment's result is the sum of the windows committed over its rounds
(Skoric et al. 2209.08552; windows/operation_results.py), so every
segment edge cuts a commit region: a window laid across one commits
through it, and a finite source's windows after it are planned again,
stretch by stretch.
"""

import dataclasses
import functools
from collections.abc import Callable
from typing import Any, Optional

import decsim.ports as ports
import decsim.records.windows as window_records
import decsim.windows.region_streams as region_streams


class SlidingStreamLayouts:
    """The sliding row's stream layouts: finite when the length is known."""

    def __init__(self, scheme: ports.WindowingScheme) -> None:
        self.scheme = scheme
        buffer_rule = region_streams.FixedBuffer()
        self.open_streams = region_streams.RegionStreamLayouts(
            buffer_rule, is_cut_at_every_segment=True
        )

    def open_stream(
        self,
        stream_id: Any,  # an opaque identity
        round_limit: Optional[int],
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ) -> ports.StreamLayout:
        """The stream's layout; round_limit is its length, if known."""
        if round_limit is None:
            return self.open_streams.open_stream(
                stream_id,
                None,
                commit_round_count=commit_round_count,
                buffer_round_count=buffer_round_count,
            )
        plan = self.scheme.plan_operation(
            stream_id,
            round_limit,
            commit_round_count=commit_round_count,
            buffer_round_count=buffer_round_count,
        )
        plan_rounds = functools.partial(
            self.scheme.plan_operation,
            stream_id,
            commit_round_count=commit_round_count,
            buffer_round_count=buffer_round_count,
        )
        return FiniteStream(plan_rounds, plan.windows, buffer_round_count)


class FiniteStream:
    """The finite construction's windows, planned again after each cut."""

    def __init__(
        self,
        plan_rounds: Callable,
        geometries: tuple,
        buffer_rounds: int,
    ) -> None:
        # the scheme's plan of the stream's next round count, as its own
        self.plan_rounds = plan_rounds
        self.geometries = geometries
        self.buffer_rounds = buffer_rounds
        self.next_window_index = 0
        # rounds a segment ends on, in the order they were cut
        self.cuts: list = []

    def begun(self, highest_known_round: int, round_cap: Optional[int]) -> list:
        """The planned geometries not laid yet whose commit has begun."""
        del round_cap
        begun = []
        unlaid = self.geometries[self.next_window_index :]
        for geometry in unlaid:
            if geometry.commit_lo > highest_known_round:
                break
            begun.append(geometry)
        # the planner lays every geometry returned, in order
        self.next_window_index += len(begun)
        return begun

    def cut_after(self, windows: list, last_round: int) -> list:
        """A segment ends on the round: clip across it, plan the rest again."""
        clipped = []
        for window in windows:
            if window.commit_lo <= last_round < window.commit_hi:
                self._clip(window, last_round)
                clipped.append(window)
        self.cuts.append(last_round)
        self._replan_after(last_round)
        return clipped

    def close_tail(
        self, windows: list, last_round: int
    ) -> window_records.TailChange:
        """A finite source's windows are its own; it closes only at its end."""
        del windows, last_round
        return window_records.TailChange()

    def relay(self, windows: list) -> list:
        """A planned window never moves once laid but by a cut."""
        del windows
        return []

    def cut_rounds(self) -> tuple:
        """The rounds a segment of the stream ends on, in the order cut."""
        return tuple(self.cuts)

    def reads_of(self, window: window_records.Window) -> tuple:
        """A sliding window reads its own stream only."""
        del window
        return ()

    def readers_of(self, window: window_records.Window) -> tuple:
        """A sliding window reads its own stream only."""
        del window
        return ()

    def readers_at(self, round_index: int) -> tuple:
        """A sliding window reads its own stream only."""
        del round_index
        return ()

    def let_go_before(self, window_index: int) -> None:
        """A declared stream's plan is laid whole from its start.

        Its geometries end with the stream, whose length the source
        declared, and no window reads across a coupling, so nothing
        here outlives a window for want of being let go.
        """
        del window_index

    def _clip(self, window: window_records.Window, last_round: int) -> None:
        """The window commits through the round and reads its buffer past it."""
        window.commit_hi = last_round
        window.buffer_hi = last_round + self.buffer_rounds
        rounds_after_start = window.buffer_hi - window.start_round
        window.round_count = rounds_after_start + 1

    def _replan_after(self, last_round: int) -> None:
        """Lay the source's windows not laid yet from the cut on.

        The window across the cut commits through it, and the scheme
        plans each stretch between the cuts after it as its own. A cut on
        a window edge, or on the source's last round, leaves the plan as
        it is.
        """
        round_limit = self.geometries[-1].commit_hi
        following_index = _first_past(self, last_round)
        unlaid_after = self.geometries[following_index:]
        if _is_plan_edge(unlaid_after, last_round, round_limit):
            return
        kept = list(self.geometries[:following_index])
        straddler = _clipped_straddler(self, unlaid_after, last_round)
        kept.extend(straddler)
        stretch_start = last_round
        stretch_ends = _stretch_ends(self.cuts, last_round, round_limit)
        for stretch_end in stretch_ends:
            stretch = self._plan_stretch(stretch_start, stretch_end)
            kept.extend(stretch)
            stretch_start = stretch_end
        self.geometries = tuple(kept)

    def _plan_stretch(self, round_before: int, last_round: int) -> list:
        """The scheme's windows over the rounds after one through another."""
        round_count = last_round - round_before
        plan = self.plan_rounds(round_count)
        return [_shifted(geometry, round_before) for geometry in plan.windows]


def _first_past(stream: FiniteStream, last_round: int) -> int:
    """The first geometry not laid yet that commits past the round.

    It is the geometry count when every one after the round is laid.
    """
    geometries = stream.geometries
    for index in range(stream.next_window_index, len(geometries)):
        if geometries[index].commit_hi > last_round:
            return index
    return len(geometries)


def _stretch_ends(cut_rounds: list, last_round: int, round_limit: int) -> list:
    """The last round of each stretch after the round: the cuts, the end."""
    later_cuts = {cut for cut in cut_rounds if last_round < cut < round_limit}
    stretch_ends = sorted(later_cuts)
    stretch_ends.append(round_limit)
    return stretch_ends


def _is_plan_edge(
    unlaid_after: tuple, last_round: int, round_limit: int
) -> bool:
    """Whether a window of the plan already ends on the round.

    The source's last round ends one, and so does a round the next
    geometry not laid yet starts after. A window already laid across
    the round was clipped to it, so the rounds after it need a plan.
    """
    if last_round >= round_limit:
        return True
    if not unlaid_after:
        return False
    first_round_after = last_round + 1
    return unlaid_after[0].commit_lo == first_round_after


def _clipped_straddler(
    stream: FiniteStream, unlaid_after: tuple, last_round: int
) -> list:
    """The geometry not laid yet across the round, clipped to it, if any."""
    if not unlaid_after:
        return []
    following = unlaid_after[0]
    if following.commit_lo > last_round:
        return []
    buffer_hi = last_round + stream.buffer_rounds
    clipped = dataclasses.replace(
        following, commit_hi=last_round, buffer_hi=buffer_hi
    )
    return [clipped]


def _shifted(
    geometry: window_records.WindowGeometry, round_shift: int
) -> window_records.WindowGeometry:
    """The geometry moved later by the rounds."""
    buffer_lo = geometry.buffer_lo + round_shift
    commit_lo = geometry.commit_lo + round_shift
    commit_hi = geometry.commit_hi + round_shift
    buffer_hi = geometry.buffer_hi + round_shift
    return dataclasses.replace(
        geometry,
        buffer_lo=buffer_lo,
        commit_lo=commit_lo,
        commit_hi=commit_hi,
        buffer_hi=buffer_hi,
    )
