"""The parallel row: Skoric block A/B windows, planned or grown.

Skoric et al. 2209.08552 section I.C: the stream is cut into A blocks
that can be decoded at the same time and B blocks that reconcile the
seams between them, so the depth is two rather than the length of the
chain. The published construction fixes ncom = nbuf = d. An A block
reads d rounds on each side of its commit and waits for no window; a B
block commits the 3d rounds between two A commits, reads nothing more,
and starts once both A blocks have committed (lines 388-405, 421-426).

A growing stream lays the same blocks as its rounds arrive, each once
its commit begins, as Skoric's online decoder hands each block its
rounds as they are collected (lines 1042-1055). The stream's end clips
the laid blocks as the plan's tail does, with one difference: a stream
never takes back a block it laid, so a tail of at most d rounds past an
A commit is a short terminal B where the plan absorbs it into the A.
"""

import dataclasses
from typing import Any, Optional

import decsim.records.windows as window_records
import decsim.refusals as refusals
import decsim.windows.schemes.window_data as window_data


class ParallelWindowScheme:
    """Skoric block A/B windows with dependency-aware seam residuals."""

    # The last block, an absorbed tail or a terminal B, commits to the
    # operation's last round and reads nothing past it.
    has_trailing_tail_context = False
    # The A blocks are decoded at the same time and the B blocks
    # reconcile them, so the commits are a depth-two graph, not a chain.
    commits_in_one_serial_chain = False
    supports_dynamic_streams = True
    reads_across_couplings = False

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The parallel row's window sizes; None is the code distance."""

        commit_rounds: Optional[int] = None
        buffer_rounds: Optional[int] = None

        def __post_init__(self) -> None:
            window_data.check_window_sizes(
                self.commit_rounds, self.buffer_rounds
            )

        # the word the reports name this row by
        name = "parallel"

        def build(self, terminal_policy: str) -> "ParallelWindowScheme":
            """The scheme; it lays its own tail and reads no other."""
            del terminal_policy
            return ParallelWindowScheme()

    def plan_operation(
        self,
        stream_id: int,
        round_count: int,
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ) -> window_records.OperationWindowPlan:
        """Skoric's depth-two block A/B schedule with endpoint rules.

        The published construction fixes ncom = nbuf = d. The first A
        commits the first 2d rounds. Interior A tasks read a disjoint 3d
        block and commit its middle d rounds. A B task commits the entire
        (up to 3d) region between adjacent A commits and depends on those
        A tasks; a terminal B at the physical boundary has only its left A
        predecessor. A short tail of at most d rounds is absorbed into
        the preceding A commit.
        """
        _check_skoric_widths(commit_round_count, buffer_round_count)
        layout = _BlockLayout(round_count, commit_round_count)
        layout.extend_to_the_end()
        return layout.plan(stream_id)

    def stream_layouts(self) -> "ParallelStreamLayouts":
        """Every growing stream laid block by block as its rounds arrive."""
        return ParallelStreamLayouts()

    def data_complete(
        self,
        window: window_records.Window,
        *,
        readiness: window_records.WindowReadiness,
    ) -> bool:
        """Whether the window has every round it reads."""
        return window_data.sliding_data_complete(window, readiness)


class ParallelStreamLayouts:
    """A run's growing streams, each laid in Skoric's blocks."""

    def open_stream(
        self,
        stream_id: Any,  # an opaque identity
        round_limit: Optional[int],
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ) -> "ParallelStream":
        """The stream's blocks; its end clips them when it is sealed.

        round_limit, a length the source declares, lays no other block:
        the seal at that length closes the tail.
        """
        del round_limit
        _check_skoric_widths(commit_round_count, buffer_round_count)
        return ParallelStream(stream_id, commit_round_count)


class ParallelStream:
    """One growing stream's A and B blocks, each laid once its commit begins.

    A B block waits for the A blocks on both sides. The A before it is
    the window before it, which the window side links
    (windows/window_manager.py); the A after it is laid later, so the B
    counts it unlaid as it is admitted, and that A links itself to the B
    as it is admitted (readers_of, reads_of), as a coupled region's
    reader does (windows/region_streams.py). Windows are admitted in
    index order, so the count always comes first.
    """

    def __init__(
        self,
        stream_id: Any,  # an opaque identity
        width: int,
    ) -> None:
        self.stream_id = stream_id
        self.width = width
        self.next_block = _first_a_block(width)

    def begun(self, highest_known_round: int, round_cap: Optional[int]) -> list:
        """The blocks not laid yet whose commit has begun, in order.

        They are the plan's blocks as if the stream never ended; the
        seal clips them (close_tail), so round_cap is not read.
        """
        del round_cap
        begun = []
        while self.next_block.commit_lo <= highest_known_round:
            begun.append(self.next_block)
            self.next_block = _block_after(self.next_block, self.width)
        return begun

    def cut_after(self, windows: list, last_round: int) -> list:
        """A segment edge would cut a block: refused, not built yet."""
        del windows
        raise ValueError(
            "not implemented yet: parallel windows on a stream that a "
            f"segment ends on round {last_round}; a segment's result is a "
            "sum of whole windows, and Skoric's blocks are laid across a "
            f"stream with no segments; {refusals.REFUSALS}"
        )

    def close_tail(
        self, windows: list, last_round: int
    ) -> window_records.TailChange:
        """The stream ends on the round: each block clipped to end there.

        A block that reads past the round reads and commits through it,
        as the plan's last blocks do. The last block, when it is a B,
        has no A after it: it waits for the A before it alone, the
        plan's terminal B, and no longer for the one it counted.
        """
        relaid = []
        for window in windows:
            if window.buffer_hi <= last_round:
                continue
            _end_on(window, last_round)
            relaid.append(window)
        last_window = windows[-1]
        released = ()
        if last_window.closed_temporal_boundaries:
            released = (last_window,)
        return window_records.TailChange(
            relaid=tuple(relaid), released=released
        )

    def relay(self, windows: list) -> list:
        """A laid block never moves but at the tail."""
        del windows
        return []

    def cut_rounds(self) -> tuple:
        """No segment ends on a round of a parallel stream."""
        return ()

    def reads_of(self, window: window_records.Window) -> tuple:
        """The B before an A block, which waits for it; none for a B.

        An A block reads the last d rounds of the B before it as its
        leading buffer. The first A starts where it commits, with no B
        before it.
        """
        if window.start_round == window.commit_lo:
            return ()
        previous_index = window.window_index - 1
        return ((self.stream_id, previous_index),)

    def readers_of(self, window: window_records.Window) -> tuple:
        """The A after a B block, counted unlaid; it links itself."""
        if window.closed_temporal_boundaries:
            return (None,)
        return ()

    def readers_at(self, round_index: int) -> tuple:
        """A parallel block reads no other stream's rounds."""
        del round_index
        return ()

    def let_go_before(self, window_index: int) -> None:
        """The stream keeps only its next block, so nothing is let go."""
        del window_index


class _BlockLayout:
    """Skoric's A/B blocks laid out from the first A to the operation's end."""

    def __init__(self, round_count: int, width: int) -> None:
        self.round_count = round_count
        self.width = width
        first_a = _first_a_block(width)
        clipped_first_a = _ending_by(first_a, round_count)
        self.windows = [clipped_first_a]
        self.edges = []
        self.current_a = 0

    def extend_to_the_end(self) -> None:
        """Add B and A blocks until the last round is committed."""
        while self.windows[self.current_a].commit_hi < self.round_count:
            remaining = (
                self.round_count - self.windows[self.current_a].commit_hi
            )
            if remaining <= self.width:
                self._absorb_tail_into_current_a()
                return
            if remaining <= 3 * self.width:
                self._add_terminal_b()
                return
            self._add_b_and_next_a()

    def _absorb_tail_into_current_a(self) -> None:
        """A short tail of at most d rounds joins the preceding A commit."""
        current = self.windows[self.current_a]
        self.windows[self.current_a] = window_records.WindowGeometry(
            buffer_lo=current.buffer_lo,
            commit_lo=current.commit_lo,
            commit_hi=self.round_count,
            buffer_hi=self.round_count,
        )

    def _add_terminal_b(self) -> None:
        """A terminal B at the physical boundary has only its left A."""
        current = self.windows[self.current_a]
        b_block = _b_block_after(current, self.width)
        terminal_b = _ending_by(b_block, self.round_count)
        b_index = len(self.windows)
        self.windows.append(terminal_b)
        self.edges.append((self.current_a, b_index))

    def _add_b_and_next_a(self) -> None:
        """An interior B between the current A and the next A."""
        current = self.windows[self.current_a]
        interior_b = _b_block_after(current, self.width)
        next_a = _a_block_after(interior_b, self.width)
        clipped_next_a = _ending_by(next_a, self.round_count)
        b_index = len(self.windows)
        self.windows.append(interior_b)
        next_a_index = len(self.windows)
        self.windows.append(clipped_next_a)
        self.edges.append((self.current_a, b_index))
        self.edges.append((next_a_index, b_index))
        self.current_a = next_a_index

    def plan(self, stream_id: int) -> window_records.OperationWindowPlan:
        """The plan: entries have no incoming edge, exits no outgoing one."""
        edge_tuple = tuple(self.edges)
        destinations = set()
        sources = set()
        for source, destination in edge_tuple:
            sources.add(source)
            destinations.add(destination)
        window_count = len(self.windows)
        entry_window_indices = []
        exit_window_indices = []
        for index in range(window_count):
            if index not in destinations:
                entry_window_indices.append(index)
            if index not in sources:
                exit_window_indices.append(index)
        return window_records.OperationWindowPlan(
            stream_id=stream_id,
            windows=tuple(self.windows),
            internal_dependencies=edge_tuple,
            entry_window_indices=tuple(entry_window_indices),
            exit_window_indices=tuple(exit_window_indices),
            windowed=True,
            batch_preceding_idle_rounds=False,
        )


def _check_skoric_widths(
    commit_round_count: int, buffer_round_count: int
) -> None:
    """The commit and the buffer are one width, Skoric's ncom = nbuf = d."""
    if commit_round_count == buffer_round_count:
        return
    raise ValueError(
        "parallel A/B decoding requires commit_round_count == "
        "buffer_round_count (Skoric ncom = nbuf = d)"
    )


def _first_a_block(width: int) -> window_records.WindowGeometry:
    """The first A: it commits the first 2d rounds and reads d more."""
    two_widths = 2 * width
    three_widths = 3 * width
    return window_records.WindowGeometry(
        buffer_lo=1, commit_lo=1, commit_hi=two_widths, buffer_hi=three_widths
    )


def _block_after(
    block: window_records.WindowGeometry, width: int
) -> window_records.WindowGeometry:
    """The block after a B is an A, and the block after an A is a B."""
    if block.closed_temporal_boundaries:
        return _a_block_after(block, width)
    return _b_block_after(block, width)


def _b_block_after(
    a_block: window_records.WindowGeometry, width: int
) -> window_records.WindowGeometry:
    """The 3d rounds after an A commit, closed at both ends."""
    first_round = a_block.commit_hi + 1
    last_round = a_block.commit_hi + 3 * width
    return window_records.WindowGeometry(
        first_round,
        first_round,
        last_round,
        last_round,
        closed_temporal_boundaries=True,
    )


def _a_block_after(
    b_block: window_records.WindowGeometry, width: int
) -> window_records.WindowGeometry:
    """The d rounds after a B, read with d rounds on each side."""
    commit_lo = b_block.commit_hi + 1
    buffer_lo = commit_lo - width
    commit_hi = b_block.commit_hi + width
    buffer_hi = commit_hi + width
    return window_records.WindowGeometry(
        buffer_lo=buffer_lo,
        commit_lo=commit_lo,
        commit_hi=commit_hi,
        buffer_hi=buffer_hi,
    )


def _ending_by(
    block: window_records.WindowGeometry, last_round: int
) -> window_records.WindowGeometry:
    """The block with no round past last_round, read or committed."""
    commit_hi = min(block.commit_hi, last_round)
    buffer_hi = min(block.buffer_hi, last_round)
    return dataclasses.replace(block, commit_hi=commit_hi, buffer_hi=buffer_hi)


def _end_on(window: window_records.Window, last_round: int) -> None:
    """A laid window reads and commits no round past last_round."""
    window.commit_hi = min(window.commit_hi, last_round)
    window.buffer_hi = last_round
    rounds_after_start = last_round - window.start_round
    window.round_count = rounds_after_start + 1
