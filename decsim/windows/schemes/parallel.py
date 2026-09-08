"""The parallel row: Skoric block A/B windows.

Skoric et al. 2209.08552 section I.C: the stream is cut into A blocks
that can be decoded at the same time and B blocks that reconcile the
seams between them, so the depth is two rather than the length of the
chain. The published construction fixes ncom = nbuf = d.
"""

import decsim.records.windows as window_records
import decsim.windows.schemes.buffer_floors as buffer_floors
import decsim.windows.schemes.window_data as window_data


class ParallelWindowScheme:
    """Skoric block A/B windows with dependency-aware seam residuals."""

    scheme_label = "parallel block A/B window (Skoric 2209.08552 sec. I.C)"

    def plan_operation(
        self,
        operation_id: int,
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
        A tasks; a terminal B at the physical boundary has only its left
        A predecessor. A short tail of at most d rounds is absorbed into
        the preceding A commit.
        """
        if commit_round_count != buffer_round_count:
            raise ValueError(
                "parallel A/B decoding requires commit_round_count == "
                "buffer_round_count (Skoric ncom = nbuf = d)"
            )
        layout = _BlockLayout(round_count, commit_round_count)
        layout.extend_to_the_end()
        return layout.plan(operation_id)

    def data_complete(
        self,
        window: window_records.Window,
        *,
        readiness: window_records.WindowReadiness,
    ) -> bool:
        """Whether the window has every round it reads."""
        return window_data.sliding_data_complete(window, readiness)

    def validate_buffer(self, geometry) -> None:
        """Reject a buffer below the two-sided floor without a justification."""
        floor = max(
            geometry.minimum_leading_buffer_round_count,
            geometry.minimum_trailing_buffer_round_count,
        )
        buffer_floors.require_buffer_floor(
            geometry, floor, "two-sided buffering floor"
        )


class _BlockLayout:
    """Skoric's A/B blocks laid out from the first A to the operation's end."""

    def __init__(self, round_count: int, width: int) -> None:
        self.round_count = round_count
        self.width = width
        two_widths = 2 * width
        three_widths = 3 * width
        first_commit_hi = min(two_widths, round_count)
        first_buffer_hi = min(three_widths, round_count)
        first_a = window_records.WindowGeometry(
            buffer_lo=1,
            commit_lo=1,
            commit_hi=first_commit_hi,
            buffer_hi=first_buffer_hi,
        )
        self.windows = [first_a]
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
        b_lo = self.windows[self.current_a].commit_hi + 1
        b_index = len(self.windows)
        terminal_b = window_records.WindowGeometry(
            b_lo,
            b_lo,
            self.round_count,
            self.round_count,
            closed_temporal_boundaries=True,
        )
        self.windows.append(terminal_b)
        self.edges.append((self.current_a, b_index))

    def _add_b_and_next_a(self) -> None:
        """An interior B between the current A and the next A."""
        b_lo = self.windows[self.current_a].commit_hi + 1
        next_a_lo = b_lo + 3 * self.width
        b_index = len(self.windows)
        b_hi = next_a_lo - 1
        interior_b = window_records.WindowGeometry(
            b_lo, b_lo, b_hi, b_hi, closed_temporal_boundaries=True
        )
        self.windows.append(interior_b)
        next_a_index = len(self.windows)
        next_a_end = next_a_lo + self.width - 1
        next_a_hi = min(next_a_end, self.round_count)
        leading_start = next_a_lo - self.width
        next_a_read_end = next_a_hi + self.width
        next_a_buffer_hi = min(next_a_read_end, self.round_count)
        next_a = window_records.WindowGeometry(
            buffer_lo=max(1, leading_start),
            commit_lo=next_a_lo,
            commit_hi=next_a_hi,
            buffer_hi=next_a_buffer_hi,
        )
        self.windows.append(next_a)
        self.edges.append((self.current_a, b_index))
        self.edges.append((next_a_index, b_index))
        self.current_a = next_a_index

    def plan(self, operation_id: int) -> window_records.OperationWindowPlan:
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
            operation_id=operation_id,
            windows=tuple(self.windows),
            internal_dependencies=edge_tuple,
            entry_window_indices=tuple(entry_window_indices),
            exit_window_indices=tuple(exit_window_indices),
            windowed=True,
            batch_preceding_idle_rounds=False,
        )
