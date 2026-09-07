"""The windowing schemes: how an operation's rounds become windows.

A scheme plans one operation's windows and their internal dependencies
and says when a window has its data; it owns no engine state and
schedules no decoder work. Table rows: sliding (Skoric et al.
2209.08552 section I.B, the overlapping recovery method), parallel
(Skoric section I.C, block A/B), sandwich (Tan et al. 2209.09219,
type-1 cores and type-2 seams), naive_online (one window per
operation).
"""

import enum
import math

import decsim.records.windows as window_records


class SlidingTerminalPolicy(enum.Enum):
    """How a finite serial stream drains its final buffered window."""

    QUITS_TAN_FLUSH = enum.auto()
    REGULAR_STRIDE_LOOKAHEAD = enum.auto()


def sliding_data_complete(
    window: window_records.Window, readiness: window_records.WindowReadiness
) -> bool:
    """Whether a window's commit and buffer rounds are present.

    A buffer that overflows past the operation's end is satisfied by a
    successor's rounds, by memory rounds, by a closed tail, or by every
    successor being exhausted. Every scheme here uses this rule.
    """
    local_rounds_needed = min(window.buffer_hi, readiness.local_round_count)
    if readiness.local_rounds_arrived < local_rounds_needed:
        return False
    overflow_rounds = window.buffer_hi - readiness.local_round_count
    if overflow_rounds <= 0 or readiness.tail_closed:
        return True
    if not readiness.successors:
        return True
    if _successor_has_rounds(readiness, overflow_rounds):
        return True
    if readiness.memory_rounds_arrived >= overflow_rounds:
        return True
    return _every_successor_exhausted(readiness)


def buffer_filled_by_memory_only(
    window: window_records.Window, readiness: window_records.WindowReadiness
) -> bool:
    """Whether memory rounds alone satisfy the buffer past the operation.

    That is a trailing buffer with no successor content standing behind it.

    Such a release is time-only: the reference systems decode the buffer
    region's content (Skoric and Tan windows, LATTE d^3+buffer blocks),
    so a window released this way carries an approximate result. The
    terminal no-successor release is the Tan
    flush and is not flagged.
    """
    overflow_rounds = window.buffer_hi - readiness.local_round_count
    if overflow_rounds <= 0 or readiness.tail_closed:
        return False
    if not readiness.successors:
        return False
    if _successor_has_rounds(readiness, overflow_rounds):
        return False
    return readiness.memory_rounds_arrived >= overflow_rounds


class SlidingWindowScheme:
    """Serial commit and look-ahead buffer windows."""

    scheme_label = "sliding-window (serial commit/buffer chain)"

    def __init__(
        self,
        terminal_policy: SlidingTerminalPolicy = (
            SlidingTerminalPolicy.QUITS_TAN_FLUSH
        ),
    ) -> None:
        self.terminal_policy = terminal_policy

    def plan_operation(
        self,
        operation_id: int,
        round_count: int,
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ) -> window_records.OperationWindowPlan:
        """The finite forward (W, F) construction.

        F is commit_round_count and W is commit_round_count plus
        buffer_round_count. Regular windows commit their first F rounds.
        Under the Tan flush the last window begins when fewer than W + F
        rounds remain and commits every remaining round (qLDPC's
        SlidingWindowDecoder tail rule; the last window is never shorter
        than W); under the lookahead policy every window strides F and
        the last commits what is left.
        """
        if self.terminal_policy is SlidingTerminalPolicy.QUITS_TAN_FLUSH:
            windows = _finite_forward_window_geometries(
                round_count, commit_round_count, buffer_round_count
            )
        else:
            windows = _lookahead_window_geometries(
                round_count, commit_round_count, buffer_round_count
            )
        window_count = len(windows)
        last_index = window_count - 1
        internal_dependencies = []
        for window_index in range(last_index):
            next_index = window_index + 1
            internal_dependencies.append((window_index, next_index))
        return window_records.OperationWindowPlan(
            operation_id=operation_id,
            windows=windows,
            internal_dependencies=tuple(internal_dependencies),
            entry_window_indices=(0,),
            exit_window_indices=(last_index,),
            windowed=True,
            batch_preceding_idle_rounds=False,
        )

    def validate_buffer(self, geometry) -> None:
        """Reject a buffer below the trailing floor without a justification."""
        _require_buffer_floor(
            geometry,
            geometry.minimum_trailing_buffer_round_count,
            "trailing buffering floor",
        )

    def data_complete(
        self,
        window: window_records.Window,
        *,
        readiness: window_records.WindowReadiness,
    ) -> bool:
        """Whether the window has every round it reads."""
        return sliding_data_complete(window, readiness)


class NaiveOnlineScheme:
    """Decode each operation as one full batch after all rounds arrive."""

    scheme_label = "naive online batch decode (no windowing)"

    def plan_operation(
        self,
        operation_id: int,
        round_count: int,
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ) -> window_records.OperationWindowPlan:
        """One window over the whole operation."""
        del commit_round_count, buffer_round_count
        whole = window_records.WindowGeometry(1, 1, round_count, round_count)
        return window_records.OperationWindowPlan(
            operation_id=operation_id,
            windows=(whole,),
            internal_dependencies=(),
            entry_window_indices=(0,),
            exit_window_indices=(0,),
            windowed=False,
            batch_preceding_idle_rounds=True,
        )

    def data_complete(
        self,
        window: window_records.Window,
        *,
        readiness: window_records.WindowReadiness,
    ) -> bool:
        """Whether the window has every round it reads."""
        return sliding_data_complete(window, readiness)

    def validate_buffer(self, geometry) -> None:
        """A batch decode has no buffer floor."""


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
        return sliding_data_complete(window, readiness)

    def validate_buffer(self, geometry) -> None:
        """Reject a buffer below the two-sided floor without a justification."""
        floor = max(
            geometry.minimum_leading_buffer_round_count,
            geometry.minimum_trailing_buffer_round_count,
        )
        _require_buffer_floor(geometry, floor, "two-sided buffering floor")


class TanSandwichScheme:
    """Tan et al.'s zero-seam sandwich decoder for graphlike memory DEMs.

    commit_round_count is the paper's step s and buffer_round_count is
    the two-sided buffer b, so the type-1 width is w = s + 2b (2209.09219
    supplement S8). The typed detector intervals are the vertex sets
    whose incident correction edges form each core; the one-layer gaps
    are the type-2 seams, using seam offset t = 0.
    """

    scheme_label = (
        "Tan zero-seam sandwich (type-1 cores / type-2 seam reconciliation)"
    )

    def plan_operation(
        self,
        operation_id: int,
        round_count: int,
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ) -> window_records.OperationWindowPlan:
        """Type-1 cores every s rounds with a type-2 seam between each pair."""
        step = commit_round_count
        buffer = buffer_round_count
        if step < 2:
            raise ValueError("Tan sandwich decoding requires step size s >= 2")
        if buffer < 1:
            raise ValueError(
                "Tan sandwich decoding requires overlapping windows (b >= 1)"
            )
        width = step + 2 * buffer
        span = round_count - width
        span_in_steps = span / step
        type_1_count = math.ceil(span_in_steps)
        type_1_count += 1
        type_1_count = max(1, type_1_count)
        cores = _TanCores(round_count, step, buffer, width, type_1_count)
        windows = [cores.type_1_window(0)]
        edges = []
        seam_count = type_1_count - 1
        for left_type_1 in range(seam_count):
            seam_round = buffer + step + left_type_1 * step
            seam_index = len(windows)
            seam = window_records.WindowGeometry(
                seam_round,
                seam_round,
                seam_round,
                seam_round,
                closed_temporal_boundaries=True,
            )
            windows.append(seam)
            right_type_1_index = len(windows)
            right_index = left_type_1 + 1
            right_type_1 = cores.type_1_window(right_index)
            windows.append(right_type_1)
            left_index = seam_index - 1
            edges.append((left_index, seam_index))
            edges.append((right_type_1_index, seam_index))
        window_count = len(windows)
        exit_window_indices = tuple(range(1, window_count, 2))
        if not exit_window_indices:
            exit_window_indices = (0,)
        return window_records.OperationWindowPlan(
            operation_id=operation_id,
            windows=tuple(windows),
            internal_dependencies=tuple(edges),
            entry_window_indices=tuple(range(0, window_count, 2)),
            exit_window_indices=exit_window_indices,
            windowed=True,
            batch_preceding_idle_rounds=False,
            protocol=window_records.WindowProtocol.TAN_ZERO_SEAM_GRAPHLIKE,
        )

    def data_complete(
        self,
        window: window_records.Window,
        *,
        readiness: window_records.WindowReadiness,
    ) -> bool:
        """Whether the window has every round it reads."""
        return sliding_data_complete(window, readiness)

    def validate_buffer(self, geometry) -> None:
        """Reject a step below 2 or a buffer below 1."""
        if geometry.commit_round_count < 2:
            raise ValueError("Tan sandwich decoding requires step size s >= 2")
        if geometry.buffer_round_count < 1:
            raise ValueError("Tan sandwich decoding requires b >= 1")


class _TanCores:
    """The type-1 windows of one Tan schedule, by index."""

    def __init__(self, round_count, step, buffer, width, type_1_count):
        self.round_count = round_count
        self.step = step
        self.buffer = buffer
        self.width = width
        self.type_1_count = type_1_count

    def type_1_window(self, index: int) -> window_records.WindowGeometry:
        """Core index reads w rounds from 1 + index*s, commits the middle s."""
        read_lo = 1 + index * self.step
        commit_lo = read_lo + self.buffer
        if index == 0:
            commit_lo = 1
        commit_hi = self.buffer + self.step - 1 + index * self.step
        if index == self.type_1_count - 1:
            commit_hi = self.round_count
        read_end = read_lo + self.width - 1
        buffer_hi = min(self.round_count, read_end)
        return window_records.WindowGeometry(
            buffer_lo=read_lo,
            commit_lo=commit_lo,
            commit_hi=commit_hi,
            buffer_hi=buffer_hi,
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


def _finite_forward_window_geometries(
    round_count: int,
    commit_round_count: int,
    buffer_round_count: int,
) -> tuple:
    """Finite forward windows with one closed all-core tail.

    Regular windows commit F rounds and read W = F + B. The tail rule is
    qLDPC's SlidingWindowDecoder rule (sinter.py, `while start < end -
    (W + s - 1)`): the last window starts as soon as fewer than W + F
    rounds remain, so it commits everything left and is never shorter
    than W. A short tail is never decoded on its own; it is absorbed by
    the last full-width window instead.
    """
    window_width = commit_round_count + buffer_round_count
    windows = []
    commit_lo = 1
    while True:
        remaining_rounds = round_count - commit_lo + 1
        if remaining_rounds < window_width + commit_round_count:
            tail = window_records.WindowGeometry(
                buffer_lo=commit_lo,
                commit_lo=commit_lo,
                commit_hi=round_count,
                buffer_hi=round_count,
            )
            windows.append(tail)
            return tuple(windows)
        regular_commit_hi = commit_lo + commit_round_count - 1
        regular_buffer_hi = regular_commit_hi + buffer_round_count
        regular = window_records.WindowGeometry(
            buffer_lo=commit_lo,
            commit_lo=commit_lo,
            commit_hi=regular_commit_hi,
            buffer_hi=regular_buffer_hi,
        )
        windows.append(regular)
        commit_lo = regular_commit_hi + 1


def _lookahead_window_geometries(
    round_count: int,
    commit_round_count: int,
    buffer_round_count: int,
) -> tuple:
    """Every window strides F and reads B past its commit, the last clipped."""
    strides = round_count / commit_round_count
    window_count = math.ceil(strides)
    window_count = max(1, window_count)
    windows = []
    for index in range(window_count):
        commit_lo = index * commit_round_count + 1
        stride_end = (index + 1) * commit_round_count
        commit_hi = min(stride_end, round_count)
        buffer_hi = commit_hi + buffer_round_count
        geometry = window_records.WindowGeometry(
            buffer_lo=commit_lo,
            commit_lo=commit_lo,
            commit_hi=commit_hi,
            buffer_hi=buffer_hi,
        )
        windows.append(geometry)
    return tuple(windows)


def _require_buffer_floor(geometry, floor: int, floor_label: str) -> None:
    """Refuse a buffer below the literature floor without a justification.

    A justification above the floor is a stale one and is refused too.
    """
    below = geometry.buffer_round_count < floor
    justification = geometry.window_floor_justification
    if below and not justification:
        raise ValueError(
            f"buffer_rounds={geometry.buffer_round_count} is below the "
            f"{floor_label} {floor} for {geometry.code_name}; windowed "
            f"accuracy degrades (Skoric 2209.08552, Tan PRX Quantum 4, "
            f"040344, Bombin 2303.04846). Raise buffer_rounds_override to "
            f"{floor}, or set window_floor_justification to run below the "
            "floor deliberately."
        )
    if justification and not below:
        raise ValueError(
            f"window_floor_justification is set but buffer_rounds="
            f"{geometry.buffer_round_count} is not below the {floor_label} "
            f"{floor}; remove the justification."
        )


def _successor_has_rounds(
    readiness: window_records.WindowReadiness, overflow_rounds: int
) -> bool:
    for successor in readiness.successors:
        if successor.rounds_arrived >= overflow_rounds:
            return True
    return False


def _every_successor_exhausted(
    readiness: window_records.WindowReadiness,
) -> bool:
    for successor in readiness.successors:
        if successor.rounds_arrived < successor.round_count:
            return False
    return True
