"""The sliding row: serial commit windows with a look-ahead buffer.

Skoric et al. 2209.08552 section I.B, the overlapping recovery method:
window i commits F rounds and reads B more, and window i + 1 begins
where window i's commit ended. How a finite stream drains its last
buffered window is the terminal policy: the Tan flush, which is qLDPC's
SlidingWindowDecoder tail rule, or a regular stride whose last commit is
what is left.
"""

import math

import decsim.records.windows as window_records
import decsim.windows.schemes.buffer_floors as buffer_floors
import decsim.windows.schemes.window_data as window_data


class SlidingWindowScheme:
    """Serial commit and look-ahead buffer windows.

    windows.terminal_policy is the one key this row reads: flush ends the
    last window at the stream's last round, Tan's QUITS flush
    (2209.09219 lines 1029-1030), and lookahead keeps the regular stride,
    so the last window still reads rounds past its own commit and a
    strong recovery has context to read.
    """

    scheme_label = "sliding-window (serial commit/buffer chain)"
    commits_in_one_serial_chain = True
    supports_dynamic_streams = True

    def __init__(
        self,
        card: window_records.WindowingSchemeCard = (
            window_records.DEFAULT_SCHEME_CARD
        ),
    ) -> None:
        self.terminal_policy = card.terminal_policy
        self.has_trailing_tail_context = card.terminal_policy == "lookahead"

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
        windows = self._window_geometries(
            round_count, commit_round_count, buffer_round_count
        )
        window_count = len(windows)
        last_index = window_count - 1
        internal_dependencies = _chain_dependencies(window_count)
        return window_records.OperationWindowPlan(
            operation_id=operation_id,
            windows=windows,
            internal_dependencies=internal_dependencies,
            entry_window_indices=(0,),
            exit_window_indices=(last_index,),
            windowed=True,
            batch_preceding_idle_rounds=False,
        )

    def _window_geometries(
        self,
        round_count: int,
        commit_round_count: int,
        buffer_round_count: int,
    ) -> tuple:
        """The geometries this operation's terminal policy lays out."""
        if self.terminal_policy == "flush":
            return _finite_forward_window_geometries(
                round_count, commit_round_count, buffer_round_count
            )
        return _lookahead_window_geometries(
            round_count, commit_round_count, buffer_round_count
        )

    def validate_buffer(self, geometry) -> None:
        """Reject a buffer below the trailing floor without a justification."""
        buffer_floors.require_buffer_floor(
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
        return window_data.sliding_data_complete(window, readiness)


def _chain_dependencies(window_count: int) -> tuple:
    """Each window waits on the one before it, in order."""
    last_index = window_count - 1
    internal_dependencies = []
    for window_index in range(last_index):
        next_index = window_index + 1
        internal_dependencies.append((window_index, next_index))
    return tuple(internal_dependencies)


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
