"""Windowing policies.

A scheme decides how rounds become windows and when a window has enough data.
It owns no engine state and schedules no decoder work.
See docs/PAPER_MODEL_MAP.md for the paper contract.
"""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import TYPE_CHECKING

from ..message import (
    OperationWindowPlan,
    WindowReadiness,
    WindowGeometry,
    WindowProtocol,
)

if TYPE_CHECKING:
    from ..message import Window


class SlidingTerminalPolicy(Enum):
    """How a finite serial stream drains its final buffered window."""

    QUITS_TAN_FLUSH = auto()
    REGULAR_STRIDE_LOOKAHEAD = auto()


def _finite_forward_window_geometries(
    round_count: int,
    commit_round_count: int,
    buffer_round_count: int,
) -> tuple[WindowGeometry, ...]:
    """Finite forward windows with one closed all-core tail.

    Regular windows commit F rounds and read W = F + B. The tail rule is
    qLDPC's SlidingWindowDecoder rule (sinter.py, `while start < end -
    (W + s - 1)`): the last window starts as soon as fewer than W + F rounds
    remain, so it commits everything left and is never shorter than W. A
    short tail is never decoded on its own; it is absorbed by the last
    full-width window instead.
    """
    window_width = commit_round_count + buffer_round_count
    windows = []
    commit_lo = 1
    while True:
        remaining_rounds = round_count - commit_lo + 1
        if remaining_rounds < window_width + commit_round_count:
            windows.append(WindowGeometry(
                buffer_lo=commit_lo,
                commit_lo=commit_lo,
                commit_hi=round_count,
                buffer_hi=round_count,
            ))
            break
        regular_commit_hi = commit_lo + commit_round_count - 1
        windows.append(WindowGeometry(
            buffer_lo=commit_lo,
            commit_lo=commit_lo,
            commit_hi=regular_commit_hi,
            buffer_hi=regular_commit_hi + buffer_round_count,
        ))
        commit_lo = regular_commit_hi + 1
    return tuple(windows)


def _require_buffer_floor(geometry, floor: int, floor_label: str) -> None:
    """Refuse a buffer below the literature floor unless the code card says
    why it runs there; a justification above the floor is a stale one."""
    below = geometry.buffer_round_count < floor
    justification = geometry.window_floor_justification
    if below and not justification:
        raise ValueError(
            f"buffer_rounds={geometry.buffer_round_count} is below the "
            f"{floor_label} {floor} for {geometry.code_name}; windowed "
            f"accuracy degrades (Skoric 2209.08552, Tan PRX Quantum 4, "
            f"040344, Bombin 2303.04846). Raise buffer_rounds_override to "
            f"{floor}, or set window_floor_justification to run below the "
            "floor deliberately.")
    if justification and not below:
        raise ValueError(
            f"window_floor_justification is set but buffer_rounds="
            f"{geometry.buffer_round_count} is not below the {floor_label} "
            f"{floor}; remove the justification.")


def sliding_data_complete(window: "Window", readiness: WindowReadiness) -> bool:
    """A window has its data once its commit and buffer rounds are present;
    a buffer that overflows past the operation's end is satisfied by a
    successor's rounds, by memory rounds, by a closed tail, or by every
    successor being exhausted. Every scheme here uses this rule."""
    local_rounds_needed = min(window.buffer_hi, readiness.local_round_count)
    if readiness.local_rounds_arrived < local_rounds_needed:
        return False
    overflow_rounds = window.buffer_hi - readiness.local_round_count
    if overflow_rounds <= 0 or readiness.tail_closed:
        return True
    if not readiness.successors:
        return True
    successor_has_data = any(
        successor.rounds_arrived >= overflow_rounds
        for successor in readiness.successors
    )
    memory_has_data = readiness.memory_rounds_arrived >= overflow_rounds
    all_successors_exhausted = all(
        successor.rounds_arrived >= successor.round_count
        for successor in readiness.successors
    )
    return successor_has_data or memory_has_data or all_successors_exhausted


class SlidingWindowScheme:
    """Serial commit and look-ahead buffer windows."""

    scheme_label = "sliding-window (serial commit/buffer chain)"

    def __init__(
        self,
        terminal_policy: SlidingTerminalPolicy = SlidingTerminalPolicy.QUITS_TAN_FLUSH,
    ) -> None:
        self.terminal_policy = terminal_policy

    def plan_operation(
        self,
        op_id: int,
        round_count: int,
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ) -> OperationWindowPlan:
        """Return the finite forward `(W,F)` construction.

        Here ``F=commit_round_count`` and
        ``W=commit_round_count+buffer_round_count``. Regular windows commit
        their first ``F`` rounds. The last window begins when fewer than
        ``W+F`` rounds remain and commits every remaining round (qLDPC's
        SlidingWindowDecoder tail rule; the last window is never shorter
        than ``W``).
        """
        if self.terminal_policy is SlidingTerminalPolicy.QUITS_TAN_FLUSH:
            windows = _finite_forward_window_geometries(
                round_count,
                commit_round_count,
                buffer_round_count,
            )
        else:
            window_count = max(1, math.ceil(round_count / commit_round_count))
            windows = tuple(
                WindowGeometry(
                    buffer_lo=index * commit_round_count + 1,
                    commit_lo=index * commit_round_count + 1,
                    commit_hi=min((index + 1) * commit_round_count, round_count),
                    buffer_hi=(
                        min((index + 1) * commit_round_count, round_count)
                        + buffer_round_count
                    ),
                )
                for index in range(window_count)
            )
        window_count = len(windows)
        return OperationWindowPlan(
            operation_id=op_id,
            windows=windows,
            internal_dependencies=tuple(
                (window_index, window_index + 1)
                for window_index in range(window_count - 1)
            ),
            entry_window_indices=(0,),
            exit_window_indices=(window_count - 1,),
            windowed=True,
            batch_preceding_idle_rounds=False,
        )

    def validate_buffer(self, geometry) -> None:
        """Reject buffers below the trailing floor (~d) without a justification."""
        _require_buffer_floor(
            geometry,
            geometry.minimum_trailing_buffer_round_count,
            "trailing buffering floor",
        )

    def data_complete(self, window: "Window", *, readiness: WindowReadiness,
                      operation) -> bool:
        return sliding_data_complete(window, readiness)


class NaiveOnlineScheme:
    """Decode each operation as one full batch after all rounds arrive."""

    scheme_label = "naive online batch decode (no windowing)"

    def plan_operation(
        self,
        op_id: int,
        round_count: int,
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ) -> OperationWindowPlan:
        return OperationWindowPlan(
            operation_id=op_id,
            windows=(WindowGeometry(1, 1, round_count, round_count),),
            internal_dependencies=(),
            entry_window_indices=(0,),
            exit_window_indices=(0,),
            windowed=False,
            batch_preceding_idle_rounds=True,
        )

    def data_complete(self, window: "Window", *, readiness: WindowReadiness,
                      operation) -> bool:
        return sliding_data_complete(window, readiness)


    def validate_buffer(self, geometry) -> None:
        pass


class ParallelWindowScheme:
    """Skoric block A/B windows with dependency-aware seam residuals."""

    scheme_label = "parallel block A/B window (Skoric 2209.08552 sec. I.C)"

    def plan_operation(
        self,
        op_id: int,
        round_count: int,
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ) -> OperationWindowPlan:
        """Return Skoric's depth-two block A/B schedule with endpoint rules.

        The published construction fixes ``ncom = nbuf = d``. The first A
        commits the first ``2d`` rounds. Interior A tasks read a disjoint
        ``3d`` block and commit its middle ``d`` rounds. A B task commits the
        entire (up to ``3d``) region between adjacent A commits and depends on
        those A tasks; a terminal B at the physical boundary has only its left
        A predecessor. A short tail of at most ``d`` rounds is absorbed into
        the preceding A commit.
        """
        if commit_round_count != buffer_round_count:
            raise ValueError(
                "parallel A/B decoding requires commit_round_count == "
                "buffer_round_count (Skoric ncom = nbuf = d)"
            )
        width = commit_round_count
        windows = [
            WindowGeometry(
                buffer_lo=1,
                commit_lo=1,
                commit_hi=min(2 * width, round_count),
                buffer_hi=min(3 * width, round_count),
            )
        ]
        edges = []
        current_a = 0

        while windows[current_a].commit_hi < round_count:
            remaining = round_count - windows[current_a].commit_hi
            if remaining <= width:
                current = windows[current_a]
                windows[current_a] = WindowGeometry(
                    buffer_lo=current.buffer_lo,
                    commit_lo=current.commit_lo,
                    commit_hi=round_count,
                    buffer_hi=round_count,
                )
                break

            b_lo = windows[current_a].commit_hi + 1
            if remaining <= 3 * width:
                b_index = len(windows)
                windows.append(WindowGeometry(
                    b_lo,
                    b_lo,
                    round_count,
                    round_count,
                    closed_temporal_boundaries=True,
                ))
                edges.append((current_a, b_index))
                break

            next_a_lo = b_lo + 3 * width
            b_index = len(windows)
            windows.append(WindowGeometry(
                b_lo,
                b_lo,
                next_a_lo - 1,
                next_a_lo - 1,
                closed_temporal_boundaries=True,
            ))

            next_a_index = len(windows)
            next_a_hi = min(next_a_lo + width - 1, round_count)
            windows.append(WindowGeometry(
                buffer_lo=max(1, next_a_lo - width),
                commit_lo=next_a_lo,
                commit_hi=next_a_hi,
                buffer_hi=min(next_a_hi + width, round_count),
            ))
            edges.extend(((current_a, b_index), (next_a_index, b_index)))
            current_a = next_a_index

        edge_tuple = tuple(edges)
        destinations = {destination for _, destination in edge_tuple}
        sources = {source for source, _ in edge_tuple}
        window_count = len(windows)
        return OperationWindowPlan(
            operation_id=op_id,
            windows=tuple(windows),
            internal_dependencies=edge_tuple,
            entry_window_indices=tuple(
                index for index in range(window_count)
                if index not in destinations
            ),
            exit_window_indices=tuple(
                index for index in range(window_count)
                if index not in sources
            ),
            windowed=True,
            batch_preceding_idle_rounds=False,
        )

    def data_complete(self, window: "Window", *, readiness: WindowReadiness,
                      operation) -> bool:
        return sliding_data_complete(window, readiness)


    def validate_buffer(self, geometry) -> None:
        """Reject buffers below the two-sided floor (~d) without a justification."""
        _require_buffer_floor(
            geometry,
            max(
                geometry.minimum_leading_buffer_round_count,
                geometry.minimum_trailing_buffer_round_count,
            ),
            "two-sided buffering floor",
        )


class TanSandwichScheme:
    """Tan et al.'s zero-seam sandwich decoder for graphlike memory DEMs.

    ``commit_round_count`` is the paper's step ``s`` and
    ``buffer_round_count`` is the two-sided buffer ``b``. Thus the type-1
    width is ``w=s+2b``. The typed detector intervals below are the vertex
    sets whose incident correction edges form each core; the one-layer gaps
    are the type-2 seams, using seam offset ``t=0``.
    """

    scheme_label = (
        "Tan zero-seam sandwich (type-1 cores / type-2 seam reconciliation)"
    )

    def plan_operation(
        self,
        op_id: int,
        round_count: int,
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ) -> OperationWindowPlan:
        step = commit_round_count
        buffer = buffer_round_count
        if step < 2:
            raise ValueError("Tan sandwich decoding requires step size s >= 2")
        if buffer < 1:
            raise ValueError("Tan sandwich decoding requires overlapping windows (b >= 1)")
        width = step + 2 * buffer
        type_1_count = max(1, math.ceil((round_count - width) / step) + 1)

        def type_1_window(index: int) -> WindowGeometry:
            read_lo = 1 + index * step
            return WindowGeometry(
                buffer_lo=read_lo,
                commit_lo=(1 if index == 0 else read_lo + buffer),
                commit_hi=(
                    round_count
                    if index == type_1_count - 1
                    else buffer + step - 1 + index * step
                ),
                buffer_hi=min(round_count, read_lo + width - 1),
            )

        windows = [type_1_window(0)]
        edges = []
        for left_type_1 in range(type_1_count - 1):
            seam_round = buffer + step + left_type_1 * step
            seam_index = len(windows)
            windows.append(WindowGeometry(
                seam_round,
                seam_round,
                seam_round,
                seam_round,
                closed_temporal_boundaries=True,
            ))
            right_type_1_index = len(windows)
            windows.append(type_1_window(left_type_1 + 1))
            edges.extend((
                (seam_index - 1, seam_index),
                (right_type_1_index, seam_index),
            ))

        return OperationWindowPlan(
            operation_id=op_id,
            windows=tuple(windows),
            internal_dependencies=tuple(edges),
            entry_window_indices=tuple(range(0, len(windows), 2)),
            exit_window_indices=(
                tuple(range(1, len(windows), 2)) or (0,)
            ),
            windowed=True,
            batch_preceding_idle_rounds=False,
            protocol=WindowProtocol.TAN_ZERO_SEAM_GRAPHLIKE,
        )

    def data_complete(self, window: "Window", *, readiness: WindowReadiness,
                      operation) -> bool:
        return sliding_data_complete(window, readiness)


    def validate_buffer(self, geometry) -> None:
        if geometry.commit_round_count < 2:
            raise ValueError("Tan sandwich decoding requires step size s >= 2")
        if geometry.buffer_round_count < 1:
            raise ValueError("Tan sandwich decoding requires b >= 1")
