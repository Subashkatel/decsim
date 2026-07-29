"""Windowing policies.

A scheme decides how rounds become windows and when a window has enough data.
It owns no engine state and schedules no decoder work.
See docs/PAPER_MODEL_MAP.md for the paper contract.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .message import (
    OperationWindowPlan,
    ResolvedCodeGeometry,
    WindowGeometry,
)

if TYPE_CHECKING:
    from .message import Operation, Window
    from .protocols import CodeModel, LayoutModel


class SlidingWindowScheme:
    """Serial commit and look-ahead buffer windows."""

    scheme_label = "sliding-window (serial commit/buffer chain)"
    def plan_operation(
        self,
        op_id: int,
        round_count: int,
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ) -> OperationWindowPlan:
        window_count = max(1, math.ceil(round_count / commit_round_count))
        windows = tuple(
            WindowGeometry(
                buffer_lo=window_index * commit_round_count + 1,
                commit_lo=window_index * commit_round_count + 1,
                commit_hi=min(
                    (window_index + 1) * commit_round_count,
                    round_count,
                ),
                buffer_hi=min(
                    (window_index + 1) * commit_round_count,
                    round_count,
                )
                + buffer_round_count,
            )
            for window_index in range(window_count)
        )
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
        """Reject buffers below the literature floor (lead, trail) ~ (d, d)."""
        if type(geometry) is not ResolvedCodeGeometry:
            raise TypeError(
                "geometry must be an exact ResolvedCodeGeometry"
            )
        if geometry.buffer_floor_override_active:
            return
        if (
            geometry.buffer_round_count
            < geometry.minimum_trailing_buffer_round_count
        ):
            raise ValueError(
                f"buffer_rounds={geometry.buffer_round_count} is below the "
                f"trailing buffering floor "
                f"{geometry.minimum_trailing_buffer_round_count} (~d) for "
                f"{geometry.code_name}; "
                f"windowed accuracy degrades (Skoric 2209.08552, Bombin "
                f"2303.04846). Raise buffer_rounds_override or use d.")

    def data_complete(
        self,
        window: "Window",
        *,
        rounds_arrived: int,
        successor_rounds: int,
        memory_rounds: int,
        round_count: int,
        has_successor: bool,
        operation,
    ) -> bool:
        """Return True once commit and buffer data are present."""
        in_operation_needed = min(window.buffer_hi, round_count)
        if rounds_arrived < in_operation_needed:
            return False
        overflow = window.buffer_hi - round_count
        if overflow > 0:
            if not has_successor:
                return True
            successor_has_data = successor_rounds >= overflow
            memory_has_data = memory_rounds >= overflow
            return successor_has_data or memory_has_data
        return True


class NaiveOnlineScheme(SlidingWindowScheme):
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

    def validate_buffer(self, geometry) -> None:
        if type(geometry) is not ResolvedCodeGeometry:
            raise TypeError(
                "geometry must be an exact ResolvedCodeGeometry"
            )
        pass


class ParallelWindowScheme(SlidingWindowScheme):
    """Two-layer parallel windows with layer-B boundary dependencies."""

    scheme_label = "parallel A/B two-layer window (Skoric 2209.08552, Tan 2209.09219)"

    def plan_operation(
        self,
        op_id: int,
        round_count: int,
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ) -> OperationWindowPlan:
        window_count = max(1, math.ceil(round_count / commit_round_count))
        windows = tuple(
            WindowGeometry(
                buffer_lo=max(
                    1,
                    window_index * commit_round_count + 1
                    - buffer_round_count,
                ),
                commit_lo=window_index * commit_round_count + 1,
                commit_hi=min(
                    (window_index + 1) * commit_round_count,
                    round_count,
                ),
                buffer_hi=min(
                    (window_index + 1) * commit_round_count,
                    round_count,
                )
                + buffer_round_count,
            )
            for window_index in range(window_count)
        )
        internal_dependencies = []
        for odd_index in range(1, window_count, 2):
            internal_dependencies.append((odd_index - 1, odd_index))
            if odd_index + 1 < window_count:
                internal_dependencies.append((odd_index + 1, odd_index))
        edges = tuple(internal_dependencies)
        destinations = {destination for _, destination in edges}
        sources = {source for source, _ in edges}
        return OperationWindowPlan(
            operation_id=op_id,
            windows=windows,
            internal_dependencies=edges,
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

    def validate_buffer(self, geometry) -> None:
        if type(geometry) is not ResolvedCodeGeometry:
            raise TypeError(
                "geometry must be an exact ResolvedCodeGeometry"
            )
        if geometry.buffer_floor_override_active:
            return
        required = max(
            geometry.minimum_leading_buffer_round_count,
            geometry.minimum_trailing_buffer_round_count,
        )
        if geometry.buffer_round_count < required:
            raise ValueError(
                f"buffer_rounds={geometry.buffer_round_count} is below the "
                f"two-sided buffering floor {required} (~d) for "
                f"{geometry.code_name}"
            )
