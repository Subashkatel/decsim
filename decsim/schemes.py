"""Windowing policies.

A scheme decides how rounds become windows and when a window has enough data.
It owns no engine state and schedules no decoder work.
See docs/PAPER_MODEL_MAP.md for the paper contract.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .message import Operation, Window
    from .protocols import CodeModel, LayoutModel


class SlidingWindowScheme:
    """Serial commit and look-ahead buffer windows."""

    scheme_label = "sliding-window (serial commit/buffer chain)"
    windowed = True

    def plan_windows(self, op_id: int, round_count: int,
                     code: CodeModel) -> list[tuple[int, int, int]]:
        """Lay out commit windows with a look-ahead buffer.

        NB when round_count % commit_rounds != 0 the last window commits a
        SHORT tail (< d rounds). Timing-wise this layout is frozen (the
        goldens pin the window count); accuracy-wise a short tail window
        decodes with reduced history — full-decode studies should pick
        round counts that divide evenly or absorb the tail in their plan
        (see benchmarks/replicate_skoric_fig6a.py)."""
        commit_rounds = code.commit_rounds()
        buffer_rounds = code.buffer_rounds()
        total_rounds = round_count
        window_count = max(1, math.ceil(total_rounds / commit_rounds))
        plan = []
        for window_index in range(window_count):
            commit_lo = window_index * commit_rounds + 1
            commit_hi = min((window_index + 1) * commit_rounds, total_rounds)
            buffer_hi = commit_hi + buffer_rounds
            plan.append((commit_lo, commit_hi, buffer_hi))
        return plan

    def validate_buffer(self, code) -> None:
        """Reject buffers below the literature floor (lead, trail) ~ (d, d)."""
        floor = getattr(code, "buffering_floor",
                        lambda scheme=None: (code.distance, code.distance))
        lead, trail = floor(self)
        if code.buffer_rounds() < trail:
            raise ValueError(
                f"buffer_rounds={code.buffer_rounds()} is below the buffering "
                f"floor {trail} (~d) for {getattr(code, 'name', code)}; "
                f"windowed accuracy degrades (Skoric 2209.08552, Bombin "
                f"2303.04846). Raise buffer_rounds_override or use d.")

    def data_complete(self, window: "Window", rounds_arrived: int, successor_rounds: int,
                      memory_rounds: int, round_count: int, has_successor: bool,
                      op: "Operation" = None, layout: "LayoutModel" = None) -> bool:
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

    batches_idle_rounds_into_next_op = True
    scheme_label = "naive online batch decode (no windowing)"
    windowed = False

    def plan_windows(self, op_id: int, round_count: int,
                     code: CodeModel) -> list[tuple[int, int, int]]:
        """One batch window: commit every round, look ahead none."""
        return [(1, round_count, round_count)]


class ParallelWindowScheme(SlidingWindowScheme):
    """Two-layer parallel windows with layer-B boundary dependencies."""

    scheme_label = "parallel A/B two-layer window (Skoric 2209.08552, Tan 2209.09219)"

    def plan_windows(self, op_id: int, round_count: int,
                     code: CodeModel) -> list[tuple[int, int, int, int]]:
        """Lay out alternating A/B commit blocks with two-sided context."""
        commit_rounds = code.commit_rounds()
        buffer_rounds = code.buffer_rounds()
        total_rounds = round_count

        plan = []
        commit_lo = 1
        while commit_lo <= total_rounds:
            commit_hi = min(commit_lo + commit_rounds - 1, total_rounds)
            buffer_lo = max(1, commit_lo - buffer_rounds)
            buffer_hi = commit_hi + buffer_rounds
            plan.append((buffer_lo, commit_lo, commit_hi, buffer_hi))
            commit_lo = commit_hi + 1
        return plan

    def wire_deps(self, windows: list) -> None:
        """Layer-B windows wait for neighboring layer-A boundary data."""
        for window_index in range(1, len(windows), 2):
            window = windows[window_index]
            window.deps.append((window.op_id, window_index - 1))
            if window_index + 1 < len(windows):
                window.deps.append((window.op_id, window_index + 1))
