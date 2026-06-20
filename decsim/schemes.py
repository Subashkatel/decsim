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
        """Lay out commit windows with a look-ahead buffer."""
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

    scheme_label = "parallel A/B two-layer window (arXiv:2511.10633 Sec II.4)"

    def plan_windows(self, op_id: int, round_count: int,
                     code: CodeModel) -> list[tuple[int, int, int, int]]:
        """Lay out interleaved layer-A windows, layer-B gaps, and a tail if needed."""
        commit_rounds = code.commit_rounds()
        buffer_rounds = code.buffer_rounds()
        total_rounds = round_count
        period = 2 * commit_rounds + 2 * buffer_rounds
        layer_a_windows = []
        layer_index = 0
        while 1 + layer_index * period <= total_rounds:
            commit_lo = 1 + layer_index * period
            commit_hi = min(commit_lo + commit_rounds - 1, total_rounds)
            buffer_lo = max(1, commit_lo - buffer_rounds)
            layer_a_windows.append(
                (buffer_lo, commit_lo, commit_hi, commit_hi + buffer_rounds))
            layer_index += 1
        plan = []
        for layer_index, layer_a_window in enumerate(layer_a_windows):
            plan.append(layer_a_window)
            if layer_index + 1 < len(layer_a_windows):
                start_round = layer_a_window[2] + 1
                end_round = layer_a_windows[layer_index + 1][1] - 1
                plan.append((start_round, start_round, end_round, end_round))
            elif layer_a_window[2] < total_rounds:
                start_round = layer_a_window[2] + 1
                plan.append((start_round, start_round, total_rounds,
                             total_rounds + buffer_rounds))
        return plan

    def wire_deps(self, windows: list) -> None:
        """Layer-B windows depend on neighboring layer-A windows."""
        for window_index in range(1, len(windows), 2):
            window = windows[window_index]
            window.deps.append((window.op_id, window_index - 1))
            if window_index + 1 < len(windows):
                window.deps.append((window.op_id, window_index + 1))
