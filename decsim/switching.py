"""Decoder switching policy: a fast weak decoder supplies DecodeResult.soft_output; low confidence escalates to the strong decoder."""

from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .message import Window, DecodeResult


class Switching:
    """Combine weak+strong decoders; run_both_at_once=True starts both and cancels strong on confidence, else escalates only when unsure."""

    def __init__(self, confidence_threshold: float, run_both_at_once: bool = False,
                 weak_keepup_ratio: Optional[float] = None,
                 bulk_strong: bool = False):
        """Store the switching policy knobs."""
        if weak_keepup_ratio is not None and not 0 < weak_keepup_ratio < 1:
            raise ValueError(f"weak_keepup_ratio must be between 0 and 1 (got {weak_keepup_ratio})")
        if bulk_strong and run_both_at_once:
            raise ValueError("bulk_strong is only meaningful in serial mode (run_both_at_once=False)")
        self.confidence_threshold = confidence_threshold
        self.run_both_at_once = run_both_at_once
        self.weak_keepup_ratio = weak_keepup_ratio
        self.bulk_strong = bulk_strong

    def keep_weak_result(self, result: "DecodeResult") -> bool:
        """Return true when the weak decoder result should be committed."""
        return (result is not None and result.soft_output is not None
                and result.soft_output >= self.confidence_threshold)

    def calculate_strong_redo_rounds(self, window: "Window") -> int:
        """Rounds reprocessed by the strong decoder for one weak window."""
        commit = window.commit_hi - window.commit_lo + 1
        buffer = window.buffer_hi - window.commit_hi
        return commit + 2 * buffer

    def check_window_size(self, commit_rounds: int, buffer_rounds: int) -> None:
        """Raise if the weak decoder cannot keep up with this window size."""
        if self.weak_keepup_ratio is None:
            return
        ratio = self.weak_keepup_ratio
        weak_decode_rounds = ratio * (commit_rounds + buffer_rounds)
        if weak_decode_rounds > commit_rounds + 1e-9:
            needed = math.ceil(ratio / (1 - ratio) * buffer_rounds - 1e-9)
            raise ValueError(
                f"commit region of {commit_rounds} rounds too short for weak_keepup_ratio={ratio} "
                f"(needs >= {needed}); use a bigger commit region or lower weak_keepup_ratio.")
