"""Decoder switching policy.

The decoder-switching paper combines a fast weak decoder with a slower, more
accurate strong decoder. The weak decoder supplies `DecodeResult.soft_output`;
low confidence sends the window to the strong decoder.
"""

from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .message import Window, DecodeResult


class Switching:
    """How to combine a fast "weak" decoder with a slow, accurate "strong" decoder.

    Each window is decoded by the weak decoder first. If its confidence is high
    enough, that answer is kept. Otherwise the strong decoder re-decodes it.

    `run_both_at_once` decides when the strong decoder starts:

    - `False`: start the strong decoder only after the weak result is unsure.
    - `True`: start weak and strong together, then cancel strong work when the
      weak result is confident.

    Override `keep_weak_result` to replace the fixed threshold rule.
    """

    def __init__(self, confidence_threshold: float, run_both_at_once: bool = False,
                 weak_keepup_ratio: Optional[float] = None,
                 bulk_strong: bool = False):
        """Store the switching policy knobs."""
        if weak_keepup_ratio is not None and not 0 < weak_keepup_ratio < 1:
            raise ValueError("weak_keepup_ratio must be between 0 and 1 (the weak decoder has to "
                             f"be faster than one syndrome round); got {weak_keepup_ratio}")
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
        needed = math.ceil(ratio / (1 - ratio) * buffer_rounds)
        if commit_rounds < needed:
            raise ValueError(
                f"Switching: a {commit_rounds}-round commit region is too short for a weak decoder "
                f"running at {ratio} of a syndrome round. It needs at least {needed} rounds (for a "
                f"{buffer_rounds}-round buffer) to keep up. Use a bigger commit region or a faster "
                f"weak decoder (lower weak_keepup_ratio).")
