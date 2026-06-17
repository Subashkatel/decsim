from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .message import Window, DecodeResult

# Decoder switching (Toshio et al., 2025, arXiv:2510.25222).
#
# Use a fast but less accurate "weak" decoder for most windows, and fall back to a slow but
# accurate "strong" decoder only for the windows the weak decoder is unsure about. The weak
# decoder reports a confidence value with each window (DecodeResult.soft_output); a low value
# means "I might be wrong here" and triggers the strong decoder on that window.


class Switching:
    """How to combine a fast "weak" decoder with a slow, accurate "strong" decoder.

    Each window is decoded by the weak decoder first. If its confidence is high enough, that
    answer is kept. If not, the strong decoder re-decodes the window.

    `run_both_at_once` decides WHEN the strong decoder starts:
      - False (default): start the strong decoder only after a window comes back unsure. The
        strong decoder runs on just the few windows that need it. (The paper's "serial" mode.)
      - True: start the strong decoder together with the weak one on every window, and cancel
        it whenever the weak answer turns out confident. The strong answer is ready sooner when
        it is needed, but the strong decoder wastes work on windows that turn out fine. (The
        paper's "parallel" mode -- "both decode".)

    Pass an instance to build_and_run(..., switching=...). To change how the keep-or-redecode
    decision is made (say, a learned predictor instead of a fixed threshold), subclass this and
    override keep_weak_result().
    """

    def __init__(self, confidence_threshold: float, run_both_at_once: bool = False,
                 weak_keepup_ratio: float = None):
        """confidence_threshold -- keep the weak answer when its confidence is at least this.
        run_both_at_once       -- see the class docstring (serial vs parallel).
        weak_keepup_ratio      -- optional; the weak decoder's time per round divided by the
                                  syndrome-generation time per round. If given, check_window_size()
                                  verifies the weak decoder is fast enough to keep up."""
        if weak_keepup_ratio is not None and not 0 < weak_keepup_ratio < 1:
            raise ValueError("weak_keepup_ratio must be between 0 and 1 (the weak decoder has to "
                             f"be faster than one syndrome round); got {weak_keepup_ratio}")
        self.confidence_threshold = confidence_threshold
        self.run_both_at_once = run_both_at_once
        self.weak_keepup_ratio = weak_keepup_ratio

    def keep_weak_result(self, result: "DecodeResult") -> bool:
        """Should we keep the weak decoder's answer (True), or hand the window to the strong decoder
        (False)? Default rule: keep it when its confidence is at least confidence_threshold. Override
        this for a different rule, for example a learned predictor."""
        return (result is not None and result.soft_output is not None
                and result.soft_output >= self.confidence_threshold)

    def calculate_strong_redo_rounds(self, window: "Window") -> int:
        """How many rounds the strong decoder reprocesses for one window: the window's own commit
        and buffer rounds, plus one extra buffer in front, so the strong decoder sees full context
        on both sides. For the usual case commit = buffer = d, this is 3*d rounds."""
        commit = window.commit_hi - window.commit_lo + 1
        buffer = window.buffer_hi - window.commit_hi
        return commit + 2 * buffer

    def check_window_size(self, commit_rounds: int, buffer_rounds: int) -> None:
        """Raise if the weak decoder is too slow for this window size (Eq. 7 of the paper): the
        commit region must be long enough that the weak decoder finishes a window before the next
        commit region's worth of syndrome data arrives. Does nothing unless weak_keepup_ratio was set."""
        import math
        if self.weak_keepup_ratio is None:
            return
        ratio = self.weak_keepup_ratio
        needed = math.ceil(ratio / (1 - ratio) * buffer_rounds)
        if commit_rounds < needed:
            raise ValueError(
                f"Switching: a {commit_rounds}-round commit region is too short for a weak decoder "
                f"running at {ratio} of a syndrome round -- it needs at least {needed} rounds (for a "
                f"{buffer_rounds}-round buffer) to keep up. Use a bigger commit region or a faster "
                f"weak decoder (lower weak_keepup_ratio).")
