"""Small window/op policy parts: boundary handoff (port 16) and idle rounds (port 17).

Both are tiny pluggable seams the window pipeline consumes:
  - BoundaryPolicy (Eager/Held) — when a committed window ships its boundary
    handoff to dependents; consumed by WindowManager.
  - IdlePolicy (Ignore/ExtendStream/SeparateDecodeJobs) — what happens to the
    idle rounds an op emits while waiting for feedback; the reaction gate
    (chip.py) branches on .mode. MODES is the one registry of mode strings.
"""

from __future__ import annotations


# ---- boundary handoff (port 16) --------------------------------------------

class Eager:
    """Speculative default: ship at weak commit and replay if strong disagrees."""

    speculative = True

    def on_commit(self, window, final: bool) -> bool:
        return True


class Held:
    """Opt-in: ship only when the committing result is final."""

    speculative = False

    def on_commit(self, window, final: bool) -> bool:
        return final


# ---- idle rounds (port 17) -------------------------------------------------

class Ignore:
    """Timing-only memory rounds (mode 'ignore', the default)."""

    mode = "ignore"

    def account(self, idle_rounds: int, op) -> None:
        pass


class ExtendStream:
    """Inject idle rounds into the op's live dynamic stream when one exists;
    falls back to memory rounds otherwise (mode 'extend_stream')."""

    mode = "extend_stream"

    def account(self, idle_rounds: int, op) -> None:
        pass


class SeparateDecodeJobs:
    """Additionally submit an external commit+buffer decode every commit
    region of idle rounds (mode 'separate_decode_jobs')."""

    mode = "separate_decode_jobs"

    def account(self, idle_rounds: int, op) -> None:
        pass


MODES = {p.mode: p for p in (Ignore, ExtendStream, SeparateDecodeJobs)}


def from_mode(mode: str):
    """Instantiate the idle policy for a mode string."""
    if mode not in MODES:
        raise ValueError(
            f"idle_round_mode must be one of {tuple(MODES)} (got {mode!r})")
    return MODES[mode]()
