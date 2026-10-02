"""The rounds a window reads: how many a scheme names, and when they are in.

Every scheme reads its window the same way: the commit rounds and the
buffer rounds must be present, and a buffer that overflows past the
operation's end is satisfied by a successor's rounds, by memory rounds,
by a closed tail, or by every successor being exhausted.
"""

import decsim.config as config
import decsim.records.windows as window_records


def check_window_sizes(scheme_settings) -> None:
    """A scheme's commit and buffer sizes, each a whole count or None.

    A window is a commit region of ncom rounds and a buffer region of
    nbuf (Skoric et al. 2209.08552 lines 194-197, nW = ncom + nbuf). A
    window that commits no round never moves the stream on, so ncom is
    at least one; a buffer may be empty. None is the code distance.
    """
    _check_window_rounds("commit_rounds", scheme_settings.commit_rounds, 1)
    _check_window_rounds("buffer_rounds", scheme_settings.buffer_rounds, 0)


def sliding_data_complete(
    window: window_records.Window, readiness: window_records.WindowReadiness
) -> bool:
    """Whether a window's commit and buffer rounds are present."""
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


def _check_window_rounds(name: str, rounds, least: int) -> None:
    if rounds is None:
        return
    if config.is_whole_count(rounds, least):
        return
    raise ValueError(
        f"{name} is a whole number of rounds, at least {least}, or None for "
        f"the code's own size (got {rounds!r})"
    )
