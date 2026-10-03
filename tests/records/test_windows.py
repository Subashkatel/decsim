"""The window records of decsim/records/windows.py.

Rounds are 1-based and ranges inclusive, so a window reads
[start_round, buffer_hi] and commits [commit_lo, commit_hi].
"""

import decsim.records.windows as window_records


def test_a_strong_context_starts_at_the_commit_and_ends_a_buffer_past_it():
    """A pinned past face reads no round behind the commit (Bombin 1456)."""
    window = window_records.Window(
        operation_id=1,
        window_index=2,
        commit_lo=7,
        commit_hi=9,
        buffer_hi=11,
        round_count=5,
    )
    bounds = window_records.strong_context_bounds(window)
    assert bounds == (7, 7, 9, 11)
