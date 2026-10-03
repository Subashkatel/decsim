"""Every window's record, as the plan laid it out and as it committed.

A listener on the window manager's window_planned and the committer's
window_committed. It holds the Window records the plan laid out at
build, so a window's stamps are read here and never from the planner.
"""

import decsim.observe.window_ledger as window_ledger_module
import decsim.records.windows as window_records


def _window(operation_id: int, window_index: int):
    """One planned window over rounds 1 to 6, reading three more."""
    return window_records.Window(
        operation_id=operation_id,
        window_index=window_index,
        commit_lo=1,
        commit_hi=6,
        buffer_hi=9,
        round_count=9,
        buffer_lo=1,
    )


def test_a_grown_stream_window_arrives_through_its_own_call():
    ledger = window_ledger_module.WindowLedger()
    grown = _window(1, 1)

    ledger.window_planned(grown)

    assert ledger.windows[grown.key] is grown
