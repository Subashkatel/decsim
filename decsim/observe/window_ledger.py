"""The window ledger: every window's record, heard as the plan grows.

A listener on the window manager's window_planned and the committer's
window_committed(window, contribution); it holds the Window records the
plan laid out at build, so the stamps a window carries (a window's
status lives on its record) are read here, never from the planner.
"""

import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records


class WindowLedger:
    """The windows by key, each the record the plan or its commit left."""

    def __init__(self) -> None:
        self.windows: dict = {}

    def load_planned(self, planned_windows: dict) -> None:
        """The windows the plan laid out before anyone could listen."""
        self.windows.update(planned_windows)

    def window_planned(self, window: window_records.Window) -> None:
        """A stream laid one more window."""
        self.windows[window.key] = window

    def window_committed(
        self,
        window: window_records.Window,
        contribution: decoding_records.LogicalContribution,
    ) -> None:
        """A window committed; its record now carries its commit."""
        del contribution
        self.windows[window.key] = window
