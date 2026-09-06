"""The window ledger: every window's record, what owns it, what absorbed it.

A listener on the committer's window_committed(window, contribution) and
the forward strong window's window_absorbed(key, owner_key); it holds
the Window records the plan laid out at build, so the stamps a window
carries (slice 5's status-on-the-record rule) are read here, never from
the planner. The switching study's final rows (run_views.py) come from
what it heard.
"""

import dataclasses
from typing import Optional

import decsim.message as message


@dataclasses.dataclass(frozen=True)
class FinalWindowRow:
    """One window at the end of the run: its extents and its owner."""

    destination_key: tuple[object, int]
    weak_buffer_lo: int
    weak_commit_lo: int
    weak_commit_hi: int
    weak_buffer_hi: int
    final_commit_lo: Optional[int]
    final_commit_hi: Optional[int]
    window_disposition: str
    absorbed_into: Optional[tuple[object, int]]
    selected_request_key: Optional[message.DecoderRequestKey]


class WindowLedger:
    """The windows by key, the contribution that owns each, the absorbed."""

    def __init__(self) -> None:
        self.windows: dict = {}
        self.contribution_by_key: dict = {}
        self.absorbed_into: dict = {}

    def load_planned(self, planned_windows: dict) -> None:
        """The windows the plan laid out before anyone could listen."""
        self.windows.update(planned_windows)

    def window_planned(self, window: message.Window) -> None:
        """A stream laid one more window."""
        self.windows[window.key] = window

    def window_committed(
        self, window: message.Window, contribution: message.LogicalContribution
    ) -> None:
        """A window committed under the contribution that owns its rounds."""
        self.windows[window.key] = window
        self.contribution_by_key[window.key] = contribution

    def window_absorbed(self, key: tuple, owner_key: tuple) -> None:
        """A strong window covers the window; the weak chain skips it."""
        self.absorbed_into[key] = owner_key

    def final_rows(self) -> tuple:
        """One row per window, in stable key order."""
        rows = []
        window_items = self.windows.items()
        for key, window in sorted(window_items, key=_first_identity_order):
            row = self._final_row(key, window)
            rows.append(row)
        return tuple(rows)

    def _final_row(self, key: tuple, window: message.Window) -> FinalWindowRow:
        if key in self.absorbed_into:
            return FinalWindowRow(
                key,
                window.start_round,
                window.commit_lo,
                window.commit_hi,
                window.buffer_hi,
                None,
                None,
                "absorbed",
                self.absorbed_into[key],
                None,
            )
        contribution = self.contribution_by_key.get(key)
        if contribution is None:
            raise RuntimeError(
                f"final window {key} has no logical contribution"
            )
        return FinalWindowRow(
            key,
            window.start_round,
            window.commit_lo,
            window.commit_hi,
            window.buffer_hi,
            contribution.commit_lo,
            contribution.commit_hi,
            contribution.ownership_kind,
            None,
            window.published_request_key,
        )


def _first_identity_order(item) -> tuple:
    return message.stable_identity_order_key(item[0])
