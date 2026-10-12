"""One value per window position, whose front a stream lets go.

A reader walks an operation's windows by position: the cursors of the
committed and final prefixes (operation_results.py), the first window
not yet queued (window_manager.py), the start floors (window_planner.py).
A stream that runs for 10^6 rounds cannot keep a value for every window
it has had, so once its windows are final and measured it lets the
front go (window_release.py), and a position keeps its meaning: the
values from first_kept on are kept, and a position before it is gone.
"""

import collections


class WindowPositions:
    """The values at positions first_kept and on, in position order."""

    def __init__(self) -> None:
        self.first_kept = 0
        self.kept: collections.deque = collections.deque()

    def __len__(self) -> int:
        """Every position there has been, the let-go ones included."""
        kept_count = len(self.kept)
        return self.first_kept + kept_count

    def __getitem__(self, position: int) -> object:
        """The value at a kept position; one let go raises."""
        if position < self.first_kept:
            raise IndexError(
                f"window position {position} was let go; positions from "
                f"{self.first_kept} on are kept"
            )
        offset = position - self.first_kept
        return self.kept[offset]

    def __setitem__(self, position: int, value: object) -> None:
        """Replace the value at a kept position."""
        if position < self.first_kept:
            raise IndexError(f"window position {position} was let go")
        offset = position - self.first_kept
        self.kept[offset] = value

    def __iter__(self):
        """The kept values, in position order."""
        return iter(self.kept)

    def append(self, value: object) -> None:
        """The value at the next position."""
        self.kept.append(value)

    def index(self, value: object) -> int:
        """The position of a kept value."""
        offset = self.kept.index(value)
        return self.first_kept + offset

    def let_go_before(self, position: int) -> None:
        """Let every position before this one go.

        A position not reached yet is let go too, so the next value
        appended may sit past the last one kept.
        """
        while self.kept and self.first_kept < position:
            self.kept.popleft()
            self.first_kept += 1
        if self.first_kept < position:
            self.first_kept = position
