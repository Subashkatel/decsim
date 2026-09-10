"""When a committed window ships its boundary to the windows after it.

A boundary policy fills the BoundaryPolicy seam (decsim/ports.py). Eager
ships at every commit, provisional or final, which keeps a serial chain
of windows moving. Held ships only once the committing result is final,
which is what a run that may revise a weak result needs: a provisional
boundary that ships is never corrected when the strong tier later
answers for the window (Toshio et al. 2510.25222 Sec. III A).
"""

import decsim.records.windows as window_records


class Eager:
    """Ships every committed boundary, final or provisional."""

    ships_provisional_boundaries = True

    def on_commit(self, window: window_records.Window, *, final: bool) -> bool:
        """Ship."""
        del window
        del final
        return True


class Held:
    """Opt-in: ship only when the committing result is final."""

    ships_provisional_boundaries = False

    def on_commit(self, window: window_records.Window, *, final: bool) -> bool:
        """Ship when final."""
        del window
        return final
