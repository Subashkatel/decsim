"""When a committed window ships its boundary to the windows after it.

A boundary policy fills the BoundaryPolicy seam (decsim/ports.py). Eager
ships at every commit, provisional or final, which keeps a serial chain
of windows moving. Held ships only once the committing result is final,
which is what a run that may revise a weak result needs: a provisional
boundary that ships is never corrected when the strong tier later
answers for the window (Toshio et al. 2510.25222 Sec. III A).
"""

import dataclasses

import decsim.records.windows as window_records


class Eager:
    """Ships every committed boundary, final or provisional."""

    ships_provisional_boundaries = True

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The eager row, which takes no setting."""

        def build(self) -> "Eager":
            """A policy that ships at every commit."""
            return Eager()

    def on_commit(self, window: window_records.Window, *, final: bool) -> bool:
        """Ship."""
        del window
        del final
        return True


class Held:
    """Opt-in: ship only when the committing result is final."""

    ships_provisional_boundaries = False

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The held row, which takes no setting."""

        def build(self) -> "Held":
            """A policy that ships once the committing result is final."""
            return Held()

    def on_commit(self, window: window_records.Window, *, final: bool) -> bool:
        """Ship when final."""
        del window
        return final
