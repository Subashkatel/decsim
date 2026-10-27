"""When a committed window ships its boundary to the windows after it.

A boundary policy fills the BoundaryPolicy seam (decsim/ports.py). Eager
ships at every commit, provisional or final, which keeps a serial chain
of windows moving. Held ships only once the committing result is final,
which is what a run that may revise a weak result needs: a provisional
boundary that ships is never corrected when the strong tier later
answers for the window (Toshio et al. 2510.25222 Sec. III A). Checked
ships at every commit, as Eager does, and ships again when the result
is final, as Held does; a window that read the provisional boundary
checks it then (windows/input_checks.py). This is SWIPER's speculation
(2412.05115: a predicted boundary, checked when the full decode ends)
with the weak decoder's own answer as the prediction.
"""

import dataclasses

import decsim.records.windows as window_records


class Eager:
    """Ships every committed boundary, final or provisional."""

    ships_provisional_boundaries = True
    rechecks_provisional_boundaries = False

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
    rechecks_provisional_boundaries = False

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


class Checked:
    """Ships every committed boundary, and the final one again to check."""

    ships_provisional_boundaries = True
    rechecks_provisional_boundaries = True

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The checked row, which takes no setting."""

        def build(self) -> "Checked":
            """A policy that ships at every commit and checks the final one."""
            return Checked()

    def on_commit(self, window: window_records.Window, *, final: bool) -> bool:
        """Ship."""
        del window
        del final
        return True
