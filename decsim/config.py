"""The tick, and the clock domains a yaml prices its cycles on.

One microsecond is TICKS_PER_MICROSECOND ticks; every duration in the
machine is an integer count of them. A yaml states its costs in cycles
of a named clock domain (XQsim's shape: domain labels with frequencies
over one tick core); ClockSettings turns cycles into microseconds once,
at load.
"""

import dataclasses
import math
from collections.abc import Mapping

TICKS_PER_MICROSECOND = 1_000_000


def microseconds_to_ticks(microseconds: float) -> int:
    """Round a duration in microseconds to whole ticks."""
    scaled = microseconds * TICKS_PER_MICROSECOND
    rounded = round(scaled)
    return int(rounded)


def microseconds(ticks: int) -> float:
    """A tick count as microseconds, to three decimals."""
    exact = ticks / TICKS_PER_MICROSECOND
    return round(exact, 3)


def format_ticks(ticks: int) -> str:
    """A tick count as the microsecond stamp every log line carries."""
    return f"{ticks / TICKS_PER_MICROSECOND:7.3f} us"


def check_duration(name: str, value: float) -> None:
    """Refuse a duration the yaml or a front call cannot mean."""
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    ticks = microseconds_to_ticks(value)
    if value > 0 and ticks == 0:
        raise ValueError(f"{name} is positive but rounds to zero ticks")


@dataclasses.dataclass(frozen=True)
class ClockSettings:
    """The clock domains of the yaml's `clocks` section: name to megahertz.

    A link, a controller, a decoder engine or a frame prices its cycles
    on the domain it names; more domains (an mK stage, a 4K SFQ decoder)
    are one more entry. Both shipped domains start at LILLIPUT's 250 MHz
    (2108.06569 Table 4).
    """

    megahertz_by_name: Mapping[str, float] = dataclasses.field(
        default_factory=dict
    )

    @classmethod
    def from_yaml(cls, section: Mapping) -> "ClockSettings":
        """The `clocks` section: every value a positive frequency."""
        megahertz_by_name = {}
        for name, megahertz in section.items():
            frequency = float(megahertz)
            if not math.isfinite(frequency) or frequency <= 0:
                raise ValueError(
                    f"clock {name} must be a positive frequency in "
                    f"megahertz, got {megahertz!r}"
                )
            megahertz_by_name[name] = frequency
        return cls(megahertz_by_name)

    def megahertz(self, clock: str) -> float:
        """The named domain's frequency."""
        if clock not in self.megahertz_by_name:
            known = sorted(self.megahertz_by_name)
            raise ValueError(
                f"clock {clock!r} is not a clocks entry; the clocks are {known}"
            )
        return self.megahertz_by_name[clock]

    def microseconds(self, cycles: float, clock: str) -> float:
        """This many cycles of the named domain, in microseconds."""
        frequency = self.megahertz(clock)
        return cycles / frequency
