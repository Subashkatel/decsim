"""The tick, and the clock domains a yaml prices its cycles on.

One microsecond is TICKS_PER_MICROSECOND ticks; every duration in the
machine is an integer count of them. A yaml states its costs in cycles
of a named clock domain (XQsim's shape: domain labels with frequencies
over one tick core); ClockSettings hands each component the Clock of the
domain it names, and the component charges its cycles on that clock's
edges.
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
    """Refuse a duration the yaml or a decsim.experiments call cannot mean."""
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    ticks = microseconds_to_ticks(value)
    if value > 0 and ticks == 0:
        raise ValueError(f"{name} is positive but rounds to zero ticks")


@dataclasses.dataclass(frozen=True)
class Clock:
    """One clock domain's period, and the edges its component charges on.

    A component charges a cost of n cycles from the edge at or after the
    tick it stands on, so work started mid-cycle lands on an edge and
    three cycles are three periods of the domain rather than three
    periods measured from an arbitrary instant. That is gem5's Clocked
    (tmp/resources/gem5/src/sim/clocked_object.hh lines 174-227):
    clockEdge aligns the current tick to the next edge before adding the
    cycles, and ticksToCycles rounds a span up to whole cycles.
    """

    period_ticks: int

    def edge(self, cycles: int, now: int) -> int:
        """The tick `cycles` periods after the edge at or after `now`."""
        aligned_cycles = self.cycles_for(now)
        edge_cycles = aligned_cycles + cycles
        return edge_cycles * self.period_ticks

    def cycles_for(self, ticks: int) -> int:
        """The whole cycles a span of ticks covers; a part cycle counts one."""
        whole_cycles = ticks // self.period_ticks
        remainder = ticks % self.period_ticks
        if remainder == 0:
            return whole_cycles
        return whole_cycles + 1


@dataclasses.dataclass(frozen=True)
class ClockSettings:
    """The clock domains of the yaml's `clocks` section: name to megahertz.

    A link, a controller, a decoder engine or a frame prices its cycles
    on the domain it names; more domains (an mK stage, a 4K SFQ decoder)
    are one more entry. Both shipped domains start at LILLIPUT's 250 MHz
    (2108.06569 Table 4). `clock` hands out the domain's Clock, which is
    what a component charges cycles on.
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

    def clock(self, clock: str) -> Clock:
        """The named domain's clock, its period rounded to whole ticks."""
        frequency = self.megahertz(clock)
        period_microseconds = 1.0 / frequency
        period_ticks = microseconds_to_ticks(period_microseconds)
        if period_ticks == 0:
            raise ValueError(
                f"clock {clock!r} at {frequency} megahertz has a period "
                f"below one tick"
            )
        return Clock(period_ticks)
