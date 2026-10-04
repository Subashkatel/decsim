"""The tick, and the clock domains a part prices its cycles on.

One microsecond is TICKS_PER_MICROSECOND ticks; every duration in the
machine is an integer count of them. A part states its costs in cycles
of a clock domain (XQsim's shape: domain labels with frequencies over
one tick core); its settings hold that domain's Clock, and the component
charges its cycles on the clock's edges. A part that names no clock runs
on the machine's (with_machine_clock).
"""

import dataclasses
import math
from typing import Optional, TypeVar

TICKS_PER_MICROSECOND = 1_000_000
# A settings record with a clock field, which a part may leave None.
ClockedSettings = TypeVar("ClockedSettings")


def microseconds_to_ticks(microseconds: float) -> int:
    """Round a duration in microseconds to whole ticks."""
    scaled = microseconds * TICKS_PER_MICROSECOND
    rounded = round(scaled)
    return int(rounded)


def ticks_to_microseconds(ticks: int) -> float:
    """Unrounded, so a report or a figure keeps every tick it was given."""
    return ticks / TICKS_PER_MICROSECOND


def format_ticks(ticks: int) -> str:
    """A tick count as the microsecond stamp every log line carries."""
    return f"{ticks / TICKS_PER_MICROSECOND:7.3f} us"


def check_duration(name: str, value: float) -> None:
    """Refuse a duration a caller cannot mean."""
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative number")
    ticks = microseconds_to_ticks(value)
    if value > 0 and ticks == 0:
        raise ValueError(f"{name} is positive but rounds to zero ticks")


def check_microseconds(name: str, value: object) -> None:
    """A card's time in microseconds: a finite number, not negative."""
    is_finite = is_number(value) and math.isfinite(value)
    if is_finite and value >= 0:
        return
    raise ValueError(f"{name} must be finite and not negative (got {value!r})")


def check_capacity_bits(key: str, value: object) -> None:
    """A memory's capacity in bits, or None for an unbounded memory.

    One owner for every memory a settings record sizes: a decoder unit's
    input memory and the two syndrome buffers.
    """
    if value is None:
        return
    if is_whole_count(value):
        return
    raise ValueError(
        f"{key} must be at least one bit, or None for an unbounded "
        f"memory (got {value!r})"
    )


def check_cycles(name: str, cycles: int) -> None:
    """A cycle count is a nonnegative integer, excluding booleans."""
    if not isinstance(cycles, int) or isinstance(cycles, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    if cycles < 0:
        raise ValueError(
            f"{name} must not be negative: cycles must be nonnegative"
        )


def check_whole_count(
    name: str, value: object, unit: str, minimum: int = 1
) -> None:
    """A count is a whole number of its unit, at least minimum."""
    if is_whole_count(value, minimum):
        return
    raise ValueError(
        f"{name} must be a whole number of {unit}, at least {minimum} "
        f"(got {value!r})"
    )


def is_whole_count(value: object, minimum: int = 1) -> bool:
    """Whether a value is a whole number of at least minimum.

    A bool is refused though Python counts it an int (bool is a subtype
    of int), so True never stands for one of anything.
    """
    if isinstance(value, bool):
        return False
    if not isinstance(value, int):
        return False
    return value >= minimum


def check_boolean(name: str, value: object) -> None:
    """An on-or-off knob is True or False.

    The test is the type, because 1 == True and 0 == False would let a
    count stand in for a knob (bool is a subtype of int).
    """
    if isinstance(value, bool):
        return
    raise ValueError(f"{name} must be true or false, got {value!r}")


def is_number(value: object) -> bool:
    """Whether a value is a number; a bool is not one."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclasses.dataclass(frozen=True)
class Clock:
    """One clock domain.

    A component charges a cost of n cycles from the edge at or after the
    tick it stands on, so work started mid-cycle lands on an edge and
    three cycles are three periods of the domain rather than three
    periods measured from an arbitrary instant. That is gem5's Clocked
    (gem5 src/sim/clocked_object.hh lines 174-227):
    clockEdge aligns the current tick to the next edge before adding the
    cycles, and ticksToCycles rounds a span up to whole cycles.
    """

    period_ticks: int

    @classmethod
    def from_megahertz(cls, megahertz: float) -> "Clock":
        """The clock of a frequency, its period rounded to whole ticks.

        gem5's Clock param takes a frequency the same way and keeps its
        period, rounded to ticks (src/python/m5/params/time_params.py
        lines 149-154 and 165-200).
        """
        if not _is_frequency(megahertz):
            raise ValueError(
                "a clock's frequency must be a positive number of "
                f"megahertz, got {megahertz!r}"
            )
        period_microseconds = 1.0 / megahertz
        period_ticks = microseconds_to_ticks(period_microseconds)
        if period_ticks == 0:
            raise ValueError(
                f"a clock at {megahertz} megahertz has a period below one tick"
            )
        return cls(period_ticks)

    def edge(self, cycles: int, now: int) -> int:
        """The tick `cycles` periods after the edge at or after `now`."""
        aligned_cycles = self.cycles_for(now)
        edge_cycles = aligned_cycles + cycles
        return edge_cycles * self.period_ticks

    def ticks_to_edge(self, cycles: int, now: int) -> int:
        """The ticks from `now` to the edge `cycles` periods away."""
        edge = self.edge(cycles, now)
        return edge - now

    def cycles_for(self, ticks: int) -> int:
        """The whole cycles a span of ticks covers; a part cycle counts one."""
        whole_cycles = ticks // self.period_ticks
        remainder = ticks % self.period_ticks
        if remainder == 0:
            return whole_cycles
        return whole_cycles + 1


def with_machine_clock(
    settings: ClockedSettings, machine_clock: Optional[Clock]
) -> ClockedSettings:
    """The settings on their own clock, or on the machine's when they name none.

    A part's clock defaults to the machine's as a gem5 ClockedObject's
    clock domain defaults to its parent's (src/sim/ClockedObject.py:50,
    `Param.ClockDomain(Parent.clk_domain)`).
    """
    if settings.clock is not None:
        return settings
    return dataclasses.replace(settings, clock=machine_clock)


def _is_frequency(value) -> bool:
    """A clock's megahertz: a finite positive number, never a flag or a word.

    Python's bool is a subclass of int (the language reference, "The
    standard type hierarchy"), so a flag would otherwise pass as one
    megahertz; a number written as text is a word, as check_cycles
    treats it.
    """
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    if not math.isfinite(value):
        return False
    return value > 0
