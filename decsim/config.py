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
import types
from collections.abc import Iterator, Mapping

TICKS_PER_MICROSECOND = 1_000_000
# A clocks section that names no domain.
_NO_DOMAINS = types.MappingProxyType({})


def microseconds_to_ticks(microseconds: float) -> int:
    """Round a duration in microseconds to whole ticks."""
    scaled = microseconds * TICKS_PER_MICROSECOND
    rounded = round(scaled)
    return int(rounded)


def ticks_to_microseconds(ticks: int) -> float:
    """A tick count as a float of microseconds."""
    return ticks / TICKS_PER_MICROSECOND


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


def check_capacity_bits(key: str, value) -> None:
    """A memory's capacity in bits, checked where the yaml enters.

    One owner for every memory the yaml sizes: a decoder unit's input
    memory and the two syndrome buffers.
    """
    if value is None:
        return
    if _is_whole_bit_count(value):
        return
    raise ValueError(
        f"{key} must be at least one bit, or null for an unbounded "
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


def whole_count(
    section: Mapping,
    section_name: str,
    key: str,
    default: int,
    unit: str,
    minimum: int = 1,
) -> int:
    """A count of at least minimum, read from a yaml section's key.

    A bool is refused though Python counts it an int, so true never
    stands for one.
    """
    value = section.get(key, default)
    is_whole = isinstance(value, int) and not isinstance(value, bool)
    if is_whole and value >= minimum:
        return value
    raise ValueError(
        f"{section_name}.{key} must be a whole number of {unit}, at least "
        f"{minimum} (got {value!r})"
    )


def boolean(section: Mapping, section_name: str, key: str) -> bool:
    """An on-or-off knob, off when the yaml is silent.

    The test is the type, because 1 == True and 0 == False would let a
    count stand in for a knob (bool is a subtype of int).
    """
    value = section.get(key, False)
    if isinstance(value, bool):
        return value
    raise ValueError(
        f"{section_name}.{key} must be true or false, got {value!r}"
    )


def is_number(value: object) -> bool:
    """Whether a yaml value is a number; a bool is not one."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclasses.dataclass(frozen=True)
class Clock:
    """One clock domain's period, and the edges its component charges on.

    A component charges a cost of n cycles from the edge at or after the
    tick it stands on, so work started mid-cycle lands on an edge and
    three cycles are three periods of the domain rather than three
    periods measured from an arbitrary instant. That is gem5's Clocked
    (gem5 src/sim/clocked_object.hh lines 174-227):
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


class ClockSettings(Mapping):
    """The clock domains of the yaml's `clocks` section: name to megahertz.

    A link, a controller, a decoder engine or a frame prices its cycles
    on the domain it names; more domains (an mK stage, a 4K SFQ decoder)
    are one more entry. Both shipped domains start at LILLIPUT's 250 MHz
    (2108.06569 Table 4). It is the section's own mapping, so a domain's
    frequency is clocks.<name> as the yaml writes it. `clock` hands out
    the domain's Clock, which is what a component charges cycles on.
    """

    def __init__(self, megahertz_of: Mapping = _NO_DOMAINS) -> None:
        self._megahertz_of = dict(megahertz_of)

    def __getitem__(self, name: str) -> float:
        return self._megahertz_of[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._megahertz_of)

    def __len__(self) -> int:
        return len(self._megahertz_of)

    def __hash__(self) -> int:
        items = self._megahertz_of.items()
        frozen = frozenset(items)
        return hash(frozen)

    @classmethod
    def from_yaml(cls, section: Mapping) -> "ClockSettings":
        """The `clocks` section: every value a positive frequency."""
        megahertz_of = {}
        for name, megahertz in section.items():
            if not _is_frequency(megahertz):
                raise ValueError(
                    f"clock {name} must be a positive frequency in "
                    f"megahertz, got {megahertz!r}"
                )
            megahertz_of[name] = float(megahertz)
        return cls(megahertz_of)

    def megahertz(self, clock: str) -> float:
        """The named domain's frequency."""
        if clock not in self:
            known = sorted(self)
            raise ValueError(
                f"clock {clock!r} is not a clocks entry; the clocks are {known}"
            )
        return self[clock]

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


def _is_whole_bit_count(value) -> bool:
    """A capacity a memory can have: a whole number of bits, never a flag."""
    if value is True or value is False:
        return False
    if not isinstance(value, int):
        return False
    return value >= 1


def _is_frequency(value) -> bool:
    """A clock's megahertz: a finite positive number, never a flag or a word.

    YAML reads `true` as a boolean, and Python's bool is a subclass of
    int (the language reference, "The standard type hierarchy"), so a
    flag would otherwise pass as one megahertz; a quoted number is a
    word, as check_cycles treats it.
    """
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    if not math.isfinite(value):
        return False
    return value > 0
