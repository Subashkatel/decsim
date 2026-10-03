"""The tick, the clock domains a yaml prices its cycles on, and yaml paths.

One microsecond is TICKS_PER_MICROSECOND ticks; every duration in the
machine is an integer count of them. A yaml states its costs in cycles
of a named clock domain (XQsim's shape: domain labels with frequencies
over one tick core); ClockSettings hands each component the Clock of the
domain it names, and the component charges its cycles on that clock's
edges. A part that names no clock runs on the machine's
(with_machine_clock).
"""

import dataclasses
import math
import types
from collections.abc import Iterator, Mapping
from typing import Optional, TypeVar

TICKS_PER_MICROSECOND = 1_000_000
# A settings record with a clock field, which a part may leave None.
ClockedSettings = TypeVar("ClockedSettings")
# A clocks section that names no domain.
_NO_DOMAINS = types.MappingProxyType({})


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
    """Refuse a duration the yaml or a decsim.experiments call cannot mean."""
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


def check_capacity_bits(key: str, value) -> None:
    """A memory's capacity in bits, checked where the yaml enters.

    One owner for every memory the yaml sizes: a decoder unit's input
    memory and the two syndrome buffers.
    """
    if value is None:
        return
    if is_whole_count(value):
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
    """A count of at least minimum, read from a yaml section's key."""
    value = section.get(key, default)
    check_whole_count(f"{section_name}.{key}", value, unit, minimum)
    return value


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
    """Whether a yaml value is a whole number of at least minimum.

    A bool is refused though Python counts it an int (bool is a subtype
    of int), so a yaml `true` never stands for one of anything.
    """
    if isinstance(value, bool):
        return False
    if not isinstance(value, int):
        return False
    return value >= minimum


def boolean(
    section: Mapping, section_name: str, key: str, default: bool = False
) -> bool:
    """An on-or-off knob, the default when the yaml is silent."""
    value = section.get(key, default)
    check_boolean(f"{section_name}.{key}", value)
    return value


def check_boolean(name: str, value: object) -> None:
    """An on-or-off knob is True or False.

    The test is the type, because 1 == True and 0 == False would let a
    count stand in for a knob (bool is a subtype of int).
    """
    if isinstance(value, bool):
        return
    raise ValueError(f"{name} must be true or false, got {value!r}")


def finite_number(
    section: Mapping, section_name: str, key: str, default: float
) -> float:
    """A finite number the yaml wrote, the default when it is silent.

    YAML 1.1 loads `1e-3`, a number with no decimal point, as text, so
    text that reads as a number is one; a bool is refused though float
    takes it, since a flag is never a rate or a probability.
    """
    value = section.get(key, default)
    sentence = f"{section_name}.{key} must be a finite number (got {value!r})"
    if isinstance(value, bool):
        raise ValueError(sentence)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(sentence) from None
    if not math.isfinite(number):
        raise ValueError(sentence)
    return number


def is_number(value: object) -> bool:
    """Whether a yaml value is a number; a bool is not one."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def setting_at(sections: Mapping, path: str, reader: str) -> object:
    """The value at a dotted yaml path, a step that is not there refused.

    One lookup serves every reader of a point's resolved sections: a
    sweep axis's section and a whole-value reference. reader names who
    asked.
    """
    value = sections
    walked = ()
    for name in path.split("."):
        if not isinstance(value, Mapping) or name not in value:
            sentence = _missing_setting_sentence(reader, path, walked, value)
            raise ValueError(sentence)
        value = value[name]
        walked += (name,)
    return value


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


class ClockSettings(Mapping):
    """The clock domains of the yaml's `clocks` section: name to megahertz.

    A link, a controller, a decoder engine or a frame prices its cycles
    on the domain it names; more domains (an mK stage, a 4K SFQ decoder)
    are one more entry. Both shipped domains start at LILLIPUT's 250 MHz
    (2108.06569 Table 4). The yaml reader alone reads it: `clock` hands
    each section the domain's Clock, which is what a component charges
    cycles on, and a settings record holds that Clock.
    """

    def __init__(self, megahertz_of: Mapping = _NO_DOMAINS) -> None:
        self._megahertz_of = dict(megahertz_of)

    def __getitem__(self, name: str) -> float:
        return self._megahertz_of[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._megahertz_of)

    def __len__(self) -> int:
        return len(self._megahertz_of)

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
        """The named domain's clock."""
        frequency = self.megahertz(clock)
        return Clock.from_megahertz(frequency)


def _missing_setting_sentence(
    reader: str, path: str, walked: tuple, value
) -> str:
    """Why a path stops where it does: a key that is not there, or a leaf."""
    where = ".".join(walked) or "the yaml"
    if not isinstance(value, Mapping):
        return f"{reader} names {path}, but {where} is {value!r}, not a section"
    names = path.split(".")
    missing = names[len(walked)]
    keys = list(value)
    return (
        f"{reader} names {path}, but {where} has no key {missing}; its keys "
        f"are {keys}"
    )


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
