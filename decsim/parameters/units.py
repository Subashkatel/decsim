"""Values with units, as gem5's params take them ("1ns", "3GHz", "32KiB").

A value is written once, with its unit, and parsed here into a Quantity
that knows its dimension. The conversions to decsim's internal units
call decsim's own functions, so a value rounds exactly as decsim rounds
it:

- a time becomes ticks with decsim.config.microseconds_to_ticks (half a
  tick rounds up, as gem5's ticks.fromSeconds);
- a time becomes cycles of a decsim Clock by the Clock's own rounding
  (cycles_on);
- a frequency becomes a decsim.config.Clock with Clock.from_megahertz.

"unlimited" and "none" are values of their own, never a silent default.
"""

import dataclasses
import difflib
import fractions
import re
from collections.abc import Iterable
from typing import Union

import decsim.config as config

TIME = "time"
FREQUENCY = "frequency"
CYCLES = "cycles"
BITS = "bits"
RATE = "rate"  # bits per microsecond
DECIBELS = "decibels"
# a cost written as a time or as cycles of the stage's clock
DURATION = (TIME, CYCLES)
UNLIMITED = "unlimited"
NONE = "none"

# unit text -> (dimension, factor to the dimension's base unit); the bases
# are microseconds, megahertz, cycles, bits, bits per microsecond and
# decibels
_UNITS = {
    "ps": (TIME, fractions.Fraction(1, 1_000_000)),
    "ns": (TIME, fractions.Fraction(1, 1000)),
    "us": (TIME, fractions.Fraction(1)),
    "µs": (TIME, fractions.Fraction(1)),
    "ms": (TIME, fractions.Fraction(1000)),
    "s": (TIME, fractions.Fraction(1_000_000)),
    "kHz": (FREQUENCY, fractions.Fraction(1, 1000)),
    "MHz": (FREQUENCY, fractions.Fraction(1)),
    "GHz": (FREQUENCY, fractions.Fraction(1000)),
    "cycle": (CYCLES, fractions.Fraction(1)),
    "cycles": (CYCLES, fractions.Fraction(1)),
    "bit": (BITS, fractions.Fraction(1)),
    "bits": (BITS, fractions.Fraction(1)),
    "byte": (BITS, fractions.Fraction(8)),
    "bytes": (BITS, fractions.Fraction(8)),
    "bits/us": (RATE, fractions.Fraction(1)),
    "bits/ns": (RATE, fractions.Fraction(1000)),
    "Mbps": (RATE, fractions.Fraction(1)),
    "Gbps": (RATE, fractions.Fraction(1000)),
    "dB": (DECIBELS, fractions.Fraction(1)),
}
# a number, with _ separators and an exponent allowed, or a fraction
# p/q, then its unit
_QUANTITY = re.compile(
    r"^\s*([-+]?[0-9][0-9_]*/[0-9]+"
    r"|[-+]?[0-9][0-9_]*(?:\.[0-9_]*)?(?:[eE][-+]?[0-9]+)?)"
    r"\s*([A-Za-zµ/]*)\s*$"
)
_EXAMPLES = {
    TIME: '"157ns" or "1.15us"',
    FREQUENCY: '"250MHz"',
    CYCLES: '"3 cycles"',
    BITS: '"128 bits" or "16 bytes"',
    RATE: '"100Gbps" or "8000 bits/us"',
    DECIBELS: '"20dB"',
}


class UnitError(ValueError):
    """A value whose unit is missing, unknown or of the wrong dimension."""


@dataclasses.dataclass(frozen=True)
class Quantity:
    """A number in its dimension's base unit, and the text it was written as."""

    value: fractions.Fraction
    dimension: str
    written: str

    def ticks(self) -> int:
        """A time as decsim's ticks, rounded as decsim rounds."""
        self._require(TIME)
        return config.microseconds_to_ticks(self.value)

    def microseconds(self) -> float:
        """A time as the float decsim's microsecond fields hold."""
        self._require(TIME)
        return float(self.value)

    def cycles_on(self, clock: config.Clock) -> int:
        """Whole cycles: a count as written, or a time on a decsim Clock.

        A time takes the Clock's own rounding (Clock.cycles_for), the
        rule decsim prices every cycle by.
        """
        if self.dimension == CYCLES:
            return _whole(self.value, self.written)
        ticks = self.ticks()
        return clock.cycles_for(ticks)

    def clock(self) -> config.Clock:
        """A frequency as a decsim clock, its period rounded to ticks."""
        self._require(FREQUENCY)
        return config.Clock.from_megahertz(float(self.value))

    def bits(self) -> int:
        """A size as a whole count of bits."""
        self._require(BITS)
        return _whole(self.value, self.written)

    def bits_per_microsecond(self) -> fractions.Fraction:
        """A rate, exact."""
        self._require(RATE)
        return self.value

    def decibels(self) -> float:
        """A ratio in decibels."""
        self._require(DECIBELS)
        return float(self.value)

    def _require(self, dimension: str) -> None:
        if self.dimension != dimension:
            raise UnitError(
                f"{self.written!r} is a {self.dimension}, not a {dimension}"
            )


def parse(
    text: object,
    dimension: Union[str, Iterable[str]],
    where: str = "",
) -> Union[Quantity, str]:
    """A written value checked against the dimensions its parameter declares.

    Returns UNLIMITED or NONE for those words; anything else must carry a
    unit of an accepted dimension.
    """
    accepted = _accepted(dimension)
    label = _label(where)
    if _is_limit_word(text):
        return text.strip()
    _refuse_unitless(text, accepted, label)
    return _parsed_text(text, accepted, label)


def did_you_mean(name: object, names: Iterable) -> str:
    """The text ' Did you mean a or b?' for the names near name, or ''."""
    written = [str(known) for known in names]
    near = difflib.get_close_matches(str(name), written, n=3)
    if not near:
        return ""
    joined = " or ".join(near)
    return f" Did you mean {joined}?"


def _accepted(dimension: Union[str, Iterable[str]]) -> tuple[str, ...]:
    if isinstance(dimension, str):
        return (dimension,)
    return tuple(dimension)


def _label(where: str) -> str:
    if not where:
        return ""
    return f"{where}: "


def _is_limit_word(text: object) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    return stripped in (UNLIMITED, NONE)


def _refuse_unitless(
    text: object, accepted: tuple[str, ...], label: str
) -> None:
    """Refuse a boolean, or a number written without its unit."""
    wanted = " or ".join(accepted)
    if isinstance(text, bool):
        raise UnitError(f"{label}{text!r} is not a {wanted}")
    if isinstance(text, (int, float)):
        raise _no_unit(text, accepted, label)


def _parsed_text(
    text: object, accepted: tuple[str, ...], label: str
) -> Quantity:
    """A text as a quantity whose unit is one of the accepted dimensions'."""
    written = str(text)
    match = _QUANTITY.match(written)
    if not match:
        wanted = " or ".join(accepted)
        example = _EXAMPLES[accepted[0]]
        raise UnitError(
            f"{label}cannot read {text!r} as a {wanted}, e.g. {example}"
        )
    number_text, unit = match.groups()
    digits = number_text.replace("_", "")
    number = fractions.Fraction(digits)
    if not unit:
        raise _no_unit(text, accepted, label)
    if unit not in _UNITS:
        known = ", ".join(sorted(_UNITS))
        raise UnitError(
            f"{label}{text!r}: unknown unit {unit!r}; known: {known}"
        )
    unit_dimension, factor = _UNITS[unit]
    _check_dimension(unit_dimension, text, accepted, label)
    value = number * factor
    stripped = written.strip()
    return Quantity(value, unit_dimension, stripped)


def _check_dimension(
    dimension: str, shown: object, accepted: tuple[str, ...], label: str
) -> None:
    if dimension in accepted:
        return
    wanted = " or ".join(accepted)
    raise UnitError(
        f"{label}{shown!r} is a {dimension}, but this takes a {wanted}"
    )


def _no_unit(text: object, accepted: tuple[str, ...], label: str) -> UnitError:
    wanted = " or ".join(accepted)
    example = _EXAMPLES[accepted[0]]
    return UnitError(
        f"{label}{text!r} has no unit; write it as a {wanted}, e.g. {example}"
    )


def _whole(value: fractions.Fraction, written: str) -> int:
    if value.denominator != 1 or value < 0:
        raise UnitError(f"{written!r} must be a whole, non-negative count")
    return int(value)
