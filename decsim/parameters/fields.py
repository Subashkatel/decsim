"""A settings record's fields as parameters, the unit moved into the value.

decsim's settings records already list what can be set, and STYLE.md
rule 2 puts each quantity's unit at the end of its field's name. So a
parameter is a field with one change: the unit word leaves the name and
is written with the value, where it is checked and converted exactly as
decsim converts it. qecboard derived its parts' parameters by the same
rule (notes/qecboard_prototype, records.py).

    decsim field                      parameter        written as
    fetch_cycles_per_round: int       fetch_per_round  "1 cycle", or "8ns"
                                                       on the part's clock
    attempt_ticks: int                attempt          "5us"
    round_period_microseconds: float  round_period     "1.1us"
    threshold_decibels: float         threshold        "20dB"
    path_mtu_bytes: int               path_mtu         "1024 bytes"
    bits: Optional[int] (a capacity)  bits             "64 bits", "unlimited"
    input_bits_per_microsecond        input            "100Gbps"
    clock: Optional[Clock]            clock            "250MHz", "none"

A name keeps its unit word when stripping it leaves nothing, a Python
keyword or the name of another field (threshold_cycles beside
threshold). A field counted in logical cycles is a plain count. Every
other field keeps its name and its plain value; a field that holds a
record is a part, addressed by path (parameters/tree.py). A field that
holds anything else, a Stim circuit, is shown and is set in Python only.
"""

import dataclasses
import fractions
import functools
import inspect
import keyword
import pathlib
import types
import typing
from typing import Any, Optional

import decsim.config as config
import decsim.parameters.units as units

# a unit word at the end of a field's name -> the dimension it is written in
UNIT_WORDS = {
    "cycles": units.DURATION,
    "ticks": units.TIME,
    "microseconds": units.TIME,
    "decibels": units.DECIBELS,
    "bytes": units.BITS,
    "bits": units.BITS,
}
# the word before "cycles" that makes the field a count of something else
COUNT_WORDS = ("logical",)
# a field ending so is a rate, exact, in bits per microsecond
RATE_SUFFIX = "_bits_per_microsecond"
PLAIN_TYPES = (bool, int, float, str, pathlib.Path, tuple)
RECORD = "record"
CLOCK = "clock"
PLAIN = "plain"
RATE = "rate"
OPAQUE = "opaque"
_BITS_PER_BYTE = 8
_CAPACITY_FIELD = "bits"


class ParameterError(ValueError):
    """A value a parameter cannot take, named by its path."""


# the default of a field that has none: it must be set
REQUIRED = dataclasses.MISSING


@dataclasses.dataclass(frozen=True)
class Parameter:
    """One field of a record, as it is set and shown.

    kind is a unit word of UNIT_WORDS, CLOCK, RATE, PLAIN, RECORD or
    OPAQUE; plain_type is the field's plain type for PLAIN; is_optional
    says None is a value it takes.
    """

    name: str
    field: str
    kind: str
    default: Any
    plain_type: Optional[type]
    is_optional: bool
    hint: Any = dataclasses.field(default=None, compare=False)

    def value_of(
        self, written: Any, clock: Optional[config.Clock], path: str
    ) -> Any:
        """The field's value for a written one, converted as decsim does.

        clock is the part's own or the machine's (config.with_machine_clock),
        read only when a time is written for a count of cycles.
        """
        if _is_nothing(written):
            return self._nothing(path)
        if self.kind == CLOCK:
            frequency = units.parse(written, units.FREQUENCY, path)
            return frequency.clock()
        if self.kind == RATE:
            rate = units.parse(written, units.RATE, path)
            return rate.bits_per_microsecond()
        if self.kind in UNIT_WORDS:
            return self._quantity_value(written, clock, path)
        return self._unless_opaque(written, path)

    def written(self, value: Any) -> str:
        """The field's value as it would be written to set it."""
        if value is None:
            return self._none_word()
        if self.kind == OPAQUE:
            type_name = _type_name(self.hint)
            return f"<{type_name}>"
        if self.kind == CLOCK:
            return _written_clock(value)
        if self.kind in _WRITERS:
            write = _WRITERS[self.kind]
            return write(value)
        return _written_plain(value)

    def _unless_opaque(self, written: Any, path: str) -> Any:
        if self.kind == OPAQUE:
            type_name = _type_name(self.hint)
            raise ParameterError(
                f"{path} holds a {type_name}, which a run file sets in "
                "Python; it is not written as text"
            )
        return _plain_value(written, self.plain_type, path)

    def _nothing(self, path: str) -> None:
        if self.is_optional:
            return None
        raise ParameterError(f"{path}: takes no {_NOTHING_WORDS}")

    def _none_word(self) -> str:
        if self.field == _CAPACITY_FIELD:
            return units.UNLIMITED
        return units.NONE

    def _quantity_value(
        self, written: Any, clock: Optional[config.Clock], path: str
    ) -> int:
        dimension = UNIT_WORDS[self.kind]
        quantity = units.parse(written, dimension, path)
        if self.kind == "cycles":
            return _cycles(quantity, clock, path)
        if self.kind == "bytes":
            return _whole_bytes(quantity, path)
        convert = getattr(quantity, _CONVERSIONS[self.kind])
        return convert()


@functools.cache
def parameters_of(record_class: type) -> dict:
    """Each field of a record class as a Parameter, by parameter name."""
    hints = typing.get_type_hints(record_class)
    record_fields = dataclasses.fields(record_class)
    field_names = {field.name for field in record_fields}
    by_name = {}
    for field in record_fields:
        hint = hints[field.name]
        parameter = _parameter(field, hint, field_names)
        by_name[parameter.name] = parameter
    return by_name


def is_record(value: Any) -> bool:
    """Whether a value is a settings record (a dataclass instance)."""
    is_class = inspect.isclass(value)
    return dataclasses.is_dataclass(value) and not is_class


def record_hint(hint: Any) -> Any:
    """Optional[X] -> X; anything else as it is."""
    arguments = typing.get_args(hint)
    origin = typing.get_origin(hint)
    if origin not in (typing.Union, types.UnionType):
        return hint
    if types.NoneType not in arguments:
        return hint
    rest = [
        argument for argument in arguments if argument is not types.NoneType
    ]
    if len(rest) == 1:
        return rest[0]
    return typing.Union[tuple(rest)]


def _parameter(
    field: dataclasses.Field, hint: Any, field_names: set
) -> Parameter:
    default = _field_default(field)
    is_optional = _allows_none(hint) or default is None
    plain_type = _plain_type(hint)
    if field.name == CLOCK:
        return Parameter(CLOCK, field.name, CLOCK, default, None, True, hint)
    if field.name.endswith(RATE_SUFFIX):
        name = _stripped_rate_name(field.name, field_names)
        return Parameter(
            name, field.name, RATE, default, None, is_optional, hint
        )
    unit = _unit_word(field.name)
    if unit is not None and _is_unit_field(unit, plain_type):
        name = _stripped_name(field.name, unit, field_names)
        return Parameter(
            name, field.name, unit, default, None, is_optional, hint
        )
    kind = _kind_of(hint, plain_type)
    return Parameter(
        field.name, field.name, kind, default, plain_type, is_optional, hint
    )


def _field_default(field: dataclasses.Field) -> Any:
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return REQUIRED


def _allows_none(hint: Any) -> bool:
    arguments = typing.get_args(hint)
    return types.NoneType in arguments


def _plain_type(hint: Any) -> Optional[type]:
    """bool, int, float, str, pathlib.Path or tuple; None for a part."""
    unwrapped = record_hint(hint)
    origin = typing.get_origin(unwrapped)
    if origin is tuple or unwrapped is tuple:
        return tuple
    is_type = inspect.isclass(unwrapped)
    if is_type and issubclass(unwrapped, pathlib.PurePath):
        return pathlib.Path
    if unwrapped in PLAIN_TYPES:
        return unwrapped
    return None


def _kind_of(hint: Any, plain_type: Optional[type]) -> str:
    """PLAIN, RECORD for a record, a union or a Protocol, else OPAQUE."""
    if plain_type is not None:
        return PLAIN
    unwrapped = record_hint(hint)
    origin = typing.get_origin(unwrapped)
    if dataclasses.is_dataclass(unwrapped) or origin is typing.Union:
        return RECORD
    if getattr(unwrapped, "_is_protocol", False):
        return RECORD
    return OPAQUE


def _type_name(hint: Any) -> str:
    unwrapped = record_hint(hint)
    return getattr(unwrapped, "__name__", str(unwrapped))


def _unit_word(field_name: str) -> Optional[str]:
    """The field's last unit word, unless it counts logical cycles."""
    words = field_name.split("_")
    positions = [
        position for position, word in enumerate(words) if word in UNIT_WORDS
    ]
    if not positions:
        return None
    position = max(positions)
    before = words[:position]
    if before and before[-1] in COUNT_WORDS:
        return None
    return words[position]


def _is_unit_field(unit: str, plain_type: Optional[type]) -> bool:
    """A float is a quantity only in microseconds or decibels."""
    if unit in ("microseconds", "decibels"):
        return True
    return plain_type is int


def _stripped_name(field_name: str, unit: str, field_names: set) -> str:
    words = field_name.split("_")
    position = len(words) - 1 - words[::-1].index(unit)
    rest = words[:position] + words[position + 1 :]
    stripped = "_".join(rest)
    if not stripped or keyword.iskeyword(stripped):
        return field_name
    if stripped in field_names:
        return field_name
    return stripped


def _stripped_rate_name(field_name: str, field_names: set) -> str:
    cut = len(RATE_SUFFIX)
    stripped = field_name[:-cut]
    if not stripped or stripped in field_names:
        return field_name
    return stripped


def _is_nothing(written: Any) -> bool:
    if written is None:
        return True
    return written in (units.NONE, units.UNLIMITED)


def _cycles(
    quantity: units.Quantity, clock: Optional[config.Clock], path: str
) -> int:
    if quantity.dimension == units.CYCLES:
        return quantity.cycles_on(clock)
    if clock is None:
        raise ParameterError(
            f"{path}: {quantity.written} is a time, and neither the part "
            "nor the machine has a clock to count it on; write it in "
            "cycles, or set a clock"
        )
    return quantity.cycles_on(clock)


def _whole_bytes(quantity: units.Quantity, path: str) -> int:
    bits = quantity.bits()
    if bits % _BITS_PER_BYTE:
        raise ParameterError(
            f"{path}: {quantity.written} is no whole number of bytes"
        )
    return bits // _BITS_PER_BYTE


def _plain_value(written: Any, plain_type: type, path: str) -> Any:
    """A plain field's value: its type as written, a list as a tuple."""
    if plain_type in _CONVERTERS:
        convert = _CONVERTERS[plain_type]
        return convert(written, path)
    is_of_type = _TYPE_TESTS[plain_type]
    if is_of_type(written):
        return written
    type_name = plain_type.__name__
    raise ParameterError(f"{path}: {written!r} is not of type {type_name}")


def _tuple_value(written: Any, path: str) -> tuple:
    if not isinstance(written, list):
        raise ParameterError(f"{path}: {written!r} is not a list [a, b, ...]")
    return tuple(written)


def _path_value(written: Any, path: str) -> pathlib.Path:
    del path
    text = str(written)
    return pathlib.Path(text)


def _float_value(written: Any, path: str) -> float:
    """A number, a whole one too, as the float the field holds."""
    if _is_flag(written) or not isinstance(written, (int, float)):
        raise ParameterError(f"{path}: {written!r} is not a number")
    return float(written)


def _is_flag(value: Any) -> bool:
    return isinstance(value, bool)


def _is_whole_number(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_text(value: Any) -> bool:
    return isinstance(value, str)


def _exact(number: float) -> str:
    """The shortest text that reads back as the same number, less any ".0"."""
    text = repr(float(number))
    if text.endswith(".0"):
        return text[:-2]
    return text


def _written_cycles(count: int) -> str:
    if count == 1:
        return f"{count} cycle"
    return f"{count} cycles"


def _written_ticks(ticks: int) -> str:
    microseconds = fractions.Fraction(ticks, config.TICKS_PER_MICROSECOND)
    return _written_time(microseconds)


def _written_microseconds(microseconds: float) -> str:
    text = repr(microseconds)
    exact = fractions.Fraction(text)
    return _written_time(exact)


def _written_time(microseconds: fractions.Fraction) -> str:
    """A time in the largest of us, ns and ps it is at least one of."""
    for unit, per_microsecond in _TIME_UNITS:
        count = microseconds * per_microsecond
        if count >= 1:
            text = _exact_fraction(count)
            return f"{text}{unit}"
    return "0us"


def _exact_fraction(number: fractions.Fraction) -> str:
    if number.denominator == 1:
        return str(number.numerator)
    return _exact(number)


def _written_rate(bits_per_microsecond: fractions.Fraction) -> str:
    text = _exact_fraction(bits_per_microsecond)
    return f"{text} bits/us"


def _written_decibels(decibels: float) -> str:
    text = _exact(decibels)
    return f"{text}dB"


def _written_bytes(count: int) -> str:
    return f"{count} bytes"


def _written_bits(count: int) -> str:
    return f"{count} bits"


def _written_clock(clock: config.Clock) -> str:
    megahertz = config.TICKS_PER_MICROSECOND / clock.period_ticks
    text = _exact(megahertz)
    return f"{text}MHz"


def _written_plain(value: Any) -> str:
    if isinstance(value, bool):
        text = str(value)
        return text.lower()
    if isinstance(value, tuple):
        items = [_written_plain(item) for item in value]
        joined = ", ".join(items)
        return f"[{joined}]"
    if isinstance(value, str):
        return _written_text(value)
    if isinstance(value, pathlib.Path):
        return str(value)
    return repr(value)


def _written_text(text: str) -> str:
    """A text as written, quoted where it would read as another value."""
    if text in _WORDS or not text:
        return repr(text)
    first = text[0]
    if first in _OTHER_VALUE_STARTS:
        return repr(text)
    return text


_NOTHING_WORDS = f"{units.NONE} or {units.UNLIMITED}"
# texts an assignment reads as something other than text, and the first
# characters of a number, a list, a part or a quoted text
_WORDS = ("true", "false", "null", units.NONE, units.UNLIMITED)
_OTHER_VALUE_STARTS = "0123456789+-.[{'\""
# a plain type -> the function that makes a written value one
_CONVERTERS = {
    tuple: _tuple_value,
    pathlib.Path: _path_value,
    float: _float_value,
}
# a plain type -> the test a written value of it passes as it is
_TYPE_TESTS = {bool: _is_flag, int: _is_whole_number, str: _is_text}
_TIME_UNITS = (("us", 1), ("ns", 1000), ("ps", 1_000_000))
# a unit word -> the Quantity method that gives decsim's value
_CONVERSIONS = {
    "ticks": "ticks",
    "microseconds": "microseconds",
    "decibels": "decibels",
    "bits": "bits",
}
_WRITERS = {
    "cycles": _written_cycles,
    "ticks": _written_ticks,
    "microseconds": _written_microseconds,
    "decibels": _written_decibels,
    "bytes": _written_bytes,
    "bits": _written_bits,
    RATE: _written_rate,
}
