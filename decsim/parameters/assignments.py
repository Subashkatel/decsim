"""Assignments written as text: `path=value`, from a flag or the environment.

`decsim run run.py --set weak_decoder.unit_count=3` writes one; the
environment variable DECSIM_SET writes several, separated by `;`, and
they come before every --set, so a flag wins (ns-3's precedence: the
command line over the environment over the run file, manual,
Attributes). A value is written as a path's parameter takes it:

    3, 0.25, -1          a number
    true, false          a flag
    10us, 250MHz, 3 cycles, 20dB, 64 bits   a quantity (parameters/units.py)
    none, unlimited      nothing set; no cap
    union_find           a word, or a record's name for path.type
    [a, b, 3]            a list, for a field that holds a tuple
    {type: x, k: v}      a part: path.type=x and path.k=v

A list or a part holds plain values, not lists or parts of its own.
"""

import os
import re
from typing import Any

import decsim.parameters.fields as fields
import decsim.parameters.tree as tree

ENVIRONMENT_VARIABLE = "DECSIM_SET"
FLAG = "--set"
SEPARATOR = ";"
_TRUE = "true"
_FALSE = "false"
_NULL = "null"
_INTEGER = re.compile(r"^[-+]?[0-9][0-9_]*$")
_REAL = re.compile(
    r"^[-+]?(?:[0-9][0-9_]*\.[0-9_]*|\.[0-9]+|[0-9][0-9_]*)"
    r"(?:[eE][-+]?[0-9]+)?$"
)
_QUOTES = ("'", '"')


def from_texts(texts: list, origin: str) -> list:
    """The assignments a list of `path=value` texts write, in order."""
    assignments = []
    for text in texts:
        written = parse(text, origin)
        assignments.extend(written)
    return assignments


def from_environment() -> list:
    """The assignments DECSIM_SET writes, in order; none if it is unset."""
    written = os.environ.get(ENVIRONMENT_VARIABLE, "")
    parts = written.split(SEPARATOR)
    stripped = [part.strip() for part in parts]
    texts = [part for part in stripped if part]
    return from_texts(texts, ENVIRONMENT_VARIABLE)


def every_assignment(flag_texts: list) -> list:
    """DECSIM_SET's assignments, then the flags', which win."""
    from_variable = from_environment()
    from_flags = from_texts(flag_texts, FLAG)
    return [*from_variable, *from_flags]


def parse(text: str, origin: str) -> list:
    """One `path=value` text as its assignments: one, or a part's several."""
    if "=" not in text:
        raise fields.ParameterError(
            f"{origin}: {text!r} is not path=value, e.g. "
            "weak_decoder.unit_count=3"
        )
    written_path, _, written_value = text.partition("=")
    path = written_path.strip()
    value_text = written_value.strip()
    _refuse_an_empty(path, value_text, text, origin)
    if _is_bracketed(value_text, "{", "}"):
        return _part_assignments(path, value_text, origin)
    value = value_of(value_text)
    return [tree.Assignment(path, value, origin)]


def value_of(text: str) -> Any:
    """A written value: a number, a flag, a word, a quantity or a list."""
    stripped = text.strip()
    if _is_bracketed(stripped, "[", "]"):
        inner = stripped[1:-1]
        items = _split_items(inner)
        return [_scalar(item) for item in items]
    return _scalar(stripped)


def _scalar(text: str) -> Any:
    if text in (_TRUE, _FALSE):
        return text == _TRUE
    if text == _NULL:
        return None
    if _INTEGER.match(text):
        digits = text.replace("_", "")
        return int(digits)
    if _REAL.match(text):
        digits = text.replace("_", "")
        return float(digits)
    return _unquoted(text)


def _unquoted(text: str) -> str:
    if len(text) < 2 or text[0] not in _QUOTES:
        return text
    if text[-1] == text[0]:
        return text[1:-1]
    return text


def _part_assignments(path: str, value_text: str, origin: str) -> list:
    """{type: x, k: v} as path.type=x and path.k=v, the type first."""
    inner = value_text[1:-1]
    items = _split_items(inner)
    assignments = []
    for item in items:
        assignment = _part_assignment(path, item, value_text, origin)
        assignments.append(assignment)
    typed = [
        assignment
        for assignment in assignments
        if assignment.pattern.endswith(f".{tree.TYPE}")
    ]
    rest = [assignment for assignment in assignments if assignment not in typed]
    return [*typed, *rest]


def _part_assignment(
    path: str, item: str, value_text: str, origin: str
) -> tree.Assignment:
    """One `key: value` of a part as path.key=value."""
    key, separator, written = item.partition(":")
    if not separator:
        raise fields.ParameterError(
            f"{origin}: {item!r} in {path}={value_text} is not key: value"
        )
    key_text = key.strip()
    child = f"{path}.{key_text}"
    value = value_of(written)
    return tree.Assignment(child, value, origin)


def _split_items(inner: str) -> list:
    parts = inner.split(",")
    stripped = [part.strip() for part in parts]
    return [part for part in stripped if part]


def _is_bracketed(text: str, opening: str, closing: str) -> bool:
    return text.startswith(opening) and text.endswith(closing)


def _refuse_an_empty(
    path: str, value_text: str, text: str, origin: str
) -> None:
    if path and value_text:
        return
    raise fields.ParameterError(
        f"{origin}: {text!r} needs a path and a value, e.g. "
        "weak_decoder.unit_count=3"
    )
