"""Constants: named numbers in a YAML file, each with its source.

    roce:
      round_trip: 2.305us  # Backline 2609.09270 Table III line 1627

A machine file reads one as ${roce.round_trip} (machine_file.py). A
config.py reads the file with load and a value by attribute,
constants.roce.round_trip; a section is a mapping, so it splats into a
record's parameters (**constants.roce, machine_file.record). Every value
read is recorded, and a file holding a value nothing read is refused
(refuse_the_unread), so a misspelt key fails rather than leaving a
number unused, as qecboard's constants were checked
(notes/qecboard_prototype). A value is overridden without editing the
file by `--set constants.roce.round_trip=2.5us`, which load applies
(parameters/in_force.py), and for one sweep cell by an axis of that
name (experiments/sweeps.py, overriding). The comment beside a value is
its source.
"""

import contextlib
import dataclasses
import pathlib
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import yaml

import decsim.parameters.fields as fields
import decsim.parameters.in_force as in_force
import decsim.parameters.units as units

# the first segment that names a constant in an assignment or an axis
PREFIX = "constants"
# a constants file's line: its key, and its value's comment as a source
_SOURCE_LINE = re.compile(r"^(\s*)([A-Za-z0-9_]+):\s*([^#]*?)\s*#\s*(.*)$")
_KEY_LINE = re.compile(r"^(\s*)([A-Za-z0-9_]+):")
# YAML 1.1's implicit booleans and numbers, which 1.2's replace
_YAML_1_1_TAGS = (
    "tag:yaml.org,2002:bool",
    "tag:yaml.org,2002:int",
    "tag:yaml.org,2002:float",
)
# YAML 1.2's core schema: (tag name, pattern, first characters)
_YAML_1_2_RESOLVERS = (
    ("bool", r"^(?:true|True|TRUE|false|False|FALSE)$", "tTfF"),
    ("int", r"^[-+]?[0-9][0-9_]*$", "-+0123456789"),
    (
        "float",
        r"^(?:[-+]?(?:\.[0-9]+|[0-9][0-9_]*(?:\.[0-9_]*)?)(?:[eE][-+]?[0-9]+)?"
        r"|[-+]?\.(?:inf|Inf|INF)"
        r"|\.(?:nan|NaN|NAN))$",
        "-+0123456789.",
    ),
)


class ConstantsError(fields.ParameterError):
    """A constant a file does not have, or one no value read."""


@dataclasses.dataclass
class Constants:
    """Named values, each with its source, and the names read so far."""

    values: dict
    sources: dict
    origin: str
    read: set = dataclasses.field(default_factory=set)

    def value(self, name: str, where: str) -> Any:
        """The constant of this name, recorded as read."""
        if name not in self.values:
            near = units.did_you_mean(name, list(self.values))
            raise ConstantsError(
                f"{where}: no constant {name!r} in {self.origin}.{near}"
            )
        self.read.add(name)
        return self.values[name]

    def unread(self, allowed: Sequence[str] = ()) -> list:
        """The names no value read, but those allowed and their sections'."""
        unread = []
        for name in self.values:
            if name in self.read:
                continue
            if _is_allowed(name, allowed):
                continue
            unread.append(name)
        return unread

    def refuse_the_unread(self, allowed: Sequence[str] = ()) -> None:
        """A constant no value read is refused, by name."""
        unread = self.unread(allowed)
        if unread:
            raise ConstantsError(
                f"{self.origin}: constants no value reads: {unread}. A "
                "misspelt key, or a value the machine forgot"
            )


class Section(Mapping):
    """A section of constants, read by attribute or by key.

    A value read is recorded; a section below is a Section; a key the
    file does not have is refused, with the near ones.
    """

    def __init__(self, constants: Constants, prefix: str = "") -> None:
        self._constants = constants
        self._prefix = prefix

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        return self[key]

    def __getitem__(self, key: str) -> Any:
        name = f"{self._prefix}{key}"
        values = self._constants.values
        if name in values:
            return self._constants.value(name, self._constants.origin)
        below = f"{name}."
        if any(known.startswith(below) for known in values):
            return Section(self._constants, below)
        keys = self._keys()
        near = units.did_you_mean(key, keys)
        raise ConstantsError(f"{self._constants.origin} has no {name!r}.{near}")

    def __iter__(self) -> Iterator[str]:
        keys = self._keys()
        return iter(keys)

    def __len__(self) -> int:
        keys = self._keys()
        return len(keys)

    def _keys(self) -> list:
        """The section's own keys, in the file's order."""
        keys = {}
        for name in self._constants.values:
            if not name.startswith(self._prefix):
                continue
            rest = name.removeprefix(self._prefix)
            key, _, _ = rest.partition(".")
            keys[key] = None
        return list(keys)


def load(path: pathlib.Path) -> Section:
    """A constants file, read by attribute, with --set's overrides applied."""
    constants = read(path)
    return Section(constants)


def read(path: pathlib.Path) -> Constants:
    """A constants file: its values by dotted name, and their sources.

    The assignments in force to constants.<name> replace the values the
    file has, and are recorded as reaching it.
    """
    path = pathlib.Path(path)
    text = path.read_text()
    document = parsed(text)
    values = {}
    if document is not None:
        _flatten(document, "", values)
    sources = _sources(text)
    overrides = in_force.constant_overrides(values, PREFIX)
    for name, assignment in overrides.items():
        values[name] = assignment.written
        sources[name] = f"{assignment.origin} {assignment.text}"
    return Constants(values, sources, str(path))


def parsed(text: str) -> Any:
    """A YAML text read as YAML 1.2 reads its scalars."""
    return yaml.load(text, Loader=_Yaml12Loader)


def held(section: Section) -> Constants:
    """The constants a section reads from."""
    return section._constants


@contextlib.contextmanager
def overriding(section: Section, values: Mapping) -> Iterator[None]:
    """Each named constant set to its value inside; each name must exist.

    A sweep cell's axes over constants.
    """
    constants = held(section)
    saved = {}
    for name, value in values.items():
        constants.value(name, "a swept constant")
        saved[name] = constants.values[name]
        constants.values[name] = value
    try:
        yield
    finally:
        constants.values.update(saved)


def _is_allowed(name: str, allowed: Sequence[str]) -> bool:
    for kept in allowed:
        if name == kept or name.startswith(f"{kept}."):
            return True
    return False


class _Yaml12Loader(yaml.SafeLoader):
    """PyYAML's safe loader with YAML 1.2's booleans and numbers."""


def _yaml_1_2_resolvers() -> dict:
    """The safe loader's implicit resolvers, 1.1's scalars swapped for 1.2's."""
    resolvers = {}
    for first, pairs in yaml.SafeLoader.yaml_implicit_resolvers.items():
        kept = [pair for pair in pairs if pair[0] not in _YAML_1_1_TAGS]
        resolvers[first] = kept
    for tag_name, pattern, firsts in _YAML_1_2_RESOLVERS:
        tag = f"tag:yaml.org,2002:{tag_name}"
        expression = re.compile(pattern)
        for first in firsts:
            pairs = resolvers.setdefault(first, [])
            pairs.append((tag, expression))
    return resolvers


def _decimal_integer(loader: yaml.SafeLoader, node: yaml.Node) -> int:
    """A YAML 1.2 integer: decimal, so 0123 is 123, not 1.1's octal 83."""
    text = loader.construct_scalar(node)
    digits = text.replace("_", "")
    return int(digits, 10)


def _flatten(document: Mapping, prefix: str, values: dict) -> None:
    for key, value in document.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            _flatten(value, f"{name}.", values)
            continue
        values[name] = value


def _sources(text: str) -> dict:
    """Each dotted key's comment, from the lines that carry one.

    A key's section is the keys above it that are indented less.
    """
    sources = {}
    sections = []
    for line in text.splitlines():
        keyed = _KEY_LINE.match(line)
        if keyed is None:
            continue
        indent_text, key = keyed.groups()
        indent = len(indent_text)
        sections = [pair for pair in sections if pair[0] < indent]
        sections.append((indent, key))
        sourced = _SOURCE_LINE.match(line)
        _add_source(sources, sections, sourced)
    return sources


def _add_source(sources: dict, sections: list, sourced: Any) -> None:
    if sourced is None:
        return
    names = [key for _, key in sections]
    dotted = ".".join(names)
    sources[dotted] = sourced.group(4)


_Yaml12Loader.yaml_implicit_resolvers = _yaml_1_2_resolvers()
_Yaml12Loader.add_constructor("tag:yaml.org,2002:int", _decimal_integer)
