"""A machine written in YAML: its parts by name, its wires, its fabric.

    clock: 250MHz
    parts:
      qpu: {type: QpuSettings, distance: 5, round_period: 1us}
      uf:
        type: DecoderPoolSettings
        placed: fridge
        algorithm: {type: UnionFindDecoder, weight_step: 0.1}
      ...
    wires:
      - qpu.readout -> controller.readout
      - wire: uf.corrections -> frame.corrections
        link: {type: ClockedLink, latency: 41 cycles}
    fabric:
      type: FridgeHostFabric
      cable: {type: RoundTripLeg, round_trip: ${roce.round_trip}, ...}

A part is its record's type and the fields that differ from its
defaults, each written as `decsim params` writes it (parameters/
fields.py), so a file holds only what a machine changes. A part that
holds a Python object, a workload's Stim circuit, is written `python`
and given in Python (load's python_parts), in its place.

`${name}` reads a constant from a constants file, a YAML of named
numbers whose comments are their sources:

    roce:
      round_trip: 2.305us  # Backline 2609.09270 Table III line 1627

A constant the machine file never reads is refused, so a misspelt name
in either file fails at once rather than leaving a number unused.
Scalars read as YAML 1.2 does: `off` is a word, 5e-4 a number, as
qecboard read them (notes/qecboard_prototype, constants.py).
"""

import dataclasses
import pathlib
import re
from collections.abc import Mapping
from typing import Any, Optional

import yaml

import decsim.observe.settings as observe_settings
import decsim.parameters.fields as fields
import decsim.parameters.kinds as kinds
import decsim.parameters.tree as tree
import decsim.parameters.units as units
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as graph

TYPE = tree.TYPE
PLACED = "placed"
# a part's document when Python gives the part
PYTHON = "python"
WIRE = "wire"
LINK = "link"
ARROW = "->"
# a constant read in a value: ${section.name}
CONSTANT = re.compile(r"\$\{([A-Za-z0-9_.]+)\}")
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
_DEFAULT_FABRIC = fabrics.ReferenceFabric()
_DEFAULT_OBSERVATION = observe_settings.ObservationSettings()


class PythonOnlyError(fields.ParameterError):
    """A value a file cannot write: an object set in Python."""


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
            raise fields.ParameterError(
                f"{where}: no constant {name!r} in {self.origin}.{near}"
            )
        self.read.add(name)
        return self.values[name]

    def refuse_the_unread(self) -> None:
        """A constant no value read is refused, by name."""
        unread = [name for name in self.values if name not in self.read]
        if unread:
            raise fields.ParameterError(
                f"{self.origin}: constants no value reads: {unread}"
            )


def load(
    path: pathlib.Path,
    constants_path: Optional[pathlib.Path] = None,
    python_parts: Optional[Mapping] = None,
) -> graph.Machine:
    """The machine a YAML file writes, its constants read from a file."""
    path = pathlib.Path(path)
    text = path.read_text()
    constants = None
    if constants_path is not None:
        constants = read_constants(constants_path)
    machine = loads(text, constants, python_parts, str(path))
    if constants is not None:
        constants.refuse_the_unread()
    return machine


def loads(
    text: str,
    constants: Optional[Constants] = None,
    python_parts: Optional[Mapping] = None,
    origin: str = "a machine file",
) -> graph.Machine:
    """The machine a YAML text writes.

    python_parts gives, by name, the records the text leaves to Python;
    a part both give is refused.
    """
    document = _parsed(text)
    resolved = _with_constants(document, constants, origin)
    clock_text = resolved.get("clock")
    clock = _clock(clock_text, origin)
    part_documents = resolved.get("parts", {})
    given = python_parts or {}
    parts = _parts(part_documents, given, clock, origin)
    wire_documents = resolved.get("wires", [])
    wires = [_wire(written, clock, origin) for written in wire_documents]
    fabric_document = resolved.get("fabric")
    fabric = _fabric(fabric_document, clock, origin)
    observation_document = resolved.get("observation")
    observation = _observation(observation_document, clock, origin)
    return graph.Machine(
        clock=clock,
        parts=tuple(parts),
        wires=tuple(wires),
        fabric=fabric,
        observation=observation,
    )


def dumps(machine: graph.Machine) -> str:
    """The machine as YAML text, each record's changed fields only.

    A part holding a Python object is written `python`, left for load's
    python_parts.
    """
    document = {}
    if machine.clock is not None:
        document["clock"] = fields.Parameter.written(_CLOCK, machine.clock)
    document["parts"] = _part_documents(machine.parts)
    document["wires"] = [_wire_document(wire) for wire in machine.wires]
    if machine.fabric != _DEFAULT_FABRIC:
        document["fabric"] = record_document(machine.fabric)
    if machine.observation != _DEFAULT_OBSERVATION:
        document["observation"] = _fields_document(machine.observation)
    return yaml.safe_dump(document, sort_keys=False, width=79)


def read_constants(path: pathlib.Path) -> Constants:
    """A constants file: its values by dotted name, and their sources."""
    path = pathlib.Path(path)
    text = path.read_text()
    document = _parsed(text)
    values = {}
    _flatten(document, "", values)
    sources = _sources(text)
    return Constants(values, sources, str(path))


def record_of(document: Mapping, clock: Any, where: str) -> Any:
    """The record a document writes: its type, then each field by path.

    Every field is set as `--set` sets it (parameters/tree.py), so a
    field none of the record's parameters is refused, with the near ones.
    """
    if TYPE not in document:
        raise fields.ParameterError(f"{where}: a record names its type")
    record_class = _record_class(document[TYPE], where)
    assignments = []
    _add_assignments(document, "", where, assignments)
    applied = tree.built(record_class, assignments, clock)
    _refuse_the_unmatched(assignments, applied, where)
    return applied.settings


def record_document(record: Any) -> dict:
    """A record as YAML: its type, then each field off its default."""
    record_class = type(record)
    name = unique_name(record_class)
    return {TYPE: name, **_fields_document(record)}


def unique_name(record_class: type) -> str:
    """The first of a record's names no other record has, else its path."""
    offered = kinds.every_record_class()
    for name in kinds.names_of(record_class):
        matches = kinds.records_named(name, offered)
        if len(matches) == 1:
            return name
    return kinds.import_path(record_class)


def _parsed(text: str) -> Any:
    return yaml.load(text, Loader=_Yaml12Loader)


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


_Yaml12Loader.yaml_implicit_resolvers = _yaml_1_2_resolvers()
_Yaml12Loader.add_constructor("tag:yaml.org,2002:int", _decimal_integer)
# the machine's clock field, which writes and reads a frequency
_MACHINE_PARAMETERS = fields.parameters_of(graph.Machine)
_CLOCK = _MACHINE_PARAMETERS[fields.CLOCK]


def _with_constants(
    value: Any, constants: Optional[Constants], where: str
) -> Any:
    """The document with every ${name} read from the constants."""
    if isinstance(value, Mapping):
        return _mapping_with_constants(value, constants, where)
    if isinstance(value, list):
        return [_with_constants(item, constants, where) for item in value]
    if not isinstance(value, str):
        return value
    return _substituted(value, constants, where)


def _mapping_with_constants(
    mapping: Mapping, constants: Optional[Constants], where: str
) -> dict:
    substituted = {}
    for key, item in mapping.items():
        substituted[key] = _with_constants(item, constants, f"{where}: {key}")
    return substituted


def _substituted(text: str, constants: Optional[Constants], where: str) -> Any:
    """A whole ${name} is the constant's value; within a text, its text."""
    found = CONSTANT.findall(text)
    if not found:
        return text
    if constants is None:
        raise fields.ParameterError(
            f"{where}: {text} reads a constant and no constants file is given"
        )
    whole = CONSTANT.fullmatch(text)
    if whole is not None:
        name = whole.group(1)
        return constants.value(name, where)
    replace = _ConstantText(constants, where)
    return CONSTANT.sub(replace, text)


@dataclasses.dataclass
class _ConstantText:
    """A ${name} within a text, replaced by the constant's own text."""

    constants: Constants
    where: str

    def __call__(self, found: re.Match) -> str:
        name = found.group(1)
        value = self.constants.value(name, self.where)
        return str(value)


def _clock(written: Any, origin: str) -> Any:
    if written is None:
        return None
    return _CLOCK.value_of(written, None, f"{origin}: clock")


def _parts(
    documents: Mapping, python_parts: Mapping, clock: Any, origin: str
) -> list:
    """Each part in the file's order, a `python` one from python_parts.

    A part given in Python that the file does not name comes last.
    """
    parts = []
    for name, document in documents.items():
        part = _part(name, document, python_parts, clock, origin)
        parts.append(part)
    for name, written in python_parts.items():
        if name not in documents:
            given = _given_part(name, written)
            parts.append(given)
    return parts


def _part(
    name: str, document: Any, python_parts: Mapping, clock: Any, origin: str
) -> graph.Part:
    """One part: its record's type and fields, and its site."""
    if document == PYTHON:
        return _python_part(name, python_parts, origin)
    fields_document = dict(document)
    placed = fields_document.pop(PLACED, None)
    settings = record_of(fields_document, clock, f"{origin}: parts.{name}")
    return graph.Part(name, settings, placed)


def _python_part(name: str, python_parts: Mapping, origin: str) -> graph.Part:
    if name not in python_parts:
        raise fields.ParameterError(
            f"{origin}: parts.{name} is given in Python, and python_parts "
            "has no part of that name"
        )
    written = python_parts[name]
    return _given_part(name, written)


def _given_part(name: str, written: Any) -> graph.Part:
    """A part given in Python: a record, or a Part for its site too."""
    placed = getattr(written, PLACED, None)
    settings = getattr(written, "settings", written)
    return graph.Part(name, settings, placed)


def _record_class(name: str, where: str) -> type:
    """The record class a name stands for, among every record decsim has."""
    offered = kinds.every_record_class()
    matches = kinds.records_named(name, offered)
    by_path = [
        record_class
        for record_class in offered
        if kinds.import_path(record_class) == name
    ]
    found = matches or by_path
    if len(found) == 1:
        return found[0]
    names = kinds.every_name(offered)
    near = units.did_you_mean(name, names)
    raise fields.ParameterError(
        f"{where}: {len(matches)} records are named {name!r}.{near} "
        "`decsim info NAME` shows one"
    )


def _add_assignments(
    document: Mapping, prefix: str, where: str, assignments: list
) -> None:
    """Each leaf of a document as path=value, a nested type at path.type."""
    for key, value in document.items():
        path = f"{prefix}{key}"
        if prefix == "" and key == TYPE:
            continue
        if isinstance(value, Mapping):
            _add_assignments(value, f"{path}.", where, assignments)
            continue
        assignment = tree.Assignment(path, value, where)
        assignments.append(assignment)


def _refuse_the_unmatched(
    assignments: list, applied: tree.Applied, where: str
) -> None:
    for position, assignment in enumerate(assignments):
        if position in applied.matched:
            continue
        settings = tree.settings_of(applied.settings)
        paths = [setting.path for setting in settings]
        near = units.did_you_mean(assignment.pattern, paths)
        raise fields.ParameterError(
            f"{where}: {assignment.pattern} is no parameter.{near}"
        )


def _wire(written: Any, clock: Any, origin: str) -> graph.Wire:
    """A wire: `a.port -> b.port`, or that and a link under wire and link."""
    text = written
    link = None
    if isinstance(written, Mapping):
        text = written[WIRE]
        link = record_of(written[LINK], clock, f"{origin}: {text}")
    written_text = str(text)
    source, arrow, target = written_text.partition(ARROW)
    if not arrow:
        raise fields.ParameterError(
            f"{origin}: {text!r} is a wire written `part.port -> part.port`"
        )
    source_port = source.strip()
    target_port = target.strip()
    return graph.Wire(source_port, target_port, link)


def _fabric(document: Any, clock: Any, origin: str) -> Any:
    if document is None:
        return _DEFAULT_FABRIC
    return record_of(document, clock, f"{origin}: fabric")


def _observation(document: Any, clock: Any, origin: str) -> Any:
    if document is None:
        return _DEFAULT_OBSERVATION
    name = unique_name(observe_settings.ObservationSettings)
    typed = {TYPE: name, **document}
    return record_of(typed, clock, f"{origin}: observation")


def _part_documents(parts: tuple) -> tuple:
    """Each part's document; `python` for a part given in Python."""
    documents = {}
    for part in parts:
        try:
            document = record_document(part.settings)
        except PythonOnlyError:
            documents[part.name] = PYTHON
            continue
        if part.placed is not None:
            document[PLACED] = part.placed
        documents[part.name] = document
    return documents


def _wire_document(wire: graph.Wire) -> Any:
    text = f"{wire.source} {ARROW} {wire.target}"
    if wire.link is None:
        return text
    link = record_document(wire.link)
    return {WIRE: text, LINK: link}


def _fields_document(record: Any) -> dict:
    """Each field whose value differs from its record's default."""
    record_class = type(record)
    parameters = fields.parameters_of(record_class)
    document = {}
    for parameter in parameters.values():
        value = getattr(record, parameter.field)
        if _is_default(parameter, value):
            continue
        document[parameter.name] = _value_document(parameter, value)
    return document


def _is_default(parameter: fields.Parameter, value: Any) -> bool:
    if parameter.default is fields.REQUIRED:
        return False
    return value == parameter.default


def _value_document(parameter: fields.Parameter, value: Any) -> Any:
    """A field's value as YAML: a record nested, a plain value as itself."""
    is_part = parameter.kind == fields.RECORD
    if is_part and fields.is_record(value):
        return record_document(value)
    if _is_python_only(parameter, value):
        raise PythonOnlyError(f"{parameter.name} holds a Python object")
    is_plain = parameter.kind == fields.PLAIN
    if is_plain and value is not None:
        return _plain_document(value)
    return parameter.written(value)


def _is_python_only(parameter: fields.Parameter, value: Any) -> bool:
    if parameter.kind == fields.OPAQUE:
        return True
    return _holds_records(value)


def _holds_records(value: Any) -> bool:
    if not isinstance(value, tuple):
        return False
    return any(fields.is_record(item) for item in value)


def _plain_document(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain_document(item) for item in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


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
