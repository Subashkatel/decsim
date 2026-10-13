"""A machine written out as YAML, and records made from written values.

dumps writes a machine as `decsim describe` prints it
(experiments/describe_command.py), and document_of gives the same
mapping for json. A part is its record's type and the fields that
differ from its defaults, each written as `decsim params` writes it
(parameters/fields.py), then its site; a wire is `part.port ->
part.port`, with its link's record when it has one. A part that holds a
Python object, a workload's Stim circuit, is written `python`. Nothing
reads the text back: a machine is written in Python, and only its
constants live in YAML (topology/constants.py).

record makes one record from values written as `--set` writes them, a
constants section among them, so a config.py prices a link from its
sourced constants; clock_of reads a frequency's text.
"""

import pathlib
from collections.abc import Mapping
from typing import Any

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
# a part's document when it holds a Python object
PYTHON = "python"
WIRE = "wire"
LINK = "link"
ARROW = "->"
_DEFAULT_FABRIC = fabrics.ReferenceFabric()
_DEFAULT_OBSERVATION = observe_settings.ObservationSettings()


class PythonOnlyError(fields.ParameterError):
    """A value a file cannot write: an object set in Python."""


def dumps(machine: graph.Machine) -> str:
    """The machine as YAML text, each record's changed fields only.

    A part holding a Python object is written `python`.
    """
    document = document_of(machine)
    return yaml.safe_dump(document, sort_keys=False, width=79)


def document_of(machine: graph.Machine) -> dict:
    """The machine as the plain mapping dumps writes, for YAML or json."""
    document = {}
    if machine.clock is not None:
        document["clock"] = fields.Parameter.written(_CLOCK, machine.clock)
    document["parts"] = _part_documents(machine.parts)
    document["wires"] = [_wire_document(wire) for wire in machine.wires]
    if machine.fabric != _DEFAULT_FABRIC:
        document["fabric"] = record_document(machine.fabric)
    if machine.observation != _DEFAULT_OBSERVATION:
        document["observation"] = _fields_document(machine.observation)
    return document


def record(
    record_class: type,
    *sections: Mapping,
    machine_clock: Any = None,
    where: str = "a record made in Python",
    **parameters: Any,
) -> Any:
    """A record of this class, its parameters written as `--set` writes them.

    Each section is a mapping, such as a constants section
    (topology/constants.py), and the keywords come after them: a key is
    a parameter, its unit in its value (fetch_per_round="1 cycle"), and
    a nested mapping a slot's record (algorithm={"type": ...}). A later
    key wins. A time on a part with no clock of its own counts cycles
    of machine_clock. A key that is no parameter is refused, with the
    near ones.
    """
    assignments = []
    for section in (*sections, parameters):
        _add_assignments(section, "", where, assignments)
    applied = tree.built(record_class, assignments, machine_clock)
    _refuse_the_unmatched(assignments, applied, where)
    return applied.settings


def clock_of(written: Any, where: str = "a clock") -> Any:
    """The clock a frequency's text writes, 250MHz; None for none."""
    if written is None:
        return None
    return _CLOCK.value_of(written, None, f"{where}: clock")


def record_document(record: Any) -> dict:
    """A record as YAML: its type, then each field off its default."""
    record_class = type(record)
    name = kinds.unique_name(record_class)
    return {TYPE: name, **_fields_document(record)}


# the machine's clock field, which writes and reads a frequency
_MACHINE_PARAMETERS = fields.parameters_of(graph.Machine)
_CLOCK = _MACHINE_PARAMETERS[fields.CLOCK]


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
