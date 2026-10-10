"""Paths over a run's settings, and assignments applied along them.

Every record in a machine's settings is a part with a path: the field
names from the root, dotted (weak_decoder.engine), a tuple's records
indexed (levels[0]). A parameter's path is its part's path and its name
(weak_decoder.engine.fetch_per_round, parameters/fields.py). A part's
type is a parameter too: path.type=name puts another record in the slot
(parameters/kinds.py) and path.type=none empties an optional one, as
OMNeT++ sets a `like` slot's typename from its ini file (simulation
manual §3.11).

An assignment names one path or many: `*` stands for one name and `**`
for any number of them, OMNeT++'s ini patterns (§10.3). Assignments
apply in order, a later one winning where two set one value, as ns-3's
command line wins over its defaults (manual, Attributes). Each record
an assignment reaches is built again, so decsim's own checks run, and a
refusal names the path. Any value a run file wrote can be set again,
but for the paths the root names in SET_IN_THE_RUN_FILE: the run file
built other values from them (the workload's circuit from the
distance), so a pattern naming one is refused, and `*` and `**` pass
them by.
"""

import dataclasses
import functools
import re
from collections.abc import Sequence
from typing import Any, Optional

import decsim.config as config
import decsim.parameters.fields as fields
import decsim.parameters.kinds as kinds
import decsim.parameters.units as units

TYPE = "type"
ANY_DEPTH = "**"
WILDCARD = "*"
# the root's class attribute naming the paths set in the run file only
RUN_FILE_PATHS = "SET_IN_THE_RUN_FILE"
_ROOT = "the machine"


@dataclasses.dataclass(frozen=True)
class Assignment:
    """One path pattern and the value written for it; origin says where."""

    pattern: str
    written: Any
    origin: str

    @property
    def text(self) -> str:
        """The assignment as it is written."""
        return f"{self.pattern}={self.written}"


@dataclasses.dataclass(frozen=True)
class Setting:
    """One settable path of a machine, its value as written, its default."""

    path: str
    written: str
    default: Optional[str]


@dataclasses.dataclass(frozen=True)
class Applied:
    """New settings, the origin of each path an assignment set, the hits.

    matched holds the position of every assignment that reached a path.
    """

    settings: Any
    origins: dict
    matched: frozenset


def apply(root: Any, assignments: Sequence[Assignment]) -> Applied:
    """The settings with the assignments applied, the later ones winning."""
    run_file_paths = getattr(root, RUN_FILE_PATHS, ())
    walk = _Walk(assignments, run_file_paths)
    clock = getattr(root, fields.CLOCK, None)
    settings = walk.node(root, (), clock, is_root=True)
    matched = frozenset(walk.matched)
    return Applied(settings, dict(walk.origins), matched)


def settings_of(root: Any) -> list:
    """Every settable path of the settings, in field order, with its value."""
    found = []
    _list_node(root, (), found)
    return found


def is_set_in_the_run_file(root: Any, path: str) -> bool:
    """Whether a path lies under one the root sets in the run file only."""
    run_file_paths = getattr(root, RUN_FILE_PATHS, ())
    segments = path.split(".")
    return _is_under_any(segments, run_file_paths)


def path_text(segments: Sequence[str]) -> str:
    """Segments as a dotted path; the root as "the machine"."""
    if not segments:
        return _ROOT
    return ".".join(segments)


def matches(pattern: str, segments: Sequence[str]) -> bool:
    """Whether a pattern reaches the path these segments spell."""
    pattern_segments = pattern.split(".")
    return _matches(tuple(pattern_segments), tuple(segments))


class _Walk:
    """One application of assignments: the hits and each set path's origin."""

    def __init__(
        self, assignments: Sequence[Assignment], run_file_paths: Sequence[str]
    ) -> None:
        self.assignments = tuple(assignments)
        self.run_file_paths = tuple(run_file_paths)
        self.matched = set()
        self.origins = {}

    def node(
        self,
        record: Any,
        segments: tuple,
        machine_clock: Optional[config.Clock],
        is_root: bool = False,
    ) -> Any:
        """One record with its own and its parts' assignments applied.

        A part with no clock of its own counts its cycles on the
        machine's; the root's clock is the machine's.
        """
        record_class = type(record)
        parameters = fields.parameters_of(record_class)
        changes = {}
        own_clock = self._clock(record, parameters, segments, changes)
        clock = own_clock or machine_clock
        if is_root:
            machine_clock = clock
        for parameter in parameters.values():
            value = getattr(record, parameter.field)
            child = (*segments, parameter.name)
            new_value = self._value(
                parameter, value, child, clock, machine_clock
            )
            if new_value is not value:
                changes[parameter.field] = new_value
        if not changes:
            return record
        return _rebuilt(record, changes, segments)

    def last_for(self, segments: tuple) -> Optional[Assignment]:
        """The last assignment reaching this path; every one is a hit."""
        position = self._last_position(segments)
        if position is None:
            return None
        return self.assignments[position]

    def _last_position(self, segments: tuple) -> Optional[int]:
        if _is_under_any(segments, self.run_file_paths):
            self._refuse_a_run_file_path(segments)
            return None
        found = None
        for position, assignment in enumerate(self.assignments):
            if matches(assignment.pattern, segments):
                self.matched.add(position)
                found = position
        return found

    def _refuse_a_run_file_path(self, segments: tuple) -> None:
        """A pattern naming a run-file path is refused; a wildcard passes."""
        for assignment in self.assignments:
            if WILDCARD in assignment.pattern:
                continue
            if not matches(assignment.pattern, segments):
                continue
            path = path_text(segments)
            raise fields.ParameterError(
                f"{path} is set in the run file, which built the "
                "workload's circuit with it; an assignment would leave the "
                "circuit behind. Change it there (e.g. the run file's "
                "distances); stage 6 builds the circuit from the "
                "parameters, and then it is set by path"
            )

    def _clock(
        self, record: Any, parameters: dict, segments: tuple, changes: dict
    ) -> Optional[config.Clock]:
        """The record's own clock, as assigned or as it stands."""
        parameter = parameters.get(fields.CLOCK)
        if parameter is None:
            return None
        clock = getattr(record, fields.CLOCK)
        child = (*segments, fields.CLOCK)
        assignment = self.last_for(child)
        if assignment is None:
            return clock
        path = path_text(child)
        new_clock = parameter.value_of(assignment.written, None, path)
        self.origins[path] = assignment.origin
        changes[fields.CLOCK] = new_clock
        return new_clock

    def _value(
        self,
        parameter: fields.Parameter,
        value: Any,
        segments: tuple,
        clock: Optional[config.Clock],
        machine_clock: Optional[config.Clock],
    ) -> Any:
        if parameter.kind == fields.CLOCK:
            return value
        if parameter.kind == fields.RECORD:
            return self._slot(parameter, value, segments, machine_clock)
        if _holds_records(value):
            return self._parts(value, segments, machine_clock)
        assignment = self.last_for(segments)
        if assignment is None:
            return value
        path = path_text(segments)
        self.origins[path] = assignment.origin
        return parameter.value_of(assignment.written, clock, path)

    def _slot(
        self,
        parameter: fields.Parameter,
        value: Any,
        segments: tuple,
        machine_clock: Optional[config.Clock],
    ) -> Any:
        """A part's slot: its type as assigned, then its own assignments."""
        if isinstance(value, tuple):
            return self._parts(value, segments, machine_clock)
        choice = self._type_choice(segments)
        if choice is not None:
            value = self._chosen(
                parameter, value, choice, segments, machine_clock
            )
        if not fields.is_record(value):
            return value
        return self.node(value, segments, machine_clock)

    def _parts(
        self,
        parts: tuple,
        segments: tuple,
        machine_clock: Optional[config.Clock],
    ) -> tuple:
        """A tuple's records, each by its index; any other tuple as it is."""
        new_parts = []
        for index, part in enumerate(parts):
            name = f"{segments[-1]}[{index}]"
            child = (*segments[:-1], name)
            new_part = part
            if fields.is_record(part):
                new_part = self.node(part, child, machine_clock)
            new_parts.append(new_part)
        unchanged = all(
            new is old for new, old in zip(new_parts, parts, strict=True)
        )
        if unchanged:
            return parts
        return tuple(new_parts)

    def _type_choice(self, segments: tuple) -> Optional[Assignment]:
        """The later of path.type=... and path=... for a slot."""
        typed = (*segments, TYPE)
        by_type = self._last_position(typed)
        whole = self._last_position(segments)
        positions = [
            position for position in (by_type, whole) if position is not None
        ]
        if not positions:
            return None
        latest = max(positions)
        return self.assignments[latest]

    def _chosen(
        self,
        parameter: fields.Parameter,
        value: Any,
        choice: Assignment,
        segments: tuple,
        machine_clock: Optional[config.Clock],
    ) -> Any:
        """The slot's record as its assigned type chooses it, or none."""
        path = path_text(segments)
        self.origins[f"{path}.{TYPE}"] = choice.origin
        if choice.written in (None, units.NONE):
            return parameter.value_of(None, None, path)
        name = str(choice.written)
        record_class = kinds.record_class_named(name, parameter.hint, path)
        return self._constructed(record_class, value, segments, machine_clock)

    def _constructed(
        self,
        record_class: type,
        old: Any,
        segments: tuple,
        machine_clock: Optional[config.Clock],
    ) -> Any:
        """A record of the chosen class in a slot.

        It keeps the old record's same-named fields and is built with
        the assignments that reach its other fields, so a record that
        checks a field as it is built (a decoder's timing) gets it; a
        required field no assignment reaches is refused.
        """
        values = _shared_values(record_class, old)
        parameters = fields.parameters_of(record_class)
        for parameter in parameters.values():
            if parameter.field in values:
                continue
            child = (*segments, parameter.name)
            is_required = parameter.default is fields.REQUIRED
            if not is_required and not self._is_reached(parameter, child):
                continue
            values[parameter.field] = self._required(
                parameter, child, machine_clock
            )
        path = path_text(segments)
        try:
            return record_class(**values)
        except (TypeError, ValueError) as refused:
            raise fields.ParameterError(
                f"{path}: decsim refused it: {refused}"
            ) from None

    def _is_reached(self, parameter: fields.Parameter, segments: tuple) -> bool:
        """Whether an assignment reaches a field: its type, or its value."""
        if parameter.kind == fields.RECORD:
            choice = self._type_choice(segments)
            return choice is not None
        assignment = self.last_for(segments)
        return assignment is not None

    def _required(
        self,
        parameter: fields.Parameter,
        segments: tuple,
        machine_clock: Optional[config.Clock],
    ) -> Any:
        """A required field's value, from the assignments that reach it."""
        path = path_text(segments)
        if parameter.kind == fields.RECORD:
            choice = self._type_choice(segments)
            _refuse_a_missing_value(choice, f"{path}.{TYPE}")
            return self._chosen(
                parameter, None, choice, segments, machine_clock
            )
        assignment = self.last_for(segments)
        _refuse_a_missing_value(assignment, path)
        return parameter.value_of(assignment.written, machine_clock, path)


def _shared_values(record_class: type, old: Any) -> dict:
    """The old record's values for the fields the new class also has."""
    if not fields.is_record(old):
        return {}
    old_fields = {field.name for field in dataclasses.fields(old)}
    values = {}
    for field in dataclasses.fields(record_class):
        if field.name in old_fields:
            values[field.name] = getattr(old, field.name)
    return values


def _rebuilt(record: Any, changes: dict, segments: tuple) -> Any:
    """The record with the changes, decsim's own checks run on it."""
    path = path_text(segments)
    try:
        return dataclasses.replace(record, **changes)
    except (TypeError, ValueError) as refused:
        raise fields.ParameterError(
            f"{path}: decsim refused it: {refused}"
        ) from None


def _refuse_a_missing_value(
    assignment: Optional[Assignment], path: str
) -> None:
    if assignment is not None:
        return
    raise fields.ParameterError(f"{path} is required: set {path}=...")


def _is_under_any(
    segments: Sequence[str], run_file_paths: Sequence[str]
) -> bool:
    """Whether the segments, indexes dropped, start with one of the paths."""
    names = [_without_index(segment) for segment in segments]
    for run_file_path in run_file_paths:
        prefix = run_file_path.split(".")
        count = len(prefix)
        if names[:count] == prefix:
            return True
    return False


def _without_index(segment: str) -> str:
    """levels[0] -> levels."""
    name, _, _ = segment.partition("[")
    return name


def _matches(pattern: tuple, segments: tuple) -> bool:
    if not pattern:
        return not segments
    head = pattern[0]
    rest = pattern[1:]
    if head == ANY_DEPTH:
        return _matches_any_depth(rest, segments)
    if not segments:
        return False
    expression = _segment_expression(head)
    if not expression.fullmatch(segments[0]):
        return False
    return _matches(rest, segments[1:])


@functools.cache
def _segment_expression(pattern_segment: str) -> re.Pattern:
    """One segment of a pattern: `*` is any text, the rest is itself.

    A path's index brackets are text here, never a character class.
    """
    pieces = pattern_segment.split("*")
    escaped = [re.escape(piece) for piece in pieces]
    expression = ".*".join(escaped)
    return re.compile(expression)


def _holds_records(value: Any) -> bool:
    """Whether a value is a non-empty tuple of records."""
    if not isinstance(value, tuple) or not value:
        return False
    return all(fields.is_record(item) for item in value)


def _matches_any_depth(rest: tuple, segments: tuple) -> bool:
    """`**` stands for none, one or more of the leading segments."""
    start_count = len(segments) + 1
    for start in range(start_count):
        if _matches(rest, segments[start:]):
            return True
    return False


def _list_node(record: Any, segments: tuple, found: list) -> None:
    record_class = type(record)
    parameters = fields.parameters_of(record_class)
    for parameter in parameters.values():
        value = getattr(record, parameter.field)
        child = (*segments, parameter.name)
        if parameter.kind == fields.RECORD:
            _list_slot(parameter, value, child, found)
            continue
        if _holds_records(value):
            _list_parts(value, child, found)
            continue
        written = parameter.written(value)
        default = _written_default(parameter)
        path = path_text(child)
        setting = Setting(path, written, default)
        found.append(setting)


def _list_slot(
    parameter: fields.Parameter, value: Any, segments: tuple, found: list
) -> None:
    path = path_text(segments)
    if isinstance(value, tuple):
        _list_parts(value, segments, found)
        return
    if not fields.is_record(value):
        written = parameter.written(value)
        nothing = Setting(f"{path}.{TYPE}", written, None)
        found.append(nothing)
        return
    record_class = type(value)
    name = kinds.written_name(record_class)
    chosen = Setting(f"{path}.{TYPE}", name, None)
    found.append(chosen)
    _list_node(value, segments, found)


def _list_parts(parts: tuple, segments: tuple, found: list) -> None:
    for index, part in enumerate(parts):
        if not fields.is_record(part):
            continue
        name = f"{segments[-1]}[{index}]"
        child = (*segments[:-1], name)
        _list_node(part, child, found)


def _written_default(parameter: fields.Parameter) -> Optional[str]:
    if parameter.default is fields.REQUIRED:
        return None
    return parameter.written(parameter.default)
