"""`decsim params` and `decsim info`: what can be set, and to what.

    decsim params <run file> [--only NAME] [--set PATH=VALUE ...]
        [--match PATTERN]
    decsim info [<record name>]

params prints every settable path of one task's machine (the first, or
the --only one; a graph's paths for a task built from a graph) with its
value as it would be written, its record's default where the run file
moved it, and the assignment that set it; then each constants file the
run read, every value with its source.
--match keeps the paths a pattern reaches, `*` and `**` as `--set`
takes them. This is gem5's `--list-sim-objects`/config.ini and
OMNeT++'s `-q params` for decsim's settings tree.

info prints one record's parameters, their units and defaults, and for
each part the records its slot takes, so a researcher swaps a part
(`path.type=NAME`) without reading the code. With no name it prints
every slot a machine has, from MachineSettings down, and the records
each takes, as sst-info lists every element it can load.
"""

import argparse
import os
from typing import Any, Optional

import decsim.experiments.experiment as experiment
import decsim.experiments.refusal as refusal
import decsim.parameters.assignments as assignments
import decsim.parameters.fields as fields
import decsim.parameters.kinds as kinds
import decsim.parameters.tree as tree
import decsim.parameters.units as units
import decsim.settings as machine_settings
import decsim.topology.constants as constants_module

_NO_ORIGIN = ""
_RUN_FILE_ONLY = "run file only"
_SETTINGS = "Settings"
# a parameter's kind -> how its value is written
_WRITTEN_AS = {
    "cycles": "cycles, or a time on the part's clock",
    "ticks": "a time",
    "microseconds": "a time",
    "decibels": "dB",
    "bytes": "a size",
    "bits": "a size",
    fields.CLOCK: "a frequency, or none for the machine's",
}


def params(argv: list) -> None:
    """Print one task's settable paths, values, defaults and setters."""
    parser = _params_parser()
    parsed = parser.parse_args(argv)
    put_in_force(parsed.set)
    study = experiment.load_one_task(parsed.run_file, parsed.only)
    task = study.tasks[0]
    root = settings_root(task)
    every_assignment = _assignments_or_refusal()
    applied = tree.apply(root, every_assignment)
    settings = tree.settings_of(root)
    print(f"# {task.name}: path = value  [record default]  (set by)")
    for setting in settings:
        if not _is_matched(parsed.match, setting):
            continue
        origin = applied.origins.get(setting.path, _NO_ORIGIN)
        if tree.is_set_in_the_run_file(root, setting.path):
            origin = _RUN_FILE_ONLY
        line = _setting_line(setting, origin)
        print(line)
    for constants in study.constants:
        _print_constants(constants, parsed.match)


def info(argv: list) -> None:
    """Print one record's parameters, and the records each part takes."""
    parser = argparse.ArgumentParser(prog="decsim info")
    parser.add_argument(
        "record",
        nargs="?",
        default=None,
        help="a record's name, e.g. union_find; none lists every slot",
    )
    parsed = parser.parse_args(argv)
    if parsed.record is None:
        _print_every_slot()
        return
    record_class = _record_class_named(parsed.record)
    name = kinds.written_name(record_class)
    print(f"{name}: {record_class.__module__}.{record_class.__qualname__}")
    parameters = fields.parameters_of(record_class)
    for parameter in parameters.values():
        line = _parameter_line(parameter)
        print(line)


def every_slot() -> dict:
    """(record class, field name): the records that slot takes.

    Every slot reached from MachineSettings through the records its
    slots take, each record read once.
    """
    slots = {}
    waiting = [machine_settings.MachineSettings]
    seen = set()
    while waiting:
        record_class = waiting.pop()
        if record_class in seen:
            continue
        seen.add(record_class)
        offered = _add_slots_of(record_class, slots)
        waiting.extend(offered)
    return slots


def settings_root(task: Any) -> Any:
    """What a task's paths name: its graph if built from one, its machine."""
    if task.graph is not None:
        return task.graph
    return task.machine


def put_in_force(flag_texts: list) -> None:
    """The --set texts appended to DECSIM_SET, after it, so a flag wins.

    A job, a worker and a Slurm script all read the variable, so every
    process of the run sees the same assignments.
    """
    if not flag_texts:
        return
    written = os.environ.get(assignments.ENVIRONMENT_VARIABLE, "")
    texts = [written, *flag_texts]
    kept = [text for text in texts if text]
    joined = assignments.SEPARATOR.join(kept)
    os.environ[assignments.ENVIRONMENT_VARIABLE] = joined


def add_set_argument(parser: argparse.ArgumentParser) -> None:
    """--set PATH=VALUE, repeatable, on a command's parser."""
    parser.add_argument(
        assignments.FLAG,
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="set a parameter, e.g. weak_decoder.unit_count=3; repeat it, "
        "a later one winning; `decsim params` lists the paths",
    )


def _params_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="decsim params")
    parser.add_argument("run_file", help="the experiment's run file")
    parser.add_argument(
        "--only", default=None, help="the task to list, by its name"
    )
    parser.add_argument(
        "--match", default=None, help="only the paths this pattern reaches"
    )
    add_set_argument(parser)
    return parser


def _assignments_or_refusal() -> list:
    try:
        return assignments.from_environment()
    except ValueError as refused:
        raise refusal.RefusalError(str(refused)) from None


def _is_matched(pattern: Optional[str], setting: tree.Setting) -> bool:
    """Whether --match keeps a setting; every one without it."""
    if pattern is None:
        return True
    segments = setting.path.split(".")
    return tree.matches(pattern, segments)


def _print_constants(constants: Any, pattern: Optional[str]) -> None:
    """A constants file's names as --set writes them, values and sources."""
    print(f"# {constants.origin}: constants.name = value  (source)")
    for name, value in constants.values.items():
        path = f"{constants_module.PREFIX}.{name}"
        segments = path.split(".")
        if pattern is not None and not tree.matches(pattern, segments):
            continue
        source = constants.sources.get(name, _NO_ORIGIN)
        line = _setting_line_of(path, value, source)
        print(line)


def _setting_line_of(path: str, written: Any, origin: str) -> str:
    line = f"{path} = {written}"
    if origin:
        line = f"{line}  ({origin})"
    return line


def _setting_line(setting: tree.Setting, origin: str) -> str:
    line = f"{setting.path} = {setting.written}"
    is_moved = (
        setting.default is not None and setting.default != setting.written
    )
    if is_moved:
        line = f"{line}  [{setting.default}]"
    if origin:
        line = f"{line}  ({origin})"
    return line


def _add_slots_of(record_class: type, slots: dict) -> list:
    """Each record slot of the class into slots; every record offered."""
    every_offered = []
    parameters = fields.parameters_of(record_class)
    for parameter in parameters.values():
        if parameter.kind != fields.RECORD:
            continue
        offered = kinds.records_for(parameter.hint)
        slots[(record_class, parameter.name)] = offered
        every_offered.extend(offered)
    return every_offered


def _print_every_slot() -> None:
    """Each slot as record.field, and the names of the records it takes."""
    print("# slot: the records it takes; `decsim info NAME` prints one")
    lines = []
    slots = every_slot()
    for (record_class, field_name), offered in slots.items():
        owner = kinds.written_name(record_class)
        names = [kinds.written_name(record) for record in offered]
        listed = ", ".join(sorted(names))
        lines.append(f"{owner}.{field_name}: {listed}")
    for line in sorted(lines):
        print(line)


def _parameter_line(parameter: fields.Parameter) -> str:
    if parameter.kind == fields.RECORD:
        offered = kinds.records_for(parameter.hint)
        names = [kinds.written_name(record_class) for record_class in offered]
        listed = ", ".join(sorted(names))
        return f"  {parameter.name}.type: one of {listed}"
    default = _written_default(parameter)
    unit = _unit_of(parameter)
    return f"  {parameter.name} = {default}{unit}"


def _written_default(parameter: fields.Parameter) -> str:
    if parameter.default is fields.REQUIRED:
        return "(required)"
    return parameter.written(parameter.default)


def _unit_of(parameter: fields.Parameter) -> str:
    if parameter.kind in _WRITTEN_AS:
        written_as = _WRITTEN_AS[parameter.kind]
        return f"  ({written_as})"
    if parameter.plain_type is None:
        return ""
    return f"  ({parameter.plain_type.__name__})"


def _record_class_named(name: str) -> type:
    """The record a name stands for, a settings record first.

    A run's state is a dataclass too (decsim.machine.Machine), so of two
    records of one name the one named ...Settings is meant. A name decsim
    does not define is read as an import path (mylab.parts.X.Settings).
    """
    every_class = kinds.every_record_class()
    named = kinds.records_named(name, every_class)
    settings = [
        record_class
        for record_class in named
        if record_class.__name__.endswith(_SETTINGS)
    ]
    matches = settings or named
    if len(matches) == 1:
        return matches[0]
    if matches:
        _refuse_an_ambiguous_name(name, matches)
    if "." in name:
        return _imported_or_refusal(name)
    names = kinds.every_name(every_class)
    near = units.did_you_mean(name, names)
    raise refusal.RefusalError(f"decsim defines no record {name}.{near}")


def _imported_or_refusal(name: str) -> type:
    try:
        return kinds.imported_class(name, "decsim info")
    except fields.ParameterError as refused:
        raise refusal.RefusalError(str(refused)) from None


def _refuse_an_ambiguous_name(name: str, matches: list) -> None:
    qualified = [kinds.import_path(record_class) for record_class in matches]
    joined = ", ".join(qualified)
    raise refusal.RefusalError(f"{name} names several records: {joined}")
