"""A sweep: a machine for each cell of a grid, and how its tasks stop.

    config = sweeps.load_beside(__file__, "config.py")
    sweep = sweeps.Sweep(
        "backup_study",
        machine=config.machine,          # machine(**arguments): a machine
        grid={
            "distance": [7, 11],          # an argument of machine
            "gpu.unit_count": [1, 2, 4],  # a path of the machine
            "constants.roce.round_trip": ["2.305us", "4.5us"],  # a constant
        },
        task_name="d{distance}_gpus{gpu.unit_count}",
        stop=decsim.CollectionSettings(max_failures=100, max_shots=10_000),
        constants=config.constants,
    )
    experiment = sweep.experiment()

The shape is qecboard's Sweep (notes/qecboard_prototype, sweep.py) and
design.md Appendix A.4. The grid runs in its axes' order, the last
fastest (experiment.grid). An axis is an argument of machine when
machine takes that name; a constant when it is written constants.<name>,
set for its cell (topology/constants.py overriding); else a path of
the cell's machine, assigned as `--set` assigns one
(parameters/tree.py), before the command line's assignments
(parameters/in_force.py). An assignment in force that reaches a swept
path or constant is refused: every cell would run its value while the
cell's name and metadata said the cell's. So is one that makes two cells
one machine, as a `--set` to the value an argument axis feeds does.

metadata is the cell's values by default, as sinter's json_metadata
holds a task's swept values; a function of the cell, or of the cell and
the task (its graph and its layout, whose card prices every hop), writes
them otherwise, so a run that keeps the ids
of results already collected writes its metadata as it did. stop is
each task's collection, or a function of the cell; stop_all the
experiment's. A constants file with a value no cell read is refused,
but the names allow_unread keeps.
"""

import contextlib
import dataclasses
import importlib.util
import inspect
import pathlib
import types
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional, Union

import decsim.experiments.collect as collect
import decsim.experiments.collection as collection_module
import decsim.experiments.experiment as experiment_module
import decsim.parameters.fields as fields
import decsim.parameters.in_force as in_force
import decsim.parameters.tree as tree
import decsim.parameters.units as units
import decsim.topology.constants as constants_module

_CONSTANT_PREFIX = f"{constants_module.PREFIX}."


@dataclasses.dataclass(frozen=True)
class Sweep:
    """One experiment: a machine for each cell of a grid, and its stops."""

    name: str
    machine: Callable[..., Any]
    grid: Mapping[str, Sequence]
    task_name: str
    stop: Union[
        None,
        collection_module.CollectionSettings,
        Callable[[dict], collection_module.CollectionSettings],
    ] = None
    stop_all: Optional[collection_module.CollectionSettings] = None
    metadata: Optional[Callable[..., Mapping]] = None
    constants: Optional[constants_module.Section] = None
    allow_unread: Sequence[str] = ()

    def cells(self) -> list:
        """The grid's cells, in experiment.grid's order."""
        return experiment_module.grid(**self.grid)

    def tasks(self) -> list:
        """A task for every cell, in the grid's order."""
        axes = _Axes(self.grid, self.machine)
        axes.refuse_an_assignment_in_force_to_a_swept_value()
        tasks = []
        for cell in self.cells():
            task = self._task(cell, axes)
            tasks.append(task)
        self._refuse_the_unread()
        _refuse_cells_made_alike(tasks)
        return tasks

    def experiment(self) -> experiment_module.Experiment:
        """What `decsim run run.py` reads."""
        tasks = self.tasks()
        files = self._constants_files()
        return experiment_module.Experiment(
            self.name, tasks, self.stop_all, files
        )

    def _task(self, cell: dict, axes: "_Axes") -> collect.Task:
        """The cell's machine, assigned its paths, named and stopped."""
        arguments = axes.arguments_of(cell)
        swept_constants = axes.constants_of(cell)
        with self._constants_set(swept_constants):
            machine = self.machine(**arguments)
        paths = axes.paths_of(cell)
        assigned = _assigned(machine, paths, self.name)
        names = _Names(cell)
        task_name = self.task_name.format_map(names)
        stop = self._stop_for(cell)
        metadata = self._metadata_for(cell, task_name, assigned)
        return collect.Task(task_name, assigned, metadata, stop)

    def _constants_set(self, swept: dict) -> Any:
        if not swept:
            return contextlib.nullcontext()
        if self.constants is None:
            names = list(swept)
            raise fields.ParameterError(
                f"the axes {names} name constants; give the sweep its "
                "constants=..."
            )
        return constants_module.overriding(self.constants, swept)

    def _stop_for(self, cell: dict) -> Any:
        if callable(self.stop):
            return self.stop(cell)
        return self.stop

    def _metadata_for(self, cell: dict, task_name: str, machine: Any) -> dict:
        """The cell's plain values, or what the metadata function writes."""
        if self.metadata is None:
            return _plain_values(cell)
        signature = inspect.signature(self.metadata)
        if len(signature.parameters) == 1:
            written = self.metadata(cell)
            return dict(written)
        task = collect.Task(task_name, machine)
        written = self.metadata(cell, task)
        return dict(written)

    def _constants_files(self) -> tuple:
        if self.constants is None:
            return ()
        held = constants_module.held(self.constants)
        return (held,)

    def _refuse_the_unread(self) -> None:
        if self.constants is None:
            return
        held = constants_module.held(self.constants)
        held.refuse_the_unread(self.allow_unread)


def load_beside(run_file: Union[str, pathlib.Path], name: str) -> Any:
    """The Python file at name, relative to the run file's folder, as a module.

    A run file is run as a module of its own (experiment.load), so a
    config.py beside it, or another experiment's
    (../switching_baseline/config.py), is loaded by its path rather than
    imported.
    """
    run_path = pathlib.Path(run_file)
    folder = run_path.parent
    path = folder / name
    module_name = f"decsim_beside_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Axes:
    """The grid's axes by kind: the machine's arguments, constants, paths."""

    def __init__(self, grid: Mapping, machine: Callable) -> None:
        signature = inspect.signature(machine)
        taken = signature.parameters
        kinds = {axis: _kind_of(axis, taken) for axis in grid}
        self.arguments = _axes_of_kind(kinds, "argument")
        self.constants = _axes_of_kind(kinds, "constant")
        self.paths = _axes_of_kind(kinds, "path")
        _refuse_a_name_that_is_no_path(self.paths, taken)

    def arguments_of(self, cell: dict) -> dict:
        return {axis: cell[axis] for axis in self.arguments}

    def constants_of(self, cell: dict) -> dict:
        return {
            axis.removeprefix(_CONSTANT_PREFIX): cell[axis]
            for axis in self.constants
        }

    def paths_of(self, cell: dict) -> dict:
        return {axis: cell[axis] for axis in self.paths}

    def refuse_an_assignment_in_force_to_a_swept_value(self) -> None:
        """--set to a swept path or constant would give every cell one value."""
        swept = (*self.paths, *self.constants)
        reaching = in_force.reaching(swept)
        if not reaching:
            return
        first = reaching[0]
        raise fields.ParameterError(
            f"{first.origin} {first.text} sets a value the run file sweeps; "
            "every cell would run it while its name said the cell's. "
            "Change the sweep's grid, or drop the axis"
        )


class _Names(dict):
    """A cell for str.format_map, a dotted axis read as {section.key}."""

    def __init__(self, cell: dict) -> None:
        plain = {key: value for key, value in cell.items() if "." not in key}
        dict.__init__(self, plain)
        for key, value in cell.items():
            if "." in key:
                self._set_dotted(key, value)

    def _set_dotted(self, key: str, value: Any) -> None:
        first, *sections, leaf = key.split(".")
        empty = types.SimpleNamespace()
        node = self.setdefault(first, empty)
        for section in sections:
            node = _child_namespace(node, section)
        setattr(node, leaf, value)


def _kind_of(axis: str, taken: Mapping) -> str:
    """An argument the machine takes, a constants.<name>, or else a path."""
    if axis in taken:
        return "argument"
    if axis.startswith(_CONSTANT_PREFIX):
        return "constant"
    return "path"


def _axes_of_kind(kinds: dict, kind: str) -> list:
    """The axes of one kind, in the grid's order."""
    return [axis for axis, its_kind in kinds.items() if its_kind == kind]


def _refuse_cells_made_alike(tasks: list) -> None:
    """Two cells the assignments in force made one machine, refused.

    An argument axis feeds the machine function, so a `--set` to the
    value it feeds would run every cell at the set value while each
    cell's name and metadata said its own.
    """
    assignments = in_force.current()
    if not assignments:
        return
    first_of = {}
    for task in tasks:
        machine_id = task.machine_id()
        first = first_of.setdefault(machine_id, task.name)
        if first != task.name:
            _refuse_alike(first, task.name, assignments)


def _refuse_alike(first: str, second: str, assignments: tuple) -> None:
    written = [f"{each.origin} {each.text}" for each in assignments]
    raise fields.ParameterError(
        f"under {written} the cells {first} and {second} run one machine, "
        "so a value the sweep varies is set for every cell. Drop the "
        "assignment, or sweep the value as an axis"
    )


def _assigned(machine: Any, paths: dict, sweep_name: str) -> Any:
    """The machine with the cell's paths assigned; an unmatched one refused."""
    if not paths:
        return machine
    origin = f"the sweep {sweep_name}'s cell"
    assignments = [
        tree.Assignment(path, value, origin) for path, value in paths.items()
    ]
    applied = tree.apply(machine, assignments)
    for position, assignment in enumerate(assignments):
        if position not in applied.matched:
            _refuse_an_unmatched_path(machine, assignment)
    return applied.settings


def _refuse_an_unmatched_path(machine: Any, assignment: Any) -> None:
    settings = tree.settings_of(machine)
    paths = [setting.path for setting in settings]
    near = units.did_you_mean(assignment.pattern, paths)
    raise fields.ParameterError(
        f"the axis {assignment.pattern} is no argument of the machine and "
        f"no path of it.{near}"
    )


def _refuse_a_name_that_is_no_path(paths: list, taken: Mapping) -> None:
    """An undotted axis the machine does not take names nothing."""
    for axis in paths:
        if "." in axis:
            continue
        arguments = list(taken)
        near = units.did_you_mean(axis, arguments)
        raise fields.ParameterError(
            f"the axis {axis!r} is no argument of the machine, whose "
            f"arguments are {arguments}.{near}"
        )


def _plain_values(cell: dict) -> dict:
    """The cell's numbers and strings, the values a results file holds."""
    return {
        key: value
        for key, value in cell.items()
        if isinstance(value, (int, float, str, bool))
    }


def _child_namespace(
    node: types.SimpleNamespace, section: str
) -> types.SimpleNamespace:
    """node.section, made empty when node has none."""
    child = getattr(node, section, None)
    if child is not None:
        return child
    child = types.SimpleNamespace()
    setattr(node, section, child)
    return child
