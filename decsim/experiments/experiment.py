"""An experiment: named tasks, each a machine to collect shots of.

A run file is a Python file that sets `experiment = Experiment(...)` at
module level; `decsim run` loads it (load). This is gem5 MultiSim's
shape, one script adding simulators by id
(src/python/gem5/utils/multisim/multisim.py:285-353), with collect.Task
after sinter's Task: a machine's settings and json metadata, collected
until a stop rule. The machine knows nothing of experiments; a task
holds its settings and every shot builds a fresh machine from them.
"""

import contextlib
import dataclasses
import itertools
import pathlib
import sys
import types
from collections.abc import Sequence
from typing import Optional, Union

import decsim.experiments.collect as collect
import decsim.experiments.collection as collection_module
import decsim.experiments.refusal as refusal

_THIS_FILE = pathlib.Path(__file__)
_RESOLVED_FILE = _THIS_FILE.resolve()
_REPOSITORY_ROOT = _RESOLVED_FILE.parents[2]
# experiments/ sits beside the decsim package, gem5's configs/ beside its
# binary; the shipped run files are what a refused path is listed with.
EXPERIMENTS_DIR = _REPOSITORY_ROOT / "experiments"
SHIPPED_RUN_FILE = "run.py"
# The module-level name a run file binds its experiment to.
EXPERIMENT_NAME_IN_A_RUN_FILE = "experiment"


@dataclasses.dataclass(frozen=True)
class Experiment:
    """A named set of tasks under one collection plan.

    A task's name is its folder, so two tasks of one name are refused, as
    is a task no collection stops.
    """

    name: str
    tasks: Sequence[collect.Task]
    collection: Optional[collection_module.CollectionSettings] = None

    def __post_init__(self) -> None:
        collect.check_folder_name(self.name, "experiment")
        if not self.tasks:
            raise ValueError(f"the experiment {self.name} has no tasks")
        _refuse_a_repeated_name(self.tasks)
        for task in self.tasks:
            self._refuse_a_task_without_a_collection(task)
            self._refuse_an_online_stop(task)

    def collection_of(
        self, task: collect.Task
    ) -> collection_module.CollectionSettings:
        """The collection a task stops by: its own, else the experiment's."""
        if task.collection is not None:
            return task.collection
        return self.collection

    def task_named(self, name: str) -> collect.Task:
        """The task of this name; a name it does not have is refused."""
        for task in self.tasks:
            if task.name == name:
                return task
        names = [task.name for task in self.tasks]
        _refuse_an_unknown_task(self.name, name, names)

    def only(self, name: str) -> "Experiment":
        """The experiment cut to its one task of this name."""
        task = self.task_named(name)
        return Experiment(self.name, [task], self.collection)

    def with_max_shots(self, shot_count: int) -> "Experiment":
        """Every task stopped at its first shot_count shots.

        Its target, minimum and time cap are dropped and its piece size
        kept. The collection is not part of a task's id, so the shots
        are the first shot_count of the full run's.
        """
        tasks = []
        for task in self.tasks:
            collection = self.collection_of(task)
            capped = collection_module.CollectionSettings(
                max_shots=shot_count, piece_rounds=collection.piece_rounds
            )
            capped_task = dataclasses.replace(task, collection=capped)
            tasks.append(capped_task)
        return Experiment(self.name, tasks, self.collection)

    def _refuse_a_task_without_a_collection(self, task: collect.Task) -> None:
        if self.collection_of(task) is not None:
            return
        raise ValueError(
            f"the task {task.name} has no collection; give the experiment "
            "a collection, or the task one of its own"
        )

    def _refuse_an_online_stop(self, task: collect.Task) -> None:
        """An online task stops at max_shots alone; any other stop is refused.

        Its shots are not independent draws, since each one's threshold
        learned from those before it, so a failure count has no interval
        a target stop could rest on, and a time cap would end its learning
        wherever the machine was fast.
        """
        if task.online_threshold is None:
            return
        settings = self.collection_of(task)
        has_only_a_shot_cap = settings.max_shots is not None
        if settings.max_failures is not None:
            has_only_a_shot_cap = False
        if settings.max_core_seconds is not None:
            has_only_a_shot_cap = False
        if has_only_a_shot_cap:
            return
        raise refusal.RefusalError(
            f"the task {task.name} calibrates its threshold online, so "
            "its shots are not independent draws and it stops at max_shots "
            "alone; its collection sets max_shots and neither max_failures "
            "nor max_core_seconds"
        )


def grid(**axes: Sequence) -> list:
    """Every combination of the axes' values, as one dict each.

    The product runs in the order the axes are given, the last axis
    fastest, as itertools.product and Hydra's multi-run do, so the
    tasks keep the order the experiment writes them in.
    """
    names = list(axes)
    value_lists = axes.values()
    combinations = itertools.product(*value_lists)
    cells = []
    for values in combinations:
        pairs = zip(names, values, strict=True)
        cells.append(dict(pairs))
    return cells


def load(path: Union[str, pathlib.Path]) -> Experiment:
    """The experiment a run file defines.

    The file is executed as a module of its own, registered under its
    name so a worker process can unpickle what it defines, and its
    module-level `experiment` is the run's.
    """
    path = pathlib.Path(path)
    _refuse_a_path_that_is_not_a_file(path)
    module = _executed_run_file(path)
    defined = getattr(module, EXPERIMENT_NAME_IN_A_RUN_FILE, None)
    if defined is None:
        raise refusal.RefusalError(
            f"{path} defines no experiment; a run file sets experiment = "
            "Experiment(...) at module level"
        )
    return defined


def load_one_task(
    path: Union[str, pathlib.Path], name: Optional[str] = None
) -> Experiment:
    """The run file's experiment cut to one task, the named or the first."""
    study = load(path)
    if name is None:
        first_task = study.tasks[0]
        name = first_task.name
    return study.only(name)


def _refuse_a_repeated_name(tasks: Sequence) -> None:
    """Each task's name is its folder, so no two tasks share one."""
    seen = set()
    for task in tasks:
        if task.name in seen:
            raise ValueError(
                f"two tasks are named {task.name}; a task's name is its "
                "folder in the results, so each task has its own"
            )
        seen.add(task.name)


def _executed_run_file(path: pathlib.Path):
    """The run file run as a module, a refusal in it one sentence.

    The source is compiled as it stands now, never from __pycache__: a
    cached module is trusted when the source's size and whole-second
    mtime match (PEP 552), so a run file rewritten within one second at
    the same length would otherwise run its old text.
    """
    stem = path.stem
    module_name = f"decsim_run_file_{stem}"
    source = path.read_text()
    code = compile(source, str(path), "exec")
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    with _refused_in(path):
        exec(code, module.__dict__)
    return module


def _refuse_an_unknown_task(
    experiment_name: str, name: str, names: list
) -> None:
    """A task name the experiment does not have, with the names it has."""
    listed = ", ".join(names)
    raise refusal.RefusalError(
        f"the experiment {experiment_name} has no task {name}; its tasks "
        f"are {listed}"
    )


def _refuse_a_path_that_is_not_a_file(path: pathlib.Path) -> None:
    """A run file that is not there, with the shipped ones to pick from."""
    if path.is_file():
        return
    names = _shipped_experiment_names()
    listed = ", ".join(names)
    raise refusal.RefusalError(
        f"{path} is not a file; the shipped experiments are {listed}"
    )


def _shipped_experiment_names() -> list:
    """Every shipped run file, experiments/<study>/run.py, by that path."""
    shipped_files = EXPERIMENTS_DIR.glob(f"*/{SHIPPED_RUN_FILE}")
    names = []
    for shipped in sorted(shipped_files):
        relative = shipped.relative_to(_REPOSITORY_ROOT)
        names.append(str(relative))
    return names


@contextlib.contextmanager
def _refused_in(path: pathlib.Path):
    """A ValueError raised inside, as one sentence naming the run file."""
    try:
        yield
    except ValueError as refused:
        raise refusal.RefusalError(f"{path}: {refused}") from refused
