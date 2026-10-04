"""An experiment: named points, each a machine to collect shots of.

A run file is a Python file that sets `experiment = Experiment(...)` at
module level; `decsim run` loads it (load). This is gem5 MultiSim's
shape, one script adding simulators by id
(src/python/gem5/utils/multisim/multisim.py:285-353), with sinter's
point: a machine's settings and json metadata, collected until a stop
rule (sinter/_data/_task.py:36-52). The machine knows nothing of
experiments; a point holds its settings and every shot builds a fresh
machine from them.
"""

import contextlib
import dataclasses
import itertools
import pathlib
import re
import sys
import types
from collections.abc import Mapping, Sequence
from typing import Optional, Union

import decsim.collect as collect
import decsim.experiments.collection as collection_module
import decsim.experiments.refusal as refusal
import decsim.settings as machine_settings

_THIS_FILE = pathlib.Path(__file__)
_RESOLVED_FILE = _THIS_FILE.resolve()
_REPOSITORY_ROOT = _RESOLVED_FILE.parents[2]
# experiments/ sits beside the decsim package, gem5's configs/ beside its
# binary; the shipped run files are what a refused path is listed with.
EXPERIMENTS_DIR = _REPOSITORY_ROOT / "experiments"
SHIPPED_RUN_FILE = "run.py"
# A name a folder can take on any filesystem the results go to.
FOLDER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# The module-level name a run file binds its experiment to.
EXPERIMENT_NAME_IN_A_RUN_FILE = "experiment"


@dataclasses.dataclass(frozen=True)
class Point:
    """One machine to collect shots of, under a name.

    name is the point's results folder and what --only picks. metadata is
    sinter's json_metadata, part of the point's id as of a sinter task's
    strong id. collection overrides the experiment's. record_options are no
    part of the id.
    """

    name: str
    machine: machine_settings.MachineSettings
    metadata: Mapping = dataclasses.field(default_factory=dict)
    collection: Optional[collection_module.CollectionSettings] = None
    record_options: collect.RecordOptions = collect.RecordOptions()

    def __post_init__(self) -> None:
        _check_folder_name(self.name, "point")


@dataclasses.dataclass(frozen=True)
class Experiment:
    """A named set of points under one collection plan.

    A point's name is its folder, so two points of one name are refused, as
    is a point no collection stops.
    """

    name: str
    points: Sequence[Point]
    collection: Optional[collection_module.CollectionSettings] = None

    def __post_init__(self) -> None:
        _check_folder_name(self.name, "experiment")
        if not self.points:
            raise ValueError(f"the experiment {self.name} has no points")
        _refuse_a_repeated_name(self.points)
        for point in self.points:
            self._refuse_a_point_without_a_collection(point)
            self._refuse_an_online_stop(point)

    def collection_of(
        self, point: Point
    ) -> collection_module.CollectionSettings:
        """The collection a point stops by: its own, else the experiment's."""
        if point.collection is not None:
            return point.collection
        return self.collection

    def point_named(self, name: str) -> Point:
        """The point of this name; a name it does not have is refused."""
        for point in self.points:
            if point.name == name:
                return point
        names = [point.name for point in self.points]
        _refuse_an_unknown_point(self.name, name, names)

    def only(self, name: str) -> "Experiment":
        """The experiment cut to its one point of this name."""
        point = self.point_named(name)
        return Experiment(self.name, [point], self.collection)

    def with_max_shots(self, shot_count: int) -> "Experiment":
        """Every point stopped at its first shot_count shots.

        Its target, minimum and time cap are dropped and its piece size
        kept. The collection is not part of a point's id, so the shots
        are the first shot_count of the full run's.
        """
        points = []
        for point in self.points:
            collection = self.collection_of(point)
            capped = collection_module.CollectionSettings(
                max_shots=shot_count, piece_rounds=collection.piece_rounds
            )
            capped_point = dataclasses.replace(point, collection=capped)
            points.append(capped_point)
        return Experiment(self.name, points, self.collection)

    def _refuse_a_point_without_a_collection(self, point: Point) -> None:
        if self.collection_of(point) is not None:
            return
        raise ValueError(
            f"the point {point.name} has no collection; give the experiment "
            "a collection, or the point one of its own"
        )

    def _refuse_an_online_stop(self, point: Point) -> None:
        """An online point stops at max_shots alone; any other stop is refused.

        Its shots are not independent draws, since each one's threshold
        learned from those before it, so a failure count has no interval
        a target stop could rest on, and a time cap would end its learning
        wherever the machine was fast.
        """
        task = task_of(point)
        if task.online_threshold is None:
            return
        settings = self.collection_of(point)
        has_only_a_shot_cap = settings.max_shots is not None
        if settings.max_failures is not None:
            has_only_a_shot_cap = False
        if settings.max_core_seconds is not None:
            has_only_a_shot_cap = False
        if has_only_a_shot_cap:
            return
        raise refusal.RefusalError(
            f"the point {point.name} calibrates its threshold online, so "
            "its shots are not independent draws and it stops at max_shots "
            "alone; its collection sets max_shots and neither max_failures "
            "nor max_core_seconds"
        )


def grid(**axes: Sequence) -> list:
    """Every combination of the axes' values, as one dict each.

    The product runs in the order the axes are given, the last axis
    fastest, as itertools.product and Hydra's multi-run do, so the
    points keep the order the experiment writes them in.
    """
    names = list(axes)
    value_lists = axes.values()
    combinations = itertools.product(*value_lists)
    points = []
    for values in combinations:
        pairs = zip(names, values, strict=True)
        points.append(dict(pairs))
    return points


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


def load_one_point(
    path: Union[str, pathlib.Path], name: Optional[str] = None
) -> Experiment:
    """The run file's experiment cut to one point, the named or the first."""
    study = load(path)
    if name is None:
        first_point = study.points[0]
        name = first_point.name
    return study.only(name)


def task_of(point: Point) -> collect.Task:
    """The task a point's shots run: its settings, metadata and state.

    The task builds the point's online calibrator, when its threshold
    learns one, which every shot's Machine.build receives.
    """
    return collect.Task(
        point.machine, point.metadata, record_options=point.record_options
    )


def _check_folder_name(name, what: str) -> None:
    """A name that becomes a folder: letters, digits, '.', '_' and '-'."""
    if isinstance(name, str) and FOLDER_NAME.fullmatch(name):
        return
    raise ValueError(
        f"the {what} name {name!r} is not a folder name; a name is letters, "
        "digits, '.', '_' and '-', starting with a letter or a digit"
    )


def _refuse_a_repeated_name(points: Sequence) -> None:
    """Each point's name is its folder, so no two points share one."""
    seen = set()
    for point in points:
        if point.name in seen:
            raise ValueError(
                f"two points are named {point.name}; a point's name is its "
                "folder in the results, so each point has its own"
            )
        seen.add(point.name)


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


def _refuse_an_unknown_point(
    experiment_name: str, name: str, names: list
) -> None:
    """A point name the experiment does not have, with the names it has."""
    listed = ", ".join(names)
    raise refusal.RefusalError(
        f"the experiment {experiment_name} has no point {name}; its points "
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
