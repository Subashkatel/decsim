"""An experiment: named points, each a machine to collect shots of.

A run file is a Python file that sets `experiment = Experiment(...)` at
module level; `decsim run` loads it (load). This is gem5 MultiSim's
shape, one script adding simulators by id
(src/python/gem5/utils/multisim/multisim.py:285-353), with sinter's
point: a machine's settings and json metadata, collected until a stop
rule (sinter/_data/_task.py:36-52). The machine knows nothing of
experiments; a point holds its settings and every shot builds a fresh
machine from them.

This module is also the one yaml reader: a yaml experiment file's
sections are handed to the packages that own them, one settings record
each (decsim.settings.MachineSettings.from_mapping), at every point of
its sweep, and the sweep becomes an Experiment's points
(ExperimentConfig.experiment). `extends: other.yaml` starts from that
file (same folder) and overrides the top-level keys this file names.
"""

import contextlib
import copy
import dataclasses
import importlib.util
import itertools
import json
import os
import pathlib
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Optional

import yaml

import decsim.collect as collect
import decsim.config as config_module
import decsim.experiments.collection as collection_module
import decsim.experiments.refusal as refusal
import decsim.machine as machine_module
import decsim.settings as machine_settings

_THIS_FILE = pathlib.Path(__file__)
_RESOLVED_FILE = _THIS_FILE.resolve()
_REPOSITORY_ROOT = _RESOLVED_FILE.parents[2]
# configs/ sits beside the decsim package, gem5's configs/ beside its
# binary; the shipped experiments are what a refused path is listed with.
CONFIGS_DIR = _REPOSITORY_ROOT / "configs"
REFERENCE_FILE = CONFIGS_DIR / "reference.yaml"
BASES_FOLDER = "bases"
SWEEP_BLOCK_KEYS = ("axes",)
# A block may also carry its own collection keys, over the file's.
OPTIONAL_SWEEP_BLOCK_KEYS = ("collection",)
# A value that is one whole reference to another setting, `${a.b}`:
# OmegaConf's node interpolation, whose value "will be the value of that
# node" (OmegaConf 2.3, "Variable interpolation"), kept to a whole value
# so that no text is spliced.
WHOLE_VALUE_REFERENCE = re.compile(r"\$\{([^${}]+)\}")
# A name a folder can take on any filesystem the results go to.
FOLDER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# A yaml point has no name of its own, so it is named by the start of
# its id: as many hex characters as git shows of a commit.
YAML_POINT_NAME_LENGTH = 12
# The module-level name a run file binds its experiment to.
EXPERIMENT_NAME_IN_A_RUN_FILE = "experiment"


@dataclasses.dataclass(frozen=True)
class Point:
    """One machine to collect shots of, under a name.

    name is the point's folder in the results and what `decsim run
    --only` picks. machine is the settings every shot builds a fresh
    machine from. metadata is sinter's json_metadata: the columns the
    point's rows carry beside its results; it is part of the point's id,
    as it is of a sinter task's strong id. collection overrides the
    experiment's. sections is the yaml a point the yaml translator made
    resolved to, which its swept cells are read from; None for a point
    built in Python, whose cells are its metadata.
    """

    name: str
    machine: machine_settings.MachineSettings
    metadata: Mapping = dataclasses.field(default_factory=dict)
    collection: Optional[collection_module.CollectionSettings] = None
    sections: Optional[Mapping] = dataclasses.field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        _check_folder_name(self.name, "point")


@dataclasses.dataclass(frozen=True)
class Experiment:
    """A named set of points and the collection they stop by.

    name names the results folder. collection is every point's unless
    the point carries its own. A point's name is its folder, so two
    points of one name are refused, as is a point no collection stops.
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


def grid(**axes) -> list:
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


def load(path) -> Experiment:
    """The experiment a run file defines, or a yaml's, translated.

    A Python file is executed as a module of its own, registered under
    its name so a worker process can unpickle what it defines, and its
    module-level `experiment` is the run's. A yaml goes through the yaml
    reader and becomes an Experiment (ExperimentConfig.experiment).
    """
    path = pathlib.Path(path)
    if path.suffix != ".py":
        config = load_experiment(path)
        return config.experiment()
    _refuse_a_path_that_is_not_a_file(path)
    module = _executed_run_file(path)
    defined = getattr(module, EXPERIMENT_NAME_IN_A_RUN_FILE, None)
    if defined is None:
        raise refusal.RefusalError(
            f"{path} defines no experiment; a run file sets experiment = "
            "Experiment(...) at module level"
        )
    return defined


def load_one_point(path, name: Optional[str] = None) -> Experiment:
    """The run file's experiment cut to one point, the named or the first.

    A yaml is read for that point alone (ExperimentConfig.one_point).
    """
    path = pathlib.Path(path)
    if path.suffix != ".py":
        config = load_experiment(path)
        return config.one_point(name)
    study = load(path)
    if name is None:
        first_point = study.points[0]
        name = first_point.name
    return study.only(name)


def run_files(path) -> tuple:
    """The files a run reads: the run file, and a yaml's extends chain.

    A yaml names its bases nearest first, as the reader follows them.
    """
    path = pathlib.Path(path)
    if path.suffix == ".py":
        return (path,)
    _sections, _folders, config_files = _yaml_sections(path)
    return config_files


def description(study: Experiment, run_file: pathlib.Path) -> list:
    """What a run resolved to, before its first shot, as gem5 dumps it.

    The run file, each section's kind and the fabric card as the first
    point's settings hold them, then every point with its metadata and
    collection, then what the run records beyond its results (gem5
    --dump-config, src/python/m5/main.py:241).
    """
    first_point = study.points[0]
    settings = first_point.machine
    lines = [f"run file: {run_file}", f"experiment: {study.name}"]
    section_lines = _section_lines(settings)
    lines.extend(section_lines)
    links_line = _links_line(settings.links)
    lines.append(links_line)
    for point in study.points:
        point_line = _point_line(study, point)
        lines.append(point_line)
    observation_lines = _observation_lines(settings.observation)
    lines.extend(observation_lines)
    return lines


def task_of(point: Point) -> collect.Task:
    """The task a point's shots run: its settings, metadata and state.

    The online threshold source is the point's state, not a setting:
    every shot of the point shares the one instance and it learns over
    them, so it is taken off the settings and carried on the task, and
    the point's id is that of the settings without it.
    """
    escalation = point.machine.escalation
    online_threshold = escalation.online_threshold
    plain_escalation = dataclasses.replace(escalation, online_threshold=None)
    settings = dataclasses.replace(point.machine, escalation=plain_escalation)
    return collect.Task(settings, point.metadata, online_threshold)


@dataclasses.dataclass(frozen=True)
class SweepBlock:
    """One cartesian product of axes, and how its points are collected.

    An axis is a yaml path and the values the sweep sets there. A block
    is every combination of its axes, in the order they are written, as
    Hydra's multi-run makes one job per combination of `key=v1,v2`
    overrides, each override a config node's dotted path (hydra.cc,
    "Multi-run"). A sweep's blocks are a union. collection is how the
    block's points are collected: the file's section, the block's keys
    over it.
    """

    axes: Mapping  # yaml path -> tuple of values
    collection: collection_module.CollectionSettings

    def points(self) -> list:
        """Its cartesian product, one {yaml path: value} per point."""
        return grid(**self.axes)


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """Everything one yaml file says: its sections, and the sweep over them."""

    name: str  # the yaml stem; suffixes the run folder
    # the sections after extends, the sweep taken out: each point reads
    # its machine from them (resolved_sections)
    sections: Mapping
    # each section's yaml folder, which its relative paths resolve against
    section_folders: Mapping
    sweep: tuple  # of SweepBlock
    # the yaml files this config was read from, nearest first (an extends
    # chain)
    config_files: tuple

    def tasks(self) -> list:
        """One task per sweep point, blocks in order, each block a product."""
        tasks = []
        for task, _collection in self.point_tasks():
            tasks.append(task)
        return tasks

    def point_tasks(self) -> list:
        """Each point's task with its block's collection, blocks in order."""
        pairs = []
        for block in self.sweep:
            for values in block.points():
                task = self.point_task(values)
                pairs.append((task, block.collection))
        return pairs

    def point_task(self, values: Mapping) -> collect.Task:
        """The task of one sweep point: its settings and metadata.

        values maps a yaml path to what the point sets there, and they
        are the point's metadata, sinter's json_metadata
        (sinter/_data/_task.py:36-52). The task's strong id covers them
        and every setting they resolve to, so a point is named by what
        it runs, as gem5's multisim names each simulation's output by
        its one id (src/python/gem5/utils/multisim/multisim.py). Every
        point is read by the one yaml reader, so an axis meets the checks
        a written value meets; then the workload's maker runs there and
        the escalation takes the threshold its source gives the point.
        """
        path = self.config_files[0]
        with _refused_in(path):
            resolved, settings = self._read_point(values)
            return _point_task_of(settings, resolved, values)

    def experiment(self) -> Experiment:
        """The sweep as an Experiment, one point per distinct sweep point.

        A point two blocks name is one point, the first block's, and two
        blocks that collect it two ways are refused, since a point stops
        by one rule. Each point carries its block's collection, its
        resolved sections, and its online threshold source on its
        escalation, where task_of takes it back off, so its id is the
        one its task has.
        """
        points_by_id = {}
        collections_by_id = {}
        for task, collection in self.point_tasks():
            point_id = task.strong_id()
            earlier = collections_by_id.setdefault(point_id, collection)
            if earlier != collection:
                _refuse_two_collections(task, earlier, collection)
            if point_id in points_by_id:
                continue
            sections = self.resolved_sections(task.metadata)
            points_by_id[point_id] = _point_of(task, collection, sections)
        points = points_by_id.values()
        return Experiment(self.name, tuple(points))

    def one_point(self, name: Optional[str] = None) -> Experiment:
        """The sweep cut to one point: the one named, else the first.

        A narrated shot needs one point, so the points are not checked
        against each other: a base whose blocks collect a point two ways
        still runs a shot, and without a name only the first point is
        read.
        """
        pairs = self.point_tasks()
        if name is None:
            first_block = self.sweep[0]
            first_task = self.first_point_task()
            pairs = [(first_task, first_block.collection)]
        names = []
        for task, collection in pairs:
            point_id = task.strong_id()
            point_name = point_id[:YAML_POINT_NAME_LENGTH]
            names.append(point_name)
            if name in (None, point_name):
                sections = self.resolved_sections(task.metadata)
                point = _point_of(task, collection, sections)
                return Experiment(self.name, (point,))
        _refuse_an_unknown_point(self.name, name, names)

    def first_point_task(self) -> collect.Task:
        """The task of the first point of the first sweep block."""
        block = self.sweep[0]
        points = block.points()
        return self.point_task(points[0])

    def resolved_sections(self, values: Mapping) -> dict:
        """The sections at one point: its axes placed, references resolved.

        An axis sets the value at its yaml path, and a mapping replaces
        the node there whole, as a Hydra config-group override replaces
        a sub-config. A whole-value reference then takes the value at its
        path, so a fact the machine and the workload share is written
        once: `distance: ${qpu.distance}`.
        """
        sections = copy.deepcopy(self.sections)
        for path, value in values.items():
            _place(sections, path, value)
        return _resolved(sections, sections, ())

    def built_machine(
        self, settings: machine_settings.MachineSettings, seed: int
    ) -> machine_module.Machine:
        """The machine the settings build, a build refusal one sentence."""
        path = self.config_files[0]
        with _refused_in(path):
            return machine_module.Machine.build(settings, seed)

    def _read_point(self, values: Mapping) -> tuple:
        """A point's resolved sections, and the settings read from them."""
        resolved = self.resolved_sections(values)
        settings = machine_settings.MachineSettings.from_mapping(
            resolved, name=self.name, section_folders=self.section_folders
        )
        return resolved, settings


def resolved_description(
    config: ExperimentConfig, settings: machine_settings.MachineSettings
) -> list:
    """What one yaml really says, after its extends chain is applied.

    settings is the first point's. An edit that did not land shows up
    here immediately: a config that extends another replaces its base's
    keys whole, so a `sweep` edited in the base never reaches a child
    that declares its own. gem5 prints the same thing with --dump-config
    (src/python/m5/main.py:241).
    """
    lines = [_files_line(config)]
    for line in _section_lines(settings):
        lines.append(line)
    links_line = _links_line(settings.links)
    lines.append(links_line)
    for index, block in enumerate(config.sweep, start=1):
        block_line = _sweep_block_line(index, block)
        lines.append(block_line)
    for line in _observation_lines(settings.observation):
        lines.append(line)
    return lines


def value_lines(
    config: ExperimentConfig, settings: machine_settings.MachineSettings
) -> list:
    """Every value the first point resolves to, and who set it where sure.

    One line per value of the settings records: its dotted path, its
    value (a value an axis sets lists the sweep's values), and in
    brackets its layer and source line when the value is one yaml key's
    (_origin_of). gem5's config.ini lists every parameter of every object
    the same way (src/python/m5/simulate.py:122-127).
    """
    written = _written_keys(config)
    documented = _documented_keys()
    swept = _swept_values(config.sweep)
    settings_value = collect.json_value(settings)
    leaves = value_leaves(settings_value, ())
    lines = []
    for path, value in leaves:
        key = _yaml_key(path)
        value = swept.get(key, value)
        value_text = json.dumps(value)
        dotted = ".".join(path)
        line = f"{dotted} = {value_text}"
        origin = _origin_of(key, swept, written, documented)
        if origin is not None:
            line = f"{line}  [{origin}]"
        lines.append(line)
    return lines


def value_leaves(value, path: tuple) -> list:
    """Every value under a json value that is not a mapping, with its path.

    An empty mapping is a leaf of its own.
    """
    if not isinstance(value, Mapping) or not value:
        return [(path, value)]
    leaves = []
    for key, item in value.items():
        item_path = path + (str(key),)
        item_leaves = value_leaves(item, item_path)
        leaves.extend(item_leaves)
    return leaves


def load_experiment(path) -> ExperimentConfig:
    """Read one yaml file, its extends chain applied, and its sweep.

    The first point's sections are read here, so a section the yaml
    reader refuses is refused when the file loads; a value an axis sets
    at a later point is read when that point's task is made.
    """
    path = pathlib.Path(path)
    _refuse_a_path_that_is_not_a_file(path)
    sections, section_folders, config_files = _yaml_sections(path)
    if not sections.get("sweep"):
        raise refusal.RefusalError(
            f"{path} has no sweep; a yaml names at least one sweep block "
            "(configs/reference.yaml)"
        )
    sweep_section = sections.pop("sweep")
    top_collection = sections.pop("collection", None)
    sweep = _sweep_blocks(sweep_section, top_collection)
    config = ExperimentConfig(
        name=path.stem,
        sections=sections,
        section_folders=section_folders,
        sweep=sweep,
        config_files=config_files,
    )
    first_block = sweep[0]
    first_points = first_block.points()
    with _refused_in(path):
        config._read_point(first_points[0])
    return config


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
    """The run file run as a module, a refusal in it one sentence."""
    stem = path.stem
    module_name = f"decsim_run_file_{stem}"
    specification = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    with _refused_in(path):
        specification.loader.exec_module(module)
    return module


def _point_of(
    task: collect.Task,
    collection: collection_module.CollectionSettings,
    sections: dict,
) -> Point:
    """A yaml point: its task's settings with the online source put back."""
    settings = task.settings
    escalation = dataclasses.replace(
        settings.escalation, online_threshold=task.online_threshold
    )
    machine = dataclasses.replace(settings, escalation=escalation)
    point_id = task.strong_id()
    name = point_id[:YAML_POINT_NAME_LENGTH]
    return Point(name, machine, task.metadata, collection, sections)


def _refuse_an_unknown_point(
    experiment_name: str, name: str, names: list
) -> None:
    """A point name the experiment does not have, with the names it has."""
    listed = ", ".join(names)
    raise refusal.RefusalError(
        f"the experiment {experiment_name} has no point {name}; its points "
        f"are {listed}"
    )


def _refuse_two_collections(task: collect.Task, first, second) -> None:
    """The sentence for a point that two blocks collect two ways."""
    metadata = collect.metadata_text(task.metadata)
    first_text = first.text()
    second_text = second.text()
    raise refusal.RefusalError(
        f"the point {metadata} is in two sweep blocks that collect it two "
        f"ways, {first_text} and {second_text}; a point stops by one rule"
    )


def _refuse_a_path_that_is_not_a_file(path: pathlib.Path) -> None:
    """A yaml that is not there, with the shipped experiments to pick from."""
    if path.is_file():
        return
    names = _shipped_experiment_names()
    listed = ", ".join(names)
    raise refusal.RefusalError(
        f"{path} is not a file; the shipped experiments are {listed}"
    )


def _shipped_experiment_names() -> list:
    """Every runnable yaml under configs/, by its path there.

    The files under configs/bases/ are starting points other files
    extend, not runs, so they are left out.
    """
    shipped_files = CONFIGS_DIR.rglob("*.yaml")
    names = []
    for shipped in sorted(shipped_files):
        relative = shipped.relative_to(CONFIGS_DIR)
        if relative.parts[0] == BASES_FOLDER:
            continue
        names.append(str(relative))
    return names


def _point_task_of(
    settings: machine_settings.MachineSettings,
    resolved: dict,
    values: Mapping,
) -> collect.Task:
    """A point's task: its workload made, its escalation's threshold set.

    The calibration table and the online source read the point's values
    by path from its resolved sections, so a point that sweeps no
    distance or error rate still finds them.
    """
    workload = settings.workload.made()
    threshold_nats = settings.escalation.threshold_nats_for(resolved)
    escalation = dataclasses.replace(
        settings.escalation, gap_threshold_nats=threshold_nats
    )
    online_threshold = escalation.online_threshold_for(resolved)
    point_settings = dataclasses.replace(
        settings, workload=workload, escalation=escalation
    )
    metadata = copy.deepcopy(dict(values))
    return collect.Task(point_settings, metadata, online_threshold)


def _place(sections: dict, path: str, value) -> None:
    """Set an axis's value at its yaml path, under a section that exists.

    The value is copied in, so an axis under a mapping another axis
    placed edits this point's copy and not the value the sweep holds.
    """
    parent_path, _, key = path.rpartition(".")
    parent = sections
    if parent_path:
        reader = f"the sweep axis {path}"
        parent = config_module.setting_at(sections, parent_path, reader)
    if not isinstance(parent, Mapping):
        raise ValueError(
            f"the sweep axis {path} sets a key under {parent_path}, which "
            f"holds {parent!r}, not a section"
        )
    parent[key] = copy.deepcopy(value)


def _resolved(value, root: dict, chain: tuple):
    """A value with every whole-value reference replaced by what it names.

    chain is the references followed to reach this value, so a reference
    that leads back to itself is refused rather than followed forever.
    """
    if isinstance(value, Mapping):
        resolved = {}
        for key, item in value.items():
            _refuse_a_key_that_is_not_text(key, value)
            resolved[key] = _resolved(item, root, chain)
        return resolved
    if isinstance(value, list):
        items = []
        for item in value:
            resolved_item = _resolved(item, root, chain)
            items.append(resolved_item)
        return items
    return _referenced(value, root, chain)


def _referenced(value, root: dict, chain: tuple):
    """The value a whole-value reference names, or the value itself."""
    if not isinstance(value, str):
        return value
    match = WHOLE_VALUE_REFERENCE.fullmatch(value)
    if match is None:
        return value
    path = match.group(1)
    followed = chain + (path,)
    if path in chain:
        cycle = " -> ".join(followed)
        raise ValueError(
            f"the references {cycle} form a cycle; each names the next, "
            "so none has a value"
        )
    target = config_module.setting_at(root, path, f"the reference {value}")
    return _resolved(target, root, followed)


def _refuse_a_key_that_is_not_text(key, mapping: Mapping) -> None:
    """A yaml key such as 1, which json would write as the text "1".

    Python's json coerces every key to str ("Keys in key/value pairs of
    JSON are always of the type str", docs.python.org, json), and a
    point's resolved record and its strong id are json, so a key of 1
    and a key of "1" would name one point.
    """
    if isinstance(key, str):
        return
    keys = list(mapping)
    raise ValueError(
        f"the yaml key {key!r} among {keys} is not text; a point's record "
        "and its id are json, whose keys are text, so "
        f"{key!r} and {str(key)!r} would name one point"
    )


@contextlib.contextmanager
def _refused_in(path: pathlib.Path):
    """A ValueError raised inside, as one sentence naming the yaml file."""
    try:
        yield
    except ValueError as refused:
        raise refusal.RefusalError(f"{path}: {refused}") from refused


def _files_line(config: ExperimentConfig) -> str:
    """The files this config was read from, nearest first."""
    names = []
    for path in config.config_files:
        shown_path = _plain_path(path)
        names.append(str(shown_path))
    joined = " <- ".join(names)
    return f"config: {joined}"


def _section_lines(settings: machine_settings.MachineSettings) -> list:
    """One line per section, its kind named where the section has one."""
    lines = []
    for field in dataclasses.fields(settings):
        section = getattr(settings, field.name)
        kind = getattr(section, "kind", None)
        if kind is None:
            continue
        lines.append(f"{field.name}: kind {kind}")
    return lines


def _links_line(links) -> str:
    """The fabric card the run resolved to."""
    return f"links: card {links.profile_name}"


def _point_line(study: Experiment, point: Point) -> str:
    """One point's name, its metadata and its collection, as one line."""
    metadata = collect.metadata_text(point.metadata)
    collection = study.collection_of(point)
    collection_text = collection.text()
    return f"point {point.name}: {metadata}; {collection_text}"


def _sweep_block_line(index: int, block: SweepBlock) -> str:
    """One sweep block's axes and its collection, as one line."""
    axes = []
    for path, values in block.axes.items():
        values_text = json.dumps(list(values))
        axes.append(f"{path} {values_text}")
    axes_text = ", ".join(axes)
    collection_text = block.collection.text()
    return f"sweep block {index}: {axes_text}; {collection_text}"


def _observation_lines(observation) -> list:
    """What the run will record beyond its results."""
    log_line = f"log: {observation.log}"
    if observation.log_component_io:
        log_line += " with component I/O"
    return [log_line, f"trace: {observation.trace}"]


def _yaml_sections(path: pathlib.Path, chain: tuple = ()) -> tuple:
    """The file's sections with `extends` applied, their folders, its files.

    The files come this file first, then the base it extends, and so on.
    A key this file names replaces the base's key whole: a child that
    declares `sweep` ignores the base's sweep entirely. So a relative
    path in a section is the one its own file wrote, and it resolves
    against that file's folder, as a relative `extends` does. chain is
    the files followed to reach this one, so a file its own chain
    reaches again is refused rather than read forever.
    """
    followed = chain + (path,)
    _refuse_an_extends_cycle(followed)
    with open(path) as handle:
        sections = yaml.safe_load(handle)
    base_name = sections.pop("extends", None)
    folders = dict.fromkeys(sections, path.parent)
    if base_name is None:
        return sections, folders, (path,)
    base_path = path.parent / base_name
    base_sections, base_folders, base_paths = _yaml_sections(
        base_path, followed
    )
    base_sections.update(sections)
    base_folders.update(folders)
    files = (path,) + base_paths
    return base_sections, base_folders, files


def _refuse_an_extends_cycle(followed: tuple) -> None:
    """The chain, when its last file is one it already followed.

    A file is the one the filesystem opens, so two spellings of one
    file, or a link to it, are the same file.
    """
    *earlier, last = followed
    earlier_files = set()
    for path in earlier:
        real_path = os.path.realpath(path)
        earlier_files.add(real_path)
    last_file = os.path.realpath(last)
    if last_file not in earlier_files:
        return
    names = []
    for path in followed:
        plain = _plain_path(path)
        names.append(str(plain))
    cycle = " -> ".join(names)
    raise refusal.RefusalError(
        f"the extends chain {cycle} forms a cycle; each file extends the "
        "next, so the chain never reaches a file without a base"
    )


def _plain_path(path: pathlib.Path) -> pathlib.Path:
    """The path as show prints it, each `..` taken out lexically.

    configs/examples/../bases/x.yaml prints as configs/bases/x.yaml. It
    names the file for a reader only: past a symbolic link the two can
    be different files, so a file is always opened by the path as
    written, which the filesystem resolves (_yaml_sections).
    """
    plain = os.path.normpath(path)
    return pathlib.Path(plain)


def _sweep_blocks(sweep_section: list, top_collection) -> tuple:
    """One SweepBlock per block of the yaml's sweep list, in order."""
    blocks = []
    for index, block in enumerate(sweep_section, start=1):
        sweep_block = _sweep_block(block, index, top_collection)
        blocks.append(sweep_block)
    return tuple(blocks)


def _sweep_block(block: dict, index: int, top_collection) -> SweepBlock:
    """One block: its axes, each a yaml path and a list of values.

    A block's own collection keys are read over the file's section.
    """
    _check_block_keys(block, index)
    axes = block["axes"]
    _check_axes(axes, index)
    values = {}
    for path, axis_values in axes.items():
        values[path] = tuple(axis_values)
    block_collection = block.get("collection")
    where = f"sweep block {index}"
    collection = collection_module.CollectionSettings.from_yaml(
        top_collection, block_collection, where
    )
    return SweepBlock(axes=values, collection=collection)


def _check_block_keys(block, index: int) -> None:
    """A block is axes, and may carry its own collection keys."""
    if not isinstance(block, Mapping):
        _refuse_the_block(block, index)
    keys = set(block)
    required = set(SWEEP_BLOCK_KEYS)
    allowed = required | set(OPTIONAL_SWEEP_BLOCK_KEYS)
    if not required <= keys <= allowed:
        _refuse_the_block(block, index)


def _refuse_the_block(block, index: int) -> None:
    """What a sweep block is, in the sentence the user reads."""
    raise refusal.RefusalError(
        f"sweep block {index} is {block!r}; a block is axes, a mapping "
        "of yaml paths to the values the sweep sets there, and may carry "
        "a collection of its own, where max_shots is"
    )


def _check_axes(axes, index: int) -> None:
    """The axes are a mapping, and each axis lists the values it sweeps."""
    if not isinstance(axes, Mapping):
        raise refusal.RefusalError(
            f"sweep block {index} axes must map yaml paths to values, got "
            f"{axes!r}"
        )
    for path, values in axes.items():
        if not isinstance(path, str):
            raise refusal.RefusalError(
                f"sweep block {index} axis {path!r} is not a yaml path; an "
                "axis names the setting it sets by its dotted path, such as "
                "qpu.distance"
            )
        if not isinstance(values, list) or not values:
            raise refusal.RefusalError(
                f"sweep block {index} axis {path} must be a list of at least "
                f"one value, got {values!r}"
            )


@dataclasses.dataclass(frozen=True)
class _WrittenKey:
    """A yaml key a file writes: the layer it belongs to and its line."""

    layer: str
    source: str  # file:line


def _origin_of(
    key: tuple, swept: dict, written: dict, documented: dict
) -> Optional[str]:
    """The layer that set a value and its yaml line, or None.

    Only a value that is one yaml key's has an origin, so a value the
    build derives cites no line rather than a guessed one.
    """
    if _is_swept(key, swept):
        sweep_key = written[("sweep",)]
        return f"sweep, {sweep_key.source}"
    if key in written:
        written_key = written[key]
        return f"{written_key.layer}, {written_key.source}"
    if key in documented:
        documented_key = documented[key]
        return f"default, {documented_key.source}"
    return None


def _written_keys(config: ExperimentConfig) -> dict:
    """Each key the extends chain writes, with its layer and line.

    A file's top-level key replaces its base's whole (_yaml_sections),
    so a section, and every key under it, comes from the nearest file
    that names the section.
    """
    written = {}
    your_file = config.config_files[0]
    for path in reversed(config.config_files):
        layer = f"preset {path.name}"
        if path == your_file:
            layer = "your file"
        shown_path = _plain_path(path)
        file_keys = _key_lines(path, shown_path, layer)
        _forget_the_sections_of(written, file_keys)
        written.update(file_keys)
    return written


def _forget_the_sections_of(written: dict, file_keys: dict) -> None:
    """Drop the base's keys of every section a nearer file names."""
    sections = set()
    for key in file_keys:
        sections.add(key[0])
    for key in list(written):
        if key[0] in sections:
            del written[key]


def _documented_keys() -> dict:
    """Each key configs/reference.yaml writes, with its line there."""
    shown_path = REFERENCE_FILE.relative_to(_REPOSITORY_ROOT)
    return _key_lines(REFERENCE_FILE, shown_path, "default")


def _key_lines(
    path: pathlib.Path, shown_path: pathlib.Path, layer: str
) -> dict:
    """Every mapping key of a yaml file, with the lines that write it.

    A key whose value is a block over several lines (the sweep's list)
    is shown as that span. The keys under a list (the sweep's blocks)
    are the list's own key.
    """
    with open(path) as handle:
        root = yaml.compose(handle)
    keys = {}
    _add_key_lines(root, (), f"{shown_path}", layer, keys)
    return keys


def _add_key_lines(node, prefix: tuple, shown_path: str, layer: str, keys):
    """The keys of one mapping node and of every mapping below it."""
    if node.id != "mapping":
        return
    for key_node, value_node in node.value:
        key = prefix + (key_node.value,)
        lines = _lines_of(key_node, value_node)
        keys[key] = _WrittenKey(layer, f"{shown_path}:{lines}")
        _add_key_lines(value_node, key, shown_path, layer, keys)


def _lines_of(key_node, value_node) -> str:
    """The line a key is on, or the span of lines its block value covers.

    A block ends on the line of its last scalar; a block's own end mark
    runs on past the blank and comment lines after it.
    """
    first_line = key_node.start_mark.line + 1
    last_node = _last_scalar(value_node)
    last_line = last_node.end_mark.line + 1
    if last_line <= first_line:
        return f"{first_line}"
    return f"{first_line}-{last_line}"


def _last_scalar(node):
    """The last scalar node under a node, which is itself when a scalar."""
    while node.id != "scalar" and node.value:
        last_entry = node.value[-1]
        node = last_entry
        if isinstance(last_entry, tuple):
            node = last_entry[1]
    return node


def _is_swept(key: tuple, swept: dict) -> bool:
    """Whether an axis sets this key, itself or a section holding it."""
    for axis in swept:
        prefix = key[: len(axis)]
        if prefix == axis:
            return True
    return False


def _swept_values(sweep: tuple) -> dict:
    """Each axis's values across the sweep blocks, each once, in order.

    The key is the axis's yaml path as a tuple of names, the shape
    _yaml_key gives a settings path.
    """
    swept = {}
    for block in sweep:
        for path, values in block.axes.items():
            names = path.split(".")
            key = tuple(names)
            known = swept.setdefault(key, [])
            _add_new_values(known, values)
    return swept


def _add_new_values(known: list, values: tuple) -> None:
    """Append the values not known yet; a mapping value is not hashable."""
    for value in values:
        if value not in known:
            known.append(value)


def _yaml_key(path: tuple) -> tuple:
    """A settings path as the yaml writes it: a row's keys sit beside kind."""
    key = []
    for name in path:
        if name != "row_settings":
            key.append(name)
    return tuple(key)
