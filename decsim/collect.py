"""The experiments layer: tasks, and the shots collected from them.

sinter's shape (sinter/_data/_task.py Task, sinter/_collection/
_collection.py collect, sinter/_data/_task_stats.py the rows) adapted
to a work unit of one seeded run. A Task is one settings record at one
sweep point with a shot count and json metadata; collect runs seeds 0 to
shots - 1 serially per task, builds and runs a Machine per seed, and
hands each Shot to the caller's measure, which returns the caller's row.
Two tasks with the same strong id (sinter's content hash, _task.py
strong_id_value: the json text of the values, sha256) are one task. The
online threshold calibrator lives on the task so every shot of a point
shares it, which is why shots stay serial inside their unit.

The work unit is one task's range of seeds, sinter's shape (a task's
shots are split into batches its workers take,
sinter/_collection/_collection_worker_state.py, capped by
--max_batch_size): the pool (sinter's --processes,
_main_collect.py:83) and the shard both run over units, so a point of a
million shots fits a Slurm array's wall clock instead of one task
having to. A unit's window error models are built by its first shot and
read by the rest, as sinter compiles its decoder once per task. Rows
come back in unit order whatever order the units finish in, so a run
that splits nothing writes its rows in task and seed order.
"""

import concurrent.futures
import dataclasses
import enum
import hashlib
import json
import numbers
import pathlib
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Optional

import numpy
import stim

import decsim.machine as machine_module
import decsim.records.results as result_records
import decsim.settings as machine_settings
import decsim.windows.built_window_models as built_window_models


@dataclasses.dataclass(frozen=True)
class Task:
    """One machine settings record to run `shots` seeds of.

    metadata is the json the caller wants to see beside every row (the
    sweep point). online_threshold is the point's online threshold
    source when the escalation asks for one, built once here and
    installed on every shot's settings.
    """

    settings: machine_settings.MachineSettings
    shots: int
    metadata: Mapping[str, Any]
    online_threshold: Optional[Any] = None

    def strong_id(self) -> str:
        """sha256 of the json text of the settings and the metadata."""
        value = {
            "settings": json_value(self.settings),
            "metadata": json_value(self.metadata),
        }
        text = json.dumps(value, sort_keys=True)
        encoded = text.encode("utf8")
        digest = hashlib.sha256(encoded)
        return digest.hexdigest()

    def shot_settings(
        self, built_models=None
    ) -> machine_settings.MachineSettings:
        """The settings one shot runs: the point's threshold and models."""
        settings = self.settings
        if self.online_threshold is not None:
            escalation = dataclasses.replace(
                settings.escalation, online_threshold=self.online_threshold
            )
            settings = dataclasses.replace(settings, escalation=escalation)
        if built_models is not None:
            workload = dataclasses.replace(
                settings.workload, built_models=built_models
            )
            settings = dataclasses.replace(settings, workload=workload)
        return settings


@dataclasses.dataclass(frozen=True)
class Unit:
    """The work unit: `seeds` seeds of one task, starting at first_seed.

    A unit is what a worker process takes and what a shard keeps or
    skips. Its seeds run serially and share one window error model
    cache.
    """

    task: Task
    first_seed: int
    seeds: int


@dataclasses.dataclass(frozen=True)
class Shot:
    """One seeded run of a task: the machine, its result, its wall time."""

    task: Task
    seed: int
    machine: machine_module.Machine
    result: result_records.RunResult
    wall_seconds: float


def collect(
    tasks: Iterable[Task],
    measure: Callable[[Shot], Any],
    on_task_done: Optional[Callable[[Task], None]] = None,
    *,
    processes: int = 1,
    shard: Optional[tuple] = None,
    shots_per_unit: Optional[int] = None,
) -> list:
    """Every shot of every task, measured; one row per shot, in order.

    A task named twice runs once, with the larger shot count (seeds are
    0 to shots - 1, so the larger count covers the smaller).
    `shots_per_unit` splits a task's seeds into units of that many, so
    the pool and the shard divide a point's shots as well as its points;
    with none, a task is one unit. on_task_done runs after the last unit
    of a task that this run holds, sinter's progress_callback. `shard`
    is (index, count) and keeps the units whose position modulo count is
    index; `processes` above one runs whole units in a worker pool, so
    `measure` must be a module-level callable or a partial of one.
    """
    unique = unique_tasks(tasks)
    units = work_units(unique, shots_per_unit)
    selected = shard_of(units, shard)
    if processes > 1:
        return _collected_in_a_pool(selected, measure, on_task_done, processes)
    rows = []
    for position, unit in enumerate(selected):
        unit_rows, ran = run_unit(unit, measure)
        rows.extend(unit_rows)
        _report_a_finished_task(on_task_done, selected, position, ran)
    return rows


def work_units(tasks: list, shots_per_unit: Optional[int] = None) -> list:
    """Every task's seeds as work units, each task's units in seed order.

    A point whose escalation calibrates its threshold online is one unit
    however small `shots_per_unit` is: the calibrator learns over the
    point's shots in order, so splitting them would split its state.
    """
    units = []
    for task in tasks:
        for unit in _units_of_one_task(task, shots_per_unit):
            units.append(unit)
    return units


def shard_of(units: list, shard: Optional[tuple]) -> list:
    """The units of one shard: position modulo count equals index.

    The command checks i and n where it reads them
    (decsim/experiments/command.py _shard_of), before a run folder exists.
    """
    if shard is None:
        return units
    index, count = shard
    selected = []
    for position, unit in enumerate(units):
        if position % count == index:
            selected.append(unit)
    return selected


def run_unit(unit: Unit, measure: Callable[[Shot], Any]) -> tuple:
    """Every seed of one unit, measured, and the task that ran them.

    The task comes back because its online threshold calibrator learned
    over these shots, and in a pool that learning happened in another
    process. The window error models are built here and not on the task,
    so a worker's models never travel back through a pickle.
    """
    built_models = built_window_models.BuiltWindowModels()
    rows = []
    past_the_last_seed = unit.first_seed + unit.seeds
    for seed in range(unit.first_seed, past_the_last_seed):
        shot = run_shot(unit.task, seed, built_models=built_models)
        row = measure(shot)
        rows.append(row)
    return rows, unit.task


def unique_tasks(tasks: Iterable[Task]) -> list:
    """The tasks in first-seen order, same strong id merged to one.

    The merged task keeps the first block's calibrator, so an online
    switching point named in two blocks calibrates once over all its
    shots. No shipped or
    frozen yaml names an online point twice.
    """
    task_by_id = {}
    for task in tasks:
        strong_id = task.strong_id()
        if strong_id not in task_by_id:
            task_by_id[strong_id] = task
            continue
        earlier = task_by_id[strong_id]
        if task.shots > earlier.shots:
            task_by_id[strong_id] = dataclasses.replace(
                earlier, shots=task.shots
            )
    unique = task_by_id.values()
    return list(unique)


def run_shot(task: Task, seed: int, *, built_models=None) -> Shot:
    """Build the task's machine for the seed and run it, timed.

    built_models is the task's window error model cache: the first shot
    fills it and the rest read it, which is most of a shot's build time
    at a large distance. A shot run on its own passes none and builds
    its own models.
    """
    settings = task.shot_settings(built_models)
    wall_start = time.perf_counter()
    machine = machine_module.Machine.build(settings, seed)
    result = machine.run()
    wall_end = time.perf_counter()
    wall_seconds = wall_end - wall_start
    return Shot(task, seed, machine, result, wall_seconds)


def metadata_text(metadata: Mapping[str, Any]) -> str:
    """A point's metadata as one line of json, its keys sorted.

    The text a progress line and `decsim diff` name a point by, sinter's
    json_metadata form (sinter/_data/_csv_out.py:35-37), so two folders'
    records of one point name it alike.
    """
    value = json_value(metadata)
    return json.dumps(value, sort_keys=True)


def json_value(value: Any) -> Any:
    """A settings record as plain json: every value by its content.

    A dataclass (a settings record, a round policy) appears as its
    fields. Every number appears exactly (a json number, or a Fraction's
    exact text), every string and flag as written, and an enum member as
    its name. A Stim circuit appears as its text, as sinter's strong id
    carries the task's circuit (sinter/_data/_task.py:193), so two tasks
    that run different circuits are two tasks. A Python-built component
    (a decoder, a device) has no record form and appears by its content
    too (_json_object).
    """
    if dataclasses.is_dataclass(value):
        return _json_record(value)
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, (list, tuple)):
        return _json_list(value)
    return _json_scalar(value)


def _units_of_one_task(task: Task, shots_per_unit: Optional[int]) -> list:
    """One task's seeds cut into units of at most `shots_per_unit`."""
    seeds_in_a_unit = task.shots
    if shots_per_unit is not None and task.online_threshold is None:
        seeds_in_a_unit = min(shots_per_unit, task.shots)
    units = []
    for first_seed in range(0, task.shots, seeds_in_a_unit):
        remaining = task.shots - first_seed
        seeds = min(seeds_in_a_unit, remaining)
        unit = Unit(task, first_seed, seeds)
        units.append(unit)
    return units


def _report_a_finished_task(
    on_task_done: Optional[Callable[[Task], None]],
    units: list,
    position: int,
    ran: Task,
) -> None:
    """The progress callback, once per task, after its last unit here."""
    if on_task_done is None:
        return
    if not _is_a_tasks_last_unit(units, position):
        return
    on_task_done(ran)


def _is_a_tasks_last_unit(units: list, position: int) -> bool:
    """Whether the next unit of this list belongs to another task.

    work_units keeps a task's units together and shard_of keeps a
    subsequence of that order, so a task's last unit in a list is the
    one whose successor does not share its task.
    """
    next_position = position + 1
    if next_position == len(units):
        return True
    unit = units[position]
    later = units[next_position]
    return later.task is not unit.task


def _collected_in_a_pool(
    units: list,
    measure: Callable[[Shot], Any],
    on_task_done: Optional[Callable[[Task], None]],
    processes: int,
) -> list:
    """Whole units in worker processes; the rows read back in unit order."""
    futures = _submitted(units, measure, processes)
    rows = []
    for position, future in enumerate(futures):
        unit_rows, ran = future.result()
        rows.extend(unit_rows)
        _report_a_finished_task(on_task_done, units, position, ran)
    return rows


def _submitted(
    units: list, measure: Callable[[Shot], Any], processes: int
) -> list:
    """Every unit handed to the pool, in unit order.

    The pool is shut down when the last future has been read, which the
    caller does in this same order, so a unit's rows are appended where
    the unit sits and not where it finished.
    """
    pool = concurrent.futures.ProcessPoolExecutor(processes)
    futures = []
    for unit in units:
        future = pool.submit(run_unit, unit, measure)
        futures.append(future)
    pool.shutdown(wait=False)
    return futures


def _json_record(record: Any) -> dict:
    """A dataclass as its fields, each one walked."""
    fields = {}
    for field in dataclasses.fields(record):
        field_value = getattr(record, field.name)
        fields[field.name] = json_value(field_value)
    return fields


def _json_mapping(mapping: Mapping) -> dict:
    """A mapping with its keys as text and its values walked."""
    items = {}
    for key, item in mapping.items():
        items[str(key)] = json_value(item)
    return items


def _json_list(sequence) -> list:
    """A list or a tuple, each item walked."""
    items = []
    for item in sequence:
        json_item = json_value(item)
        items.append(json_item)
    return items


def _json_scalar(value: Any) -> Any:
    """A number exactly, a string, flag or path as written; others named."""
    if isinstance(value, (numbers.Number, numpy.generic)):
        return _json_number(value)
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, stim.Circuit):
        return str(value)
    return _json_object(value)


def _json_object(value: Any) -> Any:
    """A Python-built component: its class, and its attributes walked.

    Two instances that hold different values are two tasks. The id is
    taken before a shot binds the component's neighbours onto it. An
    array (recorded measurements) is its values; a value with no
    attributes of its own (a lock) is its class.
    """
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    value_type = type(value)
    class_name = f"{value_type.__module__}.{value_type.__qualname__}"
    attributes = _attributes_of(value)
    if not attributes:
        return class_name
    content = _json_mapping(attributes)
    return {"class": class_name, "attributes": content}


def _attributes_of(value: Any) -> dict:
    """Its __dict__, and every __slots__ name its classes declare and set."""
    own = getattr(value, "__dict__", {})
    attributes = dict(own)
    value_type = type(value)
    for name in _slot_names(value_type):
        if hasattr(value, name):
            attributes[name] = getattr(value, name)
    return attributes


def _slot_names(value_type: type) -> list:
    """The __slots__ names of a class and its bases, a lone string as one."""
    names = []
    for value_class in value_type.__mro__:
        declared = vars(value_class)
        slots = declared.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        names.extend(slots)
    return [name for name in names if name not in ("__dict__", "__weakref__")]


def _json_number(value: Any) -> Any:
    """A number as json holds it, or as its exact text when json cannot.

    A numpy scalar is the Python number it holds. A bool, int or float
    is a json number already; any other number (a Fraction link rate, a
    Decimal) is its exact text, "80/11" or "1.10", the value Python's
    own pickle keeps (Fraction.__reduce__ is the numerator and the
    denominator, Decimal.__reduce__ its string) and the text Fraction
    and Decimal read back. sinter's json.dumps refuses such a value and
    dask's tokenize pickles it; neither lets two values share an id.
    """
    if isinstance(value, numpy.generic):
        value = value.item()
    if isinstance(value, (bool, int, float)):
        return value
    return str(value)
