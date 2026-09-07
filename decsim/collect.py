"""The front: tasks, and the shots collected from them.

sinter's shape (sinter/_data/_task.py Task, sinter/_collection/
_collection.py collect, sinter/_data/_task_stats.py the rows) adapted
to a work unit of one seeded run. A Task is one settings record at one
sweep point with a shot count and json metadata; collect runs seeds 0 to
shots - 1 serially per task, builds and runs a Machine per seed, and
hands each Shot to the caller's measure, which returns the caller's row.
Two tasks with the same strong id (sinter's content hash, _task.py
strong_id_value: the json text of the values, sha256) are one task. The
online threshold calibrator lives on the task so every shot of a point
shares it, which is why shots stay serial and the process pool is over
tasks (sinter's --processes, _main_collect.py:83); rows come back in
task order whatever order the tasks finish in. A shard runs the tasks
whose index modulo n is i, so a Slurm array covers a sweep and
`decsim combine` folds the folders.
"""

import concurrent.futures
import dataclasses
import hashlib
import json
import pathlib
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Optional

import decsim.machine as machine_module


@dataclasses.dataclass(frozen=True)
class Task:
    """One machine settings record to run `shots` seeds of.

    metadata is the json the caller wants to see beside every row (the
    sweep point). online_threshold is the point's online threshold
    source when the escalation asks for one, built once here and
    installed on every shot's settings.
    """

    settings: machine_module.MachineSettings
    shots: int
    metadata: Mapping[str, Any]
    online_threshold: Optional[Any] = None

    @classmethod
    def at_point(
        cls,
        settings: machine_module.MachineSettings,
        shots: int,
        metadata: Mapping[str, Any],
    ) -> "Task":
        """The task, with the point's online threshold when there is one."""
        physical_error_probability = (
            settings.workload.physical_error_probability
        )
        distance = settings.qpu.distance
        online_threshold = settings.escalation.online_threshold_for(
            physical_error_probability, distance
        )
        return cls(settings, shots, metadata, online_threshold)

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

    def shot_settings(self) -> machine_module.MachineSettings:
        """The settings one shot runs, with the point's online threshold."""
        if self.online_threshold is None:
            return self.settings
        escalation = dataclasses.replace(
            self.settings.escalation, online_threshold=self.online_threshold
        )
        return dataclasses.replace(self.settings, escalation=escalation)


@dataclasses.dataclass(frozen=True)
class Shot:
    """One seeded run of a task: the machine, its result, its wall time."""

    task: Task
    seed: int
    machine: machine_module.Machine
    result: machine_module.RunResult
    wall_seconds: float


def collect(
    tasks: Iterable[Task],
    measure: Callable[[Shot], Any],
    on_task_done: Optional[Callable[[Task], None]] = None,
    *,
    processes: int = 1,
    shard: Optional[tuple] = None,
) -> list:
    """Every shot of every task, measured; one row per shot, in order.

    A task named twice runs once, with the larger shot count (seeds are
    0 to shots - 1, so the larger count covers the smaller). on_task_done
    runs after a task's last shot, sinter's progress_callback. `shard`
    is (index, count) and keeps the tasks whose position modulo count is
    index; `processes` above one runs whole tasks in a worker pool, so
    `measure` must be a module-level callable or a partial of one.
    """
    unique = unique_tasks(tasks)
    selected = shard_of(unique, shard)
    if processes > 1:
        return _collected_in_a_pool(selected, measure, on_task_done, processes)
    rows = []
    for task in selected:
        task_rows, ran = run_task(task, measure)
        rows.extend(task_rows)
        if on_task_done is not None:
            on_task_done(ran)
    return rows


def shard_of(tasks: list, shard: Optional[tuple]) -> list:
    """The tasks of one shard: position modulo count equals index."""
    if shard is None:
        return tasks
    index, count = shard
    is_countable = count >= 1
    is_in_range = 0 <= index < count
    if not is_countable or not is_in_range:
        raise ValueError(
            f"shard {index}/{count} is not a shard; write i/n with n at "
            "least 1 and i between 0 and n - 1"
        )
    selected = []
    for position, task in enumerate(tasks):
        if position % count == index:
            selected.append(task)
    return selected


def run_task(task: Task, measure: Callable[[Shot], Any]) -> tuple:
    """Every seed of one task, measured, and the task that ran them.

    The task comes back because its online threshold calibrator learned
    over these shots, and in a pool that learning happened in another
    process.
    """
    rows = []
    for seed in range(task.shots):
        shot = run_shot(task, seed)
        row = measure(shot)
        rows.append(row)
    return rows, task


def _collected_in_a_pool(
    tasks: list,
    measure: Callable[[Shot], Any],
    on_task_done: Optional[Callable[[Task], None]],
    processes: int,
) -> list:
    """Whole tasks in worker processes; the rows read back in task order."""
    futures = _submitted(tasks, measure, processes)
    rows = []
    for future in futures:
        task_rows, ran = future.result()
        rows.extend(task_rows)
        if on_task_done is not None:
            on_task_done(ran)
    return rows


def _submitted(
    tasks: list, measure: Callable[[Shot], Any], processes: int
) -> list:
    """Every task handed to the pool, in task order.

    The pool is shut down when the last future has been read, which the
    caller does in this same order, so a task's rows are appended where
    the task sits and not where it finished.
    """
    pool = concurrent.futures.ProcessPoolExecutor(processes)
    futures = []
    for task in tasks:
        future = pool.submit(run_task, task, measure)
        futures.append(future)
    pool.shutdown(wait=False)
    return futures


def unique_tasks(tasks: Iterable[Task]) -> list:
    """The tasks in first-seen order, same strong id merged to one.

    The merged task keeps the first block's calibrator, so an online
    switching point named in two blocks calibrates once over all its
    shots; the old runner calibrated once per block. No shipped or
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


def run_shot(task: Task, seed: int) -> Shot:
    """Build the task's machine for the seed and run it, timed."""
    settings = task.shot_settings()
    wall_start = time.perf_counter()
    machine = machine_module.Machine.build(settings, seed)
    result = machine.run()
    wall_end = time.perf_counter()
    wall_seconds = wall_end - wall_start
    return Shot(task, seed, machine, result, wall_seconds)


def json_value(value: Any) -> Any:
    """A settings record as plain json: dataclasses walked, objects named.

    A Python-built component (a decoder, a policy) has no yaml text, so
    it appears as its class name; every number, string and flag appears
    as written.
    """
    if dataclasses.is_dataclass(value):
        fields = {}
        for field in dataclasses.fields(value):
            field_value = getattr(value, field.name)
            fields[field.name] = json_value(field_value)
        return fields
    if isinstance(value, Mapping):
        items = {}
        for key, item in value.items():
            items[str(key)] = json_value(item)
        return items
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            json_item = json_value(item)
            items.append(json_item)
        return items
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, pathlib.Path):
        return str(value)
    value_type = type(value)
    return value_type.__name__
