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
shares it, which is why shots are serial; a worker pool is a later
commit.
"""

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
    sweep point). threshold_calibrator is the point's online escalation
    controller when the escalation asks for one, built once here and
    installed on every shot's settings.
    """

    settings: machine_module.MachineSettings
    shots: int
    metadata: Mapping[str, Any]
    threshold_calibrator: Optional[Any] = None

    @classmethod
    def at_point(
        cls,
        settings: machine_module.MachineSettings,
        shots: int,
        metadata: Mapping[str, Any],
    ) -> "Task":
        """The task, with the point's online calibrator when there is one."""
        physical_error_probability = (
            settings.workload.physical_error_probability
        )
        distance = settings.qpu.distance
        calibrator = settings.escalation.online_calibrator(
            physical_error_probability, distance
        )
        return cls(settings, shots, metadata, calibrator)

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
        """The settings one shot runs: the point's calibrator installed."""
        if self.threshold_calibrator is None:
            return self.settings
        escalation = dataclasses.replace(
            self.settings.escalation,
            threshold_calibrator=self.threshold_calibrator,
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
) -> list:
    """Every shot of every task, measured; one row per shot, in order.

    A task named twice runs once, with the larger shot count (seeds are
    0 to shots - 1, so the larger count covers the smaller). on_task_done
    runs after a task's last shot, sinter's progress_callback.
    """
    rows = []
    for task in unique_tasks(tasks):
        for seed in range(task.shots):
            shot = run_shot(task, seed)
            row = measure(shot)
            rows.append(row)
        if on_task_done is not None:
            on_task_done(task)
    return rows


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
