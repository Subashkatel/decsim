"""The experiments layer: tasks, and the shots collected from them.

sinter's shape (sinter/_data/_task.py Task, sinter/_collection/
_collection.py collect, sinter/_data/_task_stats.py the rows) adapted
to a work unit of one seeded run. A Task is one settings record at one
sweep point with json metadata; a unit runs its seeds serially, builds
and runs a Machine per seed, and hands each Shot to the caller's
measure, which returns the caller's row.
Two tasks with the same strong id (sinter's content hash, _task.py
strong_id_value: the json text of the values, sha256) are one task. The
online threshold calibrator lives on the task so every shot of a point
shares it, which is why shots stay serial inside their unit.

The work unit is one task's range of seeds, sinter's shape (a task's
shots are split into batches its workers take,
sinter/_collection/_collection_worker_state.py, capped by
--max_batch_size), and the pool (sinter's --processes,
_main_collect.py:83) runs whole units. A unit's window error models are
built by its first shot and read by the rest, as sinter compiles its
decoder once per task. A pool hands each unit on the moment it ends, so
a caller saves it while slower units run.
"""

import concurrent.futures
import dataclasses
import enum
import hashlib
import importlib.metadata
import json
import numbers
import pathlib
import resource
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any, Optional

import numpy
import stim

import decsim.config as config
import decsim.machine as machine_module
import decsim.ports as ports
import decsim.records.results as result_records
import decsim.settings as machine_settings
import decsim.windows.built_window_models as built_window_models

# The key a record's class is written under beside its fields. No field
# can take it, since class is a Python keyword.
RECORD_CLASS_KEY = "class"
# confidence_shot_count's word for every shot of a point in a piece's
# record
EVERY_SHOT = "all"


@dataclasses.dataclass(frozen=True)
class RecordOptions:
    """What the run records of a point's shots beside their results.

    confidence_shot_count is how many shots of the point, from seed 0,
    write their windows' confidence gaps to window_confidence.csv when a
    confidence signal decides the escalation; None writes every scored
    shot's. The machine does not read it, so it is no part of a point's
    id, as sinter keeps its output options out of a task's strong id
    (sinter/_data/_task.py:167-204).
    """

    confidence_shot_count: Optional[int] = 100

    def __post_init__(self) -> None:
        shot_count = self.confidence_shot_count
        if shot_count is not None and not config.is_whole_count(shot_count, 0):
            raise ValueError(
                "confidence_shot_count must be a non-negative whole number "
                f"of shots or None for every shot, got {shot_count!r}"
            )

    def samples_confidence_of(self, seed: int) -> bool:
        """Whether this seed's windows go into window_confidence.csv."""
        if self.confidence_shot_count is None:
            return True
        return seed < self.confidence_shot_count


@dataclasses.dataclass(frozen=True)
class Task:
    """One machine settings record, a sweep point, to run seeds of.

    settings are the point's, its threshold read at the point
    (MachineSettings.at_point: a calibration table's row), so the strong
    id covers the number a table gives the point. metadata is the json
    the caller wants to see beside every row (the
    sweep point). online_threshold is the point's calibrator when its
    switching threshold learns across shots: point state, built here
    once from the threshold record (for_point) and handed to every
    shot's Machine.build, as the window models are. A task given one,
    the state a resumed piece saved, keeps it. record_options are what
    the run records of the shots, which the strong id leaves out with
    the calibrator.
    """

    settings: machine_settings.MachineSettings
    metadata: Mapping[str, Any]
    online_threshold: Optional[ports.ThresholdSource] = None
    record_options: RecordOptions = RecordOptions()

    def __post_init__(self):
        _refuse_a_key_that_is_not_text(self.metadata, "metadata")
        settings = self.settings.at_point()
        object.__setattr__(self, "settings", settings)
        if self.online_threshold is None:
            calibrator = _point_calibrator(settings)
            object.__setattr__(self, "online_threshold", calibrator)

    def strong_id(self) -> str:
        """sha256 of the json text of the settings and the metadata.

        A settings field declared compare=False is a label, a name or a
        source that changes nothing the machine does, and is left out,
        as sinter leaves a task's circuit_path out of its strong id
        (sinter/_data/_task.py:157, 167-204). The point's resolved
        record keeps every label (run_folder.record_point).
        """
        value = {
            "settings": json_value(self.settings, keep_labels=False),
            "metadata": json_value(self.metadata),
        }
        text = json.dumps(value, sort_keys=True)
        encoded = text.encode("utf8")
        digest = hashlib.sha256(encoded)
        return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class Unit:
    """The work unit: `seeds` seeds of one task, starting at first_seed.

    A unit is what a worker process takes. Its seeds run serially and
    share one window error model cache.
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


def run_units(
    units: list,
    measure: Callable[[Shot], Any],
    *,
    on_unit_done: Callable[[Unit, result_records.UnitOutcome], None],
    processes: int = 1,
) -> None:
    """Every unit run, each handed on the moment it ends.

    on_unit_done takes each unit with its outcome, so a caller saves a
    unit while slower ones run, and a job killed at its time limit
    loses only the units still running. The unit it takes holds the
    task as it ran, whose online calibrator learned over the unit's
    shots, in a worker process when there is a pool. With processes
    above one, whole units run in a worker pool, `processes` at a time.
    """
    outcomes = _unit_outcomes(units, measure, processes)
    for position, outcome in outcomes:
        unit = units[position]
        ran_unit = dataclasses.replace(unit, task=outcome.task)
        on_unit_done(ran_unit, outcome)


def run_unit(
    unit: Unit, measure: Callable[[Shot], Any]
) -> result_records.UnitOutcome:
    """Every seed of one unit, measured, the task that ran them, the memory.

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
    peak_memory_mb = _peak_memory_mb()
    module_versions = imported_module_versions()
    return result_records.UnitOutcome(
        rows, unit.task, peak_memory_mb, module_versions
    )


def imported_module_versions() -> dict:
    """Each third-party top-level module this process imported, its version.

    A module's own __version__ is read first, since a folder of packages
    can hold a dist-info stale against the module beside it; a module
    that states none is named by the distribution that installed it
    (relay_bp states none). The standard library, submodules and a
    module neither names are left out.
    """
    distributions_by_module = importlib.metadata.packages_distributions()
    versions = {}
    for name in sorted(sys.modules):
        version = _module_version(name, distributions_by_module)
        if version is not None:
            versions[name] = version
    return versions


def run_shot(task: Task, seed: int, *, built_models=None) -> Shot:
    """Build the task's machine for the seed and run it, timed.

    built_models is the task's window error model cache: the first shot
    fills it and the rest read it, which is most of a shot's build time
    at a large distance. A shot run on its own passes none and builds
    its own models.
    """
    wall_start = time.perf_counter()
    machine = machine_module.Machine.build(
        task.settings, seed, built_models, task.online_threshold
    )
    result = machine.run()
    wall_end = time.perf_counter()
    wall_seconds = wall_end - wall_start
    return Shot(task, seed, machine, result, wall_seconds)


def metadata_text(metadata: Mapping[str, Any]) -> str:
    """A point's metadata as one line of json, its keys sorted.

    The text a progress line and a refusal name a point by, sinter's
    json_metadata form (sinter/_data/_csv_out.py:35-37).
    """
    value = json_value(metadata)
    return json.dumps(value, sort_keys=True)


def json_value(
    value: Any, *, keep_labels: bool = True, record_classes: bool = True
) -> Any:
    """A settings record as plain json: every value by its content.

    A dataclass (a settings record, a round policy) appears as its class
    and its fields. Every number appears exactly (a json number, or a Fraction's
    exact text), every string and flag as written, and an enum member as
    its name. A Stim circuit appears as its text, as sinter's strong id
    carries the task's circuit (sinter/_data/_task.py:193), so two tasks
    that run different circuits are two tasks. A Python-built component
    (a decoder, a device) has no record form and appears by its content
    too (_json_object). keep_labels False leaves out every dataclass
    field declared compare=False, the labels a strong id does not hash.
    record_classes False leaves out each record's class, which names a
    setting's identity and not a result's.
    """
    form = _JsonForm(keep_labels, record_classes)
    return _json_in(value, form)


def _point_calibrator(
    settings: machine_settings.MachineSettings,
) -> Optional[ports.ThresholdSource]:
    """The calibrator a point's shots share, when its threshold learns one."""
    switching = settings.switching
    if switching is None:
        return None
    facts = settings.point_facts()
    return switching.threshold.for_point(facts)


@dataclasses.dataclass(frozen=True)
class _JsonForm:
    """What json_value writes beside a record's compared fields."""

    keep_labels: bool
    record_classes: bool


def _json_in(value: Any, form: _JsonForm) -> Any:
    """One value walked in the form json_value was asked for."""
    if dataclasses.is_dataclass(value):
        return _json_record(value, form)
    if isinstance(value, Mapping):
        return _json_mapping(value, form)
    if isinstance(value, (list, tuple)):
        return _json_list(value, form)
    return _json_scalar(value, form)


def _peak_memory_mb() -> float:
    """This process's peak resident memory so far, in megabytes.

    getrusage's ru_maxrss is in kilobytes on Linux and in bytes on macOS
    (getrusage(2) on each).
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    kilobytes = usage.ru_maxrss
    if sys.platform == "darwin":
        kilobytes = usage.ru_maxrss / 1024
    return kilobytes / 1024


def _module_version(
    name: str, distributions_by_module: Mapping
) -> Optional[str]:
    """One module's version, or None for a module the record leaves out."""
    if "." in name or name in sys.stdlib_module_names:
        return None
    module = sys.modules[name]
    stated = getattr(module, "__version__", None)
    if isinstance(stated, str):
        return stated
    distributions = distributions_by_module.get(name)
    if not distributions:
        return None
    return importlib.metadata.version(distributions[0])


def _unit_outcomes(units: list, measure: Callable[[Shot], Any], processes: int):
    """Each unit's position and outcome as it ends, run here or in a pool."""
    if processes <= 1:
        for position, unit in enumerate(units):
            outcome = run_unit(unit, measure)
            yield position, outcome
        return
    yield from _pooled_outcomes(units, measure, processes)


def _pooled_outcomes(
    units: list, measure: Callable[[Shot], Any], processes: int
):
    """Each unit's position and outcome as it ends, `processes` at a time.

    A unit is handed to the pool only when one ends, so what a killed
    job loses is at most the units then running (concurrent.futures.wait
    with FIRST_COMPLETED). The pool's queued work is dropped if the
    caller stops reading.
    """
    queued = enumerate(units)
    running = {}
    pool = concurrent.futures.ProcessPoolExecutor(processes)
    try:
        _submit_up_to(pool, running, queued, measure, processes)
        while running:
            yield from _ended_outcomes(running)
            _submit_up_to(pool, running, queued, measure, processes)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _ended_outcomes(running: dict):
    """The running units that have ended, each position and outcome, taken out.

    It waits until at least one has ended.
    """
    ended, _still_running = concurrent.futures.wait(
        running, return_when=concurrent.futures.FIRST_COMPLETED
    )
    for future in ended:
        position = running.pop(future)
        outcome = future.result()
        yield position, outcome


def _submit_up_to(
    pool, running: dict, queued, measure: Callable[[Shot], Any], limit: int
) -> None:
    """Queued units handed to the pool until `limit` of them run."""
    while len(running) < limit:
        queued_unit = next(queued, None)
        if queued_unit is None:
            return
        position, unit = queued_unit
        future = pool.submit(run_unit, unit, measure)
        running[future] = position


def _json_record(record: Any, form: _JsonForm) -> dict:
    """A dataclass as its class and fields, each one walked; labels when kept.

    The class sits beside the fields, as gem5's config.json writes each
    object's type (src/python/m5/SimObject.py:1175-1178), so two records
    with the same fields, two rows that take no settings, are two points.
    """
    fields = {}
    if form.record_classes:
        record_class = type(record)
        fields[RECORD_CLASS_KEY] = _qualified_name(record_class)
    for field in dataclasses.fields(record):
        if not field.compare and not form.keep_labels:
            continue
        field_value = getattr(record, field.name)
        fields[field.name] = _json_in(field_value, form)
    return fields


def _qualified_name(named_class: type) -> str:
    """A class's module and qualified name, the identity pickle writes."""
    return f"{named_class.__module__}.{named_class.__qualname__}"


def _json_mapping(mapping: Mapping, form: _JsonForm) -> dict:
    """A mapping with its keys as text and its values walked."""
    items = {}
    for key, item in mapping.items():
        items[str(key)] = _json_in(item, form)
    return items


def _refuse_a_key_that_is_not_text(value: Any, where: str) -> None:
    """The id is json of the metadata, whose keys are text, at any depth.

    A point's metadata enters here, so 1 and "1" cannot name one point.
    """
    if isinstance(value, (list, tuple)):
        for item in value:
            _refuse_a_key_that_is_not_text(item, where)
        return
    if not isinstance(value, Mapping):
        return
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(
                f"{where} holds the key {key!r}, which is not text; a "
                "point's id is the json of its metadata, whose keys are "
                f"text, so {key!r} and {str(key)!r} would name one point"
            )
        _refuse_a_key_that_is_not_text(item, f"{where}.{key}")


def _json_list(sequence, form: _JsonForm) -> list:
    """A list or a tuple, each item walked."""
    items = []
    for item in sequence:
        json_item = _json_in(item, form)
        items.append(json_item)
    return items


def _json_scalar(value: Any, form: _JsonForm) -> Any:
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
    return _json_object(value, form)


def _json_object(value: Any, form: _JsonForm) -> Any:
    """A Python-built component: its class, and its attributes walked.

    Two instances that hold different values are two tasks. The id is
    taken before a shot binds the component's neighbours onto it. An
    array (recorded measurements) is its values; a value with no
    attributes of its own (a lock) is its class. A class given as a value
    (the scheduler rule) is its module and qualified name, the identity
    pickle writes for a class (Lib/pickle.py save_global, 1056-1113): its
    attributes are code, and Python adds __annotations__ to a class the
    first time it is read.
    """
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, type):
        return _qualified_name(value)
    value_type = type(value)
    class_name = _qualified_name(value_type)
    attributes = _attributes_of(value)
    if not attributes:
        return class_name
    content = _json_mapping(attributes, form)
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
