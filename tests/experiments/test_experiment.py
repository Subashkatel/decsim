"""The Python experiment a run file defines (experiment.py).

The shape is gem5 MultiSim's, a script adding named simulators
(src/python/gem5/utils/multisim/multisim.py:285-353), and sinter's task,
settings and json metadata whose hash is its id
(sinter/_data/_task.py:167-204).
"""

import ast
import dataclasses
import os
import pathlib
import random
import re

import pytest

import decsim.confidence.complementary as complementary
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.collect as collect
import decsim.experiments.collection as collection
import decsim.experiments.experiment as experiment
import decsim.experiments.refusal as refusal
import decsim.frontends.settings as workload_settings
import decsim.producers as producers
import tests.experiments.run_files as run_files

CAPPED = collection.CollectionSettings(max_shots=4)
# a switching task whose threshold learns online from 15 dB
ONLINE = {"machine": "switching", "machine_arguments": {"online": {}}}
EXPERIMENT_FILE = pathlib.Path(experiment.__file__)
PACKAGE_DIR = EXPERIMENT_FILE.parent.parent
# The names that keep the word point: the latency points of a window's
# path (measure.POINTS), and another library's entry point or code point.
OTHER_POINT_NAMES = frozenset(
    {
        "POINTS",
        "ROUND_POINTS",
        "TIER_SPLIT_POINT",
        "window_points_us",
        "_points_held",
        "_add_latency_point_columns",
    }
)
OTHER_POINT_PHRASES = ("entry_point", "code_point")
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _minimal_settings():
    task = run_files.first_task()
    return task.machine


def _online_task_at_distance_5(workload) -> collect.Task:
    """A Python switching task on the online threshold, running workload."""
    machine = _minimal_settings()
    confidence = complementary.ComplementaryGap.Settings()
    card = threshold_sources.OnlineThreshold.Settings(threshold_decibels=15.0)
    switching = escalation_settings.SwitchingSettings(confidence, card)
    qpu = dataclasses.replace(machine.qpu, distance=5)
    running = workload_settings.WorkloadSettings.running(workload)
    machine = dataclasses.replace(
        machine,
        qpu=qpu,
        workload=running,
        strong_decoder=machine.weak_decoder,
        switching=switching,
    )
    return collect.Task("online", machine)


def test_a_grid_runs_the_last_axis_fastest_in_the_order_given():
    cells = experiment.grid(
        distance=(3, 5), physical_error_probability=(0.1, 0.2)
    )

    assert cells == [
        {"distance": 3, "physical_error_probability": 0.1},
        {"distance": 3, "physical_error_probability": 0.2},
        {"distance": 5, "physical_error_probability": 0.1},
        {"distance": 5, "physical_error_probability": 0.2},
    ]


def test_two_tasks_of_one_name_are_refused():
    settings = _minimal_settings()
    first = collect.Task("d3", settings, {"distance": 3})
    second = collect.Task("d3", settings, {"distance": 5})

    with pytest.raises(ValueError) as refused:
        experiment.Experiment("study", [first, second], CAPPED)

    message = str(refused.value)
    assert message.startswith("two tasks are named d3")


def test_a_task_name_that_is_no_folder_name_is_refused():
    settings = _minimal_settings()

    with pytest.raises(ValueError) as refused:
        collect.Task("d3/p1", settings)

    assert "is not a folder name" in str(refused.value)


def test_no_name_in_the_package_calls_a_task_a_point():
    """One configuration of an experiment is a task, sinter's word.

    A name with the word point is a latency point's or another library's.
    Names, imports and json keys are read, so a second word for a task
    does not come back.
    """
    point_names = []
    module_paths = PACKAGE_DIR.rglob("*.py")
    for path in sorted(module_paths):
        point_names += _point_names_in(path)

    assert point_names == []


def test_an_experiment_with_no_tasks_is_refused():
    with pytest.raises(ValueError) as refused:
        experiment.Experiment("study", [], CAPPED)

    assert str(refused.value) == "the experiment study has no tasks"


def test_a_task_no_collection_stops_is_refused():
    settings = _minimal_settings()
    task = collect.Task("d3", settings)

    with pytest.raises(ValueError) as refused:
        experiment.Experiment("study", [task])

    message = str(refused.value)
    assert message.startswith("the task d3 has no collection")


def test_a_tasks_own_collection_wins_over_the_experiments():
    settings = _minimal_settings()
    own = collection.CollectionSettings(max_shots=9)
    first = collect.Task("own", settings, {"distance": 3}, own)
    second = collect.Task("shared", settings, {"distance": 5})
    study = experiment.Experiment("study", [first, second], CAPPED)

    assert study.collection_of(first) == own
    assert study.collection_of(second) == CAPPED


def test_an_unknown_task_name_is_refused_with_the_names():
    settings = _minimal_settings()
    task = collect.Task("d3", settings)
    study = experiment.Experiment("study", [task], CAPPED)

    with pytest.raises(refusal.RefusalError) as refused:
        study.task_named("d5")

    assert str(refused.value) == (
        "the experiment study has no task d5; its tasks are d3"
    )


def test_a_python_task_builds_its_online_calibrator():
    """The calibrator is task state, built with the task."""
    online_task = run_files.first_task(**ONLINE)

    task = collect.Task("online", online_task.machine)

    calibrator = task.online_threshold

    expected = random.Random("online-threshold d=3 p=0.001")
    generator = calibrator.random_generator
    assert generator.getstate() == expected.getstate()


def test_a_python_task_states_its_distance_and_error_probability_once():
    """The threshold record names no fact; the seed reads the task's.

    The distance is the qpu's and the error probability the one the
    workload was made at, so the online card restates neither.
    """
    workload = producers.memory_circuit(
        "surface_code:rotated_memory_x", 6, 5, 0.002
    )

    task = _online_task_at_distance_5(workload)

    calibrator = task.online_threshold

    expected = random.Random("online-threshold d=5 p=0.002")
    generator = calibrator.random_generator
    assert generator.getstate() == expected.getstate()


def test_a_python_workload_that_states_no_probability_is_refused_by_name():
    """A workload built in Python must state its probability itself."""
    made = producers.memory_circuit(
        "surface_code:rotated_memory_x", 6, 5, 0.002
    )
    workload = dataclasses.replace(made, physical_error_probability=None)

    with pytest.raises(ValueError, match="physical_error_probability=None"):
        _online_task_at_distance_5(workload)


def test_a_run_file_is_loaded_by_its_experiment(tmp_path):
    run_file = tmp_path / "study.py"
    run_file.write_text(
        "import decsim.experiments.collect as collect\n"
        "import decsim.experiments.collection as collection\n"
        "import decsim.experiments.experiment as experiment\n"
        "import tests.experiments.run_files as run_files\n"
        "_settings = run_files.minimal_machine()\n"
        "experiment = experiment.Experiment(\n"
        "    'study',\n"
        "    [collect.Task('d3', _settings, {'d': 3})],\n"
        "    collection.CollectionSettings(max_shots=2),\n"
        ")\n"
    )

    study = experiment.load(run_file)

    assert study.name == "study"
    assert [task.name for task in study.tasks] == ["d3"]


def test_a_run_file_rewritten_within_its_second_runs_its_new_text(tmp_path):
    """No cached module stands in for the run file's text.

    Python trusts a cached module whose source keeps its size and its
    whole-second mtime (PEP 552), so a cap raised in place, at the same
    length and within the second, would otherwise run the old cap.
    """
    run_path = run_files.write_run_file(tmp_path, collection={"max_shots": 2})
    os.utime(run_path, (1_000_000_000, 1_000_000_000))
    experiment.load(run_path)
    run_files.write_run_file(tmp_path, collection={"max_shots": 4})
    os.utime(run_path, (1_000_000_000, 1_000_000_000))

    study = experiment.load(run_path)

    first_task = study.tasks[0]
    collection = study.collection_of(first_task)
    assert collection.max_shots == 4


def test_a_run_file_that_defines_no_experiment_is_refused(tmp_path):
    run_file = tmp_path / "empty.py"
    run_file.write_text("value = 1\n")

    with pytest.raises(refusal.RefusalError) as refused:
        experiment.load(run_file)

    message = str(refused.value)
    assert message.endswith(
        "defines no experiment; a run file sets experiment = "
        "Experiment(...) at module level"
    )


def test_an_online_task_with_a_target_is_refused_when_built():
    """The stop it cannot keep is refused before anything is planned."""
    study = run_files.sweep(**ONLINE)
    (task,) = study.tasks
    target = collection.CollectionSettings(max_shots=10, max_failures=5)
    targeted = dataclasses.replace(task, collection=target)

    with pytest.raises(refusal.RefusalError, match="max_shots alone"):
        experiment.Experiment("online", [targeted])


def _point_names_in(path: pathlib.Path) -> list:
    """The module's names that call a task a point, each after its file."""
    names = _names_in(path)
    point_names = []
    for name in sorted(names):
        if _calls_a_task_a_point(name):
            point_names.append(f"{path.name}: {name}")
    return point_names


def _names_in(path: pathlib.Path) -> set:
    """Every name a module defines, reads, passes, imports or quotes."""
    source = path.read_text()
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        name = _name_of(node)
        imported_or_quoted = _imported_or_quoted_name_of(node)
        names.update((name, imported_or_quoted))
    names.discard("")
    return names


def _name_of(node: ast.AST) -> str:
    """The name the node holds, or an empty string when it holds none."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, (ast.arg, ast.keyword)):
        return node.arg or ""
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    return ""


def _imported_or_quoted_name_of(node: ast.AST) -> str:
    """The name an import binds, or a string that is one (a json key)."""
    if isinstance(node, ast.alias):
        return node.asname or node.name
    if not isinstance(node, ast.Constant):
        return ""
    if isinstance(node.value, str) and IDENTIFIER.fullmatch(node.value):
        return node.value
    return ""


def _calls_a_task_a_point(name: str) -> bool:
    if name in OTHER_POINT_NAMES:
        return False
    if any(phrase in name for phrase in OTHER_POINT_PHRASES):
        return False
    split_camel_case = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", name)
    lowered = split_camel_case.lower()
    words = lowered.split("_")
    return "point" in words or "points" in words
