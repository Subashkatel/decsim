"""The Python experiment a run file defines (experiment.py).

The shape is gem5 MultiSim's, a script adding named simulators
(src/python/gem5/utils/multisim/multisim.py:285-353), and sinter's point,
settings and json metadata whose hash is its id
(sinter/_data/_task.py:167-204).
"""

import dataclasses
import os
import random

import pytest

import decsim.confidence.complementary as complementary
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.collection as collection
import decsim.experiments.experiment as experiment
import decsim.experiments.refusal as refusal
import decsim.frontends.settings as workload_settings
import decsim.producers as producers
import tests.experiments.run_files as run_files

CAPPED = collection.CollectionSettings(max_shots=4)
# a switching point whose threshold learns online from 15 dB
ONLINE = {"machine": "switching", "machine_arguments": {"online": {}}}


def _minimal_settings():
    task = run_files.first_task()
    return task.settings


def _online_point_at_distance_5(workload) -> experiment.Point:
    """A Python switching point on the online threshold, running workload."""
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
    return experiment.Point("online", machine)


def test_a_grid_runs_the_last_axis_fastest_in_the_order_given():
    points = experiment.grid(
        distance=(3, 5), physical_error_probability=(0.1, 0.2)
    )

    assert points == [
        {"distance": 3, "physical_error_probability": 0.1},
        {"distance": 3, "physical_error_probability": 0.2},
        {"distance": 5, "physical_error_probability": 0.1},
        {"distance": 5, "physical_error_probability": 0.2},
    ]


def test_two_points_of_one_name_are_refused():
    settings = _minimal_settings()
    first = experiment.Point("d3", settings, {"distance": 3})
    second = experiment.Point("d3", settings, {"distance": 5})

    with pytest.raises(ValueError) as refused:
        experiment.Experiment("study", [first, second], CAPPED)

    message = str(refused.value)
    assert message.startswith("two points are named d3")


def test_a_point_name_that_is_no_folder_name_is_refused():
    settings = _minimal_settings()

    with pytest.raises(ValueError) as refused:
        experiment.Point("d3/p1", settings)

    assert "is not a folder name" in str(refused.value)


def test_an_experiment_with_no_points_is_refused():
    with pytest.raises(ValueError) as refused:
        experiment.Experiment("study", [], CAPPED)

    assert str(refused.value) == "the experiment study has no points"


def test_a_point_no_collection_stops_is_refused():
    settings = _minimal_settings()
    point = experiment.Point("d3", settings)

    with pytest.raises(ValueError) as refused:
        experiment.Experiment("study", [point])

    message = str(refused.value)
    assert message.startswith("the point d3 has no collection")


def test_a_points_own_collection_wins_over_the_experiments():
    settings = _minimal_settings()
    own = collection.CollectionSettings(max_shots=9)
    first = experiment.Point("own", settings, {"distance": 3}, own)
    second = experiment.Point("shared", settings, {"distance": 5})
    study = experiment.Experiment("study", [first, second], CAPPED)

    assert study.collection_of(first) == own
    assert study.collection_of(second) == CAPPED


def test_an_unknown_point_name_is_refused_with_the_names():
    settings = _minimal_settings()
    point = experiment.Point("d3", settings)
    study = experiment.Experiment("study", [point], CAPPED)

    with pytest.raises(refusal.RefusalError) as refused:
        study.point_named("d5")

    assert str(refused.value) == (
        "the experiment study has no point d5; its points are d3"
    )


def test_a_python_points_task_builds_its_online_calibrator():
    """The calibrator is point state the task builds, seeded as today."""
    online_task = run_files.first_task(**ONLINE)
    point = experiment.Point("online", online_task.settings)

    task = experiment.task_of(point)

    calibrator = task.online_threshold

    expected = random.Random("online-threshold d=3 p=0.001")
    generator = calibrator.random_generator
    assert generator.getstate() == expected.getstate()


def test_a_python_point_states_its_distance_and_error_probability_once():
    """The threshold record names no fact; the seed reads the point's.

    The distance is the qpu's and the error probability the one the
    workload was made at, so the online card restates neither.
    """
    workload = producers.memory_circuit(
        "surface_code:rotated_memory_x", 6, 5, 0.002
    )
    point = _online_point_at_distance_5(workload)

    task = experiment.task_of(point)

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
    point = _online_point_at_distance_5(workload)

    with pytest.raises(ValueError, match="physical_error_probability=None"):
        experiment.task_of(point)


def test_a_run_file_is_loaded_by_its_experiment(tmp_path):
    run_file = tmp_path / "study.py"
    run_file.write_text(
        "import decsim.experiments.collection as collection\n"
        "import decsim.experiments.experiment as experiment\n"
        "import tests.experiments.run_files as run_files\n"
        "_settings = run_files.minimal_machine()\n"
        "experiment = experiment.Experiment(\n"
        "    'study',\n"
        "    [experiment.Point('d3', _settings, {'d': 3})],\n"
        "    collection.CollectionSettings(max_shots=2),\n"
        ")\n"
    )

    study = experiment.load(run_file)

    assert study.name == "study"
    assert [point.name for point in study.points] == ["d3"]


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

    first_point = study.points[0]
    collection = study.collection_of(first_point)
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


def test_an_online_point_with_a_target_is_refused_when_built():
    """The stop it cannot keep is refused before anything is planned."""
    study = run_files.sweep(**ONLINE)
    (point,) = study.points
    target = collection.CollectionSettings(max_shots=10, max_failures=5)
    targeted = dataclasses.replace(point, collection=target)

    with pytest.raises(refusal.RefusalError, match="max_shots alone"):
        experiment.Experiment("online", [targeted])
