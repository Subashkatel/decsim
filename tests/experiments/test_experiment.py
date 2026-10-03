"""The Python experiment and the yaml translated into one (experiment.py).

The shape is gem5 MultiSim's, a script adding named simulators
(src/python/gem5/utils/multisim/multisim.py:285-353), and sinter's point,
settings and json metadata whose hash is its id
(sinter/_data/_task.py:167-204). A yaml point keeps the id its task has
today, which names its pieces.
"""

import dataclasses
import random

import pytest

import decsim.collect as collect
import decsim.confidence.complementary as complementary
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.collection as collection
import decsim.experiments.experiment as experiment
import decsim.experiments.refusal as refusal
import decsim.frontends.settings as workload_settings
import decsim.producers as producers
import tests.experiments.yaml_configs as yaml_configs

CAPPED = collection.CollectionSettings(max_shots=4)


def _minimal_settings(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, {})
    config = experiment.load_experiment(config_path)
    task = config.first_point_task()
    return task.settings


def test_a_grid_runs_the_last_axis_fastest_in_the_order_given():
    points = experiment.grid(d=(3, 5), p=(0.1, 0.2))

    assert points == [
        {"d": 3, "p": 0.1},
        {"d": 3, "p": 0.2},
        {"d": 5, "p": 0.1},
        {"d": 5, "p": 0.2},
    ]


def test_two_points_of_one_name_are_refused(tmp_path):
    settings = _minimal_settings(tmp_path)
    first = experiment.Point("d3", settings, {"d": 3})
    second = experiment.Point("d3", settings, {"d": 5})

    with pytest.raises(ValueError) as refused:
        experiment.Experiment("study", [first, second], CAPPED)

    message = str(refused.value)
    assert message.startswith("two points are named d3")


def test_a_point_name_that_is_no_folder_name_is_refused(tmp_path):
    settings = _minimal_settings(tmp_path)

    with pytest.raises(ValueError) as refused:
        experiment.Point("d3/p1", settings)

    assert "is not a folder name" in str(refused.value)


def test_an_experiment_with_no_points_is_refused():
    with pytest.raises(ValueError) as refused:
        experiment.Experiment("study", [], CAPPED)

    assert str(refused.value) == "the experiment study has no points"


def test_a_point_no_collection_stops_is_refused(tmp_path):
    settings = _minimal_settings(tmp_path)
    point = experiment.Point("d3", settings)

    with pytest.raises(ValueError) as refused:
        experiment.Experiment("study", [point])

    message = str(refused.value)
    assert message.startswith("the point d3 has no collection")


def test_a_points_own_collection_wins_over_the_experiments(tmp_path):
    settings = _minimal_settings(tmp_path)
    own = collection.CollectionSettings(max_shots=9)
    first = experiment.Point("own", settings, {"d": 3}, own)
    second = experiment.Point("shared", settings, {"d": 5})
    study = experiment.Experiment("study", [first, second], CAPPED)

    assert study.collection_of(first) == own
    assert study.collection_of(second) == CAPPED


def test_an_unknown_point_name_is_refused_with_the_names(tmp_path):
    settings = _minimal_settings(tmp_path)
    point = experiment.Point("d3", settings)
    study = experiment.Experiment("study", [point], CAPPED)

    with pytest.raises(refusal.RefusalError) as refused:
        study.point_named("d5")

    assert str(refused.value) == (
        "the experiment study has no point d5; its points are d3"
    )


def test_a_yaml_point_keeps_the_id_its_task_has(tmp_path):
    overrides = yaml_configs.online_threshold()
    config_path = yaml_configs.write_config(tmp_path, overrides)
    config = experiment.load_experiment(config_path)
    (task,) = config.tasks()

    study = config.experiment()

    (point,) = study.points
    point_task = experiment.task_of(point)
    point_id = task.strong_id()
    assert point_task.strong_id() == point_id
    assert point.name == point_id[:12]


def test_a_python_points_task_builds_its_online_calibrator(tmp_path):
    """The calibrator is point state the task builds, seeded as today."""
    overrides = yaml_configs.online_threshold()
    config_path = yaml_configs.write_config(tmp_path, overrides)
    config = experiment.load_experiment(config_path)
    yaml_task = config.first_point_task()
    point = experiment.Point("online", yaml_task.settings)

    task = experiment.task_of(point)

    calibrator = task.online_threshold

    expected = random.Random("online-threshold d=3 p=0.001")
    generator = calibrator.random_generator
    assert generator.getstate() == expected.getstate()


def test_a_python_point_states_its_distance_and_error_probability_once(
    tmp_path,
):
    """The threshold record names no fact; the seed reads the point's.

    The distance is the qpu's and the error probability the one the
    workload was made at, so the online card restates neither.
    """
    machine = _minimal_settings(tmp_path)
    confidence = complementary.ComplementaryGap.Settings()
    card = threshold_sources.OnlineThreshold.Settings(threshold_decibels=15.0)
    switching = escalation_settings.SwitchingSettings(confidence, card)
    workload = producers.memory_circuit(
        "surface_code:rotated_memory_x", 6, 5, 0.002
    )
    qpu = dataclasses.replace(machine.qpu, distance=5)
    running = workload_settings.WorkloadSettings.running(workload)
    machine = dataclasses.replace(
        machine,
        qpu=qpu,
        workload=running,
        strong_decoder=machine.weak_decoder,
        switching=switching,
    )
    point = experiment.Point("online", machine)

    task = experiment.task_of(point)

    calibrator = task.online_threshold

    expected = random.Random("online-threshold d=5 p=0.002")
    generator = calibrator.random_generator
    assert generator.getstate() == expected.getstate()


def test_a_yaml_point_two_blocks_name_is_one_point(tmp_path):
    block = yaml_configs.MINIMAL_CONFIG["sweep"][0]
    config_path = yaml_configs.write_config(tmp_path, {"sweep": [block, block]})
    config = experiment.load_experiment(config_path)

    study = config.experiment()

    assert len(study.points) == 1


def test_a_run_file_is_loaded_by_its_experiment(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, {})
    run_file = tmp_path / "study.py"
    run_file.write_text(
        "import decsim.experiments.collection as collection\n"
        "import decsim.experiments.experiment as experiment\n"
        f"_config = experiment.load_experiment({str(config_path)!r})\n"
        "_settings = _config.first_point_task().settings\n"
        "experiment = experiment.Experiment(\n"
        "    'study',\n"
        "    [experiment.Point('d3', _settings, {'d': 3})],\n"
        "    collection.CollectionSettings(max_shots=2),\n"
        ")\n"
    )

    study = experiment.load(run_file)

    assert study.name == "study"
    assert [point.name for point in study.points] == ["d3"]


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


def test_one_point_reads_no_point_past_the_one_it_chooses(
    tmp_path, monkeypatch
):
    """A point's read runs its maker, so a narrated shot reads no other.

    Without a name only the first point is read; with one, the points up
    to the named one, since a yaml point is named by its id.
    """
    axes = {
        "qpu.distance": [3, 5, 7],
        "workload.arguments.physical_error_probability": [0.001],
        "qpu.round_period_microseconds": [1.0],
    }
    card = {"sweep": [{"axes": axes, "collection": {"max_shots": 2}}]}
    config_path = yaml_configs.write_config(tmp_path, card)
    config = experiment.load_experiment(config_path)
    study = config.experiment()
    second_name = study.points[1].name
    read_distances = []
    point_task = experiment.ExperimentConfig.point_task

    def counted_point_task(self, values):
        distance = values["qpu.distance"]
        read_distances.append(distance)
        return point_task(self, values)

    monkeypatch.setattr(
        experiment.ExperimentConfig, "point_task", counted_point_task
    )

    config.one_point()
    first_reads = list(read_distances)
    read_distances.clear()
    config.one_point(second_name)

    assert first_reads == [3]
    assert read_distances == [3, 5]


def test_an_online_point_with_a_target_is_refused_when_built(tmp_path):
    """The stop it cannot keep is refused before anything is planned."""
    overrides = yaml_configs.online_threshold()
    config_path = yaml_configs.write_config(tmp_path, overrides)
    config = experiment.load_experiment(config_path)
    study = config.experiment()
    (point,) = study.points
    target = collection.CollectionSettings(max_shots=10, max_failures=5)
    targeted = dataclasses.replace(point, collection=target)

    with pytest.raises(refusal.RefusalError, match="max_shots alone"):
        experiment.Experiment("online", [targeted])


def _record_options_of(tmp_path, overrides: dict):
    """The record options and settings the yaml's first point reads to."""
    config_path = yaml_configs.write_config(tmp_path, overrides)
    config = experiment.load_experiment(config_path)
    task = config.first_point_task()
    return task.record_options, task.settings


def test_the_record_options_are_read_beside_what_they_record(tmp_path):
    """They are written under observation and burst_detector, run-owned.

    The machine never reads them, so they are on the task beside the
    settings and no part of the point's id.
    """
    overrides = yaml_configs.fixed_threshold_switching()
    overrides["observation"] = {"confidence_shot_count": "all"}
    overrides["burst_detector"] = {
        "kind": "masked_regional_cusum",
        "catch_deadline_rounds": 0,
    }
    written, settings = _record_options_of(tmp_path, overrides)
    unset, _settings = _record_options_of(tmp_path, {})

    assert written == collect.RecordOptions(None, 0)
    assert unset == collect.RecordOptions(100, 300)
    a_billion = 10**9
    assert written.samples_confidence_of(a_billion)
    assert unset.samples_confidence_of(99)
    assert not unset.samples_confidence_of(100)
    assert not hasattr(settings.observation, "confidence_shot_count")
    assert not hasattr(
        settings.switching.burst_detector, "catch_deadline_rounds"
    )


@pytest.mark.parametrize("written", [-1, True, "some", 2.5])
def test_a_confidence_shot_count_that_is_no_count_is_refused(tmp_path, written):
    """A count of shots from seed 0, or the word all; nothing else."""
    overrides = {"observation": {"confidence_shot_count": written}}

    with pytest.raises(ValueError, match="confidence_shot_count must be"):
        _record_options_of(tmp_path, overrides)


def test_a_catch_deadline_is_a_whole_number_of_rounds(tmp_path):
    detector = {"kind": "masked_regional_cusum", "catch_deadline_rounds": -1}

    with pytest.raises(ValueError, match="catch_deadline_rounds must be"):
        _record_options_of(tmp_path, {"burst_detector": detector})


def test_no_detector_takes_no_catch_deadline(tmp_path):
    """With no detector there is no flag to time, so the key is refused."""
    detector = {"kind": "none", "catch_deadline_rounds": 300}

    with pytest.raises(ValueError, match="burst_detector does not know"):
        _record_options_of(tmp_path, {"burst_detector": detector})
