"""The Python experiment and the yaml translated into one (experiment.py).

The shape is gem5 MultiSim's, a script adding named simulators
(src/python/gem5/utils/multisim/multisim.py:285-353), and sinter's point,
settings and json metadata whose hash is its id
(sinter/_data/_task.py:167-204). A yaml point keeps the id its task has
today, which names its pieces.
"""

import pytest

import decsim.experiments.collection as collection
import decsim.experiments.experiment as experiment
import decsim.experiments.refusal as refusal
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
    assert point_task.online_threshold is not None
    assert point_task.settings.escalation.online_threshold is None


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
