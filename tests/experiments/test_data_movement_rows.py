"""The run folder's data-movement rows: one per shot per path.

The contract is sinter's additive shape (sinter/_data/_task_stats.py:
only fields that add are recorded and every summary is derived from
them): one row per shot per path off that shot's own counters, with the
memory class the path crosses, so a folder folded from shards carries
the rows one unsharded run would have written.
"""

import decsim.experiments.collect_command as collect_command
import decsim.experiments.command as command
import decsim.experiments.experiment as experiment
import decsim.experiments.report as sweep_report
import tests.experiments.yaml_configs as yaml_configs

COUNTING_SWEEP = {
    "observation": {"data_movement": True},
    "sweep": [
        {
            "axes": {
                "workload.arguments.physical_error_probability": [0.001],
                "qpu.distance": [3],
                "qpu.round_period_microseconds": [1.0],
            },
            "collection": {"max_shots": 2},
        }
    ],
}


def measured_shots(config_path, count):
    """That config's first point, one measured shot per seed."""
    config = experiment.load_experiment(config_path)
    measurements = []
    for seed in range(count):
        measurement = yaml_configs.measure_point_shot(
            config,
            physical_error_probability=0.001,
            distance=3,
            round_period_microseconds=1.0,
            seed=seed,
        )
        measurements.append(measurement)
    return measurements


# the counters a shot splits over its paths, one share a row
PATH_COLUMNS = (
    "copies",
    "copied_rounds",
    "copy_bits",
    "moves",
    "moved_rounds",
    "move_bits",
)


def _column_sums(rows: list, columns: tuple) -> dict:
    sums = {column: 0 for column in columns}
    for row in rows:
        for column in columns:
            sums[column] += row[column]
    return sums


def test_a_shots_rows_add_up_to_that_shots_own_counters(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, COUNTING_SWEEP)
    measurements = measured_shots(config_path, 1)
    rows = sweep_report.shot_data_movement_rows(measurements)
    counted = measurements[0].data_movement
    summed = _column_sums(rows, PATH_COLUMNS)
    references = {row["references"] for row in rows}
    referenced_rounds = {row["referenced_rounds"] for row in rows}

    assert summed == {column: counted[column] for column in PATH_COLUMNS}
    assert references == {counted["references"]}
    assert referenced_rounds == {counted["referenced_rounds"]}


def test_each_rows_memory_class_is_the_one_the_counters_placed(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, COUNTING_SWEEP)
    measurements = measured_shots(config_path, 1)
    rows = sweep_report.shot_data_movement_rows(measurements)
    class_by_path = {row["path"]: row["memory_class"] for row in rows}

    assert class_by_path["qpu_to_controller"] == "off_board"
    assert class_by_path["controller_to_weak_buffer"] == "on_board"
    assert class_by_path["readout -> controller intake"] == "on_chip"
    path = "controller assembler -> weak syndrome buffer"
    assert class_by_path[path] == "on_board"


def test_a_run_that_counted_no_movement_writes_no_rows(tmp_path, monkeypatch):
    """observation.data_movement off means no counts, which is not zero."""
    monkeypatch.chdir(tmp_path)
    silent_path = yaml_configs.write_config(
        tmp_path,
        {
            "sweep": [
                {
                    "axes": {
                        "workload.arguments.physical_error_probability": [
                            0.001
                        ],
                        "qpu.distance": [3],
                        "qpu.round_period_microseconds": [1.0],
                    },
                    "collection": {"max_shots": 1},
                }
            ]
        },
    )
    run_dir, _rows = collect_command.run_experiment(silent_path)
    movement_path = run_dir / "shot_data_movement.csv"

    assert not movement_path.exists()


def test_a_counting_run_writes_the_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = yaml_configs.write_config(tmp_path, COUNTING_SWEEP)
    run_dir, _rows = collect_command.run_experiment(config_path)
    movement_path = run_dir / "shot_data_movement.csv"
    shot_rows = sweep_report.read_rows(movement_path)
    copied = {row["memory_class"] for row in shot_rows if row["copy_bits"]}
    moved = {row["memory_class"] for row in shot_rows if row["move_bits"]}

    assert "on_chip" in copied
    assert "off_board" in moved


def test_pieces_of_one_shot_fold_to_the_whole_runs_movement_rows(tmp_path):
    """The fold's precondition: the file adds, so pieces equal one run."""
    whole_path = yaml_configs.write_config(tmp_path, COUNTING_SWEEP)
    cut_folder = tmp_path / "cut_config"
    cut_folder.mkdir()
    cut_card = {**COUNTING_SWEEP, "collection": {"piece_rounds": 1}}
    cut_path = yaml_configs.write_config(cut_folder, cut_card)
    whole_dir = tmp_path / "whole"
    cut_dir = tmp_path / "cut"
    command.main(["run", str(whole_path), "--out", str(whole_dir)])
    command.main(["run", str(cut_path), "--out", str(cut_dir)])
    whole_run_dir = whole_dir
    cut_run_dir = cut_dir
    whole_movement_path = whole_run_dir / "shot_data_movement.csv"
    folded_movement_path = cut_run_dir / "shot_data_movement.csv"
    whole_rows = sweep_report.read_rows(whole_movement_path)
    folded_rows = sweep_report.read_rows(folded_movement_path)
    cut_pieces = cut_dir.glob("pieces/*/*")

    assert len(list(cut_pieces)) == 2
    assert folded_rows == whole_rows
