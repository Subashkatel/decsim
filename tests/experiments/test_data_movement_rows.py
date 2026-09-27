"""The run folder's data-movement rows: per shot per path, then per point.

The contract is sinter's additive shape (sinter/_data/_task_stats.py:
only fields that add are recorded and every summary is derived from
them): one row per shot per path off that shot's own counters, and a
per-point file derived from those rows, so a folder folded from shards
carries the rows one unsharded run would have written. The second table
groups by memory class because the classical sources make the class the
cost: a DRAM access is "a couple of orders-of-magnitude higher than the
cost of an internal cache access" (Horowitz, ISSCC 2014 lines 232-247)
and an accelerator's access costs what the memory it reads costs (Dally,
CACM 2020 lines 231-234), so the classes are not summed into one count.
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
            "shots": 2,
        }
    ],
}
# a threshold most windows clear and some do not, so the strong tier's
# paths are in part of the point's shots and not the rest
ESCALATING_SWEEP = {
    "observation": {"data_movement": True},
    "escalation": {
        "kind": "switching",
        "gap_threshold_db": 20.0,
        "strong_window": "near_seam_pinned",
        "run_both_at_once": False,
    },
    "strong_decoder": {
        "kind": 1.0,
        "units": 1,
        "unit_memory": {"bits": None},
        "engine": {
            "clock": "fridge",
            "fetch_cycles_per_round": 1,
            "fetch_cycles_per_job": 0,
            "release_cycles_per_job": 1,
            "release_cycles_per_round": 0,
        },
    },
    "sweep": [
        {
            "axes": {
                "workload.arguments.physical_error_probability": [0.005],
                "qpu.distance": [3],
                "qpu.round_period_microseconds": [1.0],
            },
            "shots": 8,
        }
    ],
}


def measured_shots(config_path, count, probability=0.001):
    """That config's first point, one measured shot per seed."""
    config = experiment.load_experiment(config_path)
    measurements = []
    for seed in range(count):
        measurement = yaml_configs.measure_point_shot(
            config,
            physical_error_probability=probability,
            distance=3,
            round_period_microseconds=1.0,
            seed=seed,
        )
        measurements.append(measurement)
    return measurements


def lines_starting_with(lines, opening):
    """Every terminal line that opens with that label."""
    found = []
    for line in lines:
        if line.startswith(opening):
            found.append(line)
    return found


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


def _class_copy_bits(shot_rows: list, memory_class: str) -> float:
    """The copy bits of every path row in one memory class."""
    copy_bits = 0.0
    for row in shot_rows:
        if row["memory_class"] == memory_class:
            copy_bits += row["copy_bits"]
    return copy_bits


def _class_rows(point_rows: list) -> list:
    """The point rows that stand for a memory class, in listed order."""
    rows = []
    for row in point_rows:
        if row["grouping"] == "memory_class":
            rows.append(row)
    return rows


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


def test_a_memory_class_row_is_that_classs_paths_over_the_shots(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, COUNTING_SWEEP)
    measurements = measured_shots(config_path, 2)
    shot_rows = sweep_report.shot_data_movement_rows(measurements)
    point_rows = sweep_report.data_movement_rows(shot_rows)
    on_board_paths = _class_copy_bits(shot_rows, "on_board")
    class_rows = _class_rows(point_rows)
    per_shot_by_class = {
        row["name"]: row["copy_bits_per_shot"] for row in class_rows
    }
    on_board_class = per_shot_by_class["on_board"]

    assert on_board_class == on_board_paths / 2


def test_the_class_rows_are_listed_cheapest_first(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, COUNTING_SWEEP)
    measurements = measured_shots(config_path, 1)
    shot_rows = sweep_report.shot_data_movement_rows(measurements)
    point_rows = sweep_report.data_movement_rows(shot_rows)
    class_rows = _class_rows(point_rows)
    listed = [row["name"] for row in class_rows]

    assert listed == ["on_chip", "on_board", "off_board"]


def test_the_class_rows_hold_every_points_copied_and_moved_bits(tmp_path):
    """What a data movement figure is drawn from, at each swept distance.

    decsim draws no such figure; data_movement.csv holds one row per
    point per memory class, so the reader draws it from the file. An
    off-board hop of this machine moves and never copies.
    """
    sweep = {
        "observation": {"data_movement": True},
        "sweep": [
            {
                "axes": {
                    "workload.arguments.physical_error_probability": [0.001],
                    "qpu.distance": [3, 5],
                },
                "shots": 1,
            }
        ],
    }
    config_path = yaml_configs.write_config(tmp_path, sweep)
    out_dir = tmp_path / "run"
    run_dir, _rows = collect_command.run_experiment(config_path, out_dir)
    movement_path = run_dir / "data_movement.csv"

    rows = sweep_report.read_rows(movement_path)

    class_rows = [row for row in rows if row["grouping"] == "memory_class"]
    points = {row["point_id"] for row in class_rows}
    on_chip = [row for row in class_rows if row["name"] == "on_chip"]
    off_board = [row for row in class_rows if row["name"] == "off_board"]
    on_chip_copied = [row["copy_bits_per_shot"] > 0 for row in on_chip]
    off_board_copied = [row["copy_bits_per_shot"] for row in off_board]
    off_board_moved = [row["move_bits_per_shot"] > 0 for row in off_board]
    assert len(points) == 2
    assert on_chip_copied == [True, True]
    assert off_board_copied == [0, 0]
    assert off_board_moved == [True, True]


def test_a_references_column_counts_one_shots_holds_once(tmp_path):
    """A reference is a token on a store's slot, not a hop of a path.

    It repeats on every row of one shot, so the point row divides by the
    shots and not by the rows.
    """
    config_path = yaml_configs.write_config(tmp_path, COUNTING_SWEEP)
    measurements = measured_shots(config_path, 2)
    shot_rows = sweep_report.shot_data_movement_rows(measurements)
    point_rows = sweep_report.data_movement_rows(shot_rows)
    first = measurements[0].data_movement["references"]
    second = measurements[1].data_movement["references"]
    mean_references = (first + second) / 2

    per_shot = {row["references_per_shot"] for row in point_rows}
    assert per_shot == {mean_references}


def test_a_path_only_some_shots_took_still_carries_the_points_holds(
    tmp_path,
):
    """A reference belongs to the shot, so it is not cut by a rare path.

    A window reaches the strong tier only when its gap fell under the
    threshold, so the strong store's hop is in some of a point's shots
    and not others; counting the point's holds over that path's rows
    alone would divide a whole-shot counter by the wrong number of
    shots.
    """
    config_path = yaml_configs.write_config(tmp_path, ESCALATING_SWEEP)
    measurements = measured_shots(config_path, 8, probability=0.005)
    shot_rows = sweep_report.shot_data_movement_rows(measurements)
    point_rows = sweep_report.data_movement_rows(shot_rows)
    escalating_shots = _shots_with_a_path(
        shot_rows, "strong_buffer_to_strong_decoder"
    )
    references = [row["references_per_shot"] for row in point_rows]

    assert 0 < escalating_shots < len(measurements)
    assert len(set(references)) == 1


def _shots_with_a_path(shot_rows: list, ending: str) -> int:
    """How many of a point's shots have a row on a path with that ending."""
    seeds = set()
    for row in shot_rows:
        if row["path"].endswith(ending):
            seeds.add(row["seed"])
    return len(seeds)


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
                    "shots": 1,
                }
            ]
        },
    )
    run_dir, rows = collect_command.run_experiment(silent_path)
    lines = sweep_report.terminal_lines(rows, run_dir)
    movement_path = run_dir / "shot_data_movement.csv"
    point_path = run_dir / "data_movement.csv"

    assert not movement_path.exists()
    assert not point_path.exists()
    assert lines[-1].startswith("data movement: observation.data_movement")


def test_a_counting_run_writes_both_files_and_the_terminal_lines(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = yaml_configs.write_config(tmp_path, COUNTING_SWEEP)
    run_dir, rows = collect_command.run_experiment(config_path)
    lines = sweep_report.terminal_lines(rows, run_dir)
    copied = lines_starting_with(lines, "bits copied per shot")
    moved = lines_starting_with(lines, "bits moved per shot")
    movement_path = run_dir / "shot_data_movement.csv"
    point_path = run_dir / "data_movement.csv"

    assert movement_path.exists()
    assert point_path.exists()
    assert len(copied) == 1
    assert len(moved) == 1
    assert "on_chip" in copied[0]
    assert "off_board" in moved[0]


def test_pieces_of_one_shot_fold_to_the_whole_runs_movement_rows(tmp_path):
    """The fold's precondition: the file adds, so pieces equal one run."""
    whole_path = yaml_configs.write_config(tmp_path, COUNTING_SWEEP)
    cut_folder = tmp_path / "cut_config"
    cut_folder.mkdir()
    cut_card = {**COUNTING_SWEEP, "collection": {"piece_rounds": 1}}
    cut_path = yaml_configs.write_config(cut_folder, cut_card)
    whole_dir = tmp_path / "whole"
    cut_dir = tmp_path / "cut"
    command.main(["collect", str(whole_path), "--out", str(whole_dir)])
    command.main(["collect", str(cut_path), "--out", str(cut_dir)])
    whole_run_dir = yaml_configs.run_folder_of(whole_dir)
    cut_run_dir = yaml_configs.run_folder_of(cut_dir)
    whole_movement_path = whole_run_dir / "data_movement.csv"
    folded_movement_path = cut_run_dir / "data_movement.csv"
    whole_rows = sweep_report.read_rows(whole_movement_path)
    folded_rows = sweep_report.read_rows(folded_movement_path)
    cut_pieces = cut_dir.glob("pieces/*/*")

    assert len(list(cut_pieces)) == 2
    assert folded_rows == whole_rows
