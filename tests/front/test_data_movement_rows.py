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

import decsim.front.command as command
import decsim.front.report as sweep_report
from decsim.front.collect_command import run_experiment
from decsim.front.experiment import load_experiment
from tests.front.yaml_configs import measure_point_shot, write_config

COUNTING_SWEEP = {
    "observation": {"data_movement": True},
    "sweep": [
        {
            "physical_error_probability": [0.001],
            "distance": [3],
            "round_period_us": [1.0],
            "shots": 2,
        }
    ],
}
# noisy enough that some shots fold a boundary into a copy and some do
# not, so a path is in part of the point's shots and not the rest
FOLDING_SWEEP = {
    "observation": {"data_movement": True},
    "sweep": [
        {
            "physical_error_probability": [0.01],
            "distance": [3],
            "round_period_us": [1.0],
            "shots": 8,
        }
    ],
}


def measured_shots(config_path, count, probability=0.001):
    """That config's first point, one measured shot per seed."""
    config = load_experiment(config_path)
    measurements = []
    for seed in range(count):
        measurement = measure_point_shot(
            config,
            physical_error_probability=probability,
            distance=3,
            round_period_us=1.0,
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


def test_a_shots_rows_add_up_to_that_shots_own_counters(tmp_path):
    config_path = write_config(tmp_path, COUNTING_SWEEP)
    measurements = measured_shots(config_path, 1)
    rows = sweep_report.shot_data_movement_rows(measurements)
    counted = measurements[0].data_movement
    copies = 0
    copied_rounds = 0
    copy_bits = 0
    moves = 0
    moved_rounds = 0
    move_bits = 0
    for row in rows:
        copies += row["copies"]
        copied_rounds += row["copied_rounds"]
        copy_bits += row["copy_bits"]
        moves += row["moves"]
        moved_rounds += row["moved_rounds"]
        move_bits += row["move_bits"]

    assert copies == counted["copies"]
    assert copied_rounds == counted["copied_rounds"]
    assert copy_bits == counted["copy_bits"]
    assert moves == counted["moves"]
    assert moved_rounds == counted["moved_rounds"]
    assert move_bits == counted["move_bits"]
    for row in rows:
        assert row["references"] == counted["references"]
        assert row["referenced_rounds"] == counted["referenced_rounds"]


def test_each_rows_memory_class_is_the_one_the_counters_placed(tmp_path):
    config_path = write_config(tmp_path, COUNTING_SWEEP)
    measurements = measured_shots(config_path, 1)
    rows = sweep_report.shot_data_movement_rows(measurements)
    class_by_path = {}
    for row in rows:
        class_by_path[row["path"]] = row["memory_class"]

    assert class_by_path["qpu_to_controller"] == "off_board"
    assert class_by_path["controller_to_weak_buffer"] == "on_board"
    assert class_by_path["readout -> controller intake"] == "on_chip"
    assert class_by_path["controller assembler -> Buffer 0"] == "on_board"


def test_a_memory_class_row_is_that_classs_paths_over_the_shots(tmp_path):
    config_path = write_config(tmp_path, COUNTING_SWEEP)
    measurements = measured_shots(config_path, 2)
    shot_rows = sweep_report.shot_data_movement_rows(measurements)
    point_rows = sweep_report.data_movement_rows(shot_rows)
    on_board_paths = 0.0
    for row in shot_rows:
        if row["memory_class"] == "on_board":
            on_board_paths += row["copy_bits"]
    on_board_class = 0.0
    for row in point_rows:
        if row["grouping"] != "memory_class":
            continue
        if row["name"] == "on_board":
            on_board_class = row["copy_bits_per_shot"]

    assert on_board_class == on_board_paths / 2


def test_the_class_rows_are_listed_cheapest_first(tmp_path):
    config_path = write_config(tmp_path, COUNTING_SWEEP)
    measurements = measured_shots(config_path, 1)
    shot_rows = sweep_report.shot_data_movement_rows(measurements)
    point_rows = sweep_report.data_movement_rows(shot_rows)
    listed = []
    for row in point_rows:
        if row["grouping"] == "memory_class":
            listed.append(row["name"])

    assert listed == ["on_chip", "on_board", "off_board"]


def test_a_references_column_counts_one_shots_holds_once(tmp_path):
    """A reference is a token on a store's slot, not a hop of a path.

    It repeats on every row of one shot, so the point row divides by the
    shots and not by the rows.
    """
    config_path = write_config(tmp_path, COUNTING_SWEEP)
    measurements = measured_shots(config_path, 2)
    shot_rows = sweep_report.shot_data_movement_rows(measurements)
    point_rows = sweep_report.data_movement_rows(shot_rows)
    first = measurements[0].data_movement["references"]
    second = measurements[1].data_movement["references"]
    mean_references = (first + second) / 2

    for row in point_rows:
        assert row["references_per_shot"] == mean_references


def test_a_path_only_some_shots_took_still_carries_the_points_holds(
    tmp_path,
):
    """A reference belongs to the shot, so it is not cut by a rare path.

    The window's boundary is folded into a copy only when the neighbour's
    seam carried a defect, so its path is in some of a point's shots and
    not others; counting the point's holds over that path's rows alone
    would divide a whole-shot counter by the wrong number of shots.
    """
    config_path = write_config(tmp_path, FOLDING_SWEEP)
    measurements = measured_shots(config_path, 8, probability=0.01)
    shot_rows = sweep_report.shot_data_movement_rows(measurements)
    point_rows = sweep_report.data_movement_rows(shot_rows)
    folding_shots = _shots_with_a_path(shot_rows, "masked view")
    references = []
    for row in point_rows:
        references.append(row["references_per_shot"])

    assert 0 < folding_shots < len(measurements)
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
    silent_path = write_config(
        tmp_path,
        {
            "sweep": [
                {
                    "physical_error_probability": [0.001],
                    "distance": [3],
                    "round_period_us": [1.0],
                    "shots": 1,
                }
            ]
        },
    )
    run_dir, rows = run_experiment(silent_path)
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
    config_path = write_config(tmp_path, COUNTING_SWEEP)
    run_dir, rows = run_experiment(config_path)
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


def test_combining_two_shards_gives_the_whole_runs_movement_rows(tmp_path):
    """The fold's precondition: the file adds, so shards equal one run."""
    config_path = write_config(tmp_path, COUNTING_SWEEP)
    whole_dir = tmp_path / "whole"
    first_dir = tmp_path / "shard0"
    second_dir = tmp_path / "shard1"
    combined_dir = tmp_path / "combined"
    whole = ["collect", str(config_path), "--out", str(whole_dir)]
    command.main(whole)
    first = [
        "collect",
        str(config_path),
        "--out",
        str(first_dir),
        "--shard",
        "0/2",
        "--shots-per-unit",
        "1",
    ]
    command.main(first)
    second = [
        "collect",
        str(config_path),
        "--out",
        str(second_dir),
        "--shard",
        "1/2",
        "--shots-per-unit",
        "1",
    ]
    command.main(second)
    folding = [
        "combine",
        str(first_dir),
        str(second_dir),
        "--out",
        str(combined_dir),
    ]
    command.main(folding)
    whole_path = whole_dir / "data_movement.csv"
    folded_path = combined_dir / "data_movement.csv"
    whole_rows = sweep_report.read_rows(whole_path)
    folded_rows = sweep_report.read_rows(folded_path)

    assert folded_rows == whole_rows
