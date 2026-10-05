"""The rows report.py makes of measured shots, and the files a run writes.

The contract is sinter's additive shape (sinter/_data/_task_stats.py:
only fields that add are recorded and every summary is derived from
them). shot_data_movement.csv holds one row per shot per path off that
shot's own counters, with the memory class the path crosses, so a
folder folded from shards carries the rows one unsharded run would have
written. latency_samples.csv holds one measured algorithm wall clock
per decoded window, recorded only for a decoder named by a table row,
each beside its window's inter-arrival, the deadline a decode must
beat. The confidence files leave out an unscored shot, as sinter takes
a discard out of every failure-conditioned count
(sinter/_decoding/_decoding.py:120-128).
"""

import collections
import csv
import math
import types

import decsim.experiments.collect_command as collect_command
import decsim.experiments.command as command
import decsim.experiments.measure as measure
import decsim.experiments.report as report
import decsim.records.decoding as decoding_records
import tests.experiments.run_files as run_files

# one task, every copy and move counted, two shots
COUNTING = {
    "observation": {"data_movement": True},
    "collection": {"max_shots": 2},
}


def measured_shots(count: int) -> list:
    """The counting task, one measured shot per seed."""
    measurements = []
    for seed in range(count):
        measurement = run_files.measured_shot(seed, **COUNTING)
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


def test_a_shots_rows_add_up_to_that_shots_own_counters():
    measurements = measured_shots(1)
    rows = report.shot_data_movement_rows(measurements)
    counted = measurements[0].data_movement
    summed = _column_sums(rows, PATH_COLUMNS)
    references = {row["references"] for row in rows}
    referenced_rounds = {row["referenced_rounds"] for row in rows}

    assert summed == {column: counted[column] for column in PATH_COLUMNS}
    assert references == {counted["references"]}
    assert referenced_rounds == {counted["referenced_rounds"]}


def test_each_rows_memory_class_is_the_one_the_counters_placed():
    measurements = measured_shots(1)
    rows = report.shot_data_movement_rows(measurements)
    class_by_path = {row["path"]: row["memory_class"] for row in rows}

    assert class_by_path["qpu_to_controller"] == "off_board"
    assert class_by_path["controller_to_weak_buffer"] == "on_board"
    assert class_by_path["readout -> controller intake"] == "on_chip"
    path = "controller assembler -> weak syndrome buffer"
    assert class_by_path[path] == "on_board"


def test_a_run_that_counted_no_movement_writes_no_rows(tmp_path, monkeypatch):
    """observation.data_movement off means no counts, which is not zero."""
    monkeypatch.chdir(tmp_path)
    silent_path = run_files.write_run_file(tmp_path)
    run_dir, _rows = collect_command.run_experiment(silent_path)
    movement_path = run_dir / "shot_data_movement.csv"

    assert not movement_path.exists()


def _memory_classes_with(rows: list, column: str) -> set:
    """The memory classes of the rows whose column holds a count."""
    classes = set()
    for row in rows:
        if row[column]:
            classes.add(row["memory_class"])
    return classes


def test_a_counting_run_writes_the_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_path = run_files.write_run_file(tmp_path, **COUNTING)
    run_dir, _rows = collect_command.run_experiment(run_path)
    movement_path = run_dir / "shot_data_movement.csv"
    shot_rows = report.read_rows(movement_path)
    copied = _memory_classes_with(shot_rows, "copy_bits")
    moved = _memory_classes_with(shot_rows, "move_bits")

    assert "on_chip" in copied
    assert "off_board" in moved


def test_pieces_of_one_shot_fold_to_the_whole_runs_movement_rows(tmp_path):
    """The fold's precondition: the file adds, so pieces equal one run."""
    whole_path = run_files.write_run_file(tmp_path, **COUNTING)
    cut_folder = tmp_path / "cut_config"
    cut_folder.mkdir()
    cut_collection = {"max_shots": 2, "piece_rounds": 1}
    cut_path = run_files.write_run_file(
        cut_folder, **{**COUNTING, "collection": cut_collection}
    )
    whole_dir = tmp_path / "whole"
    cut_dir = tmp_path / "cut"
    command.main(["run", str(whole_path), "--out", str(whole_dir)])
    command.main(["run", str(cut_path), "--out", str(cut_dir)])
    whole_movement_path = whole_dir / "shot_data_movement.csv"
    folded_movement_path = cut_dir / "shot_data_movement.csv"
    whole_rows = report.read_rows(whole_movement_path)
    folded_rows = report.read_rows(folded_movement_path)
    cut_pieces = cut_dir.glob("pieces/*/*")

    assert len(list(cut_pieces)) == 2
    assert folded_rows == whole_rows


# PyMatching's measured wall clock on every window, two shots a task
WALL_CLOCK = {
    "machine_arguments": {"weak_decoder": "pymatching"},
    "collection": {"max_shots": 2},
}


def wall_clock_axes(distances: tuple) -> dict:
    return {
        run_files.DISTANCE_PATH: distances,
        run_files.ROUND_PERIOD_PATH: (1.0,),
        run_files.ERROR_RATE_PATH: (0.001,),
    }


def test_every_decoded_window_contributes_one_latency_sample():
    axes = wall_clock_axes((3,))
    measurement = run_files.measured_shot(axes=axes, **WALL_CLOCK)
    samples = measurement.samples["algorithm"]
    assert len(samples) == measurement.decoded_windows
    assert all(sample > 0 for sample in samples)


def test_a_latency_sample_names_the_tier_that_decoded_its_window():
    """A weak-only run's windows are all the weak tier's."""
    axes = wall_clock_axes((3,))
    measurement = run_files.measured_shot(axes=axes, **WALL_CLOCK)
    rows = report.latency_sample_rows([measurement])
    tiers = [row["tier"] for row in rows]
    assert tiers == ["weak"] * measurement.decoded_windows


def test_a_wall_clock_run_records_every_windows_sample_and_deadline(
    tmp_path, monkeypatch
):
    """A sample per decoded window, beside its task and its deadline.

    Each task holds as many samples as its shots decoded windows. The
    deadline is the window's inter-arrival, commit rounds (d when null)
    times the round period: 3 us at d 3 and 5 us at d 5.
    """
    monkeypatch.chdir(tmp_path)
    axes = wall_clock_axes((3, 5))
    run_path = run_files.write_run_file(tmp_path, axes=axes, **WALL_CLOCK)
    run_dir, rows = collect_command.run_experiment(run_path)
    samples_path = run_dir / "latency_samples.csv"
    with open(samples_path, newline="") as handle:
        reader = csv.DictReader(handle)
        samples = list(reader)
    shots_path = run_dir / "shots.csv"
    with open(shots_path, newline="") as handle:
        reader = csv.DictReader(handle)
        shots = list(reader)

    task_ids = [row["task_id"] for row in samples]
    sample_counts = collections.Counter(task_ids)
    deadlines = {
        (row["task_id"], row["qpu.distance"], row["window_period_us"])
        for row in samples
    }
    task_ids_by_distance = {
        row["qpu.distance"]: row["task_id"] for row in shots
    }
    assert sample_counts == _windows_by_task(shots)
    assert deadlines == {
        (task_ids_by_distance["3"], "3", "3.0"),
        (task_ids_by_distance["5"], "5", "5.0"),
    }


def test_a_latency_card_run_records_no_samples(tmp_path, monkeypatch):
    """The fixed-latency card is flat by construction: no raw samples."""
    monkeypatch.chdir(tmp_path)
    axes = wall_clock_axes((3, 5))
    card_path = run_files.write_run_file(tmp_path, axes=axes)
    card_run_dir, rows = collect_command.run_experiment(card_path)
    assert not (card_run_dir / "latency_samples.csv").exists()


def _windows_by_task(shots: list) -> collections.Counter:
    """The windows every task's shots decoded, off shots.csv."""
    windows = collections.Counter()
    for row in shots:
        windows[row["task_id"]] += int(row["decoded_windows"])
    return windows


def test_a_gaps_bin_is_its_tenth_of_a_decibel_below():
    ten_decibels_in_nats = 2.302585092994046

    assert report.gap_bin_low_decibels(ten_decibels_in_nats) == 10.0
    assert report.gap_bin_low_decibels(0.0) == 0.0
    assert report.gap_bin_low_decibels(math.inf) == math.inf


def test_a_window_with_no_gap_is_counted_in_the_empty_bin():
    """A gapless window escalates as the least confident: its shot's too."""
    ten_decibels_in_nats = 2.302585092994046
    windows = (
        decoding_records.WindowConfidence(
            (1, 0), ten_decibels_in_nats, False, None
        ),
        decoding_records.WindowConfidence((1, 1), None, True, None),
    )
    confidence = measure.ShotConfidence("complementary_gap", windows, True, 100)
    shot = types.SimpleNamespace(
        task_id="p",
        algorithm="pymatching",
        confidence=confidence,
        logical_failure=False,
        is_scored=True,
    )

    rows = report.confidence_histogram_rows([shot])

    assert rows == [
        _one_count("window", None, True),
        _one_count("window", 10.0, False),
        _one_count("shot_minimum", None, None),
    ]


def test_an_unscored_shot_is_in_neither_confidence_file():
    """An unscored shot is sinter's discard, and counts in no P(e|g).

    Its logical_failure reads False, so a row of it would count as a
    success beside every gap. sinter takes a discard out of every
    failure-conditioned count (sinter/_decoding/_decoding.py:120-128),
    and every row of both files carries shot_failed.
    """
    windows = (decoding_records.WindowConfidence((1, 0), 0.1, True, None),)
    confidence = measure.ShotConfidence("complementary_gap", windows, True, 100)
    shot = types.SimpleNamespace(
        task_id="p",
        algorithm="pymatching",
        seed=0,
        confidence=confidence,
        logical_failure=False,
        is_scored=False,
    )

    assert report.window_confidence_rows([shot]) == []
    assert report.confidence_histogram_rows([shot]) == []


def _one_count(histogram, gap_low_decibels, escalated) -> dict:
    """A histogram row of one window or shot of that no-failure shot."""
    return {
        "task_id": "p",
        "algorithm": "pymatching",
        "signal": "complementary_gap",
        "histogram": histogram,
        "gap_low_decibels": gap_low_decibels,
        "escalated": escalated,
        "shot_failed": False,
        "count": 1,
    }
