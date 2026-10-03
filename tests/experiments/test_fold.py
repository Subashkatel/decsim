"""The rules a fold of many pieces keeps (decsim/experiments/fold.py).

Four of them, each pinned here against the thing it claims to equal:
the merged order is the stable sort of the folders' rows, a streamed sum
is math.fsum of the values it was given, what a fold holds does not grow
with the shots the folders hold, and a summary reports the latency
points its folders' rows hold. The third is the reason the module
exists: the 500 shard folders of one weak_ler experiment hold 115
million link rows, which do not fit in memory as lists. The fourth is
why that experiment can be folded at all: its shards hold the sixteen
latency points that tree measured, and this tree measures twenty-three.

A switching run's window_confidence.csv folds as the other per-shot
files do, and its confidence_histogram.csv counts add as sinter's
custom_counts do (sinter/_data/_task_stats.py:51-71), so its pieces give
the files one uncut run writes.
"""

import csv
import json
import math
import os
import pathlib
import random
import statistics
import types

import pytest

import decsim.experiments.command as command
import decsim.experiments.fold as fold
import decsim.experiments.measure as measure
import decsim.experiments.pieces as pieces
import decsim.experiments.refusal as refusal
import decsim.experiments.report as report
import decsim.experiments.run_folder as run_folder
import decsim.records.decoding as decoding_records
import tests.experiments.yaml_configs as yaml_configs

CANCELLING = (1e100, 1.0, -1e100, 1.0)


class _WatchedRow(dict):
    """A row that counts itself alive, so a fold's peak can be read.

    CPython frees it the moment the fold drops it, so the count is the
    rows alive at that moment and not what a garbage collector got
    around to.
    """

    def __init__(self, row, alive):
        dict.__init__(self, row)
        self.alive = alive
        alive[0] += 1

    def __del__(self):
        self.alive[0] -= 1


@pytest.fixture(autouse=True)
def one_tree_reading_per_test():
    """Every test here takes its own reading of the tree, as a task does."""
    run_folder._tree_reading.cache_clear()
    yield
    run_folder._tree_reading.cache_clear()


def _write_rows(path, rows):
    field_names = list(rows[0])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def _row(place, value):
    return {"place": str(place), "value": str(value)}


def _place_of(row):
    return int(row["place"])


# A shot of the one point below runs fifteen QEC rounds: the reference
# workload's rounds_per_shot, which its resolved record reads back.
ROUNDS_PER_SHOT = 15


def _one_point_config(folder, shots, piece_shots):
    card = {
        "collection": {"piece_rounds": piece_shots * ROUNDS_PER_SHOT},
        "sweep": [
            {
                "axes": {
                    "workload.arguments.physical_error_probability": [0.001],
                    "qpu.distance": [3],
                    "qpu.round_period_microseconds": [1.0],
                },
                "collection": {"max_shots": shots},
            }
        ],
    }
    return yaml_configs.write_config(folder, card)


def _pieces_of_one_point(tmp_path, shots, piece_shots):
    """One point's shots collected in pieces; the results folder."""
    folder = tmp_path / f"of_{shots}"
    folder.mkdir()
    config_path = _one_point_config(folder, shots, piece_shots)
    experiment_dir = folder / "experiment"
    command.main(["run", str(config_path), "--out", str(experiment_dir)])
    return experiment_dir


def _piece_folders(experiment_dir) -> list:
    """The experiment's piece folders, the one point's, in seed order."""
    point_dirs = experiment_dir.glob("pieces/*")
    point_ids = [point_dir.name for point_dir in point_dirs]
    return pieces.folders_of(experiment_dir, point_ids)


def _folded(experiment_dir, folders, out_dir) -> list:
    """The pieces folded into out_dir as the collect that saved them does."""
    point_ids = [folders[0].parent.name]
    return report.fold_pieces(experiment_dir, folders, point_ids, out_dir)


def _peak_rows_alive(monkeypatch, experiment_dir, out_dir):
    """The most rows alive at once while those pieces are folded."""
    alive = [0]
    peak = []
    reading = fold.row_stream

    def watched_stream(path):
        for row in reading(path):
            watched = _WatchedRow(row, alive)
            peak.append(alive[0])
            yield watched

    folders = _piece_folders(experiment_dir)
    monkeypatch.setattr(fold, "row_stream", watched_stream)
    _folded(experiment_dir, folders, out_dir)
    return max(peak)


def _running_float_sum(values) -> float:
    """A plain running float sum, the thing an exact sum is not.

    The builtin sum is not it from CPython 3.12 on, which sums floats
    with Neumaier's compensation (What's New in Python 3.12, gh-100425).
    """
    total = 0.0
    for value in values:
        total += float(value)
    return total


def _exact_sum(values) -> fold.ExactSum:
    """An exact sum given the values in order."""
    running = fold.ExactSum()
    for value in values:
        running.add(value)
    return running


def _spread_values(generator, count: int, largest_exponent: int) -> list:
    """Random values whose decimal exponents span +-largest_exponent."""
    values = []
    for _index in range(count):
        exponent = generator.randint(-largest_exponent, largest_exponent)
        digits = generator.random()
        value = digits * 10.0**exponent
        values.append(value)
    return values


def test_an_exact_sum_equals_math_fsum_of_values_that_cancel():
    """Values whose plain float sum cancels away."""
    running = _exact_sum(CANCELLING)

    assert running.total() == math.fsum(CANCELLING)
    assert _running_float_sum(CANCELLING) != math.fsum(CANCELLING)


def test_an_exact_sum_equals_math_fsum_of_random_values_property():
    """Two hundred random spreads of fifty values against math.fsum."""
    generator = random.Random(2026)
    for _attempt in range(200):
        values = _spread_values(generator, 50, 30)
        running = _exact_sum(values)
        assert running.total() == math.fsum(values)


def test_an_exact_sum_equals_math_fsum_only_on_finite_values():
    """Where the claim stops, which is why the docstring says finite.

    The partials loop computes inf - (inf - 1.0) and lands on nan where
    math.fsum lands on inf, and overflowing partials raise a different
    exception from math.fsum's or none. These are the four cases
    CPython's own test of fsum pins (test_math.py:735-740). No column a
    fold sums can reach them, so nothing refuses them and this test
    says where the two part company rather than asking them to agree.
    """
    finite = _exact_sum(CANCELLING)
    assert finite.total() == math.fsum(CANCELLING)

    with_infinity = fold.ExactSum()
    with_infinity.add(1.0)
    with_infinity.add(math.inf)
    past_infinity = with_infinity.total()
    assert math.isnan(past_infinity)
    assert math.fsum([1.0, math.inf]) == math.inf

    overflowing = fold.ExactSum()
    overflowing.add(1e308)
    overflowing.add(1e308)
    with pytest.raises(ValueError):
        overflowing.total()
    with pytest.raises(OverflowError):
        math.fsum([1e308, 1e308])


def test_an_exact_sum_is_the_same_whichever_order_the_values_arrive_in():
    """A fold reads the pieces in whatever order it was given them.

    The values are chosen so that the thing this replaces does not have
    the property: a plain running float sum of one big value and ten
    ones loses every one added after the big value, because the gap
    between neighbouring floats at 1e16 is 2, and keeps every one added
    before it, so it reads 1e16 forwards and 1e16 + 10 backwards. The
    first assertion is that, so a reader can see the test would pass on
    anything.
    """
    swallowed = [1e16] + [1.0] * 10
    backwards_first = list(reversed(swallowed))
    assert _running_float_sum(swallowed) != _running_float_sum(backwards_first)
    forwards = _exact_sum(swallowed)
    backwards = _exact_sum(backwards_first)
    assert forwards.total() == backwards.total()
    assert forwards.total() == math.fsum(swallowed)

    generator = random.Random(7)
    values = _spread_values(generator, 200, 40)
    spread = _exact_sum(values)
    assert spread.total() == math.fsum(values)


def test_an_exact_sum_skips_a_zero_and_still_equals_math_fsum():
    """A zero changes no partial, and fsum returns 0.0 and never -0.0.

    The zeros sit inside values that cancel, so the sum is 2.0 where a
    plain running float sum reads 1.0: skipping the zeros must not cost
    the exactness around them. The two negative zeros on their own are
    the case ExactSum.add's docstring names, math.fsum of zeros being
    0.0 and never -0.0.
    """
    zeros = (1e100, 0.0, 1.0, -0.0, -1e100, 0.0, 1.0)
    assert _running_float_sum(zeros) != math.fsum(zeros)
    with_zeros = _exact_sum(zeros)
    assert with_zeros.total() == math.fsum(zeros)
    assert with_zeros.total() == 2.0
    only_zeros = fold.ExactSum()
    only_zeros.add(-0.0)
    only_zeros.add(-0.0)
    two_zeros = [-0.0, -0.0]
    total = only_zeros.total()
    assert total == math.fsum(two_zeros)
    sign = math.copysign(1.0, total)
    assert sign == 1.0


def _uniform_values(generator, count: int, largest: float) -> list:
    values = []
    for _index in range(count):
        value = generator.random() * largest
        values.append(value)
    return values


def _row_totals_of(values) -> fold.RowTotals:
    """Row totals given one row a value, under the column name value."""
    totals = fold.RowTotals(means=("value",))
    for value in values:
        totals.add({"value": value})
    return totals


def test_row_totals_mean_is_statistics_fmean_of_the_rows():
    """Which is what the summary columns were before they were streamed."""
    generator = random.Random(11)
    values = _uniform_values(generator, 500, 1e6)
    totals = _row_totals_of(values)
    assert totals.mean("value") == statistics.fmean(values)


def test_row_totals_read_a_csv_files_text_and_a_measurements_numbers_alike():
    """One accumulator serves the fold and the single run."""
    text_rows = fold.RowTotals(
        means=("value",), maxes=("value",), sums=("count",)
    )
    number_rows = fold.RowTotals(
        means=("value",), maxes=("value",), sums=("count",)
    )
    text_rows.add({"value": "0.5", "count": "3"})
    text_rows.add({"value": "1.5", "count": "4"})
    number_rows.add({"value": 0.5, "count": 3})
    number_rows.add({"value": 1.5, "count": 4})
    assert text_rows.mean("value") == number_rows.mean("value")
    assert text_rows.maxes["value"] == number_rows.maxes["value"]
    assert text_rows.sums["count"] == number_rows.sums["count"]
    assert text_rows.sums["count"] == 7


def test_merged_rows_are_the_stable_sort_of_the_files_rows(tmp_path):
    """Two files whose rows interleave, and rows of equal place in both.

    The stable sort of the concatenation keeps a row of equal place
    behind every row the earlier file gave, which is the order the
    merge yields because the file's index breaks the tie.
    """
    first_rows = [_row(1, "a"), _row(3, "b"), _row(3, "c"), _row(7, "d")]
    second_rows = [_row(2, "e"), _row(3, "f"), _row(8, "g")]
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    _write_rows(first_path, first_rows)
    _write_rows(second_path, second_rows)

    paths = [first_path, second_path]
    streamed = fold.merged_rows(paths, _place_of)
    merged = list(streamed)
    together = first_rows + second_rows
    sorted_rows = sorted(together, key=_place_of)
    assert merged == sorted_rows
    values = [row["value"] for row in merged]
    assert values == list("aebcfdg")


def test_a_merge_of_disjoint_pieces_holds_one_file_open_at_a_time(tmp_path):
    """A capped point's pieces are too many to open at once.

    Three hundred files of three rows each, seed ranges apart as a
    point's pieces are, given in reverse. The order's referent is the
    stable sort of their rows; the open files are the process's own
    descriptors, /proc/self/fd.
    """
    file_count = 300
    paths = _disjoint_piece_files(tmp_path, file_count)
    descriptors_before = _open_descriptor_count()

    merged, descriptors_peak = _merged_and_peak_descriptors(paths)

    together = _rows_of_files(paths)
    sorted_rows = sorted(together, key=_place_of)
    assert merged == sorted_rows
    assert len(merged) == 3 * file_count
    assert descriptors_peak - descriptors_before == 1


def _disjoint_piece_files(folder, count: int) -> list:
    """Files of three rows each, places 3i to 3i + 2, the last one first."""
    paths = []
    for piece in range(count):
        first_place = 3 * piece
        rows = _three_rows_from(first_place, piece)
        path = folder / f"piece_{piece}.csv"
        _write_rows(path, rows)
        paths.append(path)
    paths.reverse()
    return paths


def _three_rows_from(first_place: int, value) -> list:
    rows = []
    for offset in range(3):
        place = first_place + offset
        row = _row(place, value)
        rows.append(row)
    return rows


def _rows_of_files(paths: list) -> list:
    rows = []
    for path in paths:
        stream = fold.row_stream(path)
        rows.extend(stream)
    return rows


def _open_descriptor_count() -> int:
    descriptors = os.listdir("/proc/self/fd")
    return len(descriptors)


def _merged_and_peak_descriptors(paths: list) -> tuple:
    """The merged rows, and the most descriptors open while they came."""
    merged = []
    peak = _open_descriptor_count()
    for row in fold.merged_rows(paths, _place_of):
        merged.append(row)
        open_now = _open_descriptor_count()
        peak = max(peak, open_now)
    return merged, peak


def test_a_file_with_no_rows_of_this_kind_is_skipped(tmp_path):
    """A piece that holds no row of this kind wrote no such file."""
    path = tmp_path / "first.csv"
    only_row = _row(1, "a")
    _write_rows(path, [only_row])
    missing = tmp_path / "never_written.csv"
    paths = [path, missing]
    streamed = fold.merged_rows(paths, _place_of)
    merged = list(streamed)
    assert merged == [only_row]


def test_a_file_whose_rows_go_backwards_is_refused(tmp_path):
    """A merge assumes sorted inputs, so an unsorted file is not folded."""
    path = tmp_path / "backwards.csv"
    backwards = [_row(3, "a"), _row(1, "b")]
    _write_rows(path, backwards)
    merged = fold.merged_rows([path], _place_of)
    with pytest.raises(refusal.RefusalError) as refused:
        list(merged)
    assert "holds a row at 1 after a row at 3" in str(refused.value)


def test_a_row_file_leaves_a_column_its_row_lacks_empty(tmp_path):
    """Which is write_csv's rule, so the two agree."""
    path = tmp_path / "written.csv"
    first = _row(1, "a")
    second = {"place": "2"}
    with fold.RowFile(path, ["place", "value"]) as out_file:
        out_file.write(first)
        out_file.write(second)
    written = path.read_bytes()
    assert written == b"place,value\r\n1,a\r\n2,\r\n"


def test_a_row_file_that_got_no_row_is_not_written(tmp_path):
    """A run folder's file with nothing in it is not a file of zeros."""
    path = tmp_path / "never.csv"
    with fold.RowFile(path, ["place", "value"]):
        pass
    assert not path.exists()


def test_a_fold_holds_one_row_of_each_folder_however_many_shots_they_hold(
    tmp_path, monkeypatch
):
    """The memory rule, which is why the fold streams at all.

    Three pieces are folded twice: once holding one shot each and once
    holding three. The most rows alive at any moment is the same both
    times and is the piece count and the row being folded, not a
    piece's rows, so an experiment's pieces cost what a smoke test's do.
    """
    piece_count = 3
    one_shot_dir = _pieces_of_one_point(tmp_path, 3, 1)
    three_shot_dir = _pieces_of_one_point(tmp_path, 9, 3)
    one_shot_out = tmp_path / "folded_of_3"
    three_shot_out = tmp_path / "folded_of_9"
    one_shot_peak = _peak_rows_alive(monkeypatch, one_shot_dir, one_shot_out)
    three_shot_peak = _peak_rows_alive(
        monkeypatch, three_shot_dir, three_shot_out
    )
    assert one_shot_peak == three_shot_peak
    assert three_shot_peak <= piece_count + 1


def _rows_of(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _without_a_point(run_dir, name):
    """The folder as a tree that never measured that latency point wrote it."""
    folder = pathlib.Path(run_dir)
    shots_path = folder / "shots.csv"
    rows = _rows_of(shots_path)
    for row in rows:
        row.pop(f"{name}_mean_us")
        row.pop(f"{name}_max_us")
    _write_rows(shots_path, rows)
    samples_path = folder / "window_samples.csv"
    kept = []
    for row in _rows_of(samples_path):
        if row["name"] == name:
            continue
        kept.append(row)
    _write_rows(samples_path, kept)


def _each_without_a_point(folders, name) -> None:
    for folder in folders:
        _without_a_point(folder, name)


def _without_columns(row: dict, columns: tuple) -> dict:
    kept = {}
    for column, value in row.items():
        if column not in columns:
            kept[column] = value
    return kept


def test_a_fold_reports_the_latency_points_the_folders_rows_hold(tmp_path):
    """A newer tree folds the folders an older tree wrote.

    The 500 shard folders of one weak_ler experiment hold sixteen
    latency points and this tree measures twenty-three, so a summary that
    asked for its own columns could not read those folders at all. Here
    two pieces lose one point's columns, as an older tree's pieces lack
    them, and the fold reports the points they hold and every other
    column exactly as the fold of the same pieces whole does.
    """
    experiment_dir = _pieces_of_one_point(tmp_path, 4, 2)
    folders = _piece_folders(experiment_dir)
    whole_dir = tmp_path / "whole"
    older_dir = tmp_path / "older"
    _folded(experiment_dir, folders, whole_dir)
    _each_without_a_point(folders, "confidence")
    _folded(experiment_dir, folders, older_dir)

    whole_path = whole_dir / "sweep.csv"
    older_path = older_dir / "sweep.csv"
    whole = _rows_of(whole_path)
    older = _rows_of(older_path)
    dropped = (
        "confidence_mean_us",
        "confidence_median_us",
        "confidence_p99_us",
        "confidence_max_us",
    )
    whole_without_the_point = _without_columns(whole[0], dropped)
    assert set(dropped) <= set(whole[0])
    assert older[0] == whole_without_the_point


def test_a_fold_reports_no_strong_service_mean_for_folders_without_its_sum(
    tmp_path,
):
    """A folder whose shots hold no strong service sum gets no mean.

    The point's mean is the shots' summed service over their strong
    decodes, so a folder without the sum reports no strong_service_mean_us
    and every other column as the whole fold does, the rule a latency
    point the folder did not measure keeps.
    """
    experiment_dir = _pieces_of_one_point(tmp_path, 4, 2)
    folders = _piece_folders(experiment_dir)
    whole_dir = tmp_path / "whole"
    older_dir = tmp_path / "older"
    _folded(experiment_dir, folders, whole_dir)
    _each_without_a_shot_column(folders, "strong_service_sum_us")
    _folded(experiment_dir, folders, older_dir)

    whole_path = whole_dir / "sweep.csv"
    older_path = older_dir / "sweep.csv"
    whole = _rows_of(whole_path)
    older = _rows_of(older_path)
    dropped = ("strong_service_mean_us",)
    assert set(dropped) <= set(whole[0])
    assert older[0] == _without_columns(whole[0], dropped)


def _each_without_a_shot_column(folders, column) -> None:
    for folder in folders:
        shots_path = pathlib.Path(folder) / "shots.csv"
        rows = _rows_of(shots_path)
        for row in rows:
            row.pop(column)
        _write_rows(shots_path, rows)


def test_pieces_of_one_point_that_hold_different_columns_are_refused(
    tmp_path,
):
    """A point's pieces were measured by one tree, so they hold one set.

    Folding them would write the older piece's shots with that column
    empty, which reads as a measurement of nothing. The sentence names
    the folder that lacks the columns whichever order the folders were
    given, because the folder the walk reaches second is not always the
    one that lacks anything.
    """
    experiment_dir = _pieces_of_one_point(tmp_path, 4, 2)
    folders = _piece_folders(experiment_dir)
    _without_a_point(folders[0], "confidence")
    lacking = folders[0]
    out_dir = tmp_path / "folded"
    with pytest.raises(refusal.RefusalError) as refused:
        _folded(experiment_dir, folders, out_dir)
    said = str(refused.value)
    assert said.startswith(f"{lacking} does not hold the columns")
    assert "confidence_mean_us" in said
    assert not out_dir.exists()

    backwards = list(reversed(folders))
    with pytest.raises(refusal.RefusalError) as refused_backwards:
        _folded(experiment_dir, backwards, out_dir)
    said_backwards = str(refused_backwards.value)
    assert said_backwards.startswith(f"{lacking} does not hold the columns")
    assert not out_dir.exists()


def test_points_that_measured_different_columns_fold_to_one_header(tmp_path):
    """A quiet point and a burst point of one grid fold into one run folder.

    A burst shot measures whether it was caught in time and a quiet shot
    has no burst to catch, so their pieces hold different columns. The
    folded file takes every column any point holds, first seen first, as
    write_csv does for a run's own rows, and a point's cell for a column
    it did not measure is empty.
    """
    experiment_dir = _burst_and_quiet_pieces(tmp_path)

    run_dir = experiment_dir
    shots_path = run_dir / "shots.csv"
    sweep_path = run_dir / "sweep.csv"
    shots = _rows_of(shots_path)
    sweep = _rows_of(sweep_path)

    quiet_shots = _rows_with_qpu_kind(shots, sweep, "stim_device")
    burst_shots = _rows_with_qpu_kind(shots, sweep, "burst_stim")
    assert [row["burst_caught_in_time"] for row in quiet_shots] == [""]
    assert [row["burst_caught_in_time"] for row in burst_shots] != [""]
    quiet_points = _rows_with_qpu_kind(sweep, sweep, "stim_device")
    assert [row["caught_in_time_share"] for row in quiet_points] == [""]


def test_pieces_fold_to_the_same_bytes_whichever_order_they_come_in(tmp_path):
    """The fold orders the pieces itself, so their order is no input.

    Four pieces of one shot, folded as listed and again reversed, write
    every file byte for byte alike.
    """
    experiment_dir = _pieces_of_one_point(tmp_path, 4, 1)
    folders = _piece_folders(experiment_dir)
    backwards = list(reversed(folders))
    forwards_dir = tmp_path / "forwards"
    backwards_dir = tmp_path / "backwards"

    _folded(experiment_dir, folders, forwards_dir)
    _folded(experiment_dir, backwards, backwards_dir)

    forwards_bytes = _bytes_of_files(forwards_dir)
    backwards_bytes = _bytes_of_files(backwards_dir)
    assert forwards_bytes == backwards_bytes


def test_a_collect_run_on_to_a_raised_cap_records_every_seed(tmp_path):
    """A resumed collect's record holds the saved seeds and the new ones."""
    config_path = _one_point_config(tmp_path, 2, 1)
    experiment_dir = tmp_path / "experiment"
    command.main(["run", str(config_path), "--out", str(experiment_dir)])
    run_dir = experiment_dir
    first_seeds = _seeds_of_the_one_point(run_dir)
    _one_point_config(tmp_path, 4, 1)

    command.main(["run", str(config_path), "--out", str(experiment_dir)])

    assert first_seeds == [[0, 2]]
    assert _seeds_of_the_one_point(run_dir) == [[0, 4]]
    assert _seeds_of_every_shot(run_dir) == ["0", "1", "2", "3"]


# every file a fold writes from the pieces' rows
FOLDED_FILES = (
    "sweep.csv",
    "shots.csv",
    "shot_links.csv",
    "window_samples.csv",
)


def _bytes_of_files(run_dir) -> dict:
    contents = {}
    for name in FOLDED_FILES:
        path = run_dir / name
        contents[name] = path.read_bytes()
    return contents


def _seeds_of_every_shot(run_dir) -> list:
    shots_path = run_dir / "shots.csv"
    rows = _rows_of(shots_path)
    return [row["seed"] for row in rows]


def _seeds_of_the_one_point(run_dir) -> list:
    """The seed ranges the results folder's one point record names."""
    (record_path,) = run_dir.glob("points/*/machine.json")
    record_text = record_path.read_text()
    record = json.loads(record_text)
    return record["seeds"]


# a burst at round 5 of a shot's thirty, so a burst shot measures its catch
BURST_QPU = {
    "kind": "burst_stim",
    "burst_onset_round": 5,
    "burst_decay_rounds": 600.0,
    "burst_radius": 3.1,
    "burst_error_probability": 0.01,
}


def _burst_and_quiet_pieces(tmp_path):
    """One shot each of a quiet and a burst point, collected in pieces."""
    overrides = yaml_configs.online_threshold()
    overrides["escalation"] = {
        "kind": "switching",
        "gap_threshold_db": 20.0,
        "strong_window": "redo_window",
    }
    overrides["burst_detector"] = {"kind": "event_count"}
    # the event count's windows need 22 rounds of a shot
    overrides["workload"] = yaml_configs.memory_workload(30)
    overrides["sweep"] = [
        {
            "axes": {
                "qpu": [{"kind": "stim_device"}, BURST_QPU],
                "workload.arguments.physical_error_probability": [0.001],
                "qpu.distance": [3],
                "qpu.round_period_microseconds": [1.0],
            },
            "collection": {"max_shots": 1},
        }
    ]
    config_path = yaml_configs.write_config(tmp_path, overrides)
    experiment_dir = tmp_path / "experiment"
    command.main(["run", str(config_path), "--out", str(experiment_dir)])
    return experiment_dir


def _rows_with_qpu_kind(rows, sweep, kind) -> list:
    """The rows of the points whose QPU is that kind, sweep.csv naming it."""
    point_ids = set()
    for point in sweep:
        qpu = json.loads(point["qpu"])
        if qpu["kind"] == kind:
            point_ids.add(point["point_id"])
    selected = []
    for row in rows:
        if row["point_id"] in point_ids:
            selected.append(row)
    return selected


def test_a_switching_run_writes_both_confidence_files(tmp_path):
    overrides = yaml_configs.fixed_threshold_switching()
    run_dir = _confidence_run(tmp_path, overrides, 2)
    confidence_path = run_dir / "window_confidence.csv"
    histogram_path = run_dir / "confidence_histogram.csv"

    window_rows = _rows_of(confidence_path)
    histogram_rows = _rows_of(histogram_path)

    assert window_rows
    assert {row["signal"] for row in window_rows} == {"complementary_gap"}
    assert {row["seed"] for row in window_rows} == {"0", "1"}
    assert histogram_rows


def test_the_histogram_counts_every_window_and_every_shot(tmp_path):
    overrides = yaml_configs.fixed_threshold_switching()
    overrides["observation"] = {"confidence_shot_count": "all"}
    run_dir = _confidence_run(tmp_path, overrides, 3)
    confidence_path = run_dir / "window_confidence.csv"
    histogram_path = run_dir / "confidence_histogram.csv"

    window_rows = _rows_of(confidence_path)
    histogram_rows = _rows_of(histogram_path)

    assert _counts_of(histogram_rows, "window") == len(window_rows)
    assert _counts_of(histogram_rows, "shot_minimum") == 3


def test_no_sampled_shot_writes_only_the_histogram(tmp_path):
    overrides = yaml_configs.fixed_threshold_switching()
    overrides["observation"] = {"confidence_shot_count": 0}
    run_dir = _confidence_run(tmp_path, overrides, 2)
    confidence_path = run_dir / "window_confidence.csv"
    histogram_path = run_dir / "confidence_histogram.csv"

    histogram_rows = _rows_of(histogram_path)

    assert not confidence_path.exists()
    assert _counts_of(histogram_rows, "shot_minimum") == 2


def test_pieces_fold_to_the_confidence_files_of_one_uncut_run(tmp_path):
    """Four one-shot pieces fold to the files one four-shot run writes."""
    overrides = yaml_configs.fixed_threshold_switching()
    uncut_dir = _confidence_run(tmp_path, overrides, 4, out="uncut")
    pieces_dir = _confidence_run(tmp_path, overrides, 4, 1, out="pieces")
    uncut_windows = uncut_dir / "window_confidence.csv"
    folded_windows = pieces_dir / "window_confidence.csv"
    uncut_histogram = uncut_dir / "confidence_histogram.csv"
    folded_histogram = pieces_dir / "confidence_histogram.csv"

    assert folded_windows.read_bytes() == uncut_windows.read_bytes()
    assert folded_histogram.read_bytes() == uncut_histogram.read_bytes()


def test_a_run_whose_escalation_reads_no_confidence_writes_neither_file(
    tmp_path,
):
    run_dir = _confidence_run(tmp_path, {}, 2)
    confidence_path = run_dir / "window_confidence.csv"
    histogram_path = run_dir / "confidence_histogram.csv"

    assert not confidence_path.exists()
    assert not histogram_path.exists()


def test_a_piece_records_the_shots_its_confidence_rows_cover(tmp_path):
    """piece.json says what confidence it wrote: a count, all, or none."""
    switching = yaml_configs.fixed_threshold_switching()
    every = yaml_configs.fixed_threshold_switching()
    every["observation"] = {"confidence_shot_count": "all"}
    _confidence_run(tmp_path, switching, 1, out="sampled")
    _confidence_run(tmp_path, every, 1, out="every")
    _confidence_run(tmp_path, {}, 1, out="plain")

    sampled = _the_one_piece(tmp_path, "sampled")
    every_shot = _the_one_piece(tmp_path, "every")
    plain = _the_one_piece(tmp_path, "plain")

    assert sampled["confidence_shot_count"] == 100
    assert every_shot["confidence_shot_count"] == "all"
    assert plain["confidence_shot_count"] is None


def test_pieces_of_one_point_that_recorded_confidence_apart_are_refused(
    tmp_path,
):
    """A piece from before the confidence files would leave them short.

    Its shots are in shots.csv but in neither confidence file, so the
    histogram would count fewer shots than the sweep row. The fold
    refuses and names the pieces, before it writes anything.
    """
    overrides = yaml_configs.fixed_threshold_switching()
    _confidence_run(tmp_path, overrides, 2, 1)
    experiment_dir = tmp_path / "experiment"
    folders = _piece_folders(experiment_dir)
    older = folders[0]
    _as_a_piece_from_before_the_confidence_files(older)
    out_dir = tmp_path / "folded"

    with pytest.raises(refusal.RefusalError) as refused:
        _folded(experiment_dir, folders, out_dir)

    said = str(refused.value)
    assert "recorded confidence for different shots" in said
    assert str(older) in said
    assert not out_dir.exists()


def test_pieces_of_one_point_that_ran_different_commits_are_refused(
    tmp_path,
):
    """A resumed collect from another tree would pool two simulators.

    The second piece is saved as a process at another commit saves it;
    the fold names both trees' pieces and writes nothing.
    """
    experiment_dir = _pieces_of_one_point(tmp_path, 2, 1)
    folders = _piece_folders(experiment_dir)
    later = folders[1]
    other_commit = "b" * 40
    _as_a_piece_run_at_commit(later, other_commit)
    out_dir = tmp_path / "folded"

    with pytest.raises(refusal.RefusalError) as refused:
        _folded(experiment_dir, folders, out_dir)

    said = str(refused.value)
    assert "ran different code" in said
    assert f"{later} (commit {other_commit}" in said
    assert not out_dir.exists()


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
        point_id="p",
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
        point_id="p",
        algorithm="pymatching",
        seed=0,
        confidence=confidence,
        logical_failure=False,
        is_scored=False,
    )

    assert report.window_confidence_rows([shot]) == []
    assert report.confidence_histogram_rows([shot]) == []


def _confidence_run(
    tmp_path, overrides, shots, piece_shots=None, out="experiment"
):
    """A one-point d3 collect of shots, in pieces if asked; its run folder."""
    if piece_shots is not None:
        piece_rounds = piece_shots * ROUNDS_PER_SHOT
        overrides["collection"] = {"piece_rounds": piece_rounds}
    overrides["sweep"] = [
        {
            "axes": {
                "workload.arguments.physical_error_probability": [0.003],
                "qpu.distance": [3],
                "qpu.round_period_microseconds": [1.0],
            },
            "collection": {"max_shots": shots},
        }
    ]
    config_path = yaml_configs.write_config(tmp_path, overrides)
    experiment_dir = tmp_path / out
    command.main(["run", str(config_path), "--out", str(experiment_dir)])
    return experiment_dir


def _counts_of(rows, histogram) -> int:
    """The counts of one histogram's rows, summed."""
    counts = []
    for row in rows:
        if row["histogram"] == histogram:
            counts.append(int(row["count"]))
    return sum(counts)


def _one_count(histogram, gap_low_decibels, escalated) -> dict:
    """A histogram row of one window or shot of that no-failure shot."""
    return {
        "point_id": "p",
        "algorithm": "pymatching",
        "signal": "complementary_gap",
        "histogram": histogram,
        "gap_low_decibels": gap_low_decibels,
        "escalated": escalated,
        "shot_failed": False,
        "count": 1,
    }


def _the_one_piece(tmp_path, out) -> dict:
    """The piece.json of an experiment that saved one piece."""
    experiment_dir = tmp_path / out
    (piece_path,) = experiment_dir.glob("pieces/*/*/piece.json")
    piece_text = piece_path.read_text()
    return json.loads(piece_text)


def _as_a_piece_run_at_commit(folder, commit: str) -> None:
    """The piece as a process at another commit would have saved it."""
    piece_path = folder / "piece.json"
    piece_text = piece_path.read_text()
    piece = json.loads(piece_text)
    piece["commit"] = commit
    other_text = json.dumps(piece)
    piece_path.write_text(other_text)


def _as_a_piece_from_before_the_confidence_files(folder) -> None:
    """The piece as a tree without confidence output would have saved it."""
    (folder / "window_confidence.csv").unlink()
    (folder / "confidence_histogram.csv").unlink()
    piece_path = folder / "piece.json"
    piece_text = piece_path.read_text()
    piece = json.loads(piece_text)
    del piece["confidence_shot_count"]
    older_text = json.dumps(piece)
    piece_path.write_text(older_text)
