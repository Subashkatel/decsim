"""The rules a fold of many run folders keeps (decsim/front/fold.py).

Four of them, each pinned here against the thing it claims to equal:
the merged order is the stable sort of the folders' rows, a streamed sum
is math.fsum of the values it was given, what a fold holds does not grow
with the shots the folders hold, and a summary reports the latency
points its folders' rows hold. The third is the reason the module
exists: `decsim combine` over the 500 shard folders of the 2026-09-09
weak_ler campaign was OOM-killed at 120 GB while reading their 115
million link rows into lists. The fourth is why that campaign can be
folded at all: its shards hold the sixteen latency points that tree
measured, and this tree measures twenty-two.
"""

import csv
import math
import pathlib
import random
import statistics

import pytest

import decsim.front.command as command
import decsim.front.fold as fold
import decsim.front.refusal as refusal
import decsim.front.report as report
import decsim.front.run_folder as run_folder
import tests.front.yaml_configs as yaml_configs

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


def _one_point_config(tmp_path, shots):
    sweep = {
        "sweep": [
            {
                "physical_error_probability": [0.001],
                "distance": [3],
                "round_period_us": [1.0],
                "shots": shots,
            }
        ]
    }
    return yaml_configs.write_config(tmp_path, sweep)


def _shards_of_one_point(tmp_path, shots, shards):
    """One point's shots collected into `shards` folders, one seed a unit."""
    config_path = _one_point_config(tmp_path, shots)
    run_dirs = []
    for index in range(shards):
        run_dir = tmp_path / f"shard{index}_of_{shots}"
        run_dirs.append(str(run_dir))
        command.main(
            [
                "collect",
                str(config_path),
                "--out",
                str(run_dir),
                "--shots-per-unit",
                "1",
                "--shard",
                f"{index}/{shards}",
            ]
        )
    return run_dirs


def _peak_rows_alive(monkeypatch, run_dirs, out_dir):
    """The most rows alive at once while those folders are folded."""
    alive = [0]
    peak = []
    reading = fold.row_stream

    def watched_stream(path):
        for row in reading(path):
            watched = _WatchedRow(row, alive)
            peak.append(alive[0])
            yield watched

    monkeypatch.setattr(fold, "row_stream", watched_stream)
    report.combine(run_dirs, out_dir)
    return max(peak)


def test_an_exact_sum_equals_math_fsum_of_the_values_it_was_given():
    """Values whose plain float sum cancels away, and random ones."""
    running = fold.ExactSum()
    for value in CANCELLING:
        running.add(value)
    assert running.total() == math.fsum(CANCELLING)
    assert sum(CANCELLING) != math.fsum(CANCELLING)

    generator = random.Random(2026)
    for _attempt in range(200):
        values = []
        for _index in range(50):
            exponent = generator.randint(-30, 30)
            digits = generator.random()
            value = digits * 10.0**exponent
            values.append(value)
        running = fold.ExactSum()
        for value in values:
            running.add(value)
        assert running.total() == math.fsum(values)


def test_an_exact_sum_is_the_same_whichever_order_the_values_arrive_in():
    """A fold reads the shards in whatever order it was given them."""
    generator = random.Random(7)
    values = []
    for _index in range(200):
        exponent = generator.randint(-40, 40)
        digits = generator.random()
        value = digits * 10.0**exponent
        values.append(value)
    forwards = fold.ExactSum()
    for value in values:
        forwards.add(value)
    backwards = fold.ExactSum()
    for value in reversed(values):
        backwards.add(value)
    assert forwards.total() == backwards.total()
    assert forwards.total() == math.fsum(values)


def test_an_exact_sum_skips_a_zero_and_still_equals_math_fsum():
    """A zero changes no partial, and fsum returns 0.0 and never -0.0."""
    zeros = (0.0, 0.5, 0.0, -0.0)
    with_zeros = fold.ExactSum()
    for value in zeros:
        with_zeros.add(value)
    assert with_zeros.total() == math.fsum(zeros)
    only_zeros = fold.ExactSum()
    only_zeros.add(0.0)
    only_zeros.add(-0.0)
    two_zeros = [0.0, -0.0]
    total = only_zeros.total()
    assert total == math.fsum(two_zeros)
    sign = math.copysign(1.0, total)
    assert sign == 1.0


def test_row_totals_mean_is_statistics_fmean_of_the_rows():
    """Which is what the summary columns were before they were streamed."""
    generator = random.Random(11)
    values = []
    for _index in range(500):
        value = generator.random() * 1e6
        values.append(value)
    totals = fold.RowTotals(means=("value",))
    for value in values:
        totals.add({"value": value})
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
    behind every row the earlier file gave, which is the order
    heapq.merge yields because the stream's index breaks the tie.
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


def test_a_file_with_no_rows_of_this_kind_is_skipped(tmp_path):
    """A shard whose index selected no work unit wrote no such file."""
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


def test_a_row_file_takes_its_header_from_the_first_row(tmp_path):
    """Which is where write_csv takes it from, so the two agree."""
    path = tmp_path / "written.csv"
    first = _row(1, "a")
    second = _row(2, "b")
    with fold.RowFile(path) as out_file:
        out_file.write(first)
        out_file.write(second)
    written = path.read_bytes()
    assert written == b"place,value\r\n1,a\r\n2,b\r\n"


def test_a_row_file_that_got_no_row_is_not_written(tmp_path):
    """A run folder's file with nothing in it is not a file of zeros."""
    path = tmp_path / "never.csv"
    with fold.RowFile(path):
        pass
    assert not path.exists()


def test_a_fold_holds_one_row_of_each_folder_however_many_shots_they_hold(
    tmp_path, monkeypatch
):
    """The memory rule, which is why the fold streams at all.

    Three folders are folded twice: once holding one shot each and once
    holding three. The most rows alive at any moment is the same both
    times and is the folder count and the row being folded, not a
    folder's rows, so a campaign's folders cost what a smoke test's do.
    """
    shards = 3
    peaks = {}
    for shots in (3, 9):
        run_dirs = _shards_of_one_point(tmp_path, shots, shards)
        out_dir = tmp_path / f"combined_of_{shots}"
        peaks[shots] = _peak_rows_alive(monkeypatch, run_dirs, out_dir)
    assert peaks[3] == peaks[9]
    assert peaks[9] <= shards + 1


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


def test_a_fold_reports_the_latency_points_the_folders_rows_hold(tmp_path):
    """A newer tree folds the folders an older tree wrote.

    The 500 shard folders of the 2026-09-09 campaign hold sixteen
    latency points and this tree measures twenty-two, so a summary that
    asked for its own columns could not read those folders at all. Here
    two folders lose one point's columns, as an older tree's folders
    lack them, and the fold reports the points they hold and every other
    column exactly as the fold of the same folders whole does.
    """
    run_dirs = _shards_of_one_point(tmp_path, 4, 2)
    whole_dir = tmp_path / "whole"
    older_dir = tmp_path / "older"
    report.combine(run_dirs, whole_dir)
    for run_dir in run_dirs:
        _without_a_point(run_dir, "confidence")
    report.combine(run_dirs, older_dir)

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
    for column in dropped:
        assert column in whole[0]
        assert column not in older[0]
    for column in whole[0]:
        if column in dropped:
            continue
        assert older[0][column] == whole[0][column]


def test_folders_that_hold_different_columns_are_refused(tmp_path):
    """A folded file has one header, so its folders record one set.

    Folding them would write the older folder's shots with that column
    empty, which reads as a measurement of nothing.
    """
    run_dirs = _shards_of_one_point(tmp_path, 4, 2)
    _without_a_point(run_dirs[0], "confidence")
    out_dir = tmp_path / "combined"
    with pytest.raises(refusal.RefusalError) as refused:
        report.combine(run_dirs, out_dir)
    assert "does not hold the columns" in str(refused.value)
    assert "confidence_mean_us" in str(refused.value)
    assert not out_dir.exists()
