"""`decsim diff`: how two run folders differ."""

import argparse
import json
import math
import pathlib
import statistics
from typing import Optional

import decsim.collect as collect
import decsim.experiments.fold as fold
import decsim.experiments.refusal as refusal
import decsim.experiments.report as report
import decsim.experiments.run_folder as run_folder

# The parts of a point's resolved record diff compares: every setting,
# and the values the build derived from them (run_folder.record_point).
RESOLVED_PARTS = ("settings", "built")
# Columns diff does not compare: the host's own time per shot, which no
# setting sets, and the Wilson bounds, which the logical error rate's
# own verdict reads.
NOT_COMPARED_COLUMNS = (
    "sim_wall_seconds_per_shot",
    "ler_wilson_low",
    "ler_wilson_high",
)
# A latency point's mean over shots, beside its median, p99 and max
# (report._add_latency_point_columns); the other means are
# report.SHOT_MEANS. Only a mean has a standard error from its shots.
LATENCY_MEAN_SUFFIX = "_mean_us"
# The sweep.csv means named apart from the shots.csv column they average
# (report.summarize_point).
SHOT_COLUMN_OF_MEAN = {"windows_per_shot": "decoded_windows"}
# Two means agree within their error bars when they differ by at most
# this many standard errors of their difference (a 95% interval).
AGREEMENT_STANDARD_ERRORS = 1.96


def diff(first: pathlib.Path, second: pathlib.Path) -> list:
    """How two run folders differ: settings, inputs by hash, then results.

    A point is matched by its metadata, the values its sweep set, so two
    runs of different configs over one grid pair point by point, as
    sinter's plot groups by json_metadata across files (sinter
    _command/_main_plot.py --group_func). A result that differs is said
    to agree within its error bars or not: a logical error rate by its
    Wilson interval, a mean over shots by the standard error of its
    shots (shots.csv), which is where a decoder's measured wall clock
    moves a tick column between two runs of one yaml. A column with no
    error bar is compared exactly.
    """
    _refuse_a_folder_without_points(first)
    _refuse_a_folder_without_points(second)
    lines = _settings_lines(first, second)
    input_lines = _input_lines(first, second)
    lines.extend(input_lines)
    result_lines = _result_lines(first, second)
    lines.extend(result_lines)
    return lines


def main(argv: list) -> None:
    """The command line: two run folders, their differences printed."""
    parser = argparse.ArgumentParser(prog="decsim diff")
    parser.add_argument("first", help="the first run folder")
    parser.add_argument("second", help="the run folder to compare it with")
    parsed = parser.parse_args(argv)
    first = pathlib.Path(parsed.first)
    second = pathlib.Path(parsed.second)
    lines = diff(first, second)
    text = "\n".join(lines)
    print(text)


def _refuse_a_folder_without_points(run_dir: pathlib.Path) -> None:
    """A folder without resolved/ would read as settings that never differ."""
    if (run_dir / run_folder.RESOLVED_FOLDER).is_dir():
        return
    raise refusal.RefusalError(
        f"{run_dir} holds no resolved/ folder; name a folder decsim collect "
        "or decsim combine wrote"
    )


def _settings_lines(first: pathlib.Path, second: pathlib.Path) -> list:
    """Each point's settings and built values that differ, path by path."""
    first_points = _records_by_metadata(first)
    second_points = _records_by_metadata(second)
    lines = ["settings:"]
    unmatched = _unmatched_lines(first_points, second_points)
    lines.extend(unmatched)
    for point in _shared(first_points, second_points):
        first_values = run_folder.resolved_values(
            first_points[point], RESOLVED_PARTS
        )
        second_values = run_folder.resolved_values(
            second_points[point], RESOLVED_PARTS
        )
        point_text = _point_text(point)
        for path in _changed(first_values, second_values):
            first_value = first_values.get(path)
            second_value = second_values.get(path)
            lines.append(
                f"  {point_text} {path}: {first_value!r} -> {second_value!r}"
            )
    return _or_same(lines)


def _input_lines(first: pathlib.Path, second: pathlib.Path) -> list:
    """Each point's input files whose sha256 differs, or that one lacks."""
    first_points = _records_by_metadata(first)
    second_points = _records_by_metadata(second)
    lines = ["inputs:"]
    for point in _shared(first_points, second_points):
        first_hashes = _hashes(first, first_points[point]["id"])
        second_hashes = _hashes(second, second_points[point]["id"])
        point_text = _point_text(point)
        for name in _changed(first_hashes, second_hashes):
            lines.append(f"  {point_text} {name}: sha256 differs")
    return _or_same(lines)


def _result_lines(first: pathlib.Path, second: pathlib.Path) -> list:
    """Each point's sweep.csv columns that differ, with their verdict."""
    first_rows = _rows_by_metadata(first)
    second_rows = _rows_by_metadata(second)
    lines = ["results:"]
    unmatched = _unmatched_lines(first_rows, second_rows)
    lines.extend(unmatched)
    for point in _shared(first_rows, second_rows):
        first_row = first_rows[point]
        second_row = second_rows[point]
        first_shots = _shot_rows_of(first, first_row["point_id"])
        second_shots = _shot_rows_of(second, second_row["point_id"])
        comparison = _Comparison(point, first_shots, second_shots)
        point_lines = comparison.lines(first_row, second_row)
        lines.extend(point_lines)
    return _or_same(lines)


class _Comparison:
    """One point's results in two folders, and the error bars of each."""

    def __init__(
        self, point: str, first_shots: list, second_shots: list
    ) -> None:
        self.point_text = _point_text(point)
        self.first_shots = first_shots
        self.second_shots = second_shots

    def lines(self, first_row: dict, second_row: dict) -> list:
        """One line per compared column that differs, with its verdict."""
        lines = []
        for column in _changed(first_row, second_row):
            if column in NOT_COMPARED_COLUMNS:
                continue
            first_value = first_row.get(column)
            second_value = second_row.get(column)
            verdict = self.verdict(column, first_row, second_row)
            lines.append(
                f"  {self.point_text} {column}: {first_value} -> "
                f"{second_value}, {verdict}"
            )
        return lines

    def verdict(self, column: str, first_row: dict, second_row: dict) -> str:
        """Whether two differing values agree within their error bars."""
        if column == "logical_error_rate":
            return _wilson_verdict(first_row, second_row)
        shot_column = _shot_column_of(column)
        first_error = _standard_error(self.first_shots, shot_column)
        second_error = _standard_error(self.second_shots, shot_column)
        if first_error is None or second_error is None:
            return "no error bar: compared exactly"
        spread = math.hypot(first_error, second_error)
        difference = first_row[column] - second_row[column]
        gap = abs(difference)
        if gap <= AGREEMENT_STANDARD_ERRORS * spread:
            return "within error bars"
        return "beyond error bars"


def _shot_column_of(column: str) -> Optional[str]:
    """The shots.csv column a sweep.csv column is the mean of, or None."""
    if column in SHOT_COLUMN_OF_MEAN:
        return SHOT_COLUMN_OF_MEAN[column]
    if column in report.SHOT_MEANS or column.endswith(LATENCY_MEAN_SUFFIX):
        return column
    return None


def _wilson_verdict(first_row: dict, second_row: dict) -> str:
    """Overlapping Wilson intervals have not been shown to differ."""
    first_below = first_row["ler_wilson_high"] < second_row["ler_wilson_low"]
    second_below = second_row["ler_wilson_high"] < first_row["ler_wilson_low"]
    if first_below or second_below:
        return "beyond error bars"
    return "within error bars"


def _standard_error(shot_rows: list, column: Optional[str]) -> Optional[float]:
    """The standard error of a column's mean over shots; None without one.

    A column that is no mean (None), one shots.csv does not hold or
    holds as something other than a number, and one over fewer than two
    shots has no error bar.
    """
    values = []
    for row in shot_rows:
        value = row.get(column)
        if not _is_number(value):
            return None
        values.append(value)
    count = len(values)
    if count < 2:
        return None
    deviation = statistics.stdev(values)
    return deviation / math.sqrt(count)


def _is_number(value) -> bool:
    return isinstance(value, (int, float))


def _shot_rows_of(run_dir: pathlib.Path, point_id: str) -> list:
    """One point's rows of shots.csv, the per-shot values of its means."""
    path = run_dir / "shots.csv"
    rows = []
    for row in fold.row_stream(path):
        if row["point_id"] == point_id:
            typed = fold.typed_row(row)
            rows.append(typed)
    return rows


def _records_by_metadata(run_dir: pathlib.Path) -> dict:
    """Each point's resolved/ record, keyed by its metadata's text."""
    by_id = run_folder.resolved_by_point(run_dir)
    records = {}
    for record in by_id.values():
        point = collect.metadata_text(record["metadata"])
        records[point] = record
    return records


def _rows_by_metadata(run_dir: pathlib.Path) -> dict:
    """sweep.csv's rows, keyed by their point's metadata's text.

    A shard records every point but holds rows of the points it ran.
    """
    records = _records_by_metadata(run_dir)
    by_id = report.rows_by_point(run_dir)
    rows = {}
    for point, record in records.items():
        if record["id"] in by_id:
            rows[point] = by_id[record["id"]]
    return rows


def _hashes(run_dir: pathlib.Path, point_id: str) -> dict:
    """A point's inputs/<id>/hashes.json."""
    path = (
        run_dir / run_folder.INPUTS_FOLDER / point_id / run_folder.HASHES_FILE
    )
    text = path.read_text()
    return json.loads(text)


def _point_text(point: str) -> str:
    return f"{point}:"


def _shared(first: dict, second: dict) -> list:
    shared = []
    for point in first:
        if point in second:
            shared.append(point)
    return shared


def _changed(first: dict, second: dict) -> list:
    """The keys whose values differ, or that one of the two lacks."""
    keys = list(first)
    for key in second:
        if key not in first:
            keys.append(key)
    changed = []
    for key in keys:
        if first.get(key) != second.get(key):
            changed.append(key)
    return changed


def _unmatched_lines(first: dict, second: dict) -> list:
    """A line for each point only one of the two folders holds."""
    lines = []
    for point in first:
        if point not in second:
            point_text = _point_text(point)
            lines.append(f"  {point_text} only in the first folder")
    for point in second:
        if point not in first:
            point_text = _point_text(point)
            lines.append(f"  {point_text} only in the second folder")
    return lines


def _or_same(lines: list) -> list:
    """A section with nothing under its title says so."""
    if len(lines) == 1:
        lines.append("  the same")
    return lines
