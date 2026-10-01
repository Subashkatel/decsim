"""`decsim diff`: how two run folders differ."""

import argparse
import dataclasses
import json
import math
import pathlib
import statistics
from typing import Optional

import numpy

import decsim.collect as collect
import decsim.experiments.failure_statistics as failure_statistics
import decsim.experiments.fold as fold
import decsim.experiments.refusal as refusal
import decsim.experiments.report as report
import decsim.experiments.run_folder as run_folder

# The parts of a point's resolved record diff compares: every setting,
# and the values the build derived from them (run_folder.record_point).
RESOLVED_PARTS = ("settings", "built")
# Each estimate column and the columns of its exact 95 percent limits
# (report._add_estimate_columns), which its verdict reads.
LIMITS_OF_ESTIMATE = {
    "logical_error_rate_estimate": (
        "logical_error_rate_low",
        "logical_error_rate_high",
    ),
    "logical_error_rate_per_round": (
        "logical_error_rate_per_round_low",
        "logical_error_rate_per_round_high",
    ),
}
# Columns diff does not compare: the host's own time per shot, which no
# setting sets, and the limits, which their estimate's verdict reads.
NOT_COMPARED_COLUMNS = (
    "sim_wall_seconds_per_shot",
    "logical_error_rate_low",
    "logical_error_rate_high",
    "logical_error_rate_per_round_low",
    "logical_error_rate_per_round_high",
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


@dataclasses.dataclass(frozen=True)
class PairedComparison:
    """One point's failures in two folders on the shots both decoded.

    The field names are the columns of the csv `--out` writes. The
    shared shots are seed 0 up to where the shorter prefix stopped, and
    a pair is a shared shot both points scored. A shared shot whose
    sample_digest differs was a different draw in each folder, so a
    point with any such shot is not paired and its paired fields are
    None. The difference of rates is the first's minus the second's.
    """

    point: str
    first_point_id: str
    second_point_id: str
    shared_shots: int
    digest_mismatches: int
    unscored_shots: int
    scored_pairs: Optional[int] = None
    first_only_failures: Optional[int] = None
    second_only_failures: Optional[int] = None
    is_mixture_difference: Optional[bool] = None
    difference_low: Optional[float] = None
    difference_high: Optional[float] = None


def diff(first: pathlib.Path, second: pathlib.Path) -> tuple:
    """How two run folders differ: settings, inputs by hash, then results.

    A point is matched by its metadata, the values its sweep set, so two
    runs of different configs over one grid pair point by point, as
    sinter's plot groups by json_metadata across files (sinter
    _command/_main_plot.py --group_func). A result that differs is said
    to agree within its error bars or not: a logical error rate by its
    exact interval, a mean over shots by the standard error of its
    shots (shots.csv), which is where a decoder's measured wall clock
    moves a tick column between two runs of one yaml. A column with no
    error bar is compared exactly.

    Each shared point is also compared shot by shot (PairedComparison),
    on a line after its results, because two runs whose rates agree can
    still differ on the shots they share.

    Returns:
        The lines, and one PairedComparison per point both folders hold.
    """
    _refuse_a_folder_without_points(first)
    _refuse_a_folder_without_points(second)
    lines = _settings_lines(first, second)
    input_lines = _input_lines(first, second)
    lines.extend(input_lines)
    result_lines, paired = _result_lines(first, second)
    lines.extend(result_lines)
    return lines, paired


def main(argv: list) -> None:
    """The command line: two run folders, their differences printed."""
    parser = argparse.ArgumentParser(prog="decsim diff")
    parser.add_argument("first", help="the first run folder")
    parser.add_argument("second", help="the run folder to compare it with")
    parser.add_argument(
        "--out",
        default=None,
        help="write each shared point's paired comparison to this csv",
    )
    parsed = parser.parse_args(argv)
    first = pathlib.Path(parsed.first)
    second = pathlib.Path(parsed.second)
    lines, paired = diff(first, second)
    text = "\n".join(lines)
    print(text)
    if parsed.out is None:
        return
    rows = [dataclasses.asdict(comparison) for comparison in paired]
    out = pathlib.Path(parsed.out)
    report.write_csv(rows, out)


def _refuse_a_folder_without_points(run_dir: pathlib.Path) -> None:
    """A folder without resolved/ would read as settings that never differ."""
    if (run_dir / run_folder.RESOLVED_FOLDER).is_dir():
        return
    raise refusal.RefusalError(
        f"{run_dir} holds no resolved/ folder; name a run folder decsim "
        "collect wrote, combined/<name>-<id8>/ in its experiment folder"
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


def _result_lines(first: pathlib.Path, second: pathlib.Path) -> tuple:
    """Each point's sweep.csv columns that differ, with their verdict.

    Returns:
        The section's lines, and each shared point's PairedComparison.
    """
    first_rows = _rows_by_metadata(first)
    second_rows = _rows_by_metadata(second)
    lines = ["results:"]
    unmatched = _unmatched_lines(first_rows, second_rows)
    lines.extend(unmatched)
    paired = []
    for point in _shared(first_rows, second_rows):
        first_row = first_rows[point]
        second_row = second_rows[point]
        first_shots = _shot_rows_of(first, first_row["point_id"])
        second_shots = _shot_rows_of(second, second_row["point_id"])
        point_paired = _paired_comparison(
            point, first_row, second_row, first_shots, second_shots
        )
        paired.append(point_paired)
        comparison = _Comparison(point, first_shots, second_shots)
        point_lines = comparison.lines(first_row, second_row)
        lines.extend(point_lines)
        point_text = _point_text(point)
        paired_line = _paired_line(point_text, point_paired)
        lines.append(paired_line)
    section = _or_same(lines)
    return section, paired


class _Comparison:
    """One point's results in two folders, and the error bars of each."""

    def __init__(
        self,
        point: str,
        first_shots: list,
        second_shots: list,
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
        if column in LIMITS_OF_ESTIMATE:
            limit_columns = LIMITS_OF_ESTIMATE[column]
            return _interval_verdict(first_row, second_row, limit_columns)
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


def _paired_comparison(
    point: str,
    first_row: dict,
    second_row: dict,
    first_shots: list,
    second_shots: list,
) -> PairedComparison:
    """The point's failures in the two folders, shot by shot.

    A prefix runs from seed 0 to its stop (sweep.csv's prefix_shots),
    so the seeds both folders hold run to where the shorter stopped,
    itself a stop of the pair. The mixture test and the difference
    sequence hold at any stop (failure_statistics), so pairs cut there
    keep their 95 percent guarantee, whether a point stopped at a
    failure target or at a fixed shot count. A shots.csv cut by hand is
    read to its first missing seed, as the fold reads a prefix
    (collection.PrefixTracker).
    """
    stop = min(first_row["prefix_shots"], second_row["prefix_shots"])
    first_by_seed = _shots_by_seed(first_shots, stop)
    second_by_seed = _shots_by_seed(second_shots, stop)
    first_held = _contiguous_seed_count(first_by_seed)
    second_held = _contiguous_seed_count(second_by_seed)
    shared_shots = min(first_held, second_held)
    mismatches, unscored, first_failures, second_failures = _shared_seeds(
        first_by_seed, second_by_seed, shared_shots
    )
    unpaired = PairedComparison(
        point=point,
        first_point_id=first_row["point_id"],
        second_point_id=second_row["point_id"],
        shared_shots=shared_shots,
        digest_mismatches=mismatches,
        unscored_shots=unscored,
    )
    if mismatches:
        return unpaired
    return _with_paired_statistics(unpaired, first_failures, second_failures)


def _shots_by_seed(shot_rows: list, stop: int) -> dict:
    """The shot rows of the seeds before the stop, keyed by seed."""
    by_seed = {}
    for row in shot_rows:
        if row["seed"] < stop:
            by_seed[row["seed"]] = row
    return by_seed


def _contiguous_seed_count(by_seed: dict) -> int:
    """How many seeds from 0 the rows hold without a gap."""
    count = 0
    while count in by_seed:
        count += 1
    return count


def _shared_seeds(
    first_by_seed: dict, second_by_seed: dict, shared_shots: int
) -> tuple:
    """Digest mismatches, unscored shots, and each pair's two failures.

    A shot either folder left unscored got no correction there, so it
    has no failure bit to pair and is counted apart.
    """
    mismatches = 0
    unscored = 0
    first_failures = []
    second_failures = []
    for seed in range(shared_shots):
        first_shot = first_by_seed[seed]
        second_shot = second_by_seed[seed]
        if first_shot["sample_digest"] != second_shot["sample_digest"]:
            mismatches += 1
        if not first_shot["is_scored"] or not second_shot["is_scored"]:
            unscored += 1
            continue
        first_failures.append(first_shot["logical_failure"])
        second_failures.append(second_shot["logical_failure"])
    return mismatches, unscored, first_failures, second_failures


def _with_paired_statistics(
    unpaired: PairedComparison, first_failures: list, second_failures: list
) -> PairedComparison:
    """The discordant counts, the mixture test and the difference interval."""
    first = numpy.asarray(first_failures, dtype=bool)
    second = numpy.asarray(second_failures, dtype=bool)
    first_survived = ~first
    second_survived = ~second
    first_only = first & second_survived
    second_only = second & first_survived
    first_only_count = numpy.count_nonzero(first_only)
    second_only_count = numpy.count_nonzero(second_only)
    first_only_failures = int(first_only_count)
    second_only_failures = int(second_only_count)
    is_difference = failure_statistics.is_mixture_difference(
        first_only_failures, second_only_failures
    )
    interval = failure_statistics.difference_sequence(first, second)
    low = None
    high = None
    if interval is not None:
        low, high = interval
    return dataclasses.replace(
        unpaired,
        scored_pairs=len(first),
        first_only_failures=first_only_failures,
        second_only_failures=second_only_failures,
        is_mixture_difference=is_difference,
        difference_low=low,
        difference_high=high,
    )


def _paired_line(point_text: str, paired: PairedComparison) -> str:
    """The paired comparison in words: differ, or no difference shown."""
    if paired.digest_mismatches:
        return (
            f"  {point_text} not paired: {paired.digest_mismatches} of "
            f"{paired.shared_shots} shared shots hold a different "
            "sample_digest, so the two points did not decode the same shots"
        )
    verdict = "no difference shown"
    if paired.is_mixture_difference:
        verdict = "differ"
    interval = "none, with no pair"
    if paired.difference_low is not None:
        interval = (
            f"[{paired.difference_low:.6g}, {paired.difference_high:.6g}]"
        )
    return (
        f"  {point_text} paired on {paired.scored_pairs} of "
        f"{paired.shared_shots} shared shots ({paired.unscored_shots} "
        f"unscored in either run, left out): {paired.first_only_failures} "
        f"failed in the first only, {paired.second_only_failures} in the "
        f"second only; {verdict} by the mixture test; the difference of "
        f"rates, first minus second, is in {interval}"
    )


def _shot_column_of(column: str) -> Optional[str]:
    """The shots.csv column a sweep.csv column is the mean of, or None."""
    if column in SHOT_COLUMN_OF_MEAN:
        return SHOT_COLUMN_OF_MEAN[column]
    if column in report.SHOT_MEANS or column.endswith(LATENCY_MEAN_SUFFIX):
        return column
    return None


def _interval_verdict(
    first_row: dict, second_row: dict, limit_columns: tuple
) -> str:
    """Overlapping intervals have not been shown to differ.

    A point with no scored shot, or a cap with no failure, has no whole
    interval, and an empty or NaN limit compares false with everything,
    so without this check it would read as agreeing.
    """
    low_column, high_column = limit_columns
    first_low = first_row.get(low_column)
    first_high = first_row.get(high_column)
    second_low = second_row.get(low_column)
    second_high = second_row.get(high_column)
    limits = (first_low, first_high, second_low, second_high)
    if not all(_is_finite_number(limit) for limit in limits):
        return "no statistical comparison possible"
    first_below = first_high < second_low
    second_below = second_high < first_low
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


def _is_finite_number(value) -> bool:
    """A number other than NaN or an infinity; an empty cell is none."""
    if not _is_number(value):
        return False
    return math.isfinite(value)


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

    A folder records every point of its sweep but holds rows only of
    the points that ran a shot.
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
