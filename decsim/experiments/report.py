"""Shot measurements -> a run folder's additive facts -> the summaries.

A run folder records only facts that add up: one row per shot, one per
shot per link, and one per distinct microsecond value of each latency
point with its count. Every summary is derived from those rows at read
time, sinter's shape (sinter/_data/_task_stats.py: TaskStats.__add__
sums, rates come from the summed row), and gem5's Distribution counts
per bucket. So sweep.csv reads the same whether one process ran every
shot or many ran a piece each. The fold (experiments/fold.py) is the one
code path that produces every summary; this module says which field
plays which role.

The per-value counts file stays small because every sample is whole
ticks over the ticks in a microsecond: under a fixed-latency card a
task's windows take a handful of spans. A wall-clock decoder spreads
values wide, and those timing sweeps of hundreds of shots already write
a row per window in latency_samples.csv.
"""

import csv
import dataclasses
import functools
import json
import math
from pathlib import Path
from typing import Optional, Union

import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.collect as collect
import decsim.experiments.collection as collection
import decsim.experiments.failure_statistics as failure_statistics
import decsim.experiments.fold as fold
import decsim.experiments.measure as measure
import decsim.experiments.refusal as refusal
import decsim.experiments.run_folder as run_folder
import decsim.records.windows as window_records

# the measurement's fields that are not columns of shots.csv: the
# per-window collections, the counter tables of their own files, and the
# path of a file rather than a fact of the shot
NON_COLUMN_FIELDS = (
    "samples",
    "window_tiers",
    "means",
    "maxes",
    "link_totals",
    "data_movement",
    "window_statuses",
    "confidence",
)
# the counters a hold books for the whole shot: a reference is a token
# on a store's slot and belongs to no path, so these repeat on every row
# of one shot
REFERENCE_COUNTERS = ("references", "referenced_rounds")
# the columns that tell harder windows from an overloaded strong side, in
# sweep.csv order; a task whose shots did not keep them gets none
LOAD_MEANS = (
    "weak_syndrome_weight_mean",
    "weak_service_mean_us",
    "strong_wait_mean_us",
)
LOAD_MAXES = (
    "weak_syndrome_weight_max",
    "strong_wait_max_us",
    "strong_held_in_units_max",
    "backlog_peak_rounds",
)
# the latency point whose median and p99 sweep.csv also gives per tier:
# a window's formed-to-commit time, kept (weak) apart from escalated
# (strong), which the switching studies weigh against each other
# (Toshio et al. 2510.25222 Sec. III); every other point's tiers stay
# apart in window_samples.csv
TIER_SPLIT_POINT = "buffer0_ready_to_frame"
# which role each field of shots.csv plays in a task's row: a
# mean over the task's shots, the largest over them, a sum, or how many
# shots hold it true. The latency points' own mean and max columns are
# added to the first two per measure.POINTS.
SHOT_MEANS = (
    "decoded_windows",
    "throughput_windows_per_us",
    "throughput_rounds_per_us",
    "load",
    "sim_wall_seconds",
    "weak_busy_fraction",
    "strong_busy_fraction",
    *LOAD_MEANS,
)
SHOT_MAXES = (
    # one value per task, the code card's, so its largest is that value
    "commit_rounds",
    "window_period_us",
    "max_queued_windows",
    "weak_queue_max",
    "strong_queue_max",
    "parallel_processes_needed",
    *LOAD_MAXES,
)
# the windows by final status and the replaced provisional ones left
# uncorrected, summed per task
STATUS_SUMS = (
    *measure.WINDOW_STATUS_COLUMNS,
    "provisional_no_correction_windows",
)
SHOT_SUMS = (
    "referee_windows_checked",
    "referee_window_disagreements",
    "decoded_windows",
    "escalated_windows",
    "executed_rounds",
    "strong_decoded_rounds",
    "strong_service_sum_us",
    *STATUS_SUMS,
)
SHOT_TRUE_COUNTS = (
    "logical_failure",
    "is_scored",
)
# the files a fold reads row by row and writes back out, one header
# each: every column any piece holds, the pieces of one task alike
FOLDED_FILES = (
    "shots.csv",
    "shot_links.csv",
    "shot_data_movement.csv",
    "latency_samples.csv",
    "window_confidence.csv",
)
# the confidence histogram's bin, a tenth of a decibel: Gidney et al.
# bin gaps to the nearest whole decibel (2312.04522 main.tex:459, 505),
# and a tenth resolves the 20 dB threshold's neighbourhood
CONFIDENCE_BINS_PER_DECIBEL = 10
# the histogram's two kinds of row: every window's gap, and each shot's
# smallest window gap, the gap over many rounds that Gidney et al. take
# as the minimum of draws over fewer (main.tex:448, 513), against the
# shot's failure, Toshio's P(e|g) (2510.25222 lines 807-841)
WINDOW_HISTOGRAM = "window"
SHOT_MINIMUM_HISTOGRAM = "shot_minimum"


@dataclasses.dataclass(frozen=True)
class RunRecord:
    """The additive facts one or more run folders hold, one list per file.

    No field holds a summary, so two folders' records join by concatenation
    and the summaries of the join are those of one run over the same shots.
    """

    shots: list
    shot_links: list
    shot_data_movement: list
    window_samples: list
    latency_samples: list
    window_confidence: list
    confidence_histogram: list


def percentile_of_counts(multiset: dict, fraction: float) -> float:
    """The value at fraction of the samples, numpy's nearest method.

    The index is round((n - 1) * fraction) into the sorted samples,
    numpy.percentile(method="nearest"), half way rounding to the even index;
    not the classical nearest rank ceil(n * fraction). multiset maps a value
    to its count, so a percentile is the same however the shots were split.
    """
    total = _total_count(multiset)
    if total == 0:
        return 0.0
    last_index = total - 1
    position = fraction * last_index
    rounded = round(position)
    index = min(last_index, int(rounded))
    return _value_at_index(multiset, index)


def task_key_of(row: dict) -> tuple:
    """The task a row belongs to: its id and its algorithm.

    The id is sinter's strong_id. A row read back off csv holds text and a
    measured row numbers, so the algorithm is read as the value it was
    written from, and both name the same task.
    """
    return (row["task_id"], fold.number_of(row["algorithm"]))


def measured_task_key(measurement: measure.ShotMeasurement) -> tuple:
    """The task one measured shot belongs to."""
    return (measurement.task_id, measurement.algorithm)


def task_columns(task_key: tuple) -> dict:
    """The two columns that name a task."""
    task_id, algorithm = task_key
    return {"task_id": task_id, "algorithm": algorithm}


def summarize_task(
    task_key: tuple,
    totals: fold.RowTotals,
    counts: dict,
    prefix: collection.PrefixTracker,
) -> dict:
    """One task: means over seeds of per-shot means, max of maxes.

    The counts cover every shot the task holds; the estimate and its
    limits cover its contiguous prefix up to its stop (prefix), which is
    what the stopping rule's intervals are exact for.
    """
    failures = totals.true_counts["logical_failure"]
    shot_count = totals.rows
    scored_shots = totals.true_counts["is_scored"]
    unscored_shots = shot_count - scored_shots
    row = task_columns(task_key)
    row["shots"] = shot_count
    row["windows_per_shot"] = totals.mean("decoded_windows")
    row["logical_failures"] = failures
    row["scored_shots"] = scored_shots
    row["unscored_shots"] = unscored_shots
    _add_estimate_columns(row, prefix)
    # a bound on this sample's rate, not a confidence bound
    row["logical_error_rate_unscored_as_failures"] = (
        failure_statistics.estimate_counting_unscored(
            failures, scored_shots, unscored_shots
        )
    )
    _add_status_columns(row, totals)
    row["throughput_windows_per_us"] = totals.mean("throughput_windows_per_us")
    row["throughput_rounds_per_us"] = totals.mean("throughput_rounds_per_us")
    row["max_queued_windows"] = totals.maxes["max_queued_windows"]
    row["referee_windows_checked"] = totals.sums["referee_windows_checked"]
    row["referee_window_disagreements"] = totals.sums[
        "referee_window_disagreements"
    ]
    row["load"] = totals.mean("load")
    row["sim_wall_seconds_per_shot"] = totals.mean("sim_wall_seconds")
    _add_pool_columns(row, totals)
    _add_load_columns(row, totals)
    for name in _points_held(totals.means):
        multiset = _multiset_over_tiers(counts, task_key, name)
        _add_latency_point_columns(row, totals, name, multiset)
    _add_tier_split_columns(row, counts, task_key)
    return row


def summary_rows(totals: dict, counts: dict, prefixes: dict) -> list:
    """One row per task whose shots were totalled, in task order.

    The totals hold the tasks in the order their first shots came,
    which is the sweep's task order for one run and for a fold alike
    (a fold merges the pieces' rows in task order).
    """
    rows = []
    for task_key in totals:
        task_totals = totals[task_key]
        task_id, _algorithm = task_key
        prefix = prefixes[task_id]
        row = summarize_task(task_key, task_totals, counts, prefix)
        rows.append(row)
    return rows


def write_csv(rows: list, path: Path, swept: Optional[dict] = None) -> None:
    """The rows as a csv file, every column any row holds, first seen first.

    A task's swept values follow its task_id: the design's fixed values
    first, then the measured ones, Wickham's order (Tidy Data, J. Stat.
    Softw. 59(10), 2014, section 2.3).
    """
    if swept is not None:
        rows = _with_swept_values(rows, swept)
    columns = {}
    for row in rows:
        row_columns = dict.fromkeys(row)
        columns.update(row_columns)
    field_names = list(columns)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def shot_data_movement_rows(measurements: list) -> list:
    """One row per shot per path: that shot's own copy and move counters.

    A run with observation.data_movement off writes no row: counting
    nothing is not counting zero.
    """
    rows = []
    for measurement in measurements:
        counted = measurement.data_movement
        if counted is None:
            continue
        task_key = measured_task_key(measurement)
        for path in _paths_counted(counted):
            row = _shot_movement_row(task_key, measurement, counted, path)
            rows.append(row)
    return rows


def shot_rows(measurements: list) -> list:
    """One row per shot: its scalars and window statuses, then each point.

    Any aggregate can then be re-cut without rerunning the sweep, and a
    point's mean and max columns fold over any set of shots.
    """
    rows = []
    for measurement in measurements:
        row = _scalar_fields(measurement)
        row.update(measurement.window_statuses)
        for name in measure.POINTS:
            row[f"{name}_mean_us"] = measurement.means[name]
        for name in measure.POINTS:
            row[f"{name}_max_us"] = measurement.maxes[name]
        rows.append(row)
    return rows


def shot_link_rows(measurements: list) -> list:
    """One row per shot per link: that shot's own ledger counters.

    They are a file of their own rather than sixty more columns of
    shots.csv because the link set is data and not schema: a topology
    with another link adds rows and moves no column.
    """
    rows = []
    for measurement in measurements:
        task_key = measured_task_key(measurement)
        for path in sorted(measurement.link_totals):
            row = _shot_link_row(task_key, measurement, path)
            rows.append(row)
    return rows


def window_sample_rows(measurements: list) -> list:
    """One row per task, latency point, tier and distinct value: its count.

    That multiset is all a median or p99 needs. Kept and escalated windows
    are two multisets; a round's sample names no tier.
    """
    counts = _counts_of_samples(measurements)
    task_keys = []
    for task_key, _name, _tier in counts:
        task_keys.append(task_key)
    unique_task_keys = dict.fromkeys(task_keys)
    return _rows_of_counts(counts, list(unique_task_keys))


def latency_sample_rows(measurements: list) -> list:
    """One row per decoded window: the algorithm stage's time.

    Only algorithms named by a table row produce rows, the measured wall
    clock or the row's cycle count; a fixed latency produces none. Each row
    carries its tier and the window's inter-arrival, the deadline a decode
    must beat, so a latency figure is drawn from run folders alone.
    """
    rows = []
    for measurement in measurements:
        if not isinstance(measurement.algorithm, str):
            continue
        task_key = measured_task_key(measurement)
        samples = measurement.samples["algorithm"]
        tiers = measurement.window_tiers
        for sample_us, tier in zip(samples, tiers, strict=True):
            row = task_columns(task_key)
            row["seed"] = measurement.seed
            row["tier"] = tier
            row["algorithm_us"] = sample_us
            row["window_period_us"] = measurement.window_period_us
            rows.append(row)
    return rows


def window_confidence_rows(measurements: list) -> list:
    """One row per window of each sampled shot: its gap and its verdict.

    Toshio et al. keep each shot's gap with whether the decode was right
    (2510.25222 lines 722-731); a window has no truth of its own, so a row
    carries the shot's failure and whether the strong decode revised it.
    """
    rows = []
    for measurement in measurements:
        if not _has_counted_confidence(measurement):
            continue
        confidence = measurement.confidence
        if not confidence.is_sampled:
            continue
        for window in confidence.windows:
            row = _window_confidence_row(measurement, confidence, window)
            rows.append(row)
    return rows


def confidence_histogram_rows(measurements: list) -> list:
    """Every scored shot's window gaps and smallest gap, per 0.1 dB bin.

    The counts add across pieces as sinter's custom_counts do
    (sinter/_data/_task_stats.py:51-71). Toshio's p(g), P(e|g) and the
    brute-force cutoff come from it (2510.25222 lines 807-841, 863-900).
    """
    counts = {}
    for measurement in measurements:
        _count_the_shots_gaps(counts, measurement)
    task_keys = _task_keys_of_counts(counts)
    return _confidence_histogram_rows_of(counts, task_keys)


def gap_bin_low_decibels(gap_nats: float) -> float:
    """The lower edge, in decibels, of the 0.1 dB bin holding the gap.

    Toshio et al. histogram gaps in decibels (2510.25222 main.tex:564). An
    infinite gap keeps its own bin.
    """
    decibels = threshold_sources.nats_to_decibels(gap_nats)
    if math.isinf(decibels):
        return decibels
    scaled = decibels * CONFIDENCE_BINS_PER_DECIBEL
    tenths = math.floor(scaled)
    return tenths / CONFIDENCE_BINS_PER_DECIBEL


def confidence_shot_count_of(measurements: list) -> Optional[Union[int, str]]:
    """The shots a piece's confidence rows cover, for its piece.json.

    The first shots' count, all, or None when no confidence signal ran.
    """
    for measurement in measurements:
        confidence = measurement.confidence
        if confidence is None:
            continue
        if confidence.sampled_shot_count is None:
            return collect.EVERY_SHOT
        return confidence.sampled_shot_count
    return None


def record_of(measurements: list) -> RunRecord:
    """The additive facts of the shots this process measured."""
    shots = shot_rows(measurements)
    links = shot_link_rows(measurements)
    movement = shot_data_movement_rows(measurements)
    samples = window_sample_rows(measurements)
    latency = latency_sample_rows(measurements)
    confidence = window_confidence_rows(measurements)
    histogram = confidence_histogram_rows(measurements)
    return RunRecord(
        shots, links, movement, samples, latency, confidence, histogram
    )


def write_record(record: RunRecord, report_dir: Path, swept: dict) -> None:
    """The record's seven files; one with no rows is not written."""
    shots_path = report_dir / "shots.csv"
    _write_rows(record.shots, shots_path, swept)
    links_path = report_dir / "shot_links.csv"
    _write_rows(record.shot_links, links_path, swept)
    movement_path = report_dir / "shot_data_movement.csv"
    _write_rows(record.shot_data_movement, movement_path, swept)
    samples_path = report_dir / "window_samples.csv"
    _write_rows(record.window_samples, samples_path, swept)
    latency_path = report_dir / "latency_samples.csv"
    _write_rows(record.latency_samples, latency_path, swept)
    confidence_path = report_dir / "window_confidence.csv"
    _write_rows(record.window_confidence, confidence_path, swept)
    histogram_path = report_dir / "confidence_histogram.csv"
    _write_rows(record.confidence_histogram, histogram_path, swept)


def read_rows(path: Path) -> list:
    """One csv file's rows, each value back as the number it was written."""
    rows = []
    for row in fold.row_stream(path):
        typed = fold.typed_row(row)
        rows.append(typed)
    return rows


def fold_pieces(
    run_dir: Path,
    folders: list,
    task_ids: list,
    out_dir: Path,
    rules: Optional[dict] = None,
) -> list:
    """Pieces' additive files folded into out_dir; the sweep rows.

    A piece holds the files for a range of one task's seeds; rows come back
    in the order one run over every piece would write them. Two pieces may
    not share a shot. 500 pieces of a million-shot sweep are 115 million
    link rows, so pieces are streamed one row at a time and only fold.py's
    totals stand between reading and writing.
    """
    order = _refused_or_ordered(folders, task_ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    swept = run_folder.swept_values(run_dir, task_ids)
    return _fold_the_folders(folders, order, out_dir, swept, rules)


def strong_service_bound_us(totals: fold.RowTotals) -> float:
    """Toshio's Theorem 1 bound on one strong decode's time, in us.

    Eq. (6): tau_strong <= (1 / gamma_switch)(d / r_strong) tau_gen per round
    (2510.25222 lines 1272-1304), gamma_switch the switching rate per d
    rounds (lines 185-188). A decode reads r_strong rounds, so one decode is
    bounded by d tau_gen / gamma_switch. The proof counts escalations against
    rounds generated (lines 1335-1350), so the bound is tau_gen times the
    generated rounds over the escalated windows. The rounds are the executed
    ones, since a double window absorbs windows whose rounds were still
    generated. Infinite when nothing escalated.
    """
    escalated_windows = totals.sums["escalated_windows"]
    if escalated_windows == 0:
        return math.inf
    generated_rounds = totals.sums["executed_rounds"]
    window_period_us = totals.maxes["window_period_us"]
    commit_rounds = totals.maxes["commit_rounds"]
    round_period_us = window_period_us / commit_rounds
    generated_us = round_period_us * generated_rounds
    return generated_us / escalated_windows


def _refused_or_ordered(folders: list, task_ids: list):
    """The fold's row order, once the folders pass every refusal.

    The refusals run before anything is written, so a refused fold
    leaves no file behind.
    """
    positions = _task_positions(task_ids)
    order = functools.partial(_row_task_and_seed, positions)
    _refuse_folders_of_different_columns(folders)
    _refuse_pieces_that_recorded_confidence_apart(folders)
    _refuse_a_repeated_shot(folders, order)
    return order


def _fold_the_folders(
    folders: list, order, out_dir: Path, swept: dict, rules
) -> list:
    """Every folder's additive files into one folder's, a pass per file.

    A pass reads one file of every folder at once, writes the folded
    file row by row and feeds the totals its summary needs, so no pass
    holds a folder's rows. The derived files come off the totals the
    passes built, each task's swept values beside its rows.
    """
    shot_totals, prefixes = _fold_shots(folders, order, out_dir, swept, rules)
    counts = _folded_counts(folders)
    rows = summary_rows(shot_totals, counts, prefixes)
    sweep_path = out_dir / "sweep.csv"
    write_csv(rows, sweep_path, swept)
    samples_path = out_dir / "window_samples.csv"
    samples_rows = _rows_of_counts(counts, list(shot_totals))
    _write_rows(samples_rows, samples_path, swept)
    _fold_shot_links(folders, order, out_dir, swept)
    _fold_shot_movement(folders, order, out_dir, swept)
    _fold_latency_samples(folders, order, out_dir, swept)
    _fold_window_confidence(folders, order, out_dir, swept)
    histogram = _folded_confidence_histogram(folders)
    histogram_rows = _confidence_histogram_rows_of(histogram, list(shot_totals))
    histogram_path = out_dir / "confidence_histogram.csv"
    _write_rows(histogram_rows, histogram_path, swept)
    return rows


def _fold_shots(
    folders: list, order, out_dir: Path, swept: dict, rules
) -> tuple:
    """Every folder's shots.csv into one; each task's totals and prefix."""
    totals = {}
    prefixes = {}
    add_a_shot = functools.partial(_add_a_folded_shot, totals, prefixes, rules)
    _fold_one_file(folders, "shots.csv", order, out_dir, swept, add_a_shot)
    return totals, prefixes


def _add_a_folded_shot(
    totals: dict, prefixes: dict, rules: Optional[dict], row: dict
) -> None:
    """One folded shot row into its task's totals and its prefix."""
    _add_a_shot(totals, row)
    _add_to_the_prefix(prefixes, rules, row)


def _fold_shot_links(folders: list, order, out_dir: Path, swept: dict) -> None:
    """Every folder's shot_links.csv into one; no summary reads it."""
    name = "shot_links.csv"
    _fold_one_file(folders, name, order, out_dir, swept, _no_totals)


def _fold_shot_movement(
    folders: list, order, out_dir: Path, swept: dict
) -> None:
    """Every folder's shot_data_movement.csv into one; no summary reads it."""
    name = "shot_data_movement.csv"
    _fold_one_file(folders, name, order, out_dir, swept, _no_totals)


def _fold_latency_samples(
    folders: list, order, out_dir: Path, swept: dict
) -> None:
    """Every folder's latency_samples.csv into one; no summary reads it."""
    name = "latency_samples.csv"
    _fold_one_file(folders, name, order, out_dir, swept, _no_totals)


def _fold_window_confidence(
    folders: list, order, out_dir: Path, swept: dict
) -> None:
    """Every folder's window_confidence.csv into one; no summary reads it."""
    name = "window_confidence.csv"
    _fold_one_file(folders, name, order, out_dir, swept, _no_totals)


def _fold_one_file(
    folders: list, name: str, order, out_dir: Path, swept: dict, add_a_row
) -> None:
    """One additive file of every folder merged into out_dir's, row by row.

    A piece holds bare rows, so each row takes its task's swept values
    after its task_id here, as write_csv places them.
    """
    paths = _folder_files(folders, name)
    out_path = out_dir / name
    field_names = _folded_columns(paths, swept)
    with fold.RowFile(out_path, field_names) as out_file:
        for row in fold.merged_rows(paths, order):
            placed = _with_swept_row(row, swept)
            out_file.write(placed)
            add_a_row(row)


def _folded_columns(paths: list, swept: dict) -> list:
    """Every column a folded file's rows hold, first seen first.

    Tasks of one grid can measure different columns; a cell a task did not
    measure is empty.
    """
    columns = {"task_id": None}
    for values in swept.values():
        swept_columns = dict.fromkeys(values)
        columns.update(swept_columns)
    for path in paths:
        if not path.is_file():
            continue
        header = fold.header_of(path)
        file_columns = dict.fromkeys(header)
        columns.update(file_columns)
    return list(columns)


def _no_totals(_row: dict) -> None:
    """A file whose rows no summary totals."""


def _folded_counts(folders: list) -> dict:
    """Every folder's per-value counts added into one set of multisets.

    This is the one additive file a fold does hold, and it is the small
    one: its length follows the spread of a point's sample values and
    not the shot count, so an experiment's counts are a few thousand
    rows.
    """
    counts = {}
    for run_dir in folders:
        path = Path(run_dir) / "window_samples.csv"
        if not path.is_file():
            continue
        rows = read_rows(path)
        _add_counts_of_rows(counts, rows)
    return counts


def _folder_files(folders: list, name: str) -> list:
    """One file of every folder, whether or not the folder wrote it."""
    paths = []
    for run_dir in folders:
        path = Path(run_dir) / name
        paths.append(path)
    return paths


def _add_to_the_prefix(
    prefixes: dict, rules: Optional[dict], row: dict
) -> None:
    """One shot row onto its task's prefix, its tracker made at first."""
    task_id = row["task_id"]
    tracker = prefixes.get(task_id)
    if tracker is None:
        rule = _rule_of(rules, task_id)
        tracker = collection.PrefixTracker(rule)
        prefixes[task_id] = tracker
    tracker.add(row)


def _rule_of(rules: Optional[dict], task_id: str) -> collection.TaskRule:
    """The task's rule, or a shot count fixed in advance when none."""
    if rules is None or task_id not in rules:
        return collection.TaskRule()
    return rules[task_id]


def _add_estimate_columns(row: dict, prefix: collection.PrefixTracker) -> None:
    """The prefix's state, counts, estimate and exact limits.

    failures over scored shots is sinter's errors over shots less
    discards (sinter/_plotting.py:389). An adaptive task's shots are
    not independent draws, so it shows its counts and no estimate.
    """
    counts = prefix.counts
    row["state"] = prefix.state()
    estimate = _prefix_estimate(prefix)
    row["logical_error_rate_estimate"] = estimate.rate
    row["logical_error_rate_low"] = estimate.low
    row["logical_error_rate_high"] = estimate.high
    row["logical_error_rate_plan_unbiased"] = _plan_unbiased(prefix)
    shape = prefix.round_shape()
    row["logical_error_rate_per_round"] = _per_round(estimate.rate, shape)
    row["logical_error_rate_per_round_low"] = _per_round(estimate.low, shape)
    row["logical_error_rate_per_round_high"] = _per_round(estimate.high, shape)
    row["is_shot_rate_above_half"] = _is_above_half(estimate.rate)
    row["prefix_shots"] = counts.shots
    row["prefix_scored_shots"] = counts.scored_shots
    row["prefix_failures"] = counts.failures


def _prefix_estimate(
    prefix: collection.PrefixTracker,
) -> failure_statistics.Estimate:
    """The prefix's estimate and limits; none for an adaptive task."""
    if prefix.rule.is_adaptive:
        return failure_statistics.Estimate(None, None, None)
    counts = prefix.counts
    stop_kind = prefix.stop_kind_for_limits()
    return failure_statistics.estimate(
        counts.failures, counts.scored_shots, stop_kind
    )


def _plan_unbiased(prefix: collection.PrefixTracker) -> Optional[float]:
    """GMS's estimate, for a plan of counts fixed in advance that stopped.

    It is unbiased over every outcome of the plan, which a time cap
    makes depend on the host, and which a prefix still running has not
    reached (failure_statistics.plan_unbiased_estimate).
    """
    if prefix.rule.is_adaptive:
        return None
    settings = prefix.rule.settings
    if settings is not None and settings.max_core_seconds is not None:
        return None
    if settings is not None and prefix.stop_kind is None:
        return None
    counts = prefix.counts
    stop_kind = prefix.stop_kind_for_limits()
    return failure_statistics.plan_unbiased_estimate(
        counts.failures, counts.scored_shots, stop_kind
    )


def _per_round(shot_rate: Optional[float], round_shape: Optional[tuple]):
    """A shot's rate as one output's for one round, when both are known."""
    if shot_rate is None or round_shape is None:
        return None
    outputs, rounds = round_shape
    return failure_statistics.per_output_round_rate(shot_rate, outputs, rounds)


def _is_above_half(shot_rate: Optional[float]) -> Optional[bool]:
    """Whether the shot rate is past one half.

    With one output, there the per-round map takes its complement
    (failure_statistics.per_round_rate), and for an even round count no
    round-flip probability gives the rate at all.
    """
    if shot_rate is None:
        return None
    return shot_rate > 0.5


def _add_status_columns(row: dict, totals) -> None:
    """Windows by final status, and replaced provisional ones uncorrected."""
    for name in STATUS_SUMS:
        row[name] = totals.sums[name]


def _add_pool_columns(row: dict, totals) -> None:
    """The pool columns and Toshio's bound."""
    row["weak_queue_max"] = totals.maxes["weak_queue_max"]
    row["strong_queue_max"] = totals.maxes["strong_queue_max"]
    row["weak_busy_fraction"] = totals.mean("weak_busy_fraction")
    row["strong_busy_fraction"] = totals.mean("strong_busy_fraction")
    row["escalated_windows"] = totals.sums["escalated_windows"]
    if "strong_service_sum_us" in totals.sums:
        row["strong_service_mean_us"] = _strong_service_mean_us(totals)
    row["strong_service_bound_us"] = strong_service_bound_us(totals)
    row["parallel_processes_needed"] = totals.maxes["parallel_processes_needed"]


def _add_load_columns(row: dict, totals) -> None:
    """The difficulty and overload columns the task's shots hold.

    The escalated share rides with them: the windows the strong tier
    committed over the windows decoded.
    """
    for name in LOAD_MEANS:
        if name in totals.means:
            row[name] = totals.mean(name)
    for name in LOAD_MAXES:
        if name in totals.maxes:
            row[name] = totals.maxes[name]
    if "strong_wait_max_us" not in totals.maxes:
        return
    windows = totals.sums["decoded_windows"]
    escalated = totals.sums["escalated_windows"]
    row["escalated_fraction"] = escalated / windows


def _strong_service_mean_us(totals: fold.RowTotals) -> Optional[float]:
    """The task's strong service over its strong decodes; None for none.

    A ratio of two sums, as gem5's avgMissLatency = missLatency / misses
    (src/mem/cache/base.cc:2187-2188) and sinter divide at read time; a mean
    of shot means would weigh a one-decode shot like a ten-decode one.
    """
    escalated_windows = totals.sums["escalated_windows"]
    if escalated_windows == 0:
        return None
    return totals.sums["strong_service_sum_us"] / escalated_windows


def _shot_totals(row: dict) -> fold.RowTotals:
    """What one task's shot rows add up to, role by role.

    A role's column is totalled only when the row holds it, the rule of
    _points_held, so a folder an older tree wrote still folds.
    """
    means = _held_by(row, SHOT_MEANS)
    maxes = _held_by(row, SHOT_MAXES)
    sums = _held_by(row, SHOT_SUMS)
    for name in _points_held(row):
        means.append(f"{name}_mean_us")
        maxes.append(f"{name}_max_us")
    return fold.RowTotals(
        means=means,
        maxes=maxes,
        sums=sums,
        true_counts=SHOT_TRUE_COUNTS,
    )


def _held_by(row: dict, columns: tuple) -> list:
    """The columns of one role the row holds, in the role's order."""
    held = []
    for column in columns:
        if column in row:
            held.append(column)
    return held


def _points_held(fields) -> list:
    """The latency points a shot row or its totals hold, in POINTS order.

    A folder is read by the columns it holds, not those the reading tree
    writes, so a fold reads a folder an older tree wrote, as sinter folds
    custom_counts by adding Counters (sinter/_data/_task_stats.py:71,
    117-150). A point a folder did not measure gets no column, since zeros
    would say its windows took no time.
    """
    held = []
    for name in measure.POINTS:
        column = f"{name}_mean_us"
        if column in fields:
            held.append(name)
    return held


def _add_a_shot(totals: dict, row: dict) -> None:
    """One shot row into its own task's totals."""
    task_key = task_key_of(row)
    task_totals = totals.get(task_key)
    if task_totals is None:
        task_totals = _shot_totals(row)
        totals[task_key] = task_totals
    task_totals.add(row)


def _write_rows(rows: list, path: Path, swept: dict) -> None:
    """One file of the record, left unwritten when it has no rows."""
    if not rows:
        return
    write_csv(rows, path, swept)


def _with_swept_values(rows: list, swept: dict) -> list:
    """Each row with its task's swept values right after its task_id."""
    placed = []
    for row in rows:
        with_values = _with_swept_row(row, swept)
        placed.append(with_values)
    return placed


def _with_swept_row(row: dict, swept: dict) -> dict:
    """One row with its task's swept values right after its task_id."""
    task_id = row["task_id"]
    with_values = {"task_id": task_id}
    with_values.update(swept[task_id])
    with_values.update(row)
    return with_values


def _total_count(multiset: dict) -> int:
    """How many samples the multiset holds."""
    total = 0
    for count in multiset.values():
        total += count
    return total


def _value_at_index(multiset: dict, index: int) -> float:
    """The value at that place in the multiset's sorted samples."""
    ordered = sorted(multiset)
    at_index = ordered[-1]
    seen = 0
    for value in ordered:
        seen += multiset[value]
        if index < seen:
            at_index = value
            break
    return at_index


def _counts_of_samples(measurements: list) -> dict:
    """(task, latency point, tier) -> value -> how many carried it."""
    counts = {}
    for measurement in measurements:
        task_key = measured_task_key(measurement)
        for name, values in measurement.samples.items():
            tiers = _tiers_of_samples(measurement, name)
            _count_the_values(counts, (task_key, name), tiers, values)
    return counts


def _tiers_of_samples(measurement: measure.ShotMeasurement, name: str) -> tuple:
    """The tier each sample of a latency point sits under."""
    if name in measure.ROUND_POINTS:
        values = measurement.samples[name]
        return ("",) * len(values)
    return measurement.window_tiers


def _count_the_values(
    counts: dict, key: tuple, tiers: tuple, values: list
) -> None:
    """Add one shot's samples of one latency point, each under its tier."""
    for tier, value in zip(tiers, values, strict=True):
        tier_key = (*key, tier)
        multiset = counts.setdefault(tier_key, {})
        already = multiset.get(value, 0)
        multiset[value] = already + 1


def _multiset_over_tiers(counts: dict, task_key: tuple, name: str) -> dict:
    """One latency point's multiset at one task, every tier's added."""
    merged = {}
    for (at_task_key, at_name, _tier), multiset in counts.items():
        if (at_task_key, at_name) != (task_key, name):
            continue
        for value, count in multiset.items():
            already = merged.get(value, 0)
            merged[value] = already + count
    return merged


def _add_counts_of_rows(counts: dict, window_samples: list) -> None:
    """One file's rows added to the multisets their tasks own.

    The counts add, so the multisets of several folders' files are the
    multisets one run over the same shots would have written.
    """
    for row in window_samples:
        task_key = task_key_of(row)
        key = (task_key, row["name"], row["tier"])
        at_this_name = counts.setdefault(key, {})
        value = row["value_us"]
        already = at_this_name.get(value, 0)
        at_this_name[value] = already + row["count"]


def _rows_of_counts(counts: dict, task_keys: list) -> list:
    """The multisets as rows: tasks' order, then POINTS order, value."""
    positions = _task_positions(task_keys)
    order = functools.partial(_task_and_name_order, positions)
    rows = []
    for key in sorted(counts, key=order):
        multiset = counts[key]
        for row in _rows_of_one_multiset(key, multiset):
            rows.append(row)
    return rows


def _rows_of_one_multiset(key: tuple, multiset: dict) -> list:
    """One latency point's counts at one task, by rising value."""
    task_key, name, tier = key
    columns = task_columns(task_key)
    rows = []
    for value in sorted(multiset):
        row = dict(columns)
        row["name"] = name
        row["tier"] = tier
        row["value_us"] = value
        row["count"] = multiset[value]
        rows.append(row)
    return rows


def _task_and_name_order(positions: dict, key: tuple) -> tuple:
    """A (task, latency point, tier) key where a single run writes it."""
    task_key, name, tier = key
    place = measure.POINTS.index(name)
    task_order = positions[task_key]
    return (task_order, place, tier)


def _task_positions(task_keys: list) -> dict:
    """Each task's place in the list, the order a single run meets them."""
    positions = {}
    for position, task_key in enumerate(task_keys):
        positions[task_key] = position
    return positions


def _refuse_folders_of_different_columns(run_dirs: list) -> None:
    """The pieces of one task record the same columns in a folded file."""
    for name in FOLDED_FILES:
        _refuse_one_files_different_columns(run_dirs, name)


def _refuse_one_files_different_columns(run_dirs: list, name: str) -> None:
    """One folded file's columns, the same in every piece of one task.

    Pieces from two trees differ when a column was added between them, and
    folding them would leave it silently empty for the older shots.
    """
    first_by_task = {}
    for run_dir in run_dirs:
        path = Path(run_dir) / name
        task_id = _task_of_file(path)
        if task_id is None:
            continue
        columns = fold.header_of(path)
        first = first_by_task.setdefault(task_id, (run_dir, columns))
        first_dir, first_columns = first
        if columns == first_columns:
            continue
        _refuse_the_columns(run_dir, first_dir, name, columns, first_columns)


def _task_of_file(path: Path) -> Optional[str]:
    """The task a piece's file holds rows of, or None for no row."""
    if not path.is_file():
        return None
    rows = fold.row_stream(path)
    first_row = next(rows, None)
    rows.close()
    if first_row is None:
        return None
    return first_row["task_id"]


def _refuse_the_columns(
    run_dir, first_dir, name: str, columns: list, first_columns: list
) -> None:
    """Say which folder lacks the other's columns, which they are, and how.

    The folder the sentence is about is the one that lacks columns, and
    that is not always the folder the walk reached second: a folder
    that only added columns lacks none, and the folder it was compared
    against is then the subject.
    """
    held = set(columns)
    first_held = set(first_columns)
    absent = first_held - held
    added = held - first_held
    missing = sorted(absent)
    extra = sorted(added)
    lacking = run_dir
    compared = first_dir
    if not missing:
        lacking = first_dir
        compared = run_dir
        missing = extra
        extra = []
    raise refusal.RefusalError(
        f"{lacking} does not hold the columns {compared} holds in {name}: "
        f"missing {missing}, extra {extra}; the pieces of one task record "
        "the same columns, and two trees' pieces differ when a column was "
        "added between them"
    )


def _refuse_a_repeated_shot(run_dirs: list, order) -> None:
    """One seeded run in two folders would be counted twice.

    Rows arrive merged in seed order per task, so the check keeps each
    task's last seed. It runs before anything is written.
    """
    paths = _folder_files(run_dirs, "shots.csv")
    seen = {}
    for row in fold.merged_rows(paths, order):
        task_key = task_key_of(row)
        seed = row["seed"]
        if seen.get(task_key) == seed:
            _refuse_the_folders(row, run_dirs)
        seen[task_key] = seed


def _refuse_pieces_that_recorded_confidence_apart(run_dirs: list) -> None:
    """A task's pieces must have recorded confidence for the same shots.

    Otherwise the confidence files would count fewer shots than the sweep
    row. A piece.json without the line reads as None.
    """
    coverage_by_task = _pieces_by_task_and_value(
        run_dirs, _confidence_coverage_of
    )
    for task_id, by_coverage in coverage_by_task.items():
        if len(by_coverage) > 1:
            _refuse_the_confidence_coverage(task_id, by_coverage)


def _pieces_by_task_and_value(run_dirs: list, value_of) -> dict:
    """Each task's first piece folder for every value value_of reads."""
    folders_by_task = {}
    for run_dir in run_dirs:
        piece = _piece_of(run_dir)
        value = value_of(piece)
        by_value = folders_by_task.setdefault(piece["task_id"], {})
        by_value.setdefault(value, run_dir)
    return folders_by_task


def _confidence_coverage_of(piece: dict):
    """The shots a piece recorded confidence for; None for an older piece."""
    return piece.get("confidence_shot_count")


def _piece_of(run_dir) -> dict:
    """One piece folder's piece.json."""
    piece_path = Path(run_dir) / "piece.json"
    piece_text = piece_path.read_text()
    return json.loads(piece_text)


def _refuse_the_confidence_coverage(task_id: str, by_coverage: dict):
    """Say which pieces of the task recorded confidence for which shots."""
    pieces_named = []
    for coverage, run_dir in by_coverage.items():
        pieces_named.append(f"{run_dir} ({coverage})")
    listed = ", ".join(pieces_named)
    raise refusal.RefusalError(
        f"the pieces of task {task_id} recorded confidence for different "
        f"shots (confidence_shot_count): {listed}; folding them would "
        "leave the confidence files short of the shots shots.csv counts, "
        "so collect the task again into a new results folder"
    )


def _refuse_the_folders(row: dict, run_dirs: list) -> None:
    """Say which shot is doubled and in which folders it was found."""
    listed = _named(run_dirs)
    shot = f"{row['task_id']} seed {row['seed']}"
    raise refusal.RefusalError(
        f"the shot {shot} is in more than one of {listed}; a shot is one "
        "seeded run of one task, so folding both folders would "
        "count it twice"
    )


def _named(run_dirs: list) -> str:
    """The folders as one comma-separated list, for a refusal."""
    names = []
    for run_dir in run_dirs:
        names.append(str(run_dir))
    return ", ".join(names)


def _row_task_and_seed(positions: dict, row: dict) -> tuple:
    """One per-shot row's place: its task's position, then its seed.

    It is the order a single run wrote its rows in, so the folders'
    rows merge back into that order and a task's several rows for one
    seed (one per decoded window, or one per link) keep the order the
    run wrote them.
    """
    position = positions[row["task_id"]]
    seed = fold.number_of(row["seed"])
    return (position, seed)


def _scalar_fields(measurement) -> dict:
    """The measurement's own fields, the per-window collections left out.

    A field the shot did not measure holds None and is no column, the
    rule _points_held keeps for a latency point.
    """
    row = {}
    for name in measurement.__dataclass_fields__:
        value = getattr(measurement, name)
        if name in NON_COLUMN_FIELDS or value is None:
            continue
        row[name] = value
    return row


def _shot_link_row(task_key: tuple, measurement, path: str) -> dict:
    """One shot's ledger counters on one link."""
    row = task_columns(task_key)
    row["seed"] = measurement.seed
    row["link"] = path
    counters = measurement.link_totals[path]
    for counter, value in counters.items():
        row[counter] = value
    return row


def _add_latency_point_columns(
    row: dict, totals, name: str, multiset: dict
) -> None:
    """One latency point's mean, median, p99 and max columns.

    The mean averages the per-shot means and the max takes the largest
    per-shot max, both off the per-shot rows; the median and p99 come
    from the multiset of every decoded window of the task.
    """
    row[f"{name}_mean_us"] = totals.mean(f"{name}_mean_us")
    row[f"{name}_median_us"] = percentile_of_counts(multiset, 0.50)
    row[f"{name}_p99_us"] = percentile_of_counts(multiset, 0.99)
    row[f"{name}_max_us"] = totals.maxes[f"{name}_max_us"]


def _add_tier_split_columns(row: dict, counts: dict, task_key: tuple) -> None:
    """TIER_SPLIT_POINT's median and p99 over each tier's own windows.

    A tier that committed no window at the task gets no column, since
    a percentile of no sample is no number.
    """
    for tier in window_records.DecoderTier:
        key = (task_key, TIER_SPLIT_POINT, tier.value)
        multiset = counts.get(key)
        if not multiset:
            continue
        prefix = f"{TIER_SPLIT_POINT}_{tier.value}"
        row[f"{prefix}_median_us"] = percentile_of_counts(multiset, 0.50)
        row[f"{prefix}_p99_us"] = percentile_of_counts(multiset, 0.99)


def _paths_counted(counted: dict) -> list:
    """Every path one shot copied or moved along, in path order."""
    paths = set(counted["copies_by_path"])
    for path in counted["moves_by_path"]:
        paths.add(path)
    return sorted(paths)


def _shot_movement_row(
    task_key: tuple, measurement, counted: dict, path: str
) -> dict:
    """One shot's copies and moves on one path, plus its own references."""
    row = task_columns(task_key)
    row["seed"] = measurement.seed
    row["path"] = path
    copied = counted["copies_by_path"].get(path)
    moved = counted["moves_by_path"].get(path)
    row["memory_class"] = _class_of_path(copied, moved)
    row["copies"] = _tally_of(copied, "events")
    row["copied_rounds"] = _tally_of(copied, "rounds")
    row["copy_bits"] = _tally_of(copied, "bits")
    row["moves"] = _tally_of(moved, "events")
    row["moved_rounds"] = _tally_of(moved, "rounds")
    row["move_bits"] = _tally_of(moved, "bits")
    for counter in REFERENCE_COUNTERS:
        row[counter] = counted[counter]
    return row


def _class_of_path(copied: Optional[dict], moved: Optional[dict]) -> str:
    """The memory class the path crosses, as its counter row named it."""
    if copied is not None:
        return copied["memory_class"]
    return moved["memory_class"]


def _tally_of(counters: Optional[dict], name: str) -> int:
    """One counter of a path's row; a path with no row of that kind is 0."""
    if counters is None:
        return 0
    return counters[name]


def _window_confidence_row(
    measurement: measure.ShotMeasurement,
    confidence: measure.ShotConfidence,
    window,
) -> dict:
    """One window of one shot, as window_confidence.csv writes it."""
    task_key = measured_task_key(measurement)
    operation_id, window_index = window.window_key
    row = task_columns(task_key)
    row["seed"] = measurement.seed
    row["signal"] = confidence.signal
    row["operation_id"] = operation_id
    row["window_index"] = window_index
    row["gap_nats"] = window.gap_nats
    row["escalated"] = window.is_escalated
    row["strong_revised"] = window.is_strong_revised
    row["shot_failed"] = measurement.logical_failure
    return row


def _count_the_shots_gaps(
    counts: dict, measurement: measure.ShotMeasurement
) -> None:
    """One shot's window gaps and its smallest gap into the counts."""
    if not _has_counted_confidence(measurement):
        return
    confidence = measurement.confidence
    if not confidence.windows:
        return
    task_key = measured_task_key(measurement)
    failed = measurement.logical_failure
    for window in confidence.windows:
        key = _histogram_key(
            task_key, confidence.signal, WINDOW_HISTOGRAM, window.gap_nats
        )
        window_cell = key + (window.is_escalated, failed)
        _add_count(counts, window_cell, 1)
    smallest = _smallest_gap(confidence.windows)
    shot_key = _histogram_key(
        task_key, confidence.signal, SHOT_MINIMUM_HISTOGRAM, smallest
    )
    shot_cell = shot_key + (None, failed)
    _add_count(counts, shot_cell, 1)


def _has_counted_confidence(measurement: measure.ShotMeasurement) -> bool:
    """Whether the shot's gaps belong in the confidence files: scored only.

    An unscored shot is sinter's discard, kept out of every count
    conditioned on failure (sinter/_decoding/_decoding.py:120-128); its row
    would count as a success. sweep.csv's unscored_shots counts them.
    """
    if measurement.confidence is None:
        return False
    return measurement.is_scored


def _smallest_gap(windows: tuple) -> Optional[float]:
    """A shot's smallest window gap; None when a window had no gap.

    A window with no gap escalates as one below every threshold does
    (escalation/policies.py), so it is the shot's least confident.
    """
    gaps = []
    for window in windows:
        if window.gap_nats is None:
            return None
        gaps.append(window.gap_nats)
    return min(gaps)


def _histogram_key(task_key: tuple, signal: str, kind: str, gap_nats) -> tuple:
    """A gap's place in the histogram: task, signal, kind and bin.

    A window with no gap has the bin None, written empty.
    """
    low = None
    if gap_nats is not None:
        low = gap_bin_low_decibels(gap_nats)
    return (task_key, signal, kind, low)


def _add_count(counts: dict, key: tuple, count: int) -> None:
    """Add to one histogram cell."""
    already = counts.get(key, 0)
    counts[key] = already + count


def _task_keys_of_counts(counts: dict) -> list:
    """The tasks the counts name, in the order they first came."""
    task_keys = {}
    for key in counts:
        task_keys[key[0]] = None
    return list(task_keys)


def _confidence_histogram_rows_of(counts: dict, task_keys: list) -> list:
    """The counts as rows: tasks' order, kind, bin, escalated, failed."""
    positions = _task_positions(task_keys)
    order = functools.partial(_histogram_order, positions)
    rows = []
    for key in sorted(counts, key=order):
        task_key, signal, kind, low, escalated, failed = key
        row = task_columns(task_key)
        row["signal"] = signal
        row["histogram"] = kind
        row["gap_low_decibels"] = low
        row["escalated"] = escalated
        row["shot_failed"] = failed
        row["count"] = counts[key]
        rows.append(row)
    return rows


def _histogram_order(positions: dict, key: tuple) -> tuple:
    """A histogram cell where a single run would write it."""
    task_key, signal, kind, low, escalated, failed = key
    task_order = positions[task_key]
    kind_order = kind != WINDOW_HISTOGRAM
    low_order = _bin_order(low)
    escalated_order = escalated is True
    return (task_order, signal, kind_order, low_order, escalated_order, failed)


def _bin_order(low: Optional[float]) -> float:
    """A bin's place: the empty bin, a window with no gap, below every gap."""
    if low is None:
        return -math.inf
    return low


def _folded_confidence_histogram(folders: list) -> dict:
    """Every folder's histogram counts added into one set of cells."""
    counts = {}
    for run_dir in folders:
        path = Path(run_dir) / "confidence_histogram.csv"
        if not path.is_file():
            continue
        for row in read_rows(path):
            _add_a_histogram_row(counts, row)
    return counts


def _add_a_histogram_row(counts: dict, row: dict) -> None:
    """One histogram row read back, added to its cell."""
    task_key = task_key_of(row)
    low = _empty_as_none(row["gap_low_decibels"])
    escalated = _empty_as_none(row["escalated"])
    key = (
        task_key,
        row["signal"],
        row["histogram"],
        low,
        escalated,
        row["shot_failed"],
    )
    _add_count(counts, key, row["count"])


def _empty_as_none(value):
    """A cell read back, None where the row wrote nothing."""
    if value == "":
        return None
    return value
