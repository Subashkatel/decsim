"""Shot measurements -> a run folder's additive facts -> the summaries.

A run folder records only facts that add up: one row per shot (its
scalar fields, and each latency point's mean and max over the shot's
windows), one row per shot per link (that shot's own ledger counters),
and one row per distinct microsecond value of each latency point with
how many of the point's windows carried it. Every summary column is
derived from those rows at read time, which is sinter's shape
(sinter/_data/_task_stats.py: TaskStats holds shots, errors, discards,
seconds and a Counter of custom counts, __add__ sums them, and every
rate or interval is computed from the summed row); gem5 keeps its
Distribution statistics the same way, as counts per bucket summed
across simulations. So sweep.csv and links.csv read the same whether
one process ran every shot or many ran a piece each, and the fold of
an experiment's pieces builds them through this module's own summarize
and link_rows.

What a summary needs off those rows is a count, a true count, a sum, a
max and an exact mean per sweep point, and none of those grows with the
shots, so the rows reach them one at a time: the accumulators, the row
streams and the merge that orders them are in experiments/fold.py, and
this module says which field plays which role. A single run feeds the
rows it measured through the same accumulators a fold feeds an
experiment's pieces and folders through, so one code path produces
every summary.

The per-value counts file stays small because every sample is a whole
number of ticks divided by the ticks in a microsecond
(config.ticks_to_microseconds). Under a fixed-latency decoder card,
which is what an LER sweep of a million shots runs, a point's windows
take one of a handful of tick spans, so the file's length follows the
spread of the values and not the shot count. A wall-clock decoder
prices its measured decode into the simulated clock and spreads the
values wide, and those runs are the timing sweeps of hundreds of shots,
which already write one row per window in latency_samples.csv.

Nothing here runs a simulation.
"""

import csv
import dataclasses
import functools
import json
import math
from pathlib import Path
from typing import Optional

import decsim.escalation.settings as escalation_settings
import decsim.experiments.collection as collection
import decsim.experiments.failure_statistics as failure_statistics
import decsim.experiments.fold as fold
import decsim.experiments.measure as measure
import decsim.experiments.refusal as refusal
import decsim.experiments.run_folder as run_folder
import decsim.observe.data_movement as data_movement
import decsim.observe.settings as observe_settings

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
    "trace_path",
    "window_statuses",
    "confidence",
)
# the counters shot_data_movement.csv carries per path, as
# observe/data_movement.py's json_value names them
MOVEMENT_COUNTERS = (
    "copies",
    "copied_rounds",
    "copy_bits",
    "moves",
    "moved_rounds",
    "move_bits",
)
# the counters a hold books for the whole shot: a reference is a token
# on a store's slot and belongs to no path, so these repeat on every row
# of one shot
REFERENCE_COUNTERS = ("references", "referenced_rounds")
# the columns that tell harder windows from an overloaded strong side, in
# sweep.csv order; a point whose shots did not keep them gets none
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
# which role each field of shots.csv plays in a sweep point's row: a
# mean over the point's shots, the largest over them, a sum, or how many
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
    # one value per point, the code card's, so its largest is that value
    "commit_rounds",
    "window_period_us",
    "max_queued_windows",
    "weak_queue_max",
    "strong_queue_max",
    "parallel_processes_needed",
    *LOAD_MAXES,
)
# the windows by final status and the replaced provisional ones left
# uncorrected, summed per point and per piece (pieces.write)
STATUS_SUMS = (
    *measure.WINDOW_STATUS_COLUMNS,
    "provisional_no_correction_windows",
)
SHOT_SUMS = (
    "referee_windows_checked",
    "referee_window_disagreements",
    "decoded_windows",
    "escalated_windows",
    "strong_decoded_rounds",
    "strong_service_sum_us",
    *STATUS_SUMS,
)
SHOT_TRUE_COUNTS = (
    "logical_failure",
    "is_scored",
)
# the burst detector's shot columns, counted true per point when a run
# with a detector wrote them: a first flag round is true when the
# detector flagged a round, since rounds count from 1 and 0 is none
BURST_SHARES = {
    "burst_first_flag_round": "flagged_share",
    "burst_caught_in_time": "caught_in_time_share",
}
# the files a fold reads row by row and writes back out, one header
# each: every column any piece holds, the pieces of one point alike
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
# links.csv is the mean of these over a point's shots, one row per link
LINK_MEANS = (
    "transfers",
    "payload_bits",
    "unknown_payload_transfers",
    "queue_wait_us",
    "serialization_us",
    "propagation_us",
)


@dataclasses.dataclass(frozen=True)
class RunRecord:
    """The additive facts one or more run folders hold.

    Each field is a list of csv rows and each list is one file of the
    run folder: shots.csv, shot_links.csv, shot_data_movement.csv,
    window_samples.csv, latency_samples.csv, window_confidence.csv and
    confidence_histogram.csv. No field holds a summary, so two folders'
    records join by concatenation (the window samples' and the
    histogram's counts add), and the summaries derived from the join
    are the summaries a single run over the same shots would have
    written.
    """

    shots: list
    shot_links: list
    shot_data_movement: list
    window_samples: list
    latency_samples: list
    window_confidence: list
    confidence_histogram: list


def percentile_of_counts(multiset: dict, fraction: float) -> float:
    """The value at `fraction` of the samples, numpy's nearest method.

    The index is round((n - 1) * fraction) into the sorted samples, which
    is numpy.percentile(method="nearest"); a position half way between
    two samples takes the even index, so the median of an even count is
    the lower middle for 2 or 6 samples and the upper for 4. It is not
    the classical nearest rank, ceil(n * fraction).

    multiset maps a microsecond value to how many samples carry it.
    Expanding it and sorting gives the list this walks in place, so a
    point's percentile is the same however its shots were split across
    processes or pieces.
    """
    total = _total_count(multiset)
    if total == 0:
        return 0.0
    last_index = total - 1
    position = fraction * last_index
    rounded = round(position)
    index = min(last_index, int(rounded))
    return _value_at_index(multiset, index)


def sweep_point_of(row: dict) -> tuple:
    """The point a row belongs to: its id and its algorithm.

    The id is the point's strong id, sinter's strong_id column, so two
    points apart in any setting are two points; the algorithm rides
    with it so every derived row says what it is of, and write_csv adds
    the point's swept values. A row read back off a csv file holds text
    and a row this process measured holds numbers, so the algorithm is
    read as the value it was written from: one streamed row and one
    measured row of the same shot name the same point.
    """
    return (row["point_id"], fold.number_of(row["algorithm"]))


def measured_point(measurement: measure.ShotMeasurement) -> tuple:
    """The sweep point one measured shot belongs to."""
    return (measurement.point_id, measurement.algorithm)


def point_columns(point: tuple) -> dict:
    """The two columns that name a sweep point."""
    point_id, algorithm = point
    return {"point_id": point_id, "algorithm": algorithm}


def summarize_point(
    point: tuple,
    totals: fold.RowTotals,
    counts: dict,
    prefix: collection.PrefixTracker,
) -> dict:
    """One sweep point: means over seeds of per-shot means, max of maxes.

    The counts cover every shot the point holds; the estimate and its
    limits cover its contiguous prefix up to its stop (prefix), which is
    what the stopping rule's intervals are exact for.
    """
    failures = totals.true_counts["logical_failure"]
    shot_count = totals.rows
    scored_shots = totals.true_counts["is_scored"]
    unscored_shots = shot_count - scored_shots
    row = point_columns(point)
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
    _add_burst_columns(row, totals)
    for name in _points_held(totals.means):
        multiset = _multiset_over_tiers(counts, point, name)
        _add_latency_point_columns(row, totals, name, multiset)
    return row


def summarize(
    shots: list, window_samples: list, rules: Optional[dict] = None
) -> list:
    """One row per sweep point, in a stable order.

    The two arguments are the additive files a run folder writes: the
    per-shot rows and the per-value counts. The same code runs whether
    they were just measured in this process or read back out of an
    experiment's pieces, whose fold streams them into the same totals
    rather than holding a list of them. rules maps a point id to the
    collection.PointRule its prefix is read by; a point it does not
    name ran a shot count fixed in advance.
    """
    counts = _counts_of_rows(window_samples)
    totals = {}
    prefixes = {}
    for row in shots:
        _add_a_shot(totals, row)
        _add_to_the_prefix(prefixes, rules, row)
    return summary_rows(totals, counts, prefixes)


def summary_rows(totals: dict, counts: dict, prefixes: dict) -> list:
    """One row per sweep point whose shots were totalled, in point order.

    The totals hold the points in the order their first shots came,
    which is the sweep's task order for one run and for a fold alike
    (a fold merges the pieces' rows in task order).
    """
    rows = []
    for point in totals:
        at_point = totals[point]
        point_id, _algorithm = point
        prefix = prefixes[point_id]
        row = summarize_point(point, at_point, counts, prefix)
        rows.append(row)
    return rows


def write_csv(rows: list, path: Path, swept: Optional[dict] = None) -> None:
    """The rows as a csv file, every column any row holds, first seen first.

    swept maps a point id to its value at each swept yaml path
    (run_folder.swept_values), and a row of that point takes one column
    per path right after its point_id: the values the design fixed
    first, then what was measured, Wickham's order (Tidy Data, J. Stat.
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


def terminal_lines(rows: list, report_dir: Path) -> list:
    """The terminal summary: one labeled block per sweep point.

    Full names, no abbreviations; the full record is sweep.csv. Under
    each point come the bits its shots copied and moved, grouped by the
    memory class the hop crossed, read back from the folder's
    data_movement.csv. A run with observation.data_movement off wrote no
    such file and says so once at the end, because a run that counted
    nothing has no counts and a row of zeros would claim otherwise.
    """
    movement = _data_movement_of(report_dir)
    swept = swept_values_of(report_dir)
    blocks = []
    for row in rows:
        values = swept[row["point_id"]]
        block = _terminal_block(row, values)
        at_point = _movement_at_point(movement, row)
        with_classes = _block_with_classes(block, at_point)
        blocks.append(with_classes)
    joined_blocks = "\n\n".join(blocks)
    lines = joined_blocks.split("\n")
    if not movement:
        lines.append("")
        lines.append(
            "data movement: observation.data_movement was off, so this run "
            "counted no copies, references or moves"
        )
    return lines


def link_rows(shot_links: list) -> list:
    """One row per sweep point per link, averaged over the point's shots.

    The totals came straight off each shot's TrafficCounters; nothing
    here re-counts transfers.
    """
    totals = _link_totals_by_point(shot_links)
    return _link_rows_of(totals)


def data_movement_rows(shot_data_movement: list) -> list:
    """One row per sweep point per path, then one per memory class.

    Two tables in one file, told apart by the grouping column. The
    second one is the grouping the classical sources ask for: a DRAM
    access costs "a couple of orders-of-magnitude higher than the cost
    of an internal cache access" (Horowitz, ISSCC 2014 lines 232-247)
    and an accelerator's access costs what the memory it reads costs
    (Dally, CACM 2020 lines 231-234), so a copy into a register and a
    copy across a cryostat link may not be summed into one count. Every
    number is a mean over the point's shots.
    """
    totals = _movement_totals_by_point(shot_data_movement)
    return _movement_rows_of(totals)


def shot_data_movement_rows(measurements: list) -> list:
    """One row per shot per path: that shot's own copy and move counters.

    A path is a copy's source and target (`controller assembler -> weak
    syndrome buffer`) or a link's own name, and each row names the
    memory class that path crosses, which observe/data_movement.py
    places from the sources. A shot whose run had
    observation.data_movement off writes no row at all: a run that
    counted nothing has no counts, which is not the same fact as a run
    whose counts were zero.
    """
    rows = []
    for measurement in measurements:
        counted = measurement.data_movement
        if counted is None:
            continue
        point = measured_point(measurement)
        for path in _paths_counted(counted):
            row = _shot_movement_row(point, measurement, counted, path)
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

    links.csv is the mean of these over a point's shots. They are a
    file of their own rather than sixty more columns of shots.csv
    because the link set is data and not schema: the counters are the
    columns links.csv already has, so the summary is a mean over rows,
    and a topology with another link adds rows and moves no column.
    """
    rows = []
    for measurement in measurements:
        point = measured_point(measurement)
        for path in sorted(measurement.link_totals):
            row = _shot_link_row(point, measurement, path)
            rows.append(row)
    return rows


def window_sample_rows(measurements: list) -> list:
    """One row per point, latency point, tier and distinct value: its count.

    The multiset of a point's window samples, which is all its median
    and p99 columns need and all a piece has to record for another
    process to reach the same numbers. Each window's samples sit under
    the tier that committed it, so kept and escalated windows are two
    multisets; a round's sample has no window and names no tier.
    """
    counts = _counts_of_samples(measurements)
    points = []
    for point, _name, _tier in counts:
        points.append(point)
    unique_points = dict.fromkeys(points)
    return _rows_of_counts(counts, list(unique_points))


def latency_sample_rows(measurements: list) -> list:
    """One row per decoded window: the algorithm stage's time.

    Only algorithms named by a table row produce rows, and the time is
    what held the unit: the measured wall clock, or the row's own cycle
    count. A number instead of a name is a fixed latency and produces
    none. These are the inputs a latency figure is drawn from, so a
    reader draws one from run folders alone; each row carries the tier
    that decoded its window and the window's inter-arrival, the
    deadline a decode must beat. What such a figure computes (log
    times, densities, medians) is computed when it is drawn, not
    stored.
    """
    rows = []
    for measurement in measurements:
        if not isinstance(measurement.algorithm, str):
            continue
        point = measured_point(measurement)
        samples = measurement.samples["algorithm"]
        tiers = measurement.window_tiers
        for sample_us, tier in zip(samples, tiers, strict=True):
            row = point_columns(point)
            row["seed"] = measurement.seed
            row["tier"] = tier
            row["algorithm_us"] = sample_us
            row["window_period_us"] = measurement.window_period_us
            rows.append(row)
    return rows


def window_confidence_rows(measurements: list) -> list:
    """One row per window of each sampled shot: its gap and its verdict.

    Toshio et al. keep each shot's gap with whether the decode was right
    (2510.25222 lines 722-731); a window has no truth of its own, so a
    row carries the shot's failure and whether the strong decode
    revised the window's answer. Only the scored shots of those
    observation.confidence_shot_count names write rows; a run with no
    confidence signal writes none.
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
    (sinter/_data/_task_stats.py:51-71), so the histogram covers every
    scored shot however many window_confidence.csv lists. From it come
    Toshio's p(g) and P(e|g) and the brute-force cutoff (2510.25222
    lines 807-841, 863-900).
    """
    counts = {}
    for measurement in measurements:
        _count_the_shots_gaps(counts, measurement)
    points = _points_of_counts(counts)
    return _confidence_histogram_rows_of(counts, points)


def gap_bin_low_decibels(gap_nats: float) -> float:
    """The lower edge, in decibels, of the 0.1 dB bin holding the gap.

    A gap in nats is ln of the likelihood ratio and a decibel is
    10 log10 of it (escalation/settings.py nats_to_decibels); Toshio et
    al. histogram their gaps in decibels (2510.25222 main.tex:564). An
    infinite gap, a window whose other class has no fault set, keeps
    its own bin.
    """
    decibels = escalation_settings.nats_to_decibels(gap_nats)
    if math.isinf(decibels):
        return decibels
    scaled = decibels * CONFIDENCE_BINS_PER_DECIBEL
    tenths = math.floor(scaled)
    return tenths / CONFIDENCE_BINS_PER_DECIBEL


def confidence_shot_count_of(measurements: list):
    """The shots a piece's confidence rows cover, for its piece.json.

    The first shots' count, the word all for every shot, or None when no
    confidence signal ran and the piece wrote neither confidence file.
    A piece saved before the confidence files has no such line, and the
    fold refuses to join it to one that has
    (_refuse_pieces_that_recorded_confidence_apart).
    """
    for measurement in measurements:
        confidence = measurement.confidence
        if confidence is None:
            continue
        if confidence.sampled_shot_count is None:
            return observe_settings.EVERY_SHOT
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


def rows_by_point(run_dir: Path) -> dict:
    """sweep.csv's rows, keyed by their point id."""
    sweep_path = run_dir / "sweep.csv"
    rows = {}
    for row in read_rows(sweep_path):
        rows[row["point_id"]] = row
    return rows


def fold_pieces(
    experiment_dir: Path,
    folders: list,
    point_ids: list,
    seeds_by_point: dict,
    out_dir: Path,
    rules: Optional[dict] = None,
) -> list:
    """Pieces' additive files folded into a run folder; its sweep rows.

    A piece holds a run folder's additive files for a range of one
    point's seeds. The points' records go into the run folder with the
    seeds their pieces hold (seeds_by_point), and every folded row takes
    its point's swept values from them. The rows come back in the order
    one run over every piece would write them: its points in task order,
    then its seeds. Two pieces may not share a shot, which would be
    counted twice. rules maps a point id to the collection.PointRule
    its estimate is read by (summarize).

    An experiment's pieces can hold more rows than a process can: 500
    pieces of a million-shot sweep are 115 million link rows. So the
    pieces are read in a stream, one row of each at a time, and what
    stands between reading and writing is the totals of
    experiments/fold.py and not a list of the rows.
    """
    order = _refused_or_ordered(folders, point_ids)
    run_folder.copy_points_of(
        experiment_dir, point_ids, seeds_by_point, out_dir
    )
    return _fold_into(folders, point_ids, order, out_dir, rules)


def refuse_pieces_of_another_tree(folders: list) -> None:
    """Refuse a collect onto a point whose saved pieces ran another tree.

    A collect names this tree in its run folder's manifest before its
    first shot and folds the pieces it adds with the saved ones, so it
    asks the fold's own refusal first, this tree standing for the pieces
    it would add, and a refused collect spends no shot and leaves the
    run folder as the earlier tree wrote it.
    """
    identity = run_folder.piece_identity()
    this_code = _code_of(identity)
    code_by_point = _pieces_by_point_and_value(folders, _code_of)
    for by_code in code_by_point.values():
        by_code.setdefault(this_code, "this collect")
    _refuse_a_point_of_two_trees(code_by_point)


def swept_values_of(report_dir: Path) -> dict:
    """The swept values of every point a run folder's manifest lists."""
    manifest = _manifest_of(report_dir)
    return run_folder.swept_values(report_dir, manifest["points"])


class _PointMovement:
    """One sweep point's data-movement totals, one row at a time.

    A row carries one shot's counters on one path, so it belongs to that
    path's totals and to the totals of the memory class the path
    crosses. It also carries the shot's own hold counters, which belong
    to no path and are read once per shot: the rows of one shot are
    together in the order a run writes them and in the merged order of
    several folders, so a seed that differs from the last one is the
    next shot.
    """

    def __init__(self) -> None:
        self.by_shot = fold.RowTotals(sums=REFERENCE_COUNTERS)
        self.by_group = {}
        self.last_seed = None

    def add(self, row: dict) -> None:
        """One shot's row on one path, into the three totals it belongs to."""
        seed = row["seed"]
        if seed != self.last_seed:
            self.by_shot.add(row)
            self.last_seed = seed
        at_path = self._group_totals("path", row["path"])
        at_path.add(row)
        at_class = self._group_totals("memory_class", row["memory_class"])
        at_class.add(row)

    def paths(self) -> list:
        """Every path the point's shots copied or moved along, in order."""
        return self._names_of("path")

    def classes(self) -> list:
        """The memory classes these rows crossed, cheapest first.

        The order is observe/data_movement.py's own CLASS_ORDER, so the
        grouped table reads the way the counters group it.
        """
        crossed = self._names_of("memory_class")
        listed = []
        for memory_class in data_movement.CLASS_ORDER:
            if memory_class.value in crossed:
                listed.append(memory_class.value)
        return listed

    def totals_of(self, grouping: str, name: str):
        """One path's or one memory class's counter sums."""
        return self.by_group[(grouping, name)]

    def _group_totals(self, grouping: str, name: str):
        """The totals of one path or class, made the first time it shows."""
        key = (grouping, name)
        totals = self.by_group.get(key)
        if totals is None:
            totals = fold.RowTotals(sums=MOVEMENT_COUNTERS)
            self.by_group[key] = totals
        return totals

    def _names_of(self, grouping: str) -> list:
        """The names one grouping holds, sorted."""
        names = []
        for group, name in self.by_group:
            if group == grouping:
                names.append(name)
        return sorted(names)


def _refused_or_ordered(folders: list, point_ids: list):
    """The fold's row order, once the folders pass every refusal.

    The refusals run before anything is written, so a refused fold
    leaves no file behind.
    """
    positions = _task_positions(point_ids)
    order = functools.partial(_row_task_and_seed, positions)
    _refuse_folders_of_different_columns(folders)
    _refuse_a_point_this_tree_cannot_place(folders)
    _refuse_pieces_that_recorded_confidence_apart(folders)
    _refuse_pieces_that_ran_different_code(folders)
    _refuse_a_repeated_shot(folders, order)
    return order


def _fold_into(
    folders: list, point_ids: list, order, out_dir: Path, rules
) -> list:
    """The folders' rows into out_dir, each with its point's swept values."""
    swept = run_folder.swept_values(out_dir, point_ids)
    return _fold_the_folders(folders, order, out_dir, swept, rules)


def _fold_the_folders(
    folders: list, order, out_dir: Path, swept: dict, rules
) -> list:
    """Every folder's additive files into one folder's, a pass per file.

    A pass reads one file of every folder at once, writes the folded
    file row by row and feeds the totals its summary needs, so no pass
    holds a folder's rows. The three derived files come last, off the
    totals the passes built, each point's swept values beside its rows.
    """
    shot_totals, prefixes = _fold_shots(folders, order, out_dir, swept, rules)
    counts = _folded_counts(folders)
    rows = summary_rows(shot_totals, counts, prefixes)
    sweep_path = out_dir / "sweep.csv"
    write_csv(rows, sweep_path, swept)
    samples_path = out_dir / "window_samples.csv"
    samples_rows = _rows_of_counts(counts, list(shot_totals))
    _write_rows(samples_rows, samples_path, swept)
    link_totals = _fold_shot_links(folders, order, out_dir, swept)
    per_link = _link_rows_of(link_totals)
    links_path = out_dir / "links.csv"
    write_csv(per_link, links_path, swept)
    movement_totals = _fold_shot_movement(folders, order, out_dir, swept)
    per_movement = _movement_rows_of(movement_totals)
    movement_path = out_dir / "data_movement.csv"
    _write_rows(per_movement, movement_path, swept)
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
    """Every folder's shots.csv into one; each point's totals and prefix."""
    totals = {}
    prefixes = {}
    add_a_shot = functools.partial(_add_a_folded_shot, totals, prefixes, rules)
    _fold_one_file(folders, "shots.csv", order, out_dir, swept, add_a_shot)
    return totals, prefixes


def _add_a_folded_shot(
    totals: dict, prefixes: dict, rules: Optional[dict], row: dict
) -> None:
    """One folded shot row into its point's totals and its prefix."""
    _add_a_shot(totals, row)
    _add_to_the_prefix(prefixes, rules, row)


def _fold_shot_links(folders: list, order, out_dir: Path, swept: dict) -> dict:
    """Every folder's shot_links.csv into one, and the per-link totals."""
    totals = {}
    add_a_link = functools.partial(_add_a_shot_link, totals)
    name = "shot_links.csv"
    _fold_one_file(folders, name, order, out_dir, swept, add_a_link)
    return totals


def _fold_shot_movement(
    folders: list, order, out_dir: Path, swept: dict
) -> dict:
    """Every folder's shot_data_movement.csv into one, and its totals."""
    totals = {}
    add_a_row = functools.partial(_add_a_movement_row, totals)
    name = "shot_data_movement.csv"
    _fold_one_file(folders, name, order, out_dir, swept, add_a_row)
    return totals


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

    A piece holds bare rows, so each row takes its point's swept values
    after its point_id here, as write_csv places them.
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

    That is write_csv's header: point_id, the swept paths, then each
    file's columns. Two points of one grid can measure different
    columns, a quiet shot having no burst to catch, and a point's cell
    for a column it did not measure is empty, as a swept path a point's
    sections do not hold is (run_folder.swept_values).
    """
    columns = {"point_id": None}
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
    """One shot row onto its point's prefix, its tracker made at first."""
    point_id = row["point_id"]
    tracker = prefixes.get(point_id)
    if tracker is None:
        rule = _rule_of(rules, point_id)
        tracker = collection.PrefixTracker(rule)
        prefixes[point_id] = tracker
    tracker.add(row)


def _rule_of(rules: Optional[dict], point_id: str) -> collection.PointRule:
    """The point's rule, or a shot count fixed in advance when none."""
    if rules is None or point_id not in rules:
        return collection.PointRule()
    return rules[point_id]


def _add_estimate_columns(row: dict, prefix: collection.PrefixTracker) -> None:
    """The prefix's state, counts, estimate and exact limits.

    failures over scored shots is sinter's errors over shots less
    discards (sinter/_plotting.py:389). An adaptive point's shots are
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
    """The prefix's estimate and limits; none for an adaptive point."""
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
    """The difficulty and overload columns the point's shots hold.

    The escalated share rides with them: the windows the strong tier
    committed over the windows decoded, the escalation rate a burst
    raises when it makes the weak decoder unsure.
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


def _add_burst_columns(row: dict, totals) -> None:
    """The share of the point's shots flagged, and caught in time.

    On shots with no burst the flagged share is the share holding a
    false alarm; over one shot's time it is the false-alarm rate.
    """
    for column, share in BURST_SHARES.items():
        if column in totals.true_counts:
            flagged = totals.true_counts[column]
            row[share] = flagged / totals.rows


def _strong_service_mean_us(totals: fold.RowTotals) -> Optional[float]:
    """The point's strong service over its strong decodes; None for none.

    A ratio of two sums, gem5's Formula avgMissLatency = missLatency /
    misses (src/mem/cache/base.cc:2187-2188), whose nonan flag prints
    nothing for a zero count (src/base/stats/text.cc:288-290); sinter
    likewise sums counts and divides at read time
    (sinter/_data/_anon_task_stats.py:57-78). A mean of each shot's
    mean would weigh a shot of one decode like a shot of ten. A folder
    whose shots hold no sum gets no mean, the rule of _points_held.
    """
    escalated_windows = totals.sums["escalated_windows"]
    if escalated_windows == 0:
        return None
    return totals.sums["strong_service_sum_us"] / escalated_windows


def strong_service_bound_us(totals: fold.RowTotals) -> float:
    """Toshio's Theorem 1 bound on one strong decode's time, per point.

    Theorem 1 bounds the strong decoder's time per round (2510.25222
    line 1794), tau_strong <= (1 / gamma_switch)(d / r_strong) tau_gen
    (eq. (6), lines 1206-1214), with gamma_switch per d rounds (line
    1788). A decode reads r_strong rounds, so one decode's time is
    bounded by d tau_gen / gamma_switch. A window escalates with
    probability gamma_switch r_com / d (lines 1254-1255), the point's
    escalated windows over its windows, so the bound is tau_gen r_com
    windows / escalated windows, the same unit as
    strong_service_mean_us beside it; infinite when nothing escalated.
    tau_gen r_com is the shots' window_period_us column.
    """
    escalated_windows = totals.sums["escalated_windows"]
    if escalated_windows == 0:
        return math.inf
    windows = totals.sums["decoded_windows"]
    window_period_us = totals.maxes["window_period_us"]
    return window_period_us * windows / escalated_windows


def _shot_totals(row: dict) -> fold.RowTotals:
    """What one sweep point's shot rows add up to, role by role.

    A role's column is totalled only when the row holds it, the rule of
    _points_held, so a folder an older tree wrote still folds.
    """
    means = _held_by(row, SHOT_MEANS)
    maxes = _held_by(row, SHOT_MAXES)
    sums = _held_by(row, SHOT_SUMS)
    burst_counts = _held_by(row, tuple(BURST_SHARES))
    for name in _points_held(row):
        means.append(f"{name}_mean_us")
        maxes.append(f"{name}_max_us")
    return fold.RowTotals(
        means=means,
        maxes=maxes,
        sums=sums,
        true_counts=(*SHOT_TRUE_COUNTS, *burst_counts),
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

    A run folder is read by the columns it holds and not by the columns
    the reading tree would write: the 500 folders of one weak_ler
    sweep hold the sixteen latency points the tree that wrote them
    measured, and this tree measures twenty-three, so a summary that asked
    for its own could not read those folders at all. It is the rule that
    lets a fold read a folder an older tree wrote, which is the same
    rule sinter's counter table keeps, where a counter a file does not
    carry is simply not in the folded row: custom_counts is declared
    Counter[str] with collections.Counter as its default factory
    (.pydeps/sinter/_data/_task_stats.py:71) and folded by adding the
    two counters (:117-150). A point a folder did not measure gets no
    column, because a column of zeros would say its windows took no
    time.
    """
    held = []
    for name in measure.POINTS:
        column = f"{name}_mean_us"
        if column in fields:
            held.append(name)
    return held


def _add_a_shot(totals: dict, row: dict) -> None:
    """One shot row into its own sweep point's totals."""
    point = sweep_point_of(row)
    at_point = totals.get(point)
    if at_point is None:
        at_point = _shot_totals(row)
        totals[point] = at_point
    at_point.add(row)


def _link_totals_by_point(shot_links: list) -> dict:
    """Each sweep point's totals per link, the rows read once."""
    totals = {}
    for row in shot_links:
        _add_a_shot_link(totals, row)
    return totals


def _add_a_shot_link(totals: dict, row: dict) -> None:
    """One shot's row on one link into that point's and link's totals."""
    point = sweep_point_of(row)
    at_point = totals.get(point)
    if at_point is None:
        at_point = {}
        totals[point] = at_point
    link = row["link"]
    at_link = at_point.get(link)
    if at_link is None:
        at_link = fold.RowTotals(means=LINK_MEANS)
        at_point[link] = at_link
    at_link.add(row)


def _link_rows_of(totals: dict) -> list:
    """One row per point per link, the points and the links in order."""
    rows = []
    for point in totals:
        at_point = totals[point]
        for path in sorted(at_point):
            row = _link_row(point, path, at_point[path])
            rows.append(row)
    return rows


def _movement_totals_by_point(shot_data_movement: list) -> dict:
    """Each sweep point's data-movement totals, the rows read once."""
    totals = {}
    for row in shot_data_movement:
        _add_a_movement_row(totals, row)
    return totals


def _add_a_movement_row(totals: dict, row: dict) -> None:
    """One shot's row on one path into its sweep point's totals."""
    point = sweep_point_of(row)
    at_point = totals.get(point)
    if at_point is None:
        at_point = _PointMovement()
        totals[point] = at_point
    at_point.add(row)


def _movement_rows_of(totals: dict) -> list:
    """One point's paths and then its memory classes, point by point."""
    rows = []
    for point in totals:
        at_point = totals[point]
        for row in _movement_rows_at_point(point, at_point):
            rows.append(row)
    return rows


def _movement_rows_at_point(point: tuple, movement: _PointMovement) -> list:
    """One point's two tables: a row per path, then a row per class."""
    rows = []
    for path in movement.paths():
        row = _movement_row(point, "path", path, movement)
        rows.append(row)
    for memory_class in movement.classes():
        row = _movement_row(point, "memory_class", memory_class, movement)
        rows.append(row)
    return rows


def _write_rows(rows: list, path: Path, swept: dict) -> None:
    """One file of the record, left unwritten when it has no rows."""
    if not rows:
        return
    write_csv(rows, path, swept)


def _with_swept_values(rows: list, swept: dict) -> list:
    """Each row with its point's swept values right after its point_id."""
    placed = []
    for row in rows:
        with_values = _with_swept_row(row, swept)
        placed.append(with_values)
    return placed


def _with_swept_row(row: dict, swept: dict) -> dict:
    """One row with its point's swept values right after its point_id."""
    point_id = row["point_id"]
    with_values = {"point_id": point_id}
    with_values.update(swept[point_id])
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
    """(point, latency point, tier) -> value -> how many carried it."""
    counts = {}
    for measurement in measurements:
        point = measured_point(measurement)
        for name, values in measurement.samples.items():
            tiers = _tiers_of_samples(measurement, name)
            _count_the_values(counts, (point, name), tiers, values)
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


def _multiset_over_tiers(counts: dict, point: tuple, name: str) -> dict:
    """One latency point's multiset at one sweep point, every tier's added."""
    merged = {}
    for (at_point, at_name, _tier), multiset in counts.items():
        if (at_point, at_name) != (point, name):
            continue
        for value, count in multiset.items():
            already = merged.get(value, 0)
            merged[value] = already + count
    return merged


def _counts_of_rows(window_samples: list) -> dict:
    """The same multisets, read back off window_samples.csv rows."""
    counts = {}
    _add_counts_of_rows(counts, window_samples)
    return counts


def _add_counts_of_rows(counts: dict, window_samples: list) -> None:
    """One file's rows added to the multisets their points own.

    The counts add, so the multisets of several folders' files are the
    multisets one run over the same shots would have written.
    """
    for row in window_samples:
        point = sweep_point_of(row)
        key = (point, row["name"], row["tier"])
        at_this_name = counts.setdefault(key, {})
        value = row["value_us"]
        already = at_this_name.get(value, 0)
        at_this_name[value] = already + row["count"]


def _rows_of_counts(counts: dict, points: list) -> list:
    """The multisets as rows: points' order, then point-list order, value."""
    positions = _task_positions(points)
    order = functools.partial(_point_and_name_order, positions)
    rows = []
    for key in sorted(counts, key=order):
        multiset = counts[key]
        for row in _rows_of_one_multiset(key, multiset):
            rows.append(row)
    return rows


def _rows_of_one_multiset(key: tuple, multiset: dict) -> list:
    """One latency point's counts at one sweep point, by rising value."""
    point, name, tier = key
    columns = point_columns(point)
    rows = []
    for value in sorted(multiset):
        row = dict(columns)
        row["name"] = name
        row["tier"] = tier
        row["value_us"] = value
        row["count"] = multiset[value]
        rows.append(row)
    return rows


def _point_and_name_order(positions: dict, key: tuple) -> tuple:
    """A (point, latency point, tier) key where a single run writes it."""
    point, name, tier = key
    place = measure.POINTS.index(name)
    point_order = positions[point]
    return (point_order, place, tier)


def _task_positions(points: list) -> dict:
    """Each point's place in the list, the order a single run meets them."""
    positions = {}
    for position, point in enumerate(points):
        positions[point] = position
    return positions


def _manifest_of(run_dir) -> dict:
    """One run folder's manifest.json."""
    manifest_path = Path(run_dir) / "manifest.json"
    manifest_text = manifest_path.read_text()
    return json.loads(manifest_text)


def _refuse_folders_of_different_columns(run_dirs: list) -> None:
    """The pieces of one point record the same columns in a folded file."""
    for name in FOLDED_FILES:
        _refuse_one_files_different_columns(run_dirs, name)


def _refuse_a_point_this_tree_cannot_place(run_dirs: list) -> None:
    """Every folder's window samples name a point this tree measures.

    window_samples.csv is the one folded file read by a column's values
    and not by its header: every row names a latency point, and the
    fold writes that point's counts where the point sits among this
    tree's own (_point_and_name_order calls measure.POINTS.index). A
    folder whose rows name a point this tree does not measure, which is
    the other half of two trees measuring different points, has no
    place in that order, and the sort would fail with sweep.csv and
    shots.csv already written. It is checked here, at the boundary,
    before out_dir exists, the way the folded files' columns are, and it
    costs one pass over the small file: one folder's window_samples.csv
    of a weak_ler sweep is 188 rows and 9 KB.

    Only the names are checked, not that the folders name the same set.
    A point may hold a column and no sample at all, measured:
    csb_stall_per_round has a mean column in every shots.csv and no
    window sample on a weak-only tree, because no round reached the
    strong store. And the counts add per (sweep point, latency point),
    so a folder with no sample of a point contributes none and the
    fold's multiset is still the one a single run over the same shots
    would have written. What would be wrong, a folder written by a tree
    whose points are not this tree's, is already refused by the columns:
    shot_rows writes one _mean_us column per point of the tree that
    wrote it, so the folded files' headers already pin every folder to
    one points list.
    """
    for run_dir in run_dirs:
        path = Path(run_dir) / "window_samples.csv"
        if not path.is_file():
            continue
        names = _window_sample_names(path)
        _refuse_the_point(run_dir, names)


def _window_sample_names(path: Path) -> list:
    """The latency points one folder's window samples name, sorted."""
    names = set()
    for row in fold.row_stream(path):
        names.add(row["name"])
    return sorted(names)


def _refuse_the_point(run_dir, names: list) -> None:
    """Say which folder names which point that this tree cannot place."""
    for name in names:
        if name in measure.POINTS:
            continue
        raise refusal.RefusalError(
            f"{run_dir} holds window samples of the latency point {name!r}, "
            "which this tree does not measure; a fold writes a point's "
            "counts where that point sits among this tree's own, so a "
            "folder naming one it has no place for is refused before the "
            "fold reads a row"
        )


def _refuse_one_files_different_columns(run_dirs: list, name: str) -> None:
    """One folded file's columns, the same in every piece of one point.

    A point's pieces were measured by one tree, so they carry the same
    columns; a folder that wrote no row of this kind has none to carry.
    Two trees' pieces differ when a column was added between them, and
    folding them would leave that column empty for the older piece's
    shots rather than say so. Two points may differ, since a grid's
    points can measure different things (_folded_columns).
    """
    first_by_point = {}
    for run_dir in run_dirs:
        path = Path(run_dir) / name
        point_id = _point_of_file(path)
        if point_id is None:
            continue
        columns = fold.header_of(path)
        first = first_by_point.setdefault(point_id, (run_dir, columns))
        first_dir, first_columns = first
        if columns == first_columns:
            continue
        _refuse_the_columns(run_dir, first_dir, name, columns, first_columns)


def _point_of_file(path: Path) -> Optional[str]:
    """The point a piece's file holds rows of, or None for no row."""
    if not path.is_file():
        return None
    rows = fold.row_stream(path)
    first_row = next(rows, None)
    rows.close()
    if first_row is None:
        return None
    return first_row["point_id"]


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
        f"missing {missing}, extra {extra}; the pieces of one point record "
        "the same columns, and two trees' pieces differ when a column was "
        "added between them"
    )


def _refuse_a_repeated_shot(run_dirs: list, order) -> None:
    """One seeded run in two folders would be counted twice.

    A point may be split across folders now, since its summary is
    derived from additive rows, but a shot may not: the same seed of
    the same point is the same run. The folders' shots.csv rows arrive
    merged, one row per shot, so a point's rows come in seed order and
    the check holds the last seed of each point rather than every shot
    of the experiment. It runs before anything is written, so a refused
    fold leaves no folder behind.
    """
    paths = _folder_files(run_dirs, "shots.csv")
    seen = {}
    for row in fold.merged_rows(paths, order):
        point = sweep_point_of(row)
        seed = row["seed"]
        if seen.get(point) == seed:
            _refuse_the_folders(row, run_dirs)
        seen[point] = seed


def _refuse_pieces_that_recorded_confidence_apart(run_dirs: list) -> None:
    """A point's pieces must have recorded confidence for the same shots.

    A piece saved before the confidence files, or with another
    observation.confidence_shot_count, holds its shots in shots.csv but
    other or no rows in the confidence files, so their fold would count
    fewer shots there than in the sweep row. A piece.json without the
    line is such an older piece, and it reads as None.
    """
    coverage_by_point = _pieces_by_point_and_value(
        run_dirs, _confidence_coverage_of
    )
    for point_id, by_coverage in coverage_by_point.items():
        if len(by_coverage) > 1:
            _refuse_the_confidence_coverage(point_id, by_coverage)


def _refuse_pieces_that_ran_different_code(run_dirs: list) -> None:
    """A point's pieces must have run one tree: one commit, clean or dirty.

    A point's estimate pools its pieces' shots, and the run folder's
    manifest names one tree, the folding process's, so pieces of two
    trees would pool two simulators under one name. A dirty tree's
    changes are not recorded per piece, so two dirty pieces of one
    commit pass, and a clean and a dirty one do not.
    """
    code_by_point = _pieces_by_point_and_value(run_dirs, _code_of)
    _refuse_a_point_of_two_trees(code_by_point)


def _refuse_a_point_of_two_trees(code_by_point: dict) -> None:
    for point_id, by_code in code_by_point.items():
        if len(by_code) > 1:
            _refuse_the_code(point_id, by_code)


def _pieces_by_point_and_value(run_dirs: list, value_of) -> dict:
    """Each point's first piece folder for every value value_of reads."""
    folders_by_point = {}
    for run_dir in run_dirs:
        piece = _piece_of(run_dir)
        value = value_of(piece)
        by_value = folders_by_point.setdefault(piece["point_id"], {})
        by_value.setdefault(value, run_dir)
    return folders_by_point


def _confidence_coverage_of(piece: dict):
    """The shots a piece recorded confidence for; None for an older piece."""
    return piece.get("confidence_shot_count")


def _code_of(piece: dict) -> tuple:
    """The tree a piece ran: its commit and whether it was dirty."""
    return (piece["commit"], piece["dirty"])


def _refuse_the_code(point_id: str, by_code: dict):
    """Say which pieces of the point ran which tree."""
    pieces_named = []
    for (commit, is_dirty), run_dir in by_code.items():
        pieces_named.append(f"{run_dir} (commit {commit}, dirty {is_dirty})")
    listed = ", ".join(pieces_named)
    raise refusal.RefusalError(
        f"the pieces of point {point_id} ran different code: {listed}; "
        "one estimate would pool two simulators under one manifest, so "
        "collect the point into a new experiment folder, or move one "
        "tree's pieces out of this one"
    )


def _piece_of(run_dir) -> dict:
    """One piece folder's piece.json."""
    piece_path = Path(run_dir) / "piece.json"
    piece_text = piece_path.read_text()
    return json.loads(piece_text)


def _refuse_the_confidence_coverage(point_id: str, by_coverage: dict):
    """Say which pieces of the point recorded confidence for which shots."""
    pieces_named = []
    for coverage, run_dir in by_coverage.items():
        pieces_named.append(f"{run_dir} ({coverage})")
    listed = ", ".join(pieces_named)
    raise refusal.RefusalError(
        f"the pieces of point {point_id} recorded confidence for different "
        f"shots (confidence_shot_count): {listed}; folding them would "
        "leave the confidence files short of the shots shots.csv counts, "
        "so collect the point again into a new experiment folder"
    )


def _refuse_the_folders(row: dict, run_dirs: list) -> None:
    """Say which shot is doubled and in which folders it was found."""
    listed = _named(run_dirs)
    shot = f"{row['point_id']} seed {row['seed']}"
    raise refusal.RefusalError(
        f"the shot {shot} is in more than one of {listed}; a shot is one "
        "seeded run of one sweep point, so folding both folders would "
        "count it twice"
    )


def _named(run_dirs: list) -> str:
    """The folders as one comma-separated list, for a refusal."""
    names = []
    for run_dir in run_dirs:
        names.append(str(run_dir))
    return ", ".join(names)


def _row_task_and_seed(positions: dict, row: dict) -> tuple:
    """One per-shot row's place: its point's task position, then its seed.

    It is the order a single run wrote its rows in, so the folders'
    rows merge back into that order and a point's several rows for one
    seed (one per decoded window, or one per link) keep the order the
    run wrote them.
    """
    position = positions[row["point_id"]]
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


def _shot_link_row(point: tuple, measurement, path: str) -> dict:
    """One shot's ledger counters on one link."""
    row = point_columns(point)
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
    from the multiset of every decoded window of the point.
    """
    row[f"{name}_mean_us"] = totals.mean(f"{name}_mean_us")
    row[f"{name}_median_us"] = percentile_of_counts(multiset, 0.50)
    row[f"{name}_p99_us"] = percentile_of_counts(multiset, 0.99)
    row[f"{name}_max_us"] = totals.maxes[f"{name}_max_us"]


def _terminal_block(row: dict, values: dict) -> str:
    """One sweep point's terminal block, one labeled line per number.

    The block opens with the point's value at each swept path. The
    lines above the latency ones are columns every row has. The failures
    and their rate are of the scored shots, which the rate's label says,
    and the unscored shots are counted beside them. The latency lines
    are the points the row holds (_points_held), because a row folded
    from an older tree's folders holds only the points that tree
    measured.
    """
    algorithm = row["algorithm"]
    algorithm_text = _algorithm_text(algorithm)
    unscored_fraction = row["unscored_shots"] / row["shots"]
    rate_text = _terminal_rate_text(row)
    lines = []
    for path, value in values.items():
        lines.append(f"{path}: {value}")
    lines += [
        f"algorithm: {algorithm_text}",
        f"load (service per window / window inter-arrival): {row['load']:.2f}",
        f"logical failures: {row['logical_failures']} of "
        f"{row['scored_shots']} scored shots",
        f"logical error rate among scored shots: {rate_text}",
        f"unscored shots: {row['unscored_shots']} of {row['shots']} "
        f"({unscored_fraction:.3g})",
        f"throughput: {row['throughput_rounds_per_us']:.3f} rounds per us",
    ]
    latency_lines = _terminal_latency_lines(row)
    lines.extend(latency_lines)
    return "\n".join(lines)


def _terminal_rate_text(row: dict) -> str:
    """The estimate with its 95 percent limits and the prefix's state.

    A cap with no failure has its upper limit alone, and a point with
    no estimate, adaptive or unscored, says so.
    """
    state = row["state"]
    rate = row["logical_error_rate_estimate"]
    high = row["logical_error_rate_high"]
    if rate is not None:
        low = row["logical_error_rate_low"]
        return f"{rate:.3g}, 95% {low:.3g} to {high:.3g} ({state})"
    if high is not None:
        return f"below {high:.3g} at 95% ({state})"
    return f"none ({state})"


def _terminal_latency_lines(row: dict) -> list:
    """The block's latency lines, for the points the row holds.

    A point the row does not hold gets no line at all, not a line of
    zeros and not a placeholder, for the reason it gets no column
    either (_points_held): a zero here would say the windows took no
    time, when what happened is that nobody measured them. sinter
    prints the same way, a counter a file does not carry being simply
    absent from what the folded table shows
    (.pydeps/sinter/_data/_task_stats.py:71, custom_counts is a
    Counter[str]).
    """
    held = _points_held(row)
    lines = []
    if "queue_wait" in held:
        lines.append(f"queue wait, mean: {row['queue_wait_mean_us']:.3f} us")
    if "service" in held:
        lines.append(
            f"service time per window, mean: {row['service_mean_us']:.3f} us"
        )
    if "buffer0_ready_to_frame" in held:
        lines.append(
            f"ready to frame commit: median "
            f"{row['buffer0_ready_to_frame_median_us']:.3f} us, "
            f"p99 {row['buffer0_ready_to_frame_p99_us']:.3f} us"
        )
    return lines


def _algorithm_text(algorithm) -> str:
    """A named algorithm as its name, a latency card as its microseconds."""
    if isinstance(algorithm, str):
        return algorithm
    return f"{algorithm:g} us"


def _link_row(point: tuple, path: str, totals) -> dict:
    """One link's averaged counters at one sweep point."""
    transfers = totals.mean("transfers")
    payload_bits = totals.mean("payload_bits")
    bits_per_transfer = 0.0
    if transfers:
        bits_per_transfer = payload_bits / transfers
    row = point_columns(point)
    row["link"] = path
    row["transfers_per_shot"] = transfers
    row["payload_bits_per_shot"] = payload_bits
    row["bits_per_transfer"] = bits_per_transfer
    row["unknown_payload_transfers_per_shot"] = totals.mean(
        "unknown_payload_transfers"
    )
    row["queue_wait_us_per_shot"] = totals.mean("queue_wait_us")
    row["serialization_us_per_shot"] = totals.mean("serialization_us")
    row["propagation_us_per_shot"] = totals.mean("propagation_us")
    return row


def _paths_counted(counted: dict) -> list:
    """Every path one shot copied or moved along, in path order."""
    paths = set(counted["copies_by_path"])
    for path in counted["moves_by_path"]:
        paths.add(path)
    return sorted(paths)


def _shot_movement_row(
    point: tuple, measurement, counted: dict, path: str
) -> dict:
    """One shot's copies and moves on one path, plus its own references."""
    row = point_columns(point)
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


def _movement_row(
    point: tuple, grouping: str, name: str, movement: _PointMovement
) -> dict:
    """One point's mean over shots for one path or one memory class.

    The reference columns are the point's own, the same on every row: a
    reference belongs to the shot and not to a path, so it is counted
    over the point's shots and not over the rows of one path, which some
    of the point's shots may not have at all.
    """
    row = point_columns(point)
    row["grouping"] = grouping
    row["name"] = name
    group = movement.totals_of(grouping, name)
    shots = movement.by_shot.rows
    for counter in MOVEMENT_COUNTERS:
        total = group.sums[counter]
        row[f"{counter}_per_shot"] = total / shots
    for counter in REFERENCE_COUNTERS:
        held = movement.by_shot.sums[counter]
        row[f"{counter}_per_shot"] = held / shots
    return row


def _data_movement_of(report_dir: Path) -> list:
    """A folder's per-point data-movement rows, empty when it wrote none."""
    path = Path(report_dir) / "data_movement.csv"
    if not path.is_file():
        return []
    return read_rows(path)


def _movement_at_point(movement: list, row: dict) -> list:
    """The memory-class rows of one sweep point, in cost order."""
    point = sweep_point_of(row)
    found = []
    for movement_row in movement:
        if movement_row["grouping"] != "memory_class":
            continue
        if sweep_point_of(movement_row) == point:
            found.append(movement_row)
    return found


def _block_with_classes(block: str, at_point: list) -> str:
    """One point's block with its by-class bit lines under it."""
    if not at_point:
        return block
    lines = [block]
    copied = _class_bits_line("bits copied per shot", at_point, "copy_bits")
    lines.append(copied)
    moved = _class_bits_line("bits moved per shot", at_point, "move_bits")
    lines.append(moved)
    return "\n".join(lines)


def _class_bits_line(label: str, at_point: list, counter: str) -> str:
    """One line: a class and its bits per shot, the classes it crossed."""
    named = []
    for row in at_point:
        bits = row[f"{counter}_per_shot"]
        if bits:
            named.append(f"{row['name']} {bits:.0f}")
    if not named:
        return f"{label} by memory class: none"
    listed = ", ".join(named)
    return f"{label} by memory class: {listed}"


def _window_confidence_row(
    measurement: measure.ShotMeasurement,
    confidence: measure.ShotConfidence,
    window,
) -> dict:
    """One window of one shot, as window_confidence.csv writes it."""
    point = measured_point(measurement)
    operation_id, window_index = window.window_key
    row = point_columns(point)
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
    point = measured_point(measurement)
    failed = measurement.logical_failure
    for window in confidence.windows:
        key = _histogram_key(
            point, confidence.signal, WINDOW_HISTOGRAM, window.gap_nats
        )
        window_cell = key + (window.is_escalated, failed)
        _add_count(counts, window_cell, 1)
    smallest = _smallest_gap(confidence.windows)
    shot_key = _histogram_key(
        point, confidence.signal, SHOT_MINIMUM_HISTOGRAM, smallest
    )
    shot_cell = shot_key + (None, failed)
    _add_count(counts, shot_cell, 1)


def _has_counted_confidence(measurement: measure.ShotMeasurement) -> bool:
    """Whether the shot's gaps belong in the confidence files.

    Only a scored shot's: an unscored shot is sinter's discard, whose
    logical_failure reads False, and sinter keeps a discard out of every
    count it conditions on failure (sinter/_decoding/_decoding.py:
    120-128) as out of the rate (sinter/_plotting.py:389). Every row of
    both files carries shot_failed, so a row of an unscored shot would
    count as a success. sweep.csv's unscored_shots counts the shots
    left out.
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


def _histogram_key(point: tuple, signal: str, kind: str, gap_nats) -> tuple:
    """A gap's place in the histogram: point, signal, kind and bin.

    A window with no gap has the bin None, written empty.
    """
    low = None
    if gap_nats is not None:
        low = gap_bin_low_decibels(gap_nats)
    return (point, signal, kind, low)


def _add_count(counts: dict, key: tuple, count: int) -> None:
    """Add to one histogram cell."""
    already = counts.get(key, 0)
    counts[key] = already + count


def _points_of_counts(counts: dict) -> list:
    """The points the counts name, in the order they first came."""
    points = {}
    for key in counts:
        points[key[0]] = None
    return list(points)


def _confidence_histogram_rows_of(counts: dict, points: list) -> list:
    """The counts as rows: points' order, kind, bin, escalated, failed."""
    positions = _task_positions(points)
    order = functools.partial(_histogram_order, positions)
    rows = []
    for key in sorted(counts, key=order):
        point, signal, kind, low, escalated, failed = key
        row = point_columns(point)
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
    point, signal, kind, low, escalated, failed = key
    point_order = positions[point]
    kind_order = kind != WINDOW_HISTOGRAM
    low_order = _bin_order(low)
    escalated_order = escalated is True
    return (point_order, signal, kind_order, low_order, escalated_order, failed)


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
    point = sweep_point_of(row)
    low = _empty_as_none(row["gap_low_decibels"])
    escalated = _empty_as_none(row["escalated"])
    key = (
        point,
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
