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
one process ran every shot or a Slurm array ran a seed range each, and
`decsim combine` builds them through this module's own summarize and
link_rows.

The per-value counts file stays small because every sample is a whole
number of ticks divided by the ticks in a microsecond
(measure.ticks_to_microseconds). Under a fixed-latency decoder card,
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
import statistics
import sys
from pathlib import Path
from typing import Optional

import decsim.front.experiment as experiment
import decsim.front.measure as measure
import decsim.front.refusal as refusal
import decsim.front.run_folder as run_folder
import decsim.observe.data_movement as data_movement

BULKY_FIELDS = (
    "samples",
    "means",
    "maxes",
    "link_totals",
    "data_movement",
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


@dataclasses.dataclass(frozen=True)
class RunRecord:
    """The additive facts one or more run folders hold.

    Each field is a list of csv rows and each list is one file of the
    run folder: shots.csv, shot_links.csv, shot_data_movement.csv,
    window_samples.csv and latency_samples.csv. No field holds a
    summary, so two folders' records join by concatenation (the window
    samples' counts add), and the summaries derived from the join are
    the summaries a single run over the same shots would have written.
    """

    shots: list
    shot_links: list
    shot_data_movement: list
    window_samples: list
    latency_samples: list


def wilson_interval(failures: int, shots: int, z: float = 1.96) -> tuple:
    """Wilson 95% confidence interval for a failure fraction."""
    if shots == 0:
        return (0.0, 0.0)
    fraction = failures / shots
    denominator = 1 + z * z / shots
    center = fraction + z * z / (2 * shots)
    center = center / denominator
    variance = fraction * (1 - fraction) / shots
    continuity = z * z / (4 * shots * shots)
    inside_the_root = variance + continuity
    spread = math.sqrt(inside_the_root)
    half_width = z * spread / denominator
    low = center - half_width
    high = center + half_width
    return (max(0.0, low), min(1.0, high))


def percentile_of_counts(multiset: dict, fraction: float) -> float:
    """The value at `fraction` of the samples, nearest rank.

    multiset maps a microsecond value to how many samples carry it.
    Expanding it and sorting gives the list this walks in place, so a
    point's percentile is the same however its shots were split across
    processes or shards.
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
    """The point a row belongs to: distance, p, algorithm, round period."""
    return (
        row["distance"],
        row["physical_error_probability"],
        row["algorithm"],
        row["round_period_us"],
    )


def grouped_by_sweep_point(rows: list) -> list:
    """(sweep point, that point's rows) pairs, in a stable order."""
    points = set()
    for row in rows:
        point = sweep_point_of(row)
        points.add(point)
    ordered_points = sorted(points, key=_sweep_point_order)
    groups = []
    for point in ordered_points:
        group = _rows_at_point(rows, point)
        groups.append((point, group))
    return groups


def summarize_point(shots: list, counts: dict) -> dict:
    """One sweep point: means over seeds of per-shot means, max of maxes."""
    point = sweep_point_of(shots[0])
    distance, physical_error_probability, algorithm, round_period_us = point
    failures = _count_true(shots, "logical_failure")
    shot_count = len(shots)
    ler_low, ler_high = wilson_interval(failures, shot_count)
    row = {
        "distance": distance,
        "physical_error_probability": physical_error_probability,
        "algorithm": algorithm,
        "round_period_us": round_period_us,
        "shots": shot_count,
        "windows_per_shot": _mean_of(shots, "windows"),
        "logical_failures": failures,
        "logical_error_rate": failures / shot_count,
        "ler_wilson_low": ler_low,
        "ler_wilson_high": ler_high,
        "direct_pymatching_failures": _count_true(shots, "direct_failure"),
        "prediction_mismatches_vs_direct": _count_true(
            shots, "direct_mismatch"
        ),
        "throughput_windows_per_us": _mean_of(
            shots, "throughput_windows_per_us"
        ),
        "throughput_rounds_per_us": _mean_of(shots, "throughput_rounds_per_us"),
        "max_queued_windows": _max_of(shots, "max_queued_windows"),
        "tesseract_windows_checked": _sum_of(
            shots, "tesseract_windows_checked"
        ),
        "tesseract_window_disagreements": _sum_of(
            shots, "tesseract_window_disagreements"
        ),
        "load": _mean_of(shots, "load"),
        "sim_wall_seconds_per_shot": _mean_of(shots, "sim_wall_seconds"),
    }
    for name in measure.POINTS:
        multiset = counts.get((point, name), {})
        _add_point_columns(row, shots, name, multiset)
    return row


def summarize(shots: list, window_samples: list) -> list:
    """One row per sweep point, in a stable order.

    The two arguments are the additive files a run folder writes: the
    per-shot rows and the per-value counts. The same code runs whether
    they were just measured in this process or read back out of the
    folders `decsim combine` was given.
    """
    counts = _counts_of_rows(window_samples)
    rows = []
    for _point, group in grouped_by_sweep_point(shots):
        row = summarize_point(group, counts)
        rows.append(row)
    return rows


def write_csv(rows: list, path: Path) -> None:
    """The rows as a csv file, the first row's keys as the header."""
    field_names = list(rows[0])
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
    blocks = []
    for row in rows:
        block = _terminal_block(row)
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
    rows = []
    for point, group in grouped_by_sweep_point(shot_links):
        for path in _links_of(group):
            row = _link_row(point, group, path)
            rows.append(row)
    return rows


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
    rows = []
    for point, group in grouped_by_sweep_point(shot_data_movement):
        seeds = _seeds_of(group)
        for path in _named_values(group, "path"):
            at_path = _rows_named(group, "path", path)
            row = _movement_row(point, "path", path, at_path, seeds)
            rows.append(row)
        for memory_class in _classes_of(group):
            at_class = _rows_named(group, "memory_class", memory_class)
            row = _movement_row(
                point, "memory_class", memory_class, at_class, seeds
            )
            rows.append(row)
    return rows


def shot_data_movement_rows(measurements: list) -> list:
    """One row per shot per path: that shot's own copy and move counters.

    A path is a copy's source and target (`controller intake -> Buffer
    0`) or a link's own name, and each row names the memory class that
    path crosses, which observe/data_movement.py places from the
    sources. A shot whose run had observation.data_movement off writes
    no row at all: a run that counted nothing has no counts, which is
    not the same fact as a run whose counts were zero.
    """
    rows = []
    for measurement in measurements:
        counted = measurement.data_movement
        if counted is None:
            continue
        point = _measured_point(measurement)
        for path in _paths_counted(counted):
            row = _shot_movement_row(point, measurement, counted, path)
            rows.append(row)
    return rows


def shot_rows(measurements: list) -> list:
    """One row per shot: its scalars, then each point's mean and max.

    Any aggregate can then be re-cut without rerunning the sweep, and a
    point's mean and max columns fold over any set of shots.
    """
    rows = []
    for measurement in measurements:
        row = _scalar_fields(measurement)
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
        point = _measured_point(measurement)
        for path in sorted(measurement.link_totals):
            row = _shot_link_row(point, measurement, path)
            rows.append(row)
    return rows


def window_sample_rows(measurements: list) -> list:
    """One row per point, latency point and distinct value: its count.

    The multiset of a point's window samples, which is all its median
    and p99 columns need and all a shard has to record for another
    process to reach the same numbers.
    """
    counts = _counts_of_samples(measurements)
    return _rows_of_counts(counts)


def latency_sample_rows(measurements: list) -> list:
    """One row per decoded window: the measured algorithm wall clock.

    Only wall-clock algorithms (a name, not a latency card) produce
    rows; this is the latency figure's raw data, persisted so the
    figure, including the cross-tier combined one, rebuilds from run
    folders alone.
    """
    rows = []
    for measurement in measurements:
        if not isinstance(measurement.algorithm, str):
            continue
        for sample_us in measurement.samples["algorithm"]:
            row = {
                "distance": measurement.distance,
                "physical_error_probability": (
                    measurement.physical_error_probability
                ),
                "round_period_us": measurement.round_period_us,
                "algorithm": measurement.algorithm,
                "seed": measurement.seed,
                "algorithm_us": sample_us,
            }
            rows.append(row)
    return rows


def record_of(measurements: list) -> RunRecord:
    """The additive facts of the shots this process measured."""
    shots = shot_rows(measurements)
    links = shot_link_rows(measurements)
    movement = shot_data_movement_rows(measurements)
    samples = window_sample_rows(measurements)
    latency = latency_sample_rows(measurements)
    return RunRecord(shots, links, movement, samples, latency)


def write_record(record: RunRecord, report_dir: Path) -> None:
    """The record's five files; one with no rows is not written."""
    shots_path = report_dir / "shots.csv"
    _write_rows(record.shots, shots_path)
    links_path = report_dir / "shot_links.csv"
    _write_rows(record.shot_links, links_path)
    movement_path = report_dir / "shot_data_movement.csv"
    _write_rows(record.shot_data_movement, movement_path)
    samples_path = report_dir / "window_samples.csv"
    _write_rows(record.window_samples, samples_path)
    latency_path = report_dir / "latency_samples.csv"
    _write_rows(record.latency_samples, latency_path)


def read_rows(path: Path) -> list:
    """One csv file's rows, each value back as the number it was written."""
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            typed = _typed_row(row)
            rows.append(typed)
    return rows


def read_record(run_dirs: list, positions: dict) -> RunRecord:
    """Several folders' additive files as one record, in a run's order.

    The per-shot rows go back where the sweep says they belong, the
    task's place and then the seed, so the folder order and the shard
    indices play no part; the window samples' counts add.
    """
    shots = _rows_of_every_folder(run_dirs, "shots.csv")
    shot_links = _rows_of_every_folder(run_dirs, "shot_links.csv")
    movement = _rows_of_every_folder(run_dirs, "shot_data_movement.csv")
    samples = _rows_of_every_folder(run_dirs, "window_samples.csv")
    latency = _rows_of_every_folder(run_dirs, "latency_samples.csv")
    counts = _counts_of_rows(samples)
    ordered_shots = _in_task_and_seed_order(shots, positions)
    ordered_links = _in_task_and_seed_order(shot_links, positions)
    ordered_movement = _in_task_and_seed_order(movement, positions)
    ordered_samples = _rows_of_counts(counts)
    ordered_latency = _in_task_and_seed_order(latency, positions)
    return RunRecord(
        ordered_shots,
        ordered_links,
        ordered_movement,
        ordered_samples,
        ordered_latency,
    )


def combine(run_dirs: list, out_dir: Path) -> list:
    """Fold several run folders into one report, and return its rows.

    A run folder records only additive facts, so folding is reading
    them all back as one record and deriving the summaries from it
    through the same summarize and link_rows a single run uses. Serial,
    pooled and sharded runs of one config therefore write the same
    rows, along whatever cut `--shots-per-unit` gave the shards: a
    point split across two shards folds exactly, because its median and
    p99 come from the per-value counts and not from a shard's summary.

    The rows come back in the order one unsharded run would have
    written them, read off the rows themselves and the sweep every
    folder's manifest records, so `combine b a` writes what
    `combine a b` writes. What two folders may not share is a shot: one
    seeded run in both of them would be counted twice.

    The combined folder is itself a run folder: the additive files and a
    manifest recording the sweep every folded folder shares, so a shard
    that lands after the fold folds into it in turn, which is the shape
    a Slurm array finishing in waves has.
    """
    started_utc = run_folder.utc_now()
    recorded_config = _one_sweeps_config(run_dirs)
    positions = experiment.task_positions(recorded_config["sweep"])
    folders = _folders_that_ran_shots(run_dirs)
    record = read_record(folders, positions)
    _refuse_a_repeated_shot(record.shots, folders)
    rows = summarize(record.shots, record.window_samples)
    write_report(rows, out_dir, record)
    finished_utc = run_folder.utc_now()
    run_folder.write_combined_manifest(
        recorded_config,
        out_dir,
        run_dirs,
        started_utc,
        finished_utc=finished_utc,
    )
    return rows


def write_report(
    rows: list, report_dir: Path, record: Optional[RunRecord] = None
) -> None:
    """sweep.csv per point, and the additive files when a record comes.

    links.csv and data_movement.csv are derived from the record's
    per-shot rows, so a combined folder's files are built the way a
    single run's files were.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = report_dir / "sweep.csv"
    write_csv(rows, sweep_path)
    if record is None:
        return
    write_record(record, report_dir)
    per_link = link_rows(record.shot_links)
    links_path = report_dir / "links.csv"
    write_csv(per_link, links_path)
    per_movement = data_movement_rows(record.shot_data_movement)
    movement_path = report_dir / "data_movement.csv"
    _write_rows(per_movement, movement_path)


def _write_rows(rows: list, path: Path) -> None:
    """One file of the record, left unwritten when it has no rows."""
    if not rows:
        return
    write_csv(rows, path)


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
    """(point, latency point) -> value -> how many windows carried it."""
    counts = {}
    for measurement in measurements:
        point = _measured_point(measurement)
        for name, values in measurement.samples.items():
            at_this_name = counts.setdefault((point, name), {})
            _count_the_values(at_this_name, values)
    return counts


def _count_the_values(multiset: dict, values: list) -> None:
    """Add one shot's samples of one latency point to the multiset."""
    for value in values:
        already = multiset.get(value, 0)
        multiset[value] = already + 1


def _counts_of_rows(window_samples: list) -> dict:
    """The same multisets, read back off window_samples.csv rows."""
    counts = {}
    for row in window_samples:
        point = sweep_point_of(row)
        key = (point, row["name"])
        at_this_name = counts.setdefault(key, {})
        value = row["value_us"]
        already = at_this_name.get(value, 0)
        at_this_name[value] = already + row["count"]
    return counts


def _rows_of_counts(counts: dict) -> list:
    """The multisets as rows: point order, then point-list order, value."""
    rows = []
    for key in sorted(counts, key=_point_and_name_order):
        multiset = counts[key]
        for row in _rows_of_one_multiset(key, multiset):
            rows.append(row)
    return rows


def _rows_of_one_multiset(key: tuple, multiset: dict) -> list:
    """One latency point's counts at one sweep point, by rising value."""
    point, name = key
    columns = _point_columns(point)
    rows = []
    for value in sorted(multiset):
        row = dict(columns)
        row["name"] = name
        row["value_us"] = value
        row["count"] = multiset[value]
        rows.append(row)
    return rows


def _point_and_name_order(key: tuple) -> tuple:
    """A (point, latency point) key where a single run would write it."""
    point, name = key
    place = measure.POINTS.index(name)
    point_order = _sweep_point_order(point)
    return (point_order, place)


def _typed_row(row: dict) -> dict:
    """One csv row with its values read back as numbers and flags."""
    typed = {}
    for key, text in row.items():
        typed[key] = _typed_value(text)
    return typed


def _typed_value(text: str):
    """A csv field as the value it was written from.

    The algorithm column is a name or a latency card, so a field that
    parses as a number is a number and everything else is its text.
    """
    if text == "True":
        return True
    if text == "False":
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _one_sweeps_config(run_dirs: list) -> dict:
    """The resolved config every folded folder recorded.

    A shard's rows say which point they belong to, not where that point
    sits in the sweep, so the sweep order comes from the resolved config
    each folder's manifest records, and the combined folder records it
    again for the next fold. Folders that recorded different configs are
    refused: they are not shards of one sweep.
    """
    recorded_configs = []
    for run_dir in run_dirs:
        recorded = _manifest_config(run_dir)
        recorded_configs.append(recorded)
    _refuse_folders_of_different_sweeps(run_dirs, recorded_configs)
    return recorded_configs[0]


def _manifest_config(run_dir) -> dict:
    """One run folder's resolved config, as its manifest recorded it."""
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise refusal.RefusalError(
            f"{run_dir} has no manifest.json; combine reads the sweep a run "
            "folder records, so it can put the folded rows back in the "
            "order one unsharded run would have written them"
        )
    manifest_text = manifest_path.read_text()
    manifest = json.loads(manifest_text)
    return manifest["resolved_config"]


def _refuse_folders_of_different_sweeps(
    run_dirs: list, recorded_configs: list
) -> None:
    """Folders that ran different experiments are not one sweep's shards."""
    first_config = recorded_configs[0]
    first_dir = run_dirs[0]
    for run_dir, recorded in zip(run_dirs, recorded_configs):
        if recorded == first_config:
            continue
        raise refusal.RefusalError(
            f"{run_dir} ran a different experiment from {first_dir}; combine "
            "folds the shards of one sweep, and every shard of a sweep "
            "records the same resolved config"
        )


def _refuse_a_repeated_shot(shots: list, run_dirs: list) -> None:
    """One seeded run in two folders would be counted twice.

    A point may be split across folders now, since its summary is
    derived from additive rows, but a shot may not: the same seed of
    the same point is the same run.
    """
    seen = set()
    for row in shots:
        point = sweep_point_of(row)
        shot = (point, row["seed"])
        if shot in seen:
            _refuse_the_folders(row, run_dirs)
        seen.add(shot)


def _refuse_the_folders(row: dict, run_dirs: list) -> None:
    """Say which shot is doubled and in which folders it was found."""
    listed = _named(run_dirs)
    shot = (
        f"d {row['distance']}, p {row['physical_error_probability']}, "
        f"algorithm {row['algorithm']}, "
        f"round period {row['round_period_us']} us, seed {row['seed']}"
    )
    raise refusal.RefusalError(
        f"the shot {shot} is in more than one of {listed}; a shot is one "
        "seeded run of one sweep point, so folding both folders would "
        "count it twice"
    )


def _folders_that_ran_shots(run_dirs: list) -> list:
    """The folders with a shots.csv; one without is skipped, out loud.

    A shard whose index selected no work unit writes its manifest and
    no rows, which is not an error and is nothing to fold.
    """
    folders = []
    for run_dir in run_dirs:
        path = Path(run_dir) / "shots.csv"
        if path.is_file():
            folders.append(run_dir)
            continue
        _say_the_folder_is_skipped(run_dir)
    if not folders:
        _refuse_folders_without_shots(run_dirs)
    return folders


def _say_the_folder_is_skipped(run_dir) -> None:
    """One line on stderr for a folder that holds no shot."""
    print(
        f"decsim: {run_dir} has no shots.csv, so combine skips it; "
        "a shard whose index selected no work unit writes no rows",
        file=sys.stderr,
    )


def _refuse_folders_without_shots(run_dirs: list) -> None:
    """Nothing to fold: not one of the folders holds a shot."""
    listed = _named(run_dirs)
    raise refusal.RefusalError(
        f"none of {listed} holds a shots.csv, so there is nothing to "
        "combine; a run folder records its shots there, and a shard that "
        "selected no work unit records none"
    )


def _named(run_dirs: list) -> str:
    """The folders as one comma-separated list, for a refusal."""
    names = []
    for run_dir in run_dirs:
        names.append(str(run_dir))
    return ", ".join(names)


def _in_task_and_seed_order(rows: list, positions: dict) -> list:
    """Per-shot rows in the sweep's own order: the task, then the seed.

    Sorting is stable, so a point's several rows for one seed (one per
    decoded window, or one per link) keep the order the run wrote them.
    """
    order = functools.partial(_row_task_and_seed, positions)
    return sorted(rows, key=order)


def _row_task_and_seed(positions: dict, row: dict) -> tuple:
    """One per-shot row's place: its point's task position, then its seed."""
    point = (
        row["physical_error_probability"],
        row["distance"],
        row["round_period_us"],
    )
    position = positions[point]
    return (position, row["seed"])


def _rows_of_every_folder(run_dirs: list, name: str) -> list:
    """One file's rows from every folder that has it, folder order kept."""
    rows = []
    for run_dir in run_dirs:
        path = Path(run_dir) / name
        if not path.is_file():
            continue
        for row in read_rows(path):
            rows.append(row)
    return rows


def _rows_at_point(rows: list, point: tuple) -> list:
    """Every row of one sweep point, in the order it was written."""
    group = []
    for row in rows:
        if sweep_point_of(row) == point:
            group.append(row)
    return group


def _measured_point(measurement) -> tuple:
    """The sweep point one measured shot belongs to."""
    return (
        measurement.distance,
        measurement.physical_error_probability,
        measurement.algorithm,
        measurement.round_period_us,
    )


def _point_columns(point: tuple) -> dict:
    """The four columns that name a sweep point."""
    distance, physical_error_probability, algorithm, round_period_us = point
    return {
        "distance": distance,
        "physical_error_probability": physical_error_probability,
        "algorithm": algorithm,
        "round_period_us": round_period_us,
    }


def _scalar_fields(measurement) -> dict:
    """The measurement's own fields, the per-window collections left out."""
    row = {}
    for name in measurement.__dataclass_fields__:
        if name in BULKY_FIELDS:
            continue
        row[name] = getattr(measurement, name)
    return row


def _shot_link_row(point: tuple, measurement, path: str) -> dict:
    """One shot's ledger counters on one link."""
    row = _point_columns(point)
    row["seed"] = measurement.seed
    row["link"] = path
    counters = measurement.link_totals[path]
    for counter, value in counters.items():
        row[counter] = value
    return row


def _sweep_point_order(point: tuple) -> tuple:
    """distance, then p, then the algorithm's text, then the fastest round."""
    distance, physical_error_probability, algorithm, round_period_us = point
    algorithm_text = str(algorithm)
    return (
        distance,
        physical_error_probability,
        algorithm_text,
        -round_period_us,
    )


def _count_true(rows: list, field: str) -> int:
    """How many of the point's shots have that flag set."""
    count = 0
    for row in rows:
        flag = row[field]
        count += bool(flag)
    return count


def _sum_of(rows: list, field: str):
    """The field summed over the point's shots."""
    total = 0
    for row in rows:
        total += row[field]
    return total


def _mean_of(rows: list, field: str) -> float:
    """The field averaged over the rows, in the order they were written."""
    values = []
    for row in rows:
        values.append(row[field])
    return statistics.fmean(values)


def _max_of(rows: list, field: str):
    """The field's largest value over the rows."""
    values = []
    for row in rows:
        values.append(row[field])
    return max(values)


def _add_point_columns(
    row: dict, shots: list, name: str, multiset: dict
) -> None:
    """One latency point's mean, median, p99 and max columns.

    The mean averages the per-shot means and the max takes the largest
    per-shot max, both off the per-shot rows; the median and p99 come
    from the multiset of every decoded window of the point.
    """
    row[f"{name}_mean_us"] = _mean_of(shots, f"{name}_mean_us")
    row[f"{name}_median_us"] = percentile_of_counts(multiset, 0.50)
    row[f"{name}_p99_us"] = percentile_of_counts(multiset, 0.99)
    row[f"{name}_max_us"] = _max_of(shots, f"{name}_max_us")


def _terminal_block(row: dict) -> str:
    """One sweep point's terminal block, one labeled line per number."""
    algorithm = row["algorithm"]
    algorithm_text = _algorithm_text(algorithm)
    lines = [
        f"distance: {row['distance']}",
        f"physical error rate: {row['physical_error_probability']:g}",
        f"algorithm: {algorithm_text}",
        f"round period: {row['round_period_us']:g} us",
        f"load (service per window / window inter-arrival): {row['load']:.2f}",
        f"logical failures: {row['logical_failures']} of {row['shots']} shots",
        f"mismatches vs direct PyMatching: "
        f"{row['prediction_mismatches_vs_direct']}",
        f"throughput: {row['throughput_rounds_per_us']:.3f} rounds per us",
        f"queue wait, mean: {row['queue_wait_mean_us']:.3f} us",
        f"service time per window, mean: {row['service_mean_us']:.3f} us",
        f"ready to frame commit: median "
        f"{row['buffer0_ready_to_frame_median_us']:.3f} us, "
        f"p99 {row['buffer0_ready_to_frame_p99_us']:.3f} us",
    ]
    return "\n".join(lines)


def _algorithm_text(algorithm) -> str:
    """A named algorithm as its name, a latency card as its microseconds."""
    if isinstance(algorithm, str):
        return algorithm
    return f"{algorithm:g} us"


def _links_of(shot_links: list) -> list:
    """Every link these rows carry, in the order links.csv lists them."""
    paths = set()
    for row in shot_links:
        paths.add(row["link"])
    return sorted(paths)


def _link_row(point: tuple, shot_links: list, path: str) -> dict:
    """One link's averaged counters at one sweep point."""
    at_this_link = _rows_of_one_link(shot_links, path)
    transfers = _mean_of(at_this_link, "transfers")
    payload_bits = _mean_of(at_this_link, "payload_bits")
    bits_per_transfer = 0.0
    if transfers:
        bits_per_transfer = payload_bits / transfers
    row = _point_columns(point)
    row["link"] = path
    row["transfers_per_shot"] = transfers
    row["payload_bits_per_shot"] = payload_bits
    row["bits_per_transfer"] = bits_per_transfer
    row["unknown_payload_transfers_per_shot"] = _mean_of(
        at_this_link, "unknown_payload_transfers"
    )
    row["queue_wait_us_per_shot"] = _mean_of(at_this_link, "queue_wait_us")
    row["serialization_us_per_shot"] = _mean_of(
        at_this_link, "serialization_us"
    )
    row["propagation_us_per_shot"] = _mean_of(at_this_link, "propagation_us")
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
    row = _point_columns(point)
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


def _seeds_of(rows: list) -> list:
    """Every shot of one sweep point, in the order the rows were written."""
    seeds = []
    for row in rows:
        if row["seed"] not in seeds:
            seeds.append(row["seed"])
    return seeds


def _named_values(rows: list, column: str) -> list:
    """Every distinct value of one column, in its own sorted order."""
    values = set()
    for row in rows:
        values.add(row[column])
    return sorted(values)


def _classes_of(rows: list) -> list:
    """The memory classes these rows crossed, cheapest first.

    The order is observe/data_movement.py's own CLASS_ORDER, so the
    grouped table reads the way the counters group it.
    """
    crossed = _named_values(rows, "memory_class")
    listed = []
    for memory_class in data_movement.CLASS_ORDER:
        if memory_class.value in crossed:
            listed.append(memory_class.value)
    return listed


def _rows_named(rows: list, column: str, value) -> list:
    """The rows whose column holds that value, in the order written."""
    found = []
    for row in rows:
        if row[column] == value:
            found.append(row)
    return found


def _movement_row(
    point: tuple, grouping: str, name: str, rows: list, seeds: list
) -> dict:
    """One point's mean over shots for one path or one memory class."""
    row = _point_columns(point)
    row["grouping"] = grouping
    row["name"] = name
    shots = len(seeds)
    for counter in MOVEMENT_COUNTERS:
        total = _sum_of(rows, counter)
        row[f"{counter}_per_shot"] = total / shots
    for counter in REFERENCE_COUNTERS:
        total = _summed_once_per_shot(rows, counter)
        row[f"{counter}_per_shot"] = total / shots
    return row


def _summed_once_per_shot(rows: list, counter: str) -> int:
    """A whole-shot counter added up over the shots these rows carry.

    A reference belongs to the shot and not to a path, so it repeats on
    every row of that shot and only the first row of each is read.
    """
    seen = {}
    for row in rows:
        seed = row["seed"]
        if seed in seen:
            continue
        seen[seed] = row[counter]
    total = 0
    for value in seen.values():
        total += value
    return total


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


def _rows_of_one_link(shot_links: list, path: str) -> list:
    """Every shot's row on one link, in the order the shots ran."""
    rows = []
    for row in shot_links:
        if row["link"] == path:
            rows.append(row)
    return rows
