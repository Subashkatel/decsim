"""Shot measurements -> one row per sweep point -> sweep.csv and links.csv.

The summary pools every decoded window of a point's shots for its medians
and p99s, and averages the per-shot means for its means; the logical error
rate carries its Wilson 95% interval. Nothing here runs a simulation.
"""

import csv
import math
import statistics
from pathlib import Path
from typing import Optional

import decsim.front.measure as measure

BULKY_FIELDS = ("samples", "means", "maxes", "link_totals")
# The files combine folds, and how each one is put back in the order a
# single run would have written it: the summaries are written point by
# point in sweep-point order (summarize, link_rows), the per-shot files
# in the order the tasks ran, which is the shards taken in turn.
POINT_ORDERED_FILES = ("sweep.csv", "links.csv")
TASK_ORDERED_FILES = ("shots.csv", "latency_samples.csv")


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


def percentile(values: list, fraction: float) -> float:
    """The value at `fraction` of the sorted samples, nearest rank."""
    if not values:
        return 0.0
    ordered = sorted(values)
    last_index = len(ordered) - 1
    position = fraction * last_index
    rounded = round(position)
    index = min(last_index, int(rounded))
    return ordered[index]


def sweep_point_of(measurement) -> tuple:
    """The point a shot belongs to: distance, p, algorithm, round period."""
    return (
        measurement.distance,
        measurement.physical_error_probability,
        measurement.algorithm,
        measurement.round_period_us,
    )


def grouped_by_sweep_point(measurements: list) -> list:
    """(sweep point, that point's shots) pairs, in a stable order."""
    points = set()
    for measurement in measurements:
        point = sweep_point_of(measurement)
        points.add(point)
    ordered_points = sorted(points, key=_sweep_point_order)
    groups = []
    for point in ordered_points:
        group = _shots_at_point(measurements, point)
        groups.append((point, group))
    return groups


def summarize_point(group: list) -> dict:
    """One sweep point: means over seeds of per-shot means, max of maxes."""
    point = sweep_point_of(group[0])
    distance, physical_error_probability, algorithm, round_period_us = point
    failures = _count_true(group, "logical_failure")
    shots = len(group)
    ler_low, ler_high = wilson_interval(failures, shots)
    row = {
        "distance": distance,
        "physical_error_probability": physical_error_probability,
        "algorithm": algorithm,
        "round_period_us": round_period_us,
        "shots": shots,
        "windows_per_shot": _mean_of(group, "windows"),
        "logical_failures": failures,
        "logical_error_rate": failures / shots,
        "ler_wilson_low": ler_low,
        "ler_wilson_high": ler_high,
        "direct_pymatching_failures": _count_true(group, "direct_failure"),
        "prediction_mismatches_vs_direct": _count_true(
            group, "direct_mismatch"
        ),
        "throughput_windows_per_us": _mean_of(
            group, "throughput_windows_per_us"
        ),
        "throughput_rounds_per_us": _mean_of(group, "throughput_rounds_per_us"),
        "max_queued_windows": _max_of(group, "max_queued_windows"),
        "tesseract_windows_checked": _sum_of(
            group, "tesseract_windows_checked"
        ),
        "tesseract_window_disagreements": _sum_of(
            group, "tesseract_window_disagreements"
        ),
        "load": _mean_of(group, "load"),
        "sim_wall_seconds_per_shot": _mean_of(group, "sim_wall_seconds"),
    }
    for name in measure.POINTS:
        _add_point_columns(row, group, name)
    return row


def summarize(measurements: list) -> list:
    """One row per sweep point, in a stable order."""
    rows = []
    for _point, group in grouped_by_sweep_point(measurements):
        row = summarize_point(group)
        rows.append(row)
    return rows


def write_csv(rows: list, path: Path) -> None:
    """The rows as a csv file, the first row's keys as the header."""
    field_names = list(rows[0])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def terminal_lines(rows: list) -> list:
    """The terminal summary: one labeled block per sweep point.

    Full names, no abbreviations; the full record is sweep.csv.
    """
    blocks = []
    for row in rows:
        block = _terminal_block(row)
        blocks.append(block)
    joined_blocks = "\n\n".join(blocks)
    return joined_blocks.split("\n")


def link_rows(measurements: list) -> list:
    """One row per sweep point per link, averaged over the point's shots.

    The totals come straight off each run's TrafficCounters; nothing
    here re-counts transfers.
    """
    rows = []
    for point, group in grouped_by_sweep_point(measurements):
        paths = sorted(group[0].link_totals)
        for path in paths:
            row = _link_row(point, group, path)
            rows.append(row)
    return rows


def shot_rows(measurements: list) -> list:
    """One row per shot: every scalar field plus each point's mean.

    Any aggregate can then be re-cut without rerunning the sweep.
    """
    rows = []
    for measurement in measurements:
        row = _scalar_fields(measurement)
        for name in measure.POINTS:
            row[f"{name}_mean_us"] = measurement.means[name]
        rows.append(row)
    return rows


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


def read_rows(path: Path) -> list:
    """One csv file's rows, each value back as the number it was written."""
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            typed = _typed_row(row)
            rows.append(typed)
    return rows


def combine(run_dirs: list, out_dir: Path) -> list:
    """Fold several run folders' rows into one report, and return its rows.

    A shard runs whole tasks (decsim/collect.py shard_of), so two shards
    of one sweep never hold the same point and folding them is a
    concatenation, not a re-aggregation: sinter's combine adds rows with
    the same strong id, and here no two rows share one. A point that
    does appear twice is refused, because summing two summaries of the
    same point is not the summary of their shots.
    """
    summaries = _rows_of_every_folder(run_dirs, "sweep.csv")
    combined = _in_sweep_point_order(summaries)
    _refuse_a_repeated_point(combined, run_dirs)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in POINT_ORDERED_FILES:
        rows = _rows_of_every_folder(run_dirs, name)
        if not rows:
            continue
        ordered = _in_sweep_point_order(rows)
        _write_combined(ordered, out_dir, name)
    for name in TASK_ORDERED_FILES:
        by_folder = _rows_by_folder(run_dirs, name)
        ordered = _in_task_order(by_folder)
        if not ordered:
            continue
        _write_combined(ordered, out_dir, name)
    _refuse_a_repeated_point(combined, run_dirs)
    return combined


def write_report(
    rows: list, report_dir: Path, measurements: Optional[list] = None
) -> None:
    """sweep.csv per point, and the per-shot files when measurements come.

    shots.csv is one row per shot, links.csv the per-link ledger totals
    and latency_samples.csv one row per decoded window of a wall-clock
    algorithm.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = report_dir / "sweep.csv"
    write_csv(rows, sweep_path)
    if not measurements:
        return
    shots_path = report_dir / "shots.csv"
    per_shot = shot_rows(measurements)
    write_csv(per_shot, shots_path)
    links_path = report_dir / "links.csv"
    per_link = link_rows(measurements)
    write_csv(per_link, links_path)
    samples = latency_sample_rows(measurements)
    if samples:
        samples_path = report_dir / "latency_samples.csv"
        write_csv(samples, samples_path)


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


def _write_combined(rows: list, out_dir: Path, name: str) -> None:
    """One folded file, written where the combined report goes."""
    path = out_dir / name
    write_csv(rows, path)


def _rows_by_folder(run_dirs: list, name: str) -> list:
    """One file's rows per folder, in the order the folders were named."""
    by_folder = []
    for run_dir in run_dirs:
        path = Path(run_dir) / name
        if not path.is_file():
            continue
        rows = read_rows(path)
        by_folder.append(rows)
    return by_folder


def _in_task_order(by_folder: list) -> list:
    """The folders' per-shot rows back in the order the tasks ran.

    Shard i holds the tasks whose index modulo n is i, so taking one
    task from each shard in turn, again and again, is the sweep's own
    task order.
    """
    groups_by_folder = []
    for rows in by_folder:
        groups = _point_groups(rows)
        groups_by_folder.append(groups)
    ordered = []
    task_count = _most_tasks_in_a_shard(groups_by_folder)
    for position in range(task_count):
        _extend_with_one_task_each(ordered, groups_by_folder, position)
    return ordered


def _most_tasks_in_a_shard(groups_by_folder: list) -> int:
    """How many tasks the fullest shard ran; zero when there are none."""
    counts = [0]
    for groups in groups_by_folder:
        counts.append(len(groups))
    return max(counts)


def _extend_with_one_task_each(
    ordered: list, groups_by_folder: list, position: int
) -> None:
    """One round of the interleave: each shard's task at this position."""
    for groups in groups_by_folder:
        if position < len(groups):
            ordered.extend(groups[position])


def _point_groups(rows: list) -> list:
    """One folder's rows cut into runs of one sweep point, order kept."""
    groups = []
    current = []
    current_point = None
    for row in rows:
        point = _row_point_order(row)
        if point != current_point and current:
            groups.append(current)
            current = []
        current_point = point
        current.append(row)
    if current:
        groups.append(current)
    return groups


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


def _in_sweep_point_order(rows: list) -> list:
    """Csv rows sorted the way a single run would have written them."""
    return sorted(rows, key=_row_point_order)


def _row_point_order(row: dict) -> tuple:
    """A csv row's sweep point, in the order summarize writes points."""
    point = (
        row["distance"],
        row["physical_error_probability"],
        row["algorithm"],
        row["round_period_us"],
    )
    return _sweep_point_order(point)


def _refuse_a_repeated_point(rows: list, run_dirs: list) -> None:
    """A point in two folders means these are not shards of one sweep."""
    seen = set()
    for row in rows:
        point = _row_point_order(row)
        if point in seen:
            _refuse_the_folders(row, run_dirs)
        seen.add(point)


def _refuse_the_folders(row: dict, run_dirs: list) -> None:
    """Say which point is doubled and in which folders it was found."""
    names = []
    for run_dir in run_dirs:
        names.append(str(run_dir))
    listed = ", ".join(names)
    point = (
        f"d {row['distance']}, p {row['physical_error_probability']}, "
        f"algorithm {row['algorithm']}, "
        f"round period {row['round_period_us']} us"
    )
    raise ValueError(
        f"the point {point} is in more than one of {listed}; combine folds "
        "shards of one sweep, and two shards never run the same point"
    )


def _shots_at_point(measurements: list, point: tuple) -> list:
    """Every shot of one sweep point, in the order it was measured."""
    group = []
    for measurement in measurements:
        if sweep_point_of(measurement) == point:
            group.append(measurement)
    return group


def _scalar_fields(measurement) -> dict:
    """The measurement's own fields, the per-window collections left out."""
    row = {}
    for name in measurement.__dataclass_fields__:
        if name in BULKY_FIELDS:
            continue
        row[name] = getattr(measurement, name)
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


def _count_true(group: list, field: str) -> int:
    """How many of the point's shots have that flag set."""
    count = 0
    for measurement in group:
        flag = getattr(measurement, field)
        count += bool(flag)
    return count


def _sum_of(group: list, field: str):
    """The field summed over the point's shots."""
    total = 0
    for measurement in group:
        total += getattr(measurement, field)
    return total


def _mean_of(group: list, field: str) -> float:
    """The field averaged over the point's shots."""
    values = []
    for measurement in group:
        value = getattr(measurement, field)
        values.append(value)
    return statistics.fmean(values)


def _max_of(group: list, field: str):
    """The field's largest value over the point's shots."""
    values = []
    for measurement in group:
        value = getattr(measurement, field)
        values.append(value)
    return max(values)


def _add_point_columns(row: dict, group: list, name: str) -> None:
    """One latency point's mean, median, p99 and max columns.

    The mean averages the per-shot means; the median and p99 come from
    every decoded window of every shot pooled.
    """
    pooled = []
    per_shot_means = []
    per_shot_maxes = []
    for measurement in group:
        pooled.extend(measurement.samples[name])
        per_shot_means.append(measurement.means[name])
        per_shot_maxes.append(measurement.maxes[name])
    row[f"{name}_mean_us"] = statistics.fmean(per_shot_means)
    row[f"{name}_median_us"] = percentile(pooled, 0.50)
    row[f"{name}_p99_us"] = percentile(pooled, 0.99)
    row[f"{name}_max_us"] = max(per_shot_maxes)


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


def _link_row(point: tuple, group: list, path: str) -> dict:
    """One link's averaged counters at one sweep point."""
    distance, physical_error_probability, algorithm, round_period_us = point
    per_shot = []
    for measurement in group:
        per_shot.append(measurement.link_totals[path])
    transfers = _mean_of_counter(per_shot, "transfers")
    payload_bits = _mean_of_counter(per_shot, "payload_bits")
    bits_per_transfer = 0.0
    if transfers:
        bits_per_transfer = payload_bits / transfers
    return {
        "distance": distance,
        "physical_error_probability": physical_error_probability,
        "algorithm": algorithm,
        "round_period_us": round_period_us,
        "link": path,
        "transfers_per_shot": transfers,
        "payload_bits_per_shot": payload_bits,
        "bits_per_transfer": bits_per_transfer,
        "unknown_payload_transfers_per_shot": _mean_of_counter(
            per_shot, "unknown_payload_transfers"
        ),
        "queue_wait_us_per_shot": _mean_of_counter(per_shot, "queue_wait_us"),
        "serialization_us_per_shot": _mean_of_counter(
            per_shot, "serialization_us"
        ),
        "propagation_us_per_shot": _mean_of_counter(per_shot, "propagation_us"),
    }


def _mean_of_counter(per_shot: list, counter: str) -> float:
    """One ledger counter averaged over a point's shots."""
    values = []
    for totals in per_shot:
        values.append(totals[counter])
    return statistics.fmean(values)
