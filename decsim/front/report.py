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
