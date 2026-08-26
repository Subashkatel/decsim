"""Shot measurements -> one row per sweep point -> sweep.csv and sweep.md.

The summary pools every decoded window of a point's shots for its medians
and p99s, and averages the per-shot means for its means; the logical error
rate carries its Wilson 95% interval. Nothing here runs a simulation.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

from experiments.measure_shot import POINTS, ShotMeasurement


def wilson_interval(failures: int, shots: int, z: float = 1.96) -> tuple:
    """Wilson 95% confidence interval for a failure fraction."""
    if shots == 0:
        return (0.0, 0.0)
    fraction = failures / shots
    denominator = 1 + z * z / shots
    center = (fraction + z * z / (2 * shots)) / denominator
    half_width = z * math.sqrt(fraction * (1 - fraction) / shots
                               + z * z / (4 * shots * shots)) / denominator
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def percentile(values: list, fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def sweep_point_of(measurement: ShotMeasurement) -> tuple:
    return (measurement.physical_error_probability,
            measurement.algorithm_latency_us,
            measurement.round_period_us)


def summarize_point(group: list) -> dict:
    """One sweep point: means over seeds of the per-shot means, maxes of maxes."""
    physical_error_probability, algorithm_latency_us, round_period_us = \
        sweep_point_of(group[0])
    failures = sum(m.logical_failure for m in group)
    ler_low, ler_high = wilson_interval(failures, len(group))
    row = {"physical_error_probability": physical_error_probability,
           "algorithm_latency_us": algorithm_latency_us,
           "round_period_us": round_period_us,
           "shots": len(group),
           "windows_per_shot": statistics.fmean(m.windows for m in group),
           "logical_failures": failures,
           "logical_error_rate": failures / len(group),
           "ler_wilson_low": ler_low,
           "ler_wilson_high": ler_high,
           "direct_pymatching_failures": sum(m.direct_failure for m in group),
           "prediction_mismatches_vs_direct": sum(m.direct_mismatch for m in group),
           "backlog_trajectory": group[0].backlog,
           "throughput_windows_per_us": statistics.fmean(m.throughput_windows_per_us for m in group),
           "throughput_rounds_per_us": statistics.fmean(m.throughput_rounds_per_us for m in group),
           "decoder_utilization": statistics.fmean(m.decoder_utilization for m in group),
           "max_queued_windows": max(m.max_queued_windows for m in group),
           "load": statistics.fmean(m.load for m in group),
           "sim_wall_seconds_per_shot": statistics.fmean(m.sim_wall_seconds for m in group)}
    for point in POINTS:
        pooled = []
        for measurement in group:
            pooled.extend(measurement.samples[point])
        row[f"{point}_mean_us"] = statistics.fmean(m.means[point] for m in group)
        row[f"{point}_median_us"] = percentile(pooled, 0.50)
        row[f"{point}_p99_us"] = percentile(pooled, 0.99)
        row[f"{point}_max_us"] = max(m.maxes[point] for m in group)
    return row


def summarize(measurements: list) -> list:
    """One row per sweep point, in a stable order."""
    sweep_points = sorted({sweep_point_of(m) for m in measurements},
                          key=lambda point: (point[0], str(point[1]), -point[2]))
    rows = []
    for sweep_point in sweep_points:
        group = [m for m in measurements if sweep_point_of(m) == sweep_point]
        rows.append(summarize_point(group))
    return rows


def write_csv(rows: list, path: Path) -> None:
    """Every scalar column; the backlog trajectory is a list and stays out."""
    scalar_rows = [{key: value for key, value in row.items()
                    if key != "backlog_trajectory"} for row in rows]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)


def table_lines(rows: list) -> list:
    head = (["p", "algo us", "round us", "rate MHz", "load", "LER",
             "fails/shots", "direct fails", "mismatch vs direct", "win/us",
             "rounds/us", "util", "max q"] + list(POINTS)
            + ["ready->frame median", "ready->frame p99"])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in rows:
        algorithm = row["algorithm_latency_us"]
        cells = [f"{row['physical_error_probability']:g}",
                 algorithm if isinstance(algorithm, str) else f"{algorithm:g}",
                 f"{row['round_period_us']:g}",
                 f"{1 / row['round_period_us']:g}",
                 f"{row['load']:.2f}",
                 f"{row['logical_error_rate']:.3f}",
                 f"{row['logical_failures']}/{row['shots']}",
                 f"{row['direct_pymatching_failures']}",
                 f"{row['prediction_mismatches_vs_direct']}",
                 f"{row['throughput_windows_per_us']:.4f}",
                 f"{row['throughput_rounds_per_us']:.3f}",
                 f"{row['decoder_utilization']:.3f}",
                 f"{row['max_queued_windows']}"]
        cells += [f"{row[f'{point}_mean_us']:.3f}" for point in POINTS]
        cells.append(f"{row['buffer0_ready_to_frame_median_us']:.3f}")
        cells.append(f"{row['buffer0_ready_to_frame_p99_us']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def terminal_lines(rows: list) -> list:
    """The terminal summary: one labeled block per sweep point, full names,
    no abbreviations. The full record is sweep.csv and sweep.md."""
    blocks = []
    for row in rows:
        algorithm = row["algorithm_latency_us"]
        algorithm_text = (algorithm if isinstance(algorithm, str)
                          else f"{algorithm:g} us")
        blocks.append("\n".join([
            f"physical error rate: {row['physical_error_probability']:g}",
            f"algorithm latency: {algorithm_text}",
            f"round period: {row['round_period_us']:g} us",
            f"load (service per window / window inter-arrival): {row['load']:.2f}",
            f"logical failures: {row['logical_failures']} of {row['shots']} shots",
            f"mismatches vs direct PyMatching: {row['prediction_mismatches_vs_direct']}",
            f"throughput: {row['throughput_rounds_per_us']:.3f} rounds per us",
            f"decoder utilization (fraction of the run a unit computes): "
            f"{row['decoder_utilization']:.3f}",
            f"queue wait, mean: {row['queue_wait_mean_us']:.3f} us",
            f"service time per window, mean: {row['service_mean_us']:.3f} us",
            f"ready to frame commit: median "
            f"{row['buffer0_ready_to_frame_median_us']:.3f} us, "
            f"p99 {row['buffer0_ready_to_frame_p99_us']:.3f} us"]))
    return "\n\n".join(blocks).split("\n")


def write_report(rows: list, report_dir: Path) -> None:
    """sweep.csv (every column) and sweep.md (the table)."""
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, report_dir / "sweep.csv")
    (report_dir / "sweep.md").write_text("\n".join(table_lines(rows)) + "\n")
