r"""The gap thresholds that escalate each target share, from the gap run.

    python calibrate.py RESULTS

RESULTS is gap_calibration.py's folded folder. Its
confidence_histogram.csv counts every window's cluster gap in bins of
0.1 dB (report.confidence_histogram_rows). A switching run escalates a
window whose gap is below the threshold, and a window with no gap at
any threshold (threshold_sources.FixedThreshold, records/decoding.py
WindowConfidence). The gap takes only multiples of union-find's weight
step, 0.5 nats or 2.17 dB, so the share a threshold escalates is a
staircase. For each target share this takes the step whose share is
nearest the target and puts the threshold halfway to the step above, so
a gap on the step escalates and the next one does not. It writes
calibration.csv into RESULTS and prints it; calibration.py holds what it
printed.
"""

import collections
import csv
import dataclasses
import functools
import itertools
import math
import pathlib
import sys

import decsim.experiments.report as report

TARGET_SHARES = (0.05, 0.1, 0.2)
# two steps closer than two bins could share a bin edge, and the bins
# could not tell which side of a threshold between them a gap falls
SMALLEST_STEP_DECIBELS = 2 / report.CONFIDENCE_BINS_PER_DECIBEL
PROBABILITY_COLUMN = "workload.arguments.physical_error_probability"
COLUMNS = (
    "distance",
    "physical_error_probability",
    "target_share",
    "threshold_decibels",
    "escalated_share",
    "windows",
)


@dataclasses.dataclass(frozen=True)
class EscalationStep:
    """A threshold between two gap steps, and the share it escalates."""

    threshold_decibels: float
    escalated_share: float
    windows: int


def main(results: pathlib.Path) -> None:
    """Write and print each point's threshold for each target share."""
    histogram_path = results / "confidence_histogram.csv"
    rows = _read(histogram_path)
    counts_by_point = window_gap_counts(rows)
    calibrated = []
    for point, gap_counts in counts_by_point.items():
        steps = escalation_steps(gap_counts)
        for target in TARGET_SHARES:
            row = calibrated_row(point, steps, target)
            calibrated.append(row)
    calibration_path = results / "calibration.csv"
    _write(calibrated, calibration_path)
    for row in calibrated:
        print(row)


def window_gap_counts(rows: list) -> dict:
    """Per (distance, p), the windows at each gap bin's low edge in dB.

    A window with no gap counts under None.
    """
    counts_by_point = collections.defaultdict(collections.Counter)
    for row in rows:
        if row["histogram"] != report.WINDOW_HISTOGRAM:
            continue
        distance = int(row["qpu.distance"])
        probability = float(row[PROBABILITY_COLUMN])
        point = (distance, probability)
        gap_low = _gap_low(row["gap_low_decibels"])
        counts_by_point[point][gap_low] += int(row["count"])
    return counts_by_point


def escalation_steps(gap_counts: collections.Counter) -> list:
    """Each threshold between two gap steps, lowest first.

    The first is 0 dB, which escalates only the windows with no gap.
    """
    windows = gap_counts.total()
    escalated = gap_counts[None]
    share = escalated / windows
    first = EscalationStep(0.0, share, windows)
    steps = [first]
    lows = sorted(low for low in gap_counts if low is not None)
    for low, next_low in itertools.pairwise(lows):
        if math.isinf(next_low):
            break
        _check_steps_apart(low, next_low)
        escalated += gap_counts[low]
        threshold = (low + next_low) / 2
        share = escalated / windows
        step = EscalationStep(threshold, share, windows)
        steps.append(step)
    return steps


def calibrated_row(point: tuple, steps: list, target: float) -> dict:
    """The step whose share is nearest the target, as one row."""
    share_error = functools.partial(_share_error, target)
    nearest = min(steps, key=share_error)
    distance, probability = point
    return {
        "distance": distance,
        "physical_error_probability": probability,
        "target_share": target,
        "threshold_decibels": round(nearest.threshold_decibels, 2),
        "escalated_share": round(nearest.escalated_share, 4),
        "windows": nearest.windows,
    }


def _share_error(target: float, step: EscalationStep) -> float:
    error = step.escalated_share - target
    return abs(error)


def _check_steps_apart(low: float, next_low: float) -> None:
    if next_low - low >= SMALLEST_STEP_DECIBELS:
        return
    raise ValueError(
        f"gap bins at {low} and {next_low} dB are closer than "
        f"{SMALLEST_STEP_DECIBELS} dB, so a threshold between them cannot "
        "be placed from 0.1 dB bins"
    )


def _gap_low(written: str):
    if written == "":
        return None
    return float(written)


def _read(path: pathlib.Path) -> list:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _write(rows: list, path: pathlib.Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    results_folder = pathlib.Path(sys.argv[1])
    main(results_folder)
