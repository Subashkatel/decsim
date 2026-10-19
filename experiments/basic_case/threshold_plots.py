"""The threshold sweep's figures and calibration table.

    python threshold_plots.py SWEEP_FOLDER PILOT_FOLDER

SWEEP_FOLDER is threshold_sweep.py's results, PILOT_FOLDER pilot.py's,
which ran weak alone and strong alone on the same seeds. Writes into
SWEEP_FOLDER/plots/:
  - threshold.png: switching's error rate per shot against the threshold,
    weak alone and strong alone as lines (Toshio et al. 2510.25222
    Fig. 17), a panel per p;
  - frontier.png: the error rate against the windows escalated, with the
    escalated share the chain sustains, q*, dotted, strong alone dashed
    and weak alone dash-dotted;
  - calibration.csv: per point, the threshold that escalates each target
    share, interpolated between the swept thresholds, and q*.

q* is where a window's mean service equals the d us between windows:
(d - S_keep) / (S_esc - S_keep), each tier's mean service the pilot's
data-ready-to-commit time less its wait for the boundary (the chain's
Lindley form, tests/escalation/test_redo_chain_queue.py). It reads the
belief-matching times in constants.yaml, which are placeholders.
"""

import collections
import csv
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROBABILITY = "workload.arguments.physical_error_probability"
DISTANCE = "qpu.distance"
TARGETS = (0.01, 0.02, 0.05, 0.1, 0.2)
COLORS = {5: "tab:blue", 7: "tab:orange", 9: "tab:green", 11: "tab:red"}
CALIBRATION_FIELDS = ("d", "p", "q_star", *(f"target_{t:g}" for t in TARGETS))


def main(sweep: pathlib.Path, pilot: pathlib.Path) -> None:
    """Both figures and the table."""
    folder = sweep / "plots"
    folder.mkdir(exist_ok=True)
    sweep_path = sweep / "sweep.csv"
    pilot_path = pilot / "sweep.csv"
    samples_path = pilot / "window_samples.csv"
    swept = _read(sweep_path)
    references = _read(pilot_path)
    samples = _read(samples_path)
    capacities = q_stars(samples)
    plot_threshold(swept, references, folder)
    plot_frontier(swept, references, capacities, folder)
    write_calibration(swept, capacities, folder)


def q_stars(samples: list) -> dict:
    """q* per (d, p), from the pilot's switching windows."""
    means = _tier_means(samples)
    capacities = {}
    for distance, probability, tier, name in means:
        if tier != "weak" or name != "dep_block":
            continue
        point = (distance, probability)
        keep = _service(means, point, "weak")
        escalated = _service(means, point, "strong")
        capacities[point] = (distance - keep) / (escalated - keep)
    return capacities


def _tier_means(samples: list) -> dict:
    """Mean us of the two latency points, by (d, p, tier, point)."""
    sums = collections.defaultdict(float)
    counts = collections.Counter()
    for row in samples:
        if row["decoding"] != "switching":
            continue
        if row["name"] not in ("buffer0_ready_to_frame", "dep_block"):
            continue
        key = (*_point(row), row["tier"], row["name"])
        count = int(row["count"])
        sums[key] += float(row["value_us"]) * count
        counts[key] += count
    return {key: sums[key] / counts[key] for key in sums}


def _service(means: dict, point: tuple, tier: str) -> float:
    """A tier's mean data-ready-to-commit less its mean boundary wait."""
    ready = means[(*point, tier, "buffer0_ready_to_frame")]
    wait = means[(*point, tier, "dep_block")]
    return ready - wait


def plot_threshold(swept: list, references: list, folder: pathlib.Path):
    """Fig. 17: error rate against threshold, the two runs alone as lines."""
    panels = _by_probability(swept)
    figure, axes_row = _panels(len(panels))
    panel_items = panels.items()
    pairs = zip(axes_row, panel_items, strict=True)
    for axes, (probability, by_distance) in pairs:
        for distance, rows in by_distance.items():
            point = (distance, probability)
            _draw_threshold(axes, point, rows, references)
        axes.set_xlabel("gap threshold (dB)")
        _finish(axes, f"p = {probability:g}")
    _save(figure, folder, "threshold.png")


def _draw_threshold(axes, point: tuple, rows: list, references: list):
    """One d's switching line and its weak and strong references."""
    distance, _probability = point
    color = COLORS[distance]
    ordered = sorted(rows, key=_decibels)
    thresholds = [_decibels(row) for row in ordered]
    rates = [_rate(row) for row in ordered]
    axes.plot(thresholds, rates, "o-", ms=3, color=color, label=f"d={distance}")
    for decoding, style in (("weak_alone", ":"), ("strong_alone", "--")):
        reference = _reference(references, point, decoding)
        rate = _rate(reference)
        axes.axhline(rate, color=color, ls=style, lw=1)


def plot_frontier(swept, references, capacities, folder: pathlib.Path):
    """Error rate against windows escalated, q* as a vertical line."""
    panels = _by_probability(swept)
    figure, axes_row = _panels(len(panels))
    panel_items = panels.items()
    pairs = zip(axes_row, panel_items, strict=True)
    for axes, (probability, by_distance) in pairs:
        for distance, rows in by_distance.items():
            point = (distance, probability)
            _draw_frontier(axes, point, rows, references, capacities)
        axes.set_xscale("log")
        axes.set_xlabel("windows escalated")
        _finish(axes, f"p = {probability:g}")
    _save(figure, folder, "frontier.png")


def _draw_frontier(axes, point, rows, references, capacities) -> None:
    """One d's frontier, its strong-alone rate and its q*."""
    distance, _probability = point
    color = COLORS[distance]
    # a threshold that escalates nothing is weak alone, a line of its own
    escalating = [row for row in rows if _escalated(row) > 0]
    ordered = sorted(escalating, key=_escalated)
    fractions = [_escalated(row) for row in ordered]
    rates = [_rate(row) for row in ordered]
    axes.plot(fractions, rates, "o-", ms=3, color=color, label=f"d={distance}")
    for decoding, style in (("weak_alone", "-."), ("strong_alone", "--")):
        reference = _reference(references, point, decoding)
        rate = _rate(reference)
        axes.axhline(rate, color=color, ls=style, lw=1)
    axes.axvline(capacities[point], color=color, ls=":", lw=1)


def write_calibration(swept, capacities, folder: pathlib.Path) -> None:
    """Per point, the threshold that escalates each target share."""
    grouped = _by_point(swept)
    rows = []
    for point in sorted(grouped):
        ordered = sorted(grouped[point], key=_decibels)
        thresholds = [_decibels(row) for row in ordered]
        fractions = [_escalated(row) for row in ordered]
        row = [*point, round(capacities[point], 4)]
        for target in TARGETS:
            threshold = _threshold_at(thresholds, fractions, target)
            row.append(threshold)
        rows.append(row)
    path = folder / "calibration.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CALIBRATION_FIELDS)
        writer.writerows(rows)
    print(f"wrote {path}")


def _threshold_at(thresholds: list, fractions: list, target: float):
    """The threshold, linear between swept ones, escalating target; or ''."""
    for index in range(1, len(thresholds)):
        low_fraction = fractions[index - 1]
        high_fraction = fractions[index]
        if low_fraction <= target <= high_fraction:
            low = (thresholds[index - 1], low_fraction)
            high = (thresholds[index], high_fraction)
            return _between(low, high, target)
    return ""


def _between(low: tuple, high: tuple, target: float) -> float:
    """The threshold on the line from low to high where target falls."""
    low_threshold, low_fraction = low
    high_threshold, high_fraction = high
    span = high_fraction - low_fraction
    if span == 0:
        return round(low_threshold, 2)
    share = (target - low_fraction) / span
    step = share * (high_threshold - low_threshold)
    threshold = low_threshold + step
    return round(threshold, 2)


def _by_probability(rows: list) -> dict:
    """Rows by p, then by d, both in order."""
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        distance, probability = _point(row)
        grouped[probability][distance].append(row)
    panels = {}
    for probability in sorted(grouped):
        by_distance = grouped[probability]
        distance_items = by_distance.items()
        ordered = sorted(distance_items)
        panels[probability] = dict(ordered)
    return panels


def _panels(count: int) -> tuple:
    """A figure of count panels side by side, and their axes."""
    width = 5 * count
    figure, axes_grid = plt.subplots(
        1, count, figsize=(width, 4), squeeze=False
    )
    return figure, axes_grid[0]


def _finish(axes, title: str) -> None:
    axes.set_title(title)
    axes.set_yscale("log")
    axes.set_ylabel("logical error rate per shot")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(fontsize=7)


def _by_point(rows: list) -> dict:
    grouped = collections.defaultdict(list)
    for row in rows:
        point = _point(row)
        grouped[point].append(row)
    return grouped


def _reference(references: list, point: tuple, decoding: str) -> dict:
    for row in references:
        if row["decoding"] == decoding and _point(row) == point:
            return row
    raise KeyError(f"the pilot has no {decoding} at {point}")


def _point(row: dict) -> tuple:
    distance = int(row[DISTANCE])
    probability = float(row[PROBABILITY])
    return (distance, probability)


def _decibels(row: dict) -> float:
    text = row["threshold"]
    number = text.removesuffix("dB")
    return float(number)


def _rate(row: dict) -> float:
    return float(row["logical_error_rate_estimate"])


def _escalated(row: dict) -> float:
    return float(row["escalated_fraction"])


def _read(path: pathlib.Path) -> list:
    with path.open() as rows_file:
        reader = csv.DictReader(rows_file)
        return list(reader)


def _save(figure, folder: pathlib.Path, name: str) -> None:
    figure.tight_layout()
    path = folder / name
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")


if __name__ == "__main__":
    sweep_folder = pathlib.Path(sys.argv[1])
    pilot_folder = pathlib.Path(sys.argv[2])
    main(sweep_folder, pilot_folder)
