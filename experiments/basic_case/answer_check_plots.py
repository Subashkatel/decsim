"""The answer checks' table and confidence plots.

    python answer_check_plots.py FOLDER

FOLDER holds answer_check.py's json lines. Writes into FOLDER/plots/:
  - answer_check.csv: per point, the shots and how many failed under
    union-find alone, belief matching alone, the switching machine, the
    per-shot best-of-two line (both alone failed) and the per-window
    line (the best-of-two chain), each rate with its exact 95% interval;
  - confidence_d{d}_p{p}.png: against the weak gap, how often the weak
    answer fails where the strong one passes in another logical class,
    the reverse, both, and, faint, one fails and the other passes in the
    same class (the boundary decides), with the 20 dB threshold marked
    (decsim/experiments/answer_check_bins.py).
"""

import collections
import csv
import json
import math
import pathlib
import sys

import matplotlib

import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.answer_check_bins as answer_check_bins
import decsim.experiments.failure_statistics as failure_statistics

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# the paper's threshold (constants.yaml escalation.threshold)
THRESHOLD = threshold_sources.FixedThreshold.Settings(threshold_decibels=20.0)
# one-nat bins to 14 nats, then every larger gap and the infinite one
GAP_EDGES = (*range(15), math.inf)
RUNS = ("weak", "strong", "switching", "per_shot_line", "per_window_line")
FIELDS = ("d", "p", "shots") + tuple(
    f"{run}_{part}"
    for run in RUNS
    for part in ("failures", "rate", "low", "high")
)
# each curve's bin field, label, color and opacity: the boundary's
# curve is faint, as it says nothing of the window's own class
CURVES = (
    ("weak_only", "weak fails, strong passes", "tab:red", 1.0),
    ("strong_only", "strong fails, weak passes", "tab:blue", 1.0),
    (
        "boundary_decides",
        "one fails, same class: the boundary",
        "tab:purple",
        0.35,
    ),
    ("both", "both fail", "tab:gray", 1.0),
)


def main(folder: pathlib.Path) -> None:
    """The table and a confidence plot a point."""
    out = folder / "plots"
    out.mkdir(exist_ok=True)
    lines_by_point = lines_by_point_of(folder)
    rows = [
        point_row(point, lines)
        for point, lines in sorted(lines_by_point.items())
    ]
    write_table(rows, out)
    for point, lines in sorted(lines_by_point.items()):
        plot_confidence(point, lines, out)


def write_table(rows: list, out: pathlib.Path) -> None:
    """answer_check.csv, and the same rows printed."""
    path = out / "answer_check.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        rates = ", ".join(f"{run} {row[f'{run}_failures']}" for run in RUNS)
        print(f"d={row['d']} p={row['p']} shots={row['shots']}: {rates}")


def plot_confidence(point: tuple, lines: list, out: pathlib.Path) -> None:
    """The four rates against the weak gap, on the best-of-two chain."""
    distance, probability = point
    streams = [line["windows"]["best_of_two"] for line in lines]
    bins = answer_check_bins.gap_bins(streams, GAP_EDGES)
    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    plotted = False
    for name, label, color, opacity in CURVES:
        centers, rates, lows, highs = _curve(bins, name)
        plotted = plotted or bool(rates)
        errors = [
            [rate - low for rate, low in zip(rates, lows)],
            [high - rate for rate, high in zip(rates, highs)],
        ]
        axis.errorbar(
            centers,
            rates,
            yerr=errors,
            label=label,
            color=color,
            alpha=opacity,
            marker="o",
            markersize=3,
            capsize=2,
            linewidth=1,
        )
    axis.axvline(
        THRESHOLD.threshold_nats,
        color="black",
        linestyle="--",
        linewidth=1,
        label="20 dB threshold",
    )
    # a log axis needs a rate above zero
    if plotted:
        axis.set_yscale("log")
    axis.set_xlabel("weak gap (nats); the last point holds every gap over 14")
    axis.set_ylabel("fraction of windows")
    axis.set_title(
        f"d = {distance}, p = {probability}: each window's two answers"
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out / f"confidence_d{distance}_p{probability}.png", dpi=150)
    plt.close(figure)


def lines_by_point_of(folder: pathlib.Path) -> dict:
    """Every json line in the folder, under its (distance, p) point."""
    lines_by_point = collections.defaultdict(list)
    for path in sorted(folder.glob("*.jsonl")):
        for text in path.read_text().splitlines():
            line = json.loads(text)
            point = (line["distance"], line["p"])
            lines_by_point[point].append(line)
    return lines_by_point


def point_row(point: tuple, lines: list) -> dict:
    """The point's shots and each run's failures, with exact intervals."""
    distance, probability = point
    failures = {
        "weak": sum(line["failed"]["weak"] for line in lines),
        "strong": sum(line["failed"]["strong"] for line in lines),
        "switching": sum(line["switching_failed"] for line in lines),
        "per_shot_line": sum(
            line["failed"]["weak"] and line["failed"]["strong"]
            for line in lines
        ),
        "per_window_line": sum(line["failed"]["best_of_two"] for line in lines),
    }
    shots = len(lines)
    row = {"d": distance, "p": probability, "shots": shots}
    for run in RUNS:
        estimate = failure_statistics.estimate(
            failures[run], shots, failure_statistics.StopKind.CAP
        )
        row[f"{run}_failures"] = failures[run]
        row[f"{run}_rate"] = estimate.rate
        row[f"{run}_low"] = estimate.low
        row[f"{run}_high"] = estimate.high
    return row


def _curve(bins: list, name: str) -> tuple:
    """Each bin with a failure: its center, rate and interval."""
    centers, rates, lows, highs = [], [], [], []
    for gap_bin in bins:
        estimate = getattr(gap_bin, name)
        if estimate.rate is None:
            continue
        high_edge = min(gap_bin.high_nats, gap_bin.low_nats + 1)
        centers.append((gap_bin.low_nats + high_edge) / 2)
        rates.append(estimate.rate)
        lows.append(estimate.low)
        highs.append(estimate.high)
    return centers, rates, lows, highs


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
