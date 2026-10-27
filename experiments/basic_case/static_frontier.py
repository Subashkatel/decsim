"""The static frontier: each static threshold's failure rate and share.

    python static_frontier.py FOLDER

FOLDER holds answer_check.py's json lines, run with a top gap so each
stream keeps its static chains (observe/answer_checks.py). Writes into
FOLDER/plots/, per point:
  - static_frontier_d{d}_p{p}.csv: per threshold, the shots, the windows
    escalated and decided, the escalated share, and how many shots
    failed, with the exact 95% interval;
  - static_frontier_d{d}_p{p}.png: failure rate against escalated share
    for the static chains, with union-find alone (share 0), belief
    matching alone (share 1) and the switching machine marked, and the
    per-shot and per-window best-of-two lines drawn across, each with
    its interval.
"""

import collections
import csv
import pathlib
import sys

import matplotlib

import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.failure_statistics as failure_statistics
import decsim.experiments.sweeps as sweeps

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

answer_check_plots = sweeps.load_beside(__file__, "answer_check_plots.py")
FIXED_COUNT = failure_statistics.StopKind.CAP
FIELDS = (
    "threshold_decibels",
    "threshold_nats",
    "shots",
    "escalated_windows",
    "window_count",
    "escalated_share",
    "failures",
    "rate",
    "low",
    "high",
)
# each marked run's label, color and marker
MARKS = (
    ("weak", "union-find alone", "tab:red", "s"),
    ("strong", "belief matching alone", "tab:blue", "s"),
    ("switching", "switching machine, 20 dB", "black", "*"),
)
# each line drawn across: its label and color
LINES = (
    ("per_shot_line", "per-shot best-of-two line", "tab:green"),
    ("per_window_line", "per-window best-of-two line", "tab:olive"),
)


def main(folder: pathlib.Path) -> None:
    """A table and a plot a point that kept static chains."""
    out = folder / "plots"
    out.mkdir(exist_ok=True)
    lines_by_point = answer_check_plots.lines_by_point_of(folder)
    by_point = lines_by_point.items()
    points = sorted(by_point)
    for point, lines in points:
        rows = frontier_rows(lines)
        if not rows:
            continue
        distance, probability = point
        name = f"static_frontier_d{distance}_p{probability}"
        table = out / f"{name}.csv"
        write_rows(rows, table)
        figure_path = out / f"{name}.png"
        plot_frontier(point, lines, rows, figure_path)


def frontier_rows(lines: list) -> list:
    """One row a static threshold, summed over the point's streams."""
    chains_by_threshold = collections.defaultdict(list)
    for line in lines:
        for chain in line["static"]:
            decibels = chain["threshold_decibels"]
            chains_by_threshold[decibels].append(chain)
    rows = []
    by_threshold = chains_by_threshold.items()
    thresholds = sorted(by_threshold)
    for decibels, chains in thresholds:
        row = _frontier_row(decibels, chains)
        rows.append(row)
    return rows


def write_rows(rows: list, path: pathlib.Path) -> None:
    """The frontier's rows, a threshold each."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_frontier(
    point: tuple, lines: list, rows: list, path: pathlib.Path
) -> None:
    """The static chains' rates against their shares, the runs marked."""
    distance, probability = point
    runs = answer_check_plots.point_row(point, lines)
    window_count = rows[0]["window_count"]
    switching_share = sum(line["escalated_windows"] for line in lines)
    switching_share /= window_count
    shares = {"weak": 0.0, "strong": 1.0, "switching": switching_share}
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    _draw_static(axis, rows)
    for run, label, color, marker in MARKS:
        _draw_mark(axis, runs, run, shares[run], (label, color, marker))
    for run, label, color in LINES:
        _draw_line(axis, runs, run, label, color)
    axis.set_yscale("log")
    axis.set_xlabel("escalated share of windows")
    axis.set_ylabel("logical failure rate per stream")
    shots = runs["shots"]
    axis.set_title(f"d = {distance}, p = {probability}, {shots} streams")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _frontier_row(decibels: float, chains: list) -> dict:
    shots = len(chains)
    failures = sum(chain["failed"] for chain in chains)
    escalated = sum(chain["escalated_windows"] for chain in chains)
    window_count = sum(chain["window_count"] for chain in chains)
    estimate = failure_statistics.estimate(failures, shots, FIXED_COUNT)
    return {
        "threshold_decibels": decibels,
        "threshold_nats": threshold_sources.decibels_to_nats(decibels),
        "shots": shots,
        "escalated_windows": escalated,
        "window_count": window_count,
        "escalated_share": escalated / window_count,
        "failures": failures,
        "rate": estimate.rate,
        "low": estimate.low,
        "high": estimate.high,
    }


def _draw_static(axis, rows: list) -> None:
    # a threshold with no failure has no rate to draw on a log axis
    rows = [row for row in rows if row["rate"] is not None]
    shares = [row["escalated_share"] for row in rows]
    rates = [row["rate"] for row in rows]
    errors = _error_bars(rows)
    axis.errorbar(
        shares,
        rates,
        yerr=errors,
        color="tab:purple",
        marker="o",
        markersize=2.5,
        linewidth=1,
        capsize=1.5,
        label="static thresholds",
    )


def _error_bars(rows: list) -> list:
    """Each row's distance from its rate down to low and up to high."""
    lows = [row["rate"] - row["low"] for row in rows]
    highs = [row["high"] - row["rate"] for row in rows]
    return [lows, highs]


def _draw_mark(axis, runs: dict, run: str, share: float, style: tuple) -> None:
    label, color, marker = style
    rate = runs[f"{run}_rate"]
    if rate is None:
        return
    low = rate - runs[f"{run}_low"]
    high = runs[f"{run}_high"] - rate
    axis.errorbar(
        [share],
        [rate],
        yerr=[[low], [high]],
        color=color,
        marker=marker,
        markersize=8,
        capsize=3,
        linestyle="none",
        label=label,
    )


def _draw_line(axis, runs: dict, run: str, label: str, color: str) -> None:
    rate = runs[f"{run}_rate"]
    if rate is None:
        return
    axis.axhline(rate, color=color, linestyle="--", linewidth=1, label=label)
    low = runs[f"{run}_low"]
    high = runs[f"{run}_high"]
    axis.axhspan(low, high, color=color, alpha=0.15)


if __name__ == "__main__":
    folder = pathlib.Path(sys.argv[1])
    main(folder)
