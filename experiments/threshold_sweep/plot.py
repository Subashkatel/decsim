"""The threshold sweep's figures, one plain question each.

`python plot.py <results folder>` reads the folder's status.csv and
writes plots/: the share of windows sent to the strong decoder, how
many times better switching is than union-find alone, how long an
escalated window takes (median and p99), and how many times too slow
the strong decoder is. Every figure has one panel, the threshold
across, and a line per distance and physical error rate.

Union-find alone, the no-switching reference, comes from the switching
baseline's weak_alone points beside this folder: the same machine and
shots, run with no escalation, so its rate is the same at every
threshold. A point with no failures has no rate, so the improvement
figure draws physical error rate 0.003 only, where every point has 100
failures; at 0.001 at most one point has failures.
"""

import csv
import pathlib
import sys

import matplotlib.pyplot as pyplot

DISTANCES = [9, 11]
ERROR_RATES = [0.001, 0.003]
THRESHOLDS_DB = [5.0, 10.0, 15.0, 20.0, 30.0]
RATE = "logical_error_rate_per_round"
BASELINE_FOLDER = "2026-10-01_switching_baseline"
BASELINE_NAME = "weak_alone"
FIGURE_SIZE_INCHES = (4.8, 3.6)
DOTS_PER_INCH = 150
GRID_ALPHA = 0.3
LEGEND_FONT_SIZE = 8
THRESHOLD_LABEL = "threshold (dB)"
# the rates whose switching points all have failures to compare
IMPROVEMENT_RATES = [0.003]


def main(folder: pathlib.Path) -> None:
    """Every figure of the folder's status.csv, into its plots/."""
    rows = rows_of(folder / "status.csv")
    baseline_rows = union_find_alone_rows(folder.parent / BASELINE_FOLDER)
    plots_folder = folder / "plots"
    plots_folder.mkdir(exist_ok=True)
    for old_figure in plots_folder.glob("*.png"):
        old_figure.unlink()
    escalated_figure(rows, plots_folder / "1_windows_escalated.png")
    improvement_figure(
        rows, baseline_rows, plots_folder / "2_switching_improvement.png"
    )
    latency_figure(rows, "median", plots_folder / "3_escalated_median.png")
    latency_figure(rows, "p99", plots_folder / "4_escalated_p99.png")
    keep_up_figure(rows, plots_folder / "5_strong_too_slow.png")


def rows_of(status_path: pathlib.Path) -> list:
    """The sweep's points with data, as numbers."""
    with open(status_path) as handle:
        statuses = list(csv.DictReader(handle))
    rows = []
    for status in statuses:
        if status["state"] == "no data":
            continue
        row = numbers_of(status)
        row["threshold"] = float(status["escalation.gap_threshold_db"])
        rows.append(row)
    return rows


def union_find_alone_rows(folder: pathlib.Path) -> list:
    """The switching baseline's union-find alone points, as numbers."""
    with open(folder / "configurations.csv") as handle:
        configurations = list(csv.DictReader(handle))
    baseline_ids = [
        configuration["configuration_id"]
        for configuration in configurations
        if configuration["name"] == BASELINE_NAME
    ]
    with open(folder / "status.csv") as handle:
        statuses = list(csv.DictReader(handle))
    return [
        numbers_of(status)
        for status in statuses
        if status["configuration_id"] in baseline_ids
    ]


def numbers_of(status: dict) -> dict:
    """A status.csv row with numbers parsed and its distance and rate named."""
    row = {column: number(text) for column, text in status.items()}
    row["d"] = int(status["qpu.distance"])
    row["error_rate"] = float(
        status["workload.arguments.physical_error_probability"]
    )
    return row


def number(text: str):
    """A status.csv value as a float, or None where it is empty."""
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


def series_of(rows: list, error_rates: list) -> list:
    """(label, colour, rows by threshold) per distance and error rate.

    A series keeps its colour in every figure, by its place among all
    the sweep's distances and error rates.
    """
    series = []
    for error_rate in error_rates:
        for distance in DISTANCES:
            place = ERROR_RATES.index(error_rate) * len(DISTANCES)
            place += DISTANCES.index(distance)
            colour = f"C{place}"
            label = f"d={distance}, physical error rate {error_rate:g}"
            chosen = [
                row
                for row in rows
                if row["d"] == distance and row["error_rate"] == error_rate
            ]
            chosen.sort(key=lambda row: row["threshold"])
            series.append((label, colour, chosen))
    return series


def new_figure(title: str, y_label: str) -> tuple:
    """One panel in the house style, the threshold across."""
    figure, axis = pyplot.subplots(
        figsize=FIGURE_SIZE_INCHES, layout="constrained"
    )
    axis.set_title(title)
    axis.set_xlabel(THRESHOLD_LABEL)
    axis.set_ylabel(y_label)
    axis.set_xticks(THRESHOLDS_DB, [f"{value:g}" for value in THRESHOLDS_DB])
    axis.grid(alpha=GRID_ALPHA)
    return figure, axis


def save(figure, axis, path: pathlib.Path) -> None:
    """Legend, write, close."""
    axis.legend(fontsize=LEGEND_FONT_SIZE)
    figure.savefig(path, dpi=DOTS_PER_INCH)
    pyplot.close(figure)


def escalated_figure(rows: list, path: pathlib.Path) -> None:
    """The share of windows each threshold sends to the strong decoder."""
    figure, axis = new_figure(
        "Windows sent to the strong decoder", "windows sent (%)"
    )
    for label, colour, series_rows in series_of(rows, ERROR_RATES):
        thresholds = [row["threshold"] for row in series_rows]
        shares = [100 * row["escalated_fraction"] for row in series_rows]
        axis.plot(thresholds, shares, "o-", color=colour, label=label)
    axis.set_ylim(bottom=0)
    save(figure, axis, path)


def improvement_figure(
    rows: list, baseline_rows: list, path: pathlib.Path
) -> None:
    """Union-find alone's rate over switching's, where both have failures.

    The bar spans the ratio of the two rates' ranges taken the far way
    round: union-find's low over switching's high, and its high over
    switching's low.
    """
    figure, axis = new_figure(
        "How many times better switching is",
        "union-find alone rate / switching rate",
    )
    for label, colour, series_rows in series_of(rows, IMPROVEMENT_RATES):
        points = improvement_points(series_rows, baseline_rows)
        if not points:
            continue
        thresholds, ratios, below, above = zip(*points)
        axis.errorbar(
            thresholds,
            ratios,
            yerr=[below, above],
            fmt="o-",
            capsize=3,
            color=colour,
            label=label,
        )
    axis.axhline(1, color="black", linestyle="--", linewidth=1)
    axis.set_ylim(bottom=0)
    save(figure, axis, path)


def improvement_points(series_rows: list, baseline_rows: list) -> list:
    """(threshold, ratio, below, above) for each point both have a rate for."""
    points = []
    for row in series_rows:
        baseline = baseline_for(row, baseline_rows)
        if baseline is None or not has_rate(row) or not has_rate(baseline):
            continue
        ratio = baseline[RATE] / row[RATE]
        lowest = baseline[f"{RATE}_low"] / row[f"{RATE}_high"]
        highest = baseline[f"{RATE}_high"] / row[f"{RATE}_low"]
        points.append((row["threshold"], ratio, ratio - lowest, highest - ratio))
    return points


def baseline_for(row: dict, baseline_rows: list):
    """The union-find alone point at the row's distance and error rate."""
    for baseline in baseline_rows:
        same_distance = baseline["d"] == row["d"]
        if same_distance and baseline["error_rate"] == row["error_rate"]:
            return baseline
    return None


def has_rate(row: dict) -> bool:
    """Whether a point failed, on at most half its shots."""
    if row["logical_failures"] == 0:
        return False
    return row["is_shot_rate_above_half"] == "False"


def latency_figure(rows: list, statistic: str, path: pathlib.Path) -> None:
    """How long an escalated window takes, formed to Pauli frame, in ms."""
    figure, axis = new_figure(
        f"Time for an escalated window ({statistic})", "time (ms)"
    )
    column = f"buffer0_ready_to_frame_strong_{statistic}_us"
    for label, colour, series_rows in series_of(rows, ERROR_RATES):
        thresholds = [row["threshold"] for row in series_rows]
        milliseconds = [row[column] / 1000 for row in series_rows]
        axis.plot(thresholds, milliseconds, "o-", color=colour, label=label)
    axis.set_ylim(bottom=0)
    save(figure, axis, path)


def keep_up_figure(rows: list, path: pathlib.Path) -> None:
    """The strong decoder's mean decode time over Toshio's Theorem 1 limit.

    Above 1 the strong decoder cannot keep up with the windows it is
    sent (2510.25222 Eq. (6)); report.strong_service_bound_us gives the
    limit for the point's own escalation rate.
    """
    figure, axis = new_figure(
        "How many times too slow the strong decoder is",
        "decode time / longest time that keeps up",
    )
    for label, colour, series_rows in series_of(rows, ERROR_RATES):
        thresholds = [row["threshold"] for row in series_rows]
        ratios = [
            row["strong_service_mean_us"] / row["strong_service_bound_us"]
            for row in series_rows
        ]
        axis.plot(thresholds, ratios, "o-", color=colour, label=label)
    axis.axhline(1, color="black", linestyle="--", linewidth=1)
    axis.set_yscale("log")
    save(figure, axis, path)


if __name__ == "__main__":
    results_folder = pathlib.Path(sys.argv[1])
    main(results_folder)
