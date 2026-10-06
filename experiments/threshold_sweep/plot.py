"""The threshold sweep's figures, drawn from its status.csv.

`python plot.py <results folder>` writes the folder's plots/, one
figure per question: how many windows each cluster-gap threshold sends
to Relay-BP-5, what logical error rate that share buys, what it costs
in latency for kept and for escalated windows, and whether the strong
decoder keeps up. Union-find alone, the no-switching reference, comes
from the switching baseline's weak_alone points beside this folder:
the same machine and shots, run with no escalation, so its rate is the
same at every threshold. A panel is one physical error rate and a curve one
distance. A point with no failures is a hollow marker at its upper
bound, as decsim.plots.bounds draws it, and a point failing more than
half its shots has no per-round rate and is left out.
"""

import csv
import pathlib
import sys

import decsim.plots as plots

DISTANCES = [9, 11]
ERROR_RATES = [0.001, 0.003]
THRESHOLDS_DB = [5.0, 10.0, 15.0, 20.0, 30.0]
X = "physical error rate"
THRESHOLD = "cluster-gap threshold (dB)"
SHARE = "windows escalated (%)"
RATE = "logical_error_rate_per_round"
WINDOW = "window"
TIERS = {"kept": "weak", "escalated": "strong"}
TIER_CURVES = [
    f"d={distance} {tier}" for distance in DISTANCES for tier in TIERS
]
KEEP_UP = "strong decode time / Toshio bound"
SWITCHING = "switching"
SWITCHING_CURVES = [f"d={distance} switching" for distance in DISTANCES]
BASELINE_FOLDER = "2026-10-01_switching_baseline"
BASELINE_NAME = "weak_alone"


def main(folder: pathlib.Path) -> None:
    """Every figure of the folder's status.csv, into its plots/."""
    rows = rows_of(folder)
    plots_folder = folder / "plots"
    plots_folder.mkdir(exist_ok=True)
    escalated_figure(rows, plots_folder / "escalated.png")
    error_rate_figure(rows, plots_folder / "logical_error_rate.png")
    baseline_rows = union_find_alone_rows(folder.parent / BASELINE_FOLDER)
    versus_union_find_figure(
        rows, baseline_rows, plots_folder / "versus_union_find.png"
    )
    latency_figure(rows, plots_folder / "latency.png")
    keep_up_figure(rows, plots_folder / "strong_keeps_up.png")


def rows_of(folder: pathlib.Path) -> list:
    """The points with data, as numbers."""
    status_path = folder / "status.csv"
    with open(status_path) as handle:
        statuses = list(csv.DictReader(handle))
    rows = []
    for status in statuses:
        if status["state"] == "no data":
            continue
        row = {column: number(text) for column, text in status.items()}
        row["d"] = int(status["qpu.distance"])
        row[X] = float(status["workload.arguments.physical_error_probability"])
        row[THRESHOLD] = float(status["escalation.gap_threshold_db"])
        row[SHARE] = 100 * row["escalated_fraction"]
        service = row["strong_service_mean_us"]
        row[KEEP_UP] = service / row["strong_service_bound_us"]
        rows.append(row)
    return rows


def number(text: str):
    """A status.csv value as a float, or None where it is empty."""
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


def escalated_figure(rows: list, path: pathlib.Path) -> None:
    """The share of windows each threshold sends to Relay-BP-5."""
    figure, axis_by_rate = plots.panels(X, ERROR_RATES, columns=2)
    for error_rate, axis in axis_by_rate.items():
        rate_rows = [row for row in rows if row[X] == error_rate]
        plots.values(axis, rate_rows, THRESHOLD, SHARE, "d", order=DISTANCES)
        threshold_axis(axis)
        axis.set_ylabel(SHARE)
    plots.save(figure, path)


def error_rate_figure(rows: list, path: pathlib.Path) -> None:
    """The logical error rate per round against the share escalated.

    Each point is one threshold; a point with failures has its number
    in dB written beside it, so the share it escalates and the rate it
    reaches read together. A point with none yet is its hollow bound.
    """
    figure, axis_by_rate = plots.panels(X, ERROR_RATES, columns=2)
    for error_rate, axis in axis_by_rate.items():
        rate_rows = [row for row in rows if row[X] == error_rate]
        failed_rows = [row for row in rate_rows if has_per_round_rate(row)]
        error_free_rows = [
            row for row in rate_rows if row["logical_failures"] == 0
        ]
        if failed_rows:
            plots.values(
                axis,
                failed_rows,
                SHARE,
                RATE,
                "d",
                low=f"{RATE}_low",
                high=f"{RATE}_high",
                order=DISTANCES,
            )
        plots.bounds(
            axis, error_free_rows, SHARE, f"{RATE}_high", "d", order=DISTANCES
        )
        for row in failed_rows:
            threshold_label = f" {row[THRESHOLD]:g}"
            axis.annotate(
                threshold_label,
                (row[SHARE], row[RATE]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
        axis.set_yscale("log")
        axis.set_xlabel(SHARE)
        axis.set_ylabel("logical error rate per round")
    plots.save(figure, path)


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
    rows = []
    for status in statuses:
        if status["configuration_id"] not in baseline_ids:
            continue
        row = {column: number(text) for column, text in status.items()}
        row["d"] = int(status["qpu.distance"])
        row[X] = float(status["workload.arguments.physical_error_probability"])
        rows.append(row)
    return rows


def versus_union_find_figure(
    rows: list, baseline_rows: list, path: pathlib.Path
) -> None:
    """Switching's logical error rate per threshold against union-find alone.

    Switching is a curve per distance over the thresholds; union-find
    alone is a dashed line in the same colour with its band shaded, or
    a dotted line at its upper bound while it has no failures.
    """
    figure, axis_by_rate = plots.panels(X, ERROR_RATES, columns=2)
    for error_rate, axis in axis_by_rate.items():
        rate_rows = [row for row in rows if row[X] == error_rate]
        failed_rows = [row for row in rate_rows if has_per_round_rate(row)]
        error_free_rows = [
            row for row in rate_rows if row["logical_failures"] == 0
        ]
        for row in rate_rows:
            row[SWITCHING] = f"d={row['d']} switching"
        plots.values(
            axis,
            failed_rows,
            THRESHOLD,
            RATE,
            SWITCHING,
            low=f"{RATE}_low",
            high=f"{RATE}_high",
            order=SWITCHING_CURVES,
        )
        plots.bounds(
            axis,
            error_free_rows,
            THRESHOLD,
            f"{RATE}_high",
            SWITCHING,
            order=SWITCHING_CURVES,
        )
        for row in baseline_rows:
            if row[X] == error_rate and row["d"] in DISTANCES:
                draw_union_find_alone(axis, row)
        axis.set_yscale("log")
        axis.set_ylabel("logical error rate per round")
        axis.legend(fontsize=8)
        threshold_axis(axis)
    plots.save(figure, path)


def draw_union_find_alone(axis, row: dict) -> None:
    """One distance's union-find alone rate across every threshold."""
    colour = f"C{DISTANCES.index(row['d'])}"
    label = f"d={row['d']} union-find alone"
    if not has_per_round_rate(row):
        bound = row[f"{RATE}_high"]
        label = f"{label}, no failures yet (rate is below)"
        axis.axhline(bound, color=colour, linestyle=":", label=label)
        return
    axis.axhline(row[RATE], color=colour, linestyle="--", label=label)
    low = row[f"{RATE}_low"]
    high = row[f"{RATE}_high"]
    axis.axhspan(low, high, color=colour, alpha=0.12, linewidth=0)


def has_per_round_rate(row: dict) -> bool:
    """Whether a point failed, on at most half its shots."""
    if row["logical_failures"] == 0:
        return False
    return row["is_shot_rate_above_half"] == "False"


def latency_figure(rows: list, path: pathlib.Path) -> None:
    """Window formed to Pauli frame, median and p99, kept against escalated.

    A kept window's correction is the union-find answer; an escalated
    window's is Relay-BP-5's, so its time holds the strong side's wait.
    """
    statistics = ["median", "p99"]
    panel_names = [
        f"{X} = {error_rate:g}, {statistic}"
        for error_rate in ERROR_RATES
        for statistic in statistics
    ]
    figure, axis_by_panel = plots.panels("latency", panel_names, columns=2)
    for panel_name, axis in axis_by_panel.items():
        rate_text, statistic = panel_name.split(", ")
        error_rate = float(rate_text.split(" = ")[1])
        rate_rows = [row for row in rows if row[X] == error_rate]
        tier_rows = rows_by_tier(rate_rows, statistic)
        plots.values(
            axis, tier_rows, THRESHOLD, statistic, WINDOW, order=TIER_CURVES
        )
        axis.set_title(panel_name)
        axis.set_ylabel("window formed to Pauli frame (µs)")
        axis.set_yscale("log")
        threshold_axis(axis)
    plots.save(figure, path)


def rows_by_tier(rows: list, statistic: str) -> list:
    """One row per point and tier, its latency under the statistic's name."""
    tier_rows = []
    for row in rows:
        for tier, column_tier in TIERS.items():
            column = f"buffer0_ready_to_frame_{column_tier}_{statistic}_us"
            tier_row = {
                THRESHOLD: row[THRESHOLD],
                WINDOW: f"d={row['d']} {tier}",
                statistic: row[column],
            }
            tier_rows.append(tier_row)
    return tier_rows


def keep_up_figure(rows: list, path: pathlib.Path) -> None:
    """The strong decoder's mean decode time over Toshio's Theorem 1 bound.

    Above 1 the strong decoder cannot keep up with the windows it is
    sent (2510.25222 Eq. (6)); report.strong_service_bound_us gives the
    bound for the point's own escalation rate.
    """
    figure, axis_by_rate = plots.panels(X, ERROR_RATES, columns=2)
    for error_rate, axis in axis_by_rate.items():
        rate_rows = [row for row in rows if row[X] == error_rate]
        plots.values(axis, rate_rows, THRESHOLD, KEEP_UP, "d", order=DISTANCES)
        axis.axhline(1, color="black", linewidth=0.8, linestyle="--")
        axis.set_yscale("log")
        axis.set_ylabel(KEEP_UP)
        threshold_axis(axis)
    plots.save(figure, path)


def threshold_axis(axis) -> None:
    """A linear axis over the swept thresholds, labelled at every one."""
    axis.set_xscale("linear")
    threshold_labels = [f"{threshold:g}" for threshold in THRESHOLDS_DB]
    axis.set_xticks(THRESHOLDS_DB, threshold_labels)
    axis.set_xlim(0, 35)
    axis.set_xlabel(THRESHOLD)


if __name__ == "__main__":
    results_folder = pathlib.Path(sys.argv[1])
    main(results_folder)
