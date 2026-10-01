"""The switching baseline's figures, drawn from its status.csv.

`python plot.py <results folder>` writes the folder's plots/, one
figure per question: the share of windows escalated, the formed-to-
commit latency, the logical error rate of switching against union-find
alone, and whether the strong decoder keeps up. The folder holds
status.csv, as `decsim status` writes it, and configurations.csv, each
configuration's id and name. A point with no failures has no rate and is
left out of the rate figure, as decsim.plots.error_rate leaves it.
"""

import csv
import pathlib
import sys

import matplotlib.pyplot as pyplot

import decsim.plots as plots

DISTANCES = [5, 7, 9, 11, 13]
ERROR_RATES = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005]
SWITCHING = "switching"
UNION_FIND_ALONE = "union-find alone"
# configurations.csv's names, in the figures' words
SHOWN_NAME = {
    "switching_baseline": SWITCHING,
    "weak_alone": UNION_FIND_ALONE,
}
X = "physical error rate"


def main(folder: pathlib.Path) -> None:
    """Every figure of the folder's status.csv, into its plots/."""
    rows = rows_of(folder)
    plots_folder = folder / "plots"
    plots_folder.mkdir(exist_ok=True)
    switching_rows = [row for row in rows if row["decoder"] == SWITCHING]
    escalated_figure(switching_rows, plots_folder / "escalated.png")
    latency_figure(rows, plots_folder / "latency.png")
    error_rate_figure(rows, plots_folder / "logical_error_rate.png")
    strong_figure(switching_rows, plots_folder / "strong_decoder.png")


def rows_of(folder: pathlib.Path) -> list:
    """The points with data, as numbers, each named by its decoder."""
    name_by_id = {}
    with open(folder / "configurations.csv") as handle:
        for configuration in csv.DictReader(handle):
            name = configuration["name"]
            name_by_id[configuration["configuration_id"]] = SHOWN_NAME[name]
    rows = []
    with open(folder / "status.csv") as handle:
        for status in csv.DictReader(handle):
            if status["state"] == "no data":
                continue
            row = {"decoder": name_by_id[status["configuration_id"]]}
            row["d"] = int(status["qpu.distance"])
            row[X] = float(
                status["workload.arguments.physical_error_probability"]
            )
            for column, text in status.items():
                if column not in row:
                    row[column] = number(text)
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
    """The share of windows the weak decoder sends to the strong one."""
    for row in rows:
        row["windows escalated (%)"] = 100 * row["escalated_fraction"]
    figure, axis = single_panel("windows sent to the strong decoder")
    plots.values(axis, rows, X, "windows escalated (%)", "d", order=DISTANCES)
    rate_axis(axis)
    plots.save(figure, path)


def latency_figure(rows: list, path: pathlib.Path) -> None:
    """Formed-to-commit latency, median and p99, for both decoders."""
    for row in rows:
        row["median (us)"] = row["buffer0_ready_to_frame_median_us"]
        row["p99 (us)"] = row["buffer0_ready_to_frame_p99_us"]
    panel_names = [
        f"{SWITCHING}, median",
        f"{SWITCHING}, p99",
        f"{UNION_FIND_ALONE}, median",
        f"{UNION_FIND_ALONE}, p99",
    ]
    figure, axis_by_panel = plots.panels("latency", panel_names, columns=2)
    for panel_name, axis in axis_by_panel.items():
        decoder, statistic = panel_name.split(", ")
        column = f"{statistic} (us)"
        decoder_rows = [row for row in rows if row["decoder"] == decoder]
        plots.values(axis, decoder_rows, X, column, "d", order=DISTANCES)
        axis.set_title(panel_name)
        axis.set_ylabel("window formed to correction committed (us)")
        axis.set_yscale("log")
        rate_axis(axis)
    plots.save(figure, path)


def error_rate_figure(rows: list, path: pathlib.Path) -> None:
    """Switching against union-find alone, a panel per distance."""
    failed_rows = [row for row in rows if row["logical_failures"]]
    figure, axis_by_distance = plots.panels("d", DISTANCES)
    for distance, axis in axis_by_distance.items():
        distance_rows = [row for row in failed_rows if row["d"] == distance]
        plots.values(
            axis,
            distance_rows,
            X,
            "logical_error_rate_per_round",
            "decoder",
            low="logical_error_rate_per_round_low",
            high="logical_error_rate_per_round_high",
            order=[UNION_FIND_ALONE, SWITCHING],
        )
        axis.set_ylabel("logical error rate per round")
        axis.set_yscale("log")
        rate_axis(axis)
    plots.save(figure, path)


def strong_figure(rows: list, path: pathlib.Path) -> None:
    """The strong decoder's busy time, and its decode time over its budget.

    The budget is strong_service_bound_us, the longest mean decode time
    at which the strong decoder keeps up with the escalations
    (decsim/experiments/report.py strong_service_bound_us).
    """
    escalating_rows = []
    for row in rows:
        row["time busy (%)"] = 100 * row["strong_busy_fraction"]
        mean = row["strong_service_mean_us"]
        if mean is None:
            continue
        bound = row["strong_service_bound_us"]
        row["mean decode time / budget"] = mean / bound
        escalating_rows.append(row)
    # the two panels' values differ in kind, so they share no y axis
    figure, figure_axes = pyplot.subplots(
        1,
        2,
        figsize=(2 * plots.PANEL_WIDTH_INCHES, plots.PANEL_HEIGHT_INCHES),
        layout="constrained",
    )
    busy_axis, budget_axis = figure_axes
    plots.values(busy_axis, rows, X, "time busy (%)", "d", order=DISTANCES)
    busy_axis.set_title("strong decoder, time busy")
    plots.values(
        budget_axis,
        escalating_rows,
        X,
        "mean decode time / budget",
        "d",
        order=DISTANCES,
    )
    budget_axis.set_title("strong decoder, mean decode time over budget")
    budget_axis.set_yscale("log")
    for axis in figure_axes:
        rate_axis(axis)
    plots.save(figure, path)


def single_panel(title: str) -> tuple:
    """One axis, the size of one of decsim.plots' panels."""
    figure, axis = pyplot.subplots(
        figsize=(plots.PANEL_WIDTH_INCHES, plots.PANEL_HEIGHT_INCHES),
        layout="constrained",
    )
    axis.set_title(title)
    return figure, axis


def rate_axis(axis) -> None:
    """A log axis over the swept rates, labelled at every one."""
    axis.set_xscale("log")
    left_limit = min(ERROR_RATES) * 0.8
    right_limit = max(ERROR_RATES) * 1.25
    axis.set_xlim(left_limit, right_limit)
    rate_labels = [f"{rate:g}" for rate in ERROR_RATES]
    axis.set_xticks(ERROR_RATES, rate_labels, rotation=45)
    axis.xaxis.set_minor_formatter(pyplot.NullFormatter())
    axis.set_xlabel(X)


if __name__ == "__main__":
    results_folder = pathlib.Path(sys.argv[1])
    main(results_folder)
