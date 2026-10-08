"""The switching baseline's figures, drawn from its status.csv.

`python plot.py <results folder>` writes the folder's plots/, one
figure per question: the share of windows escalated, the formed-to-
commit latency, the logical error rate of switching against union-find
alone, whether the strong decoder keeps up, each decoder's decode time
per window by distance, and where a window's time goes. The folder
holds status.csv, as `decsim status` writes it, configurations.csv,
each configuration's id and name, and two tables of each
configuration's window_samples.csv: decode_time.csv, the "algorithm"
counts, and stage_means.csv, each stage's mean time per window. In
the rate figure a point with no failures is a hollow marker at its
upper bound, and a point failing more than half its shots is left
out, as decsim.plots.error_rate draws them.
"""

import csv
import json
import pathlib
import sys

import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.pyplot as pyplot
import numpy

import decsim.experiments.failure_statistics as failure_statistics
import decsim.plots as plots

DISTANCES = [5, 7, 9, 11, 13]
ERROR_RATES = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005]
SWITCHING = "switching"
UNION_FIND_ALONE = "union-find alone"
TESSERACT = "Tesseract alone (whole shot)"
# the decoder baseline beside this folder: sinter's Tesseract on the same
# Stim circuit, the Z memory at one rate on all four noise channels and
# 100 rounds, decoding each shot whole rather than in windows
TESSERACT_FOLDER = "2026-09-27_decoder_baseline"
TESSERACT_DECODER = "tesseract-short-beam"
TESSERACT_ROUNDS = 100
TESSERACT_TARGET_ERRORS = 100
# configurations.csv's names, in the figures' words
SHOWN_NAME = {
    "switching_baseline": SWITCHING,
    "weak_alone": UNION_FIND_ALONE,
}
X = "physical error rate"
RATE = "logical_error_rate_per_round"
TIER_COLOR = {"weak": "C0", "strong": "C1"}
TIER_NAME = {"weak": "union-find (weak)", "strong": "Relay-BP-5 (strong)"}
VIOLIN_HALF_WIDTH = 0.8
VIOLIN_BINS = 40
# A window's path through the machine, one segment per place in the
# order the window passes it, after Liu et al. 2603.16203 Fig. 8. Each
# segment has its own hue, from the Okabe-Ito and Paul Tol palettes; a
# segment in both paths keeps one color. Each
# segment is the recorded points of decsim/experiments/measure.py it
# adds, less any it subtracts, and a path's segments add up to its
# span. A kept window's path starts when the window is formed, an
# escalated window's at its first round.
#
# dep_block, the wait between the verdict and the decode's start, holds
# two places on an escalated path: the escalation link,
# escalation_link_per_window, which ends before the strong job is
# queued, and the wait for a free input slot on the strong decoder
# after it. On a kept path it is the wait for the previous window's
# boundary.
KEPT_PATH = {
    "union-find queue": ("#f0e442", ["admission_wait", "queue_wait"], []),
    "wait for previous window's boundary": ("#e69f00", ["dep_block"], []),
    "into union-find memory": ("#88ccee", ["input_link_per_window"], []),
    "union-find busy with the window ahead": (
        "#332288",
        ["compute_wait"],
        [],
    ),
    "union-find decode": ("#0072b2", ["fetch", "algorithm", "release"], []),
    "confidence check": ("#882255", ["confidence"], []),
    "to the Pauli frame": (
        "#009e73",
        ["selection_wait", "output_link_per_window", "frame_commit"],
        [],
    ),
}
ESCALATED_PATH = {
    "rounds arriving in the weak buffer": ("#bbbbbb", ["buffer_fill"], []),
    "union-find queue": ("#f0e442", ["admission_wait", "queue_wait"], []),
    "union-find decode and confidence check": (
        "#0072b2",
        ["weak_attempt", "confidence"],
        [],
    ),
    "extra rounds and link to Relay-BP-5": (
        "#44aa99",
        ["escalation_link_per_window"],
        [],
    ),
    "Relay-BP-5 queue: both input slots full": (
        "#cc79a7",
        ["dep_block"],
        ["escalation_link_per_window"],
    ),
    "into Relay-BP-5 memory": ("#ddcc77", ["input_link_per_window"], []),
    "Relay-BP-5 busy with the window ahead": (
        "#d55e00",
        ["compute_wait"],
        [],
    ),
    "Relay-BP-5 decode": ("#6a3d9a", ["fetch", "algorithm", "release"], []),
    "to the Pauli frame": (
        "#009e73",
        ["selection_wait", "output_link_per_window", "frame_commit"],
        [],
    ),
}
# each tier's path, the recorded span it adds up to, and where it starts
TIER_PATH = {
    "weak": (KEPT_PATH, "buffer0_ready_to_frame", "window formed"),
    "strong": (ESCALATED_PATH, "buffer0_first_round_to_frame", "first round"),
}
BREAKDOWN_ERROR_RATE = 0.003
# the escalated window the path figure spreads out, one segment a bar
PATH_DETAIL_DISTANCE = 9
TIER_UNIT = {"weak": ("µs", 1.0), "strong": ("ms", 1e3)}
TIER_WINDOWS = {
    "weak": "windows union-find kept",
    "strong": "windows escalated to Relay-BP-5",
}


def main(folder: pathlib.Path) -> None:
    """Every figure of the folder's status.csv, into its plots/."""
    rows = rows_of(folder)
    plots_folder = folder / "plots"
    plots_folder.mkdir(exist_ok=True)
    switching_rows = [row for row in rows if row["decoder"] == SWITCHING]
    escalated_path = plots_folder / "escalated.png"
    latency_path = plots_folder / "latency.png"
    rate_path = plots_folder / "logical_error_rate.png"
    rate_with_tesseract_path = (
        plots_folder / "logical_error_rate_with_tesseract.png"
    )
    strong_path = plots_folder / "strong_decoder.png"
    decode_time_path = plots_folder / "decode_time.png"
    escalated_figure(switching_rows, escalated_path)
    latency_figure(rows, latency_path)
    tesseract = tesseract_rows(folder.parent / TESSERACT_FOLDER)
    error_rate_figure(rows, rate_path)
    error_rate_figure(rows + tesseract, rate_with_tesseract_path)
    strong_figure(switching_rows, strong_path)
    breakdown_path = plots_folder / "latency_breakdown.png"
    histograms = decode_time_histograms(folder)
    decode_time_figure(histograms, decode_time_path)
    stage_means = stage_means_of(folder)
    breakdown_figure(stage_means, breakdown_path)
    path_detail_path = plots_folder / "escalated_path.png"
    path_detail_figure(stage_means, path_detail_path)


def rows_of(folder: pathlib.Path) -> list:
    """The points with data, as numbers, each named by its decoder."""
    configurations_path = folder / "configurations.csv"
    status_path = folder / "status.csv"
    with open(configurations_path) as handle:
        configuration_reader = csv.DictReader(handle)
        configurations = list(configuration_reader)
    with open(status_path) as handle:
        status_reader = csv.DictReader(handle)
        statuses = list(status_reader)
    name_by_id = {}
    for configuration in configurations:
        name = configuration["name"]
        name_by_id[configuration["configuration_id"]] = SHOWN_NAME[name]
    rows = []
    for status in statuses:
        if status["state"] != "no data":
            row = row_of(status, name_by_id)
            rows.append(row)
    return rows


def row_of(status: dict, name_by_id: dict) -> dict:
    """One status.csv row as numbers, named by its decoder."""
    error_rate_text = status["workload.arguments.physical_error_probability"]
    row = {"decoder": name_by_id[status["configuration_id"]]}
    row["d"] = int(status["qpu.distance"])
    row[X] = float(error_rate_text)
    for column, text in status.items():
        if column not in row:
            row[column] = number(text)
    return row


def tesseract_rows(folder: pathlib.Path) -> list:
    """The decoder baseline's Tesseract points in Z, as error rate rows.

    A point that reached the target error count stopped on it; one short
    of it was stopped by the run's time budget, a cap. The per-round
    rates go through the same map as the machine's rows.
    """
    with open(folder / "stats.csv") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        header = [column.strip() for column in next(reader)]
        stats = [dict(zip(header, values)) for values in reader]
    rows = []
    for stat in stats:
        metadata = json.loads(stat["json_metadata"])
        is_tesseract = stat["decoder"] == TESSERACT_DECODER
        if not is_tesseract or metadata["basis"] != "z":
            continue
        if metadata["d"] not in DISTANCES:
            continue
        failures = int(stat["errors"])
        shots = int(stat["shots"]) - int(stat["discards"])
        stop_kind = failure_statistics.StopKind.CAP
        if failures >= TESSERACT_TARGET_ERRORS:
            stop_kind = failure_statistics.StopKind.TARGET
        estimate = failure_statistics.estimate(failures, shots, stop_kind)
        row = {"decoder": TESSERACT, "d": metadata["d"], X: metadata["p"]}
        row["logical_failures"] = failures
        row["is_shot_rate_above_half"] = str(
            estimate.rate is not None and estimate.rate > 0.5
        )
        row[RATE] = per_round(estimate.rate)
        row[f"{RATE}_low"] = per_round(estimate.low)
        row[f"{RATE}_high"] = per_round(estimate.high)
        rows.append(row)
    return rows


def per_round(shot_rate):
    """A shot rate over the baseline's rounds as a per-round rate."""
    if shot_rate is None:
        return None
    return failure_statistics.per_round_rate(shot_rate, TESSERACT_ROUNDS)


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
    """Switching against union-find alone, and Tesseract alone if in rows.

    A point failing more than half its shots has no per-round rate
    (decsim/experiments/report.py _is_above_half).
    """
    decoders = [UNION_FIND_ALONE, SWITCHING, TESSERACT]
    failed_rows = [row for row in rows if has_per_round_rate(row)]
    error_free_rows = [row for row in rows if row["logical_failures"] == 0]
    figure, axis_by_distance = plots.panels("d", DISTANCES)
    for distance, axis in axis_by_distance.items():
        distance_rows = [row for row in failed_rows if row["d"] == distance]
        bound_rows = [row for row in error_free_rows if row["d"] == distance]
        plots.values(
            axis,
            distance_rows,
            X,
            RATE,
            "decoder",
            low=f"{RATE}_low",
            high=f"{RATE}_high",
            order=decoders,
        )
        plots.bounds(
            axis, bound_rows, X, f"{RATE}_high", "decoder", order=decoders
        )
        axis.set_ylabel("logical error rate per round")
        axis.set_yscale("log")
        rate_axis(axis)
    plots.save(figure, path)


def has_per_round_rate(row: dict) -> bool:
    """Whether a point failed, on at most half its shots."""
    if row["logical_failures"] == 0:
        return False
    return row["is_shot_rate_above_half"] == "False"


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
    figure_width = 2 * plots.PANEL_WIDTH_INCHES
    figure_size = (figure_width, plots.PANEL_HEIGHT_INCHES)
    figure, figure_axes = pyplot.subplots(
        1, 2, figsize=figure_size, layout="constrained"
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


def decode_time_histograms(folder: pathlib.Path) -> dict:
    """The switching run's decode times, a value-to-count map per window kind.

    The key is (physical error rate, distance, tier, round period in us).
    """
    decode_time_path = folder / "decode_time.csv"
    with open(decode_time_path) as handle:
        decode_time_rows = list(csv.DictReader(handle))
    histograms = {}
    for row in decode_time_rows:
        if SHOWN_NAME[row["configuration"]] != SWITCHING:
            continue
        error_rate = float(row["physical_error_rate"])
        distance = int(row["distance"])
        round_period = float(row["round_period_us"])
        key = (error_rate, distance, row["tier"], round_period)
        histogram = histograms.setdefault(key, {})
        value = float(row["value_us"])
        histogram[value] = histogram.get(value, 0) + int(row["count"])
    return histograms


def decode_time_figure(histograms: dict, path: pathlib.Path) -> None:
    """Decode time per window against distance, a panel per error rate.

    Each violin is one tier's windows at one distance and rate, its width
    the share of windows at that time; a line marks the median and a
    triangle the slowest window. The dashed line is the time the chip
    takes to measure one window, distance rounds of one round period.
    """
    figure, axis_by_rate = plots.panels(X, ERROR_RATES)
    for (error_rate, distance, tier, _), histogram in histograms.items():
        axis = axis_by_rate[error_rate]
        color = TIER_COLOR[tier]
        draw_violin(axis, distance, histogram, color)
    round_periods = {key[3] for key in histograms}
    (round_period,) = round_periods
    window_times = [distance * round_period for distance in DISTANCES]
    for axis in axis_by_rate.values():
        axis.plot(DISTANCES, window_times, "--", color="gray")
        axis.set_yscale("log")
        axis.set_xticks(DISTANCES)
        axis.set_xlabel("code distance")
        axis.set_ylabel("decode time per window (us)")
        axis.grid(True, which="major", alpha=plots.MINOR_GRID_ALPHA)
    legend_handles = decode_time_legend(round_period)
    figure.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncols=len(legend_handles),
    )
    plots.save(figure, path)


def draw_violin(axis, position: int, histogram: dict, color: str) -> None:
    """One violin of a value-to-count map, on a log axis.

    The widths are counts in equal steps of log time, so a tier spanning
    four decades keeps its shape on the log axis.
    """
    values = numpy.array(sorted(histogram))
    counts = numpy.array([histogram[value] for value in values])
    log_values = numpy.log10(values)
    median = weighted_median(values, counts)
    slowest = values[-1]
    axis.hlines(median, position - 0.5, position + 0.5, color=color)
    axis.plot(position, slowest, "v", color=color)
    if log_values[0] == log_values[-1]:
        return
    bin_counts, bin_edges = numpy.histogram(
        log_values, bins=VIOLIN_BINS, weights=counts
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    half_widths = VIOLIN_HALF_WIDTH * bin_counts / bin_counts.max()
    center_times = 10**bin_centers
    left = position - half_widths
    right = position + half_widths
    axis.fill_betweenx(center_times, left, right, color=color, alpha=0.5)


def weighted_median(values, counts) -> float:
    """The value at which half the counted windows are at or below."""
    cumulative_counts = numpy.cumsum(counts)
    half_count = cumulative_counts[-1] / 2
    median_index = numpy.searchsorted(cumulative_counts, half_count)
    return values[median_index]


def decode_time_legend(round_period: float) -> list:
    """The decode-time figure's legend: the tiers, the marks, the window."""
    handles = []
    for tier, color in TIER_COLOR.items():
        tier_patch = patches.Patch(
            color=color, alpha=0.5, label=TIER_NAME[tier]
        )
        handles.append(tier_patch)
    median_line = lines.Line2D([], [], color="black", label="median")
    slowest_mark = lines.Line2D(
        [],
        [],
        color="black",
        marker="v",
        linestyle="None",
        label="slowest window",
    )
    window_label = f"one window measured ({round_period:g} us rounds)"
    window_line = lines.Line2D(
        [], [], color="gray", linestyle="--", label=window_label
    )
    handles.extend([median_line, slowest_mark, window_line])
    return handles


def stage_means_of(folder: pathlib.Path) -> dict:
    """The switching run's mean time per window of each stage, in us.

    The key is (physical error rate, distance, tier, stage).
    """
    stage_means_path = folder / "stage_means.csv"
    with open(stage_means_path) as handle:
        stage_rows = list(csv.DictReader(handle))
    stage_means = {}
    for row in stage_rows:
        if SHOWN_NAME[row["configuration"]] != SWITCHING:
            continue
        error_rate = float(row["physical_error_rate"])
        distance = int(row["distance"])
        key = (error_rate, distance, row["tier"], row["stage"])
        stage_means[key] = float(row["mean_us"])
    return stage_means


def breakdown_figure(stage_means: dict, path: pathlib.Path) -> None:
    """Where a window's time goes at one error rate, kept and escalated.

    Each bar is one distance's mean time per window along its tier's
    path (TIER_PATH). Means, unlike medians, add up, so a bar's length
    is the mean of the path's recorded span: the number at its end.
    """
    tiers = list(TIER_WINDOWS)
    figure_width = len(tiers) * plots.PANEL_WIDTH_INCHES * 1.3
    figure_height = plots.PANEL_HEIGHT_INCHES * 1.4
    figure, axes = pyplot.subplots(
        1,
        len(tiers),
        figsize=(figure_width, figure_height),
        layout="constrained",
    )
    for axis, tier in zip(axes, tiers, strict=True):
        draw_breakdown(axis, stage_means, BREAKDOWN_ERROR_RATE, tier)
    plots.save(figure, path)


def segment_mean_us(stage_means: dict, point: tuple, segment: tuple) -> float:
    """A segment's mean time per window: its points added, less any dropped."""
    _, added_stages, subtracted_stages = segment
    total_us = 0.0
    for stage in added_stages:
        total_us += stage_means[(*point, stage)]
    for stage in subtracted_stages:
        total_us -= stage_means[(*point, stage)]
    return total_us


def draw_breakdown(
    axis, stage_means: dict, error_rate: float, tier: str
) -> None:
    """One tier's stacked bars at one rate, its path's legend underneath."""
    unit_name, unit_us = TIER_UNIT[tier]
    path_segments, span_stage, span_start = TIER_PATH[tier]
    distances = []
    for distance in DISTANCES:
        if (error_rate, distance, tier, span_stage) in stage_means:
            distances.append(distance)
    lefts = [0.0] * len(distances)
    positions = list(range(len(distances)))
    legend_handles = []
    for segment_name, segment in path_segments.items():
        color = segment[0]
        widths = []
        for distance in distances:
            point = (error_rate, distance, tier)
            mean_us = segment_mean_us(stage_means, point, segment)
            widths.append(mean_us / unit_us)
        axis.barh(positions, widths, left=lefts, height=0.6, color=color)
        lefts = [
            left + width for left, width in zip(lefts, widths, strict=True)
        ]
        segment_patch = patches.Patch(color=color, label=segment_name)
        legend_handles.append(segment_patch)
    for position, total in zip(positions, lefts, strict=True):
        axis.text(total, position, f" {total:.4g}", va="center", fontsize=8)
    distance_labels = [f"d={distance}" for distance in distances]
    axis.set_yticks(positions, distance_labels)
    axis.invert_yaxis()
    widest = max(lefts, default=1.0)
    axis.set_xlim(0, widest * 1.2)
    axis.set_xlabel(
        f"mean time per window, {span_start} to Pauli frame ({unit_name})"
    )
    axis.set_title(f"{TIER_WINDOWS[tier]}, {X} = {error_rate:g}")
    axis.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncols=2,
        fontsize=8,
        frameon=False,
    )


def path_detail_figure(stage_means: dict, path: pathlib.Path) -> None:
    """One escalated window's path, a bar per segment, on a log axis.

    The segments of an escalated window run from microseconds to tens of
    milliseconds, so on the breakdown's linear axis the first ones are
    under a pixel; here each has its own bar, in the order the window
    passes them, with its mean time at its end. A segment of zero time
    has no bar on a log axis, so it shows its value alone.
    """
    tier = "strong"
    path_segments, span_stage, span_start = TIER_PATH[tier]
    point = (BREAKDOWN_ERROR_RATE, PATH_DETAIL_DISTANCE, tier)
    names = list(path_segments)
    means_us = []
    for segment in path_segments.values():
        mean_us = segment_mean_us(stage_means, point, segment)
        means_us.append(mean_us)
    colors = [segment[0] for segment in path_segments.values()]
    positions = list(range(len(names)))
    figure, axis = pyplot.subplots(
        figsize=(plots.PANEL_WIDTH_INCHES * 1.8, plots.PANEL_HEIGHT_INCHES),
        layout="constrained",
    )
    axis.barh(positions, means_us, height=0.6, color=colors, log=True)
    smallest = min(mean for mean in means_us if mean > 0)
    for position, mean_us in zip(positions, means_us, strict=True):
        label_at = max(mean_us, smallest / 2)
        axis.text(
            label_at, position, f" {mean_us:,.3f}", va="center", fontsize=8
        )
    axis.set_yticks(positions, names)
    axis.invert_yaxis()
    axis.set_xlim(smallest / 4, max(means_us) * 20)
    axis.set_xlabel("mean time per window (µs, log scale)")
    total_us = stage_means[(*point, span_stage)]
    axis.set_title(
        f"{TIER_WINDOWS[tier]}, d={PATH_DETAIL_DISTANCE}, "
        f"{X} = {BREAKDOWN_ERROR_RATE:g}\n"
        f"{span_start} to Pauli frame: {total_us:,.3f} µs in all"
    )
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
    no_labels = pyplot.NullFormatter()
    axis.xaxis.set_minor_formatter(no_labels)
    axis.set_xlabel(X)


if __name__ == "__main__":
    results_folder = pathlib.Path(sys.argv[1])
    main(results_folder)
