"""The switching-per-window run's figures, one question each.

`python plot.py <results folder>` reads the tables tables.py wrote
there and writes its plots/. Each figure answers one question:

Shots
 01 What is each configuration's logical error rate (per round)?
 02 How many windows does each configuration decode in a shot?
Union-find alone
 03 What share of union-find's windows are wrong?
All configurations
 04 What share of each configuration's windows are wrong alone?
Switching
 05 What share of union-find's windows does switching escalate?
 06 Of the escalated windows, what share were false positives?
 07 Of union-find's windows, what share are wrong, by decision?
 08 Does the gap separate union-find's wrong windows from right ones?
 09 Do the kept windows union-find gets wrong have more detection events?
Relay-BP-5 on the escalated windows
 10 Of those union-find got wrong, what share did Relay-BP-5 fix?
 11 Of those union-find got right, what share did Relay-BP-5 break?
 12 On what share did Relay-BP-5 not converge?
Windows and shots
 13 Of the shots with a wrong window, what share fail?

A share is a count over a total, drawn with its 95% Wilson interval
when the total holds at least SHOWN_TOTAL windows or shots. Figure 01
draws a rate only with SHOWN_FAILURES failures (LOG.md, stop rule).
"""

import collections
import csv
import pathlib
import sys

import matplotlib.pyplot as pyplot
import matplotlib.ticker as ticker
import numpy
import tables

import decsim.experiments.failure_statistics as failure_statistics

DISTANCES = (5, 7, 9, 11)
PHYSICAL_ERROR_RATES = (0.002, 0.003, 0.004, 0.005)
# Experiment 1's switching threshold, on the cluster gap
THRESHOLD_DECIBELS = 20.0
ROUNDS_PER_SHOT = 100
SHOWN_FAILURES = tables.SHOWN_FAILURES
SHOWN_TOTAL = 20
# each configuration's name and color, the same in every figure
CONFIGURATIONS = {
    "union_find_alone": ("union-find alone", "C0"),
    "switching": ("switching", "C1"),
    "relay_bp5_alone": ("Relay-BP-5 alone", "C2"),
    "tesseract_alone": ("Tesseract alone", "C3"),
}
# each distance's color, apart from the configurations' four
DISTANCE_COLORS = {5: "C4", 7: "C5", 9: "C6", 11: "C7"}
ANSWER_NAMES = {
    "right": ("right", "C2"),
    "wrong_alone": ("wrong alone", "C3"),
    "wrong_paired": ("wrong in a seam pair", "C1"),
}
GAP_BIN_DECIBELS = 2
EVENT_BINS = 40
# the gap figure's last bin holds every gap at or above this
GAP_TOP_DECIBELS = 60
GAP_X_LABEL = "cluster gap (dB); the last bin holds 60 dB and above"
ESCALATED = tuple(f"escalated_{name}" for name in tables.ANSWER_CLASSES)
KEPT = tuple(f"kept_{name}" for name in tables.ANSWER_CLASSES)
WRONG_CLASSES = tables.ANSWER_CLASSES[1:]
# the gap figure's lines: every wrong window together, as an escalated
# window's seam partner is unseen
GAP_GROUPS = {
    "right": (("right",), "C2"),
    "wrong": (WRONG_CLASSES, "C3"),
}
# the detection-event figure holds kept windows, whose pairs are seen
EVENT_GROUPS = {
    "right": (("right",), "C2"),
    "wrong alone": (("wrong_alone",), "C3"),
    "wrong in a seam pair": (("wrong_paired",), "C1"),
    "wrong beside a skipped window": (("wrong_partner_unseen",), "C7"),
}
WRONG_BARS = {
    "kept_wrong_alone": (
        "kept, wrong alone (false negative)",
        ("kept_wrong_alone",),
        "C3",
    ),
    "kept_wrong_paired": (
        "kept, wrong in a seam pair",
        ("kept_wrong_paired",),
        "C1",
    ),
    "kept_wrong_partner_unseen": (
        "kept, wrong beside a window union-find skipped",
        ("kept_wrong_partner_unseen",),
        "C7",
    ),
    "escalated_wrong": (
        "escalated, wrong (correct escalation)",
        tuple(f"escalated_{name}" for name in WRONG_CLASSES),
        "C4",
    ),
}
WRONG_BAR_NAMES = {key: (bar[0], bar[2]) for key, bar in WRONG_BARS.items()}
SINGLE_SIZE_INCHES = (4.8, 3.6)
PANELS_SIZE_INCHES = (9.6, 7.2)
DOTS_PER_INCH = 150
GRID_ALPHA = 0.3
LEGEND_FONT_SIZE = 8
TITLE_FONT_SIZE = 10
BAR_GROUP_WIDTH = 0.8
BAR_HEADROOM = 1.35
X_LABEL = "physical error rate"
X_MARGIN = 0.0003


def main(results_dir: pathlib.Path) -> None:
    """Every figure of the results folder's tables, into its plots/."""
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    shots = rows_of(results_dir / "shot_summary.csv")
    windows = rows_of(results_dir / "window_summary.csv")
    switching = rows_of(results_dir / "switching_summary.csv")
    gaps = rows_of(results_dir / "union_find_by_gap.csv")
    events = rows_of(results_dir / "kept_by_detection_events.csv")
    failure_figure(shots, plots_dir)
    decoded_figure(shots, plots_dir)
    union_find_wrong_figure(windows, plots_dir)
    wrong_alone_figure(windows, plots_dir)
    switching_figures(switching, plots_dir)
    wrong_windows_figure(switching, plots_dir)
    gap_figures(gaps, plots_dir)
    event_figures(events, plots_dir)
    strong_figures(switching, plots_dir)
    wrong_shots_figure(shots, plots_dir)


def failure_figure(rows: list, plots_dir: pathlib.Path) -> None:
    """01: each configuration's logical error rate per round, per d.

    A shot's failure rate and its Wilson bounds are each turned into a
    rate per round over the shot's 100 rounds, the form memory papers
    and Experiment 1 report.
    """
    title = "01 What is each configuration's logical error rate?"
    figure, axis_by_distance = distance_panels(title)
    for distance, axis in axis_by_distance.items():
        for configuration, (name, color) in CONFIGURATIONS.items():
            points = column_points(
                rows, configuration, distance, "failed_shots", "shots"
            )
            draw_shares(axis, points, name, color, SHOWN_FAILURES, per_round)
        axis.set_ylabel("logical error rate per round")
        axis.set_yscale("log")
        axis.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        rate_axis(axis)
    save(figure, plots_dir / "01_logical_error_rate.png")


def per_round(shot_share: float) -> float:
    """A share of failed shots, in %, as a rate per round."""
    shot_rate = shot_share / 100
    return failure_statistics.per_round_rate(shot_rate, ROUNDS_PER_SHOT)


def decoded_figure(rows: list, plots_dir: pathlib.Path) -> None:
    """02: windows decoded per shot, a bar per configuration."""
    title = "02 How many windows does each configuration decode in a shot?"
    figure, axis_by_distance = distance_panels(title)
    for distance, axis in axis_by_distance.items():
        values = {}
        for configuration in CONFIGURATIONS:
            points = column_points(
                rows, configuration, distance, "decoded_windows", "shots"
            )
            values[configuration] = per_shot(points)
        draw_bars(axis, values, CONFIGURATIONS)
        axis.set_ylabel("windows decoded per shot")
        bar_axis(axis)
    save(figure, plots_dir / "02_windows_decoded.png")


def union_find_wrong_figure(rows: list, plots_dir: pathlib.Path) -> None:
    """03: union-find alone's wrong windows, alone and in seam pairs."""
    title = "03 What share of union-find's windows are wrong?"
    figure, axis_by_distance = distance_panels(title)
    wrong_names = {key: ANSWER_NAMES[key] for key in ANSWER_NAMES}
    del wrong_names["right"]
    for distance, axis in axis_by_distance.items():
        values = {}
        for answer_class in wrong_names:
            values[answer_class] = answer_shares(
                rows, "union_find_alone", distance, answer_class
            )
        draw_bars(axis, values, wrong_names)
        axis.set_ylabel("union-find alone's windows (%)")
        bar_axis(axis)
    save(figure, plots_dir / "03_union_find_wrong.png")


def wrong_alone_figure(rows: list, plots_dir: pathlib.Path) -> None:
    """04: each configuration's windows wrong alone, a bar each."""
    title = "04 What share of each configuration's windows are wrong alone?"
    figure, axis_by_distance = distance_panels(title)
    for distance, axis in axis_by_distance.items():
        values = {}
        for configuration in CONFIGURATIONS:
            values[configuration] = answer_shares(
                rows, configuration, distance, "wrong_alone"
            )
        draw_bars(axis, values, CONFIGURATIONS)
        axis.set_ylabel("windows wrong alone (%)")
        bar_axis(axis)
    save(figure, plots_dir / "04_windows_wrong_alone.png")


def switching_figures(rows: list, plots_dir: pathlib.Path) -> None:
    """05 and 06: switching's escalations and its false positives."""
    single_figure(
        rows,
        (ESCALATED, ESCALATED + KEPT),
        "05 What share of union-find's windows\ndoes switching escalate?",
        "windows escalated (%)",
        plots_dir / "05_escalated.png",
    )
    single_figure(
        rows,
        (("escalated_right",), ESCALATED),
        "06 Of the escalated windows,\nwhat share were false positives?",
        "escalated windows union-find had right (%)",
        plots_dir / "06_false_positives.png",
    )


def wrong_windows_figure(rows: list, plots_dir: pathlib.Path) -> None:
    """07: union-find's wrong windows by what switching did with them.

    An escalated window's seam partner is unseen, so its wrong windows
    are one bar.
    """
    title = "07 Of union-find's windows, what share are wrong, by decision?"
    figure, axis_by_distance = distance_panels(title)
    for distance, axis in axis_by_distance.items():
        values = {}
        for key, (_name, columns, _color) in WRONG_BARS.items():
            values[key] = switching_shares(rows, distance, columns)
        draw_bars(axis, values, WRONG_BAR_NAMES)
        axis.set_ylabel("union-find's windows in switching (%)")
        bar_axis(axis)
    save(figure, plots_dir / "07_wrong_by_decision.png")


def gap_figures(rows: list, plots_dir: pathlib.Path) -> None:
    """08: the gap of right and wrong windows, a figure per rate."""
    edges = numpy.arange(0, GAP_TOP_DECIBELS + 1, GAP_BIN_DECIBELS)
    for rate in PHYSICAL_ERROR_RATES:
        title = (
            "08 Does the gap separate union-find's wrong windows from"
            f" right ones? (physical error rate {rate})"
        )
        figure, axis_by_distance = distance_panels(title)
        for distance, axis in axis_by_distance.items():
            counts = class_counts(rows, distance, rate, gap_bin_of)
            draw_distributions(axis, counts, GAP_GROUPS, edges)
            axis.axvline(
                THRESHOLD_DECIBELS,
                color="black",
                linestyle="--",
                label="threshold",
            )
            axis.set_xlabel(GAP_X_LABEL)
            finish_distribution(axis)
        save(figure, plots_dir / f"08_gap_p{rate}.png")


def event_figures(rows: list, plots_dir: pathlib.Path) -> None:
    """09: detection events of kept right and wrong windows, per rate."""
    for rate in PHYSICAL_ERROR_RATES:
        title = (
            "09 Do the kept windows union-find gets wrong have more"
            f" detection events? (physical error rate {rate})"
        )
        figure, axis_by_distance = distance_panels(title)
        for distance, axis in axis_by_distance.items():
            counts = class_counts(rows, distance, rate, events_of)
            edges = event_edges(counts)
            counts = rebinned(counts, edges)
            draw_distributions(axis, counts, EVENT_GROUPS, edges)
            axis.set_xlabel("detection events in the kept window")
            finish_distribution(axis)
        save(figure, plots_dir / f"09_detection_events_p{rate}.png")


def strong_figures(rows: list, plots_dir: pathlib.Path) -> None:
    """10 to 12: Relay-BP-5 on the escalated windows."""
    every_outcome = tables.STRONG_OUTCOMES
    single_figure(
        rows,
        (("fixed",), ("fixed", "both_wrong")),
        "10 Of the escalated windows union-find got wrong,\n"
        "what share did Relay-BP-5 fix?",
        "fixed by Relay-BP-5 (%)",
        plots_dir / "10_fixed.png",
    )
    single_figure(
        rows,
        (("broke",), ("broke", "both_right")),
        "11 Of the escalated windows union-find got right,\n"
        "what share did Relay-BP-5 break?",
        "broken by Relay-BP-5 (%)",
        plots_dir / "11_broken.png",
    )
    single_figure(
        rows,
        (("not_converged",), every_outcome),
        "12 On what share of the escalated windows\n"
        "did Relay-BP-5 not converge?",
        "escalated windows not converged (%)",
        plots_dir / "12_not_converged.png",
    )


def wrong_shots_figure(rows: list, plots_dir: pathlib.Path) -> None:
    """13: failed shots among the shots with a wrong window."""
    title = "13 Of the shots with a wrong window, what share fail?"
    figure, axis_by_distance = distance_panels(title)
    for distance, axis in axis_by_distance.items():
        for configuration, (name, color) in CONFIGURATIONS.items():
            points = column_points(
                rows,
                configuration,
                distance,
                "failed_shots",
                "shots_with_a_wrong_window",
            )
            draw_shares(axis, points, name, color, SHOWN_TOTAL)
        axis.set_ylabel("of these shots, failed (%)")
        rate_axis(axis)
    save(figure, plots_dir / "13_failed_among_wrong_shots.png")


def single_figure(
    rows: list, columns: tuple, title: str, y_label: str, path: pathlib.Path
) -> None:
    """One panel, a line per distance, of a share of switching's rows."""
    figure, axis = pyplot.subplots(
        figsize=SINGLE_SIZE_INCHES, layout="constrained"
    )
    axis.set_title(title, fontsize=TITLE_FONT_SIZE)
    draw_distance_lines(axis, rows, columns)
    axis.set_ylabel(y_label)
    rate_axis(axis)
    save(figure, path)


def draw_distance_lines(axis: pyplot.Axes, rows: list, columns: tuple) -> None:
    """A line per distance: the summed count columns over the total ones."""
    count_columns, total_columns = columns
    for distance in DISTANCES:
        points = {}
        for row in rows:
            if int(row["distance"]) != distance:
                continue
            count = column_sum(row, count_columns)
            total = column_sum(row, total_columns)
            points[float(row["physical_error_rate"])] = (count, total)
        color = DISTANCE_COLORS[distance]
        draw_shares(axis, points, f"d = {distance}", color, SHOWN_TOTAL)


def draw_shares(
    axis: pyplot.Axes,
    points: dict,
    label: str,
    color: str,
    shown: int,
    transform=None,
) -> None:
    """Count over total in % at each x, with Wilson bars.

    A point whose shown quantity (the count for failures, else the
    total) is below the rule is a gap in its line.
    """
    xs = sorted(points)
    ys = []
    below = []
    above = []
    for x in xs:
        share, low, high = shown_share(points[x], shown, transform)
        bar_below = share - low
        bar_above = high - share
        ys.append(share)
        below.append(bar_below)
        above.append(bar_above)
    is_hidden = numpy.isnan(ys)
    if is_hidden.all():
        return
    axis.errorbar(
        xs,
        ys,
        yerr=[below, above],
        fmt="o-",
        color=color,
        label=label,
        capsize=3,
    )


def shown_share(point: tuple, shown: int, transform=None) -> tuple:
    """A point's share and Wilson bounds in %, not a number when hidden.

    Failures are held to the count rule, other shares to the total.
    """
    count, total = point
    held = count if shown == SHOWN_FAILURES else total
    if held < shown or total == 0:
        return numpy.nan, numpy.nan, numpy.nan
    share = 100 * count / total
    low, high = tables.wilson_interval(count, total)
    if transform is None:
        return share, 100 * low, 100 * high
    return transform(share), transform(100 * low), transform(100 * high)


def draw_bars(axis: pyplot.Axes, values: dict, names: dict) -> None:
    """Grouped bars: a group per rate, a bar per key of values."""
    positions = numpy.arange(len(PHYSICAL_ERROR_RATES))
    bar_width = BAR_GROUP_WIDTH / len(values)
    for index, (key, heights) in enumerate(values.items()):
        name, color = names[key]
        offset = (index - (len(values) - 1) / 2) * bar_width
        bar_positions = positions + offset
        axis.bar(bar_positions, heights, bar_width, color=color, label=name)
    axis.set_xticks(positions, labels=PHYSICAL_ERROR_RATES)


def draw_distributions(
    axis: pyplot.Axes, counts: dict, groups: dict, edges
) -> None:
    """Each group's share of its windows per bin, as a histogram line.

    edges are the bins' low edges, the last bin's high edge one bin on.
    """
    bin_width = edges[1] - edges[0]
    stair_edges = numpy.append(edges, edges[-1] + bin_width)
    for name, (classes, color) in groups.items():
        windows = group_windows(counts, classes, edges)
        total = windows.sum()
        if total < SHOWN_TOTAL:
            continue
        shares = 100 * windows / total
        label = f"{name} ({total} windows)"
        axis.stairs(shares, stair_edges, color=color, label=label)
    axis.set_ylabel("share of the line's windows per bin (%)")


def group_windows(counts: dict, classes: tuple, edges) -> numpy.ndarray:
    """The windows of the classes in each bin, zero where none."""
    windows = numpy.zeros(len(edges), dtype=int)
    for answer_class in classes:
        class_bins = counts.get(answer_class, {})
        for position, edge in enumerate(edges):
            windows[position] += class_bins.get(edge, 0)
    return windows


def event_edges(counts: dict) -> numpy.ndarray:
    """EVENT_BINS bins of whole detection-event counts, from 0 up."""
    largest = 0
    for class_bins in counts.values():
        largest = max(largest, *class_bins)
    bin_width = max(1, -(-largest // EVENT_BINS))
    top = largest + bin_width
    return numpy.arange(0, top, bin_width)


def rebinned(counts: dict, edges) -> dict:
    """Counts by single value moved into the bins whose low edges these are."""
    bin_width = edges[1] - edges[0]
    moved = collections.defaultdict(collections.Counter)
    for answer_class, class_bins in counts.items():
        for value, windows in class_bins.items():
            low_edge = value // bin_width * bin_width
            moved[answer_class][low_edge] += windows
    return moved


def class_counts(rows: list, distance: int, rate: float, bin_of) -> dict:
    """Windows of one setting by answer class and by bin."""
    counts = collections.defaultdict(collections.Counter)
    for row in rows:
        if not _is_setting(row, distance, rate):
            continue
        class_bins = counts[row["union_find_answer"]]
        class_bins[bin_of(row)] += int(row["windows"])
    return counts


def gap_bin_of(row: dict) -> int:
    """A 1 dB row's bin of GAP_BIN_DECIBELS, the top bin holding the rest."""
    gap_low = float(row["gap_low_db"])
    bins = gap_low // GAP_BIN_DECIBELS
    gap_bin = int(bins * GAP_BIN_DECIBELS)
    return min(gap_bin, GAP_TOP_DECIBELS)


def switching_shares(rows: list, distance: int, columns: tuple) -> list:
    """The columns' share of union-find's windows in switching, per rate."""
    shares = []
    for rate in PHYSICAL_ERROR_RATES:
        share = numpy.nan
        for row in rows:
            if _is_setting(row, distance, rate):
                windows = column_sum(row, ESCALATED + KEPT)
                share = 100 * column_sum(row, columns) / windows
        shares.append(share)
    return shares


def events_of(row: dict) -> int:
    return int(row["detection_events"])


def column_points(
    rows: list, configuration: str, distance: int, count: str, total: str
) -> dict:
    """One configuration's (count, total) columns by physical error rate."""
    points = {}
    for row in rows:
        if row["configuration"] != configuration:
            continue
        if int(row["distance"]) == distance:
            rate = float(row["physical_error_rate"])
            points[rate] = (int(row[count]), int(row[total]))
    return points


def answer_shares(
    rows: list, configuration: str, distance: int, answer_class: str
) -> list:
    """One answer class's share of a configuration's windows, per rate."""
    shares = []
    for rate in PHYSICAL_ERROR_RATES:
        share = numpy.nan
        for row in rows:
            is_row = row["configuration"] == configuration
            if is_row and _is_setting(row, distance, rate):
                windows = column_sum(row, tables.ANSWER_CLASSES)
                share = 100 * int(row[answer_class]) / windows
        shares.append(share)
    return shares


def per_shot(points: dict) -> list:
    """Each rate's count per shot, not a number where it has no shots."""
    values = []
    for rate in PHYSICAL_ERROR_RATES:
        count, shots = points.get(rate, (0, 0))
        value = count / shots if shots else numpy.nan
        values.append(value)
    return values


def column_sum(row: dict, columns: tuple) -> int:
    total = 0
    for column in columns:
        total += int(row[column])
    return total


def distance_panels(title: str) -> tuple:
    """A 2 by 2 figure, a panel per distance, the question on top."""
    figure, axes = pyplot.subplots(
        2, 2, figsize=PANELS_SIZE_INCHES, layout="constrained"
    )
    figure.suptitle(title)
    axis_by_distance = {}
    for distance, axis in zip(DISTANCES, axes.flat, strict=True):
        axis.set_title(f"d = {distance}")
        axis_by_distance[distance] = axis
    return figure, axis_by_distance


def rate_axis(axis: pyplot.Axes) -> None:
    """The physical error rate on x, its four values as ticks."""
    axis.set_xlabel(X_LABEL)
    axis.set_xticks(PHYSICAL_ERROR_RATES)
    left = PHYSICAL_ERROR_RATES[0] - X_MARGIN
    right = PHYSICAL_ERROR_RATES[-1] + X_MARGIN
    axis.set_xlim(left, right)
    finish_axis(axis)


def bar_axis(axis: pyplot.Axes) -> None:
    """The rate groups on x, and headroom above the bars for the legend."""
    axis.set_xlabel(X_LABEL)
    _bottom, top = axis.get_ylim()
    axis.set_ylim(0, top * BAR_HEADROOM)
    finish_axis(axis)


def finish_distribution(axis: pyplot.Axes) -> None:
    axis.set_ylim(bottom=0)
    finish_axis(axis)


def finish_axis(axis: pyplot.Axes) -> None:
    """A light grid, and a legend when lines have labels."""
    axis.grid(alpha=GRID_ALPHA)
    handles, _labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(fontsize=LEGEND_FONT_SIZE)


def save(figure: pyplot.Figure, path: pathlib.Path) -> None:
    """The figure written to path and closed."""
    figure.savefig(path, dpi=DOTS_PER_INCH)
    pyplot.close(figure)


def rows_of(path: pathlib.Path) -> list:
    """A table's rows as dicts of text."""
    with path.open(newline="") as table_file:
        reader = csv.DictReader(table_file)
        return list(reader)


def _is_setting(row: dict, distance: int, rate: float) -> bool:
    is_distance = int(row["distance"]) == distance
    return is_distance and float(row["physical_error_rate"]) == rate


if __name__ == "__main__":
    results_folder = pathlib.Path(sys.argv[1])
    main(results_folder)
