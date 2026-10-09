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
 08 At each cluster gap, what share of union-find's windows are wrong?
 09 Do the kept windows union-find gets wrong have more detection events?
Relay-BP-5 on the escalated windows, seam pairs counted right
 10 Of those union-find got wrong, what share did Relay-BP-5 fix?
 11 Of those union-find got right, what share did Relay-BP-5 break?
 12 On what share did Relay-BP-5 not converge?
Windows and shots
 13 Of the shots with a wrong window, what share fail?
Switching, every window
 14 Of union-find's windows, which were kept or escalated, right or wrong?
Latency
 15 How long does each configuration take to react to a window?
 16 How long does each decoder take on a window, against the time the
    QPU takes to measure one?
Relay-BP-5 on the escalated windows, seam pairs counted right
 17 Where Relay-BP-5 and union-find alone differ, who was right?

A share is a count over a total, drawn with its 95% Wilson interval
when the total holds at least SHOWN_TOTAL windows or shots (TOTAL_RULE).
Figure 01 draws a rate only with SHOWN_FAILURES failures (FAILURE_RULE,
LOG.md, stop rule). Windows of one shot are not independent, so a
window share's interval is narrower than a shot-resampled one would be.
"""

import collections
import csv
import math
import pathlib
import sys
from collections.abc import Callable
from typing import Optional

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
# which number a point's display rule holds to, and its minimum
FAILURE_RULE = ("count", SHOWN_FAILURES)
TOTAL_RULE = ("total", SHOWN_TOTAL)
TABLE_NAMES = (
    "shot_summary",
    "window_summary",
    "switching_summary",
    "union_find_by_gap",
    "kept_by_detection_events",
    "reaction_time",
    "decode_time",
)
# each configuration's name and color, the same in every figure
CONFIGURATIONS = {
    "union_find_alone": ("Union-find alone", "C0", "o"),
    "switching": ("Switching", "C1", "s"),
    "relay_bp5_alone": ("Relay-BP-5 alone", "C2", "^"),
    "tesseract_alone": ("Tesseract alone", "C3", "D"),
}
# each window answer's color, never a configuration's
RIGHT_COLOR = "C7"
WRONG_ALONE_COLOR = "black"
WRONG_PAIRED_COLOR = "C8"
WRONG_UNSEEN_COLOR = "#8c6d31"
CORRECT_ESCALATION_COLOR = "#253494"
# each distance's line in the one-panel figures, never another color
DISTANCE_LINES = {
    5: ("d = 5", "C4", "o"),
    7: ("d = 7", "C5", "^"),
    9: ("d = 9", "C6", "s"),
    11: ("d = 11", "C9", "D"),
}
UNION_FIND_WRONG_NAMES = {
    "wrong_alone": ("Wrong alone", WRONG_ALONE_COLOR),
    "wrong_paired": ("Wrong in a seam pair", WRONG_PAIRED_COLOR),
}
GAP_BIN_DECIBELS = 4
# the gap figure's last bin holds every gap at or above this
GAP_TOP_DECIBELS = 60
GAP_X_LABEL = "Cluster gap (dB); last point: 60 dB and above"
EVENT_BINS = 40
ESCALATED = tuple(f"escalated_{name}" for name in tables.ANSWER_CLASSES)
KEPT = tuple(f"kept_{name}" for name in tables.ANSWER_CLASSES)
DECIDED = ESCALATED + KEPT
WRONG_CLASSES = tables.ANSWER_CLASSES[1:]
# the detection-event figure holds kept windows only
EVENT_GROUPS = {
    "Right": (("right",), RIGHT_COLOR),
    "Wrong alone": (("wrong_alone",), WRONG_ALONE_COLOR),
    "Wrong in a seam pair": (("wrong_paired",), WRONG_PAIRED_COLOR),
    "Wrong beside a skipped window": (
        ("wrong_partner_unseen",),
        WRONG_UNSEEN_COLOR,
    ),
}
# figure 07's bars: name, the columns each sums, color
WRONG_BARS = {
    "kept_wrong_alone": (
        "Kept, wrong alone (false negative)",
        ("kept_wrong_alone",),
        WRONG_ALONE_COLOR,
    ),
    "kept_wrong_paired": (
        "Kept, wrong in a seam pair",
        ("kept_wrong_paired",),
        WRONG_PAIRED_COLOR,
    ),
    "kept_wrong_partner_unseen": (
        "Kept, wrong beside a skipped window",
        ("kept_wrong_partner_unseen",),
        WRONG_UNSEEN_COLOR,
    ),
    "escalated_wrong": (
        "Escalated, wrong (correct escalation)",
        tuple(f"escalated_{name}" for name in WRONG_CLASSES),
        CORRECT_ESCALATION_COLOR,
    ),
}
WRONG_BAR_NAMES = {key: (bar[0], bar[2]) for key, bar in WRONG_BARS.items()}
# figure 17's bars: Relay-BP-5 against union-find alone on the escalated
# windows, seam pairs counted right; both right is left out, as figure 07
# leaves out union-find's right windows
RELAY_BARS = {
    "fixed": (
        "Fixed: Relay-BP-5 right, union-find wrong",
        ("fixed",),
        "#1b7837",
    ),
    "broke": (
        "Broke: Relay-BP-5 wrong, union-find right",
        ("broke",),
        "#c51b7d",
    ),
    "both_wrong": ("Both wrong", ("both_wrong",), "#4d4d4d"),
    "not_converged": (
        "Relay-BP-5 did not converge",
        (tables.NOT_CONVERGED,),
        "#e6ab02",
    ),
}
RELAY_BAR_NAMES = {key: (bar[0], bar[2]) for key, bar in RELAY_BARS.items()}
# one-panel figures of switching_summary.csv: file, (count columns,
# total columns), title, y label
SHARE_FIGURES = (
    (
        "05_escalated.png",
        (ESCALATED, DECIDED),
        "Windows switching escalates",
        "Escalated (% of windows)"
    ),
    (
        "06_false_positives.png",
        (("escalated_right",), ESCALATED),
        "False positives among escalations",
        "Union-find was right (% of escalated)"
    ),
    (
        "10_fixed.png",
        (("fixed",), ("fixed", "both_wrong")),
        "Escalated windows that union-find got wrong",
        "Relay-BP-5 got them right (%)"
    ),
    (
        "11_broken.png",
        (("broke",), ("broke", "both_right")),
        "Escalated windows that union-find got right",
        "Relay-BP-5 got them wrong (%)"
    ),
    (
        "12_not_converged.png",
        (("not_converged",), tables.STRONG_OUTCOMES),
        "Escalated windows",
        "Relay-BP-5 did not converge (%)"
    ),
)
# figure 14's parts of a bar: name, the columns each sums, color
DECISION_PARTS = (
    ("Kept, union-find right", ("kept_right",), "#d9d9d9"),
    (
        "Kept, union-find wrong (false negative)",
        tuple(f"kept_{name}" for name in WRONG_CLASSES),
        "#252525",
    ),
    (
        "Escalated, union-find right (false positive)",
        ("escalated_right",),
        "#fdae6b",
    ),
    (
        "Escalated, union-find wrong (correct escalation)",
        tuple(f"escalated_{name}" for name in WRONG_CLASSES),
        "#d94801",
    ),
)
# the space between one distance's bars and the next's, in bar heights
DECISION_GROUP_GAP = 0.6
DECISION_SIZE_INCHES = (13.5, 6.8)
# figure 16's decoders, each from the configuration that runs it alone
DECODERS = {
    "union_find_alone": ("Union-find", "C0", "o"),
    "relay_bp5_alone": ("Relay-BP-5", "C2", "^"),
    "tesseract_alone": ("Tesseract", "C3", "D"),
}
DECODE_DISTANCE_LIMITS = (4.5, 12)
SINGLE_SIZE_INCHES = (4.8, 3.6)
PANELS_SIZE_INCHES = (9.6, 7.2)
DOTS_PER_INCH = 150
GRID_ALPHA = 0.3
LEGEND_FONT_SIZE = 8
TITLE_FONT_SIZE = 10
BAR_GROUP_WIDTH = 0.8
BAR_HEADROOM = 1.35
X_LABEL = "Physical error rate"
X_MARGIN = 0.0003


def main(results_dir: pathlib.Path) -> None:
    """Every figure of the results folder's tables, into its plots/."""
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    rows = {}
    for name in TABLE_NAMES:
        table_path = results_dir / f"{name}.csv"
        rows[name] = rows_of(table_path)
    switching = rows["switching_summary"]
    failure_figure(rows["shot_summary"], plots_dir)
    decoded_figure(rows["shot_summary"], plots_dir)
    union_find_wrong_figure(rows["window_summary"], plots_dir)
    wrong_alone_figure(rows["window_summary"], plots_dir)
    for figure_text in SHARE_FIGURES:
        share_figure(switching, figure_text, plots_dir)
    wrong_windows_figure(switching, plots_dir)
    gap_figures(rows["union_find_by_gap"], plots_dir)
    event_figures(rows["kept_by_detection_events"], plots_dir)
    wrong_shots_figure(rows["shot_summary"], plots_dir)
    decisions_figure(switching, plots_dir)
    reaction_figures(rows["reaction_time"], plots_dir)
    decode_figure(rows["decode_time"], plots_dir)
    relay_outcomes_figure(switching, plots_dir)


def failure_figure(rows: list, plots_dir: pathlib.Path) -> None:
    """01: each configuration's logical error rate per round, per d.

    A shot's failure rate and its Wilson bounds are each turned into a
    rate per round over the shot's 100 rounds, the form memory papers
    and Experiment 1 report.
    """
    title = "Logical error rate per round"
    figure, axis_by_distance = distance_panels(title)
    plain_numbers = ticker.FormatStrFormatter("%g")
    for distance, axis in axis_by_distance.items():
        for configuration, line in CONFIGURATIONS.items():
            columns = ("failed_shots", "shots")
            points = column_points(rows, configuration, distance, columns)
            draw_shares(axis, points, line, FAILURE_RULE, per_round)
        axis.set_ylabel("Logical error rate per round")
        axis.set_yscale("log")
        axis.yaxis.set_major_formatter(plain_numbers)
        rate_axis(axis)
    save(figure, plots_dir, "01_logical_error_rate.png")


def decoded_figure(rows: list, plots_dir: pathlib.Path) -> None:
    """02: windows decoded per shot, a bar per configuration."""
    title = "Windows decoded per shot"
    figure, axis_by_distance = distance_panels(title)
    for distance, axis in axis_by_distance.items():
        values = {}
        for configuration in CONFIGURATIONS:
            columns = ("decoded_windows", "shots")
            points = column_points(rows, configuration, distance, columns)
            values[configuration] = per_shot(points)
        draw_bars(axis, values, CONFIGURATIONS)
        axis.set_ylabel("Windows per shot")
        bar_axis(axis)
    save(figure, plots_dir, "02_windows_decoded.png")


def union_find_wrong_figure(rows: list, plots_dir: pathlib.Path) -> None:
    """03: union-find alone's wrong windows, alone and in seam pairs."""
    title = "Union-find alone: wrong windows"
    figure, axis_by_distance = distance_panels(title)
    for distance, axis in axis_by_distance.items():
        values = {}
        for answer_class in UNION_FIND_WRONG_NAMES:
            values[answer_class] = answer_shares(
                rows, "union_find_alone", distance, answer_class
            )
        draw_bars(axis, values, UNION_FIND_WRONG_NAMES)
        axis.set_ylabel("Wrong (% of windows)")
        bar_axis(axis)
    save(figure, plots_dir, "03_union_find_wrong.png")


def wrong_alone_figure(rows: list, plots_dir: pathlib.Path) -> None:
    """04: each configuration's windows wrong alone, a bar each."""
    title = "Windows wrong alone, by configuration"
    figure, axis_by_distance = distance_panels(title)
    for distance, axis in axis_by_distance.items():
        values = {}
        for configuration in CONFIGURATIONS:
            values[configuration] = answer_shares(
                rows, configuration, distance, "wrong_alone"
            )
        draw_bars(axis, values, CONFIGURATIONS)
        axis.set_ylabel("Wrong alone (% of windows)")
        bar_axis(axis)
    save(figure, plots_dir, "04_windows_wrong_alone.png")


def share_figure(
    rows: list, figure_text: tuple, plots_dir: pathlib.Path
) -> None:
    """One panel, a line per distance, of a share of switching's rows."""
    file_name, columns, title, y_label = figure_text
    figure, axis = single_panel(title)
    for distance in DISTANCES:
        points = distance_points(rows, distance, columns)
        draw_shares(axis, points, DISTANCE_LINES[distance], TOTAL_RULE)
    axis.set_ylabel(y_label)
    rate_axis(axis)
    save(figure, plots_dir, file_name)


def wrong_windows_figure(rows: list, plots_dir: pathlib.Path) -> None:
    """07: union-find's wrong windows by what switching did with them.

    An escalated window's seam partner is unseen, so its wrong windows
    are one bar.
    """
    title = "Union-find's wrong windows, by switching's decision"
    figure, axis_by_distance = distance_panels(title)
    for distance, axis in axis_by_distance.items():
        values = {}
        for key, (_name, columns, _color) in WRONG_BARS.items():
            values[key] = switching_shares(rows, distance, columns)
        draw_bars(axis, values, WRONG_BAR_NAMES)
        axis.set_ylabel("Windows (% of windows decided)")
        bar_axis(axis)
    save(figure, plots_dir, "07_wrong_by_decision.png")


def relay_outcomes_figure(rows: list, plots_dir: pathlib.Path) -> None:
    """17: Relay-BP-5 against union-find alone on the escalated windows.

    A window Relay-BP-5 did not converge on keeps its committed answer, so
    it is also counted in fixed, broke or both wrong.
    """
    title = "Escalated windows: Relay-BP-5 against union-find alone"
    figure, axis_by_distance = distance_panels(title)
    for distance, axis in axis_by_distance.items():
        values = {}
        for key, (_name, columns, _color) in RELAY_BARS.items():
            values[key] = switching_shares(
                rows, distance, columns, tables.STRONG_OUTCOMES
            )
        draw_bars(axis, values, RELAY_BAR_NAMES)
        axis.set_ylabel("Windows (% of escalated windows)")
        bar_axis(axis)
    save(figure, plots_dir, "17_relay_against_union_find.png")


def gap_figures(rows: list, plots_dir: pathlib.Path) -> None:
    """08: union-find's wrong share at each cluster gap, a figure per rate.

    Wrong means union-find's answer differs from the label of the
    window's own rounds, which does not depend on the escalation
    decision, so escalated and kept windows count alike.
    """
    for rate in PHYSICAL_ERROR_RATES:
        title = f"Union-find wrong, by cluster gap (p = {rate})"
        figure, axis = single_panel(title)
        for distance in DISTANCES:
            points = gap_points(rows, distance, rate)
            draw_shares(axis, points, DISTANCE_LINES[distance], TOTAL_RULE)
        axis.axvline(
            THRESHOLD_DECIBELS,
            color="black",
            linestyle="--",
            label="Threshold (20 dB)",
        )
        axis.set_ylim(bottom=0)
        axis.set_xlabel(GAP_X_LABEL)
        axis.set_ylabel("Union-find's answer wrong\n(% of windows at this gap)")
        finish_axis(axis)
        save(figure, plots_dir, f"08_gap_p{rate}.png")


def event_figures(rows: list, plots_dir: pathlib.Path) -> None:
    """09: detection events of kept right and wrong windows, per rate."""
    for rate in PHYSICAL_ERROR_RATES:
        title = (
            "Detection events in the windows switching kept"
            f" (physical error rate {rate})"
        )
        figure, axis_by_distance = distance_panels(title)
        for distance, axis in axis_by_distance.items():
            counts = class_counts(rows, distance, rate, events_of)
            draw_event_panel(axis, counts)
        save(figure, plots_dir, f"09_detection_events_p{rate}.png")


def draw_event_panel(axis: pyplot.Axes, counts: dict) -> None:
    """One setting's detection-event histograms.

    A setting with no kept windows keeps its axis labels only.
    """
    axis.set_xlabel("Detection events in the window")
    if counts:
        edges = event_edges(counts)
        binned = rebinned(counts, edges)
        draw_distributions(axis, binned, EVENT_GROUPS, edges)
    finish_distribution(axis)


def wrong_shots_figure(rows: list, plots_dir: pathlib.Path) -> None:
    """13: failed shots among the shots with a wrong window."""
    title = "Failed shots among shots with a wrong window"
    figure, axis_by_distance = distance_panels(title)
    columns = ("failed_shots", "shots_with_a_wrong_window")
    for distance, axis in axis_by_distance.items():
        for configuration, line in CONFIGURATIONS.items():
            points = column_points(rows, configuration, distance, columns)
            draw_shares(axis, points, line, TOTAL_RULE)
        axis.set_ylabel("Failed (% of shots\nwith a wrong window)")
        rate_axis(axis)
    save(figure, plots_dir, "13_failed_among_wrong_shots.png")


def decisions_figure(rows: list, plots_dir: pathlib.Path) -> None:
    """14: each setting's union-find windows as one 100% bar.

    The bar's parts are kept or escalated, union-find right or wrong;
    the text beside it gives the escalated share and, of those, the
    share union-find already had right.
    """
    figure, axis = pyplot.subplots(
        figsize=DECISION_SIZE_INCHES, layout="constrained"
    )
    labels = []
    positions = []
    position = 0.0
    for distance in DISTANCES:
        for rate in PHYSICAL_ERROR_RATES:
            row = _setting_row_of(rows, distance, rate)
            draw_decision_bar(axis, row, position)
            labels.append(f"d = {distance}, p = {rate}")
            positions.append(position)
            position += 1
        position += DECISION_GROUP_GAP
    axis.set_yticks(positions, labels=labels, fontsize=LEGEND_FONT_SIZE)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("Windows union-find decided in switching (%)")
    axis.set_title("What switching did with each window union-find decided")
    axis.legend(
        fontsize=LEGEND_FONT_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
    )
    axis.grid(axis="x", alpha=GRID_ALPHA)
    path = plots_dir / "14_decisions.png"
    figure.savefig(path, dpi=DOTS_PER_INCH)
    pyplot.close(figure)


def draw_decision_bar(axis: pyplot.Axes, row: dict, position: float) -> None:
    """One setting's bar of DECISION_PARTS, and its two shares as text."""
    decided = column_sum(row, DECIDED)
    left = 0.0
    for name, columns, color in DECISION_PARTS:
        width = 100 * column_sum(row, columns) / decided
        # the first bar names the parts once in the legend
        label = name if position == 0 else None
        axis.barh(
            position,
            width,
            left=left,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=label,
        )
        left += width
    escalated = column_sum(row, ESCALATED)
    escalated_share = 100 * escalated / decided
    false_positive_share = 100 * int(row["escalated_right"]) / escalated
    text = (
        f"escalated {escalated_share:.1f}%,"
        f" false positive {false_positive_share:.1f}%"
    )
    axis.text(101, position, text, va="center", fontsize=LEGEND_FONT_SIZE)


def reaction_figures(rows: list, plots_dir: pathlib.Path) -> None:
    """15: each configuration's reaction times, a figure per rate."""
    for rate in PHYSICAL_ERROR_RATES:
        title = f"Reaction time per window (p = {rate})"
        figure, axis_by_distance = distance_panels(title, share_x=True)
        for distance, axis in axis_by_distance.items():
            for configuration, line in CONFIGURATIONS.items():
                setting = (configuration, distance, rate)
                draw_reaction_histogram(axis, rows, setting, line)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel("Reaction time (µs)")
            axis.set_ylabel("Windows (% per bin)")
            axis.label_outer()
            is_first_panel = distance == DISTANCES[0]
            finish_axis(axis, has_legend=is_first_panel)
        save(figure, plots_dir, f"15_reaction_time_p{rate}.png")


def draw_reaction_histogram(
    axis: pyplot.Axes, rows: list, setting: tuple, line: tuple
) -> None:
    """One setting's reaction-time bins as a histogram line, in %."""
    configuration, distance, rate = setting
    label, color, _marker = line
    windows_by_bin = {}
    for row in rows:
        if row["configuration"] != configuration:
            continue
        if not _is_setting(row, distance, rate):
            continue
        reaction_bin = reaction_bin_of(row)
        windows_by_bin[reaction_bin] = int(row["windows"])
    if not windows_by_bin:
        return
    # the empty bins between the first and last are drawn at zero
    bins = range(min(windows_by_bin), max(windows_by_bin) + 1)
    windows = numpy.array([windows_by_bin.get(index, 0) for index in bins])
    edges = [tables.reaction_edge(index) for index in bins]
    last_edge = tables.reaction_edge(bins[-1] + 1)
    edges.append(last_edge)
    shares = 100 * windows / windows.sum()
    axis.stairs(shares, edges, color=color, label=label, linewidth=1.5)


def reaction_bin_of(row: dict) -> int:
    """A reaction_time.csv row's bin, from its low edge."""
    low_decades = math.log10(float(row["low_us"]))
    decades = low_decades - tables.REACTION_LOWEST_DECADE
    # the edge is a power of ten written as a float, so round, not floor
    return round(decades * tables.REACTION_BINS_PER_DECADE)


def decode_figure(rows: list, plots_dir: pathlib.Path) -> None:
    """16: each decoder's time on a window against distance, per rate.

    The dashed line is the window period: a decoder above it falls
    further behind with every window.
    """
    figure, axes = pyplot.subplots(
        2,
        2,
        figsize=PANELS_SIZE_INCHES,
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    figure.suptitle("Decode time per window: median, bar to 99th percentile")
    for rate, axis in zip(PHYSICAL_ERROR_RATES, axes.flat, strict=True):
        axis.set_title(f"p = {rate}")
        for configuration, line in DECODERS.items():
            draw_decode_times(axis, rows, (configuration, rate), line)
        periods = window_periods(rows, rate)
        axis.plot(
            DISTANCES,
            periods,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Time to measure one window's new rounds",
        )
        decode_axis(axis)
    first_axis = axes.flat[0]
    handles, labels = first_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="Decoder",
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
        loc="outside lower center",
        ncols=4,
    )
    save(figure, plots_dir, "16_decode_time.png")


def draw_decode_times(
    axis: pyplot.Axes, rows: list, key: tuple, line: tuple
) -> None:
    """One decoder's median per distance, a bar up to its 99th percentile."""
    configuration, rate = key
    label, color, marker = line
    medians = []
    bars_above = []
    for distance in DISTANCES:
        row = _decode_row_of(rows, configuration, distance, rate)
        median = float(row["median_us"])
        p99 = float(row["p99_us"])
        medians.append(median)
        bars_above.append(p99 - median)
    bars_below = [0.0] * len(medians)
    axis.errorbar(
        DISTANCES,
        medians,
        yerr=[bars_below, bars_above],
        color=color,
        marker=marker,
        capsize=3,
        label=label,
    )


def window_periods(rows: list, rate: float) -> list:
    """Each distance's window period, the same for every configuration."""
    periods = []
    for distance in DISTANCES:
        row = _decode_row_of(rows, "union_find_alone", distance, rate)
        periods.append(float(row["window_period_us"]))
    return periods


def decode_axis(axis: pyplot.Axes) -> None:
    """Log distance on x with its four values as ticks, log time on y."""
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(*DECODE_DISTANCE_LIMITS)
    distance_labels = [str(distance) for distance in DISTANCES]
    axis.set_xticks(DISTANCES, labels=distance_labels)
    axis.set_xticks([], minor=True)
    axis.set_xlabel("Code distance")
    axis.set_ylabel("Decode time per window (µs)")
    axis.label_outer()
    finish_axis(axis, has_legend=False)


def gap_points(rows: list, distance: int, rate: float) -> dict:
    """(wrong, windows) by gap bin centre, for one setting."""
    wrong_by_centre = collections.Counter()
    windows_by_centre = collections.Counter()
    for row in rows:
        if not _is_setting(row, distance, rate):
            continue
        centre = gap_bin_of(row) + GAP_BIN_DECIBELS / 2
        windows = int(row["windows"])
        windows_by_centre[centre] += windows
        if row["union_find_answer"] == "wrong":
            wrong_by_centre[centre] += windows
    points = {}
    for centre, windows in windows_by_centre.items():
        points[centre] = (wrong_by_centre[centre], windows)
    return points


def distance_points(rows: list, distance: int, columns: tuple) -> dict:
    """(summed count columns, summed total columns) by rate, for one d."""
    count_columns, total_columns = columns
    points = {}
    for row in rows:
        if int(row["distance"]) != distance:
            continue
        count = column_sum(row, count_columns)
        total = column_sum(row, total_columns)
        points[float(row["physical_error_rate"])] = (count, total)
    return points


def draw_shares(
    axis: pyplot.Axes,
    points: dict,
    line: tuple,
    rule: tuple,
    transform: Optional[Callable] = None,
) -> None:
    """Count over total in % at each physical error rate, Wilson bars.

    line is the (label, color) pair, rule FAILURE_RULE or TOTAL_RULE. A
    point below the rule is a gap in its line.
    """
    label, color, marker = line
    rates = sorted(points)
    shares = []
    below = []
    above = []
    for rate in rates:
        share, low, high = shown_share(points[rate], rule, transform)
        bar_below = share - low
        bar_above = high - share
        shares.append(share)
        below.append(bar_below)
        above.append(bar_above)
    is_hidden = numpy.isnan(shares)
    if is_hidden.all():
        return
    axis.errorbar(
        rates,
        shares,
        yerr=[below, above],
        fmt="-",
        marker=marker,
        color=color,
        label=label,
        capsize=3,
    )


def shown_share(
    point: tuple, rule: tuple, transform: Optional[Callable] = None
) -> tuple:
    """A point's share and Wilson bounds in %, not a number when hidden."""
    count, total = point
    held_number, minimum = rule
    held = total
    if held_number == "count":
        held = count
    if held < minimum or total == 0:
        return numpy.nan, numpy.nan, numpy.nan
    share = 100 * count / total
    low, high = tables.wilson_interval(count, total)
    low_share = 100 * low
    high_share = 100 * high
    if transform is None:
        return share, low_share, high_share
    return transform(share), transform(low_share), transform(high_share)


def per_round(shot_share: float) -> float:
    """A share of failed shots, in %, as a rate per round."""
    shot_rate = shot_share / 100
    return failure_statistics.per_round_rate(shot_rate, ROUNDS_PER_SHOT)


def draw_bars(axis: pyplot.Axes, values: dict, names: dict) -> None:
    """Grouped bars: a group per rate, a bar per key of values.

    values maps a key to its heights per rate, or to its (count, total)
    points per rate, drawn as shares in % with Wilson bars.
    """
    positions = numpy.arange(len(PHYSICAL_ERROR_RATES))
    bar_width = BAR_GROUP_WIDTH / len(values)
    first_offset = (len(values) - 1) / 2
    entries = values.items()
    for index, (key, bar_values) in enumerate(entries):
        name, color = names[key][:2]
        offset = (index - first_offset) * bar_width
        bar_positions = positions + offset
        heights, error_bars = bar_heights(bar_values)
        axis.bar(
            bar_positions,
            heights,
            bar_width,
            yerr=error_bars,
            capsize=2,
            color=color,
            label=name,
        )
    axis.set_xticks(positions, labels=PHYSICAL_ERROR_RATES)


def bar_heights(bar_values: list) -> tuple:
    """Heights and Wilson bars of shares, or plain heights and no bars."""
    is_shares = any(isinstance(value, tuple) for value in bar_values)
    if not is_shares:
        return bar_values, None
    heights = []
    below = []
    above = []
    for point in bar_values:
        share, low, high = shown_share(point, TOTAL_RULE)
        bar_below = share - low
        bar_above = high - share
        heights.append(share)
        below.append(bar_below)
        above.append(bar_above)
    return heights, [below, above]


def draw_distributions(
    axis: pyplot.Axes, counts: dict, groups: dict, edges: numpy.ndarray
) -> None:
    """Each group's share of its windows per bin, as a histogram line.

    edges are the bins' low edges, the last bin's high edge one bin on.
    """
    bin_width = edges[1] - edges[0]
    last_edge = edges[-1] + bin_width
    stair_edges = numpy.append(edges, last_edge)
    for name, (classes, color) in groups.items():
        windows = group_windows(counts, classes, edges)
        total = windows.sum()
        if total < SHOWN_TOTAL:
            continue
        shares = 100 * windows / total
        label = f"{name}, {total:,} windows"
        axis.stairs(
            shares, stair_edges, color=color, label=label, linewidth=1.5
        )
    axis.set_ylabel("Fraction of windows (%)")


def group_windows(
    counts: dict, classes: tuple, edges: numpy.ndarray
) -> numpy.ndarray:
    """The windows of the classes in each bin, zero where none."""
    windows = numpy.zeros(len(edges), dtype=int)
    for answer_class in classes:
        class_bins = counts.get(answer_class, {})
        for position, edge in enumerate(edges):
            windows[position] += class_bins.get(edge, 0)
    return windows


def event_edges(counts: dict) -> numpy.ndarray:
    """About EVENT_BINS bins of whole detection-event counts, from 0."""
    largest = 0
    for class_bins in counts.values():
        largest = max(largest, *class_bins)
    rounded_up_width = -(-largest // EVENT_BINS)
    bin_width = max(1, rounded_up_width)
    # one bin past the largest, so even an all-zero count has two edges
    top = largest + 2 * bin_width
    return numpy.arange(0, top, bin_width)


def rebinned(counts: dict, edges: numpy.ndarray) -> dict:
    """Counts by single value moved into the bins whose low edges these are."""
    bin_width = edges[1] - edges[0]
    moved = collections.defaultdict(collections.Counter)
    for answer_class, class_bins in counts.items():
        for value, windows in class_bins.items():
            low_edge = value // bin_width * bin_width
            moved[answer_class][low_edge] += windows
    return moved


def class_counts(
    rows: list, distance: int, rate: float, bin_of: Callable
) -> dict:
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
    gap_low_edge = bins * GAP_BIN_DECIBELS
    gap_bin = int(gap_low_edge)
    return min(gap_bin, GAP_TOP_DECIBELS)


def events_of(row: dict) -> int:
    """A row's detection-event count."""
    return int(row["detection_events"])


def switching_shares(
    rows: list, distance: int, columns: tuple, total_columns: tuple = DECIDED
) -> list:
    """The columns' (count, total) points, per rate.

    The total is union-find's windows unless total_columns names another.
    """
    points = []
    for rate in PHYSICAL_ERROR_RATES:
        row = _setting_row_of(rows, distance, rate)
        point = (0, 0)
        if row is not None:
            count = column_sum(row, columns)
            total = column_sum(row, total_columns)
            point = (count, total)
        points.append(point)
    return points


def answer_shares(
    rows: list, configuration: str, distance: int, answer_class: str
) -> list:
    """One answer class's (count, windows) points, per rate."""
    configuration_rows = [
        row for row in rows if row["configuration"] == configuration
    ]
    points = []
    for rate in PHYSICAL_ERROR_RATES:
        row = _setting_row_of(configuration_rows, distance, rate)
        point = (0, 0)
        if row is not None:
            windows = column_sum(row, tables.ANSWER_CLASSES)
            point = (int(row[answer_class]), windows)
        points.append(point)
    return points


def column_points(
    rows: list, configuration: str, distance: int, columns: tuple
) -> dict:
    """One configuration's (count, total) columns by physical error rate."""
    count_column, total_column = columns
    points = {}
    for row in rows:
        if row["configuration"] != configuration:
            continue
        if int(row["distance"]) == distance:
            rate = float(row["physical_error_rate"])
            points[rate] = (int(row[count_column]), int(row[total_column]))
    return points


def per_shot(points: dict) -> list:
    """Each rate's count per shot, not a number where it has no shots."""
    values = []
    for rate in PHYSICAL_ERROR_RATES:
        count, shots = points.get(rate, (0, 0))
        if shots == 0:
            values.append(numpy.nan)
            continue
        windows_per_shot = count / shots
        values.append(windows_per_shot)
    return values


def column_sum(row: dict, columns: tuple) -> int:
    """The sum of a row's columns, as whole numbers."""
    total = 0
    for column in columns:
        total += int(row[column])
    return total


def single_panel(title: str) -> tuple:
    """A one-panel figure with its title."""
    figure, axis = pyplot.subplots(
        figsize=SINGLE_SIZE_INCHES, layout="constrained"
    )
    axis.set_title(title, fontsize=TITLE_FONT_SIZE)
    return figure, axis


def distance_panels(title: str, share_x: bool = False) -> tuple:
    """A 2 by 2 figure, a panel per distance, one y scale, the title on top."""
    figure, axes = pyplot.subplots(
        2,
        2,
        figsize=PANELS_SIZE_INCHES,
        layout="constrained",
        sharex=share_x,
        sharey=True,
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
    axis.needs_headroom = True
    finish_axis(axis)


def finish_distribution(axis: pyplot.Axes) -> None:
    """A share axis from zero, with headroom above the lines for the legend."""
    axis.needs_headroom = True
    finish_axis(axis)


def finish_axis(axis: pyplot.Axes, has_legend: bool = True) -> None:
    """A light grid, and a legend when lines have labels."""
    axis.grid(which="major", alpha=GRID_ALPHA)
    if axis.get_yscale() == "log":
        axis.grid(which="minor", axis="y", alpha=GRID_ALPHA / 3)
    handles, _labels = axis.get_legend_handles_labels()
    if has_legend and len(handles) > 1:
        axis.legend(fontsize=LEGEND_FONT_SIZE)


def save(figure: pyplot.Figure, plots_dir: pathlib.Path, name: str) -> None:
    """The figure written into plots_dir under name, and closed."""
    tops = [
        axis.dataLim.y1
        for axis in figure.axes
        if getattr(axis, "needs_headroom", False)
    ]
    for axis in figure.axes:
        if tops:
            axis.set_ylim(0, max(tops) * BAR_HEADROOM)
        if axis.get_subplotspec().colspan.start == 1:
            axis.set_ylabel("")
    path = plots_dir / name
    figure.savefig(path, dpi=DOTS_PER_INCH)
    pyplot.close(figure)


def rows_of(path: pathlib.Path) -> list:
    """A table's rows as dicts of text."""
    with path.open(newline="") as table_file:
        reader = csv.DictReader(table_file)
        return list(reader)


def _decode_row_of(
    rows: list, configuration: str, distance: int, rate: float
) -> dict:
    """decode_time.csv's row of one configuration and setting."""
    for row in rows:
        is_configuration = row["configuration"] == configuration
        if is_configuration and _is_setting(row, distance, rate):
            return row
    setting = f"{configuration} d={distance} p={rate}"
    raise ValueError(f"decode_time.csv has no row for {setting}")


def _setting_row_of(rows: list, distance: int, rate: float):
    """The row of one setting, or None."""
    for row in rows:
        if _is_setting(row, distance, rate):
            return row
    return None


def _is_setting(row: dict, distance: int, rate: float) -> bool:
    is_distance = int(row["distance"]) == distance
    return is_distance and float(row["physical_error_rate"]) == rate


if __name__ == "__main__":
    results_folder = pathlib.Path(sys.argv[1])
    main(results_folder)
