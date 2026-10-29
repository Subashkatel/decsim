r"""The pilot's figures, one message each, in the papers' style.

    python plots.py RESULTS DATE

RESULTS is the five results' results folder and DATE the runs' folder
name, as summary.py reads them. The figures go to RESULTS/plots_DATE/,
each as PDF and PNG; figure_style.py holds their look. Each plot
function's docstring is the one thing its figure says.

- decoder_accuracy: R1 shots, each decoder alone's error per round;
- decoder_reaction: R1 shots, each window's reaction against the
  deadline (SWIPER 2412.05115 Fig. 3 draws latency as violins);
- waiting_line: R1 streams with series rows, each decoder's waiting
  windows over time against the queue's theory (Toshio et al.
  2510.25222 Figs. 10 and 13 draw backlog over time);
- share_steps: the calibration's gap histogram as the share each
  threshold sends;
- switching_accuracy, switching_lateness, speculation_reaction,
  second_core, repairs: the switching pilot, against both decoders
  alone on the same shots.

Reaction is buffer0_ready_to_frame, as in summary.py.
"""

import csv
import json
import math
import pathlib
import sys

import config
import figure_style
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import numpy
import scipy.stats

import decsim.experiments.paired_shots as paired_shots

R1_SHOTS = "r1_shots"
R1_STREAMS = "r1_streams"
CALIBRATION = "gap_calibration"
SWITCHING_PILOT = "switching_pilot"
DECODINGS = ("weak_alone", "strong_alone")
RULES = ("sliding", "speculation", "double")
POOLS = {"weak_alone": "weak", "strong_alone": "strong"}
WEAK_ALONE = {"decoding": "weak_alone"}
STRONG_ALONE = {"decoding": "strong_alone"}
# the pilot's point (experiments.md); the R1 streams' series rows are
# in the streams folder's series/ (B7's series_every_rounds)
PILOT_DISTANCE = 9
PILOT_PROBABILITY = 0.003
STREAM_SEED = 1
REACTION = "buffer0_ready_to_frame"
REACTION_MEAN = f"{REACTION}_mean_us"
REACTION_TAIL = f"{REACTION}_p99_us"
DEADLINE_MICROSECONDS = config.ROUND_PERIOD_MICROSECONDS * PILOT_DISTANCE
ACCURACY_TICKS = (1e-4, 2e-4, 5e-4, 1e-3)
ACCURACY_LIMITS = (1e-4, 1.5e-3)
SHARE_LIMIT_PERCENT = 20
SHARE_TICK_PERCENT = 5
LATENESS_LIMIT_PERCENT = 70
STEPS_LIMIT = 45
STEPS_TICK = 10
# where the note on the step across the highest target sits, in data
STEP_NOTE_AT = (1, 38)
TAIL_LIMIT_MICROSECONDS = 290
REPAIRS_LIMIT = 520
VIOLIN_HALF_WIDTH = 0.38
VIOLIN_GRID_POINTS = 300
MEDIAN_HALF_HEIGHT = 0.3
# how many series rows apart the waiting line puts a marker
MARKER_EVERY_ROWS = 10
BAR_WIDTH = 0.36
BAR_EDGE = {"edgecolor": "white", "linewidth": 0.6}
SPECULATION_HATCH = "////"
TWO_CORE_HATCH = "////"
TWO_CORE_ALPHA = 0.55
SWATCH_GRAY = "#888888"
REPAIRED_COLOR = "#4d4d4d"
BROKEN_COLOR = "#c7c7c7"
THEORY_COLOR = "#b0b0b0"
INK = "#333333"
ANNOTATION = {"textcoords": "offset points", "fontsize": 6.5}


def main(results: pathlib.Path, date: str) -> None:
    """Draw every figure from the pilot's finished runs."""
    figure_style.use()
    folder = results / f"plots_{date}"
    r1_folder = results / R1_SHOTS / date
    r1_sweep_path = r1_folder / "sweep.csv"
    r1_sweep = _read(r1_sweep_path)
    plot_decoder_accuracy(r1_sweep, folder)
    r1_samples_path = r1_folder / "window_samples.csv"
    r1_samples = _read(r1_samples_path)
    plot_decoder_reaction(r1_samples, folder)
    series_folder = results / R1_STREAMS / date / "series"
    plot_waiting_line(series_folder, folder)
    calibration_folder = results / CALIBRATION / date
    plot_share_steps(calibration_folder, folder)
    switching_folder = results / SWITCHING_PILOT / date
    sweep_path = switching_folder / "sweep.csv"
    sweep = _read(sweep_path)
    plot_switching_accuracy(sweep, r1_sweep, folder)
    plot_switching_lateness(sweep, folder)
    plot_speculation_reaction(sweep, folder)
    plot_second_core(sweep, folder)
    shots_path = switching_folder / "shots.csv"
    switching_shots = _read(shots_path)
    r1_shots_path = r1_folder / "shots.csv"
    r1_shots = _read(r1_shots_path)
    plot_repairs(sweep, switching_shots, r1_shots, folder)
    print(f"figures in {folder}")


def plot_decoder_accuracy(r1_sweep: list, folder: pathlib.Path) -> None:
    """Correlated matching alone errs 3.6 times less often than union-find."""
    figure, axes = figure_style.figure(height=1.2)
    for position, decoding in enumerate(DECODINGS):
        row = _only(r1_sweep, decoding)
        _draw_rate_point(axes, row, position, decoding)
    _name_decoder_rows(axes)
    axes.set_xscale("log")
    axes.set_xlim(*ACCURACY_LIMITS)
    labels = [_tick_text(tick) for tick in ACCURACY_TICKS]
    axes.set_xticks(ACCURACY_TICKS, labels)
    no_labels = ticker.NullFormatter()
    axes.xaxis.set_minor_formatter(no_labels)
    axes.set_xlabel("Logical error per round")
    figure_style.save(figure, folder, "decoder_accuracy")


def plot_decoder_reaction(r1_samples: list, folder: pathlib.Path) -> None:
    """Union-find is almost never late; correlated matching is always late.

    Union-find misses the deadline on 6 of 100,000 windows, correlated
    matching on every window.
    """
    figure, axes = figure_style.figure(height=1.4)
    for position, decoding in enumerate(DECODINGS):
        values, counts = _samples(r1_samples, REACTION, decoding)
        style = figure_style.DECODER_STYLES[decoding]
        _violin(axes, values, counts, position, style["color"])
    figure_style.deadline(axes, DEADLINE_MICROSECONDS, vertical=True)
    _name_decoder_rows(axes)
    axes.set_xscale("log")
    axes.set_xlabel("Reaction time per window (µs)")
    figure_style.save(figure, folder, "decoder_reaction")


def plot_waiting_line(
    series_folder: pathlib.Path, folder: pathlib.Path
) -> None:
    """Correlated matching's waiting line grows; union-find's stays empty.

    The line grows without limit, at the queue's theoretical rate.
    """
    figure, axes = figure_style.figure()
    end_us = 0.0
    for decoding in DECODINGS:
        rows = _series_rows(series_folder, decoding)
        _draw_waiting(axes, rows, decoding)
        last = rows[-1]
        end_us = max(end_us, last["time_us"])
    _draw_queue_theory(axes, end_us)
    end_ms = end_us / 1000
    axes.set_xlim(0, end_ms)
    axes.set_ylim(bottom=-20)
    axes.set_xlabel("Time (ms)")
    axes.set_ylabel("Windows waiting")
    axes.legend(loc="upper left")
    figure_style.save(figure, folder, "waiting_line")


def plot_share_steps(
    calibration_folder: pathlib.Path, folder: pathlib.Path
) -> None:
    """The gap takes few values, so the share sent rises in steps.

    No threshold sends 20%.
    """
    lows, shares = _share_curve(calibration_folder)
    figure, axes = figure_style.figure()
    # a threshold inside a bin sends the windows of that bin and below
    axes.step(lows, shares, where="post", color=INK, linewidth=1.0)
    calibration_path = calibration_folder / "calibration.csv"
    calibration = _read(calibration_path)
    for row in calibration:
        _draw_target(axes, row)
        _draw_calibrated(axes, row)
    targets = [_percent(row, "target_share") for row in calibration]
    highest_target = max(targets)
    _name_step_across(axes, lows, shares, highest_target)
    axes.set_xlim(0, STEPS_LIMIT)
    axes.set_ylim(0, STEPS_LIMIT)
    tens = ticker.MultipleLocator(STEPS_TICK)
    axes.xaxis.set_major_locator(tens)
    axes.set_xlabel("Threshold on the cluster gap (dB)")
    axes.set_ylabel("Windows sent (%)")
    figure_style.save(figure, folder, "share_steps")


def plot_switching_accuracy(
    sweep: list, r1_sweep: list, folder: pathlib.Path
) -> None:
    """Sending 17% of windows recovers about 90% of the strong accuracy.

    Of the gap between the two decoders alone, sliding closes 87% at a
    16.5% share and the double window 90% at an 18.5% share.
    """
    figure, axes = figure_style.figure()
    for decoding in DECODINGS:
        row = _only(r1_sweep, decoding)
        _draw_reference(axes, row, decoding)
    for rule in ("sliding", "double"):
        rows = _design_rows(sweep, rule, "1")
        _draw_rule_rates(axes, rows, rule)
    axes.set_yscale("log")
    _share_axis(axes)
    axes.set_ylabel("Logical error per round")
    axes.legend(loc="center right", bbox_to_anchor=(1, 0.62))
    figure_style.save(figure, folder, "switching_accuracy")


def plot_switching_lateness(sweep: list, folder: pathlib.Path) -> None:
    """The double window is late about three times less often than sliding.

    Speculation is late about as often as sliding.
    """
    figure, axes = figure_style.figure()
    for rule in RULES:
        rows = _design_rows(sweep, rule, "1")
        shares = [_percent(row, "escalated_fraction") for row in rows]
        late = [_percent(row, "deadline_missed_fraction") for row in rows]
        color = figure_style.RULE_COLORS[rule]
        marker = figure_style.RULE_MARKERS[rule]
        name = figure_style.RULE_NAMES[rule]
        axes.plot(shares, late, color=color, marker=marker, label=name)
    _share_axis(axes)
    axes.set_ylim(0, LATENESS_LIMIT_PERCENT)
    axes.set_ylabel("Windows past the deadline (%)")
    axes.legend(loc="upper left")
    figure_style.save(figure, folder, "switching_lateness")


def plot_speculation_reaction(sweep: list, folder: pathlib.Path) -> None:
    """Speculation reacts about 15% sooner than sliding, with equal answers."""
    figure, axes = figure_style.figure()
    sliding = _design_rows(sweep, "sliding", "1")
    speculation = _design_rows(sweep, "speculation", "1")
    pairs = zip(sliding, speculation, strict=True)
    for index, pair in enumerate(pairs):
        _draw_reaction_pair(axes, index, pair)
    labels = [_share_text(row) for row in sliding]
    _name_groups(axes, labels)
    axes.set_xlabel("Windows sent to correlated matching")
    axes.set_ylabel("Mean reaction time (µs)")
    sliding_color = figure_style.RULE_COLORS["sliding"]
    speculation_color = figure_style.RULE_COLORS["speculation"]
    handles = [
        _swatch(sliding_color, None, 1.0),
        _swatch(speculation_color, SPECULATION_HATCH, 1.0),
    ]
    names = [
        figure_style.RULE_NAMES["sliding"],
        figure_style.RULE_NAMES["speculation"],
    ]
    axes.legend(handles, names, loc="upper left")
    figure_style.save(figure, folder, "speculation_reaction")


def plot_second_core(sweep: list, folder: pathlib.Path) -> None:
    """A second correlated matching core shortens only the double window's tail.

    Sliding and speculation decode one window at a time.
    """
    figure, axes = figure_style.figure()
    for index, rule in enumerate(RULES):
        one_core = _highest_share(sweep, rule, "1")
        two_cores = _highest_share(sweep, rule, "2")
        _draw_core_pair(axes, index, rule, one_core, two_cores)
    names = [figure_style.RULE_NAMES[rule] for rule in RULES]
    _name_groups(axes, names)
    axes.set_ylim(0, TAIL_LIMIT_MICROSECONDS)
    axes.set_ylabel("99th percentile reaction (µs)")
    handles = [
        _swatch(SWATCH_GRAY, None, 1.0),
        _swatch(SWATCH_GRAY, TWO_CORE_HATCH, TWO_CORE_ALPHA),
    ]
    axes.legend(handles, ["One core", "Two cores"], loc="upper left", ncols=2)
    figure_style.save(figure, folder, "second_core")


def plot_repairs(
    sweep: list, switching_shots: list, r1_shots: list, folder: pathlib.Path
) -> None:
    """Switching repairs about seven union-find failures per failure it adds."""
    weak = paired_shots.failures_by_seed(r1_shots, WEAK_ALONE)
    strong = paired_shots.failures_by_seed(r1_shots, STRONG_ALONE)
    figure, axes = figure_style.figure()
    rows = _design_rows(sweep, "sliding", "1")
    for index, row in enumerate(rows):
        selection = {
            "scheme": "sliding",
            "strong_unit_count": "1",
            "share": row["share"],
        }
        design = paired_shots.failures_by_seed(switching_shots, selection)
        pair = paired_shots.paired_repairs(weak, strong, design)
        _draw_repair_pair(axes, index, pair)
    labels = [_share_text(row) for row in rows]
    _name_groups(axes, labels)
    axes.set_ylim(0, REPAIRS_LIMIT)
    axes.set_xlabel("Windows sent to correlated matching (sliding window)")
    axes.set_ylabel("Shots")
    handles = [
        _swatch(REPAIRED_COLOR, None, 1.0),
        _swatch(BROKEN_COLOR, None, 1.0),
    ]
    axes.legend(handles, ["Repaired", "Broken"], loc="upper left")
    figure_style.save(figure, folder, "repairs")


def _draw_rate_point(axes, row: dict, position: int, decoding: str) -> None:
    """One decoder's error per round, its 95% interval and its value."""
    rate, below, above = _rate_and_errors(row)
    style = figure_style.DECODER_STYLES[decoding]
    errors = [[below], [above]]
    axes.errorbar(
        rate,
        position,
        xerr=errors,
        color=style["color"],
        marker=style["marker"],
        linestyle="none",
        markersize=5,
    )
    text = _scientific_text(rate)
    axes.annotate(
        text, (rate, position), xytext=(0, 6), ha="center", **ANNOTATION
    )


def _name_decoder_rows(axes) -> None:
    """The decoders' names on the y axis, one row each."""
    names = [figure_style.DECODER_NAMES[decoding] for decoding in DECODINGS]
    count = len(DECODINGS)
    positions = range(count)
    axes.set_yticks(positions, names)
    top = count - 0.4
    axes.set_ylim(-0.6, top)
    axes.grid(axis="y", visible=False)


def _violin(axes, values, counts, position: int, color: str) -> None:
    """A horizontal violin of the values' logarithm, with a tick at the median.

    A log axis then shows its shape undistorted.
    """
    logs = numpy.log10(values)
    density = scipy.stats.gaussian_kde(logs, weights=counts)
    lowest = logs.min()
    highest = logs.max()
    grid = numpy.linspace(lowest, highest, VIOLIN_GRID_POINTS)
    heights = density(grid)
    tallest = heights.max()
    half = VIOLIN_HALF_WIDTH * heights / tallest
    times = 10**grid
    bottom = position - half
    top = position + half
    axes.fill_between(
        times,
        bottom,
        top,
        color=color,
        alpha=0.35,
        linewidth=0.6,
        edgecolor=color,
    )
    median = _weighted_median(values, counts)
    span = [position - MEDIAN_HALF_HEIGHT, position + MEDIAN_HALF_HEIGHT]
    axes.plot([median, median], span, color=color, linewidth=1.2)


def _weighted_median(values, counts) -> float:
    """The value at the middle of the counted samples."""
    order = numpy.argsort(values)
    sorted_values = values[order]
    sorted_counts = counts[order]
    cumulative = numpy.cumsum(sorted_counts)
    half = cumulative[-1] / 2
    middle = numpy.searchsorted(cumulative, half)
    return sorted_values[middle]


def _series_rows(series_folder: pathlib.Path, decoding: str) -> list:
    """One R1 stream's series rows."""
    name = _stream_name(decoding)
    path = series_folder / name
    text = path.read_text()
    record = json.loads(text)
    return record["measurement"]["series"]


def _draw_waiting(axes, rows: list, decoding: str) -> None:
    """One decoder's waiting windows at each series row."""
    pool = POOLS[decoding]
    times_us = [row["time_us"] for row in rows]
    times_ms = numpy.divide(times_us, 1000)
    waiting = [row["queued_jobs"][pool] for row in rows]
    style = figure_style.DECODER_STYLES[decoding]
    name = figure_style.DECODER_NAMES[decoding]
    axes.plot(
        times_ms,
        waiting,
        color=style["color"],
        marker=style["marker"],
        markevery=MARKER_EVERY_ROWS,
        linewidth=0.9,
        markersize=3.5,
        label=name,
    )


def _draw_queue_theory(axes, end_us: float) -> None:
    """The queue's theory: the line grows by arrival rate less service rate.

    A window arrives every period; correlated matching ends one every
    service time.
    """
    service_us = _strong_service_microseconds()
    arrival_rate = 1 / DEADLINE_MICROSECONDS
    service_rate = 1 / service_us
    slope = arrival_rate - service_rate
    end_ms = end_us / 1000
    end_waiting = slope * end_us
    label = f"Theory: 1/{DEADLINE_MICROSECONDS:g} − 1/{service_us:g} per µs"
    axes.plot(
        [0, end_ms],
        [0, end_waiting],
        color=THEORY_COLOR,
        linewidth=2.6,
        zorder=0,
        label=label,
    )


def _share_curve(calibration_folder: pathlib.Path) -> tuple:
    """Each gap bin's low edge, and the percent of windows in it and below."""
    path = calibration_folder / "confidence_histogram.csv"
    histogram = _read(path)
    counts = {}
    for row in histogram:
        if row["histogram"] != "window":
            continue
        low = float(row["gap_low_decibels"])
        count = int(row["count"])
        before = counts.get(low, 0)
        counts[low] = before + count
    lows = sorted(counts)
    in_order = [counts[low] for low in lows]
    cumulative = numpy.cumsum(in_order)
    total = cumulative[-1]
    shares = cumulative / total * 100
    return lows, shares


def _draw_target(axes, row: dict) -> None:
    """A target share as a dotted line, named at the right."""
    target = _percent(row, "target_share")
    axes.axhline(target, color="#999999", linestyle=":", linewidth=0.8)
    text = f"target {target:g}%"
    axes.annotate(
        text,
        (1, target),
        xycoords=("axes fraction", "data"),
        xytext=(-2, 1.5),
        ha="right",
        va="bottom",
        color="#555555",
        **ANNOTATION,
    )


def _draw_calibrated(axes, row: dict) -> None:
    """A calibrated threshold and the share it sends."""
    threshold = float(row["threshold_decibels"])
    sent = _percent(row, "escalated_share")
    axes.plot(threshold, sent, marker="o", color="#000000", markersize=4)
    text = f"{threshold:g} dB: {sent:.1f}%"
    # right of and below its point the curve has not yet risen
    axes.annotate(
        text,
        (threshold, sent),
        xytext=(4, -3),
        ha="left",
        va="top",
        **ANNOTATION,
    )


def _name_step_across(axes, lows: list, shares, percent: float) -> None:
    """Point at the step that jumps past a share, with its rise."""
    reached = shares >= percent
    first = numpy.argmax(reached)
    index = int(first)
    below = shares[index - 1]
    rise = shares[index] - below
    middle = below + rise / 2
    where = lows[index]
    text = f"{rise:.0f}% of windows share one gap value"
    arrow = {"arrowstyle": "-|>", "linewidth": 0.6, "color": INK}
    axes.annotate(
        text,
        (where, middle),
        xytext=STEP_NOTE_AT,
        textcoords="data",
        ha="left",
        va="center",
        fontsize=6.5,
        arrowprops=arrow,
    )


def _draw_reference(axes, row: dict, decoding: str) -> None:
    """A decoder alone as a line across, with its 95% band and its name."""
    rate, below, above = _rate_and_errors(row)
    low = rate - below
    high = rate + above
    style = figure_style.DECODER_STYLES[decoding]
    axes.axhspan(
        low,
        high,
        color=style["color"],
        alpha=figure_style.BAND_ALPHA,
        linewidth=0,
    )
    axes.axhline(
        rate, color=style["color"], linestyle=style["linestyle"], linewidth=1.0
    )
    name = figure_style.DECODER_NAMES[decoding]
    axes.annotate(
        name,
        (1, rate),
        xycoords=("axes fraction", "data"),
        xytext=(-2, 2),
        ha="right",
        va="bottom",
        **ANNOTATION,
    )


def _draw_rule_rates(axes, rows: list, rule: str) -> None:
    """One window rule's error per round against the share it sent."""
    shares = [_percent(row, "escalated_fraction") for row in rows]
    rates = []
    belows = []
    aboves = []
    for row in rows:
        rate, below, above = _rate_and_errors(row)
        rates.append(rate)
        belows.append(below)
        aboves.append(above)
    name = _accuracy_name(rule)
    axes.errorbar(
        shares,
        rates,
        yerr=[belows, aboves],
        color=figure_style.RULE_COLORS[rule],
        marker=figure_style.RULE_MARKERS[rule],
        label=name,
    )


def _accuracy_name(rule: str) -> str:
    """The rule's name; sliding speaks for speculation.

    Their answers are the same on every shot.
    """
    name = figure_style.RULE_NAMES[rule]
    if rule == "sliding":
        return f"{name} (= speculation)"
    return name


def _share_axis(axes) -> None:
    """The x axis of the share sent to correlated matching."""
    axes.set_xlim(0, SHARE_LIMIT_PERCENT)
    fives = ticker.MultipleLocator(SHARE_TICK_PERCENT)
    axes.xaxis.set_major_locator(fives)
    axes.set_xlabel("Windows sent to correlated matching (%)")


def _draw_reaction_pair(axes, index: int, pair: tuple) -> None:
    """Sliding's and speculation's mean reaction side by side, and the cut."""
    sliding_row, speculation_row = pair
    sliding = float(sliding_row[REACTION_MEAN])
    speculation = float(speculation_row[REACTION_MEAN])
    left, right = _pair_positions(index)
    sliding_color = figure_style.RULE_COLORS["sliding"]
    speculation_color = figure_style.RULE_COLORS["speculation"]
    axes.bar(left, sliding, BAR_WIDTH, color=sliding_color, **BAR_EDGE)
    axes.bar(
        right,
        speculation,
        BAR_WIDTH,
        color=speculation_color,
        hatch=SPECULATION_HATCH,
        **BAR_EDGE,
    )
    cut = 1 - speculation / sliding
    text = f"−{cut:.0%}"
    axes.annotate(
        text,
        (right, speculation),
        xytext=(0, 2),
        ha="center",
        va="bottom",
        **ANNOTATION,
    )


def _draw_core_pair(
    axes, index: int, rule: str, one_core: dict, two_cores: dict
) -> None:
    """One rule's reaction tail on one core and on two, with each value."""
    left, right = _pair_positions(index)
    color = figure_style.RULE_COLORS[rule]
    one_tail = float(one_core[REACTION_TAIL])
    two_tail = float(two_cores[REACTION_TAIL])
    axes.bar(left, one_tail, BAR_WIDTH, color=color, **BAR_EDGE)
    axes.bar(
        right,
        two_tail,
        BAR_WIDTH,
        color=color,
        alpha=TWO_CORE_ALPHA,
        hatch=TWO_CORE_HATCH,
        **BAR_EDGE,
    )
    _label_bar(axes, left, one_tail)
    _label_bar(axes, right, two_tail)


def _draw_repair_pair(axes, index: int, pair) -> None:
    """The shots switching repaired and broke, and their ratio."""
    left, right = _pair_positions(index)
    axes.bar(left, pair.repaired, BAR_WIDTH, color=REPAIRED_COLOR, **BAR_EDGE)
    axes.bar(right, pair.broken, BAR_WIDTH, color=BROKEN_COLOR, **BAR_EDGE)
    ratio = pair.repaired / pair.broken
    text = f"{ratio:.1f}×"
    axes.annotate(
        text,
        (index, pair.repaired),
        xytext=(0, 3),
        ha="center",
        va="bottom",
        **ANNOTATION,
    )


def _label_bar(axes, position: float, value: float) -> None:
    text = f"{value:.0f}"
    axes.annotate(
        text,
        (position, value),
        xytext=(0, 1.5),
        ha="center",
        va="bottom",
        **ANNOTATION,
    )


def _pair_positions(index: int) -> tuple:
    """The left and right bar of a group's pair."""
    half = BAR_WIDTH / 2
    left = index - half
    right = index + half
    return left, right


def _name_groups(axes, labels: list) -> None:
    """One label under each group of bars."""
    count = len(labels)
    positions = range(count)
    axes.set_xticks(positions, labels)
    axes.grid(axis="x", visible=False)


def _swatch(color: str, hatch, alpha: float):
    """A legend swatch for a bar style."""
    return patches.Patch(
        facecolor=color, hatch=hatch, alpha=alpha, edgecolor="white"
    )


def _samples(rows: list, name: str, decoding: str) -> tuple:
    """One decoding's values and counts of one window measure."""
    values = []
    counts = []
    for row in rows:
        if row["name"] != name:
            continue
        if row["decoding"] != decoding:
            continue
        value = float(row["value_us"])
        count = int(row["count"])
        values.append(value)
        counts.append(count)
    return numpy.array(values), numpy.array(counts)


def _design_rows(sweep: list, rule: str, units: str) -> list:
    """One rule's designs on so many cores, by the share they target."""
    rows = []
    for row in sweep:
        if row["scheme"] == rule and row["strong_unit_count"] == units:
            rows.append(row)
    return sorted(rows, key=_target_share)


def _highest_share(sweep: list, rule: str, units: str) -> dict:
    rows = _design_rows(sweep, rule, units)
    return rows[-1]


def _target_share(row: dict) -> float:
    return float(row["share"])


def _percent(row: dict, column: str) -> float:
    share = float(row[column])
    return share * 100


def _share_text(row: dict) -> str:
    share = float(row["escalated_fraction"])
    return f"{share:.1%}"


def _rate_and_errors(row: dict) -> tuple:
    """The error per round and its distances to the 95% limits."""
    rate = float(row["logical_error_rate_per_round"])
    low = float(row["logical_error_rate_per_round_low"])
    high = float(row["logical_error_rate_per_round_high"])
    below = rate - low
    above = high - rate
    return rate, below, above


def _scientific_text(number: float) -> str:
    """A rate as a mantissa of one decimal times a power of ten."""
    power = math.log10(number)
    exponent = math.floor(power)
    scale = 10**exponent
    mantissa = number / scale
    return f"${mantissa:.1f} \\times 10^{{{exponent}}}$"


def _tick_text(number: float) -> str:
    """A log-axis tick: a power of ten, or a digit times one."""
    power = math.log10(number)
    exponent = math.floor(power)
    scale = 10**exponent
    mantissa = number / scale
    digit = round(mantissa)
    if digit == 1:
        return f"$10^{{{exponent}}}$"
    return f"${digit} \\times 10^{{{exponent}}}$"


def _strong_service_microseconds() -> float:
    """Correlated matching's fixed time for one sliding window."""
    per_round = config.CORRELATED_MATCHING_ROUND_MICROSECONDS_BY_DISTANCE[
        PILOT_DISTANCE
    ]
    distances = config.STRONG_DECODE_DISTANCES_BY_SCHEME["sliding"]
    rounds = distances * PILOT_DISTANCE
    service = per_round * rounds
    return round(service, 6)


def _stream_name(decoding: str) -> str:
    return (
        f"{decoding}_d{PILOT_DISTANCE}_p{PILOT_PROBABILITY}_s{STREAM_SEED}.json"
    )


def _only(rows: list, decoding: str) -> dict:
    """The one row of a decoding."""
    matching = []
    for row in rows:
        if row["decoding"] == decoding:
            matching.append(row)
    if len(matching) != 1:
        raise ValueError(f"no single row of {decoding}")
    return matching[0]


def _read(path: pathlib.Path) -> list:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        return list(reader)


if __name__ == "__main__":
    results_folder = pathlib.Path(sys.argv[1])
    main(results_folder, sys.argv[2])
