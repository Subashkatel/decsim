"""The burst detection figures, drawn from the folder's saved rows.

`python plot.py <results folder>` writes the folder's plots/, one set
per d and p: the share of bursts caught within 300 rounds of onset and
the median delay of the caught ones, size by strength, a panel per
alarm line; the share caught against the false alarms measured on the
quiet stream, a curve per strength, a panel per size; the false alarms
measured against those asked for; and each class's example trace, a
figure per alarm line. A burst is caught when copy B alarms at most
300 rounds after its onset, half its 600-round decay.
"""

import collections
import csv
import itertools
import pathlib
import sys

import matplotlib.axes
import matplotlib.pyplot as pyplot
import numpy
import run
import scipy.stats
import sinter

import decsim.plots as plots

CATCH_DEADLINE_ROUNDS = 300
# the exact Poisson interval's two tails, 95 percent between them
LOWER_TAIL = 0.025
UPPER_TAIL = 0.975
SIZES = list(run.RADIUS_BY_SIZE)
STRENGTHS = list(run.FIRING_MULTIPLE_BY_STRENGTH)
LINES = list(run.FALSE_ALARMS_PER_SECOND)
TRACE_CEILING = 2.0


def main(folder: pathlib.Path) -> None:
    """Every figure of the folder's rows, into its plots/."""
    trials_path = folder / "trials.csv"
    quiet_path = folder / "quiet.csv"
    traces_path = folder / "traces.csv"
    trials = _read(trials_path)
    quiet_parts = _read(quiet_path)
    quiet = quiet_totals(quiet_parts)
    traces = _read(traces_path)
    plots_folder = folder / "plots"
    plots_folder.mkdir(exist_ok=True)
    rate_points = sorted({(row["d"], row["p"]) for row in quiet})
    for distance, error_rate in rate_points:
        suffix = f"d{distance}_p{error_rate:g}"
        point_trials = plots.chosen(trials, d=distance, p=error_rate)
        point_quiet = plots.chosen(quiet, d=distance, p=error_rate)
        point_traces = plots.chosen(traces, d=distance, p=error_rate)
        class_rows = class_summaries(point_trials)
        draw_heatmaps(class_rows, plots_folder, suffix)
        draw_catch_against_false_alarms(
            class_rows, point_quiet, plots_folder, suffix
        )
        draw_false_alarm_check(point_quiet, plots_folder, suffix)
        draw_traces(point_traces, plots_folder, suffix)


def class_summaries(trials: list) -> list:
    """Per class and line: bursts, caught in time, median delay caught.

    The delay is the rounds from the onset to copy B's first alarm; the
    median is over the bursts B alarmed on at all, NaN where none.
    """
    delays_by_key = collections.defaultdict(list)
    for row in trials:
        key = (row["size"], row["strength"], row["alarm_line"])
        delay = _delay(row)
        delays_by_key[key].append(delay)
    rows = []
    for (size, strength, line), delays in delays_by_key.items():
        caught_delays = [delay for delay in delays if delay is not None]
        in_time = [
            delay for delay in caught_delays if delay <= CATCH_DEADLINE_ROUNDS
        ]
        median = numpy.nan
        if caught_delays:
            middle_delay = numpy.median(caught_delays)
            median = float(middle_delay)
        row = {
            "size": size,
            "strength": strength,
            "size_place": SIZES.index(size),
            "strength_place": STRENGTHS.index(strength),
            "alarm_line": line,
            "bursts": len(delays),
            "caught": len(in_time),
            "caught_share": len(in_time) / len(delays),
            "median_delay": median,
        }
        rows.append(row)
    return rows


def quiet_totals(quiet_parts: list) -> list:
    """Each (d, p, line)'s quiet seconds and alarms, summed over parts."""
    totals = {}
    for row in quiet_parts:
        key = (row["d"], row["p"], row["alarm_line"])
        total = totals.setdefault(key, {"quiet_seconds": 0.0, "alarms": 0})
        total["quiet_seconds"] += row["quiet_seconds"]
        total["alarms"] += row["alarms"]
    rows = []
    for (distance, error_rate, line), total in totals.items():
        labels = {"d": distance, "p": error_rate, "alarm_line": line}
        row = labels | total
        rows.append(row)
    return rows


def draw_heatmaps(class_rows: list, folder: pathlib.Path, suffix: str) -> None:
    """The share caught in time, then the median delay, size by strength.

    Each cell holds its value. One colour bar serves a figure's panels,
    so its limits span them all: a share 0 to 1, a delay the figure's
    shortest to longest median.
    """
    caught_label = f"share caught within {CATCH_DEADLINE_ROUNDS} rounds"
    delays = [row["median_delay"] for row in class_rows]
    shortest_delay = numpy.nanmin(delays)
    longest_delay = numpy.nanmax(delays)
    delay_label = "median rounds to the first alarm"
    kinds = [
        ("caught_share", caught_label, (0.0, 1.0), "{:.1%}"),
        ("median_delay", delay_label, (shortest_delay, longest_delay), "{:g}"),
    ]
    for value, label, limits, cell_format in kinds:
        figure, axis_by_line = plots.panels("false alarms per s", LINES)
        for line, axis in axis_by_line.items():
            line_rows = plots.chosen(class_rows, alarm_line=line)
            mesh = plots.heatmap(
                axis,
                line_rows,
                x="size_place",
                y="strength_place",
                value=value,
                is_log_scale=False,
                limits=limits,
                cell_format=cell_format,
            )
            _name_places(axis)
        panel_axes = axis_by_line.values()
        panel_list = list(panel_axes)
        figure.colorbar(mesh, ax=panel_list, label=label)
        figure.suptitle(f"{label}, {suffix}")
        path = folder / f"{value}_{suffix}.png"
        plots.save(figure, path)


def draw_catch_against_false_alarms(
    class_rows: list, quiet: list, folder: pathlib.Path, suffix: str
) -> None:
    """The share caught in time against false alarms measured, per line."""
    measured_by_line = {}
    for row in quiet:
        measured_by_line[row["alarm_line"]] = _measured_rate(row)
    stats = []
    for row in class_rows:
        stat = _catch_stat(row, measured_by_line)
        stats.append(stat)
    figure, axis_by_size = plots.panels("size", SIZES, columns=2)
    for size, axis in axis_by_size.items():
        size_stats = plots.chosen(stats, size=size)
        plots.count_rate(
            axis,
            size_stats,
            x="false alarms per s",
            hits="caught",
            total="bursts",
            curve="strength",
            order=STRENGTHS,
        )
    figure.suptitle(f"caught within {CATCH_DEADLINE_ROUNDS} rounds, {suffix}")
    path = folder / f"catch_against_false_alarms_{suffix}.png"
    plots.save(figure, path)


def draw_false_alarm_check(
    quiet: list, folder: pathlib.Path, suffix: str
) -> None:
    """False alarms measured on the quiet stream against those asked for."""
    rows = []
    for row in quiet:
        interval = _poisson_interval(row["alarms"], row["quiet_seconds"])
        checked = {
            "asked": row["alarm_line"],
            "measured": _measured_rate(row),
            "low": interval[0],
            "high": interval[1],
            "stream": f"{row['quiet_seconds']:.0f} s quiet",
        }
        rows.append(checked)
    figure, axis = pyplot.subplots(layout="constrained")
    plots.values(
        axis,
        rows,
        x="asked",
        y="measured",
        curve="stream",
        low="low",
        high="high",
    )
    axis.plot(LINES, LINES, color="grey", linestyle="--", label="asked")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("false alarms per s asked")
    axis.set_ylabel("false alarms per s measured, 95% Poisson")
    axis.set_title(suffix)
    path = folder / f"false_alarm_check_{suffix}.png"
    plots.save(figure, path)


def draw_traces(traces: list, folder: pathlib.Path, suffix: str) -> None:
    """Each class's example trial, copies A and B, a figure per line.

    The score is the largest group score over its level, so the line
    fires at one; it is drawn up to TRACE_CEILING, where the crossing
    shows, and a stronger score runs off the top.
    """
    class_pairs = itertools.product(SIZES, STRENGTHS)
    classes = list(class_pairs)
    class_names = [f"{size}, {strength}" for size, strength in classes]
    for line in LINES:
        figure, axis_by_class = plots.panels("burst", class_names, columns=3)
        for (size, strength), name in zip(classes, class_names, strict=True):
            axis = axis_by_class[name]
            rows = plots.chosen(
                traces, size=size, strength=strength, alarm_line=line
            )
            plots.values(
                axis, rows, x="round", y="score_over_level", curve="copy"
            )
            axis.axhline(1.0, color="grey", linestyle="--")
            axis.axvline(run.ONSET_ROUND, color="grey", linestyle=":")
            axis.set_ylim(0.0, TRACE_CEILING)
        figure.suptitle(
            f"example trials at {line:g} false alarms per s, {suffix}"
        )
        path = folder / f"traces_{line:g}_{suffix}.png"
        plots.save(figure, path)


def _read(path: pathlib.Path) -> list:
    """A results file's rows, numbers read as numbers, an empty cell None."""
    with path.open(newline="") as results_file:
        reader = csv.DictReader(results_file)
        rows = []
        for row in reader:
            typed = {key: _cell(text) for key, text in row.items()}
            rows.append(typed)
    return rows


def _cell(text: str):
    """An int, else a float, else the text; empty is None."""
    if text == "":
        return None
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            continue
    return text


def _delay(row: dict):
    first_alarm = row["first_alarm_b"]
    if first_alarm is None:
        return None
    return first_alarm - run.ONSET_ROUND


def _name_places(axis: matplotlib.axes.Axes) -> None:
    """The size and strength names on the heatmap's place ticks."""
    size_places = range(len(SIZES))
    strength_places = range(len(STRENGTHS))
    axis.set_xticks(size_places, SIZES)
    axis.set_yticks(strength_places, STRENGTHS)
    axis.set_xlabel("size")
    axis.set_ylabel("strength")


def _measured_rate(quiet_row: dict) -> float:
    return quiet_row["alarms"] / quiet_row["quiet_seconds"]


def _catch_stat(row: dict, measured_by_line: dict) -> sinter.TaskStats:
    """A class's catches at one line, as the counts count_rate reads."""
    measured = measured_by_line[row["alarm_line"]]
    metadata = {
        "size": row["size"],
        "strength": row["strength"],
        "false alarms per s": measured,
    }
    counts = collections.Counter(
        {"caught": row["caught"], "bursts": row["bursts"]}
    )
    strong_id = f"{row['size']} {row['strength']} {row['alarm_line']}"
    return sinter.TaskStats(
        strong_id=strong_id,
        decoder="masked_regional_cusum",
        json_metadata=metadata,
        shots=row["bursts"],
        errors=0,
        custom_counts=counts,
    )


def _poisson_interval(alarms: int, seconds: float) -> tuple:
    """The exact 95 percent interval of a Poisson rate (Garwood 1936)."""
    low_count = 0.0
    if alarms > 0:
        low_degrees = 2 * alarms
        low_quantile = scipy.stats.chi2.ppf(LOWER_TAIL, low_degrees)
        low_count = low_quantile / 2
    high_degrees = 2 * alarms + 2
    high_quantile = scipy.stats.chi2.ppf(UPPER_TAIL, high_degrees)
    high_count = high_quantile / 2
    return low_count / seconds, high_count / seconds


if __name__ == "__main__":
    results_folder = pathlib.Path(sys.argv[1])
    main(results_folder)
