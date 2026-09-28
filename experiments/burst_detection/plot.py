"""The burst detection figures, drawn from the folder's saved rows.

`python plot.py <results folder>` writes the folder's plots/. Each d
and p gets its own folder, plots/d<d>_p<p>/, holding: the share of
bursts caught within 300 rounds of onset and the median delay of the
caught ones, size by strength, a panel per alarm line, each cell
holding its value; the share caught against the false alarm rate each
line asks for, a curve per strength, a panel per size; the false
alarms measured against those asked for; and each class's example
trace, a figure per alarm line. plots/ itself holds the figures across
d and p, one per alarm line: the share caught against d and the median
delay against d, a row of panels per strength and a column per p; the
share caught against p, a column per d; and one false alarm check with
every d and p. A burst is caught when copy B alarms at most 300 rounds
after its onset, half its 600-round decay.
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
CAUGHT_LABEL = f"share caught within {CATCH_DEADLINE_ROUNDS} rounds"
DELAY_LABEL = "median rounds to the first alarm"
ASKED_LABEL = "false alarms per s asked"


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
    class_rows = class_summaries(trials)
    checked_rows = false_alarm_rows(quiet)
    stats = catch_stats(class_rows)
    rate_points = sorted({(row["d"], row["p"]) for row in quiet})
    for distance, error_rate in rate_points:
        point = {"d": distance, "p": error_rate}
        suffix = _suffix(point)
        point_folder = plots_folder / suffix
        point_folder.mkdir(exist_ok=True)
        point_classes = plots.chosen(class_rows, **point)
        point_stats = plots.chosen(stats, **point)
        point_checks = plots.chosen(checked_rows, **point)
        point_traces = plots.chosen(traces, **point)
        draw_heatmaps(point_classes, point_folder, point)
        draw_catch_against_false_alarms(point_stats, point_folder, point)
        draw_false_alarm_check(point_checks, point_folder, point)
        draw_traces(point_traces, point_folder, point)
    draw_share_against_distance(stats, plots_folder)
    draw_share_against_error_rate(stats, plots_folder)
    draw_delay_against_distance(class_rows, plots_folder)
    draw_every_false_alarm_check(checked_rows, plots_folder)


def class_summaries(trials: list) -> list:
    """Per d, p, class and line: bursts, caught in time, median delay.

    The delay is the rounds from the onset to copy B's first alarm; the
    median is over the bursts B alarmed on at all, NaN where none.
    """
    delays_by_key = collections.defaultdict(list)
    for row in trials:
        key = (row["d"], row["p"], row["size"], row["strength"])
        line_key = (*key, row["alarm_line"])
        delay = _delay(row)
        delays_by_key[line_key].append(delay)
    rows = []
    for key, delays in delays_by_key.items():
        row = _class_row(key, delays)
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


def false_alarm_rows(quiet: list) -> list:
    """Each (d, p, line)'s false alarms per s asked and measured.

    low and high bound the measured rate's exact 95 percent Poisson
    interval.
    """
    rows = []
    for row in quiet:
        interval = _poisson_interval(row["alarms"], row["quiet_seconds"])
        checked = {
            "d": row["d"],
            "p": row["p"],
            "asked": row["alarm_line"],
            "measured": _measured_rate(row),
            "low": interval[0],
            "high": interval[1],
            "stream": f"{row['quiet_seconds']:.0f} s quiet",
        }
        rows.append(checked)
    return rows


def catch_stats(class_rows: list) -> list:
    """Each class summary as the counts count_rate reads."""
    stats = []
    for row in class_rows:
        stat = _catch_stat(row)
        stats.append(stat)
    return stats


def draw_heatmaps(class_rows: list, folder: pathlib.Path, point: dict) -> None:
    """The share caught in time, then the median delay, size by strength.

    Each cell holds its value. One colour bar serves a figure's panels,
    so its limits span them all: a share 0 to 1, a delay the figure's
    shortest to longest median.
    """
    suffix = _suffix(point)
    title = _title(point)
    delays = [row["median_delay"] for row in class_rows]
    shortest_delay = numpy.nanmin(delays)
    longest_delay = numpy.nanmax(delays)
    kinds = [
        ("caught_share", CAUGHT_LABEL, (0.0, 1.0), "{:.1%}"),
        ("median_delay", DELAY_LABEL, (shortest_delay, longest_delay), "{:g}"),
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
        figure.suptitle(f"{label}, {title}")
        path = folder / f"{value}_{suffix}.png"
        plots.save(figure, path)


def draw_catch_against_false_alarms(
    stats: list, folder: pathlib.Path, point: dict
) -> None:
    """The share caught in time against the rate each line asks for.

    The asked rate, not the measured one, places a line: a line whose
    quiet stream saw no false alarm measures zero, which a log axis
    cannot show, and the false alarm check already sets measured
    against asked.
    """
    suffix = _suffix(point)
    title = _title(point)
    figure, axis_by_size = plots.panels("size", SIZES, columns=2)
    for size, axis in axis_by_size.items():
        size_stats = plots.chosen(stats, size=size)
        plots.count_rate(
            axis,
            size_stats,
            x=ASKED_LABEL,
            hits="caught",
            total="bursts",
            curve="strength",
            order=STRENGTHS,
            is_y_log_scale=False,
        )
    figure.suptitle(f"caught within {CATCH_DEADLINE_ROUNDS} rounds, {title}")
    path = folder / f"catch_against_false_alarms_{suffix}.png"
    plots.save(figure, path)


def draw_false_alarm_check(
    checked_rows: list, folder: pathlib.Path, point: dict
) -> None:
    """False alarms measured on the quiet stream against those asked for."""
    suffix = _suffix(point)
    title = _title(point)
    figure, axis = pyplot.subplots(layout="constrained")
    plots.values(
        axis,
        checked_rows,
        x="asked",
        y="measured",
        curve="stream",
        low="low",
        high="high",
    )
    _name_false_alarm_axes(axis)
    axis.set_title(title)
    path = folder / f"false_alarm_check_{suffix}.png"
    plots.save(figure, path)


def draw_traces(traces: list, folder: pathlib.Path, point: dict) -> None:
    """Each class's example trial, copies A and B, a figure per line.

    The score is the largest group score over its level, so the line
    fires at one; it is drawn up to TRACE_CEILING, where the crossing
    shows, and a stronger score runs off the top.
    """
    suffix = _suffix(point)
    title = _title(point)
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
            f"example trials at {line:g} false alarms per s, {title}"
        )
        path = folder / f"traces_{line:g}_{suffix}.png"
        plots.save(figure, path)


def draw_share_against_distance(stats: list, folder: pathlib.Path) -> None:
    """The share caught in time against d, a column of panels per p."""
    _draw_share_across(stats, folder, ("d", "p"), is_x_log_scale=False)


def draw_share_against_error_rate(stats: list, folder: pathlib.Path) -> None:
    """The share caught in time against p, a column of panels per d."""
    _draw_share_across(stats, folder, ("p", "d"), is_x_log_scale=True)


def draw_delay_against_distance(class_rows: list, folder: pathlib.Path) -> None:
    """The median delay against d, a figure per line.

    It is laid out as the share against d: a row of panels per strength,
    a column per p, a curve per size. The y axis is log, since a
    saturating burst is caught in a few rounds and a weak one in
    hundreds, and the rows share it.
    """
    distances = sorted({row["d"] for row in class_rows})
    error_rates = sorted({row["p"] for row in class_rows})
    for line in LINES:
        figure, axis_by_pair = plots.panel_grid(
            "strength", STRENGTHS, "p", error_rates
        )
        for (strength, error_rate), axis in axis_by_pair.items():
            rows = plots.chosen(
                class_rows, alarm_line=line, strength=strength, p=error_rate
            )
            plots.values(
                axis, rows, x="d", y="median_delay", curve="size", order=SIZES
            )
            axis.set_yscale("log")
            axis.set_ylabel(DELAY_LABEL)
            _tick_each(axis, distances)
        figure.suptitle(f"{DELAY_LABEL} at {line:g} false alarms per s")
        path = folder / f"median_delay_against_d_{line:g}.png"
        plots.save(figure, path)


def draw_every_false_alarm_check(
    checked_rows: list, folder: pathlib.Path
) -> None:
    """False alarms measured against asked, a panel per d, a curve per p."""
    distances = sorted({row["d"] for row in checked_rows})
    error_rates = sorted({row["p"] for row in checked_rows})
    figure, axis_by_distance = plots.panels("d", distances)
    for distance, axis in axis_by_distance.items():
        rows = plots.chosen(checked_rows, d=distance)
        plots.values(
            axis,
            rows,
            x="asked",
            y="measured",
            curve="p",
            low="low",
            high="high",
            order=error_rates,
        )
        _name_false_alarm_axes(axis)
    figure.suptitle("false alarms on the quiet stream, every d and p")
    path = folder / "false_alarm_check.png"
    plots.save(figure, path)


def _draw_share_across(
    stats: list, folder: pathlib.Path, keys: tuple, is_x_log_scale: bool
) -> None:
    """The share caught in time against one key, a figure per line.

    keys is (x key, column key): a row of panels per strength, a column
    per value of the column key, a curve per size.
    """
    x_key, column_key = keys
    x_values = sorted({stat.json_metadata[x_key] for stat in stats})
    column_values = sorted({stat.json_metadata[column_key] for stat in stats})
    for line in LINES:
        figure, axis_by_pair = plots.panel_grid(
            "strength", STRENGTHS, column_key, column_values
        )
        for (strength, column_value), axis in axis_by_pair.items():
            wanted = {"alarm_line": line, "strength": strength}
            wanted[column_key] = column_value
            panel_stats = plots.chosen(stats, **wanted)
            plots.count_rate(
                axis,
                panel_stats,
                x=x_key,
                hits="caught",
                total="bursts",
                curve="size",
                order=SIZES,
                is_x_log_scale=is_x_log_scale,
                is_y_log_scale=False,
            )
            _tick_each(axis, x_values)
        figure.suptitle(f"{CAUGHT_LABEL} at {line:g} false alarms per s")
        path = folder / f"caught_share_against_{x_key}_{line:g}.png"
        plots.save(figure, path)


def _tick_each(axis: matplotlib.axes.Axes, x_values: list) -> None:
    """A labelled tick at each swept value and none between.

    The p values span about one decade, where a log axis labels only
    10^-3, and d's odd values fall between a linear axis's own ticks.
    """
    labels = [f"{value:g}" for value in x_values]
    axis.set_xticks(x_values, labels)
    axis.set_xticks([], minor=True)


def _name_false_alarm_axes(axis: matplotlib.axes.Axes) -> None:
    """Log axes, the labels and the dashed line where measured is asked."""
    axis.plot(LINES, LINES, color="grey", linestyle="--", label="asked")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(ASKED_LABEL)
    axis.set_ylabel("false alarms per s measured, 95% Poisson")


def _suffix(point: dict) -> str:
    """A (d, p)'s folder name and file suffix, d5_p0.003."""
    return f"d{point['d']}_p{point['p']:g}"


def _title(point: dict) -> str:
    """A (d, p)'s title, d = 5, p = 0.003."""
    return f"d = {point['d']}, p = {point['p']:g}"


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


def _class_row(key: tuple, delays: list) -> dict:
    """One class's summary: key is (d, p, size, strength, line)."""
    distance, error_rate, size, strength, line = key
    caught_delays = [delay for delay in delays if delay is not None]
    in_time = [
        delay for delay in caught_delays if delay <= CATCH_DEADLINE_ROUNDS
    ]
    median = numpy.nan
    if caught_delays:
        middle_delay = numpy.median(caught_delays)
        median = float(middle_delay)
    return {
        "d": distance,
        "p": error_rate,
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


def _catch_stat(row: dict) -> sinter.TaskStats:
    """A class's catches at one line, as the counts count_rate reads."""
    metadata = {
        "d": row["d"],
        "p": row["p"],
        "size": row["size"],
        "strength": row["strength"],
        "alarm_line": row["alarm_line"],
        ASKED_LABEL: row["alarm_line"],
    }
    counts = collections.Counter(
        {"caught": row["caught"], "bursts": row["bursts"]}
    )
    strong_id = (
        f"{row['d']} {row['p']} {row['size']} {row['strength']} "
        f"{row['alarm_line']}"
    )
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
    alarm_degrees = 2 * alarms
    low_count = 0.0
    if alarms > 0:
        low_quantile = scipy.stats.chi2.ppf(LOWER_TAIL, alarm_degrees)
        low_count = low_quantile / 2
    high_degrees = alarm_degrees + 2
    high_quantile = scipy.stats.chi2.ppf(UPPER_TAIL, high_degrees)
    high_count = high_quantile / 2
    low_rate = low_count / seconds
    high_rate = high_count / seconds
    return low_rate, high_rate


if __name__ == "__main__":
    results_folder = pathlib.Path(sys.argv[1])
    main(results_folder)
