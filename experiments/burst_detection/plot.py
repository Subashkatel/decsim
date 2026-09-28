"""The burst detection figures, drawn from a results folder's rows.

`python plot.py <results folder>` writes two kinds into its plots/:
rounds_to_90_p<p>.png, one per p, the rounds until 90% of bursts are
caught against burst size, a line per strength and a panel per d, at 1
false alarm per second, a class that never gets there within the 1,000
rounds after onset marked open at "never"; and
false_alarms.png, the false alarms measured against each target line.
"""

import collections
import csv
import math
import pathlib
import sys

import numpy
import run
import scipy.stats

import decsim.plots as plots

LINE = 1.0
SHARE = 0.9
HORIZON_ROUNDS = run.ROUNDS - run.ONSET_ROUND
SIZES = list(run.RADIUS_BY_SIZE)
PLACES = numpy.arange(len(SIZES))
NEVER_OFFSET = {"weak": -0.12, "strong": 0.0, "saturating": 0.12}
NEVER_ROUNDS = 1800
ROUND_LIMITS = (0.8, 3000)
ROUND_TICKS = [1, 3, 10, 30, 100, 300, 1000, NEVER_ROUNDS]
ROUND_LABELS = ["1", "3", "10", "30", "100", "300", "1000", "never"]


def main(folder: pathlib.Path) -> None:
    """Both kinds of figure, into the folder's plots/."""
    trials_path = folder / "trials.csv"
    quiet_path = folder / "quiet.csv"
    plots_folder = folder / "plots"
    plots_folder.mkdir(exist_ok=True)
    delays_by_class = collections.defaultdict(list)
    for row in _read(trials_path):
        key = (row["p"], row["d"], row["strength"], row["size"])
        delay = _delay(row)
        if row["alarm_line"] == LINE:
            delays_by_class[key].append(delay)
    rounds_by_class = {}
    for key, delays in delays_by_class.items():
        rounds_by_class[key] = rounds_to_share(delays)
    error_rates = sorted({key[0] for key in rounds_by_class})
    for error_rate in error_rates:
        draw_rounds_to_share(rounds_by_class, error_rate, plots_folder)
    quiet_rows = _read(quiet_path)
    draw_false_alarms(quiet_rows, plots_folder)


def rounds_to_share(delays: list) -> float:
    """Rounds after onset until SHARE of the bursts are caught, inf never."""
    ordered = sorted(delays)
    exact_place = SHARE * len(ordered)
    place = math.ceil(exact_place) - 1
    return ordered[place]


def draw_rounds_to_share(
    rounds_by_class: dict, error_rate: float, folder: pathlib.Path
) -> None:
    """One p's figure: a panel per d, a line per strength over sizes.

    A class the rows lack is drawn as never reaching SHARE.
    """
    distances = sorted({key[1] for key in rounds_by_class})
    figure, axis_by_distance = plots.panels("d", distances)
    for distance, axis in axis_by_distance.items():
        for strength in run.MULTIPLE_BY_STRENGTH:
            keys = [(error_rate, distance, strength, size) for size in SIZES]
            rounds = [rounds_by_class.get(key, math.nan) for key in keys]
            _draw_strength(axis, strength, rounds)
        _name_round_axes(axis)
    figure.suptitle(f"p = {error_rate:g}, 1 false alarm per second")
    path = folder / f"rounds_to_90_p{error_rate:g}.png"
    plots.save(figure, path)


def draw_false_alarms(quiet_rows: list, folder: pathlib.Path) -> None:
    """Measured against target: a panel per line, x = p, a curve per d."""
    seconds_by_key = collections.defaultdict(float)
    alarms_by_key = collections.defaultdict(int)
    for row in quiet_rows:
        key = (row["d"], row["p"], row["alarm_line"])
        seconds_by_key[key] += row["quiet_seconds"]
        alarms_by_key[key] += row["alarms"]
    rows = []
    for (distance, error_rate, line), seconds in seconds_by_key.items():
        alarms = alarms_by_key[(distance, error_rate, line)]
        interval = _poisson_interval(alarms)
        row = {"d": distance, "p": error_rate, "alarm_line": line}
        row["measured"] = alarms / seconds
        row["low"] = interval[0] / seconds
        row["high"] = interval[1] / seconds
        rows.append(row)
    error_rates = sorted({row["p"] for row in rows})
    rate_labels = [f"{rate:g}" for rate in error_rates]
    lines = run.FALSE_ALARMS_PER_SECOND
    figure, axis_by_line = plots.panels("target false alarms per second", lines)
    for line, axis in axis_by_line.items():
        line_rows = plots.chosen(rows, alarm_line=line)
        plots.values(axis, line_rows, "p", "measured", "d", "low", "high")
        axis.axhline(line, color="grey", linestyle="--")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(error_rates, rate_labels)
        axis.set_xticks([], minor=True)
        # six rates in one decade: slanted, their labels do not meet
        axis.tick_params(axis="x", labelrotation=45)
        axis.set_xlabel("physical error rate p")
        axis.set_ylabel("false alarms per second")
    path = folder / "false_alarms.png"
    plots.save(figure, path)


def _draw_strength(axis, strength: str, rounds: list) -> None:
    """One strength's line; a class never at SHARE is open at never."""
    values = numpy.array(rounds)
    is_never = values > HORIZON_ROUNDS
    caught = numpy.where(is_never, numpy.nan, values)
    # a catch in the onset round is drawn at 1, the log axis's floor
    shown = numpy.maximum(caught, 1)
    (line,) = axis.plot(PLACES, shown, "o-", label=strength)
    never_places = PLACES[is_never] + NEVER_OFFSET[strength]
    never_rounds = numpy.full(len(never_places), NEVER_ROUNDS)
    color = line.get_color()
    open_style = {"color": color, "markerfacecolor": "none"}
    axis.plot(never_places, never_rounds, "o", **open_style)


def _name_round_axes(axis) -> None:
    axis.set_yscale("log")
    axis.set_ylim(*ROUND_LIMITS)
    axis.set_yticks(ROUND_TICKS, ROUND_LABELS)
    axis.minorticks_off()
    axis.set_xticks(PLACES, SIZES)
    axis.set_xlabel("burst size")
    axis.set_ylabel("rounds until 90% of bursts are caught")
    axis.grid(alpha=plots.MINOR_GRID_ALPHA)
    axis.legend(title="burst strength")
    axis.label_outer()


def _delay(row: dict) -> float:
    if row["first_alarm_b"] is None:
        return math.inf
    return row["first_alarm_b"] - run.ONSET_ROUND


def _poisson_interval(alarms: int) -> tuple:
    """The exact 95 percent interval of a Poisson count (Garwood 1936)."""
    alarm_degrees = 2 * alarms
    low_quantile = 0.0
    if alarms > 0:
        low_quantile = scipy.stats.chi2.ppf(0.025, alarm_degrees)
    high_degrees = alarm_degrees + 2
    high_quantile = scipy.stats.chi2.ppf(0.975, high_degrees)
    return low_quantile / 2, high_quantile / 2


def _read(path: pathlib.Path) -> list:
    """A results file's rows, numbers as numbers, an empty cell None."""
    rows = []
    with path.open(newline="") as results_file:
        for row in csv.DictReader(results_file):
            typed = {key: _cell(text) for key, text in row.items()}
            rows.append(typed)
    return rows


def _cell(text: str):
    if text == "":
        return None
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            continue
    return text


if __name__ == "__main__":
    results_folder = pathlib.Path(sys.argv[1])
    main(results_folder)
