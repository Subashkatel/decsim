"""The figure kinds an experiment's plot.py draws, styled in one place.

Rates follow sinter: an error rate is drawn with sinter.plot_error_rate,
its line through the most likely rate and its band over every rate
within a likelihood factor of it (sinter/_plotting.py:317-419). Values
read plain rows, one dict per row, as decsim.results.load
returns them. A key names a column of a row; on a sinter.TaskStats it
names a json_metadata field, except "decoder", which is the stat's own.

A curve's colour and marker come from its value's place in `order`, the
list of every value the curve key takes across a figure, so a value
keeps its style in a panel that lacks another; without `order` the
place is among the values the call draws, sorted.

A point with no failures has no likeliest rate, only an upper bound; it
is drawn as a hollow marker at that bound in its curve's style, off the
line, so a reader sees where it stands and that its rate lies below.
"""

import collections
import functools
import math
import operator
import pathlib
from collections.abc import Mapping
from typing import Optional, Union

import matplotlib.pyplot as pyplot
import matplotlib.ticker
import numpy
import sinter

DOTS_PER_INCH = 150
PANEL_WIDTH_INCHES = 5.0
PANEL_HEIGHT_INCHES = 4.2
MINOR_GRID_ALPHA = 0.3
# sinter.plot_error_rate's default band (sinter/_plotting.py), so a
# point's bound reads on the same scale as the other points' bands
LIKELIHOOD_FACTOR = 1e3
# sinter's marker order (sinter/_plotting.py:15), so a curve reads the
# same here as in a figure sinter draws itself
MARKERS = "ov*sp^<>8PhH+xXDd"


def panels(key: str, values: list, columns: int = 3) -> tuple:
    """A figure with one axis per value, titled "key = value", axes shared.

    Returns the figure and a dict from each value to its axis.
    """
    panel_count = len(values)
    column_count = min(columns, panel_count)
    exact_rows = panel_count / column_count
    row_count = math.ceil(exact_rows)
    width = PANEL_WIDTH_INCHES * column_count
    height = PANEL_HEIGHT_INCHES * row_count
    figure, axes = pyplot.subplots(
        row_count,
        column_count,
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=(width, height),
        layout="constrained",
    )
    flat_axes = list(axes.flat)
    shown_axes = flat_axes[:panel_count]
    axis_by_value = {}
    for axis, value in zip(shown_axes, values, strict=True):
        axis.set_title(f"{key} = {value}")
        axis_by_value[value] = axis
    for axis in flat_axes[panel_count:]:
        axis.set_visible(False)
    return figure, axis_by_value


def chosen(items: list, **wanted) -> list:
    """The stats or rows whose values under the wanted keys are the wanted."""
    kept = []
    for item in items:
        if _holds(item, wanted):
            kept.append(item)
    return kept


def error_rate(
    axis: pyplot.Axes,
    stats: list,
    x: str,
    curve: str,
    marker: Optional[str] = None,
    rounds: int = 1,
    order: Optional[list] = None,
) -> None:
    """The logical error rate against x, a curve per value of curve.

    marker, when given, keeps curve's colours and gives each of its
    values a marker and a line style. rounds is the rounds every shot of
    the call holds; above one the rate is per round, sinter's
    failure_units_per_shot_func (sinter/_plotting.py:338-342), and one is
    sinter's own default, a rate per shot.

    A point with no errors is drawn as its bound (the module says how):
    sinter draws it as a band from zero with no marker
    (sinter/_plotting.py:399-400), which on a log axis runs to the floor.
    A point whose likeliest shot rate is above one half is left out of a
    per-round figure: for an even round count no per-round rate gives
    it (decsim/experiments/report.py _is_above_half), and sinter maps it
    to its complement, near one.
    """
    x_of = functools.partial(_value, key=x)
    group_of = functools.partial(_group, curve=curve, marker=marker)
    places = _places(stats, curve, order)
    style_of = functools.partial(
        _place_style, curve=curve, marker=marker, places=places
    )
    has_rate = functools.partial(_has_rate, rounds=rounds)
    sinter.plot_error_rate(
        ax=axis,
        stats=stats,
        x_func=x_of,
        failure_units_per_shot_func=lambda _stat: rounds,
        group_func=group_of,
        filter_func=has_rate,
        plot_args_func=style_of,
    )
    for stat in stats:
        if stat.errors == 0:
            _draw_error_free(axis, stat, (x_of, curve, rounds), places)
    unit = "shot"
    if rounds > 1:
        unit = "round"
    legend_title = curve
    if marker is not None:
        legend_title = f"{curve}, {marker}"
    _log_axes(axis)
    _style(axis, x, f"logical error rate per {unit}", legend_title)


def values(
    axis: pyplot.Axes,
    rows: list,
    x: str,
    y: str,
    curve: str,
    low: Optional[str] = None,
    high: Optional[str] = None,
    order: Optional[list] = None,
) -> None:
    """The y column against x, a curve per curve value; low, high as bars."""
    curves = _curves(rows, curve, x)
    for curve_value, curve_rows in curves.items():
        style = _curve_style(curve_value, curves, order)
        style["label"] = str(curve_value)
        _draw_values(axis, curve_rows, (x, y, low, high), style)
    _style(axis, x, y, curve)


def bounds(
    axis: pyplot.Axes,
    rows: list,
    x: str,
    high: str,
    curve: str,
    order: Optional[list] = None,
) -> None:
    """Points with no failures, as hollow markers at their high column."""
    curves = _curves(rows, curve, x)
    for curve_value, curve_rows in curves.items():
        style = _curve_style(curve_value, curves, order)
        xs = [row[x] for row in curve_rows]
        ys = [row[high] for row in curve_rows]
        _draw_bound(axis, xs, ys, style)


def save(figure: pyplot.Figure, path: Union[str, pathlib.Path]) -> None:
    """One figure written and closed, so a run of them holds one open."""
    figure.savefig(path, dpi=DOTS_PER_INCH)
    pyplot.close(figure)


def _value(item, key: str):
    """A row's column, or a stat's decoder or json_metadata field."""
    if isinstance(item, Mapping):
        return item[key]
    if key == "decoder":
        return item.decoder
    return item.json_metadata[key]


def _holds(item, wanted: dict) -> bool:
    """Whether the item holds every wanted value."""
    for key, wanted_value in wanted.items():
        found = _value(item, key)
        if found != wanted_value:
            return False
    return True


def _group(stat: sinter.TaskStats, curve: str, marker: Optional[str]) -> dict:
    """A stat's curve, in the keys sinter.plot_custom reads off a dict.

    sinter orders the curves by "sort" and gives one marker and line
    style per "marker" and "linestyle" value (sinter/_plotting.py:350-357);
    the colour, and the marker when no second key is given, come from
    _place_style instead, since sinter ranks them among the curves of
    one call only.
    """
    curve_value = _value(stat, curve)
    label = str(curve_value)
    group = {"label": label, "sort": curve_value}
    if marker is None:
        return group
    marker_value = _value(stat, marker)
    group["label"] = f"{curve_value}, {marker_value}"
    group["marker"] = marker_value
    group["linestyle"] = marker_value
    group["sort"] = (curve_value, marker_value)
    return group


def _curves(rows: list, curve: str, order: str) -> dict:
    """The rows of each curve value, the values sorted, each curve by order."""
    rows_by_curve = collections.defaultdict(list)
    for row in rows:
        curve_value = row[curve]
        rows_by_curve[curve_value].append(row)
    order_of = operator.itemgetter(order)
    curves = {}
    for curve_value in sorted(rows_by_curve):
        curve_rows = rows_by_curve[curve_value]
        curves[curve_value] = sorted(curve_rows, key=order_of)
    return curves


def _draw_values(
    axis: pyplot.Axes, rows: list, columns: tuple, style: dict
) -> None:
    """One curve of values, with error bars when low and high are named.

    columns is (x, y, low, high); style holds the curve's colour, marker
    and label.
    """
    x, y, low, high = columns
    xs = [row[x] for row in rows]
    ys = numpy.array([row[y] for row in rows])
    if low is None:
        axis.plot(xs, ys, **style)
        return
    lows = numpy.array([row[low] for row in rows])
    highs = numpy.array([row[high] for row in rows])
    below = ys - lows
    above = highs - ys
    bars = numpy.array([below, above])
    axis.errorbar(xs, ys, yerr=bars, capsize=3, **style)


def _has_rate(stat: sinter.TaskStats, rounds: int) -> bool:
    """Whether a point has a likeliest rate in the figure's unit."""
    if stat.errors == 0:
        return False
    if rounds == 1:
        return True
    scored_shots = stat.shots - stat.discards
    shot_rate = stat.errors / scored_shots
    return shot_rate <= 0.5


def _places(stats: list, curve: str, order: Optional[list]) -> list:
    """Every curve value in style order: order, or the values sorted."""
    if order is not None:
        return order
    curve_values = {_value(stat, curve) for stat in stats}
    return sorted(curve_values)


def _draw_error_free(
    axis: pyplot.Axes, stat: sinter.TaskStats, keys: tuple, places: list
) -> None:
    """A point with no errors at its bound, in its curve's style.

    keys is (x_of, curve, rounds). The bound is the high end of sinter's
    band for no hits, in the figure's unit.
    """
    x_of, curve, rounds = keys
    scored_shots = stat.shots - stat.discards
    fit = sinter.fit_binomial(
        num_shots=scored_shots,
        num_hits=0,
        max_likelihood_factor=LIKELIHOOD_FACTOR,
    )
    bound = sinter.shot_error_rate_to_piece_error_rate(fit.high, pieces=rounds)
    curve_value = _value(stat, curve)
    style = _place(curve_value, places)
    x_value = x_of(stat)
    _draw_bound(axis, [x_value], [bound], style)


def _draw_bound(axis: pyplot.Axes, xs: list, ys: list, style: dict) -> None:
    """Hollow markers off the curve's line, kept out of the legend."""
    axis.plot(
        xs,
        ys,
        linestyle="none",
        markerfacecolor="none",
        label="_nolegend_",
        **style,
    )


def _place_style(
    _index: int,
    _group_key: dict,
    group_stats: list,
    curve: str,
    marker: Optional[str],
    places: list,
) -> dict:
    """One sinter curve's colour, and its marker when marker is None."""
    first_stat = group_stats[0]
    curve_value = _value(first_stat, curve)
    style = _place(curve_value, places)
    if marker is not None:
        del style["marker"]
    return style


def _curve_style(curve_value, curves: dict, order: Optional[list]) -> dict:
    """A row curve's colour and marker, by its value's place."""
    places = order
    if places is None:
        places = list(curves)
    return _place(curve_value, places)


def _place(curve_value, places: list) -> dict:
    """The colour and marker of the value at its place in places."""
    index = places.index(curve_value)
    marker_index = index % len(MARKERS)
    return {"color": f"C{index}", "marker": MARKERS[marker_index]}


def _log_axes(axis: pyplot.Axes) -> None:
    """A rate spans decades on both axes, labelled at the decades only.

    Matplotlib labels minor ticks on an axis that spans few decades
    (LogFormatter's minor_thresholds), and at 3 and 4 times a decade
    those labels run into each other.
    """
    axis.set_xscale("log")
    axis.set_yscale("log")
    no_labels = matplotlib.ticker.NullFormatter()
    axis.xaxis.set_minor_formatter(no_labels)
    axis.yaxis.set_minor_formatter(no_labels)


def _style(
    axis: pyplot.Axes, x_label: str, y_label: str, legend_title: str
) -> None:
    """The labels, both grids and the legend every kind's axis carries."""
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(which="major")
    axis.grid(which="minor", alpha=MINOR_GRID_ALPHA)
    axis.legend(title=legend_title)
