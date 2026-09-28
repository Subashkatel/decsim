"""The figure kinds an experiment's plot.py draws, styled in one place.

Rates follow sinter: a binomial rate is drawn with sinter.plot_custom,
its line through the most likely rate and its band over every rate
within a likelihood factor of it (sinter/_plotting.py:317-419). The
other kinds read plain rows, one dict per row, as decsim.results.load
returns them. A key names a column of a row; on a sinter.TaskStats it
names a json_metadata field, except "decoder", which is the stat's own.
"""

import collections
import functools
import math
import operator
import pathlib
from collections.abc import Mapping
from typing import Optional, Union

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as pyplot
import numpy
import sinter

DOTS_PER_INCH = 150
PANEL_WIDTH_INCHES = 5.0
PANEL_HEIGHT_INCHES = 4.2
# sinter.plot_error_rate's own band, 1e3 times less likely than the best
# fit (sinter/_plotting.py:328)
LIKELIHOOD_FACTOR = 1e3
HISTOGRAM_BINS = 30
MINOR_GRID_ALPHA = 0.3


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
) -> None:
    """The logical error rate against x, a curve per value of curve.

    marker, when given, keeps curve's colours and gives each of its
    values a marker and a line style. rounds is the rounds every shot of
    the call holds; above one the rate is per round, sinter's
    failure_units_per_shot_func (sinter/_plotting.py:338-342), and one is
    sinter's own default, a rate per shot.
    """
    x_of = functools.partial(_value, key=x)
    group_of = functools.partial(_group, curve=curve, marker=marker)
    sinter.plot_error_rate(
        ax=axis,
        stats=stats,
        x_func=x_of,
        failure_units_per_shot_func=lambda _stat: rounds,
        group_func=group_of,
    )
    unit = "shot"
    if rounds > 1:
        unit = "round"
    legend_title = curve
    if marker is not None:
        legend_title = f"{curve}, {marker}"
    _log_axes(axis)
    _style(axis, x, f"logical error rate per {unit}", legend_title)


def count_rate(
    axis: pyplot.Axes, stats: list, x: str, hits: str, total: str, curve: str
) -> None:
    """The share hits of total against x, both custom counts, with a band.

    The band is the one error_rate draws, from sinter.fit_binomial over
    the two counts (sinter/_probability_util.py:327).
    """
    x_of = functools.partial(_value, key=x)
    share_of = functools.partial(_count_share, hits=hits, total=total)
    group_of = functools.partial(_group, curve=curve, marker=None)
    sinter.plot_custom(
        ax=axis, stats=stats, x_func=x_of, y_func=share_of, group_func=group_of
    )
    _log_axes(axis)
    _style(axis, x, f"{hits} / {total}", curve)


def values(
    axis: pyplot.Axes,
    rows: list,
    x: str,
    y: str,
    curve: str,
    low: Optional[str] = None,
    high: Optional[str] = None,
) -> None:
    """The y column against x, a curve per curve value; low, high as bars."""
    curves = _curves(rows, curve, x)
    for curve_value, curve_rows in curves.items():
        label = str(curve_value)
        _draw_values(axis, curve_rows, x, y, low, high, label)
    _style(axis, x, y, curve)


def distribution(
    axis: pyplot.Axes,
    rows: list,
    value: str,
    curve: str,
    count: Optional[str] = None,
) -> None:
    """A histogram of value per curve, each summing to one, its median dashed.

    count names a column holding how many times its row's value was
    seen; without it every row is seen once.
    """
    all_values = [row[value] for row in rows]
    edges = numpy.histogram_bin_edges(all_values, bins=HISTOGRAM_BINS)
    curves = _curves(rows, curve, value)
    for index, curve_value in enumerate(curves):
        curve_rows = curves[curve_value]
        seen_values, counts = _seen_values(curve_rows, value, count)
        weights = counts / counts.sum()
        color = f"C{index}"
        label = str(curve_value)
        axis.hist(
            seen_values,
            bins=edges,
            weights=weights,
            histtype="step",
            color=color,
            label=label,
        )
        median = _median(seen_values, counts)
        axis.axvline(median, color=color, linestyle="--")
    _style(axis, value, "share", curve)


def heatmap(
    axis: pyplot.Axes, rows: list, x: str, y: str, value: str, label: str
) -> None:
    """The value column on an x by y grid, log colours, zero cells white."""
    xs = sorted({row[x] for row in rows})
    ys = sorted({row[y] for row in rows})
    grid = numpy.zeros((len(ys), len(xs)))
    for row in rows:
        column = xs.index(row[x])
        line = ys.index(row[y])
        grid[line, column] = row[value]
    masked = numpy.ma.masked_equal(grid, 0)
    colormap = matplotlib.colormaps["viridis"]
    white_zeros = colormap.with_extremes(bad="white")
    norm = matplotlib.colors.LogNorm()
    mesh = axis.pcolormesh(
        xs, ys, masked, shading="nearest", cmap=white_zeros, norm=norm
    )
    figure = axis.figure
    figure.colorbar(mesh, ax=axis, label=label)
    axis.set_xlabel(x)
    axis.set_ylabel(y)


def share_below(values: list, counts: list, thresholds: list) -> numpy.ndarray:
    """The share of the counted values strictly below each threshold."""
    value_array = numpy.asarray(values)
    count_array = numpy.asarray(counts)
    order = numpy.argsort(value_array)
    sorted_values = value_array[order]
    sorted_counts = count_array[order]
    running = numpy.cumsum(sorted_counts)
    cumulative = numpy.concatenate(([0], running))
    positions = numpy.searchsorted(sorted_values, thresholds, side="left")
    below = cumulative[positions]
    return below / cumulative[-1]


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

    sinter gives one colour per "color" value and one marker and line
    style per "marker" and "linestyle" value, and orders the curves by
    "sort" (sinter/_plotting.py:350-357).
    """
    curve_value = _value(stat, curve)
    label = str(curve_value)
    group = {
        "label": label,
        "color": curve_value,
        "marker": curve_value,
        "sort": curve_value,
    }
    if marker is None:
        return group
    marker_value = _value(stat, marker)
    group["label"] = f"{curve_value}, {marker_value}"
    group["marker"] = marker_value
    group["linestyle"] = marker_value
    group["sort"] = (curve_value, marker_value)
    return group


def _count_share(stat: sinter.TaskStats, hits: str, total: str) -> sinter.Fit:
    """The share of hits in total, the likeliest value and its band."""
    hit_count = stat.custom_counts[hits]
    total_count = stat.custom_counts[total]
    return sinter.fit_binomial(
        num_shots=total_count,
        num_hits=hit_count,
        max_likelihood_factor=LIKELIHOOD_FACTOR,
    )


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
    axis: pyplot.Axes,
    rows: list,
    x: str,
    y: str,
    low: Optional[str],
    high: Optional[str],
    label: str,
) -> None:
    """One curve of values, with error bars when low and high are named."""
    xs = [row[x] for row in rows]
    ys = numpy.array([row[y] for row in rows])
    if low is None:
        axis.plot(xs, ys, marker="o", label=label)
        return
    lows = numpy.array([row[low] for row in rows])
    highs = numpy.array([row[high] for row in rows])
    below = ys - lows
    above = highs - ys
    bars = numpy.array([below, above])
    axis.errorbar(xs, ys, yerr=bars, marker="o", capsize=3, label=label)


def _seen_values(rows: list, value: str, count: Optional[str]) -> tuple:
    """A curve's values and how many times each was seen."""
    seen = numpy.array([row[value] for row in rows])
    if count is None:
        return seen, numpy.ones(len(seen))
    counts = numpy.array([row[count] for row in rows])
    return seen, counts


def _median(seen_values: numpy.ndarray, counts: numpy.ndarray) -> float:
    """The lower median of values seen counts times each."""
    order = numpy.argsort(seen_values)
    sorted_values = seen_values[order]
    sorted_counts = counts[order]
    cumulative = numpy.cumsum(sorted_counts)
    half = cumulative[-1] / 2
    position = numpy.searchsorted(cumulative, half, side="left")
    return sorted_values[position]


def _log_axes(axis: pyplot.Axes) -> None:
    """A rate spans decades on both axes."""
    axis.set_xscale("log")
    axis.set_yscale("log")


def _style(
    axis: pyplot.Axes, x_label: str, y_label: str, legend_title: str
) -> None:
    """The labels, both grids and the legend every kind's axis carries."""
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(which="major")
    axis.grid(which="minor", alpha=MINOR_GRID_ALPHA)
    axis.legend(title=legend_title)
