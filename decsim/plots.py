"""The figure kinds an experiment's plot.py draws, styled in one place.

Rates follow sinter: a binomial rate is drawn with sinter.plot_custom,
its line through the most likely rate and its band over every rate
within a likelihood factor of it (sinter/_plotting.py:317-419). The
other kinds read plain rows, one dict per row, as decsim.results.load
returns them. A key names a column of a row; on a sinter.TaskStats it
names a json_metadata field, except "decoder", which is the stat's own.

A curve's colour and marker come from its value's place in `order`, the
list of every value the curve key takes across a figure, so a value
keeps its style in a panel that lacks another; without `order` the
place is among the values the call draws, sorted.
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
import matplotlib.ticker
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
    figure, axes = _panel_figure(row_count, column_count)
    flat_axes = list(axes.flat)
    shown_axes = flat_axes[:panel_count]
    axis_by_value = {}
    for axis, value in zip(shown_axes, values, strict=True):
        axis.set_title(f"{key} = {value}")
        axis_by_value[value] = axis
    for axis in flat_axes[panel_count:]:
        axis.set_visible(False)
    return figure, axis_by_value


def panel_grid(
    row_key: str, row_values: list, column_key: str, column_values: list
) -> tuple:
    """A figure with one axis per row value and column value, axes shared.

    Each axis is titled "row_key = row, column_key = column". Returns the
    figure and a dict from each (row value, column value) to its axis.
    """
    row_count = len(row_values)
    column_count = len(column_values)
    figure, axes = _panel_figure(row_count, column_count)
    axis_by_pair = {}
    for row_index, row_value in enumerate(row_values):
        for column_index, column_value in enumerate(column_values):
            axis = axes[row_index, column_index]
            axis.set_title(
                f"{row_key} = {row_value}, {column_key} = {column_value}"
            )
            axis_by_pair[(row_value, column_value)] = axis
    return figure, axis_by_pair


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

    A point with no errors is left out: it has no likeliest rate, only a
    bound, and sinter draws it as a band from zero with no marker
    (sinter/_plotting.py:399-400), which on a log axis runs to the floor.
    """
    x_of = functools.partial(_value, key=x)
    group_of = functools.partial(_group, curve=curve, marker=marker)
    style_of = _sinter_style(stats, curve, marker, order)
    sinter.plot_error_rate(
        ax=axis,
        stats=stats,
        x_func=x_of,
        failure_units_per_shot_func=lambda _stat: rounds,
        group_func=group_of,
        filter_func=_has_errors,
        plot_args_func=style_of,
    )
    unit = "shot"
    if rounds > 1:
        unit = "round"
    legend_title = curve
    if marker is not None:
        legend_title = f"{curve}, {marker}"
    _scale_axes(axis, is_x_log_scale=True, is_y_log_scale=True)
    _style(axis, x, f"logical error rate per {unit}", legend_title)


def count_rate(
    axis: pyplot.Axes,
    stats: list,
    x: str,
    hits: str,
    total: str,
    curve: str,
    order: Optional[list] = None,
    is_x_log_scale: bool = True,
    is_y_log_scale: bool = True,
) -> None:
    """The share hits of total against x, both custom counts, with a band.

    The band is the one error_rate draws, from sinter.fit_binomial over
    the two counts (sinter/_probability_util.py:327). A share spans
    decades when it is rare, so both axes are log by default; a linear
    share axis runs from 0 to 1, the whole range a share can take.
    """
    x_of = functools.partial(_value, key=x)
    share_of = functools.partial(_count_share, hits=hits, total=total)
    group_of = functools.partial(_group, curve=curve, marker=None)
    style_of = _sinter_style(stats, curve, None, order)
    sinter.plot_custom(
        ax=axis,
        stats=stats,
        x_func=x_of,
        y_func=share_of,
        group_func=group_of,
        plot_args_func=style_of,
    )
    _scale_axes(axis, is_x_log_scale, is_y_log_scale)
    if not is_y_log_scale:
        axis.set_ylim(0.0, 1.0)
    _style(axis, x, f"{hits} / {total}", curve)


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


def distribution(
    axis: pyplot.Axes,
    rows: list,
    value: str,
    curve: str,
    count: Optional[str] = None,
    order: Optional[list] = None,
) -> None:
    """A histogram of value per curve, each summing to one, its median dashed.

    count names a column holding how many times its row's value was
    seen; without it every row is seen once.
    """
    all_values = [row[value] for row in rows]
    edges = numpy.histogram_bin_edges(all_values, bins=HISTOGRAM_BINS)
    curves = _curves(rows, curve, value)
    for curve_value, curve_rows in curves.items():
        seen_values, counts = _seen_values(curve_rows, value, count)
        weights = counts / counts.sum()
        style = _curve_style(curve_value, curves, order)
        color = style["color"]
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
    axis: pyplot.Axes,
    rows: list,
    x: str,
    y: str,
    value: str,
    label: str,
    is_log_scale: bool = True,
    limits: Optional[tuple] = None,
) -> None:
    """The value column on an x by y grid, cells no row fills white.

    On the log scale a zero cell is white too; the linear scale, for a
    share, colours a zero like any value, and a NaN value is white.
    limits, (low, high), fixes the colour scale, so panels drawn apart
    give one value one colour; None spans the cells drawn.
    """
    xs = sorted({row[x] for row in rows})
    ys = sorted({row[y] for row in rows})
    grid = numpy.full((len(ys), len(xs)), numpy.nan)
    for row in rows:
        column = xs.index(row[x])
        line = ys.index(row[y])
        grid[line, column] = row[value]
    masked = numpy.ma.masked_invalid(grid)
    norm = matplotlib.colors.Normalize()
    if is_log_scale:
        masked = numpy.ma.masked_equal(masked, 0)
        norm = matplotlib.colors.LogNorm()
    if limits is not None:
        norm.vmin, norm.vmax = limits
    colormap = matplotlib.colormaps["viridis"]
    white_blanks = colormap.with_extremes(bad="white")
    mesh = axis.pcolormesh(
        xs, ys, masked, shading="nearest", cmap=white_blanks, norm=norm
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

    sinter orders the curves by "sort" and gives one marker and line
    style per "marker" and "linestyle" value (sinter/_plotting.py:350-357);
    the colour, and the marker when no second key is given, come from
    _sinter_style instead, since sinter ranks them among the curves of
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


def _has_errors(stat: sinter.TaskStats) -> bool:
    """Whether a point saw an error, so it has a likeliest rate."""
    return stat.errors > 0


def _sinter_style(
    stats: list, curve: str, marker: Optional[str], order: Optional[list]
) -> functools.partial:
    """The plot_args_func sinter calls: a curve's colour and marker by place."""
    curve_values = {_value(stat, curve) for stat in stats}
    places = order
    if places is None:
        places = sorted(curve_values)
    return functools.partial(
        _place_style, curve=curve, marker=marker, places=places
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


def _panel_figure(row_count: int, column_count: int) -> tuple:
    """A figure of row_count by column_count axes, shared, panel sized."""
    width = PANEL_WIDTH_INCHES * column_count
    height = PANEL_HEIGHT_INCHES * row_count
    return pyplot.subplots(
        row_count,
        column_count,
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=(width, height),
        layout="constrained",
    )


def _scale_axes(
    axis: pyplot.Axes, is_x_log_scale: bool, is_y_log_scale: bool
) -> None:
    """A rate spans decades, so a log axis is labelled at decades only.

    Matplotlib labels minor ticks on an axis that spans few decades
    (LogFormatter's minor_thresholds), and at 3 and 4 times a decade
    those labels run into each other.
    """
    no_labels = matplotlib.ticker.NullFormatter()
    if is_x_log_scale:
        axis.set_xscale("log")
        axis.xaxis.set_minor_formatter(no_labels)
    if is_y_log_scale:
        axis.set_yscale("log")
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
