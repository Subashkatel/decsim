"""decsim.plots: each kind draws what its rows hold, sinter's rates beside it.

The rate kinds are checked against sinter's own arithmetic on the same
counts (sinter.fit_binomial and shot_error_rate_to_piece_error_rate,
sinter/_probability_util.py), the rest against numbers worked by hand.
"""

import collections

import matplotlib.pyplot as pyplot
import pytest
import sinter

import decsim.plots as plots


def test_panels_title_one_axis_per_value_and_hide_the_rest():
    figure, axis_by_value = plots.panels("d", [5, 7, 9, 11], columns=3)

    titles = [axis.get_title() for axis in axis_by_value.values()]
    hidden = [axis for axis in figure.axes if not axis.get_visible()]
    assert titles == ["d = 5", "d = 7", "d = 9", "d = 11"]
    assert len(figure.axes) == 6
    assert len(hidden) == 2
    pyplot.close(figure)


def test_panel_grid_titles_one_axis_per_row_and_column_value():
    figure, axis_by_pair = plots.panel_grid(
        "strength", ["weak", "strong"], "p", [0.001, 0.003, 0.005]
    )

    corner_axis = axis_by_pair[("strong", 0.005)]
    assert len(axis_by_pair) == 6
    assert len(figure.axes) == 6
    assert corner_axis.get_title() == "strength = strong, p = 0.005"
    pyplot.close(figure)


def test_chosen_reads_a_stat_decoder_its_metadata_and_a_row_column():
    kept_stat = stat("pymatching", basis="x", d=5)
    other_stat = stat("pymatching", basis="z", d=5)
    kept_row = {"decoder": "pymatching", "basis": "x"}
    other_row = {"decoder": "union-find", "basis": "x"}
    items = [kept_stat, other_stat, kept_row, other_row]

    kept = plots.chosen(items, decoder="pymatching", basis="x")

    assert kept == [kept_stat, kept_row]


def test_error_rate_draws_sinter_per_round_rate_and_band_per_curve():
    stats = [
        stat("pymatching", d=5, p=0.001, errors=10),
        stat("pymatching", d=5, p=0.002, errors=40),
        stat("pymatching", d=7, p=0.001, errors=5),
        stat("pymatching", d=7, p=0.002, errors=20),
    ]
    figure, axis = pyplot.subplots()

    plots.error_rate(axis, stats, x="p", curve="d", rounds=10)

    first_line = axis.lines[0]
    first_curve = first_line.get_ydata()
    expected = sinter.shot_error_rate_to_piece_error_rate(0.01, pieces=10)
    assert len(axis.lines) == 2
    assert len(axis.collections) == 2
    assert first_curve[0] == pytest.approx(expected, rel=1e-12)
    assert axis.get_ylabel() == "logical error rate per round"
    assert axis.get_xscale() == "log"
    pyplot.close(figure)


def test_error_rate_styles_a_curve_by_its_place_in_order():
    """The panel lacks d = 5, and d = 7 keeps its second style."""
    stats = [
        stat("pymatching", d=7, p=0.001, errors=5),
        stat("pymatching", d=7, p=0.002, errors=20),
    ]
    figure, axis = pyplot.subplots()

    plots.error_rate(axis, stats, x="p", curve="d", order=[5, 7])

    (line,) = axis.lines
    assert line.get_color() == "C1"
    assert line.get_marker() == "v"
    pyplot.close(figure)


def test_error_rate_draws_nothing_for_a_point_with_no_errors():
    stats = [
        stat("pymatching", d=5, p=0.001, errors=0),
        stat("pymatching", d=5, p=0.002, errors=10),
        stat("pymatching", d=5, p=0.003, errors=30),
    ]
    figure, axis = pyplot.subplots()

    plots.error_rate(axis, stats, x="p", curve="d")

    (line,) = axis.lines
    band = axis.collections[0]
    (band_path,) = band.get_paths()
    band_xs = band_path.vertices[:, 0]
    drawn_rates = line.get_xdata()
    assert list(drawn_rates) == [0.002, 0.003]
    assert min(band_xs) == 0.002
    pyplot.close(figure)


def test_count_rate_draws_the_share_of_two_custom_counts():
    stats = [
        stat("union-find", p=0.001, counts={"escalated": 3, "windows": 60}),
        stat("union-find", p=0.002, counts={"escalated": 12, "windows": 60}),
    ]
    figure, axis = pyplot.subplots()

    plots.count_rate(
        axis, stats, x="p", hits="escalated", total="windows", curve="decoder"
    )

    line = axis.lines[0]
    shares = line.get_ydata()
    assert list(shares) == pytest.approx([0.05, 0.2], rel=1e-9)
    assert len(axis.collections) == 1
    assert axis.get_ylabel() == "escalated / windows"
    pyplot.close(figure)


def test_count_rate_draws_a_share_on_linear_axes_from_zero_to_one():
    stats = [
        stat("union-find", d=5, counts={"caught": 150, "bursts": 200}),
        stat("union-find", d=7, counts={"caught": 190, "bursts": 200}),
    ]
    figure, axis = pyplot.subplots()

    plots.count_rate(
        axis,
        stats,
        x="d",
        hits="caught",
        total="bursts",
        curve="decoder",
        is_x_log_scale=False,
        is_y_log_scale=False,
    )

    assert axis.get_xscale() == "linear"
    assert axis.get_yscale() == "linear"
    assert axis.get_ylim() == (0.0, 1.0)
    pyplot.close(figure)


def test_values_draw_error_bars_from_the_low_and_high_columns():
    rows = [
        {"d": 5, "p": 0.001, "rate": 0.2, "low": 0.1, "high": 0.4},
        {"d": 5, "p": 0.002, "rate": 0.3, "low": 0.25, "high": 0.5},
        {"d": 7, "p": 0.001, "rate": 0.1, "low": 0.05, "high": 0.2},
    ]
    figure, axis = pyplot.subplots()

    plots.values(axis, rows, x="p", y="rate", curve="d", low="low", high="high")

    first_bars = axis.containers[0]
    segments = first_bars.lines[2][0].get_segments()
    assert len(axis.containers) == 2
    assert segments[0][:, 1] == pytest.approx([0.1, 0.4])
    pyplot.close(figure)


def test_distribution_dashes_each_curve_at_its_count_weighted_median():
    rows = [
        {"decoder": "a", "microseconds": 1.0, "windows": 1},
        {"decoder": "a", "microseconds": 2.0, "windows": 1},
        {"decoder": "a", "microseconds": 3.0, "windows": 5},
        {"decoder": "b", "microseconds": 1.0, "windows": 4},
        {"decoder": "b", "microseconds": 4.0, "windows": 1},
    ]
    figure, axis = pyplot.subplots()

    plots.distribution(
        axis, rows, value="microseconds", curve="decoder", count="windows"
    )

    first_median, second_median = axis.lines
    first_histogram = axis.patches[0]
    outline = first_histogram.get_xy()
    heights = outline[:, 1]
    most_seen_share = 5 / 7
    assert first_median.get_xdata() == [3.0, 3.0]
    assert second_median.get_xdata() == [1.0, 1.0]
    assert len(axis.patches) == 2
    assert max(heights) == pytest.approx(most_seen_share)
    pyplot.close(figure)


def test_heatmap_leaves_a_zero_cell_blank_and_labels_its_colour_bar():
    rows = [
        {"d": 5, "p": 0.001, "count": 3},
        {"d": 5, "p": 0.002, "count": 0},
        {"d": 7, "p": 0.001, "count": 30},
        {"d": 7, "p": 0.002, "count": 300},
    ]
    figure, axis = pyplot.subplots()

    plots.heatmap(axis, rows, x="p", y="d", value="count", label="windows")

    mesh = axis.collections[0]
    grid = mesh.get_array()
    colour_bar_axis = figure.axes[1]
    assert grid.mask.tolist() == [[False, True], [False, False]]
    assert colour_bar_axis.get_ylabel() == "windows"
    pyplot.close(figure)


def test_a_linear_heatmap_colours_a_zero_and_leaves_a_missing_cell_blank():
    rows = [
        {"d": 5, "p": 0.001, "share": 0.0},
        {"d": 5, "p": 0.002, "share": 0.5},
        {"d": 7, "p": 0.001, "share": float("nan")},
        {"d": 7, "p": 0.002, "share": 1.0},
    ]
    figure, axis = pyplot.subplots()

    plots.heatmap(
        axis,
        rows,
        x="p",
        y="d",
        value="share",
        label="share",
        is_log_scale=False,
    )

    mesh = axis.collections[0]
    grid = mesh.get_array()
    norm = mesh.norm
    assert grid.mask.tolist() == [[False, False], [True, False]]
    assert (norm.vmin, norm.vmax) == (0.0, 1.0)
    pyplot.close(figure)


def test_heatmap_limits_fix_the_colour_scale_past_the_values_drawn():
    rows = [
        {"d": 5, "p": 0.001, "share": 0.2},
        {"d": 5, "p": 0.002, "share": 0.5},
    ]
    figure, axis = pyplot.subplots()

    plots.heatmap(
        axis,
        rows,
        x="p",
        y="d",
        value="share",
        label="share",
        is_log_scale=False,
        limits=(0.0, 1.0),
    )

    norm = axis.collections[0].norm
    assert (norm.vmin, norm.vmax) == (0.0, 1.0)
    pyplot.close(figure)


def test_share_below_counts_each_value_strictly_below_a_threshold():
    shares = plots.share_below([3, 1, 2], [1, 2, 3], [1, 2, 2.5, 4])

    assert shares.tolist() == [0.0, 2 / 6, 5 / 6, 1.0]


def test_save_writes_a_png_and_closes_the_figure(tmp_path):
    figure, axis = pyplot.subplots()
    axis.plot([1, 2], [1, 2])
    path = tmp_path / "line.png"

    plots.save(figure, path)

    written = path.read_bytes()
    header = written[:8]
    assert header == b"\x89PNG\r\n\x1a\n"
    assert not pyplot.fignum_exists(figure.number)


def stat(decoder, errors=1, counts=None, **metadata) -> sinter.TaskStats:
    """One sinter row of 1000 shots with the given metadata."""
    custom_counts = collections.Counter(counts)
    return sinter.TaskStats(
        strong_id=f"{decoder} {metadata} {errors}",
        decoder=decoder,
        json_metadata=metadata,
        shots=1000,
        errors=errors,
        custom_counts=custom_counts,
    )
