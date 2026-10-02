"""decsim.plots: each kind draws what its rows hold, sinter's rates beside it.

The error rate is checked against sinter's own arithmetic on the same
counts (shot_error_rate_to_piece_error_rate, sinter/_probability_util.py),
the rest against numbers worked by hand.
"""

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


def test_error_rate_marks_a_point_with_no_errors_at_its_bound():
    stats = [
        stat("pymatching", d=5, p=0.001, errors=0),
        stat("pymatching", d=5, p=0.002, errors=10),
        stat("pymatching", d=5, p=0.003, errors=30),
    ]
    figure, axis = pyplot.subplots()

    plots.error_rate(axis, stats, x="p", curve="d", rounds=10)

    line, bound = axis.lines
    band = axis.collections[0]
    (band_path,) = band.get_paths()
    band_xs = band_path.vertices[:, 0]
    fit = sinter.fit_binomial(
        num_shots=1000, num_hits=0, max_likelihood_factor=1e3
    )
    expected = sinter.shot_error_rate_to_piece_error_rate(fit.high, pieces=10)
    line_xs = line.get_xdata()
    bound_xs = bound.get_xdata()
    bound_ys = bound.get_ydata()
    legend = axis.get_legend()
    legend_lines = legend.get_lines()
    assert list(line_xs) == [0.002, 0.003]
    assert min(band_xs) == 0.002
    assert list(bound_xs) == [0.001]
    assert bound_ys[0] == pytest.approx(expected, rel=1e-12)
    assert bound.get_markerfacecolor() == "none"
    assert bound.get_color() == line.get_color()
    assert len(legend_lines) == 1
    pyplot.close(figure)


def test_error_rate_leaves_a_shot_rate_above_half_out_of_a_per_round_figure():
    """Per round, sinter maps 600 errors in 1000 shots to about 0.99."""
    stats = [
        stat("pymatching", d=5, p=0.001, errors=100),
        stat("pymatching", d=5, p=0.002, errors=600),
    ]
    figure, axis = pyplot.subplots()

    plots.error_rate(axis, stats, x="p", curve="d", rounds=10)

    drawn_xs = {x for line in axis.lines for x in line.get_xdata()}
    assert drawn_xs == {0.001}
    pyplot.close(figure)


def test_error_rate_keeps_a_shot_rate_above_half_per_shot():
    stats = [stat("pymatching", d=5, p=0.002, errors=600)]
    figure, axis = pyplot.subplots()

    plots.error_rate(axis, stats, x="p", curve="d")

    curve_line = axis.lines[0]
    curve_rates = curve_line.get_ydata()
    assert list(curve_rates) == pytest.approx([0.6])
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


def test_bounds_draw_hollow_markers_at_the_high_column_by_place():
    rows = [
        {"d": 7, "p": 0.001, "high": 0.02},
        {"d": 7, "p": 0.002, "high": 0.03},
    ]
    figure, axis = pyplot.subplots()

    plots.bounds(axis, rows, x="p", high="high", curve="d", order=[5, 7])

    (bound,) = axis.lines
    bound_ys = bound.get_ydata()
    assert list(bound_ys) == [0.02, 0.03]
    assert bound.get_linestyle() == "None"
    assert bound.get_markerfacecolor() == "none"
    assert bound.get_color() == "C1"
    pyplot.close(figure)


def test_save_writes_a_png_and_closes_the_figure(tmp_path):
    figure, axis = pyplot.subplots()
    axis.plot([1, 2], [1, 2])
    path = tmp_path / "line.png"

    plots.save(figure, path)

    written = path.read_bytes()
    header = written[:8]
    assert header == b"\x89PNG\r\n\x1a\n"
    assert not pyplot.fignum_exists(figure.number)


def stat(decoder, errors=1, **metadata) -> sinter.TaskStats:
    """One sinter row of 1000 shots with the given metadata."""
    return sinter.TaskStats(
        strong_id=f"{decoder} {metadata} {errors}",
        decoder=decoder,
        json_metadata=metadata,
        shots=1000,
        errors=errors,
    )
