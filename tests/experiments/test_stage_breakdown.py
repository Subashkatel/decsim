"""The stage-breakdown figure: one stacked bar per sweep point.

The figure's contract: each stage's width is the MEDIAN over a point's
shots of that shot's per-window mean (one slow shot cannot drag a bar
the way a mean of means would), stages stack in pipeline order, no two
points' shots share a bar, and a run without a shots.csv is refused
rather than silently skipped.
"""

import csv
import json

import matplotlib.legend
import matplotlib.pyplot
import pytest

import decsim.experiments.plots as plots
import decsim.experiments.refusal as refusal


def shots_csv_row(point_id, stage_us, stage_columns, algorithm):
    row = {"point_id": point_id, "algorithm": algorithm}
    for column in stage_columns:
        row[column] = stage_us.get(column, 0.0)
    return row


def write_point_record(resolved_dir, point_id):
    """A resolved/ record whose metadata sets the distance its id ends in.

    The other two paths are the long ones a real sweep sets.
    """
    distance = int(point_id[-1])
    metadata = {
        "qpu.distance": distance,
        "qpu.round_period_microseconds": 1.0,
        "workload.arguments.physical_error_probability": 0.001,
    }
    sections = {
        "qpu": {"distance": distance, "round_period_microseconds": 1.0},
        "workload": {"arguments": {"physical_error_probability": 0.001}},
    }
    record = {"id": point_id, "metadata": metadata, "sections": sections}
    record_path = resolved_dir / f"{point_id}.json"
    record_text = json.dumps(record)
    record_path.write_text(record_text)


def write_run(run_dir, rows, algorithm="pymatching"):
    """rows: (point id, {stage column: us}) pairs, one per shot."""
    run_dir.mkdir()
    resolved_dir = run_dir / "resolved"
    resolved_dir.mkdir()
    stage_columns = [column for column, _ in plots.STAGE_BREAKDOWN_STAGES]
    field_names = ["point_id", "algorithm"] + stage_columns
    shots_csv_path = run_dir / "shots.csv"
    with open(shots_csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for point_id, stage_us in rows:
            row = shots_csv_row(point_id, stage_us, stage_columns, algorithm)
            writer.writerow(row)
            write_point_record(resolved_dir, point_id)


def flat_stages(algorithm_us):
    return {"algorithm_mean_us": algorithm_us}


def drawn_axis(run_dir, figure_path, monkeypatch):
    """The one axis stage_breakdown_plot drew, caught as it closes."""
    drawn = []
    monkeypatch.setattr(matplotlib.pyplot, "close", drawn.append)
    plots.stage_breakdown_plot(run_dir, figure_path)
    (figure,) = drawn
    (axis,) = figure.axes
    return axis


def stage_widths(axis, stage_label) -> list:
    """One stage's segment width in ms, one per bar, in the bars' order."""
    (segments,) = [
        container
        for container in axis.containers
        if container.get_label() == stage_label
    ]
    return [segment.get_width() for segment in segments]


def test_stage_widths_are_medians_over_one_points_shots(tmp_path, monkeypatch):
    """The algorithm segment of each bar is that point's median.

    The 1000 us outlier shot would move a mean to 343 us; the median
    stays at 20 us, 0.020 ms.
    """
    run_dir = tmp_path / "run"
    shot_rows = [
        ("point3", flat_stages(10.0)),
        ("point3", flat_stages(20.0)),
        ("point3", flat_stages(1000.0)),
        ("point5", flat_stages(40.0)),
    ]
    write_run(run_dir, shot_rows)
    figure_path = tmp_path / "stage_breakdown.png"

    axis = drawn_axis(run_dir, figure_path, monkeypatch)

    algorithm_widths = stage_widths(axis, "algorithm")
    assert algorithm_widths == pytest.approx([0.020, 0.040])


def test_a_bar_is_labelled_by_its_points_values_and_decoder(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    one_shot = ("point3", flat_stages(12.0))
    write_run(run_dir, [one_shot])
    figure_path = tmp_path / "stage_breakdown.png"

    axis = drawn_axis(run_dir, figure_path, monkeypatch)

    (tick_label,) = axis.get_yticklabels()
    assert tick_label.get_text() == (
        "qpu.distance=3\n"
        "qpu.round_period_microseconds=1.0\n"
        "workload.arguments.physical_error_probability=0.001\n"
        "algorithm pymatching"
    )


def test_a_bar_is_labelled_by_the_value_its_point_ran_with(
    tmp_path, monkeypatch
):
    """A swept reference reads as the value it resolved to, not its text."""
    run_dir = tmp_path / "run"
    one_shot = ("point3", flat_stages(12.0))
    write_run(run_dir, [one_shot])
    record = {
        "id": "point3",
        "metadata": {"qpu.distance": "${windows.commit_rounds}"},
        "sections": {"qpu": {"distance": 3}},
    }
    record_path = run_dir / "resolved" / "point3.json"
    record_text = json.dumps(record)
    record_path.write_text(record_text)
    figure_path = tmp_path / "stage_breakdown.png"

    axis = drawn_axis(run_dir, figure_path, monkeypatch)

    (tick_label,) = axis.get_yticklabels()
    assert tick_label.get_text() == "qpu.distance=3\nalgorithm pymatching"


def test_every_label_the_legend_and_the_title_fit_a_four_point_figure(
    tmp_path, monkeypatch
):
    """Nothing is cut at the figure's edge, and the key covers no bar.

    The decoder is a latency card, which the label names in its unit.
    """
    run_dir = tmp_path / "run"
    shot_rows = [
        ("point3", flat_stages(12.0)),
        ("point5", flat_stages(25.0)),
        ("point7", flat_stages(40.0)),
        ("point9", flat_stages(60.0)),
    ]
    write_run(run_dir, shot_rows, algorithm="0.028")
    figure_path = tmp_path / "stage_breakdown.png"

    axis = drawn_axis(run_dir, figure_path, monkeypatch)

    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    (legend,) = figure.findobj(matplotlib.legend.Legend)
    texts = [*axis.get_yticklabels(), axis.title, axis.xaxis.label, legend]
    boxes = [text.get_window_extent(renderer) for text in texts]
    axis_box = axis.get_window_extent(renderer)
    legend_box = legend.get_window_extent(renderer)
    tick_labels = axis.get_yticklabels()
    first_label = tick_labels[0]
    first_text = first_label.get_text()
    label_lines = first_text.splitlines()
    assert label_lines[-1] == "algorithm 0.028 us"
    assert all(_inside(box, figure.bbox) for box in boxes)
    assert not legend_box.overlaps(axis_box)


def test_figure_is_written_per_point(tmp_path):
    run_dir = tmp_path / "run"
    shot_rows = [
        ("point3", flat_stages(12.0)),
        ("point5", flat_stages(25.0)),
    ]
    write_run(run_dir, shot_rows)
    figure_path = tmp_path / "stage_breakdown.png"

    plots.stage_breakdown_plot(run_dir, figure_path)

    figure_status = figure_path.stat()
    assert figure_status.st_size > 0


def test_a_run_without_shots_csv_is_refused(tmp_path):
    empty_run = tmp_path / "empty"
    empty_run.mkdir()
    figure_path = tmp_path / "figure.png"
    with pytest.raises(refusal.RefusalError, match="shots.csv"):
        plots.stage_breakdown_plot(empty_run, figure_path)


def _inside(box, outer) -> bool:
    """Whether a drawn box lies within an outer one, edges included."""
    lower_left = outer.contains(box.x0, box.y0)
    upper_right = outer.contains(box.x1, box.y1)
    return lower_left and upper_right
