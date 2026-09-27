"""The stage-breakdown figure: one stacked bar per x value from shots.csv.

The figure's contract: each stage's width is the MEDIAN over shots of
that shot's per-window mean (one slow shot cannot drag a bar the way a
mean of means would), stages stack in pipeline order, and a run
without a shots.csv is refused rather than silently skipped.
"""

import csv
import json

import matplotlib.pyplot
import pytest

import decsim.experiments.plots as plots
import decsim.experiments.refusal as refusal

BY_DISTANCE = plots.Selection("qpu.distance", None, {})


def shots_csv_row(distance, stage_us, stage_columns):
    metadata = json.dumps({"qpu.distance": distance})
    row = {"metadata": metadata, "algorithm": "pymatching"}
    for column in stage_columns:
        row[column] = stage_us.get(column, 0.0)
    return row


def write_shots_csv(run_dir, rows):
    """rows: (distance, {stage column: us}) pairs; other columns filled."""
    run_dir.mkdir()
    stage_columns = [column for column, _ in plots.STAGE_BREAKDOWN_STAGES]
    field_names = ["metadata", "algorithm"] + stage_columns
    shots_csv_path = run_dir / "shots.csv"
    with open(shots_csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for distance, stage_us in rows:
            row = shots_csv_row(distance, stage_us, stage_columns)
            writer.writerow(row)


def flat_stages(algorithm_us):
    return {"algorithm_mean_us": algorithm_us}


def stage_widths(axis, stage_label) -> list:
    """One stage's segment width in ms, one per bar, in the bars' order."""
    (segments,) = [
        container
        for container in axis.containers
        if container.get_label() == stage_label
    ]
    return [segment.get_width() for segment in segments]


def test_stage_widths_are_medians_over_shots(tmp_path, monkeypatch):
    """The algorithm segment of each bar is that stage's median.

    The bars are d = 3 and d = 5. The 1000 us outlier shot would move a
    mean to 343 us; the median stays at 20 us, 0.020 ms.
    """
    run_dir = tmp_path / "run"
    shot_rows = [
        (3, flat_stages(10.0)),
        (3, flat_stages(20.0)),
        (3, flat_stages(1000.0)),
        (5, flat_stages(40.0)),
    ]
    write_shots_csv(run_dir, shot_rows)
    figure_path = tmp_path / "stage_breakdown.png"
    drawn = []
    monkeypatch.setattr(matplotlib.pyplot, "close", drawn.append)

    plots.stage_breakdown_plot(run_dir, BY_DISTANCE, figure_path)

    (figure,) = drawn
    (axis,) = figure.axes
    algorithm_widths = stage_widths(axis, "algorithm")
    assert algorithm_widths == pytest.approx([0.020, 0.040])


def test_figure_is_written_per_x_value(tmp_path):
    run_dir = tmp_path / "run"
    shot_rows = [
        (3, flat_stages(12.0)),
        (5, flat_stages(25.0)),
    ]
    write_shots_csv(run_dir, shot_rows)
    figure_path = tmp_path / "stage_breakdown.png"
    plots.stage_breakdown_plot(run_dir, BY_DISTANCE, figure_path)
    assert figure_path.exists()
    figure_status = figure_path.stat()
    assert figure_status.st_size > 0


def test_a_run_without_shots_csv_is_refused(tmp_path):
    empty_run = tmp_path / "empty"
    empty_run.mkdir()
    figure_path = tmp_path / "figure.png"
    with pytest.raises(refusal.RefusalError, match="shots.csv"):
        plots.stage_breakdown_plot(empty_run, BY_DISTANCE, figure_path)
