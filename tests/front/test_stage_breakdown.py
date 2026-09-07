"""The stage-breakdown figure: one stacked bar per distance from shots.csv.

The figure's contract: each stage's width is the MEDIAN over shots of
that shot's per-window mean (one slow shot cannot drag a bar the way a
mean of means would), stages stack in pipeline order, and a run
without a shots.csv is refused rather than silently skipped.
"""

import csv

import pytest

import decsim.front.refusal as refusal
from decsim.front.plots import (
    STAGE_BREAKDOWN_STAGES,
    _median_stage_us_by_distance,
    stage_breakdown_plot,
)


def shots_csv_row(distance, stage_us, stage_columns):
    row = {"distance": distance, "algorithm": "pymatching"}
    for column in stage_columns:
        row[column] = stage_us.get(column, 0.0)
    return row


def write_shots_csv(run_dir, rows):
    """rows: (distance, {stage column: us}) pairs; other columns filled."""
    run_dir.mkdir()
    stage_columns = [column for column, _ in STAGE_BREAKDOWN_STAGES]
    field_names = ["distance", "algorithm"] + stage_columns
    shots_csv_path = run_dir / "shots.csv"
    with open(shots_csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for distance, stage_us in rows:
            row = shots_csv_row(distance, stage_us, stage_columns)
            writer.writerow(row)


def flat_stages(algorithm_us):
    return {"algorithm_mean_us": algorithm_us}


def test_stage_widths_are_medians_over_shots(tmp_path):
    run_dir = tmp_path / "run"
    shot_rows = [
        (3, flat_stages(10.0)),
        (3, flat_stages(20.0)),
        (3, flat_stages(1000.0)),
        (5, flat_stages(40.0)),
    ]
    write_shots_csv(run_dir, shot_rows)
    shots_csv_path = run_dir / "shots.csv"
    with open(shots_csv_path) as handle:
        reader = csv.DictReader(handle)
        shot_records = list(reader)
    medians = _median_stage_us_by_distance(shot_records)
    stage_columns = [column for column, _ in STAGE_BREAKDOWN_STAGES]
    algorithm_index = stage_columns.index("algorithm_mean_us")
    # the 1000 us outlier shot moves a mean to 343 but the median to 20
    assert medians[3][algorithm_index] == pytest.approx(20.0)
    assert medians[5][algorithm_index] == pytest.approx(40.0)


def test_figure_is_written_per_distance(tmp_path):
    run_dir = tmp_path / "run"
    shot_rows = [
        (3, flat_stages(12.0)),
        (5, flat_stages(25.0)),
    ]
    write_shots_csv(run_dir, shot_rows)
    figure_path = tmp_path / "stage_breakdown.png"
    stage_breakdown_plot(run_dir, figure_path)
    assert figure_path.exists()
    figure_status = figure_path.stat()
    assert figure_status.st_size > 0


def test_a_run_without_shots_csv_is_refused(tmp_path):
    empty_run = tmp_path / "empty"
    empty_run.mkdir()
    figure_path = tmp_path / "figure.png"
    with pytest.raises(refusal.RefusalError, match="shots.csv"):
        stage_breakdown_plot(empty_run, figure_path)
