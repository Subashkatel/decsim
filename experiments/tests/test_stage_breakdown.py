"""The stage-breakdown figure: one stacked bar per distance from shots.csv.

The figure's contract: each stage's width is the MEDIAN over shots of
that shot's per-window mean (one slow shot cannot drag a bar the way a
mean of means would), stages stack in pipeline order, and a run
without a shots.csv is refused rather than silently skipped.
"""

import csv

import pytest

from experiments.plots import (
    STAGE_BREAKDOWN_STAGES,
    _median_stage_us_by_distance,
    stage_breakdown_plot,
)


def write_shots_csv(run_dir, rows):
    """rows: (distance, {stage column: us}) pairs; other columns filled."""
    run_dir.mkdir()
    stage_columns = [column for column, _ in STAGE_BREAKDOWN_STAGES]
    field_names = ["distance", "algorithm"] + stage_columns
    with open(run_dir / "shots.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for distance, stage_us in rows:
            row = {"distance": distance, "algorithm": "pymatching"}
            for column in stage_columns:
                row[column] = stage_us.get(column, 0.0)
            writer.writerow(row)


def flat_stages(algorithm_us):
    return {"algorithm_mean_us": algorithm_us}


def test_stage_widths_are_medians_over_shots(tmp_path):
    run_dir = tmp_path / "run"
    write_shots_csv(run_dir, [
        (3, flat_stages(10.0)),
        (3, flat_stages(20.0)),
        (3, flat_stages(1000.0)),
        (5, flat_stages(40.0)),
    ])
    medians = _median_stage_us_by_distance(
        list(csv.DictReader(open(run_dir / "shots.csv"))))
    algorithm_index = [column for column, _ in STAGE_BREAKDOWN_STAGES
                      ].index("algorithm_mean_us")
    # the 1000 us outlier shot moves a mean to 343 but the median to 20
    assert medians[3][algorithm_index] == pytest.approx(20.0)
    assert medians[5][algorithm_index] == pytest.approx(40.0)


def test_figure_is_written_per_distance(tmp_path):
    run_dir = tmp_path / "run"
    write_shots_csv(run_dir, [
        (3, flat_stages(12.0)),
        (5, flat_stages(25.0)),
    ])
    figure_path = tmp_path / "stage_breakdown.png"
    stage_breakdown_plot(run_dir, figure_path)
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0


def test_a_run_without_shots_csv_is_refused(tmp_path):
    empty_run = tmp_path / "empty"
    empty_run.mkdir()
    with pytest.raises(FileNotFoundError, match="shots.csv"):
        stage_breakdown_plot(empty_run, tmp_path / "figure.png")
