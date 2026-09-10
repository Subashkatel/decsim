"""The LER-vs-distance figure: both tiers at one p from their sweep.csv.

The figure's contract: measured points carry Wilson bars, a
zero-failure point is left off (a log axis cannot hold zero, so the
curve ends at the last distance that saw failures), and a run that
never swept the requested p is refused rather than silently dropped.

The two folders here are `decsim collect` folders written by hand: the
columns are the first ten of sweep.csv, in the order report.summarize
writes them, which is every column the figure reads. Writing them
rather than running the sweep is what lets one tier hold a
zero-failure point at d 5.
"""

import csv

import pytest

import decsim.front.refusal as refusal
from decsim.front.plots import ler_vs_distance_plot

# The first ten columns of sweep.csv (decsim/front/report.py
# summarize_point), which is all the figure reads.
SWEEP_FIELDS = [
    "distance",
    "physical_error_probability",
    "algorithm",
    "round_period_us",
    "shots",
    "windows_per_shot",
    "logical_failures",
    "logical_error_rate",
    "ler_wilson_low",
    "ler_wilson_high",
]


def write_sweep_csv(run_dir, algorithm, points):
    """points: (distance, p, shots, failures, rate, low, high) rows."""
    run_dir.mkdir()
    sweep_path = run_dir / "sweep.csv"
    with open(sweep_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_FIELDS)
        writer.writeheader()
        for distance, p, shots, failures, rate, low, high in points:
            writer.writerow(
                {
                    "distance": distance,
                    "physical_error_probability": p,
                    "algorithm": algorithm,
                    "round_period_us": 1.0,
                    "shots": shots,
                    "windows_per_shot": distance,
                    "logical_failures": failures,
                    "logical_error_rate": rate,
                    "ler_wilson_low": low,
                    "ler_wilson_high": high,
                }
            )
    return run_dir


def two_tier_runs(tmp_path):
    weak_run_dir = tmp_path / "weak"
    weak_points = [
        (3, 0.001, 1000, 7, 7e-3, 3e-3, 1.4e-2),
        (5, 0.001, 1000, 1, 1e-3, 2e-4, 6e-3),
    ]
    weak = write_sweep_csv(weak_run_dir, "0.028", weak_points)
    strong_run_dir = tmp_path / "strong"
    strong_points = [
        (3, 0.001, 1000, 6, 6e-3, 2e-3, 1.2e-2),
        (5, 0.001, 1000, 0, 0.0, 0.0, 3.8e-3),
    ]
    strong = write_sweep_csv(strong_run_dir, "belief_matching", strong_points)
    return weak, strong


def test_figure_written_with_zero_failure_point_left_off(tmp_path):
    weak, strong = two_tier_runs(tmp_path)
    figure_path = tmp_path / "ler_vs_d.png"
    ler_vs_distance_plot([weak, strong], 0.001, figure_path)
    assert figure_path.exists()


def test_run_without_the_requested_p_is_refused(tmp_path):
    weak, strong = two_tier_runs(tmp_path)
    figure_path = tmp_path / "ler_vs_d.png"
    with pytest.raises(ValueError, match="p=0.002"):
        ler_vs_distance_plot([weak, strong], 0.002, figure_path)


def test_a_run_without_sweep_csv_is_refused(tmp_path):
    empty_run = tmp_path / "no_sweep"
    empty_run.mkdir()
    figure_path = tmp_path / "ler_vs_d.png"
    with pytest.raises(refusal.RefusalError, match="sweep.csv"):
        ler_vs_distance_plot([empty_run], 0.001, figure_path)
