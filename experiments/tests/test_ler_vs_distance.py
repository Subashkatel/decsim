"""The LER-vs-distance figure: both tiers at one p from their ler.csv.

The figure's contract: measured points carry Wilson bars, a
zero-failure point is left off (a log axis cannot hold zero, so the
curve ends at the last distance that saw failures), and a run that
never swept the requested p is refused rather than silently dropped.
"""

import csv

import pytest

from experiments.plots import ler_vs_distance_plot

LER_FIELDS = ["distance", "physical_error_probability", "algorithm",
              "shots", "failures", "logical_error_rate",
              "ler_per_d_rounds", "ler_wilson_low", "ler_wilson_high",
              "wall_seconds_per_shot", "round_period_us"]


def write_ler_csv(run_dir, algorithm, points):
    """points: (distance, p, shots, failures, rate, low, high) rows."""
    run_dir.mkdir()
    with open(run_dir / "ler.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LER_FIELDS)
        writer.writeheader()
        for distance, p, shots, failures, rate, low, high in points:
            writer.writerow({
                "distance": distance, "physical_error_probability": p,
                "algorithm": algorithm, "shots": shots,
                "failures": failures, "logical_error_rate": rate,
                "ler_per_d_rounds": rate / 10, "ler_wilson_low": low,
                "ler_wilson_high": high, "wall_seconds_per_shot": 0.01,
                "round_period_us": 0.0})
    return run_dir


def two_tier_runs(tmp_path):
    weak = write_ler_csv(tmp_path / "weak", "0.028", [
        (3, 0.001, 1000, 7, 7e-3, 3e-3, 1.4e-2),
        (5, 0.001, 1000, 1, 1e-3, 2e-4, 6e-3)])
    strong = write_ler_csv(tmp_path / "strong", "belief_matching", [
        (3, 0.001, 1000, 6, 6e-3, 2e-3, 1.2e-2),
        (5, 0.001, 1000, 0, 0.0, 0.0, 3.8e-3)])
    return weak, strong


def test_figure_written_with_zero_failure_point_left_off(tmp_path):
    weak, strong = two_tier_runs(tmp_path)
    figure_path = tmp_path / "ler_vs_d.png"
    ler_vs_distance_plot([weak, strong], 0.001, figure_path)
    assert figure_path.exists()


def test_run_without_the_requested_p_is_refused(tmp_path):
    weak, strong = two_tier_runs(tmp_path)
    with pytest.raises(ValueError, match="p=0.002"):
        ler_vs_distance_plot([weak, strong], 0.002,
                             tmp_path / "ler_vs_d.png")
