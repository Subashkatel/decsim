"""Run one experiment: python -m experiments.run configs/<name>.yaml

The config is the experiment; this module only orchestrates. It runs every
shot of every sweep point, summarizes one row per point, and writes
sweep.csv, sweep.md and the figures to experiments/results/<name>/.
Rerunning the same config reproduces the same rows (seeds 0..shots-1 per
point; only the wall-clock column varies).
"""

from __future__ import annotations

import sys
from pathlib import Path

from experiments.experiment_config import ExperimentConfig, load_experiment
from experiments.measure_shot import measure_shot
from experiments.sweep_report import summarize, write_report

CONFIGS_DIR = Path(__file__).parent / "configs"


def run_sweep(config: ExperimentConfig) -> list:
    """Every shot of every point of the config's sweep blocks; a point and
    seed named by more than one block runs once."""
    measurements = {}
    for block in config.sweep:
        for physical_error_probability in block.physical_error_probabilities:
            for algorithm_latency_us in block.algorithm_latencies_us:
                for round_period_us in block.round_periods_us:
                    for seed in range(block.shots):
                        shot_key = (physical_error_probability, algorithm_latency_us,
                                    round_period_us, seed)
                        if shot_key in measurements:
                            continue
                        measurements[shot_key] = measure_shot(
                            config,
                            physical_error_probability=physical_error_probability,
                            round_period_us=round_period_us,
                            algorithm_latency_us=algorithm_latency_us, seed=seed)
                    print(f"p {physical_error_probability}, "
                          f"algorithm {algorithm_latency_us} us, "
                          f"round period {round_period_us} us: "
                          f"{block.shots} shots done", file=sys.stderr)
    return list(measurements.values())


def run_experiment(config_path) -> Path:
    """One full experiment: sweep, summary, report, figures. Returns the
    results folder."""
    config = load_experiment(config_path)
    rows = summarize(run_sweep(config))
    results_dir = config.results_dir
    write_report(rows, results_dir)
    from experiments.plots import plots
    plots(rows, results_dir)
    return results_dir


def main(argv) -> None:
    if len(argv) != 2:
        names = sorted(path.stem for path in CONFIGS_DIR.glob("*.yaml"))
        print("usage: python -m experiments.run configs/<name>.yaml\n"
              f"configs: {', '.join(names)}", file=sys.stderr)
        raise SystemExit(2)
    results_dir = run_experiment(argv[1])
    print((results_dir / "sweep.md").read_text())


if __name__ == "__main__":
    main(sys.argv)
