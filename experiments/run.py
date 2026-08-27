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

from experiments.experiment_config import (MODE_TIER, ExperimentConfig,
                                           load_experiment)
from experiments.measure_shot import measure_shot
from experiments.sweep_report import summarize, terminal_lines, write_report

CONFIGS_DIR = Path(__file__).parent / "configs"


def resolved_description(config: ExperimentConfig) -> list:
    """What this run will actually do, echoed before the first shot.

    An edit that did not land shows up here immediately: a config that
    extends another replaces its base's keys whole, so a `sweep` edited in
    the base never reaches a child that declares its own.
    """
    files = " <- ".join(str(path) for path in config.config_files)
    unit = config.active_decoder
    lines = [f"config: {files}",
             f"mode: {config.mode}, {config.code_task}, distance "
             f"{config.distance}, {config.rounds_per_shot} rounds per shot",
             f"windows: {config.windowing.scheme}",
             f"decoder: the {MODE_TIER[config.mode]} tier, algorithm "
             f"{unit.algorithm}, {unit.units} unit(s), "
             f"engine clock {unit.engine.clock}"]
    for index, block in enumerate(config.sweep, start=1):
        lines.append(
            f"sweep block {index}: "
            f"p {list(block.physical_error_probabilities)}, "
            f"round period {list(block.round_periods_us)} us, "
            f"{block.shots} shots")
    lines.append(f"trace: {config.trace}"
                 + (" with component I/O" if config.trace_io else ""))
    return lines


def run_sweep(config: ExperimentConfig) -> list:
    """Every shot of every point of the config's sweep blocks; a point and
    seed named by more than one block runs once."""
    measurements = {}
    for block in config.sweep:
        for physical_error_probability in block.physical_error_probabilities:
            for round_period_us in block.round_periods_us:
                for seed in range(block.shots):
                    shot_key = (physical_error_probability,
                                round_period_us, seed)
                    if shot_key in measurements:
                        continue
                    measurements[shot_key] = measure_shot(
                        config,
                        physical_error_probability=physical_error_probability,
                        round_period_us=round_period_us, seed=seed)
                print(f"p {physical_error_probability}, "
                      f"round period {round_period_us} us: "
                      f"{block.shots} shots done", file=sys.stderr)
    return list(measurements.values())


def run_experiment(config_path) -> tuple:
    """One full experiment: sweep, summary, report, figures. Returns the
    results folder and the summary rows."""
    config = load_experiment(config_path)
    print("\n".join(resolved_description(config)) + "\n", file=sys.stderr)
    measurements = run_sweep(config)
    rows = summarize(measurements)
    results_dir = config.results_dir
    write_report(rows, results_dir, measurements)
    from experiments.plots import plots
    plots(config, rows, results_dir)
    return results_dir, rows


def main(argv) -> None:
    if len(argv) != 2:
        names = sorted(path.stem for path in CONFIGS_DIR.glob("*.yaml"))
        print("usage: python -m experiments.run configs/<name>.yaml\n"
              f"configs: {', '.join(names)}", file=sys.stderr)
        raise SystemExit(2)
    results_dir, rows = run_experiment(argv[1])
    print("\n".join(terminal_lines(rows)))
    print(f"\nfull table: {results_dir}/sweep.md   every column: sweep.csv")


if __name__ == "__main__":
    main(sys.argv)
