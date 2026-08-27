"""Run one experiment: python -m experiments.run configs/<name>.yaml

The config is the experiment; this module only orchestrates. It runs every
shot of every sweep point, summarizes one row per point, and writes
sweep.csv, links.csv and the figures to experiments/results/<name>/.
Rerunning the same config reproduces the same rows (seeds 0..shots-1 per
point; only the wall-clock column varies).
"""

from __future__ import annotations

import dataclasses
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
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
             f"mode: {config.mode}, {config.code_task}, "
             f"{config.rounds_per_shot} rounds per shot",
             f"windows: {config.windowing.scheme}",
             f"decoder: the {MODE_TIER[config.mode]} tier, algorithm "
             f"{unit.algorithm}, {unit.units} unit(s), "
             f"engine clock {unit.engine.clock}"]
    for index, block in enumerate(config.sweep, start=1):
        lines.append(
            f"sweep block {index}: "
            f"p {list(block.physical_error_probabilities)}, "
            f"d {list(block.distances)}, "
            f"round period {list(block.round_periods_us)} us, "
            f"{block.shots} shots")
    lines.append(f"trace: {config.trace}"
                 + (" with component I/O" if config.trace_io else ""))
    return lines


def new_run_dir(config: ExperimentConfig) -> Path:
    """results/<name>/<UTC stamp>/, never reused; `latest` points at it."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = config.results_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    latest = config.results_dir / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(stamp)
    return run_dir


def _git_state() -> dict:
    # the container image has no git; the manifest records what it can
    def output(*arguments):
        try:
            return subprocess.run(arguments, capture_output=True,
                                  text=True).stdout.strip()
        except FileNotFoundError:
            return None
    commit = output("git", "rev-parse", "HEAD")
    porcelain = output("git", "status", "--porcelain")
    return {"commit": commit,
            "dirty": None if porcelain is None else bool(porcelain)}


def _versions() -> dict:
    import numpy
    import pymatching
    import stim
    return {"python": sys.version.split()[0], "stim": stim.__version__,
            "pymatching": pymatching.__version__,
            "numpy": numpy.__version__}


def write_manifest(config: ExperimentConfig, run_dir: Path,
                   started_utc: str, finished_utc: str = None) -> None:
    """The run's identity: everything needed to interpret or reproduce it
    without the source tree. Sampling is deterministic from (stim version,
    code_task, distance, rounds, p, seed), so the manifest plus seeds are
    the raw data."""
    import os
    manifest = {
        "config_files": [str(path) for path in config.config_files],
        "resolved_config": json.loads(json.dumps(
            dataclasses.asdict(config), default=str)),
        "git": _git_state(),
        "container": os.environ.get("APPTAINER_CONTAINER")
                     or os.environ.get("SINGULARITY_CONTAINER"),
        "versions": _versions(),
        "host": platform.node(),
        "argv": sys.argv,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def run_sweep(config: ExperimentConfig, run_dir: Path = None) -> list:
    """Every shot of every point of the config's sweep blocks; a point and
    seed named by more than one block runs once."""
    measurements = {}
    for block in config.sweep:
        for physical_error_probability in block.physical_error_probabilities:
            for distance in block.distances:
                for round_period_us in block.round_periods_us:
                    for seed in range(block.shots):
                        shot_key = (physical_error_probability, distance,
                                    round_period_us, seed)
                        if shot_key in measurements:
                            continue
                        measurements[shot_key] = measure_shot(
                            config,
                            physical_error_probability=physical_error_probability,
                            distance=distance,
                            round_period_us=round_period_us, seed=seed,
                            run_dir=run_dir)
                    print(f"p {physical_error_probability}, d {distance}, "
                          f"round period {round_period_us} us: "
                          f"{block.shots} shots done", file=sys.stderr)
    return list(measurements.values())


def run_experiment(config_path) -> tuple:
    """One full experiment: sweep, summary, report, figures. Returns the
    results folder and the summary rows."""
    config = load_experiment(config_path)
    run_dir = new_run_dir(config)
    started_utc = datetime.now(timezone.utc).isoformat()
    write_manifest(config, run_dir, started_utc)
    print("\n".join(resolved_description(config))
          + f"\nrun dir: {run_dir}\n", file=sys.stderr)
    measurements = run_sweep(config, run_dir)
    rows = summarize(measurements)
    write_report(rows, run_dir, measurements)
    from experiments.plots import plots
    plots(config, rows, run_dir)
    write_manifest(config, run_dir, started_utc,
                   finished_utc=datetime.now(timezone.utc).isoformat())
    return run_dir, rows


def main(argv) -> None:
    if len(argv) != 2:
        names = sorted(path.stem for path in CONFIGS_DIR.glob("*.yaml"))
        print("usage: python -m experiments.run configs/<name>.yaml\n"
              f"configs: {', '.join(names)}", file=sys.stderr)
        raise SystemExit(2)
    results_dir, rows = run_experiment(argv[1])
    print("\n".join(terminal_lines(rows)))
    print(f"\nevery column: {results_dir}/sweep.csv")


if __name__ == "__main__":
    main(sys.argv)
