"""Run one experiment: python -m experiments.run configs/<name>.yaml.

The config is the experiment; this module only orchestrates. It runs every
shot of every sweep point, summarizes one row per point, and writes
sweep.csv, links.csv and the figures to experiments/results/<name>/.
Rerunning the same config reproduces the same rows (seeds 0..shots-1 per
point; only the wall-clock column varies).
"""

import csv
import dataclasses
import datetime
import itertools
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy
import pymatching
import stim

import experiments.build_run as build_run
import experiments.experiment_config as experiment_config
import experiments.measure_shot as measure_shot
import experiments.plots as plots
import experiments.sweep_report as sweep_report

_THIS_FILE = Path(__file__)
CONFIGS_DIR = _THIS_FILE.parent / "configs"
NATS_TO_DECIBELS = 10.0 / math.log(10.0)


def resolved_description(config: experiment_config.ExperimentConfig) -> list:
    """What this run will actually do, echoed before the first shot.

    An edit that did not land shows up here immediately: a config that
    extends another replaces its base's keys whole, so a `sweep` edited in
    the base never reaches a child that declares its own.
    """
    file_names = []
    for path in config.config_files:
        file_names.append(str(path))
    files = " <- ".join(file_names)
    unit = config.active_decoder
    tier = experiment_config.DECODE_PATH_TIER[config.decode_path]
    lines = [
        f"config: {files}",
        f"decode_path: {config.decode_path}, {config.circuit}, "
        f"{config.rounds_per_shot} rounds per shot",
        f"windows: {config.windowing.scheme}",
        f"decoder: the {tier} tier, algorithm {unit.algorithm}, "
        f"{unit.units} unit(s), engine clock {unit.engine.clock}",
    ]
    for index, block in enumerate(config.sweep, start=1):
        probabilities = list(block.physical_error_probabilities)
        distances = list(block.distances)
        periods = list(block.round_periods_microseconds)
        lines.append(
            f"sweep block {index}: p {probabilities}, d {distances}, "
            f"round period {periods} us, {block.shots} shots"
        )
    trace_line = f"trace: {config.trace}"
    if config.log_component_io:
        trace_line += " with component I/O"
    lines.append(trace_line)
    return lines


def new_run_dir(config: experiment_config.ExperimentConfig) -> Path:
    """results/<UTC stamp>-<config name>/, never reused; sorted by time.

    The stamp is whole seconds, so a second run started within the same
    second gets a numeric suffix instead of clobbering the first.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H-%M-%SZ")
    base = Path("experiments/results") / f"{stamp}-{config.name}"
    run_dir = base
    suffix = 2
    while True:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            run_dir = base.with_name(f"{base.name}-{suffix}")
            suffix += 1


def snapshot_code_state(
    config: experiment_config.ExperimentConfig, run_dir: Path
) -> None:
    """Copy the run's exact inputs next to its results.

    The config chain goes verbatim into config/, and any uncommitted code
    into code_state.patch, so manifest commit + patch + config = the
    whole experiment.
    """
    config_dir = run_dir / "config"
    config_dir.mkdir()
    for config_file in config.config_files:
        source = Path(config_file)
        target = config_dir / source.name
        shutil.copy2(config_file, target)
    diff = _git_output("git", "diff", "HEAD")
    if diff:
        patch_path = run_dir / "code_state.patch"
        patch_path.write_text(diff)


def write_manifest(
    config: experiment_config.ExperimentConfig,
    run_dir: Path,
    started_utc: str,
    finished_utc: Optional[str] = None,
) -> None:
    """Write the run's identity: enough to interpret or reproduce it.

    Sampling is deterministic from (stim version, circuit, distance,
    rounds, p, seed), so the manifest plus seeds are the raw data.
    """
    # default=str turns Paths and cards into strings; the round trip
    # leaves a plain json-safe dict.
    config_as_dict = dataclasses.asdict(config)
    config_text = json.dumps(config_as_dict, default=str)
    json_safe_config = json.loads(config_text)
    config_files = []
    for path in config.config_files:
        config_files.append(str(path))
    container = os.environ.get("APPTAINER_CONTAINER")
    if not container:
        container = os.environ.get("SINGULARITY_CONTAINER")
    manifest = {
        "config_files": config_files,
        "resolved_config": json_safe_config,
        "git": _git_state(),
        "container": container,
        "versions": _versions(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "argv": sys.argv,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_text = json.dumps(manifest, indent=2)
    manifest_path.write_text(manifest_text)


def run_sweep(
    config: experiment_config.ExperimentConfig, run_dir: Optional[Path] = None
) -> list:
    """Every shot of every point of the config's sweep blocks.

    A point and seed named by more than one block runs once.
    """
    measurements = {}
    for block in config.sweep:
        points = itertools.product(
            block.physical_error_probabilities,
            block.distances,
            block.round_periods_microseconds,
        )
        for physical_error_probability, distance, round_period_us in points:
            _measure_point(
                config,
                measurements,
                run_dir,
                physical_error_probability=physical_error_probability,
                distance=distance,
                round_period_us=round_period_us,
                shots=block.shots,
            )
    ordered = measurements.values()
    return list(ordered)


def run_experiment(config_path) -> tuple:
    """One full experiment: sweep, summary, report, figures.

    Returns the results folder and the summary rows.
    """
    config = experiment_config.load_experiment(config_path)
    run_dir = new_run_dir(config)
    snapshot_code_state(config, run_dir)
    started_utc = _utc_now()
    write_manifest(config, run_dir, started_utc)
    description = resolved_description(config)
    description.append(f"run dir: {run_dir}\n")
    description_text = "\n".join(description)
    print(description_text, file=sys.stderr)
    measurements = run_sweep(config, run_dir)
    rows = sweep_report.summarize(measurements)
    sweep_report.write_report(rows, run_dir, measurements)
    plots.plots(config, rows, run_dir, measurements)
    finished_utc = _utc_now()
    write_manifest(config, run_dir, started_utc, finished_utc=finished_utc)
    return run_dir, rows


def main(argv) -> None:
    """The command line: one config path."""
    if len(argv) != 2:
        names = []
        found = CONFIGS_DIR.glob("*.yaml")
        for path in sorted(found):
            names.append(path.stem)
        listed = ", ".join(names)
        print(
            "usage: python -m experiments.run configs/<name>.yaml\n"
            f"configs: {listed}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    run_dir, rows = run_experiment(argv[1])
    lines = sweep_report.terminal_lines(rows)
    report_text = "\n".join(lines)
    print(report_text)
    print(f"\nevery column: {run_dir}/sweep.csv")


def _measure_point(
    config: experiment_config.ExperimentConfig,
    measurements: dict,
    run_dir: Optional[Path],
    *,
    physical_error_probability: float,
    distance: int,
    round_period_us: float,
    shots: int,
) -> None:
    """Every seed of one sweep point, sharing one online calibrator."""
    threshold_calibrator = build_run.online_threshold_calibrator(
        config,
        physical_error_probability=physical_error_probability,
        distance=distance,
    )
    for seed in range(shots):
        shot_key = (physical_error_probability, distance, round_period_us, seed)
        if shot_key in measurements:
            continue
        measurements[shot_key] = measure_shot.measure_shot(
            config,
            distance=distance,
            seed=seed,
            physical_error_probability=physical_error_probability,
            round_period_us=round_period_us,
            run_dir=run_dir,
            threshold_calibrator=threshold_calibrator,
        )
    print(
        f"p {physical_error_probability}, d {distance}, "
        f"round period {round_period_us} us: {shots} shots done",
        file=sys.stderr,
    )
    if threshold_calibrator is not None:
        _write_online_threshold_record(
            threshold_calibrator,
            run_dir,
            physical_error_probability=physical_error_probability,
            distance=distance,
            round_period_us=round_period_us,
        )


def _write_online_threshold_record(
    calibrator,
    run_dir: Optional[Path],
    *,
    physical_error_probability: float,
    distance: int,
    round_period_us: float,
) -> None:
    """One csv per sweep point with the online threshold's trajectory.

    Every audit and target move, plus every 100th window, and a summary
    line on stderr. The gap unit inside the calibrator is nats; the csv
    converts to the paper's decibels.
    """
    summary = calibrator.summary()
    threshold_db = summary["threshold"] * NATS_TO_DECIBELS
    print(
        f"    online threshold: {summary['windows']} windows, "
        f"escalation rate {summary['escalation_rate']:.4f}, "
        f"threshold {threshold_db:.2f} dB, "
        f"{summary['audited']} audits ({summary['audited_bad']} bad), "
        f"{summary['raises']} raises, {summary['relaxes']} relaxes, "
        f"{summary['pending_audits']} pending",
        file=sys.stderr,
    )
    if run_dir is None:
        return
    record_path = Path(run_dir) / (
        f"online_threshold_p{physical_error_probability}_d{distance}"
        f"_round{round_period_us}us.csv"
    )
    with open(record_path, "w", newline="") as record_file:
        writer = csv.writer(record_file)
        writer.writerow(["window_count", "threshold_db", "event"])
        for window_count, threshold_nats, event in calibrator.trajectory:
            row_db = threshold_nats * NATS_TO_DECIBELS
            writer.writerow([window_count, row_db, event])
        writer.writerow([summary["windows"], threshold_db, "end"])


def _git_state() -> dict:
    """The commit and whether the tree is dirty; None where git is absent.

    The container image has no git binary, so the commit falls back to
    reading .git directly; dirty stays a host-side best effort.
    """
    commit = _git_output("git", "rev-parse", "HEAD")
    if not commit:
        commit = _commit_from_git_files()
    porcelain = _git_output("git", "status", "--porcelain")
    is_dirty = None
    if porcelain is not None:
        is_dirty = bool(porcelain)
    return {"commit": commit, "dirty": is_dirty}


def _git_output(*arguments) -> Optional[str]:
    try:
        completed = subprocess.run(arguments, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return completed.stdout.strip()


def _commit_from_git_files() -> Optional[str]:
    head_path = Path(".git/HEAD")
    if not head_path.exists():
        return None
    head_text = head_path.read_text()
    head = head_text.strip()
    if not head.startswith("ref: "):
        return head
    reference = head[len("ref: ") :]
    reference_path = Path(".git") / reference
    if reference_path.exists():
        reference_text = reference_path.read_text()
        return reference_text.strip()
    packed = Path(".git/packed-refs")
    if packed.exists():
        return _packed_reference(packed, reference)
    return None


def _packed_reference(packed: Path, reference: str) -> Optional[str]:
    packed_text = packed.read_text()
    for line in packed_text.splitlines():
        if line.endswith(reference):
            words = line.split()
            return words[0]
    return None


def _versions() -> dict:
    version_words = sys.version.split()
    python_version = version_words[0]
    return {
        "python": python_version,
        "stim": stim.__version__,
        "pymatching": pymatching.__version__,
        "numpy": numpy.__version__,
    }


def _utc_now() -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.isoformat()


if __name__ == "__main__":
    main(sys.argv)
