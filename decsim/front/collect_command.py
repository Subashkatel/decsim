"""Collect one experiment: every shot of every sweep point of one yaml.

The config is the experiment; this module only orchestrates. It collects
every shot of every sweep point (decsim.collect), summarizes one row per
point, and writes sweep.csv, links.csv and the figures into the run
folder (run_folder). Rerunning the same config reproduces the same rows
(seeds 0..shots-1 per point; only the wall-clock column varies).
"""

import csv
import math
import sys
from pathlib import Path
from typing import Optional

import decsim.collect as collect
import decsim.front.experiment as experiment
import decsim.front.measure as measure
import decsim.front.plots as plots
import decsim.front.report as report
import decsim.front.run_folder as run_folder

CONFIGS_DIR = Path("configs")
NATS_TO_DECIBELS = 10.0 / math.log(10.0)


def resolved_description(config) -> list:
    """What this run will actually do, echoed before the first shot.

    An edit that did not land shows up here immediately: a config that
    extends another replaces its base's keys whole, so a `sweep` edited in
    the base never reaches a child that declares its own.
    """
    file_names = []
    for path in config.config_files:
        file_names.append(str(path))
    files = " <- ".join(file_names)
    settings = config.settings
    unit = config.active_decoder
    tier = settings.escalation.decodes_on
    lines = [
        f"config: {files}",
        f"escalation: {settings.escalation.kind}, "
        f"{settings.workload.code_task}, "
        f"{settings.workload.rounds_per_shot} rounds per shot",
        f"windows: {settings.windows.kind}",
        f"decoder: the {tier} tier, kind {unit.kind}, "
        f"{unit.units} unit(s), engine at {unit.engine_megahertz} MHz",
    ]
    for index, block in enumerate(config.sweep, start=1):
        block_line = _sweep_block_line(index, block)
        lines.append(block_line)
    log_line = _log_line(settings)
    lines.append(log_line)
    return lines


def run_sweep(config, run_dir: Optional[Path] = None) -> list:
    """Every shot of every point of the config's sweep blocks, measured.

    A point named by more than one block runs once; run_dir receives
    the trace files and the online threshold records.
    """
    tasks = config.tasks()

    def measure_shot(shot: collect.Shot) -> measure.ShotMeasurement:
        return measure.measure_shot(shot, run_dir)

    def on_task_done(task: collect.Task) -> None:
        _report_point_done(task, run_dir)

    return collect.collect(tasks, measure_shot, on_task_done)


def run_experiment(config_path) -> tuple:
    """One full experiment: sweep, summary, report, figures.

    Returns the results folder and the summary rows.
    """
    config = experiment.load_experiment(config_path)
    run_dir = run_folder.new_run_dir(config)
    run_folder.snapshot_code_state(config, run_dir)
    started_utc = run_folder.utc_now()
    run_folder.write_manifest(config, run_dir, started_utc)
    description = resolved_description(config)
    description.append(f"run dir: {run_dir}\n")
    description_text = "\n".join(description)
    print(description_text, file=sys.stderr)
    measurements = run_sweep(config, run_dir)
    rows = report.summarize(measurements)
    report.write_report(rows, run_dir, measurements)
    plots.plots(config, rows, run_dir, measurements)
    finished_utc = run_folder.utc_now()
    run_folder.write_manifest(
        config, run_dir, started_utc, finished_utc=finished_utc
    )
    return run_dir, rows


def main(argv) -> None:
    """The command line: one config path."""
    if len(argv) != 2:
        usage = _usage()
        print(usage, file=sys.stderr)
        raise SystemExit(2)
    run_dir, rows = run_experiment(argv[1])
    lines = report.terminal_lines(rows)
    report_text = "\n".join(lines)
    print(report_text)
    print(f"\nevery column: {run_dir}/sweep.csv")


def _sweep_block_line(index: int, block) -> str:
    """One sweep block's three axes and its shot count, as one line."""
    probabilities = list(block.physical_error_probabilities)
    distances = list(block.distances)
    periods = list(block.round_periods_microseconds)
    return (
        f"sweep block {index}: p {probabilities}, d {distances}, "
        f"round period {periods} us, {block.shots} shots"
    )


def _log_line(settings) -> str:
    """What the engine narrator will record for this run."""
    log_line = f"log: {settings.observation.log}"
    if settings.observation.log_component_io:
        log_line += " with component I/O"
    return log_line


def _usage() -> str:
    """The command line, with the shipped config names listed."""
    names = []
    found = CONFIGS_DIR.glob("*.yaml")
    for path in sorted(found):
        names.append(path.stem)
    listed = ", ".join(names)
    return (
        "usage: python -m decsim.front.collect_command configs/<name>.yaml\n"
        f"configs: {listed}"
    )


def _report_point_done(task: collect.Task, run_dir: Optional[Path]) -> None:
    """The progress line, and the online threshold's record when it ran."""
    point = task.metadata
    physical_error_probability = point["physical_error_probability"]
    distance = point["distance"]
    round_period_us = point["round_period_us"]
    print(
        f"p {physical_error_probability}, d {distance}, "
        f"round period {round_period_us} us: {task.shots} shots done",
        file=sys.stderr,
    )
    if task.online_threshold is not None:
        _write_online_threshold_record(
            task.online_threshold,
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
    summary_line = _threshold_summary_line(summary, threshold_db)
    print(summary_line, file=sys.stderr)
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


def _threshold_summary_line(summary: dict, threshold_db: float) -> str:
    """The online threshold's one-line summary for a finished point."""
    return (
        f"    online threshold: {summary['windows']} windows, "
        f"escalation rate {summary['escalation_rate']:.4f}, "
        f"threshold {threshold_db:.2f} dB, "
        f"{summary['audited']} audits ({summary['audited_bad']} bad), "
        f"{summary['raises']} raises, {summary['relaxes']} relaxes, "
        f"{summary['pending_audits']} pending"
    )


if __name__ == "__main__":
    main(sys.argv)
