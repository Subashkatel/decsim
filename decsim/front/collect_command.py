"""`decsim collect`: every shot of every sweep point of one yaml.

The config is the experiment; this module only orchestrates. It collects
every shot of every sweep point (decsim.collect), records the shots'
additive facts, derives one row per point from them, and writes both
plus the figures into the run folder (report, run_folder, plots).
Rerunning the same config reproduces the same rows
(seeds 0..shots-1 per point; only the wall-clock column varies), and so
does running it with a process pool or in shards that `decsim combine`
folds back together.
"""

import csv
import functools
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

NATS_TO_DECIBELS = 10.0 / math.log(10.0)


def run_sweep(
    config,
    run_dir: Optional[Path] = None,
    *,
    processes: int = 1,
    shard: Optional[tuple] = None,
    shots_per_unit: Optional[int] = None,
) -> list:
    """Every shot of every point of the config's sweep blocks, measured.

    A point named by more than one block runs once; run_dir receives the
    trace files and the online threshold records. The measure handed to
    the pool is a partial of a module-level function, so a worker
    process can unpickle it.
    """
    tasks = config.tasks()
    measure_shot = functools.partial(measure.measure_shot, run_dir=run_dir)
    on_task_done = functools.partial(_report_point_done, run_dir=run_dir)
    return collect.collect(
        tasks,
        measure_shot,
        on_task_done,
        processes=processes,
        shard=shard,
        shots_per_unit=shots_per_unit,
    )


def run_experiment(
    config_path,
    out_dir: Optional[Path] = None,
    *,
    processes: int = 1,
    shard: Optional[tuple] = None,
    shots_per_unit: Optional[int] = None,
) -> tuple:
    """One full experiment: sweep, summary, report, figures.

    Returns the results folder and the summary rows.
    """
    config = experiment.load_experiment(config_path)
    run_dir = run_folder.run_dir_for(config, out_dir)
    run_folder.snapshot_code_state(config, run_dir)
    started_utc = run_folder.utc_now()
    run_folder.write_manifest(config, run_dir, started_utc)
    _echo_description(config, run_dir, shard)
    measurements = run_sweep(
        config,
        run_dir,
        processes=processes,
        shard=shard,
        shots_per_unit=shots_per_unit,
    )
    if not measurements:
        _report_no_work_unit()
        _finish_the_manifest(config, run_dir, started_utc)
        return run_dir, []
    record = report.record_of(measurements)
    rows = report.summarize(record.shots, record.window_samples)
    report.write_report(rows, run_dir, record)
    plots.plots(config, rows, run_dir, measurements)
    _finish_the_manifest(config, run_dir, started_utc)
    return run_dir, rows


def _report_no_work_unit() -> None:
    """One line for a run that held no work unit of the sweep.

    A Slurm array sized above the sweep's unit count leaves its last
    shards nothing to run. That is the array's shape and not a
    failure, so the folder keeps its manifest, no csv is written and
    the command exits 0; `decsim combine` skips such a folder.
    """
    print(
        "no work unit of this sweep fell to this run, so it wrote no rows "
        "beyond its manifest",
        file=sys.stderr,
    )


def _finish_the_manifest(config, run_dir: Path, started_utc: str) -> None:
    """The manifest again, now carrying the time the run ended."""
    finished_utc = run_folder.utc_now()
    run_folder.write_manifest(
        config, run_dir, started_utc, finished_utc=finished_utc
    )


def _echo_description(config, run_dir: Path, shard: Optional[tuple]) -> None:
    """The resolved experiment, before the first shot, as gem5 dumps it."""
    description = experiment.resolved_description(config)
    if shard is not None:
        index, count = shard
        description.append(f"shard: {index} of {count}")
    description.append(f"run dir: {run_dir}\n")
    description_text = "\n".join(description)
    print(description_text, file=sys.stderr)


def _report_point_done(
    task: collect.Task, run_dir: Optional[Path] = None
) -> None:
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
