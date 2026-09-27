"""`decsim collect`: every shot of every sweep point of one yaml.

The config is the experiment; this module only orchestrates. It collects
every shot of every sweep point (decsim.collect), records the shots'
additive facts, derives one row per point from them, and writes both
plus the figures into the run folder (report, run_folder, plots), and
the residence and wait table of the shots that were traced (residence).
Rerunning the same config reproduces the same rows
(seeds 0..shots-1 per point; only the wall-clock column varies), and so
does running it with a process pool or in shards that `decsim combine`
folds back together.
"""

import csv
import functools
import pathlib
import sys
from typing import Optional

import decsim.collect as collect
import decsim.escalation.settings as escalation_settings
import decsim.experiments.experiment as experiment
import decsim.experiments.measure as measure
import decsim.experiments.plots as plots
import decsim.experiments.report as report
import decsim.experiments.residence as residence
import decsim.experiments.run_folder as run_folder


def run_sweep(
    tasks: list,
    run_dir: Optional[pathlib.Path] = None,
    *,
    processes: int = 1,
    shard: Optional[tuple] = None,
    shots_per_unit: Optional[int] = None,
) -> list:
    """Every shot of every task of a sweep (config.tasks()), measured.

    A point named by more than one block runs once; run_dir receives the
    trace files and the online threshold records. The measure handed to
    the pool is a partial of a module-level function, so a worker
    process can unpickle it.
    """
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
    out_dir: Optional[pathlib.Path] = None,
    *,
    processes: int = 1,
    shard: Optional[tuple] = None,
    shots_per_unit: Optional[int] = None,
) -> tuple:
    """One full experiment: sweep, summary, report, figures.

    Returns the results folder and the summary rows. A folder a run
    already finished is skipped, so rerunning a Slurm array reruns only
    the shards that did not finish.
    """
    config = experiment.load_experiment(config_path)
    run_dir = run_folder.run_dir_for(config, out_dir)
    if run_folder.is_finished(run_dir):
        _report_the_finished_folder(run_dir)
        return run_dir, []
    how_it_ran = {"shard": shard, "shots_per_unit": shots_per_unit}
    tasks = config.tasks()
    point_ids = _point_ids(tasks)
    started_utc = run_folder.start_run(config, run_dir, point_ids, **how_it_ran)
    _record_the_points(run_dir, tasks, **how_it_ran)
    first_task = tasks[0]
    _echo_description(config, first_task.settings, run_dir, shard)
    measurements = run_sweep(
        tasks,
        run_dir,
        processes=processes,
        shard=shard,
        shots_per_unit=shots_per_unit,
    )
    if not measurements:
        _report_no_work_unit()
        run_folder.finish_run(
            config, run_dir, point_ids, started_utc, **how_it_ran
        )
        return run_dir, []
    record = report.record_of(measurements)
    rows = report.summarize(record.shots, record.window_samples)
    report.write_report(rows, run_dir, record)
    residence_rows = residence.rows_of(measurements)
    residence.write_residence(residence_rows, run_dir)
    plots.plots(run_dir)
    run_folder.finish_run(config, run_dir, point_ids, started_utc, **how_it_ran)
    return run_dir, rows


def _point_ids(tasks: list) -> list:
    """The sweep's point ids in task order, a point named twice once."""
    unique = collect.unique_tasks(tasks)
    point_ids = []
    for task in unique:
        point_id = task.strong_id()
        point_ids.append(point_id)
    return point_ids


def _record_the_points(
    run_dir: pathlib.Path, tasks: list, shard, shots_per_unit
) -> None:
    """The maker, and every point's values and workload, before any shot.

    producer.json holds the first point's maker and arguments; every
    point's own arguments are in its resolved/ record.

    Every shard records every point, with its own units' seeds, so a
    shard's folder alone says what its sweep was and combine folds the
    content-named files into one set. Recording builds each point's
    plan, so a point the build refuses stops the run before any shot.
    """
    first_task = tasks[0]
    run_folder.write_producer(run_dir, first_task.settings.workload)
    unique = collect.unique_tasks(tasks)
    units = collect.work_units(unique, shots_per_unit)
    selected = collect.shard_of(units, shard)
    for task in unique:
        seeds = _seeds_of(task, selected)
        run_folder.record_point(run_dir, task, seeds)


def _seeds_of(task, units: list) -> list:
    """The seed ranges of the task's units among these."""
    ranges = []
    for unit in units:
        if unit.task is task:
            ranges.append((unit.first_seed, unit.seeds))
    return run_folder.seed_ranges(ranges)


def _report_the_finished_folder(run_dir: pathlib.Path) -> None:
    """One line for a folder a run already finished, which is left as it is."""
    print(
        f"{run_dir} holds a finished run, so this run leaves it as it is; "
        "name another --out to run the sweep again",
        file=sys.stderr,
    )


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


def _echo_description(
    config, settings, run_dir: pathlib.Path, shard: Optional[tuple]
) -> None:
    """The resolved experiment, before the first shot, as gem5 dumps it.

    settings is the first point's, whose sections the lines name.
    """
    description = experiment.resolved_description(config, settings)
    if shard is not None:
        index, count = shard
        description.append(f"shard: {index} of {count}")
    description.append(f"run dir: {run_dir}\n")
    description_text = "\n".join(description)
    print(description_text, file=sys.stderr)


def _report_point_done(
    task: collect.Task, run_dir: Optional[pathlib.Path] = None
) -> None:
    """The progress line, and the online threshold's record when it ran."""
    metadata = collect.metadata_text(task.metadata)
    print(f"{metadata}: {task.shots} shots done", file=sys.stderr)
    if task.online_threshold is not None:
        point_id = task.strong_id()
        _write_online_threshold_record(task.online_threshold, run_dir, point_id)


def _write_online_threshold_record(
    calibrator, run_dir: Optional[pathlib.Path], point_id: str
) -> None:
    """One csv per sweep point with the online threshold's trajectory.

    Every audit and target move, plus every 100th window, and a summary
    line on stderr. The gap unit inside the calibrator is nats; the csv
    converts to the paper's decibels. The file is named by the point's
    id, as its resolved/ record is.
    """
    summary = calibrator.summary()
    final_threshold_nats = summary["threshold"]
    threshold_db = escalation_settings.nats_to_decibels(final_threshold_nats)
    summary_line = _threshold_summary_line(summary, threshold_db)
    print(summary_line, file=sys.stderr)
    if run_dir is None:
        return
    record_path = pathlib.Path(run_dir) / f"online_threshold_{point_id}.csv"
    with open(record_path, "w", newline="") as record_file:
        writer = csv.writer(record_file)
        writer.writerow(["window_count", "threshold_db", "event"])
        for window_count, threshold_nats, event in calibrator.trajectory:
            row_db = escalation_settings.nats_to_decibels(threshold_nats)
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
