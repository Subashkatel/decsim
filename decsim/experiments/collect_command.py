"""`decsim collect`: every shot of every sweep point of one yaml.

The config is the experiment; this module only orchestrates. It runs
every sweep point's shots (decsim.collect) and saves each work unit's
additive facts as a piece of the experiment folder the moment the unit
ends (pieces). Then it folds the pieces into the configuration's run
folder, derives one row per point, and writes the figures and the
residence and wait table of the traced shots there (report, run_folder,
plots, residence). A piece already saved is skipped, so a killed collect
resumes where it stopped; rerunning the same config reproduces the same
rows (seeds 0..shots-1 per point; only the wall-clock column varies),
and so does running it with a process pool.
"""

import functools
import pathlib
import sys
from typing import Optional

import decsim.collect as collect
import decsim.escalation.settings as escalation_settings
import decsim.experiments.experiment as experiment
import decsim.experiments.measure as measure
import decsim.experiments.pieces as pieces
import decsim.experiments.plots as plots
import decsim.experiments.report as report
import decsim.experiments.residence as residence
import decsim.experiments.run_folder as run_folder


def run_sweep(tasks: list, *, processes: int = 1) -> list:
    """Every shot of every task of a sweep (config.tasks()), measured.

    A point named by more than one block runs once, and nothing is
    written. The measure handed to the pool is a partial of a
    module-level function, so a worker process can unpickle it.
    """
    measure_shot = _shot_measure(tasks, None)
    return collect.collect(
        tasks,
        measure_shot,
        _report_point_done,
        processes=processes,
    )


def run_experiment(
    config_path,
    out_dir: Optional[pathlib.Path] = None,
    *,
    processes: int = 1,
) -> tuple:
    """One full experiment: pieces, then the run folder folded from them.

    out_dir is the experiment folder. Each work unit is saved as a
    piece when it ends, and a piece already saved is skipped, so a
    killed collect run again runs only what it had not saved. Returns
    the configuration's run folder, combined/<name>-<id8>/, and its
    summary rows.
    """
    config = experiment.load_experiment(config_path)
    experiment_dir = run_folder.run_dir_for(config, out_dir)
    report_dir = run_folder.combined_folder(experiment_dir, config)
    point_tasks = config.point_tasks()
    unique = _unique_tasks(point_tasks)
    point_ids = _point_ids(unique)
    started_utc = run_folder.start_run(config, report_dir, point_ids)
    _record_the_points(experiment_dir, config, unique)
    first_task = unique[0]
    _echo_description(config, first_task.settings, report_dir)
    unsaved = _unsaved_units(experiment_dir, point_tasks, unique)
    measure_shot = _shot_measure(unique, report_dir)
    swept = run_folder.swept_values(experiment_dir, point_ids)
    traced = _run_the_pieces(
        unsaved, experiment_dir, measure_shot, swept, processes
    )
    folders = pieces.folders_of(experiment_dir, point_ids)
    rows = _fold_the_pieces(experiment_dir, folders, point_ids, report_dir)
    residence_rows = residence.rows_of(traced)
    residence.write_residence(residence_rows, report_dir)
    plots.plots(report_dir)
    run_folder.finish_run(config, report_dir, point_ids, started_utc)
    return report_dir, rows


def _shot_measure(tasks: list, run_dir):
    """The measure every shot runs through, written for a worker process.

    It is a partial of a module-level function, so a pool can pickle it.
    """
    only_traced_shot = _traces_one_shot(tasks)
    return functools.partial(
        measure.measure_shot,
        run_dir=run_dir,
        only_traced_shot=only_traced_shot,
    )


def _traces_one_shot(tasks: list) -> bool:
    """Whether the sweep is one point that traces one shot."""
    unique = collect.unique_tasks(tasks)
    if len(unique) != 1:
        return False
    observation = unique[0].settings.observation
    return len(observation.trace_shots) == 1


def _point_ids(unique: list) -> list:
    """The sweep's point ids in task order."""
    point_ids = []
    for task in unique:
        point_id = task.strong_id()
        point_ids.append(point_id)
    return point_ids


def _record_the_points(
    experiment_dir: pathlib.Path,
    config: experiment.ExperimentConfig,
    unique: list,
) -> None:
    """Every point's values, maker and workload, and the configuration.

    Recording builds each point's plan, so a point the build refuses
    stops the run before any shot. The seeds a point ran are its
    pieces', which its run folder's record gathers when it folds.
    """
    for task in unique:
        sections = config.resolved_sections(task.metadata)
        run_folder.record_point(experiment_dir, task, None, sections)
    run_folder.record_configuration(experiment_dir, config)


def _unique_tasks(point_tasks: list) -> list:
    """The tasks of (task, collection) pairs, a point named twice once."""
    tasks = []
    for task, _collection in point_tasks:
        tasks.append(task)
    return collect.unique_tasks(tasks)


def _collection_by_point(point_tasks: list) -> dict:
    """Each point's collection, the first block that names it deciding."""
    collections = {}
    for task, collection in point_tasks:
        point_id = task.strong_id()
        collections.setdefault(point_id, collection)
    return collections


def _piece_units(
    experiment_dir: pathlib.Path, unique: list, collections: dict
) -> list:
    """Every point's seeds cut into pieces of its collection's rounds.

    A point whose escalation calibrates its threshold online is one
    piece: its calibrator learns over the point's shots in order, and
    its state is not saved between pieces.
    """
    records = run_folder.resolved_by_point(experiment_dir)
    units = []
    for task in unique:
        point_id = task.strong_id()
        rounds_per_shot = records[point_id]["rounds_per_shot"]
        collection = collections[point_id]
        piece_shots = collection.piece_shots(rounds_per_shot)
        if task.online_threshold is not None:
            piece_shots = task.shots
        point_units = _units_of(task, piece_shots)
        units.extend(point_units)
    return units


def _units_of(task: collect.Task, piece_shots: int) -> list:
    """One task's seeds 0 to shots - 1 in pieces of piece_shots."""
    units = []
    for first_seed in range(0, task.shots, piece_shots):
        remaining = task.shots - first_seed
        count = min(piece_shots, remaining)
        unit = collect.Unit(task, first_seed, count)
        units.append(unit)
    return units


def _unsaved_units(
    experiment_dir: pathlib.Path, point_tasks: list, unique: list
) -> list:
    """Every point's pieces not saved yet, in order, as work units."""
    collections = _collection_by_point(point_tasks)
    units = _piece_units(experiment_dir, unique, collections)
    unsaved = []
    for unit in units:
        point_id = unit.task.strong_id()
        is_saved = pieces.is_written(
            experiment_dir, point_id, unit.first_seed, unit.seeds
        )
        if not is_saved:
            unsaved.append(unit)
    return unsaved


def _run_the_pieces(
    units: list,
    experiment_dir: pathlib.Path,
    measure_shot,
    swept: dict,
    processes: int,
) -> list:
    """Each unit run and saved as a piece; the traced shots' measurements.

    The traced shots feed the residence table; the rest of a unit's
    measurements are its piece's files and are not held. swept is each
    point's swept values, which an online threshold record's rows carry.
    """
    traced = []
    report_dir = measure_shot.keywords["run_dir"]
    on_task_done = functools.partial(
        _report_point_done, run_dir=report_dir, swept=swept
    )
    save = functools.partial(_save_the_piece, experiment_dir, traced)
    collect.run_units(
        units,
        measure_shot,
        on_task_done,
        on_unit_done=save,
        processes=processes,
    )
    return traced


def _save_the_piece(
    experiment_dir: pathlib.Path, traced: list, unit: collect.Unit, rows: list
) -> None:
    """One unit's measurements saved as its piece; its traced shots kept."""
    point_id = unit.task.strong_id()
    pieces.write(experiment_dir, point_id, unit.first_seed, rows, {})
    for measurement in rows:
        if measurement.trace_path is not None:
            traced.append(measurement)


def _fold_the_pieces(
    experiment_dir: pathlib.Path,
    folders: list,
    point_ids: list,
    report_dir: pathlib.Path,
) -> list:
    """The configuration's pieces folded into its run folder; its rows."""
    seeds_by_point = pieces.seed_ranges_of(folders)
    return report.fold_pieces(
        experiment_dir, folders, point_ids, seeds_by_point, report_dir
    )


def _echo_description(config, settings, run_dir: pathlib.Path) -> None:
    """The resolved experiment, before the first shot, as gem5 dumps it.

    settings is the first point's, whose sections the lines name.
    """
    description = experiment.resolved_description(config, settings)
    description.append(f"run dir: {run_dir}\n")
    description_text = "\n".join(description)
    print(description_text, file=sys.stderr)


def _report_point_done(
    task: collect.Task,
    run_dir: Optional[pathlib.Path] = None,
    swept: Optional[dict] = None,
) -> None:
    """The progress line, and the online threshold's record when it ran."""
    metadata = collect.metadata_text(task.metadata)
    print(f"{metadata}: {task.shots} shots done", file=sys.stderr)
    if task.online_threshold is not None:
        _write_online_threshold_record(task, run_dir, swept)


def _write_online_threshold_record(
    task: collect.Task,
    run_dir: Optional[pathlib.Path],
    swept: Optional[dict],
) -> None:
    """One csv per sweep point with the online threshold's trajectory.

    Every audit and target move, plus every 100th window, and a summary
    line on stderr. The gap unit inside the calibrator is nats; the csv
    converts to the paper's decibels. The file is named by the point's
    id, as its resolved/ record is, and its rows carry the point's id,
    swept values and algorithm, as every other csv's rows do.
    """
    calibrator = task.online_threshold
    summary = calibrator.summary()
    final_threshold_nats = summary["threshold"]
    threshold_db = escalation_settings.nats_to_decibels(final_threshold_nats)
    summary_line = _threshold_summary_line(summary, threshold_db)
    print(summary_line, file=sys.stderr)
    if run_dir is None:
        return
    point_id = task.strong_id()
    algorithm = measure.active_decoder_kind(task.settings)
    point = report.point_columns((point_id, algorithm))
    rows = []
    for window_count, threshold_nats, event in calibrator.trajectory:
        row_db = escalation_settings.nats_to_decibels(threshold_nats)
        row = {**point, "window_count": window_count}
        row["threshold_db"] = row_db
        row["event"] = event
        rows.append(row)
    end_row = {**point, "window_count": summary["windows"]}
    end_row["threshold_db"] = threshold_db
    end_row["event"] = "end"
    rows.append(end_row)
    record_path = pathlib.Path(run_dir) / f"online_threshold_{point_id}.csv"
    report.write_csv(rows, record_path, swept)


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
