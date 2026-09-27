"""`decsim collect`: every sweep point of one yaml, each until it stops.

The config is the experiment; this module only orchestrates. It runs
each sweep point's seeds in order, a piece at a time (decsim.collect),
and saves each piece's additive facts in the experiment folder the
moment it ends (pieces). A point stops by its collection's rule on the
contiguous prefix of its seeds (collection), and no piece past the stop
is started. Then the pieces are folded into the configuration's run
folder, one row per point, beside the figures and the residence and
wait table of the traced shots (report, run_folder, plots, residence).
A piece already saved is counted and not run again, so a killed collect
resumes where it stopped; rerunning the same config reproduces the same
rows (only the wall-clock column varies), and so does running it with a
process pool, as long as each point stops at the same piece.
"""

import dataclasses
import functools
import math
import pathlib
import sys
from typing import Optional

import decsim.collect as collect
import decsim.escalation.settings as escalation_settings
import decsim.experiments.collection as collection_module
import decsim.experiments.experiment as experiment
import decsim.experiments.failure_statistics as failure_statistics
import decsim.experiments.measure as measure
import decsim.experiments.pieces as pieces
import decsim.experiments.plots as plots
import decsim.experiments.refusal as refusal
import decsim.experiments.report as report
import decsim.experiments.residence as residence
import decsim.experiments.run_folder as run_folder


@dataclasses.dataclass
class PointCollection:
    """One point's collection as it runs: its next seed and its prefix.

    saved maps the first seed of each piece saved before this collect
    to its count, whatever collection cut it. pending holds the (first
    seed, count) of the pieces handed out past the counted prefix, in
    seed order; they are counted once they are all saved, so the prefix
    stays contiguous whatever order the pool ends them in.
    """

    task: collect.Task
    settings: collection_module.CollectionSettings
    piece_shots: int
    rounds_per_shot: int
    saved: dict
    next_seed: int = 0
    counts: collection_module.PrefixCounts = dataclasses.field(
        default_factory=collection_module.PrefixCounts
    )
    stop_kind: Optional[failure_statistics.StopKind] = None
    pending: list = dataclasses.field(default_factory=list)

    def next_units(self, experiment_dir: pathlib.Path, wanted: int) -> list:
        """Up to `wanted` unsaved pieces past the prefix, as work units.

        A saved piece met before any unsaved one is counted at once, so
        a point whose saved pieces reach its stop starts nothing.
        """
        units = []
        while self.stop_kind is None and len(units) < wanted:
            count = self._next_piece_count()
            if count == 0:
                break
            unit = self._hand_out(count)
            if unit is None and not units:
                self.count_the_pending(experiment_dir)
            if unit is not None:
                units.append(unit)
        return units

    def count_the_pending(self, experiment_dir: pathlib.Path) -> None:
        """The pending pieces' counts onto the prefix, until it stops."""
        point_id = self.task.strong_id()
        for first_seed, count in self.pending:
            if self.stop_kind is not None:
                break
            folder = pieces.piece_dir(
                experiment_dir, point_id, first_seed, count
            )
            piece_counts = _prefix_counts_of(folder)
            self.counts.add(piece_counts)
            self.stop_kind = self.settings.stop_kind(self.counts)
        self.pending = []
        if self.stop_kind is not None:
            _say_the_point_stopped(self)

    def rule(self) -> collection_module.PointRule:
        """What the point's summary reads its prefix by."""
        is_adaptive = self.task.online_threshold is not None
        return collection_module.PointRule(
            self.settings, is_adaptive, self.rounds_per_shot
        )

    def _hand_out(self, count: int) -> Optional[collect.Unit]:
        """The next piece made pending; its unit, or None when it is saved.

        A saved piece that starts at the next seed is taken whole, and a
        new one ends where the next saved piece starts, so a collect run
        again under a raised cap or target runs no seed twice.
        """
        first_seed = self.next_seed
        saved_count = self.saved.get(first_seed)
        if saved_count is not None:
            self._make_pending(first_seed, saved_count)
            return None
        seeds_free = self._seeds_before_a_saved_piece(first_seed)
        new_count = min(count, seeds_free)
        self._make_pending(first_seed, new_count)
        return collect.Unit(self.task, first_seed, new_count)

    def _make_pending(self, first_seed: int, count: int) -> None:
        self.next_seed = first_seed + count
        self.pending.append((first_seed, count))

    def _seeds_before_a_saved_piece(self, first_seed: int) -> float:
        """The seeds from first_seed to the next saved piece, or infinity."""
        later_starts = []
        for saved_first in self.saved:
            if saved_first > first_seed:
                later_starts.append(saved_first)
        if not later_starts:
            return math.inf
        return min(later_starts) - first_seed

    def _next_piece_count(self) -> int:
        """The next piece's shots: a piece, or what is left below max_shots."""
        max_shots = self.settings.max_shots
        if max_shots is None:
            return self.piece_shots
        seeds_left = max_shots - self.next_seed
        remaining = max(seeds_left, 0)
        return min(self.piece_shots, remaining)


def run_sweep(tasks: list, shots: int, *, processes: int = 1) -> list:
    """Seeds 0 to shots - 1 of every task of a sweep, measured.

    A point named by more than one block runs once, and nothing is
    written. The measure handed to the pool is a partial of a
    module-level function, so a worker process can unpickle it.
    """
    measure_shot = _shot_measure(tasks, None)
    on_task_done = functools.partial(
        _write_online_threshold_record, run_dir=None, swept=None
    )
    return collect.collect(
        tasks,
        shots,
        measure_shot,
        on_task_done,
        processes=processes,
    )


def run_experiment(
    config_path,
    out_dir: Optional[pathlib.Path] = None,
    *,
    processes: int = 1,
) -> tuple:
    """One full experiment: pieces until every point stops, then the fold.

    out_dir is the experiment folder. Each piece is saved when it ends,
    and a piece already saved is counted and skipped, so a killed
    collect run again runs only what it had not saved. Returns the
    configuration's run folder, combined/<name>-<id8>/, and its summary
    rows.
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
    points = _point_collections(experiment_dir, point_tasks, unique)
    measure_shot = _shot_measure(unique, report_dir)
    swept = run_folder.swept_values(experiment_dir, point_ids)
    configuration_id = run_folder.configuration_id(config)
    traced = _collect_until_stopped(
        points,
        experiment_dir,
        configuration_id,
        measure_shot,
        swept,
        processes,
    )
    folders = pieces.folders_of(experiment_dir, point_ids)
    rows = _fold_the_pieces(experiment_dir, folders, points, report_dir)
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
    """Each point's collection; a point two blocks name has one.

    Two collections for one point would give it two stopping rules, so
    that is refused, as sinter refuses a task given twice
    (sinter/_collection/_collection_manager.py:224-226).
    """
    collections = {}
    for task, collection in point_tasks:
        point_id = task.strong_id()
        earlier = collections.setdefault(point_id, collection)
        if earlier != collection:
            _refuse_two_collections(task, earlier, collection)
    return collections


def _refuse_two_collections(task: collect.Task, first, second) -> None:
    """The sentence for a point that two blocks collect two ways."""
    metadata = collect.metadata_text(task.metadata)
    first_text = first.text()
    second_text = second.text()
    raise refusal.RefusalError(
        f"the point {metadata} is in two sweep blocks that collect it two "
        f"ways, {first_text} and {second_text}; a point stops by one rule"
    )


def _point_collections(
    experiment_dir: pathlib.Path, point_tasks: list, unique: list
) -> list:
    """Every point's collection state, its pieces sized by its rounds."""
    collections = _collection_by_point(point_tasks)
    records = run_folder.resolved_by_point(experiment_dir)
    points = []
    for task in unique:
        point_id = task.strong_id()
        settings = collections[point_id]
        rounds_per_shot = records[point_id]["rounds_per_shot"]
        piece_shots = _piece_shots_of(task, settings, rounds_per_shot)
        saved = pieces.saved_counts(experiment_dir, point_id)
        point = PointCollection(
            task, settings, piece_shots, rounds_per_shot, saved
        )
        points.append(point)
    return points


def _piece_shots_of(
    task: collect.Task,
    settings: collection_module.CollectionSettings,
    rounds_per_shot: int,
) -> int:
    """How many shots one of the point's pieces holds.

    A point whose escalation calibrates its threshold online is one
    piece of max_shots: its calibrator learns over the point's shots in
    order and its state is not saved between pieces, so neither a
    target nor a time cap can stop it partway.
    """
    if task.online_threshold is None:
        return settings.piece_shots(rounds_per_shot)
    has_other_stops = settings.max_failures is not None
    if settings.max_core_seconds is not None:
        has_other_stops = True
    if settings.max_shots is None or has_other_stops:
        _refuse_an_online_stop(task)
    return settings.max_shots


def _refuse_an_online_stop(task: collect.Task) -> None:
    """The sentence for an online point given a stop it cannot keep."""
    metadata = collect.metadata_text(task.metadata)
    raise refusal.RefusalError(
        f"the point {metadata} calibrates its threshold online, so it runs "
        "as one piece of max_shots shots; its collection sets max_shots "
        "and neither max_failures nor max_core_seconds"
    )


def _collect_until_stopped(
    points: list,
    experiment_dir: pathlib.Path,
    configuration_id: str,
    measure_shot,
    swept: dict,
    processes: int,
) -> list:
    """Every point's pieces, a round at a time, until each has stopped.

    A round hands the pool about as many pieces as it has processes,
    shared among the points still running, so no point runs far past
    its stop. Each piece names its configuration. Returns the traced
    shots' measurements.
    """
    traced = []
    rounds_by_point = {}
    for point in points:
        point_id = point.task.strong_id()
        rounds_by_point[point_id] = point.rounds_per_shot
    save = functools.partial(
        _save_the_piece,
        experiment_dir,
        configuration_id,
        rounds_by_point,
        traced,
    )
    while True:
        units = _next_round(points, experiment_dir, processes)
        if not units:
            return traced
        _run_the_pieces(units, measure_shot, swept, processes, save)
        for point in points:
            point.count_the_pending(experiment_dir)


def _next_round(
    points: list, experiment_dir: pathlib.Path, processes: int
) -> list:
    """The next round's pieces: the running points' shares of the pool."""
    running = []
    for point in points:
        if point.stop_kind is None:
            running.append(point)
    if not running:
        return []
    share = processes / len(running)
    wanted = math.ceil(share)
    units = []
    for point in running:
        point_units = point.next_units(experiment_dir, wanted)
        units.extend(point_units)
    return units


def _prefix_counts_of(folder: pathlib.Path) -> collection_module.PrefixCounts:
    """One saved piece's counts, read from its piece.json."""
    piece = pieces.read_piece(folder)
    scored_shots = piece["scored_shots"]
    shots = scored_shots + piece["unscored_shots"]
    return collection_module.PrefixCounts(
        shots=shots,
        scored_shots=scored_shots,
        failures=piece["failures"],
        core_seconds=piece["core_seconds"],
    )


def _say_the_point_stopped(point: PointCollection) -> None:
    """The progress line of a point that stopped, and why."""
    metadata = collect.metadata_text(point.task.metadata)
    shots = point.counts.shots
    reason = point.stop_kind.value
    print(f"{metadata}: {shots} shots done ({reason})", file=sys.stderr)


def _run_the_pieces(
    units: list,
    measure_shot,
    swept: dict,
    processes: int,
    save,
) -> None:
    """Each unit run, then handed to save, which writes it as a piece.

    save keeps a unit's traced shots, which feed the residence table;
    the rest of its measurements are its piece's files and are not
    held. swept is each point's swept values, which an online threshold
    record's rows carry.
    """
    report_dir = measure_shot.keywords["run_dir"]
    on_task_done = functools.partial(
        _write_online_threshold_record, run_dir=report_dir, swept=swept
    )
    collect.run_units(
        units,
        measure_shot,
        on_task_done,
        on_unit_done=save,
        processes=processes,
    )


def _save_the_piece(
    experiment_dir: pathlib.Path,
    configuration_id: str,
    rounds_by_point: dict,
    traced: list,
    unit: collect.Unit,
    rows: list,
) -> None:
    """One unit's measurements saved as its piece; its traced shots kept.

    The piece names its configuration and counts its rounds, since a
    shot's cost grows with its rounds.
    """
    point_id = unit.task.strong_id()
    rounds_per_shot = rounds_by_point[point_id]
    facts = {
        "configuration_id": configuration_id,
        "rounds": rounds_per_shot * len(rows),
    }
    pieces.write(experiment_dir, point_id, unit.first_seed, rows, facts)
    for measurement in rows:
        if measurement.trace_path is not None:
            traced.append(measurement)


def _fold_the_pieces(
    experiment_dir: pathlib.Path,
    folders: list,
    points: list,
    report_dir: pathlib.Path,
) -> list:
    """The configuration's pieces folded into its run folder; its rows."""
    seeds_by_point = pieces.seed_ranges_of(folders)
    point_ids = []
    rules = {}
    for point in points:
        point_id = point.task.strong_id()
        point_ids.append(point_id)
        rules[point_id] = point.rule()
    return report.fold_pieces(
        experiment_dir, folders, point_ids, seeds_by_point, report_dir, rules
    )


def _echo_description(config, settings, run_dir: pathlib.Path) -> None:
    """The resolved experiment, before the first shot, as gem5 dumps it.

    settings is the first point's, whose sections the lines name.
    """
    description = experiment.resolved_description(config, settings)
    description.append(f"run dir: {run_dir}\n")
    description_text = "\n".join(description)
    print(description_text, file=sys.stderr)


def _write_online_threshold_record(
    task: collect.Task,
    run_dir: Optional[pathlib.Path],
    swept: Optional[dict],
) -> None:
    """One csv per online sweep point with its threshold's trajectory.

    Nothing for a point whose threshold is not calibrated online.

    Every audit and target move, plus every 100th window, and a summary
    line on stderr. The gap unit inside the calibrator is nats; the csv
    converts to the paper's decibels. The file is named by the point's
    id, as its resolved/ record is, and its rows carry the point's id,
    swept values and algorithm, as every other csv's rows do.
    """
    calibrator = task.online_threshold
    if calibrator is None:
        return
    summary = calibrator.summary()
    final_threshold_nats = summary["threshold"]
    threshold_db = escalation_settings.nats_to_decibels(final_threshold_nats)
    summary_line = _threshold_summary_line(summary, threshold_db)
    print(summary_line, file=sys.stderr)
    if run_dir is None:
        return
    point_id = task.strong_id()
    rows = _trajectory_rows(task, summary, threshold_db)
    record_path = pathlib.Path(run_dir) / f"online_threshold_{point_id}.csv"
    report.write_csv(rows, record_path, swept)


def _trajectory_rows(
    task: collect.Task, summary: dict, threshold_db: float
) -> list:
    """The trajectory's rows, then the end row at the final threshold."""
    point_id = task.strong_id()
    algorithm = measure.active_decoder_kind(task.settings)
    point = report.point_columns((point_id, algorithm))
    rows = []
    for window_count, threshold_nats, event in task.online_threshold.trajectory:
        row_db = escalation_settings.nats_to_decibels(threshold_nats)
        row = {**point, "window_count": window_count}
        row["threshold_db"] = row_db
        row["event"] = event
        rows.append(row)
    end_row = {**point, "window_count": summary["windows"]}
    end_row["threshold_db"] = threshold_db
    end_row["event"] = "end"
    rows.append(end_row)
    return rows


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
