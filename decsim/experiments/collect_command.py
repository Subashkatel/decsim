"""`decsim run`: every point of one experiment, each until it stops.

Each point's seeds run in order, a piece at a time, each piece saved the
moment it ends. A point stops by its collection's rule on the
contiguous prefix of its seeds, and no piece past the stop starts. Then
the pieces are folded, one row per point. A saved piece is counted, not
rerun, so a killed collect resumes; the same experiment reproduces the
same rows (only the wall-clock column varies), with a pool too, as long
as each point stops at the same piece. On Slurm each array task collects
one point and one fold job folds them all.
"""

import dataclasses
import functools
import math
import pathlib
import sys
import tempfile
from typing import Optional, Union

import decsim.collect as collect
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.collection as collection_module
import decsim.experiments.experiment as experiment
import decsim.experiments.fold as fold
import decsim.experiments.measure as measure
import decsim.experiments.pieces as pieces
import decsim.experiments.refusal as refusal
import decsim.experiments.report as report
import decsim.experiments.run_folder as run_folder
import decsim.machine as machine_module
import decsim.records.results as result_records
import decsim.settings as machine_settings


@dataclasses.dataclass
class PointCollection:
    """One point's collection as it runs: its next seed and its prefix.

    tracker reads the prefix shot by shot off the saved shots.csv, as the
    report does, so the collector stops on the report's shot. pending holds
    pieces handed out past the counted prefix, counted once all are saved,
    so the prefix stays contiguous whatever order a pool ends them in.

    An adaptive point's calibrator learns over its shots in order, so its
    pieces run one at a time, each from the state the piece before saved.
    """

    name: str
    task: collect.Task
    settings: collection_module.CollectionSettings
    piece_shots: int
    rounds_per_shot: int
    saved: dict
    next_seed: int = 0
    pending: list = dataclasses.field(default_factory=list)
    last_piece: Optional[tuple] = None
    tracker: collection_module.PrefixTracker = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        rule = self.rule()
        self.tracker = collection_module.PrefixTracker(rule)

    def next_units(self, run_dir: pathlib.Path, wanted: int) -> list:
        """Up to `wanted` unsaved pieces past the prefix, as work units.

        A saved piece met before any unsaved one is counted at once, so
        a point whose saved pieces reach its stop starts nothing. An
        adaptive point hands out one piece at a time.
        """
        if self.task.online_threshold is not None:
            wanted = 1
        units = []
        while self.tracker.stop_kind is None and len(units) < wanted:
            count = self._next_piece_count()
            if count == 0:
                break
            unit = self._hand_out(run_dir, count)
            if unit is None and not units:
                self.count_the_pending(run_dir)
            if unit is not None:
                units.append(unit)
        return units

    def count_the_pending(self, run_dir: pathlib.Path) -> None:
        """The pending pieces' shots onto the prefix, until it stops.

        A stopped point counts nothing more, so its line is said once.
        """
        if self.tracker.stop_kind is not None:
            return
        point_id = self.task.strong_id()
        for first_seed, count in self.pending:
            if self.tracker.stop_kind is not None:
                break
            folder = pieces.piece_dir(run_dir, point_id, first_seed, count)
            shots_path = folder / "shots.csv"
            for row in fold.row_stream(shots_path):
                self.tracker.add(row)
        self.pending = []
        if self.tracker.stop_kind is not None:
            _say_the_point_stopped(self)

    def rule(self) -> collection_module.PointRule:
        """What the point's summary reads its prefix by."""
        is_adaptive = self.task.online_threshold is not None
        return collection_module.PointRule(self.settings, is_adaptive)

    def task_as_its_last_piece_left_it(
        self, run_dir: pathlib.Path
    ) -> collect.Task:
        """The task, an adaptive point's calibrator the last piece's state."""
        if self.task.online_threshold is None or self.last_piece is None:
            return self.task
        point_id = self.task.strong_id()
        first_seed, count = self.last_piece
        folder = pieces.piece_dir(run_dir, point_id, first_seed, count)
        return _task_as_the_piece_left_it(self.task, folder)

    def _hand_out(
        self, run_dir: pathlib.Path, count: int
    ) -> Optional[collect.Unit]:
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
        seeds_free = self._seeds_before_the_next_piece(first_seed)
        new_count = min(count, seeds_free)
        task = self.task_as_its_last_piece_left_it(run_dir)
        self._make_pending(first_seed, new_count)
        return collect.Unit(task, first_seed, new_count)

    def _make_pending(self, first_seed: int, count: int) -> None:
        self.next_seed = first_seed + count
        self.pending.append((first_seed, count))
        self.last_piece = (first_seed, count)

    def _seeds_before_the_next_piece(self, first_seed: int) -> float:
        """The seeds from first_seed to the next saved piece, or infinity."""
        later_starts = [start for start in self.saved if start > first_seed]
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


def run_experiment(
    run_file: Union[str, pathlib.Path],
    out_dir: Optional[pathlib.Path] = None,
    *,
    processes: int = 1,
    only: Optional[str] = None,
    shot_count: Optional[int] = None,
) -> tuple:
    """Every point of one run file collected into its results folder.

    out_dir is the results folder, a new dated one when None. only
    names the one point to collect, and shot_count stops every point at
    that many shots (Experiment.with_max_shots). Returns the folder and
    its folded rows.
    """
    _check_processes(processes)
    run_path = pathlib.Path(run_file)
    study = experiment.load(run_path)
    if shot_count is not None:
        study = study.with_max_shots(shot_count)
    run_dir = run_folder.run_dir_for(study.name, out_dir)
    rows = collect_experiment(
        study, run_dir, run_path, processes=processes, only=only
    )
    return run_dir, rows


def collect_experiment(
    study: experiment.Experiment,
    run_dir: pathlib.Path,
    run_file: pathlib.Path,
    *,
    processes: int = 1,
    only: Optional[str] = None,
) -> list:
    """One experiment: pieces until every point stops, then the fold.

    only names the one point to collect, every point when None. Returns the
    folded rows.
    """
    chosen = _chosen_points(study, only)
    every_id, points, started_utc = start_the_folder(
        study, chosen, run_dir, run_file
    )
    print(f"run dir: {run_dir}\n", file=sys.stderr)
    measure_shot = _shot_measure(points, run_dir)
    _collect_until_stopped(points, run_dir, measure_shot, processes)
    folded_ids = run_folder.recorded_point_ids(run_dir, every_id)
    folders = pieces.folders_of(run_dir, folded_ids)
    rows = fold_the_folder(run_dir, folded_ids, folders)
    run_folder.finish_run(run_dir, run_file, every_id, started_utc)
    return rows


def start_the_folder(
    study: experiment.Experiment,
    chosen: experiment.Experiment,
    run_dir: pathlib.Path,
    run_file: pathlib.Path,
) -> tuple:
    """The folder checked, the chosen points recorded, run.json written.

    Returns every point's id in the experiment's order, the chosen
    points' collections, and the start time.
    """
    _refuse_another_tree(run_dir)
    every_id = _experiment_point_ids(study)
    points = recorded_points(run_dir, chosen)
    run_folder.accept_raised_stop_rules(run_dir, run_file, every_id)
    started_utc = run_folder.start_run(run_dir, run_file, every_id)
    return every_id, points, started_utc


def run_task(
    run_file: pathlib.Path,
    run_dir: pathlib.Path,
    index: int,
    *,
    processes: int = 1,
) -> None:
    """One array task: the experiment's index-th point to its stop.

    The launcher wrote run.json and the run file's copy, so a task checks
    the folder ran this tree and this run file, records its point again
    and collects from its saved pieces. It folds nothing.
    """
    _check_processes(processes)
    run_folder.copy_the_run_file(run_file, run_dir)
    study = experiment.load(run_file)
    _refuse_another_tree(run_dir)
    point = study.points[index]
    chosen = study.only(point.name)
    points = recorded_points(run_dir, chosen)
    measure_shot = _shot_measure(points, run_dir)
    _collect_until_stopped(points, run_dir, measure_shot, processes)


def fold_the_run(run_dir: pathlib.Path) -> list:
    """Every recorded point's saved pieces folded; nothing run.

    The points are the ones the folder recorded, in run.json's order,
    so a point the run file no longer makes is still folded.
    """
    run_path = run_dir / run_folder.RUN_FILE
    run_record = run_folder.read_json(run_path)
    point_ids = run_folder.recorded_point_ids(run_dir, run_record["points"])
    folders = pieces.folders_of(run_dir, point_ids)
    return fold_the_folder(run_dir, point_ids, folders)


def run_one_shot(
    study: experiment.Experiment,
    run_file: pathlib.Path,
    seed: int,
    out_dir: Optional[pathlib.Path] = None,
    *,
    log: Optional[str] = None,
    trace: bool = False,
) -> list:
    """The experiment's first point at one seed, run and narrated.

    log and trace override the point's observation for this shot, as gem5's
    --debug-flags (src/python/m5/main.py:280, 299). A point the folder
    recorded with other settings is refused. Returns the lines to print.
    """
    point = study.points[0]
    task = experiment.task_of(point)
    settings = _with_observation(task.settings, log, trace)
    task = dataclasses.replace(task, settings=settings)
    machine = machine_module.Machine.build(
        task.settings, seed, online_threshold=task.online_threshold
    )
    run_dir = run_folder.run_dir_for(study.name, out_dir)
    _refuse_another_tree(run_dir)
    point_id = task.strong_id()
    _refuse_a_name_recorded_for_another_point(run_dir, point.name, point_id)
    run_points = _run_points(run_dir, point_id)
    started_utc = run_folder.start_run(run_dir, run_file, run_points)
    _record_a_new_point(run_dir, point, task, seed)
    result = machine.run()
    label = measure.shot_label(point_id, seed)
    run_folder.write_shot(machine, settings, run_dir, label, result)
    run_folder.finish_run(run_dir, run_file, run_points, started_utc)
    return _shot_lines(study, task, seed, result, run_dir)


def fold_the_folder(
    run_dir: pathlib.Path, point_ids: list, folders: list
) -> list:
    """The points' saved pieces folded into the folder; the rows they give.

    Every file comes from what the folder recorded, not from what this run
    ran. The fold is built in a staging folder and moved in only then, so a
    refused fold leaves the last one as it was.
    """
    pieces.refuse_pieces_of_another_tree(run_dir, folders)
    with tempfile.TemporaryDirectory(dir=run_dir, prefix=".") as staged:
        staging = pathlib.Path(staged)
        rows = _fold_into_the_staging(run_dir, folders, point_ids, staging)
        run_folder.publish_the_fold(staging, run_dir)
    seeds_by_point = pieces.seed_ranges_of(folders)
    run_folder.record_seeds(run_dir, seeds_by_point)
    return rows


def recorded_points(
    run_dir: pathlib.Path, study: experiment.Experiment
) -> list:
    """Every point of the experiment, recorded once all are accepted.

    Every record is built first, which runs its build, so a refused point
    stops the run before any shot and before anything is written. Returns
    each point's collection state, in order.
    """
    resolved = _resolved_points(study)
    for resolved_point in resolved:
        name = resolved_point.record["name"]
        point_id = resolved_point.record["id"]
        _refuse_a_name_recorded_for_another_point(run_dir, name, point_id)
    for resolved_point in resolved:
        run_folder.write_point_record(
            run_dir, resolved_point.task, resolved_point.record
        )
    points = []
    for resolved_point in resolved:
        point = _point_collection(run_dir, resolved_point)
        points.append(point)
    return points


def _refuse_another_tree(run_dir: pathlib.Path) -> None:
    """This tree, and every piece the folder saved, are run.json's tree.

    A new folder has neither run.json nor a piece, so its pieces are
    not asked about.
    """
    run_folder.refuse_another_tree(run_dir)
    saved = pieces.every_folder(run_dir)
    if saved:
        pieces.refuse_pieces_of_another_tree(run_dir, saved)


def _check_processes(processes) -> None:
    """The worker count is a whole number of at least one.

    A share of the pool gives each point its part of the processes, so
    zero gives no piece and the collect would finish having run nothing;
    sinter
    refuses the same count (sinter/_command/_main_collect.py:319-327).
    """
    if isinstance(processes, int) and processes >= 1:
        return
    raise refusal.RefusalError(
        f"processes must be a whole number of at least 1, got {processes!r}"
    )


def _task_as_the_piece_left_it(
    task: collect.Task, folder: pathlib.Path
) -> collect.Task:
    """The task with the online calibrator the piece in folder saved."""
    state = pieces.read_state(folder)
    return dataclasses.replace(task, online_threshold=state)


def _fold_into_the_staging(
    run_dir: pathlib.Path,
    folders: list,
    point_ids: list,
    staging: pathlib.Path,
) -> list:
    """Every file of the fold written into staging; the rows.

    An online point's calibrator, as its prefix left it, gives its
    trajectory file and one row of threshold_summary.csv.
    """
    records = run_folder.point_records(run_dir)
    swept = run_folder.swept_values(run_dir, point_ids)
    rules = {}
    summary_rows = []
    for point_id in point_ids:
        record = records[point_id]
        rules[point_id] = collection_module.PointRule.from_record(record)
        calibrator = _recorded_calibrator(run_dir, folders, record)
        if calibrator is None:
            continue
        algorithm = record["experiment"]["algorithm"]
        point = report.point_columns((point_id, algorithm))
        _write_online_threshold_record(point, calibrator, staging, swept)
        summary_row = _threshold_summary_row(point, calibrator)
        summary_rows.append(summary_row)
    if summary_rows:
        summary_path = staging / "threshold_summary.csv"
        report.write_csv(summary_rows, summary_path, swept)
    rows = report.fold_pieces(run_dir, folders, point_ids, staging, rules)
    return rows


def _recorded_calibrator(
    run_dir: pathlib.Path, folders: list, record: dict
) -> Optional[threshold_sources.OnlineThreshold]:
    """An online point's calibrator as its prefix ended; None if none."""
    facts = record["experiment"]
    if not facts["adaptive"]:
        return None
    point_id = record["id"]
    point_folders = pieces.point_folders(folders, point_id)
    saved = pieces.saved_counts(point_folders)
    prefix = pieces.contiguous_ranges(saved, 0)
    if not prefix:
        return None
    first_seed, count = prefix[-1]
    folder = pieces.piece_dir(run_dir, point_id, first_seed, count)
    return pieces.read_state(folder)


def _shot_measure(points: list, run_dir: pathlib.Path):
    """The measure every shot of these points runs through.

    It is a partial of a module-level function, so a pool can pickle it.
    """
    tasks = [point.task for point in points]
    only_traced_shot = _traces_one_shot(tasks)
    return functools.partial(
        measure.measure_shot,
        run_dir=run_dir,
        only_traced_shot=only_traced_shot,
    )


def _traces_one_shot(tasks: list) -> bool:
    """Whether the run is one point that traces one shot."""
    if len(tasks) != 1:
        return False
    observation = tasks[0].settings.observation
    return len(observation.trace_shots) == 1


def _experiment_point_ids(study: experiment.Experiment) -> list:
    """Every point's id, in the experiment's order."""
    point_ids = []
    for point in study.points:
        task = experiment.task_of(point)
        point_id = task.strong_id()
        point_ids.append(point_id)
    return point_ids


def _chosen_points(
    study: experiment.Experiment, only: Optional[str]
) -> experiment.Experiment:
    """The experiment, or its one point only names."""
    if only is None:
        return study
    return study.only(only)


def _with_observation(
    settings: machine_settings.MachineSettings,
    log: Optional[str],
    trace: bool,
) -> machine_settings.MachineSettings:
    """The flags this shot was given, over the point's observation."""
    changes = {}
    if log is not None:
        changes["log"] = log
    if trace:
        changes["trace"] = "chrome"
    if not changes:
        return settings
    observation = dataclasses.replace(settings.observation, **changes)
    return dataclasses.replace(settings, observation=observation)


def _shot_lines(
    study: experiment.Experiment,
    task: collect.Task,
    seed: int,
    result: result_records.RunResult,
    run_dir: pathlib.Path,
) -> list:
    """The point, the terminal status, the ticks and every result."""
    metadata = collect.metadata_text(task.metadata)
    lines = [
        f"config: {study.name}",
        f"point: {metadata} seed {seed}",
        f"terminal status: {result.terminal_status}",
        f"execution done: {result.execution_done_ticks} ticks",
        f"fully done: {result.fully_done_ticks} ticks",
    ]
    for row in result.operation_results:
        operation_line = _operation_line(row)
        lines.append(operation_line)
    lines.append(f"run dir: {run_dir}")
    return lines


def _operation_line(row) -> str:
    """One operation's status, prediction and truth, as the gate reads it."""
    return (
        f"operation {row.operation_id}: {row.result_status}, "
        f"observables {row.logical_observables}, "
        f"truth {row.observable_truth}"
    )


@dataclasses.dataclass(frozen=True)
class _ResolvedPoint:
    """One point, its task and collection, its record built and not written."""

    task: collect.Task
    settings: collection_module.CollectionSettings
    record: dict


def _resolved_points(study: experiment.Experiment) -> list:
    """Each point checked and its record built, in the experiment's order.

    Two points of one id would share their pieces, so they are refused.
    """
    resolved = []
    names_by_id = {}
    for point in study.points:
        task = experiment.task_of(point)
        point_id = task.strong_id()
        earlier_name = names_by_id.setdefault(point_id, point.name)
        if earlier_name != point.name:
            _refuse_two_points_of_one_id(earlier_name, point.name)
        settings = study.collection_of(point)
        facts = _experiment_facts(task, settings)
        record = run_folder.point_record(point.name, task, None, facts)
        resolved_point = _ResolvedPoint(task, settings, record)
        resolved.append(resolved_point)
    return resolved


def _refuse_two_points_of_one_id(first_name: str, second_name: str) -> None:
    """Two points that run one machine with one metadata are one point."""
    raise refusal.RefusalError(
        f"the points {first_name} and {second_name} run the same settings "
        "with the same metadata, so they would share their shots; give "
        "them metadata that tells them apart"
    )


def _refuse_a_name_recorded_for_another_point(
    run_dir: pathlib.Path, name: str, point_id: str
) -> None:
    """A name the folder recorded for another id is refused.

    The folder names a point's record by its name and its pieces by its
    id, so a name whose settings or metadata changed would fold another
    point's shots under it.
    """
    record_path = _record_path(run_dir, name)
    if not record_path.is_file():
        return
    recorded = run_folder.read_json(record_path)
    recorded_id = recorded["id"]
    if recorded_id == point_id:
        return
    raise refusal.RefusalError(
        f"{run_dir} recorded the point {name} with other settings or "
        f"metadata (id {recorded_id[:12]}) than it has now (id "
        f"{point_id[:12]}); give the point another name, or the run "
        "another folder with --out"
    )


def _record_path(run_dir: pathlib.Path, name: str) -> pathlib.Path:
    """Where the folder keeps the named point's machine.json."""
    point_dir = run_dir / run_folder.POINTS_FOLDER / name
    return point_dir / run_folder.RECORD_FILE


def _record_a_new_point(
    run_dir: pathlib.Path,
    point: experiment.Point,
    task: collect.Task,
    seed: int,
) -> None:
    """The shot's point recorded, unless the folder recorded it already."""
    record_path = _record_path(run_dir, point.name)
    if record_path.is_file():
        return
    seeds = [(seed, 1)]
    run_folder.record_point(run_dir, point.name, task, seeds)


def _run_points(run_dir: pathlib.Path, point_id: str) -> list:
    """run.json's points for one shot: the folder's own when it has them.

    A shot replayed into a collection folder leaves its points, the
    order its fold writes rows in, as the collection recorded them.
    """
    run_path = run_dir / run_folder.RUN_FILE
    if not run_path.is_file():
        return [point_id]
    recorded = run_folder.read_json(run_path)
    return recorded["points"]


def _point_collection(
    run_dir: pathlib.Path, resolved: _ResolvedPoint
) -> PointCollection:
    """A recorded point's collection state, its pieces sized by its rounds."""
    point_id = resolved.record["id"]
    rounds_per_shot = resolved.record["rounds_per_shot"]
    settings = resolved.settings
    piece_shots = settings.piece_shots(rounds_per_shot)
    point_folders = pieces.folders_of(run_dir, [point_id])
    saved = pieces.saved_counts(point_folders)
    return PointCollection(
        resolved.record["name"],
        resolved.task,
        settings,
        piece_shots,
        rounds_per_shot,
        saved,
    )


def _experiment_facts(
    task: collect.Task, settings: collection_module.CollectionSettings
) -> dict:
    """What folding the point needs besides its pieces.

    Its collection, which reads its prefix; whether its threshold learns
    online; and the kind of the tier that decodes its windows, which
    names its trajectory's rows.
    """
    return {
        "collection": dataclasses.asdict(settings),
        "adaptive": task.online_threshold is not None,
        "algorithm": measure.active_decoder_kind(task.settings),
    }


def _collect_until_stopped(
    points: list,
    run_dir: pathlib.Path,
    measure_shot,
    processes: int,
) -> None:
    """Every point's pieces, a share of the pool at a time, until each stops.

    Each share hands the pool about as many pieces as it has processes,
    split among the points still running, so no point runs far past its
    stop.
    """
    save = functools.partial(_save_the_piece, run_dir)
    while True:
        units = _next_share_of_the_pool(points, run_dir, processes)
        if not units:
            return
        collect.run_units(
            units, measure_shot, on_unit_done=save, processes=processes
        )
        for point in points:
            point.count_the_pending(run_dir)


def _next_share_of_the_pool(
    points: list, run_dir: pathlib.Path, processes: int
) -> list:
    """The next pieces the pool runs: the running points' shares of it."""
    running = []
    for point in points:
        if point.tracker.stop_kind is None:
            running.append(point)
    if not running:
        return []
    share = processes / len(running)
    wanted = math.ceil(share)
    units = []
    for point in running:
        point_units = point.next_units(run_dir, wanted)
        units.extend(point_units)
    return units


def _say_the_point_stopped(point: PointCollection) -> None:
    """The progress line of a point that stopped, and why.

    The line names the point, since two points may share their metadata.
    A piece runs whole, so the shots of the stop's piece past its stop,
    and of any piece handed out beside it, ran and count nowhere; the
    line says how many.
    """
    shots = point.tracker.counts.shots
    reason = point.tracker.stop_kind.value
    line = f"{point.name}: {shots} shots done ({reason})"
    past_the_stop = point.next_seed - shots
    if past_the_stop > 0:
        line += f"; {past_the_stop} more ran past the stop"
    print(line, file=sys.stderr)


def _save_the_piece(
    run_dir: pathlib.Path,
    unit: collect.Unit,
    outcome: result_records.UnitOutcome,
) -> None:
    """One unit's measurements saved as its piece.

    The piece records the peak memory and the package versions of the
    process that ran it. An adaptive point's piece keeps its calibrator
    as the unit's shots left it.
    """
    point_id = unit.task.strong_id()
    rows = outcome.rows
    piece_facts = {
        "peak_memory_mb": outcome.peak_memory_mb,
        "packages": outcome.module_versions,
    }
    state = unit.task.online_threshold
    pieces.write(run_dir, point_id, unit.first_seed, rows, piece_facts, state)


def _say_the_threshold(calibrator) -> None:
    """The online threshold's one-line summary on stderr."""
    summary = calibrator.summary()
    threshold_db = _final_threshold_db(summary)
    summary_line = _threshold_summary_line(summary, threshold_db)
    print(summary_line, file=sys.stderr)


def _final_threshold_db(summary: dict) -> float:
    """The calibrator's final threshold, in the paper's decibels."""
    final_threshold_nats = summary["threshold"]
    return threshold_sources.nats_to_decibels(final_threshold_nats)


def _write_online_threshold_record(
    point: dict,
    calibrator,
    run_dir: pathlib.Path,
    swept: dict,
) -> None:
    """One online point's csv of its threshold's trajectory.

    Every audit and target move plus every 100th window; the calibrator's
    nats are converted to the paper's decibels.
    """
    _say_the_threshold(calibrator)
    summary = calibrator.summary()
    threshold_db = _final_threshold_db(summary)
    rows = _trajectory_rows(point, calibrator, summary, threshold_db)
    point_id = point["point_id"]
    record_path = pathlib.Path(run_dir) / f"online_threshold_{point_id}.csv"
    report.write_csv(rows, record_path, swept)


def _threshold_summary_row(point: dict, calibrator) -> dict:
    """The calibrator's counters at the point's end, its threshold in dB.

    Every counter calibrator.summary() holds, the ones the stderr line
    prints among them, with the final threshold converted from nats.
    """
    summary = calibrator.summary()
    threshold_db = _final_threshold_db(summary)
    row = {**point, **summary}
    del row["threshold"]
    row["threshold_db"] = threshold_db
    return row


def _trajectory_rows(
    point: dict, calibrator, summary: dict, threshold_db: float
) -> list:
    """The trajectory's rows, then the end row at the final threshold."""
    rows = []
    for window_count, threshold_nats, event in calibrator.trajectory:
        row_db = threshold_sources.nats_to_decibels(threshold_nats)
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
