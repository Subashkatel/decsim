"""`decsim run`: every point of one experiment, each until it stops.

The experiment says what runs; this module only orchestrates. It runs
each point's seeds in order, a piece at a time (decsim.collect), and
saves each piece's additive facts in the results folder the moment it
ends (pieces). A point stops by its collection's rule on the contiguous
prefix of its seeds (collection), and no piece past the stop is
started. Then the pieces are folded into the results folder, one row
per point, beside the figures and the residence and wait table of the
traced shots (report, run_folder, plots, residence). A piece already
saved is counted and not run again, so a killed collect resumes where
it stopped; rerunning the same experiment reproduces the same rows
(only the wall-clock column varies), and so does running it with a
process pool, as long as each point stops at the same piece.
"""

import dataclasses
import functools
import math
import pathlib
import sys
import tempfile
from typing import Optional

import decsim.collect as collect
import decsim.escalation.settings as escalation_settings
import decsim.experiments.collection as collection_module
import decsim.experiments.experiment as experiment
import decsim.experiments.fold as fold
import decsim.experiments.measure as measure
import decsim.experiments.pieces as pieces
import decsim.experiments.plots as plots
import decsim.experiments.refusal as refusal
import decsim.experiments.report as report
import decsim.experiments.residence as residence
import decsim.experiments.run_folder as run_folder
import decsim.machine as machine_module
import decsim.records.results as result_records
import decsim.records.round_plans as round_plans
import decsim.settings as machine_settings


@dataclasses.dataclass
class PointCollection:
    """One point's collection as it runs: its next seed and its prefix.

    tracker reads the prefix shot by shot off the saved pieces'
    shots.csv, as the report does (collection.PrefixTracker), so the
    collector stops on the shot the report's row stops on. saved maps
    the first seed of each piece saved before this collect to its
    count, whatever collection cut it, and planned lists the (first
    seed, count) of the pieces batch plans named. pending holds the (first seed,
    count) of the pieces handed out past the counted prefix, in seed
    order; they are counted once they are all saved, so the prefix
    stays contiguous whatever order the pool ends them in. last_piece
    is the (first seed, count) of the piece handed out last.

    An adaptive point's calibrator learns over its shots in order, so
    its pieces run one at a time, each from the calibrator state the
    piece before it saved (design 6.5), whether that one ran in this
    collect or in one that was killed.
    """

    task: collect.Task
    settings: collection_module.CollectionSettings
    piece_shots: int
    rounds_per_shot: int
    saved: dict
    planned: list = dataclasses.field(default_factory=list)
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
        """The pending pieces' shots onto the prefix, until it stops."""
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

    def count_the_saved(self, run_dir: pathlib.Path) -> None:
        """The saved pieces from the next seed on, up to a gap, counted.

        A plan and a status read a point's prefix this way, shot by
        shot, as a collect reads it before it runs a piece.
        """
        prefix = pieces.contiguous_ranges(self.saved, self.next_seed)
        for first_seed, count in prefix:
            self._make_pending(first_seed, count)
        self.count_the_pending(run_dir)

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
        new one ends where the next saved piece starts or a planned piece
        starts or ends, so a collect run again under a raised cap or
        target, or beside a batch's tasks, runs no seed twice.
        """
        first_seed = self.next_seed
        saved_count = self.saved.get(first_seed)
        if saved_count is not None:
            self._make_pending(first_seed, saved_count)
            return None
        seeds_free = self._seeds_before_a_boundary(first_seed)
        new_count = min(count, seeds_free)
        task = self.task_as_its_last_piece_left_it(run_dir)
        self._make_pending(first_seed, new_count)
        return collect.Unit(task, first_seed, new_count)

    def _make_pending(self, first_seed: int, count: int) -> None:
        self.next_seed = first_seed + count
        self.pending.append((first_seed, count))
        self.last_piece = (first_seed, count)

    def _seeds_before_a_boundary(self, first_seed: int) -> float:
        """The seeds from first_seed to the next piece's edge, or infinity.

        An edge is where a saved piece starts, or where a planned piece
        starts or ends.
        """
        edges = list(self.saved)
        for planned_first, planned_count in self.planned:
            planned_end = planned_first + planned_count
            edges.append(planned_first)
            edges.append(planned_end)
        later_edges = [edge for edge in edges if edge > first_seed]
        if not later_edges:
            return math.inf
        return min(later_edges) - first_seed

    def _next_piece_count(self) -> int:
        """The next piece's shots: a piece, or what is left below max_shots."""
        max_shots = self.settings.max_shots
        if max_shots is None:
            return self.piece_shots
        seeds_left = max_shots - self.next_seed
        remaining = max(seeds_left, 0)
        return min(self.piece_shots, remaining)


def run_experiment(
    run_file,
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

    run_dir is the results folder, and only names the one point to
    collect, every point when None. Each piece is saved when it ends,
    and a piece already saved is counted and skipped, so a killed
    run started again runs only what it had not saved. run.json lists
    every point of the experiment, the order the fold writes its rows
    in whichever points ran. Returns the folded rows.
    """
    _check_processes(processes)
    run_folder.refuse_another_tree(run_dir)
    every_id = _experiment_point_ids(study)
    chosen = _chosen_points(study, only)
    points = recorded_points(run_dir, chosen)
    point_ids = _point_ids(points)
    saved_pieces = pieces.folders_of(run_dir, point_ids)
    report.refuse_pieces_of_another_tree(saved_pieces)
    run_folder.accept_raised_stop_rules(run_dir, run_file, every_id)
    started_utc = run_folder.start_run(run_dir, run_file, every_id)
    _echo_description(chosen, run_file, run_dir)
    measure_shot = _shot_measure(points, run_dir)
    _collect_until_stopped(points, run_dir, measure_shot, processes)
    folded_ids = run_folder.recorded_point_ids(run_dir, every_id)
    folders = pieces.folders_of(run_dir, folded_ids)
    rows = fold_the_folder(run_dir, folded_ids, folders)
    run_folder.finish_run(run_dir, run_file, every_id, started_utc)
    return rows


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

    log and trace override the point's observation for this shot, as
    gem5's --debug-flags and --debug-file set what the config script did
    not (src/python/m5/main.py:280, 299). The shot writes its results
    folder: run.json, the run file, the point's machine.json and
    workload, and the shot's files (run_folder.write_shot). A shot
    replayed into a folder that recorded its point keeps that record,
    which holds what the collection knew of it, and a point the folder
    recorded with other settings is refused. Returns the lines the
    command prints.
    """
    point = study.points[0]
    task = experiment.task_of(point)
    settings = _with_observation(task.settings, log, trace)
    task = dataclasses.replace(task, settings=settings)
    shot_settings = task.shot_settings()
    machine = _built_machine(shot_settings, seed, run_file)
    run_dir = run_folder.run_dir_for(study.name, out_dir)
    run_folder.refuse_another_tree(run_dir)
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

    point_ids are every recorded point, in the rows' order
    (run_folder.recorded_point_ids), and folders their pieces, read
    once (pieces.folders_of), so a piece saved while the fold runs is
    in none of the files. Every file comes from what the folder
    recorded, not from what this run ran or what a run file makes now:
    each point's collection and rounds from its machine.json, the rest
    from its pieces. The
    fold is built whole in a staging folder and moved in only then, in
    place of the last fold, so a fold that is refused (two pieces of a
    point with different columns) leaves the last one as it was. Then
    each record names the seeds its pieces hold, and the figure is
    drawn.
    """
    with tempfile.TemporaryDirectory(dir=run_dir, prefix=".") as staged:
        staging = pathlib.Path(staged)
        rows = _fold_into_the_staging(run_dir, folders, point_ids, staging)
        run_folder.publish_the_fold(staging, run_dir)
    seeds_by_point = pieces.seed_ranges_of(folders)
    run_folder.record_seeds(run_dir, seeds_by_point)
    plots.plots(run_dir)
    return rows


def run_planned(
    run_dir: pathlib.Path,
    batch_number: int,
    task_number: int,
    *,
    processes: int = 1,
) -> None:
    """One task of a batch's plan, its pieces run and saved.

    run_dir is the results folder. The plan recorded its points, so the
    task loads the run file run.json names and only reads the records.
    A piece already saved is skipped, so a task run again runs only what
    it had not saved. Each piece records its batch and task, and the
    task's run.json goes in batches/<k>/<task>/.
    """
    _check_processes(processes)
    run_dir = pathlib.Path(run_dir)
    batch_folder = pieces.batch_dir(run_dir, batch_number)
    plan_path = batch_folder / pieces.PLAN_FILE
    task_pieces = _pieces_of_the_task(plan_path, task_number)
    run_file = run_folder.recorded_run_file(run_dir)
    study = experiment.load(run_file)
    point_ids = [piece.point_id for piece in task_pieces]
    task_dir = batch_folder / str(task_number)
    started_utc = run_folder.start_run(task_dir, None, point_ids)
    facts = {"batch": batch_number, "task": task_number}
    _run_the_planned_pieces(study, run_dir, task_pieces, facts, processes)
    run_folder.finish_run(task_dir, None, point_ids, started_utc)


def recorded_points(
    run_dir: pathlib.Path, study: experiment.Experiment
) -> list:
    """Every point of the experiment, recorded once all are accepted.

    Every point's record is built first, which runs its build, so a
    point the build refuses stops the run before any shot, and a point
    whose name the folder holds for another machine is refused. Only
    then are the records written, so a refused plan or collect leaves
    the results folder as it was. Returns each point's collection
    state, in the experiment's order.
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
    planned = pieces.planned_pieces(run_dir)
    points = []
    for resolved_point in resolved:
        point = _point_collection(run_dir, resolved_point, planned)
        points.append(point)
    return points


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
    """Every file of the fold written into staging; the rows."""
    records = run_folder.point_records(run_dir)
    swept = run_folder.swept_values(run_dir, point_ids)
    rules = {}
    for point_id in point_ids:
        record = records[point_id]
        rules[point_id] = collection_module.PointRule.from_record(record)
        _write_the_recorded_trajectory(run_dir, folders, record, staging, swept)
    rows = report.fold_pieces(run_dir, folders, point_ids, staging, rules)
    residence_rows = residence.rows_in(folders)
    residence.write_residence(residence_rows, staging, swept)
    return rows


def _write_the_recorded_trajectory(
    run_dir: pathlib.Path,
    folders: list,
    record: dict,
    staging: pathlib.Path,
    swept: dict,
) -> None:
    """An online point's trajectory, from the state its prefix ended on."""
    facts = record["experiment"]
    if not facts["adaptive"]:
        return
    point_id = record["id"]
    point_folders = pieces.point_folders(folders, point_id)
    saved = pieces.saved_counts(point_folders)
    prefix = pieces.contiguous_ranges(saved, 0)
    if not prefix:
        return
    first_seed, count = prefix[-1]
    folder = pieces.piece_dir(run_dir, point_id, first_seed, count)
    calibrator = pieces.read_state(folder)
    algorithm = facts["algorithm"]
    _write_online_threshold_record(
        point_id, algorithm, calibrator, staging, swept
    )


def _pieces_of_the_task(plan_path: pathlib.Path, task_number: int) -> list:
    """The plan's pieces dealt to one task; a task with none is refused."""
    task_pieces = []
    for task, piece in pieces.read_plan(plan_path):
        if task == task_number:
            task_pieces.append(piece)
    if not task_pieces:
        raise refusal.RefusalError(
            f"{plan_path} deals no piece to task {task_number}"
        )
    return task_pieces


def _run_the_planned_pieces(
    study: experiment.Experiment,
    run_dir: pathlib.Path,
    task_pieces: list,
    facts: dict,
    processes: int,
) -> None:
    """The task's planned pieces run and saved; saved ones skipped.

    The independent pieces share the pool; then an online point's
    pieces run one after another in seed order, each from the
    calibrator the piece before it saved.
    """
    task_by_point = {}
    for point in study.points:
        task = experiment.task_of(point)
        task_by_point[task.strong_id()] = task
    units = _planned_units(run_dir, task_pieces, task_by_point)
    tasks = task_by_point.values()
    measure_shot = _shot_measure_of_tasks(list(tasks), run_dir)
    save = functools.partial(_save_the_piece, run_dir, facts)
    independent, online = _independent_and_online(units)
    collect.run_units(
        independent, measure_shot, on_unit_done=save, processes=processes
    )
    for unit in online:
        resumed = _resumed_from_the_piece_before(run_dir, unit)
        collect.run_units([resumed], measure_shot, on_unit_done=save)


def _independent_and_online(units: list) -> tuple:
    """The units split, since only independent shots may run in any order.

    An online piece starts from the calibrator its piece before saved, so
    it runs after that piece, one at a time; the rest share the pool.
    """
    independent = []
    online = []
    for unit in units:
        if unit.task.online_threshold is None:
            independent.append(unit)
        else:
            online.append(unit)
    return independent, online


def _planned_units(
    run_dir: pathlib.Path, task_pieces: list, task_by_point: dict
) -> list:
    """The planned pieces' seeds no saved piece holds, as work units.

    A piece saved whole, or its seeds saved under another cut by a plain
    collect, runs nothing; one partly saved runs the seeds left, so no
    seed is saved twice.
    """
    units = []
    for piece in task_pieces:
        task = task_by_point.get(piece.point_id)
        if task is None:
            _refuse_a_point_gone_from_the_run_file(run_dir, piece)
        point_folders = pieces.folders_of(run_dir, [piece.point_id])
        saved = pieces.saved_counts(point_folders)
        unsaved = pieces.uncovered_ranges(saved, piece.first_seed, piece.count)
        for first_seed, count in unsaved:
            unit = collect.Unit(task, first_seed, count)
            units.append(unit)
    return units


def _refuse_a_point_gone_from_the_run_file(
    run_dir: pathlib.Path, piece: round_plans.PlannedPiece
) -> None:
    """The sentence for a planned point its run file no longer makes."""
    run_file = run_folder.recorded_run_file(run_dir)
    raise refusal.RefusalError(
        f"the plan's point {piece.point_id} is no point of {run_file} now; "
        "the run file or a maker it calls changed since the plan was "
        "written, so plan again"
    )


def _resumed_from_the_piece_before(
    run_dir: pathlib.Path, unit: collect.Unit
) -> collect.Unit:
    """An online point's unit, its calibrator as the piece before it left it.

    The plan deals an online point's pieces to one task in seed order,
    so the piece before is saved, by an earlier batch or just now.
    """
    if unit.first_seed == 0:
        return unit
    point_id = unit.task.strong_id()
    point_folders = pieces.folders_of(run_dir, [point_id])
    saved = pieces.saved_counts(point_folders)
    for first_seed, count in saved.items():
        if first_seed + count != unit.first_seed:
            continue
        folder = pieces.piece_dir(run_dir, point_id, first_seed, count)
        task = _task_as_the_piece_left_it(unit.task, folder)
        return dataclasses.replace(unit, task=task)
    raise refusal.RefusalError(
        f"the point {point_id} has no saved piece ending at seed "
        f"{unit.first_seed}, whose calibrator its piece starts from; plan "
        "again"
    )


def _shot_measure(points: list, run_dir: pathlib.Path):
    """The measure every shot of these points' collections runs through."""
    tasks = [point.task for point in points]
    return _shot_measure_of_tasks(tasks, run_dir)


def _shot_measure_of_tasks(tasks: list, run_dir: pathlib.Path):
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


def _built_machine(
    settings: machine_settings.MachineSettings,
    seed: int,
    run_file: pathlib.Path,
) -> machine_module.Machine:
    """The shot's machine; a build refusal one sentence naming the file."""
    try:
        return machine_module.Machine.build(settings, seed)
    except ValueError as refused:
        raise refusal.RefusalError(f"{run_file}: {refused}") from refused


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


def _point_ids(points: list) -> list:
    """The points' ids in the experiment's order."""
    point_ids = []
    for point in points:
        point_id = point.task.strong_id()
        point_ids.append(point_id)
    return point_ids


@dataclasses.dataclass(frozen=True)
class _ResolvedPoint:
    """One point, its task and collection, its record built and not written."""

    task: collect.Task
    settings: collection_module.CollectionSettings
    record: dict


def _resolved_points(study: experiment.Experiment) -> list:
    """Each point checked and its record built, in the experiment's order.

    Two points of one id would share their pieces, and an online point
    stops at max_shots alone; either is refused.
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
        _check_the_stop_of_an_online_point(task, settings)
        facts = _experiment_facts(task, settings)
        record = run_folder.point_record(
            point.name, task, None, point.sections, facts
        )
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
    run_folder.record_point(run_dir, point.name, task, seeds, point.sections)


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
    run_dir: pathlib.Path, resolved: _ResolvedPoint, planned: dict
) -> PointCollection:
    """A recorded point's collection state, its pieces sized by its rounds.

    planned maps a point id to its planned pieces, as batch plans name
    them (pieces.planned_pieces). A point whose shots send no detector
    data has no rounds to size a piece by or to score, and is refused
    here, before any shot runs, as sinter's Task refuses a task it cannot
    count when it is built (sinter/_data/_task.py:127-149).
    """
    point_id = resolved.record["id"]
    rounds_per_shot = resolved.record["rounds_per_shot"]
    if rounds_per_shot == 0:
        metadata = collect.metadata_text(resolved.task.metadata)
        raise refusal.RefusalError(
            f"the point {metadata} runs no operation that sends detector "
            "data, so its shots have no rounds to size a piece by or to "
            "score; decsim run needs a workload whose operations emit "
            "detector data (decsim run times the others)"
        )
    settings = resolved.settings
    piece_shots = settings.piece_shots(rounds_per_shot)
    point_folders = pieces.folders_of(run_dir, [point_id])
    saved = pieces.saved_counts(point_folders)
    point_planned = planned.get(point_id, [])
    return PointCollection(
        resolved.task,
        settings,
        piece_shots,
        rounds_per_shot,
        saved,
        point_planned,
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


def _check_the_stop_of_an_online_point(
    task: collect.Task, settings: collection_module.CollectionSettings
) -> None:
    """An online point stops at max_shots alone; any other stop is refused.

    Its shots are not independent draws, since each one's threshold
    learned from those before it, so a failure count has no interval a
    target stop could rest on, and a time cap would end its learning
    wherever the machine was fast.
    """
    if task.online_threshold is None:
        return
    has_other_stops = settings.max_failures is not None
    if settings.max_core_seconds is not None:
        has_other_stops = True
    if settings.max_shots is None or has_other_stops:
        _refuse_an_online_stop(task)


def _refuse_an_online_stop(task: collect.Task) -> None:
    """The sentence for an online point given a stop it cannot keep."""
    metadata = collect.metadata_text(task.metadata)
    raise refusal.RefusalError(
        f"the point {metadata} calibrates its threshold online, so its "
        "shots are not independent draws and it stops at max_shots alone; "
        "its collection sets max_shots and neither max_failures nor "
        "max_core_seconds"
    )


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
    save = functools.partial(_save_the_piece, run_dir, {})
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

    A piece runs whole, so the shots of the stop's piece past its stop,
    and of any piece handed out beside it, ran and count nowhere; the
    line says how many.
    """
    metadata = collect.metadata_text(point.task.metadata)
    shots = point.tracker.counts.shots
    reason = point.tracker.stop_kind.value
    line = f"{metadata}: {shots} shots done ({reason})"
    past_the_stop = point.next_seed - shots
    if past_the_stop > 0:
        line += f"; {past_the_stop} more ran past the stop"
    print(line, file=sys.stderr)


def _save_the_piece(
    run_dir: pathlib.Path,
    facts: dict,
    unit: collect.Unit,
    outcome: result_records.UnitOutcome,
) -> None:
    """One unit's measurements saved as its piece.

    facts are the lines every piece of the run takes in piece.json: a
    planned piece's batch and task, none for a local run. The piece
    counts the rounds its shots ran, since a shot's cost grows with its
    rounds, and the peak memory and the package versions of the process
    that ran it. An adaptive
    point's piece keeps its calibrator as the unit's shots left it.
    """
    point_id = unit.task.strong_id()
    rows = outcome.rows
    rounds = sum(row.executed_rounds for row in rows)
    piece_facts = {
        **facts,
        "rounds": rounds,
        "peak_memory_mb": outcome.peak_memory_mb,
        "packages": outcome.module_versions,
    }
    state = unit.task.online_threshold
    pieces.write(run_dir, point_id, unit.first_seed, rows, piece_facts, state)


def _echo_description(
    study: experiment.Experiment,
    run_file: pathlib.Path,
    run_dir: pathlib.Path,
) -> None:
    """The resolved experiment, before the first shot, on stderr."""
    description = experiment.description(study, run_file)
    description.append(f"run dir: {run_dir}\n")
    description_text = "\n".join(description)
    print(description_text, file=sys.stderr)


def _say_the_threshold(calibrator) -> None:
    """The online threshold's one-line summary on stderr."""
    summary = calibrator.summary()
    threshold_db = _final_threshold_db(summary)
    summary_line = _threshold_summary_line(summary, threshold_db)
    print(summary_line, file=sys.stderr)


def _final_threshold_db(summary: dict) -> float:
    """The calibrator's final threshold, in the paper's decibels."""
    final_threshold_nats = summary["threshold"]
    return escalation_settings.nats_to_decibels(final_threshold_nats)


def _write_online_threshold_record(
    point_id: str,
    algorithm,
    calibrator,
    run_dir: pathlib.Path,
    swept: dict,
) -> None:
    """One online point's csv of its threshold's trajectory.

    Every audit and target move, plus every 100th window, and a summary
    line on stderr. The gap unit inside the calibrator is nats; the csv
    converts to the paper's decibels. The file is named by the point's
    id, as its resolved/ record is, and its rows carry the point's id,
    swept values and algorithm, as every other csv's rows do.
    """
    _say_the_threshold(calibrator)
    summary = calibrator.summary()
    threshold_db = _final_threshold_db(summary)
    point = report.point_columns((point_id, algorithm))
    rows = _trajectory_rows(point, calibrator, summary, threshold_db)
    record_path = pathlib.Path(run_dir) / f"online_threshold_{point_id}.csv"
    report.write_csv(rows, record_path, swept)


def _trajectory_rows(
    point: dict, calibrator, summary: dict, threshold_db: float
) -> list:
    """The trajectory's rows, then the end row at the final threshold."""
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
