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
import decsim.experiments.stim_batch as stim_batch
import decsim.records.results as result_records
import decsim.records.round_plans as round_plans

# What a yaml's top-level `sampling` key names: a module that runs a
# point's units in place of the machine, saves them as pieces, reads a
# piece's shots back and folds the pieces. A point with none runs the
# machine.
SAMPLINGS = {"stim_batch": stim_batch}


@dataclasses.dataclass
class PointCollection:
    """One point's collection as it runs: its next seed and its prefix.

    tracker reads the prefix shot by shot off the saved pieces'
    shots.csv, as the report does (collection.PrefixTracker), so the
    collector stops on the shot the report's row stops on. saved maps
    the first seed of each piece saved before this collect to its
    count, whatever collection cut it, and planned lists the (first
    seed, count) of the pieces round plans named. pending holds the (first seed,
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

    def next_units(self, experiment_dir: pathlib.Path, wanted: int) -> list:
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
            unit = self._hand_out(experiment_dir, count)
            if unit is None and not units:
                self.count_the_pending(experiment_dir)
            if unit is not None:
                units.append(unit)
        return units

    def count_the_pending(self, experiment_dir: pathlib.Path) -> None:
        """The pending pieces' shots onto the prefix, until it stops."""
        point_id = self.task.strong_id()
        for first_seed, count in self.pending:
            if self.tracker.stop_kind is not None:
                break
            folder = pieces.piece_dir(
                experiment_dir, point_id, first_seed, count
            )
            for row in _shot_rows_of(self.task, folder):
                self.tracker.add(row)
        self.pending = []
        if self.tracker.stop_kind is not None:
            _say_the_point_stopped(self)

    def count_the_saved(self, experiment_dir: pathlib.Path) -> None:
        """The saved pieces from the next seed on, up to a gap, counted.

        A plan and a status read a point's prefix this way, shot by
        shot, as a collect reads it before it runs a piece.
        """
        prefix = pieces.contiguous_ranges(self.saved, self.next_seed)
        for first_seed, count in prefix:
            self._make_pending(first_seed, count)
        self.count_the_pending(experiment_dir)

    def rule(self) -> collection_module.PointRule:
        """What the point's summary reads its prefix by."""
        is_adaptive = self.task.online_threshold is not None
        return collection_module.PointRule(
            self.settings, is_adaptive, self.rounds_per_shot
        )

    def task_as_its_last_piece_left_it(
        self, experiment_dir: pathlib.Path
    ) -> collect.Task:
        """The task, an adaptive point's calibrator the last piece's state."""
        if self.task.online_threshold is None or self.last_piece is None:
            return self.task
        point_id = self.task.strong_id()
        first_seed, count = self.last_piece
        folder = pieces.piece_dir(experiment_dir, point_id, first_seed, count)
        return _task_as_the_piece_left_it(self.task, folder)

    def _hand_out(
        self, experiment_dir: pathlib.Path, count: int
    ) -> Optional[collect.Unit]:
        """The next piece made pending; its unit, or None when it is saved.

        A saved piece that starts at the next seed is taken whole, and a
        new one ends where the next saved piece starts or a planned piece
        starts or ends, so a collect run again under a raised cap or
        target, or beside a round's tasks, runs no seed twice.
        """
        first_seed = self.next_seed
        saved_count = self.saved.get(first_seed)
        if saved_count is not None:
            self._make_pending(first_seed, saved_count)
            return None
        seeds_free = self._seeds_before_a_boundary(first_seed)
        new_count = min(count, seeds_free)
        task = self.task_as_its_last_piece_left_it(experiment_dir)
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


def run_sweep(tasks: list, shots: int, *, processes: int = 1) -> list:
    """Seeds 0 to shots - 1 of every task of a sweep, measured.

    A point named by more than one block runs once, and nothing is
    written. The measure handed to the pool is a partial of a
    module-level function, so a worker process can unpickle it.
    """
    measure_shot = _shot_measure(tasks, None)
    return collect.collect(
        tasks,
        shots,
        measure_shot,
        _say_the_online_threshold,
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
    configuration_id = run_folder.configuration_id(config)
    owned_points = recorded_points(experiment_dir, {configuration_id: [config]})
    report_dir = run_folder.combined_folder(experiment_dir, config)
    points = [point for _configuration_id, point in owned_points]
    unique = [point.task for point in points]
    point_ids = _point_ids(unique)
    started_utc = run_folder.start_run(config, report_dir, point_ids)
    first_task = unique[0]
    _echo_description(config, first_task.settings, report_dir)
    measure_shot = _shot_measure(unique, report_dir)
    _collect_until_stopped(
        points, experiment_dir, configuration_id, measure_shot, processes
    )
    folders = pieces.folders_of(experiment_dir, point_ids)
    rows = write_the_run_folder(experiment_dir, folders, point_ids, report_dir)
    run_folder.finish_run(config, report_dir, point_ids, started_utc)
    return report_dir, rows


def write_the_run_folder(
    experiment_dir: pathlib.Path,
    folders: list,
    point_ids: list,
    report_dir: pathlib.Path,
) -> list:
    """The points' saved pieces folded into the run folder, and what they give.

    Every file comes from what the experiment folder recorded, not from
    what this collect ran or what a yaml makes now: each point's
    collection and rounds from its resolved/ record, the rest from its
    pieces, folders, as pieces.folders_of gave them. Every file reads
    that one list, so a piece saved while the fold runs is in none of
    them. The fold is built whole in a staging folder beside report_dir
    and moved in only then, in place of the last fold, so a fold that is
    refused (two pieces of a point with different columns) leaves the
    last one as it was, and nothing of a point the folder no longer
    holds stays. So a collect that found its pieces saved, or a status
    after a yaml changed, writes the folder whole: the online
    thresholds' trajectories from their prefixes' last states, the
    residence table from the pieces' traced shots, and the figures.
    Returns the summary rows.
    """
    combined_dir = report_dir.parent
    combined_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=combined_dir, prefix=".") as staged:
        staging = pathlib.Path(staged)
        rows = _fold_into_the_staging(
            experiment_dir, folders, point_ids, staging
        )
        run_folder.publish_the_fold(staging, report_dir)
    plots.plots(report_dir)
    return rows


def run_planned(
    plan_path: pathlib.Path, task_number: int, *, processes: int = 1
) -> None:
    """One task of a round's plan, its pieces run and saved.

    The experiment folder is the one the round's folder sits in, and its
    points were recorded by the plan, so the task only reads them. A
    piece already saved is skipped, so a task run again runs only what
    it had not saved. Each piece records its round and task, and the
    task's manifest goes in round<k>/<task>/.
    """
    round_dir = plan_path.parent
    experiment_dir = round_dir.parent
    task_pieces = _pieces_of_the_task(plan_path, task_number)
    configurations = run_folder.recorded_configurations(experiment_dir)
    point_ids = [piece.point_id for piece in task_pieces]
    task_dir = round_dir / str(task_number)
    started_utc = run_folder.start_run(None, task_dir, point_ids)
    round_number = pieces.round_number_of(round_dir)
    facts = {"round": round_number, "task": task_number}
    for configuration_id, config_pieces in _by_configuration(task_pieces):
        configs = configurations[configuration_id]
        _run_the_planned_pieces(
            configs, experiment_dir, config_pieces, facts, processes
        )
    run_folder.finish_run(None, task_dir, point_ids, started_utc)


def recorded_points(experiment_dir: pathlib.Path, configs_by_id: dict) -> list:
    """Every point of every configuration, recorded once all are accepted.

    configs_by_id maps a configuration id to its yamls, which may split
    its sweep, one file a distance; a point two of them name is one
    point. Every yaml is resolved and every point's record built first,
    which runs its build, so a point the build refuses stops the run
    before any shot. A point two configurations reach, when their yamls
    differ only in a setting both sweeps set, is under the one given
    last, and refused if they collect it two ways, since a point stops
    by one rule. Only then are the records and the configuration lines
    written, so a refused plan or collect leaves the experiment folder
    as it was. Returns (configuration id, point) pairs.
    """
    owners = {}
    for configuration_id, configs in configs_by_id.items():
        for resolved in _resolved_points(configuration_id, configs):
            point_id = resolved.record["id"]
            earlier = owners.get(point_id, resolved)
            _check_one_collection(earlier, resolved)
            owners[point_id] = resolved
    owned = owners.values()
    accepted = list(owned)
    _write_the_records(experiment_dir, accepted, configs_by_id)
    planned = pieces.planned_pieces(experiment_dir)
    owned_points = []
    for resolved in accepted:
        point = _point_collection(experiment_dir, resolved, planned)
        owned_points.append((resolved.configuration_id, point))
    return owned_points


def _task_as_the_piece_left_it(
    task: collect.Task, folder: pathlib.Path
) -> collect.Task:
    """The task with the online calibrator the piece in folder saved."""
    state = pieces.read_state(folder)
    return dataclasses.replace(task, online_threshold=state)


def _recorded_rule(record: dict) -> collection_module.PointRule:
    """What a point's summary reads its prefix by, as its record says."""
    facts = record["experiment"]
    settings = collection_module.CollectionSettings(**facts["collection"])
    return collection_module.PointRule(
        settings, facts["adaptive"], record["rounds_per_shot"]
    )


def _fold_into_the_staging(
    experiment_dir: pathlib.Path,
    folders: list,
    point_ids: list,
    staging: pathlib.Path,
) -> list:
    """Every file of the fold written into staging; the summary rows."""
    records = run_folder.resolved_by_point(experiment_dir)
    sampling = _sampling_of_the_points(records, point_ids)
    if sampling is not None:
        return sampling.fold_into_the_staging(
            experiment_dir, folders, point_ids, staging
        )
    swept = run_folder.swept_values(experiment_dir, point_ids)
    rules = {}
    for point_id in point_ids:
        record = records[point_id]
        rules[point_id] = _recorded_rule(record)
        _write_the_recorded_trajectory(
            experiment_dir, folders, record, staging, swept
        )
    seeds_by_point = pieces.seed_ranges_of(folders)
    rows = report.fold_pieces(
        experiment_dir, folders, point_ids, seeds_by_point, staging, rules
    )
    residence_rows = residence.rows_in(folders)
    residence.write_residence(residence_rows, staging, swept)
    return rows


def _write_the_recorded_trajectory(
    experiment_dir: pathlib.Path,
    folders: list,
    record: dict,
    report_dir: pathlib.Path,
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
    folder = pieces.piece_dir(experiment_dir, point_id, first_seed, count)
    calibrator = pieces.read_state(folder)
    algorithm = facts["algorithm"]
    _write_online_threshold_record(
        point_id, algorithm, calibrator, report_dir, swept
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


def _by_configuration(task_pieces: list) -> list:
    """The pieces grouped by configuration id, first seen first."""
    grouped = {}
    for piece in task_pieces:
        config_pieces = grouped.setdefault(piece.configuration_id, [])
        config_pieces.append(piece)
    grouped_pairs = grouped.items()
    return list(grouped_pairs)


def _run_the_planned_pieces(
    configs: list,
    experiment_dir: pathlib.Path,
    config_pieces: list,
    facts: dict,
    processes: int,
) -> None:
    """One configuration's planned pieces run and saved; saved ones skipped.

    Traces go in the configuration's run folder, named by its first
    yaml, as a collect of it writes them. The independent pieces share
    the pool; then an online point's pieces run one after another in
    seed order, each from the calibrator the piece before it saved.
    """
    config = configs[0]
    point_tasks = _point_tasks_of(configs)
    unique = _unique_tasks(point_tasks)
    task_by_point = {task.strong_id(): task for task in unique}
    units = _planned_units(
        experiment_dir, config_pieces, task_by_point, configs
    )
    report_dir = run_folder.combined_folder(experiment_dir, config)
    report_dir.mkdir(parents=True, exist_ok=True)
    measure_shot = _shot_measure(unique, report_dir)
    save = _planned_piece_saver(experiment_dir, config, units, facts)
    independent, online = _independent_and_online(units)
    collect.run_units(
        independent,
        measure_shot,
        on_unit_done=save,
        processes=processes,
        unit_runner=_run_unit,
    )
    for unit in online:
        resumed = _resumed_from_the_piece_before(experiment_dir, unit)
        collect.run_units([resumed], measure_shot, on_unit_done=save)


def _planned_piece_saver(
    experiment_dir: pathlib.Path,
    config: experiment.ExperimentConfig,
    units: list,
    facts: dict,
):
    """The unit callback that saves a planned unit as its piece.

    Each piece names its configuration beside the round's facts, and
    counts its rounds by its point's record, which the plan wrote.
    """
    records = run_folder.resolved_by_point(experiment_dir)
    rounds_by_point = {}
    for unit in units:
        point_id = unit.task.strong_id()
        rounds_by_point[point_id] = records[point_id]["rounds_per_shot"]
    configuration_id = run_folder.configuration_id(config)
    piece_facts = {"configuration_id": configuration_id, **facts}
    return functools.partial(
        _save_the_piece, experiment_dir, piece_facts, rounds_by_point
    )


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
    experiment_dir: pathlib.Path,
    config_pieces: list,
    task_by_point: dict,
    configs: list,
) -> list:
    """The planned pieces' seeds no saved piece holds, as work units.

    A piece saved whole, or its seeds saved under another cut by a plain
    collect, runs nothing; one partly saved runs the seeds left, so no
    seed is saved twice.
    """
    units = []
    for piece in config_pieces:
        task = task_by_point.get(piece.point_id)
        if task is None:
            _refuse_a_point_gone_from_its_yamls(piece, configs)
        point_folders = pieces.folders_of(experiment_dir, [piece.point_id])
        saved = pieces.saved_counts(point_folders)
        unsaved = pieces.uncovered_ranges(saved, piece.first_seed, piece.count)
        for first_seed, count in unsaved:
            unit = collect.Unit(task, first_seed, count)
            units.append(unit)
    return units


def _refuse_a_point_gone_from_its_yamls(
    piece: round_plans.PlannedPiece, configs: list
) -> None:
    """The sentence for a planned point its yamls no longer make."""
    files = ", ".join(str(config.config_files[0]) for config in configs)
    raise refusal.RefusalError(
        f"the plan's point {piece.point_id} is no point of {files} now; a "
        "yaml or its maker changed since the plan was written, so plan "
        "again"
    )


def _resumed_from_the_piece_before(
    experiment_dir: pathlib.Path, unit: collect.Unit
) -> collect.Unit:
    """An online point's unit, its calibrator as the piece before it left it.

    The plan deals an online point's pieces to one task in seed order,
    so the piece before is saved, by an earlier round or just now.
    """
    if unit.first_seed == 0:
        return unit
    point_id = unit.task.strong_id()
    point_folders = pieces.folders_of(experiment_dir, [point_id])
    saved = pieces.saved_counts(point_folders)
    for first_seed, count in saved.items():
        if first_seed + count != unit.first_seed:
            continue
        folder = pieces.piece_dir(experiment_dir, point_id, first_seed, count)
        task = _task_as_the_piece_left_it(unit.task, folder)
        return dataclasses.replace(unit, task=task)
    raise refusal.RefusalError(
        f"the point {point_id} has no saved piece ending at seed "
        f"{unit.first_seed}, whose calibrator its piece starts from; plan "
        "again"
    )


def _point_tasks_of(configs: list) -> list:
    """Every (task, collection) pair of the yamls, file by file."""
    point_tasks = []
    for config in configs:
        config_tasks = config.point_tasks()
        point_tasks.extend(config_tasks)
    return point_tasks


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


@dataclasses.dataclass(frozen=True)
class _ResolvedPoint:
    """One point as its yaml resolves it, its record built and not written."""

    configuration_id: str
    task: collect.Task
    settings: collection_module.CollectionSettings
    record: dict


def _resolved_points(configuration_id: str, configs: list) -> list:
    """One configuration's points, each once, checked and its record built.

    A point one yaml sweeps twice, or two of them, is one point with one
    collection, and an online point stops at max_shots alone; either is
    refused otherwise.
    """
    made = []
    for config in configs:
        for task, collection in config.point_tasks():
            made.append((config, task, collection))
    point_tasks = [(task, settings) for _config, task, settings in made]
    collections = _collection_by_point(point_tasks)
    resolved = {}
    for config, task, _collection in made:
        point_id = task.strong_id()
        if point_id in resolved:
            continue
        settings = collections[point_id]
        _check_the_stop_of_an_online_point(task, settings)
        _check_the_sampling(task)
        sections = config.resolved_sections(task.metadata)
        facts = _experiment_facts(task, configuration_id, settings)
        record = run_folder.point_record(task, None, sections, facts)
        resolved[point_id] = _ResolvedPoint(
            configuration_id, task, settings, record
        )
    resolved_points = resolved.values()
    return list(resolved_points)


def _write_the_records(
    experiment_dir: pathlib.Path, accepted: list, configs_by_id: dict
) -> None:
    """The accepted points' records, and every yaml's configuration line."""
    for resolved in accepted:
        run_folder.write_point_record(
            experiment_dir, resolved.task, resolved.record
        )
    for configs in configs_by_id.values():
        for config in configs:
            run_folder.record_configuration(experiment_dir, config)


def _check_one_collection(
    earlier: _ResolvedPoint, later: _ResolvedPoint
) -> None:
    """Two configurations of one point give it one collection, or refused."""
    if earlier.settings == later.settings:
        return
    metadata = collect.metadata_text(later.task.metadata)
    earlier_text = earlier.settings.text()
    later_text = later.settings.text()
    earlier_id = earlier.configuration_id[:8]
    later_id = later.configuration_id[:8]
    raise refusal.RefusalError(
        f"the point {metadata} is in two configurations, {earlier_id} "
        f"and {later_id}, that collect it two ways, {earlier_text} and "
        f"{later_text}; a point stops by one rule"
    )


def _point_collection(
    experiment_dir: pathlib.Path, resolved: _ResolvedPoint, planned: dict
) -> PointCollection:
    """A recorded point's collection state, its pieces sized by its rounds.

    planned maps a point id to its planned pieces, as round plans name
    them (pieces.planned_pieces).
    """
    point_id = resolved.record["id"]
    rounds_per_shot = resolved.record["rounds_per_shot"]
    settings = resolved.settings
    piece_shots = settings.piece_shots(rounds_per_shot)
    point_folders = pieces.folders_of(experiment_dir, [point_id])
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
    task: collect.Task,
    configuration_id: str,
    settings: collection_module.CollectionSettings,
) -> dict:
    """What folding the point needs besides its pieces.

    Its configuration, which the last yaml to record it belongs to; its
    collection, which reads its prefix; whether its threshold learns
    online; and the kind of the tier that decodes its windows, which
    names its trajectory's rows.
    """
    facts = {
        "configuration_id": configuration_id,
        "collection": dataclasses.asdict(settings),
        "adaptive": task.online_threshold is not None,
        "algorithm": measure.active_decoder_kind(task.settings),
    }
    if task.sampling is not None:
        facts["sampling"] = task.sampling
    return facts


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
    experiment_dir: pathlib.Path,
    configuration_id: str,
    measure_shot,
    processes: int,
) -> None:
    """Every point's pieces, a round at a time, until each has stopped.

    A round hands the pool about as many pieces as it has processes,
    shared among the points still running, so no point runs far past
    its stop. Each piece names its configuration.
    """
    facts = {"configuration_id": configuration_id}
    rounds_by_point = {}
    for point in points:
        point_id = point.task.strong_id()
        rounds_by_point[point_id] = point.rounds_per_shot
    save = functools.partial(
        _save_the_piece, experiment_dir, facts, rounds_by_point
    )
    while True:
        units = _next_round(points, experiment_dir, processes)
        if not units:
            return
        collect.run_units(
            units,
            measure_shot,
            on_unit_done=save,
            processes=processes,
            unit_runner=_run_unit,
        )
        for point in points:
            point.count_the_pending(experiment_dir)


def _next_round(
    points: list, experiment_dir: pathlib.Path, processes: int
) -> list:
    """The next round's pieces: the running points' shares of the pool."""
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
        point_units = point.next_units(experiment_dir, wanted)
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
    experiment_dir: pathlib.Path,
    facts: dict,
    rounds_by_point: dict,
    unit: collect.Unit,
    outcome: result_records.UnitOutcome,
) -> None:
    """One unit's measurements saved as its piece.

    facts are the lines every piece of the run takes in piece.json: its
    configuration id, and a planned piece's round and task. The piece
    counts its rounds, since a shot's cost grows with its rounds, and
    the peak memory of the process that ran it. An adaptive point's
    piece keeps its calibrator as the unit's shots left it. A point's
    sampling saves its own piece.
    """
    if unit.task.sampling is not None:
        sampling = SAMPLINGS[unit.task.sampling]
        sampling.write_piece(
            experiment_dir, facts, rounds_by_point, unit, outcome
        )
        return
    point_id = unit.task.strong_id()
    rows = outcome.rows
    rounds_per_shot = rounds_by_point[point_id]
    piece_facts = {
        **facts,
        "rounds": rounds_per_shot * len(rows),
        "peak_memory_mb": outcome.peak_memory_mb,
    }
    state = unit.task.online_threshold
    pieces.write(
        experiment_dir, point_id, unit.first_seed, rows, piece_facts, state
    )


def _run_unit(unit: collect.Unit, measure_shot) -> result_records.UnitOutcome:
    """One unit run by the machine, or by the sampling its point names.

    It is a module-level function, so a pool can pickle it.
    """
    if unit.task.sampling is None:
        return collect.run_unit(unit, measure_shot)
    sampling = SAMPLINGS[unit.task.sampling]
    return sampling.run_unit(unit, measure_shot)


def _shot_rows_of(task: collect.Task, folder: pathlib.Path):
    """A saved piece's shot rows in seed order, as its sampling kept them."""
    if task.sampling is not None:
        sampling = SAMPLINGS[task.sampling]
        return sampling.shot_rows(folder)
    shots_path = folder / "shots.csv"
    return fold.row_stream(shots_path)


def _sampling_of_the_points(records: dict, point_ids: list):
    """The sampling module the points' records name, None for the machine.

    The key is a configuration's, so its points share it.
    """
    if not point_ids:
        return None
    facts = records[point_ids[0]]["experiment"]
    name = facts.get("sampling")
    if name is None:
        return None
    return SAMPLINGS[name]


def _check_the_sampling(task: collect.Task) -> None:
    """A point's sampling is a row of SAMPLINGS that can run it, or refused."""
    if task.sampling is None:
        return
    sampling = SAMPLINGS.get(task.sampling)
    if sampling is None:
        rows = sorted(SAMPLINGS)
        raise refusal.RefusalError(
            f"sampling {task.sampling!r} is not a row of its table; the "
            f"rows are {rows}"
        )
    sampling.check_task(task)


def _echo_description(config, settings, run_dir: pathlib.Path) -> None:
    """The resolved experiment, before the first shot, as gem5 dumps it.

    settings is the first point's, whose sections the lines name.
    """
    description = experiment.resolved_description(config, settings)
    description.append(f"run dir: {run_dir}\n")
    description_text = "\n".join(description)
    print(description_text, file=sys.stderr)


def _say_the_online_threshold(task: collect.Task) -> None:
    """A finished point's online threshold summary, when it learns one."""
    calibrator = task.online_threshold
    if calibrator is None:
        return
    _say_the_threshold(calibrator)


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
