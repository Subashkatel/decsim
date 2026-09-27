"""`decsim plan`: the next round of an experiment's pieces, dealt to tasks.

A round is one Slurm array running one written plan. Between rounds the
plan reads what the pieces say and decides what runs next, as OpenMC
checks its triggers between batches and extends the run by the ratio of
the uncertainty it has to the one it wants, squared (openmc
src/simulation.cpp, trigger.cpp). Per point, on its contiguous prefix of
saved pieces, read shot by shot as a collect reads it:

1. A point whose prefix has stopped by its collection's rule gets
   nothing.
2. A planned piece whose seeds no piece holds, a task that died, is
   planned again with its own seeds, before anything new; its point
   waits for it.
3. Otherwise the point is extended above its highest planned seed to
   shots x (target / failures), the shots its failure rate so far needs
   for the target (the ratio squared, since the relative error goes as
   one over the square root of the failures); with no failure yet, to
   twice its shots; with nothing saved yet, by one piece; by at least
   one piece, then cut to the shot cap and to the time cap at the
   prefix's seconds a shot, so its last piece may be short.

Every piece is sized by the collection's piece_rounds, so a piece takes
about as long at any history length. The pieces are then dealt to tasks,
the longest first to the least loaded task (Graham's LPT rule, SIAM J.
Appl. Math. 17, 1969), each costed by its point's measured seconds a
shot, or by rounds where nothing was measured yet. An online point's
pieces are dealt together, in seed order, to one task, since each starts
from the calibrator the piece before it saved. round<k>/plan.csv
holds the pieces and round<k>/tasks.csv each task's cores, memory and
time, which slurm/round.sh submits.
"""

import argparse
import csv
import dataclasses
import math
import pathlib
import sys
from typing import Optional

import decsim.experiments.collect_command as collect_command
import decsim.experiments.collection as collection
import decsim.experiments.experiment as experiment
import decsim.experiments.pieces as pieces
import decsim.experiments.refusal as refusal
import decsim.experiments.run_folder as run_folder
import decsim.records.round_plans as round_plans

TASKS_FILE = "tasks.csv"
TASK_COLUMNS = ("task", "cores", "memory_mb", "hours", "estimated_core_hours")
# The margin on a measured peak memory, configs/experiments_2026_09/PLAN.md's
# "margin of one half": a later piece of the point may hold more.
MEMORY_MARGIN = 1.5
SECONDS_PER_HOUR = 3600


@dataclasses.dataclass(frozen=True)
class JobShape:
    """What one task asks Slurm for: cores, walltime, and a first memory.

    memory_mb is the memory one piece is given before its point has a
    measured peak; a task runs `cores` pieces at once.
    """

    cores: int
    hours: int
    memory_mb: int


@dataclasses.dataclass(frozen=True)
class PointCost:
    """What a point's saved pieces measured: seconds a shot, peak memory.

    None where nothing is saved yet.
    """

    rounds_per_shot: int
    seconds_per_shot: Optional[float] = None
    peak_memory_mb: Optional[float] = None


def main(argv: list) -> None:
    """Plan the next round and print where it went and how to submit it."""
    parser = _parser()
    parsed = parser.parse_args(argv)
    job = JobShape(parsed.cores, parsed.hours, parsed.memory_mb)
    experiment_dir = pathlib.Path(parsed.out)
    round_dir = plan_round(parsed.configs, experiment_dir, parsed.tasks, job)
    if round_dir is None:
        print(
            f"every point has stopped; decsim status {experiment_dir}",
            file=sys.stderr,
        )
        return
    plan_path = round_dir / pieces.PLAN_FILE
    print(plan_path)
    round_number = pieces.round_number_of(round_dir)
    print(f"submit it: slurm/round.sh {experiment_dir} {round_number}")


def plan_round(
    config_paths: list,
    experiment_dir: pathlib.Path,
    task_count: int,
    job: JobShape,
) -> Optional[pathlib.Path]:
    """The next round's plan written in its folder; None when nothing is left.

    Every configuration's points are recorded first, so the tasks that
    run the round only read them. Yamls of one configuration id, which
    split its sweep, are planned as one configuration, and a point two
    configurations reach is planned once (collect_command.recorded_points).
    A shape below one is refused before anything is written.
    """
    _refuse_a_shape_below_one(task_count, job)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    planned = pieces.planned_pieces(experiment_dir)
    bundles = []
    costs = {}
    configs_by_id = _configs_by_id(config_paths)
    owned_points = collect_command.recorded_points(
        experiment_dir, configs_by_id
    )
    for configuration_id, point in owned_points:
        point_id = point.task.strong_id()
        point_pieces = _next_pieces(
            experiment_dir, configuration_id, point, planned
        )
        point_bundles = _bundles_of(point, point_pieces)
        bundles.extend(point_bundles)
        costs[point_id] = _cost_of(experiment_dir, point)
    if not bundles:
        return None
    round_dir = _new_round_dir(experiment_dir)
    tasks = _dealt(bundles, costs, task_count)
    _write_plan(round_dir, tasks)
    _write_tasks(round_dir, tasks, costs, job)
    return round_dir


def _parser() -> argparse.ArgumentParser:
    """The plan command's arguments."""
    parser = argparse.ArgumentParser(prog="decsim plan")
    parser.add_argument("configs", nargs="+", help="the experiment's yamls")
    parser.add_argument("--out", required=True, help="the experiment folder")
    parser.add_argument(
        "--tasks", type=int, required=True, help="the most tasks a round has"
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=4,
        help="pieces a task runs at once, one per core",
    )
    parser.add_argument(
        "--hours", type=int, default=24, help="a task's walltime"
    )
    parser.add_argument(
        "--memory-mb",
        type=int,
        default=4096,
        help="one piece's memory before its point has a measured peak",
    )
    return parser


def _refuse_a_shape_below_one(task_count: int, job: JobShape) -> None:
    """A round has a task, and a task a core, an hour and some memory."""
    shape = {
        "--tasks": task_count,
        "--cores": job.cores,
        "--hours": job.hours,
        "--memory-mb": job.memory_mb,
    }
    for flag, value in shape.items():
        if value < 1:
            raise refusal.RefusalError(
                f"{flag} must be at least 1, got {value}"
            )


def _configs_by_id(config_paths: list) -> dict:
    """The yamls, each loaded by its absolute path, grouped by configuration id.

    The absolute path is what configurations.csv records, so a task
    started in another folder loads the same file.
    """
    configs_by_id = {}
    for config_path in config_paths:
        given_path = pathlib.Path(config_path)
        absolute_path = given_path.resolve()
        config = experiment.load_experiment(absolute_path)
        configuration_id = run_folder.configuration_id(config)
        configs = configs_by_id.setdefault(configuration_id, [])
        configs.append(config)
    return configs_by_id


def _next_pieces(
    experiment_dir: pathlib.Path,
    configuration_id: str,
    point: collect_command.PointCollection,
    planned: dict,
) -> list:
    """One point's pieces for the next round (the module's three steps)."""
    point_id = point.task.strong_id()
    point.count_the_saved(experiment_dir)
    if point.tracker.stop_kind is not None:
        return []
    saved_items = point.saved.items()
    saved = set(saved_items)
    planned_ranges = planned.get(point_id, [])
    missing = _unsaved_seeds(point.saved, planned_ranges)
    if missing:
        return _pieces_of(configuration_id, point_id, missing)
    highest_seed = _highest_seed(planned_ranges, saved)
    shot_count = _extension(point, point.tracker.counts, highest_seed)
    ranges = _cut(highest_seed, shot_count, point.piece_shots)
    return _pieces_of(configuration_id, point_id, ranges)


def _unsaved_seeds(saved: dict, planned_ranges: list) -> list:
    """The planned seeds no saved piece holds, as (first seed, count).

    A planned piece a task never saved comes back with its own seeds;
    one saved under another cut, by a plain collect, comes back only for
    the seeds that cut left out.
    """
    missing = []
    for first_seed, count in planned_ranges:
        unsaved = pieces.uncovered_ranges(saved, first_seed, count)
        missing.extend(unsaved)
    return missing


def _highest_seed(planned_ranges: list, saved: set) -> int:
    """One past the highest seed any round planned or any piece holds."""
    highest_seed = 0
    for first_seed, count in [*planned_ranges, *saved]:
        end = first_seed + count
        highest_seed = max(highest_seed, end)
    return highest_seed


def _extension(
    point, counts: collection.PrefixCounts, highest_seed: int
) -> int:
    """The shots to plan above highest_seed: to the wanted total, capped.

    At least one piece, since the point has not stopped, then cut to
    the caps: the shot cap, and the time cap at the prefix's seconds a
    shot, so a round's last piece may be short.
    """
    settings = point.settings
    wanted_shot_count = _wanted_shots(settings, counts, point.piece_shots)
    wanted_shot_count = max(wanted_shot_count, settings.min_shots)
    shot_count = wanted_shot_count - highest_seed
    shot_count = max(shot_count, point.piece_shots)
    end_seed = highest_seed + shot_count
    capped_end_seed = _capped_end_seed(settings, counts, end_seed)
    shot_count = capped_end_seed - highest_seed
    return max(shot_count, 0)


def _capped_end_seed(
    settings: collection.CollectionSettings,
    counts: collection.PrefixCounts,
    end_seed: int,
) -> int:
    """end_seed, or the lower end the shot cap or the time cap allows."""
    ends = [end_seed]
    if settings.max_shots is not None:
        ends.append(settings.max_shots)
    time_capped = _shots_in_the_time_cap(settings, counts)
    if time_capped is not None:
        ends.append(time_capped)
    return min(ends)


def _wanted_shots(
    settings: collection.CollectionSettings,
    counts: collection.PrefixCounts,
    piece_shots: int,
) -> int:
    """The total shots the point needs by what its prefix shows so far.

    A point with no target runs to its shot cap. Nothing saved: one
    piece. No failure yet: twice what it has. Otherwise the target over
    the failure fraction, OpenMC's ratio squared.
    """
    if settings.max_failures is None and settings.max_shots is not None:
        return settings.max_shots
    if counts.shots == 0:
        return piece_shots
    if settings.max_failures is None or counts.failures == 0:
        return 2 * counts.shots
    needed = counts.shots * settings.max_failures / counts.failures
    return math.ceil(needed)


def _shots_in_the_time_cap(
    settings: collection.CollectionSettings, counts: collection.PrefixCounts
) -> Optional[int]:
    """The shots the time cap allows at the prefix's seconds a shot."""
    if settings.max_core_seconds is None or counts.core_seconds <= 0:
        return None
    seconds_per_shot = counts.core_seconds / counts.shots
    allowed = settings.max_core_seconds / seconds_per_shot
    return math.ceil(allowed)


def _cut(first_seed: int, shot_count: int, piece_shots: int) -> list:
    """Seeds [first_seed, first_seed + shot_count) as (first, count) pieces."""
    ranges = []
    end = first_seed + shot_count
    for piece_first in range(first_seed, end, piece_shots):
        remaining_shot_count = end - piece_first
        count = min(piece_shots, remaining_shot_count)
        ranges.append((piece_first, count))
    return ranges


def _pieces_of(configuration_id: str, point_id: str, ranges: list) -> list:
    """Planned pieces of one point, one per (first seed, count)."""
    planned = []
    for first_seed, count in ranges:
        piece = round_plans.PlannedPiece(
            configuration_id, point_id, first_seed, count
        )
        planned.append(piece)
    return planned


def _cost_of(experiment_dir: pathlib.Path, point) -> PointCost:
    """The point's measured seconds a shot and peak memory, if any."""
    point_id = point.task.strong_id()
    folders = pieces.folders_of(experiment_dir, [point_id])
    shot_count = 0
    core_seconds = 0.0
    peaks = []
    for folder in folders:
        piece = pieces.read_piece(folder)
        shot_count += piece["count"]
        core_seconds += piece["core_seconds"]
        peaks.append(piece["peak_memory_mb"])
    peak_memory_mb = None
    if peaks:
        peak_memory_mb = max(peaks)
    seconds_per_shot = None
    if shot_count > 0 and core_seconds > 0:
        seconds_per_shot = core_seconds / shot_count
    return PointCost(point.rounds_per_shot, seconds_per_shot, peak_memory_mb)


def _bundles_of(point, point_pieces: list) -> list:
    """What the dealer hands out whole: each piece, or all of an online point's.

    An online point's pieces run one after another, each from the
    calibrator the piece before it saved, so one task runs them all, in
    seed order.
    """
    if not point_pieces:
        return []
    if point.task.online_threshold is not None:
        return [point_pieces]
    return [[piece] for piece in point_pieces]


def _dealt(bundles: list, costs: dict, task_count: int) -> list:
    """The bundles dealt to at most task_count tasks, longest first (LPT).

    Returns each task's pieces, task 0 first; a bundle's cost is the sum
    of its pieces', and a piece's is its point's seconds a shot times
    its count, or its rounds where no point was measured, priced by the
    measured points' seconds a round.
    """
    seconds_per_round = _measured_seconds_per_round(costs)
    priced = []
    for position, bundle in enumerate(bundles):
        cost = _bundle_cost(bundle, costs, seconds_per_round)
        priced.append((-cost, position, bundle))
    priced.sort()
    used_task_count = min(task_count, len(bundles))
    tasks = [[] for _ in range(used_task_count)]
    loads = [0.0] * used_task_count
    for negative_cost, _position, bundle in priced:
        lightest = loads.index(min(loads))
        tasks[lightest].extend(bundle)
        loads[lightest] -= negative_cost
    return tasks


def _bundle_cost(
    bundle: list, costs: dict, seconds_per_round: Optional[float]
) -> float:
    """The summed cost of a bundle's pieces."""
    cost = 0.0
    for piece in bundle:
        cost += _piece_cost(piece, costs, seconds_per_round)
    return cost


def _measured_seconds_per_round(costs: dict) -> Optional[float]:
    """Seconds a QEC round over every measured point, None if none is."""
    seconds = 0.0
    round_count = 0
    for cost in costs.values():
        if cost.seconds_per_shot is None:
            continue
        seconds += cost.seconds_per_shot
        round_count += cost.rounds_per_shot
    if round_count == 0:
        return None
    return seconds / round_count


def _piece_cost(
    piece: round_plans.PlannedPiece,
    costs: dict,
    seconds_per_round: Optional[float],
) -> float:
    """A piece's seconds, or its rounds when nothing is measured."""
    cost = costs[piece.point_id]
    if cost.seconds_per_shot is not None:
        return cost.seconds_per_shot * piece.count
    round_count = cost.rounds_per_shot * piece.count
    if seconds_per_round is None:
        return round_count
    return seconds_per_round * round_count


def _new_round_dir(experiment_dir: pathlib.Path) -> pathlib.Path:
    """round<k>/, k one past the last round that holds a plan."""
    existing = pieces.round_dirs(experiment_dir)
    round_number = 1
    if existing:
        round_number = pieces.round_number_of(existing[-1]) + 1
    round_dir = experiment_dir / f"{pieces.ROUND_PREFIX}{round_number}"
    round_dir.mkdir(exist_ok=True)
    return round_dir


def _write_plan(round_dir: pathlib.Path, tasks: list) -> None:
    """plan.csv: a row per piece, its task first."""
    rows = []
    for task, task_pieces in enumerate(tasks):
        for piece in task_pieces:
            row = {"task": task, **dataclasses.asdict(piece)}
            rows.append(row)
    plan_path = round_dir / pieces.PLAN_FILE
    _write_rows_atomically(plan_path, pieces.PLAN_COLUMNS, rows)


def _write_tasks(
    round_dir: pathlib.Path, tasks: list, costs: dict, job: JobShape
) -> None:
    """tasks.csv: each task's cores, memory, walltime and estimated time.

    The memory is `cores` pieces at the largest measured peak among its
    points with MEMORY_MARGIN, or the job's first memory for a point
    with none. The estimate is empty until every one of its points has
    measured seconds.
    """
    rows = []
    for task, task_pieces in enumerate(tasks):
        piece_memory_mb = _piece_memory_mb(task_pieces, costs, job)
        estimate = _estimated_core_hours(task_pieces, costs)
        row = {
            "task": task,
            "cores": job.cores,
            "memory_mb": job.cores * piece_memory_mb,
            "hours": job.hours,
            "estimated_core_hours": estimate,
        }
        rows.append(row)
    tasks_path = round_dir / TASKS_FILE
    _write_rows_atomically(tasks_path, TASK_COLUMNS, rows)


def _piece_memory_mb(task_pieces: list, costs: dict, job: JobShape) -> int:
    """The memory one of the task's pieces is given, whole megabytes."""
    largest = 0.0
    for piece in task_pieces:
        peak = costs[piece.point_id].peak_memory_mb
        memory_mb = job.memory_mb
        if peak is not None:
            memory_mb = peak * MEMORY_MARGIN
        largest = max(largest, memory_mb)
    return math.ceil(largest)


def _estimated_core_hours(task_pieces: list, costs: dict) -> Optional[float]:
    """The task's measured core hours, None if a point has no measure."""
    seconds = 0.0
    for piece in task_pieces:
        seconds_per_shot = costs[piece.point_id].seconds_per_shot
        if seconds_per_shot is None:
            return None
        seconds += seconds_per_shot * piece.count
    return seconds / SECONDS_PER_HOUR


def _write_rows_atomically(
    path: pathlib.Path, columns: tuple, rows: list
) -> None:
    """A csv written under a temporary name and renamed into place."""
    staging = run_folder.staging_path(path)
    with open(staging, "w", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(rows)
    staging.replace(path)
