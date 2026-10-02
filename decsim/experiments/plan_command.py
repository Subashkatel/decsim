"""`decsim run --slurm`: an experiment's batches, planned and submitted.

A batch is one written plan of pieces that Slurm arrays run. Each step
of the loop plans the next batch from what the saved pieces say,
submits one array per shape of job, and queues the next step behind
those arrays with afterany, so a task that died or ran out of time is
planned again, until every point has stopped. Before the first batch,
seed 0 of the cheapest point of each machine shape runs once in the
interpreter the jobs will use, so a broken import, build or config is
caught before anything is queued.

Between batches the plan decides what runs next, as OpenMC checks its
triggers between batches and extends the run by the ratio of the
uncertainty it has to the one it wants, squared (openmc
src/simulation.cpp, trigger.cpp). Per point, on its contiguous prefix of
saved pieces, read shot by shot as a local run reads it:

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
from the calibrator the piece before it saved. batches/<k>/plan.csv
holds the pieces and batches/<k>/tasks.csv each task's cores, memory and
time.

No account, partition or QOS is written: sbatch reads them from
SBATCH_ACCOUNT, SBATCH_PARTITION and SBATCH_QOS in the environment
(sbatch(1), "INPUT ENVIRONMENT VARIABLES"), and every job inherits the
submitting environment.
"""

import csv
import dataclasses
import getpass
import json
import math
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from typing import Optional

import decsim.collect as collect
import decsim.experiments.collect_command as collect_command
import decsim.experiments.collection as collection
import decsim.experiments.experiment as experiment
import decsim.experiments.pieces as pieces
import decsim.experiments.refusal as refusal
import decsim.experiments.run_folder as run_folder
import decsim.records.round_plans as round_plans

TASKS_FILE = "tasks.csv"
TASK_COLUMNS = ("task", "cores", "memory_mb", "hours", "estimated_core_hours")
# The margin on a measured peak memory, the "margin of one half" of the
# September 2026 plan (configs/experiments_2026_09/PLAN.md at d759e38f):
# a later piece of the point may hold more.
MEMORY_MARGIN = 1.5
SECONDS_PER_HOUR = 3600
ALLOW_DIRTY_VARIABLE = "ALLOW_DIRTY"
PYTHON_VARIABLE = "DECSIM_PYTHON"
SUBMIT_LIMIT_VARIABLE = "SUBMIT_LIMIT"
# Della's short QOS: 1,000 submitted jobs a user, each array task one.
DEFAULT_SUBMIT_LIMIT = 1000
# The values a shape keeps: each record's class, and the kind a row
# names when one class serves several kinds.
SHAPE_KEYS = (collect.RECORD_CLASS_KEY, "kind")
# The next step only plans and submits, so it asks for one core, 16 GB
# and twelve hours, what the loop's planning job has run with.
NEXT_STEP_SHAPE = (
    "--nodes",
    "1",
    "--ntasks",
    "1",
    "--cpus-per-task",
    "1",
    "--mem",
    "16G",
    "--time",
    "12:00:00",
)
# What a dry run names the arrays it did not submit, in the next step's
# dependency.
DRY_RUN_JOB_ID = "<array job id>"


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


@dataclasses.dataclass(frozen=True)
class SlurmRequest:
    """One `decsim run --slurm`: what runs, where, and in what jobs.

    run_file and results_dir are absolute, since every job of the loop
    reads them from wherever Slurm starts it. task_count is the most
    tasks a batch has; dry_run prints the sbatch lines and submits
    nothing.
    """

    run_file: pathlib.Path
    results_dir: pathlib.Path
    task_count: int
    job: JobShape
    dry_run: bool = False


def launch(
    run_file,
    out_dir: Optional[pathlib.Path],
    task_count: int,
    job: JobShape,
    *,
    dry_run: bool = False,
) -> None:
    """One step of the batch loop: check, plan the next batch, submit it.

    A tree git does not vouch for is refused first, before the results
    folder is made: out_dir, or a new dated one that every later step
    is handed. Before the first batch one shot per machine shape is
    run. A batch past the submit limit is refused before its first
    array. The next step is queued behind the batch's arrays; when every
    point has stopped there is no batch and the loop ends.
    """
    _refuse_a_shape_below_one(task_count, job)
    refuse_an_unnamed_tree()
    given_run_file = pathlib.Path(run_file)
    absolute_run_file = given_run_file.resolve()
    study = experiment.load(absolute_run_file)
    run_dir = run_folder.run_dir_for(study.name, out_dir)
    results_dir = run_dir.resolve()
    request = SlurmRequest(
        absolute_run_file, results_dir, task_count, job, dry_run
    )
    python = _job_python()
    is_first_batch = not pieces.batch_dirs(results_dir)
    if is_first_batch:
        _check_one_shot_per_shape(study, absolute_run_file, python)
    _plan_and_submit(request, python)


def _plan_and_submit(request: SlurmRequest, python: str) -> None:
    """The next batch planned, then its arrays and the next step queued."""
    batch_folder = plan_batch(
        request.run_file, request.results_dir, request.task_count, request.job
    )
    if batch_folder is None:
        print(f"every point has stopped; decsim status {request.results_dir}")
        return
    task_rows = _task_rows(batch_folder)
    _refuse_a_batch_past_the_submit_limit(batch_folder, task_rows)
    job_ids = _submit_the_arrays(batch_folder, task_rows, python, request)
    _submit_the_next_step(batch_folder, job_ids, python, request)


def plan_batch(
    run_file,
    results_dir: pathlib.Path,
    task_count: int,
    job: JobShape,
) -> Optional[pathlib.Path]:
    """The next batch's plan written in its folder; None when nothing is left.

    Every point is recorded first, and run.json names the run file, so
    the tasks that run the batch load it again and only read the
    records. A shape below one is refused before anything is written.
    """
    _refuse_a_shape_below_one(task_count, job)
    run_folder.refuse_another_tree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    given_run_file = pathlib.Path(run_file)
    absolute_run_file = given_run_file.resolve()
    study = experiment.load(absolute_run_file)
    planned = pieces.planned_pieces(results_dir)
    points = collect_command.recorded_points(results_dir, study)
    point_ids = [point.task.strong_id() for point in points]
    started_utc = run_folder.start_run(
        results_dir, absolute_run_file, point_ids
    )
    bundles, costs = _bundles_and_costs(results_dir, points, planned)
    run_folder.finish_run(
        results_dir, absolute_run_file, point_ids, started_utc
    )
    if not bundles:
        return None
    batch_folder = _new_batch_dir(results_dir)
    tasks = _dealt(bundles, costs, task_count)
    _write_plan(batch_folder, tasks)
    _write_tasks(batch_folder, tasks, costs, job)
    return batch_folder


def refuse_an_unnamed_tree() -> None:
    """A tree git does not vouch for is refused, unless ALLOW_DIRTY is set.

    Every task imports the tree as it stands when that task starts, so
    an edit or a commit landing while an array is still launching gives
    different tasks different code, and the pieces would name a commit
    none of them ran; a tree git cannot read cannot name its code at
    all. The reading is taken fresh, at submission and again in every
    task, and handed to the run records through TREE_DIRTY_VARIABLE,
    since a job's interpreter may have no git of its own. gem5 prints
    its version, build date, host and command line at every start for
    the same reason (gem5 src/python/m5/main.py:524-537).
    """
    checkout, commit, is_dirty = run_folder.fresh_tree_reading()
    dirty_text = _dirty_text(is_dirty)
    commit_text = commit or "unknown"
    print(f"decsim tree: {checkout}")
    print(f"decsim commit: {commit_text}, dirty: {dirty_text}")
    if is_dirty is not None:
        os.environ[run_folder.TREE_DIRTY_VARIABLE] = dirty_text
    if os.environ.get(ALLOW_DIRTY_VARIABLE):
        return
    if is_dirty:
        raise refusal.RefusalError(
            f"refusing to start: {checkout} has uncommitted changes, so a "
            "task would import whatever the tree holds when it starts; "
            "commit them, submit from a worktree pinned at a commit, or "
            f"set {ALLOW_DIRTY_VARIABLE}=1"
        )
    if is_dirty is None:
        raise refusal.RefusalError(
            f"refusing to start: git says nothing about {checkout}, so a "
            "task cannot name the code it ran; submit from a git checkout "
            f"pinned at a commit, or set {ALLOW_DIRTY_VARIABLE}=1"
        )


def _dirty_text(is_dirty: Optional[bool]) -> str:
    """The dirty flag as the launcher exports it: 1, 0 or unknown."""
    if is_dirty is None:
        return "unknown"
    return str(int(is_dirty))


def _job_python() -> str:
    """The interpreter every job runs: DECSIM_PYTHON, else this one."""
    named = os.environ.get(PYTHON_VARIABLE)
    if named:
        return named
    return sys.executable


def _check_one_shot_per_shape(
    study: experiment.Experiment, run_file: pathlib.Path, python: str
) -> None:
    """Seed 0 of each shape's cheapest point, run in the jobs' interpreter.

    The shot runs as `decsim run --only NAME --seed 0` in a scratch
    folder and is thrown away: it proves the jobs' interpreter imports
    decsim, builds the machine and runs it, before any array is queued.
    """
    for point in _cheapest_point_of_each_shape(study):
        with tempfile.TemporaryDirectory() as scratch:
            command = [
                python,
                "-m",
                "decsim",
                "run",
                str(run_file),
                "--only",
                point.name,
                "--seed",
                "0",
                "--out",
                scratch,
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
        _refuse_a_failed_check(point, python, completed)
        print(f"checked: one shot of {point.name} ran in {python}")


def _refuse_a_failed_check(point, python: str, completed) -> None:
    """A check shot that failed, refused with the last line it printed."""
    if completed.returncode == 0:
        return
    printed = completed.stderr.strip() or completed.stdout.strip()
    lines = printed.splitlines() or ["(nothing printed)"]
    raise refusal.RefusalError(
        f"the one-shot check of point {point.name} failed in {python}: "
        f"{lines[-1]}"
    )


def _cheapest_point_of_each_shape(study: experiment.Experiment) -> list:
    """Per machine shape, the point with the fewest QEC rounds a shot.

    A shape is the settings with every value but the records' classes
    and the rows' kinds left out, so points that differ only in numbers
    (a distance, an error rate) or in a workload made from them are one
    shape. Ties go to the point the experiment lists first.
    """
    cheapest_by_shape = {}
    for point in study.points:
        task = experiment.task_of(point)
        settings_json = collect.json_value(task.settings)
        shape = _shape_of(settings_json)
        shape_text = json.dumps(shape, sort_keys=True)
        rounds = run_folder.rounds_per_shot(task)
        earlier = cheapest_by_shape.get(shape_text)
        if earlier is None or rounds < earlier[0]:
            cheapest_by_shape[shape_text] = (rounds, point)
    return [point for _rounds, point in cheapest_by_shape.values()]


def _shape_of(value):
    """The json value with every leaf but a class or kind made None."""
    if isinstance(value, Mapping):
        return _shape_of_a_mapping(value)
    if isinstance(value, list):
        return [_shape_of(child) for child in value]
    return None


def _shape_of_a_mapping(value: Mapping) -> dict:
    """A mapping's shape: its keys, class and kind, children's shapes."""
    shape = {}
    for key, child in value.items():
        if key in SHAPE_KEYS:
            shape[key] = child
            continue
        shape[key] = _shape_of(child)
    return shape


def _task_rows(batch_folder: pathlib.Path) -> list:
    """tasks.csv's rows, as the plan wrote them."""
    tasks_path = batch_folder / TASKS_FILE
    with open(tasks_path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _refuse_a_batch_past_the_submit_limit(
    batch_folder: pathlib.Path, task_rows: list
) -> None:
    """A batch whose tasks and the user's queued jobs pass the limit.

    The QOS counts each array task as a job and rejects a submission
    past its limit, so a batch past it would be half submitted, leaving
    tasks no array holds; it is refused before the first array.
    """
    limit = _submit_limit()
    queued = _queued_job_count()
    task_count = len(task_rows)
    if task_count + queued <= limit:
        return
    batch_number = pieces.batch_number_of(batch_folder)
    raise refusal.RefusalError(
        f"refusing to submit: batch {batch_number} has {task_count} tasks "
        f"and {queued} of your jobs are queued, past the submit limit of "
        f"{limit} ({SUBMIT_LIMIT_VARIABLE}); run again with fewer --tasks, "
        "or wait for queued jobs to end"
    )


def _submit_limit() -> int:
    """SUBMIT_LIMIT, a whole number of jobs of at least one."""
    text = os.environ.get(SUBMIT_LIMIT_VARIABLE, str(DEFAULT_SUBMIT_LIMIT))
    if text.isdigit() and int(text) >= 1:
        return int(text)
    raise refusal.RefusalError(
        f"refusing to submit: {SUBMIT_LIMIT_VARIABLE} must be a whole "
        f"number of jobs of at least 1, got {text!r}"
    )


def _queued_job_count() -> int:
    """The user's queued and running jobs, each array task one; 0 off Slurm."""
    if shutil.which("squeue") is None:
        return 0
    user = getpass.getuser()
    command = ["squeue", "-h", "-r", "-u", user]
    completed = subprocess.run(command, capture_output=True, text=True)
    lines = completed.stdout.splitlines()
    return len(lines)


def _submit_the_arrays(
    batch_folder: pathlib.Path,
    task_rows: list,
    python: str,
    request: SlurmRequest,
) -> list:
    """One array per shape of job in tasks.csv; the job ids submitted.

    A Slurm array has one memory request, so the tasks of one cores,
    memory and hours share an array. Each array task runs its share of
    the plan, one process per core.
    """
    tasks_by_shape = {}
    for row in task_rows:
        shape = (int(row["cores"]), int(row["memory_mb"]), int(row["hours"]))
        shape_tasks = tasks_by_shape.setdefault(shape, [])
        shape_tasks.append(row["task"])
    job_ids = []
    for shape in sorted(tasks_by_shape):
        tasks = tasks_by_shape[shape]
        line = _array_line(batch_folder, shape, tasks, python, request)
        job_id = _submitted(line, request.dry_run, batch_folder, tasks)
        job_ids.append(job_id)
    return job_ids


def _array_line(
    batch_folder: pathlib.Path,
    shape: tuple,
    tasks: list,
    python: str,
    request: SlurmRequest,
) -> list:
    """The sbatch line of one array: its shape, its tasks, its task command."""
    cores, memory_mb, hours = shape
    batch_number = pieces.batch_number_of(batch_folder)
    wrapped = _task_command(python, request, batch_number)
    return [
        "sbatch",
        "--parsable",
        "--job-name",
        f"decsim-batch{batch_number}",
        "--array",
        ",".join(tasks),
        "--nodes",
        "1",
        "--ntasks",
        "1",
        "--cpus-per-task",
        str(cores),
        "--mem",
        f"{memory_mb}M",
        "--time",
        f"{hours}:00:00",
        "--output",
        f"{batch_folder}/%a/log.txt",
        "--wrap",
        wrapped,
    ]


def _task_command(python: str, request: SlurmRequest, batch_number: int) -> str:
    """What each array task runs: its share of the batch's plan."""
    task_command = [
        python,
        "-m",
        "decsim",
        "run",
        "--out",
        str(request.results_dir),
        "--batch",
        str(batch_number),
    ]
    task_text = shlex.join(task_command)
    # The array index and the cores are the job's own, so the shell Slurm
    # wraps the command in reads them when the task starts.
    return (
        f"{task_text} --task $SLURM_ARRAY_TASK_ID "
        "--processes $SLURM_CPUS_PER_TASK"
    )


def _submitted(
    line: list, dry_run: bool, batch_folder: pathlib.Path, tasks: list
) -> str:
    """The array submitted, or printed on a dry run; its job id.

    Slurm opens each task's log before the task runs, so its folder is
    made first; the task's run.json goes in the same folder.
    """
    if dry_run:
        line_text = shlex.join(line)
        print(line_text)
        return DRY_RUN_JOB_ID
    for task in tasks:
        task_folder = batch_folder / task
        task_folder.mkdir(exist_ok=True)
    return _sbatch(line)


def _submit_the_next_step(
    batch_folder: pathlib.Path,
    job_ids: list,
    python: str,
    request: SlurmRequest,
) -> None:
    """The next step of the loop, queued behind every array of this batch.

    afterany starts it however the arrays ended, so a task that died
    or ran out of time leaves seeds the next plan gives out again.
    """
    batch_number = pieces.batch_number_of(batch_folder)
    step_command = _step_command(python, request)
    dependency = ":".join(job_ids)
    line = [
        "sbatch",
        "--parsable",
        "--dependency",
        f"afterany:{dependency}",
        "--job-name",
        f"decsim-after-batch{batch_number}",
        "--output",
        f"{batch_folder}/next_step.log",
        *NEXT_STEP_SHAPE,
        "--wrap",
        shlex.join(step_command),
    ]
    if request.dry_run:
        line_text = shlex.join(line)
        print(line_text)
        return
    _sbatch(line)


def _step_command(python: str, request: SlurmRequest) -> list:
    """What the next step runs: this command again, with the same shape."""
    job = request.job
    return [
        python,
        "-m",
        "decsim",
        "run",
        str(request.run_file),
        "--slurm",
        "--out",
        str(request.results_dir),
        "--tasks",
        str(request.task_count),
        "--cores",
        str(job.cores),
        "--hours",
        str(job.hours),
        "--memory-mb",
        str(job.memory_mb),
    ]


def _sbatch(line: list) -> str:
    """One submission; the job id sbatch --parsable printed."""
    completed = subprocess.run(line, capture_output=True, text=True)
    if completed.returncode != 0:
        said = completed.stderr.strip()
        raise refusal.RefusalError(f"sbatch refused the submission: {said}")
    printed = completed.stdout.strip()
    # --parsable prints the job id, then ";cluster" on a federated Slurm
    fields = printed.split(";")
    job_id = fields[0]
    print(f"Submitted batch job {job_id}")
    return job_id


def _refuse_a_shape_below_one(task_count: int, job: JobShape) -> None:
    """A batch has a task, and a task a core, an hour and some memory."""
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


def _bundles_and_costs(
    results_dir: pathlib.Path, points: list, planned: dict
) -> tuple:
    """Every point's next pieces as the dealer's bundles, and its cost."""
    bundles = []
    costs = {}
    for point in points:
        point_id = point.task.strong_id()
        point_pieces = _next_pieces(results_dir, point, planned)
        point_bundles = _bundles_of(point, point_pieces)
        bundles.extend(point_bundles)
        costs[point_id] = _cost_of(results_dir, point)
    return bundles, costs


def _next_pieces(
    results_dir: pathlib.Path,
    point: collect_command.PointCollection,
    planned: dict,
) -> list:
    """One point's pieces for the next batch (the module's three steps)."""
    point_id = point.task.strong_id()
    point.count_the_saved(results_dir)
    if point.tracker.stop_kind is not None:
        return []
    saved_items = point.saved.items()
    saved = set(saved_items)
    planned_ranges = planned.get(point_id, [])
    missing = _unsaved_seeds(point.saved, planned_ranges)
    if missing:
        return _pieces_of(point_id, missing)
    highest_seed = _highest_seed(planned_ranges, saved)
    shot_count = _extension(point, point.tracker.counts, highest_seed)
    ranges = _cut(highest_seed, shot_count, point.piece_shots)
    return _pieces_of(point_id, ranges)


def _unsaved_seeds(saved: dict, planned_ranges: list) -> list:
    """The planned seeds no saved piece holds, each once, as (first, count).

    A planned piece a task never saved comes back with its own seeds;
    one saved under another cut, by a local run, comes back only for
    the seeds that cut left out. Two batches may plan the same seeds
    under different cuts, a piece and then its unsaved part, so a seed
    an earlier range already gave back is not given again.
    """
    missing = []
    covered = dict(saved)
    for first_seed, count in planned_ranges:
        unsaved = pieces.uncovered_ranges(covered, first_seed, count)
        missing.extend(unsaved)
        covered.update(unsaved)
    return missing


def _highest_seed(planned_ranges: list, saved: set) -> int:
    """One past the highest seed any batch planned or any piece holds."""
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
    shot, so a batch's last piece may be short.
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


def _pieces_of(point_id: str, ranges: list) -> list:
    """Planned pieces of one point, one per (first seed, count)."""
    planned = []
    for first_seed, count in ranges:
        piece = round_plans.PlannedPiece(point_id, first_seed, count)
        planned.append(piece)
    return planned


def _cost_of(results_dir: pathlib.Path, point) -> PointCost:
    """The point's measured seconds a shot and peak memory, if any."""
    point_id = point.task.strong_id()
    folders = pieces.folders_of(results_dir, [point_id])
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


def _new_batch_dir(results_dir: pathlib.Path) -> pathlib.Path:
    """batches/<k>/, k one past the last batch that holds a plan."""
    existing = pieces.batch_dirs(results_dir)
    batch_number = 1
    if existing:
        batch_number = pieces.batch_number_of(existing[-1]) + 1
    batch_folder = pieces.batch_dir(results_dir, batch_number)
    batch_folder.mkdir(parents=True, exist_ok=True)
    return batch_folder


def _write_plan(batch_folder: pathlib.Path, tasks: list) -> None:
    """plan.csv: a row per piece, its task first."""
    rows = []
    for task, task_pieces in enumerate(tasks):
        for piece in task_pieces:
            row = {"task": task, **dataclasses.asdict(piece)}
            rows.append(row)
    plan_path = batch_folder / pieces.PLAN_FILE
    _write_rows_atomically(plan_path, pieces.PLAN_COLUMNS, rows)


def _write_tasks(
    batch_folder: pathlib.Path, tasks: list, costs: dict, job: JobShape
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
    tasks_path = batch_folder / TASKS_FILE
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
    with run_folder.staged_replacement(path) as staging:
        with open(staging, "w", newline="") as handle:
            writer = csv.DictWriter(handle, columns)
            writer.writeheader()
            writer.writerows(rows)
