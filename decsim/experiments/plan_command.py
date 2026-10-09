"""`decsim run --slurm`: fixed-shot pieces packed into Slurm jobs, then a fold.

gem5 MultiSim's list-then-run-one-id pattern, with the work packed
first. Every task stops at a fixed shot count and gives an estimate of
a shot's core seconds (CollectionSettings.core_seconds_per_shot), so
its work is known before launch. The launcher records every task,
which runs each build, so a refused task queues nothing; cuts each
task's unsaved seeds into pieces of at most a twentieth of a core's
budget; and packs them onto cores longest first (Graham, "Bounds on
multiprocessing timing anomalies", SIAM J. Appl. Math. 17, 1969). A
core's budget is USE_TARGET of the --hours limit, the share of its
walltime Princeton Research Computing asked our jobs to use. The pieces
keep the folders a local run writes, so the fold and a resume work
unchanged: submitting again re-packs only the pieces not yet saved.

Three cores a job: Della's short QOS runs at most 400 jobs and 1000
cores of one user at once (sacctmgr show qos, 2026-10-09), so three is
the fewest cores a job with which 333 running jobs reach the core
limit; two would stop at 800 cores, and a smaller job waits less for a
slot. --hours sets the job count: fewer hours, more and shorter jobs.

Slurm runs an array's jobs "with identical parameters" (sbatch(1),
--array; https://slurm.schedmd.com/job_array.html: "All jobs must have
the same initial options (e.g., size, time limit)"), so every job asks
the walltime of the busiest core. The pieces are dealt over every core
of every job at once, so no two cores differ by more than one piece, a
twentieth of a budget. run.sbatch is the array, jobs.json lists the
pieces of each job i, and fold.sbatch waits on the array with afterany
(sbatch(1), --dependency). No account, partition or QOS is written:
sbatch reads SBATCH_ACCOUNT, SBATCH_PARTITION and SBATCH_QOS from the
environment (sbatch(1)).
"""

import dataclasses
import heapq
import math
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Optional, Union

import decsim.experiments.collect as collect
import decsim.experiments.collect_command as collect_command
import decsim.experiments.collection as collection_module
import decsim.experiments.experiment as experiment
import decsim.experiments.pieces as pieces
import decsim.experiments.refusal as refusal
import decsim.experiments.run_folder as run_folder

RUN_SCRIPT = "run.sbatch"
FOLD_SCRIPT = "fold.sbatch"
JOBS_FILE = "jobs.json"
LOGS_FOLDER = "logs"
# the fold reads every piece once and runs no shot
FOLD_SHAPE = "--cpus-per-task=1 --mem=16G --time=12:00:00"
# the share of its walltime a job's estimated work fills, Princeton
# Research Computing's request of 2026-10
USE_TARGET = 0.8
# a job under an hour lands in Della's test QOS, two jobs at a time
SHORTEST_WALLTIME_MINUTES = 61
# a piece is at most this share of a core's budget, so packed cores
# differ by at most a twentieth of it (Graham 1969: a core ends within
# one piece of the mean)
SLICES_PER_CORE = 20
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60


@dataclasses.dataclass(frozen=True)
class JobShape:
    """The Slurm request of every job: cores, the walltime limit, memory.

    Cores and memory are at least 1: sbatch reads a memory of zero as all
    of each node's memory (sbatch(1), --mem). The limit is at least two
    hours, since a job's walltime is at least SHORTEST_WALLTIME_MINUTES.
    """

    cores: int
    hours: int
    memory_mb: int

    def __post_init__(self) -> None:
        minimums = {
            "--cores": (self.cores, 1),
            "--hours": (self.hours, 2),
            "--memory-mb": (self.memory_mb, 1),
        }
        for flag, (value, minimum) in minimums.items():
            if value < minimum:
                raise refusal.RefusalError(
                    f"{flag} must be at least {minimum}, got {value}"
                )

    def core_budget_seconds(self) -> float:
        """The estimated work one core may hold: USE_TARGET of the limit."""
        limit_minutes = self.hours * MINUTES_PER_HOUR
        limit_seconds = limit_minutes * SECONDS_PER_MINUTE
        return USE_TARGET * limit_seconds


@dataclasses.dataclass(frozen=True)
class PlannedPiece:
    """Seeds [first_seed, first_seed + count) of one task, not yet saved."""

    task_name: str
    first_seed: int
    count: int
    estimated_seconds: float


@dataclasses.dataclass(frozen=True)
class PackedJob:
    """One Slurm job: lanes holds one list of pieces per core."""

    lanes: tuple

    def pieces_longest_first(self) -> list:
        """The job's pieces in the order its pool takes them.

        A pool that hands the next piece to the first free core, longest
        first, is Graham's list scheduling: with exact estimates it runs
        the lanes planned here, and when a core runs early it takes the
        next piece, so no core idles while another has a queue.
        """
        job_pieces = []
        for lane in self.lanes:
            job_pieces.extend(lane)
        return sorted(job_pieces, key=_estimated_seconds_of, reverse=True)

    def busiest_core_seconds(self) -> float:
        """The estimated work of the job's busiest core."""
        lane_seconds = [_lane_seconds(lane) for lane in self.lanes]
        return max(lane_seconds)

    def estimated_use(self, walltime_minutes: int) -> float:
        """The share of the cores' requested time the estimates fill."""
        cores = len(self.lanes)
        lane_seconds = [_lane_seconds(lane) for lane in self.lanes]
        work_seconds = sum(lane_seconds)
        walltime_seconds = walltime_minutes * SECONDS_PER_MINUTE
        requested_seconds = cores * walltime_seconds
        return work_seconds / requested_seconds


def launch(
    run_file: Union[str, pathlib.Path],
    out_dir: Optional[pathlib.Path],
    job: JobShape,
    *,
    dry_run: bool = False,
) -> None:
    """Every task recorded, its unsaved pieces packed, the jobs submitted.

    A tree git does not vouch for, and a task the packing cannot size,
    are refused first, before the results folder is made: out_dir, or a
    new dated one. A dry run writes the files, prints the plan and
    submits nothing.
    """
    refuse_an_unnamed_tree()
    given_path = pathlib.Path(run_file)
    run_path = given_path.resolve()
    study = experiment.load(run_path)
    for task in study.tasks:
        _refuse_a_task_it_cannot_pack(study, task, job)
    named_dir = run_folder.run_dir_for(study.name, out_dir)
    run_dir = named_dir.resolve()
    _every_id, collections, _started_utc = collect_command.start_the_folder(
        study, study, run_dir, run_path
    )
    budget_seconds = job.core_budget_seconds()
    planned = _planned_pieces(collections, budget_seconds)
    if not planned:
        print(f"every piece is saved; fold them with --fold --out {run_dir}")
        return
    jobs = pack(planned, job)
    minutes = walltime_minutes(jobs, job.hours)
    _write_the_job_files(run_path, run_dir, jobs, minutes, job)
    table = plan_table(jobs, minutes)
    print(table)
    if dry_run:
        return
    _submit(run_dir)


def pack(planned: list, job: JobShape) -> list:
    """The jobs that run these pieces, the busiest first.

    The job count is the fewest whose cores hold the work when each core
    keeps one piece of room: a core packed longest first ends within one
    piece of the mean (Graham 1969), so no core passes its budget and
    no walltime passes the limit. The cores are sorted by their work and
    taken in turn, so a job's cores carry about the same work.
    """
    budget_seconds = job.core_budget_seconds()
    piece_seconds = [piece.estimated_seconds for piece in planned]
    longest_piece = max(piece_seconds)
    total_seconds = sum(piece_seconds)
    room_seconds = budget_seconds - longest_piece
    job_seconds = job.cores * room_seconds
    jobs_needed = total_seconds / job_seconds
    job_count = math.ceil(jobs_needed)
    lane_count = job_count * job.cores
    lanes = _balanced_lanes(planned, lane_count)
    lanes.sort(key=_lane_seconds, reverse=True)
    jobs = []
    for first_lane in range(0, lane_count, job.cores):
        past_the_last_lane = first_lane + job.cores
        job_lanes = lanes[first_lane:past_the_last_lane]
        packed = PackedJob(tuple(job_lanes))
        jobs.append(packed)
    return jobs


def walltime_minutes(jobs: list, hours: int) -> int:
    """The walltime every job asks: the busiest core's work over USE_TARGET.

    Whole minutes, at least SHORTEST_WALLTIME_MINUTES and at most the
    limit, which only rounding could pass.
    """
    core_seconds = [packed.busiest_core_seconds() for packed in jobs]
    busiest_seconds = max(core_seconds)
    walltime_seconds = busiest_seconds / USE_TARGET
    exact_minutes = walltime_seconds / SECONDS_PER_MINUTE
    minutes = math.ceil(exact_minutes)
    floored_minutes = max(minutes, SHORTEST_WALLTIME_MINUTES)
    limit_minutes = hours * MINUTES_PER_HOUR
    return min(floored_minutes, limit_minutes)


def plan_table(jobs: list, minutes: int) -> str:
    """The walltime, then one line per job: cores, estimated use, pieces."""
    walltime = _walltime_text(minutes)
    lines = [f"every job asks {walltime}", "job  cores  use  pieces"]
    for index, packed in enumerate(jobs):
        cores = len(packed.lanes)
        use = packed.estimated_use(minutes)
        job_pieces = packed.pieces_longest_first()
        lines.append(
            f"{index:>3}  {cores:>5}  {use:>4.0%}  {len(job_pieces):>6}"
        )
    return "\n".join(lines)


def run_job(
    run_file: pathlib.Path,
    run_dir: pathlib.Path,
    index: int,
    *,
    processes: int = 1,
) -> None:
    """One Slurm job: the pieces jobs.json packed into job index."""
    jobs_path = run_dir / JOBS_FILE
    plan = run_folder.read_json(jobs_path)
    job_pieces = plan["jobs"][index]
    collect_command.run_pieces(
        run_file, run_dir, job_pieces, processes=processes
    )


def refuse_an_unnamed_tree() -> None:
    """A tree git does not vouch for is refused, unless ALLOW_DIRTY is set.

    Every job imports the tree as it stands when it starts, so an edit
    while the array is queued gives jobs different code, and the pieces
    would name a commit none ran. The reading is taken at submission and in
    every job, and passed through TREE_DIRTY_VARIABLE and
    TREE_PATCH_VARIABLE, since a job's interpreter may have no git.
    """
    checkout, commit, is_dirty = run_folder.fresh_tree_reading()
    dirty_text = _dirty_text(is_dirty)
    commit_text = commit or "unknown"
    print(f"decsim tree: {checkout}")
    print(f"decsim commit: {commit_text}, dirty: {dirty_text}")
    if is_dirty is not None:
        os.environ[run_folder.TREE_DIRTY_VARIABLE] = dirty_text
        patch_sha256 = run_folder.code_state_sha256(checkout)
        os.environ[run_folder.TREE_PATCH_VARIABLE] = patch_sha256 or ""
    if os.environ.get(run_folder.ALLOW_DIRTY_VARIABLE):
        return
    if is_dirty:
        raise refusal.RefusalError(
            f"refusing to start: {checkout} has uncommitted changes, so a "
            "job would import whatever the tree holds when it starts; "
            "commit them, submit from a worktree pinned at a commit, or "
            f"set {run_folder.ALLOW_DIRTY_VARIABLE}=1"
        )
    if is_dirty is None:
        raise refusal.RefusalError(
            f"refusing to start: git says nothing about {checkout}, so a "
            "job cannot name the code it ran; submit from a git checkout "
            f"pinned at a commit, or set {run_folder.ALLOW_DIRTY_VARIABLE}=1"
        )


def _refuse_a_task_it_cannot_pack(
    study: experiment.Experiment, task: collect.Task, job: JobShape
) -> None:
    """A task is packed only when its shots and their seconds are known.

    An online task's pieces run one after another from the state the
    piece before saved, so they cannot share cores. A task whose shot
    outlasts a slice of a core's budget would unbalance its cores.
    """
    settings = study.collection_of(task)
    if task.online_threshold is not None:
        raise refusal.RefusalError(
            f"the task {task.name} calibrates its threshold online, so its "
            "pieces run one after another and --slurm cannot pack them; "
            "run it without --slurm"
        )
    if not _has_a_fixed_shot_count(settings):
        raise refusal.RefusalError(
            f"the task {task.name} stops on failures or time, so its work "
            "is unknown before launch and --slurm cannot pack it; give it "
            "max_shots alone, or run it without --slurm"
        )
    if settings.core_seconds_per_shot is None:
        raise refusal.RefusalError(
            f"the task {task.name} has no core_seconds_per_shot, so "
            "--slurm cannot size its work; set it in its collection from "
            "an earlier run's sim_wall_seconds"
        )
    budget_seconds = job.core_budget_seconds()
    slice_seconds = budget_seconds / SLICES_PER_CORE
    if settings.core_seconds_per_shot > slice_seconds:
        raise refusal.RefusalError(
            f"a shot of the task {task.name} is estimated at "
            f"{settings.core_seconds_per_shot} core seconds, more than a "
            f"slice of a core's {budget_seconds:.0f} at --hours "
            f"{job.hours}, so its cores cannot be balanced; raise --hours"
        )


def _has_a_fixed_shot_count(
    settings: collection_module.CollectionSettings,
) -> bool:
    """Whether the task stops at max_shots and at nothing else."""
    if settings.max_failures is not None:
        return False
    if settings.max_core_seconds is not None:
        return False
    return settings.max_shots is not None


def _planned_pieces(collections: list, budget_seconds: float) -> list:
    """Every task's unsaved seeds cut into pieces, task by task."""
    slice_seconds = budget_seconds / SLICES_PER_CORE
    planned = []
    for collection in collections:
        task_pieces = _task_pieces(collection, slice_seconds)
        planned.extend(task_pieces)
    return planned


def _task_pieces(
    collection: collect_command.TaskCollection, slice_seconds: float
) -> list:
    """One task's unsaved seeds below max_shots, in pieces.

    A piece holds the shots of one slice, or the collection's own piece
    size when that is fewer, since piece_rounds bounds what a piece
    holds in memory and loses when killed.
    """
    settings = collection.settings
    seconds_per_shot = settings.core_seconds_per_shot
    whole_shots = slice_seconds // seconds_per_shot
    slice_shots = int(whole_shots)
    piece_shots = min(collection.piece_shots, slice_shots)
    ranges = pieces.unsaved_ranges(collection.saved, settings.max_shots)
    planned = []
    for first_seed, count in ranges:
        for piece_first, piece_count in _cut(first_seed, count, piece_shots):
            estimated_seconds = piece_count * seconds_per_shot
            piece = PlannedPiece(
                collection.name, piece_first, piece_count, estimated_seconds
            )
            planned.append(piece)
    return planned


def _cut(first_seed: int, count: int, piece_shots: int) -> list:
    """Seeds [first_seed, first_seed + count) as (first, count) pieces."""
    range_end = first_seed + count
    cut = []
    for piece_first in range(first_seed, range_end, piece_shots):
        seeds_left = range_end - piece_first
        piece_count = min(piece_shots, seeds_left)
        cut.append((piece_first, piece_count))
    return cut


def _balanced_lanes(planned: list, lane_count: int) -> list:
    """The pieces dealt longest first, each to the least loaded lane."""
    lanes = [[] for _lane in range(lane_count)]
    loads = [(0.0, index) for index in range(lane_count)]
    longest_first = sorted(planned, key=_estimated_seconds_of, reverse=True)
    for piece in longest_first:
        load, index = heapq.heappop(loads)
        lanes[index].append(piece)
        new_load = load + piece.estimated_seconds
        heapq.heappush(loads, (new_load, index))
    return lanes


def _write_the_job_files(
    run_path: pathlib.Path,
    run_dir: pathlib.Path,
    jobs: list,
    minutes: int,
    job: JobShape,
) -> None:
    """jobs.json, each job's pieces longest first; run.sbatch; fold.sbatch."""
    logs_dir = run_dir / LOGS_FOLDER
    logs_dir.mkdir(exist_ok=True)
    job_records = [_job_record(packed) for packed in jobs]
    jobs_path = run_dir / JOBS_FILE
    run_folder.write_json(jobs_path, {"jobs": job_records})
    array = f"0-{len(jobs) - 1}"
    walltime = _walltime_text(minutes)
    run_script = run_dir / RUN_SCRIPT
    run_lines = _run_lines(run_path, run_dir, array, walltime, job)
    run_script.write_text(run_lines)
    fold_script = run_dir / FOLD_SCRIPT
    fold_lines = _fold_lines(run_dir)
    fold_script.write_text(fold_lines)


def _job_record(packed: PackedJob) -> list:
    """One job of jobs.json: its pieces, longest first."""
    job_pieces = packed.pieces_longest_first()
    return [_piece_record(piece) for piece in job_pieces]


def _piece_record(piece: PlannedPiece) -> list:
    return [piece.task_name, piece.first_seed, piece.count]


def _submit(run_dir: pathlib.Path) -> None:
    """The array submitted, then the fold behind it."""
    run_script = run_dir / RUN_SCRIPT
    array_job = _submitted([str(run_script)])
    dependency = f"--dependency=afterany:{array_job}"
    fold_script = run_dir / FOLD_SCRIPT
    fold_job = _submitted([dependency, str(fold_script)])
    print(f"job array {array_job}, fold job {fold_job}")


def _walltime_text(minutes: int) -> str:
    """A walltime in minutes as sbatch's hours:minutes:seconds."""
    hours, minutes_past = divmod(minutes, MINUTES_PER_HOUR)
    return f"{hours}:{minutes_past:02d}:00"


def _dirty_text(is_dirty: Optional[bool]) -> str:
    """The dirty flag as the launcher exports it: 1, 0 or unknown."""
    if is_dirty is None:
        return "unknown"
    return str(int(is_dirty))


def _run_lines(
    run_path: pathlib.Path,
    run_dir: pathlib.Path,
    array: str,
    walltime: str,
    job: JobShape,
) -> str:
    """run.sbatch: the array, job i running its pieces into the folder.

    A job runs the run file where it stands, as MultiSim runs its
    config, so a file the run file reads beside itself is found. Its log
    is named by the array's id and its index, so a resubmission keeps
    the last one's logs.

    Every path is quoted, which bash and sbatch's own reading of an
    #SBATCH line both undo, and the job's array index is left for bash
    to expand.
    """
    shape = (
        f"--array={array} --cpus-per-task={job.cores} "
        f"--mem={job.memory_mb}M --time={walltime}"
    )
    log = run_dir / LOGS_FOLDER / "%A_%a.log"
    literal = [sys.executable, "-m", "decsim", "run", str(run_path)]
    literal += ["--out", str(run_dir)]
    quoted = shlex.join(literal)
    command = f"{quoted} --job $SLURM_ARRAY_TASK_ID --processes {job.cores}"
    output = shlex.quote(f"--output={log}")
    lines = ["#!/bin/bash", f"#SBATCH {shape}", f"#SBATCH {output}"]
    lines.append(command)
    return "\n".join(lines) + "\n"


def _fold_lines(run_dir: pathlib.Path) -> str:
    """fold.sbatch: every saved piece folded into the folder's csv files."""
    log = run_dir / LOGS_FOLDER / "fold.log"
    literal = [sys.executable, "-m", "decsim", "run", "--fold"]
    literal += ["--out", str(run_dir)]
    command = shlex.join(literal)
    output = shlex.quote(f"--output={log}")
    lines = ["#!/bin/bash", f"#SBATCH {FOLD_SHAPE}", f"#SBATCH {output}"]
    lines.append(command)
    return "\n".join(lines) + "\n"


def _submitted(arguments: list) -> str:
    """One sbatch submission; the job id it answered.

    --parsable answers "id" or "id;cluster" (sbatch(1)).
    """
    completed = subprocess.run(
        ["sbatch", "--parsable", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        argument_text = " ".join(arguments)
        error_text = completed.stderr.strip()
        message = (
            f"sbatch {argument_text} exited {completed.returncode}: "
            f"{error_text}"
        )
        raise refusal.RefusalError(message)
    answer = completed.stdout.strip()
    fields = answer.split(";")
    return fields[0]


def _estimated_seconds_of(piece: PlannedPiece) -> float:
    return piece.estimated_seconds


def _lane_seconds(lane: list) -> float:
    """A core's estimated work: its pieces' seconds added up."""
    piece_seconds = [piece.estimated_seconds for piece in lane]
    return sum(piece_seconds)
