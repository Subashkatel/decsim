"""`decsim run --slurm`: one array task per point, then one fold.

gem5 MultiSim's list-then-run-one-id pattern. The launcher records every
point, which runs each build, so a refused point queues nothing, and
writes run.sbatch, one array whose task i runs point i to its stop from
the saved pieces, and fold.sbatch, which waits on the array with
afterany and folds every saved piece. Submitting again resumes.

No account, partition or QOS is written: sbatch reads SBATCH_ACCOUNT,
SBATCH_PARTITION and SBATCH_QOS from the environment (sbatch(1)).
"""

import dataclasses
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Optional, Union

import decsim.experiments.collect_command as collect_command
import decsim.experiments.experiment as experiment
import decsim.experiments.refusal as refusal
import decsim.experiments.run_folder as run_folder

RUN_SCRIPT = "run.sbatch"
FOLD_SCRIPT = "fold.sbatch"
LOGS_FOLDER = "logs"
# the fold reads every piece once and runs no shot
FOLD_SHAPE = "--cpus-per-task=1 --mem=16G --time=12:00:00"


@dataclasses.dataclass(frozen=True)
class JobShape:
    """The Slurm request of one array task (cores, walltime, memory).

    Each is at least 1: sbatch reads a time limit of zero as no limit
    and a memory of zero as all of each node's memory (sbatch(1),
    --time and --mem), so a 0 would run a job nobody asked for.
    """

    cores: int
    hours: int
    memory_mb: int

    def __post_init__(self) -> None:
        shape = {
            "--cores": self.cores,
            "--hours": self.hours,
            "--memory-mb": self.memory_mb,
        }
        for flag, value in shape.items():
            if value < 1:
                raise refusal.RefusalError(
                    f"{flag} must be at least 1, got {value}"
                )


def launch(
    run_file: Union[str, pathlib.Path],
    out_dir: Optional[pathlib.Path],
    job: JobShape,
    *,
    dry_run: bool = False,
) -> None:
    """Every point recorded, the two sbatch files written, then submitted.

    A tree git does not vouch for is refused first, before the results
    folder is made: out_dir, or a new dated one. A dry run writes the
    files and submits nothing.
    """
    refuse_an_unnamed_tree()
    given_path = pathlib.Path(run_file)
    run_path = given_path.resolve()
    study = experiment.load(run_path)
    named_dir = run_folder.run_dir_for(study.name, out_dir)
    run_dir = named_dir.resolve()
    collect_command.start_the_folder(study, study, run_dir, run_path)
    logs_dir = run_dir / LOGS_FOLDER
    logs_dir.mkdir(exist_ok=True)
    point_count = len(study.points)
    run_script = run_dir / RUN_SCRIPT
    run_lines = _run_lines(run_path, run_dir, point_count, job)
    run_script.write_text(run_lines)
    fold_script = run_dir / FOLD_SCRIPT
    fold_lines = _fold_lines(run_dir)
    fold_script.write_text(fold_lines)
    print(run_script)
    print(fold_script)
    if dry_run:
        return
    array_job = _submitted([str(run_script)])
    dependency = f"--dependency=afterany:{array_job}"
    fold_job = _submitted([dependency, str(fold_script)])
    print(f"array job {array_job}, fold job {fold_job}")


def refuse_an_unnamed_tree() -> None:
    """A tree git does not vouch for is refused, unless ALLOW_DIRTY is set.

    Every task imports the tree as it stands when it starts, so an edit
    while the array is queued gives tasks different code, and the pieces
    would name a commit none ran. The reading is taken at submission and in
    every task, and passed through TREE_DIRTY_VARIABLE and
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
            "task would import whatever the tree holds when it starts; "
            "commit them, submit from a worktree pinned at a commit, or "
            f"set {run_folder.ALLOW_DIRTY_VARIABLE}=1"
        )
    if is_dirty is None:
        raise refusal.RefusalError(
            f"refusing to start: git says nothing about {checkout}, so a "
            "task cannot name the code it ran; submit from a git checkout "
            f"pinned at a commit, or set {run_folder.ALLOW_DIRTY_VARIABLE}=1"
        )


def _dirty_text(is_dirty: Optional[bool]) -> str:
    """The dirty flag as the launcher exports it: 1, 0 or unknown."""
    if is_dirty is None:
        return "unknown"
    return str(int(is_dirty))


def _run_lines(
    run_path: pathlib.Path,
    run_dir: pathlib.Path,
    point_count: int,
    job: JobShape,
) -> str:
    """run.sbatch: the array, task i running point i into the folder.

    A task runs the run file where it stands, as MultiSim runs its
    config, so a file the run file reads beside itself is found.

    Every path is quoted, which bash and sbatch's own reading of an
    #SBATCH line both undo, and the task index is left for bash to
    expand.
    """
    last_task = point_count - 1
    shape = (
        f"--array=0-{last_task} --cpus-per-task={job.cores} "
        f"--mem={job.memory_mb}M --time={job.hours}:00:00"
    )
    log = run_dir / LOGS_FOLDER / "%a.log"
    literal = [sys.executable, "-m", "decsim", "run", str(run_path)]
    literal += ["--out", str(run_dir)]
    quoted = shlex.join(literal)
    command = f"{quoted} --task $SLURM_ARRAY_TASK_ID --processes {job.cores}"
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
