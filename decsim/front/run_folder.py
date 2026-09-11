"""The run folder: where a sweep's results, config and identity land.

results/<utc stamp>-<name>/, never reused unless the caller names one
with --out; the config chain copied verbatim beside the rows, any
uncommitted code as a patch, and a manifest that says which commit,
container, host and package versions produced them. gem5 writes its
output to m5out/ the same way, out of the code tree and never
overwritten (src/python/m5/main.py --outdir).
"""

import datetime
import functools
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy
import pymatching
import stim

import decsim.collect as collect

RESULTS_DIR = Path("results")
# What the launcher saw of the tree it was about to run, for a process
# whose interpreter has no git of its own (slurm/slurm_run.sh exports
# it): "1" dirty, "0" clean, unset means nobody looked.
TREE_DIRTY_VARIABLE = "DECSIM_TREE_DIRTY"


def run_dir_for(config, out_dir=None) -> Path:
    """Where this run writes: the folder asked for, or a fresh stamped one.

    gem5's --outdir names the folder and makes it (src/python/m5/main.py:
    102); with no --outdir it writes m5out/. A named folder is reused as
    the caller asked, so a Slurm array can point every shard at a folder
    of its own.
    """
    if out_dir is None:
        return new_run_dir(config)
    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def combined_run_dir(out_dir=None) -> Path:
    """Where `decsim combine` writes: the folder asked for, or a fresh one.

    Only the path: the report makes the folder once the fold is accepted,
    so a refused combine leaves nothing behind.
    """
    if out_dir is not None:
        return Path(out_dir)
    stamp = _utc_stamp()
    return RESULTS_DIR / f"{stamp}-combined"


def new_run_dir(config) -> Path:
    """results/<UTC stamp>-<config name>/, never reused; sorted by time.

    The stamp is whole seconds, so a second run started within the same
    second gets a numeric suffix instead of clobbering the first.
    """
    stamp = _utc_stamp()
    base = RESULTS_DIR / f"{stamp}-{config.name}"
    run_dir = base
    suffix = 2
    while True:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            run_dir = base.with_name(f"{base.name}-{suffix}")
            suffix += 1


def snapshot_code_state(config, run_dir: Path) -> None:
    """Copy the run's exact inputs next to its results.

    The config chain goes verbatim into config/, and any uncommitted code
    into code_state.patch, so manifest commit + patch + config = the
    whole experiment. The patch is the imported tree's, for the reason
    the manifest's commit is (_checkout).
    """
    config_dir = run_dir / "config"
    config_dir.mkdir(exist_ok=True)
    for config_file in config.config_files:
        source = Path(config_file)
        target = config_dir / source.name
        shutil.copy2(config_file, target)
    checkout = _checkout()
    diff = _git_output("git", "-C", str(checkout), "diff", "HEAD")
    if diff:
        patch_path = run_dir / "code_state.patch"
        patch_path.write_text(diff)


def write_manifest(
    config,
    run_dir: Path,
    started_utc: str,
    finished_utc: Optional[str] = None,
    *,
    shard: Optional[tuple] = None,
    shots_per_unit: Optional[int] = None,
) -> None:
    """Write the run's identity: enough to interpret or reproduce it.

    Sampling is deterministic from (stim version, circuit, distance,
    rounds, p, seed), so the manifest plus seeds are the raw data.

    `shard` and `shots_per_unit` are facts of how this run ran, the way
    the host and the slurm job id are: which share of the sweep's work
    units fell to it, and how its points were cut into units. Nothing
    reads them to fold the rows, since `decsim combine` puts the folded
    rows in the order the recorded sweep gives; they say what a folder
    holds when a Slurm array leaves a hundred of them behind.
    """
    json_safe_config = collect.json_value(config)
    config_files = []
    for path in config.config_files:
        config_files.append(str(path))
    manifest = {
        "config_files": config_files,
        "resolved_config": json_safe_config,
        "shard": _shard_text(shard),
        "shots_per_unit": shots_per_unit,
    }
    how_it_ran = _how_it_ran()
    manifest.update(how_it_ran)
    manifest["started_utc"] = started_utc
    manifest["finished_utc"] = finished_utc
    _write_the_manifest(manifest, run_dir)


def write_combined_manifest(
    resolved_config: dict,
    run_dir: Path,
    folded: list,
    started_utc: str,
    finished_utc: Optional[str] = None,
) -> None:
    """A combined folder's identity: the sweep it folds, and what it folded.

    A combined folder is a run folder, so it carries a manifest like any
    other and `decsim combine` can fold it again with a shard that
    landed later. The resolved config is the one every folded folder
    recorded, which is what combine reads the sweep order off; the
    folded folders' names are a fact of how this folder came about, and
    like a run's shard they order nothing.
    """
    folded_names = []
    for run_folder_path in folded:
        folded_names.append(str(run_folder_path))
    manifest = {
        "resolved_config": resolved_config,
        "folded": folded_names,
    }
    how_it_ran = _how_it_ran()
    manifest.update(how_it_ran)
    manifest["started_utc"] = started_utc
    manifest["finished_utc"] = finished_utc
    _write_the_manifest(manifest, run_dir)


def utc_now() -> str:
    """This moment as an iso timestamp, for the manifest's times."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.isoformat()


def _write_the_manifest(manifest: dict, run_dir: Path) -> None:
    """manifest.json, indented, in the folder it describes."""
    manifest_path = run_dir / "manifest.json"
    manifest_text = json.dumps(manifest, indent=2)
    manifest_path.write_text(manifest_text)


def _how_it_ran() -> dict:
    """Where and by what this process ran, for either kind of manifest."""
    return {
        "git": _git_state(),
        "container": _container(),
        "versions": _versions(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "argv": sys.argv,
    }


def _shard_text(shard: Optional[tuple]) -> Optional[str]:
    """The shard as the user wrote it, i/n; None when there was none."""
    if shard is None:
        return None
    index, count = shard
    return f"{index}/{count}"


def _utc_stamp() -> str:
    """This moment as a folder-name stamp, whole seconds."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y-%m-%dT%H-%M-%SZ")


def _container() -> Optional[str]:
    """The apptainer image the run is inside, when it is inside one."""
    container = os.environ.get("APPTAINER_CONTAINER")
    if not container:
        container = os.environ.get("SINGULARITY_CONTAINER")
    return container


def _checkout() -> Path:
    """The tree this code was imported from, which is the code that ran.

    Not the working directory: a cluster task starts in the folder its
    job was submitted from and may import a checkout pinned at a commit
    somewhere else (docs/how-to/run_a_sweep_on_slurm.md), so a folder
    that named the working directory's commit would name code no part
    of the run read.
    """
    this_file = Path(__file__)
    here = this_file.resolve()
    return here.parents[2]


def _git_state() -> dict:
    """The manifest's git block: the one reading this process took.

    gem5 prints its version, its build date, the host and the command
    line at every start, so a result says what produced it
    (tmp/resources/gem5/src/python/m5/main.py:524-537), and sinter
    carries the decoder and the task's metadata in every row of its csv
    for the same reason (sinter/_data/_task_stats.py:196-204 through
    _data/_csv_out.py:56-65). decsim's run folder is where that belongs
    here, so the manifest names the commit and says whether the tree had
    uncommitted changes.
    """
    commit, is_dirty = _tree_reading()
    return {"commit": commit, "dirty": is_dirty}


@functools.lru_cache(maxsize=1)
def _tree_reading() -> tuple:
    """(commit, dirty) of the tree this code came from, read once.

    Read at the first ask, which is before the first shot, and reused by
    every later ask. A run writes its manifest at its start and again at
    its end, and a tree can move between the two: a Slurm array running
    for hours out of a checkout somebody commits to would name, in every
    folder, whatever the tree held when that task finished, which is
    code no part of the run read. A manifest whose commit is not the
    code's is worse than none.

    Both referents record provenance before the work and not after.
    gem5 prints its version, its build date, its host, its pid and its
    command line at :524-556 of
    tmp/resources/gem5/src/python/m5/main.py, then executes the
    simulation script at :687. sinter writes its csv header into the
    save file before the collect loop
    (sinter/_collection/_collection.py:385-397 against the loop at
    :401), and every row's strong id is computed from the task and
    cached on it rather than recomputed per row
    (sinter/_data/_task.py:248-270).

    The launcher's answer about dirtiness wins when it is given, because
    the launcher looked at the tree as the job started, which is the
    tree the process went on to import; this process's own git looks
    later, and edits made after the import did not run. The container
    image ships no git binary at all, which is why the commit falls back
    to reading the tree's own git files and why the dirty flag is None
    rather than clean when nobody could answer.
    """
    checkout = _checkout()
    commit = _git_output("git", "-C", str(checkout), "rev-parse", "HEAD")
    if not commit:
        commit = _commit_from_git_files(checkout)
    is_dirty = _dirty_from_the_launcher()
    if is_dirty is None:
        porcelain = _git_output(
            "git", "-C", str(checkout), "status", "--porcelain"
        )
        if porcelain is not None:
            is_dirty = bool(porcelain)
    return commit, is_dirty


def _dirty_from_the_launcher() -> Optional[bool]:
    """What the launcher saw, when it exported it (slurm/slurm_run.sh)."""
    said = os.environ.get(TREE_DIRTY_VARIABLE)
    if said is None or said == "":
        return None
    return said != "0"


def _git_output(*arguments) -> Optional[str]:
    """One git command's output; None when git could not answer."""
    try:
        completed = subprocess.run(arguments, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _commit_from_git_files(checkout: Path) -> Optional[str]:
    """The checkout's own HEAD, read without git."""
    git_dir = _git_dir(checkout)
    if git_dir is None:
        return None
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return None
    head_text = head_path.read_text()
    head = head_text.strip()
    if not head.startswith("ref: "):
        return head
    reference = head[len("ref: ") :]
    common = _common_git_dir(git_dir)
    for directory in (git_dir, common):
        found = _reference_in(directory, reference)
        if found is not None:
            return found
    return None


def _reference_in(directory: Path, reference: str) -> Optional[str]:
    """One reference in one git directory, loose or packed."""
    reference_path = directory / reference
    if reference_path.exists():
        reference_text = reference_path.read_text()
        return reference_text.strip()
    packed = directory / "packed-refs"
    if not packed.exists():
        return None
    return _packed_reference(packed, reference)


def _git_dir(checkout: Path) -> Optional[Path]:
    """The checkout's .git, or where it points when it is a worktree."""
    git_path = checkout / ".git"
    if git_path.is_dir():
        return git_path
    if not git_path.is_file():
        return None
    pointer = git_path.read_text()
    text = pointer.strip()
    prefix = "gitdir: "
    if not text.startswith(prefix):
        return None
    return Path(text[len(prefix) :])


def _common_git_dir(git_dir: Path) -> Path:
    """Where a worktree's git dir keeps the refs it shares with the repo."""
    commondir = git_dir / "commondir"
    if not commondir.exists():
        return git_dir
    written = commondir.read_text()
    relative = written.strip()
    common = git_dir / relative
    return common.resolve()


def _packed_reference(packed: Path, reference: str) -> Optional[str]:
    packed_text = packed.read_text()
    for line in packed_text.splitlines():
        if line.endswith(reference):
            words = line.split()
            return words[0]
    return None


def _versions() -> dict:
    version_words = sys.version.split()
    python_version = version_words[0]
    return {
        "python": python_version,
        "stim": stim.__version__,
        "pymatching": pymatching.__version__,
        "numpy": numpy.__version__,
    }
