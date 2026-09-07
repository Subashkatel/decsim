"""The run folder: where a sweep's results, config and identity land.

results/<utc stamp>-<name>/, never reused unless the caller names one
with --out; the config chain copied verbatim beside the rows, any
uncommitted code as a patch, and a manifest that says which commit,
container, host and package versions produced them. gem5 writes its
output to m5out/ the same way, out of the code tree and never
overwritten (src/python/m5/main.py --outdir).
"""

import datetime
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
    whole experiment.
    """
    config_dir = run_dir / "config"
    config_dir.mkdir(exist_ok=True)
    for config_file in config.config_files:
        source = Path(config_file)
        target = config_dir / source.name
        shutil.copy2(config_file, target)
    diff = _git_output("git", "diff", "HEAD")
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


def _git_state() -> dict:
    """The commit and whether the tree is dirty; None where git is absent.

    The container image has no git binary, so the commit falls back to
    reading .git directly; dirty stays a host-side best effort.
    """
    commit = _git_output("git", "rev-parse", "HEAD")
    if not commit:
        commit = _commit_from_git_files()
    porcelain = _git_output("git", "status", "--porcelain")
    is_dirty = None
    if porcelain is not None:
        is_dirty = bool(porcelain)
    return {"commit": commit, "dirty": is_dirty}


def _git_output(*arguments) -> Optional[str]:
    try:
        completed = subprocess.run(arguments, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return completed.stdout.strip()


def _commit_from_git_files() -> Optional[str]:
    head_path = Path(".git/HEAD")
    if not head_path.exists():
        return None
    head_text = head_path.read_text()
    head = head_text.strip()
    if not head.startswith("ref: "):
        return head
    reference = head[len("ref: ") :]
    reference_path = Path(".git") / reference
    if reference_path.exists():
        reference_text = reference_path.read_text()
        return reference_text.strip()
    packed = Path(".git/packed-refs")
    if packed.exists():
        return _packed_reference(packed, reference)
    return None


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
