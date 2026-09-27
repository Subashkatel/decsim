"""The run folder: where a run's results, config and identity land.

results/<utc stamp>-<name>/, or the folder --out names, holds the config
chain, the code state as a patch, a manifest of the commit, container,
host and packages, each sweep point's values and maker (resolved/) and
workload (inputs/), and the finished flag last. gem5
writes m5out/ the same way, out of the code tree and never overwritten
(src/python/m5/main.py --outdir).
"""

import datetime
import functools
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping
from typing import Optional

import decsim.build.escalation as escalation_build
import decsim.build.plan as plan_build
import decsim.collect as collect
import decsim.compiled_libraries as compiled_libraries
import decsim.experiments.experiment as experiment
import decsim.frontends.settings as workload_settings
import decsim.frontends.workload_files as workload_files

RESULTS_DIR = pathlib.Path("results")
RESOLVED_FOLDER = "resolved"
INPUTS_FOLDER = "inputs"
HASHES_FILE = "hashes.json"
# Written last, once a run's rows, report and manifest are all in place;
# a folder without it is a run that stopped or is still running.
FINISHED_FILE = "finished"
# What the launcher saw of the tree it was about to run, for a process
# whose interpreter has no git of its own (slurm/slurm_run.sh exports
# it): "1" dirty, "0" clean, unset means nobody looked.
TREE_DIRTY_VARIABLE = "DECSIM_TREE_DIRTY"


def run_dir_for(config, out_dir=None) -> pathlib.Path:
    """Where this run writes: the folder asked for, or a fresh stamped one.

    gem5's --outdir names the folder and makes it (src/python/m5/main.py:
    102); with no --outdir it writes m5out/. A named folder is reused as
    the caller asked, so a Slurm array can point every shard at a folder
    of its own.
    """
    if out_dir is None:
        return new_run_dir(config)
    run_dir = pathlib.Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def combined_run_dir(out_dir=None) -> pathlib.Path:
    """Where `decsim combine` writes: the folder asked for, or a fresh one.

    Only the path: the report makes the folder once the fold is accepted,
    so a refused combine leaves nothing behind.
    """
    if out_dir is not None:
        return pathlib.Path(out_dir)
    stamp = _utc_stamp()
    return RESULTS_DIR / f"{stamp}-combined"


def new_run_dir(config) -> pathlib.Path:
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


def start_run(
    config: Optional[experiment.ExperimentConfig],
    run_dir: pathlib.Path,
    point_ids: list,
    *,
    shard: Optional[tuple] = None,
    shots_per_unit: Optional[int] = None,
) -> str:
    """The code state and the manifest, before the first shot; the time.

    config is None for a run no yaml describes (the tools/ examples).
    point_ids are the run's points in task order (write_manifest).
    """
    snapshot_code_state(config, run_dir)
    started_utc = utc_now()
    how_it_ran = {"shard": shard, "shots_per_unit": shots_per_unit}
    write_manifest(config, run_dir, point_ids, started_utc, **how_it_ran)
    return started_utc


def finish_run(
    config: Optional[experiment.ExperimentConfig],
    run_dir: pathlib.Path,
    point_ids: list,
    started_utc: str,
    *,
    shard: Optional[tuple] = None,
    shots_per_unit: Optional[int] = None,
) -> None:
    """The manifest again with the time the run ended, then the flag."""
    finished_utc = utc_now()
    how_it_ran = {"shard": shard, "shots_per_unit": shots_per_unit}
    write_manifest(
        config, run_dir, point_ids, started_utc, finished_utc, **how_it_ran
    )
    mark_finished(run_dir)


def snapshot_code_state(
    config: Optional[experiment.ExperimentConfig], run_dir: pathlib.Path
) -> None:
    """Copy the run's exact inputs next to its results.

    The config chain goes verbatim into config/, and any uncommitted code
    into code_state.patch, an untracked file as a patch that creates it
    (git diff --no-index), so manifest commit + patch + config = the
    whole experiment. The patch is the imported tree's, for the reason
    the manifest's commit is (_checkout).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    if config is not None:
        _copy_the_config_chain(config.config_files, run_dir)
    checkout = _checkout()
    diff = _git_output("git", "-C", str(checkout), "diff", "HEAD")
    patches = []
    if diff:
        patches.append(diff)
    untracked = _untracked_files(checkout) or []
    for relative in untracked:
        patch = _untracked_patch(checkout, relative)
        patches.append(patch)
    if patches:
        patch_path = run_dir / "code_state.patch"
        patch_text = "\n".join(patches)
        patch_lines = patch_text + "\n"
        patch_path.write_text(patch_lines)


def read_the_tree() -> None:
    """Take this process's reading of the tree, before its work starts.

    A manifest carries the reading the process took at its first ask,
    and the ask belongs before the work rather than after
    (_tree_reading). A run needs no call here: it writes a manifest
    before its first shot, so its own first ask is at its start.
    `decsim combine` writes one manifest and writes it at the end, so
    it asks here instead, beside the time it started and before the
    first folder is opened, and a fold long enough for the tree to move
    still names the commit its code came from.
    """
    _tree_reading()


def write_manifest(
    config: Optional[experiment.ExperimentConfig],
    run_dir: pathlib.Path,
    point_ids: list,
    started_utc: str,
    finished_utc: Optional[str] = None,
    *,
    shard: Optional[tuple] = None,
    shots_per_unit: Optional[int] = None,
) -> None:
    """Write the run's identity: enough to interpret or reproduce it.

    Sampling is deterministic from (stim version, circuit, distance,
    rounds, p, seed), so the manifest plus seeds are the raw data. config
    is None for a run no yaml describes (tools/deltakit_example.py),
    whose every value is in resolved/. point_ids are the sweep's points
    in task order, each the name of its resolved/ record, which is the
    order `decsim combine` writes a fold's rows in.

    `shard` and `shots_per_unit` are facts of how this run ran, the way
    the host and the slurm job id are: which share of the sweep's work
    units fell to it, and how its points were cut into units. Nothing
    reads them to fold the rows, since `decsim combine` puts the folded
    rows in the order the recorded sweep gives; they say what a folder
    holds when a Slurm array leaves a hundred of them behind.
    """
    experiment_config = None
    config_files = []
    if config is not None:
        experiment_config = collect.json_value(config)
        for path in config.config_files:
            config_files.append(str(path))
    manifest = {
        "config_files": config_files,
        "experiment_config": experiment_config,
        "points": point_ids,
        "shard": _shard_text(shard),
        "shots_per_unit": shots_per_unit,
    }
    how_it_ran = _how_it_ran()
    manifest.update(how_it_ran)
    manifest["started_utc"] = started_utc
    manifest["finished_utc"] = finished_utc
    _write_the_manifest(manifest, run_dir)


def write_combined_manifest(
    experiment_config: dict,
    point_ids: list,
    run_dir: pathlib.Path,
    folded: list,
    started_utc: str,
    finished_utc: Optional[str] = None,
) -> None:
    """A combined folder's identity: the sweep it folds, and what it folded.

    A combined folder is a run folder, so it carries a manifest like any
    other and `decsim combine` can fold it again with a shard that
    landed later. The experiment config and the point ids are the ones
    every folded folder recorded, and the ids are what combine reads the
    sweep order off; the folded folders' names are a fact of how this
    folder came about, and like a run's shard they order nothing.
    """
    folded_names = []
    for run_folder_path in folded:
        folded_names.append(str(run_folder_path))
    manifest = {
        "experiment_config": experiment_config,
        "points": point_ids,
        "folded": folded_names,
    }
    how_it_ran = _how_it_ran()
    manifest.update(how_it_ran)
    manifest["started_utc"] = started_utc
    manifest["finished_utc"] = finished_utc
    _write_the_manifest(manifest, run_dir)


def record_point(
    run_dir: pathlib.Path,
    task: collect.Task,
    seeds: Optional[list] = None,
    sections: Optional[Mapping] = None,
) -> str:
    """One sweep point's values and workload, named by its strong id.

    resolved/<id>.json holds the metadata, the seed ranges run, the
    sections the point's yaml resolved to (its axes placed and its
    references resolved, as Hydra keeps each job's composed config in
    .hydra/config.yaml), the maker a producer workload called with the
    point's own arguments, every setting and the values the build
    derives; a point a Python caller built has no sections. inputs/<id>/ holds
    the workload as the files row reads it, with each file's sha256 in
    hashes.json, so a rerun needs no maker installed. Returns the id.
    """
    point_id = task.strong_id()
    settings = task.settings
    shot_settings = task.shot_settings()
    resolved = {
        "id": point_id,
        "metadata": collect.json_value(task.metadata),
        "seeds": seeds,
        "sections": collect.json_value(sections),
        "producer": _producer(settings.workload),
        "settings": collect.json_value(settings),
        "built": _built_values(shot_settings),
    }
    resolved_dir = run_dir / RESOLVED_FOLDER
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = resolved_dir / f"{point_id}.json"
    write_json(resolved_path, resolved)
    record = settings.workload.workload_record
    if record is not None:
        inputs_dir = run_dir / INPUTS_FOLDER / point_id
        _write_inputs(inputs_dir, record)
    return point_id


def mark_finished(run_dir: pathlib.Path) -> None:
    """The finished flag, the last thing a run writes: the time it ended."""
    finished_utc = utc_now()
    finished_line = finished_utc + "\n"
    finished_path = run_dir / FINISHED_FILE
    finished_path.write_text(finished_line)


def is_finished(run_dir: pathlib.Path) -> bool:
    """Whether a run into this folder ended, which a rerun then skips."""
    finished_path = pathlib.Path(run_dir) / FINISHED_FILE
    return finished_path.exists()


def resolved_by_point(run_dir: pathlib.Path) -> dict:
    """Each point's resolved/ record, keyed by its point id."""
    records = {}
    resolved_dir = pathlib.Path(run_dir) / RESOLVED_FOLDER
    paths = resolved_dir.glob("*.json")
    for path in sorted(paths):
        text = path.read_text()
        record = json.loads(text)
        records[record["id"]] = record
    return records


def swept_values(run_dir: pathlib.Path, point_ids: list) -> dict:
    """Each point's value at every yaml path the sweep sets, as csv cells.

    One column per swept path is Wickham's tidy table, each variable a
    column and each observation a row (Tidy Data, J. Stat. Softw.
    59(10), 2014, section 2.3); the paths come in the order the points
    first set them. Every cell is the value the point resolved to, from
    its record's sections, whether its block set the path or not, so a
    swept reference is the value it names; a path its sections do not
    hold is an empty cell, not available. A value other than a string
    or a number is one cell of compact json, as sinter writes
    json_metadata (sinter/_data/_csv_out.py:35-37); the typed value is
    in the record.
    """
    records = resolved_by_point(run_dir)
    paths = {}
    for point_id in point_ids:
        metadata = records[point_id]["metadata"]
        new_paths = dict.fromkeys(metadata)
        paths.update(new_paths)
    values = {}
    for point_id in point_ids:
        record = records[point_id]
        values[point_id] = _cells_of(record, list(paths))
    return values


def resolved_values(record: dict, names: tuple) -> dict:
    """The named parts of a resolved record, each value at its dotted path.

    resolved_values(record, ("settings",)) gives
    settings.qpu.distance and every other setting as one mapping.
    """
    values = {}
    for name in names:
        part = record.get(name)
        for path, value in experiment.value_leaves(part, (name,)):
            dotted = ".".join(path)
            values[dotted] = value
    return values


def seed_ranges(ranges: list) -> list:
    """Seed ranges as [first, how many], in order, touching ones joined.

    A combine into a folder that already holds the fold sees the same
    seeds again, so ranges that overlap are joined too.
    """
    joined = []
    for first, count in sorted(ranges):
        if joined and sum(joined[-1]) >= first:
            last = joined[-1]
            last_end = sum(last)
            new_end = first + count
            end = max(last_end, new_end)
            last[1] = end - last[0]
            continue
        joined.append([first, count])
    return joined


def copy_point_records(run_dirs: list, out_dir: pathlib.Path) -> None:
    """The folded folders' resolved/ and inputs/, in one.

    A point's files are named by its content, so the same point in two
    shards is the same file, and the union is every point's. Its record
    holds the seeds every folder ran of it.
    """
    for run_dir_name in run_dirs:
        run_dir = pathlib.Path(run_dir_name)
        _fold_resolved(run_dir, out_dir)
        source = run_dir / INPUTS_FOLDER
        target = out_dir / INPUTS_FOLDER
        shutil.copytree(source, target, dirs_exist_ok=True)


def write_json(path: pathlib.Path, value) -> None:
    """A json file of the run folder: indented, ending in a newline."""
    text = json.dumps(value, indent=2)
    lines = text + "\n"
    path.write_text(lines)


def utc_now() -> str:
    """This moment as an iso timestamp, for the manifest's times."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.isoformat()


def _copy_the_config_chain(config_files: tuple, run_dir: pathlib.Path) -> None:
    """Every yaml of the chain into config/, each at its place."""
    config_dir = run_dir / "config"
    config_dir.mkdir(exist_ok=True)
    chain_folder = _chain_folder(config_files)
    for config_file in config_files:
        config_path = pathlib.Path(config_file)
        source = config_path.resolve()
        place = source.relative_to(chain_folder)
        target = config_dir / place
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _fold_resolved(run_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    """One folder's point records into the fold's, their seeds joined."""
    target_dir = out_dir / RESOLVED_FOLDER
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = (run_dir / RESOLVED_FOLDER).glob("*.json")
    for path in sorted(paths):
        record_text = path.read_text()
        record = json.loads(record_text)
        target = target_dir / path.name
        if target.exists():
            folded_text = target.read_text()
            folded = json.loads(folded_text)
            ranges = folded["seeds"] + record["seeds"]
            record["seeds"] = seed_ranges(ranges)
        write_json(target, record)


def _chain_folder(config_files: tuple) -> pathlib.Path:
    """The deepest folder that holds every file of the extends chain.

    Each copy keeps its place below it, so two files of one name in two
    folders stay two copies, and every copy's `extends` still names its
    base's copy.
    """
    folders = []
    for config_file in config_files:
        config_path = pathlib.Path(config_file)
        source = config_path.resolve()
        folders.append(str(source.parent))
    common_folder = os.path.commonpath(folders)
    return pathlib.Path(common_folder)


def _write_the_manifest(manifest: dict, run_dir: pathlib.Path) -> None:
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
        "compiled_libraries": _compiled_library_hashes(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "argv": sys.argv,
    }


def _built_values(settings) -> dict:
    """The values the build derives from the settings, before it wires.

    The code card at the point's distance with the window sizes a null
    commit_rounds or buffer_rounds resolves to, the rows the plan built
    (the boundary and terminal defaults the escalation row names among
    them), and the run plan: every operation's rounds and windows.
    """
    escalation_policy = escalation_build.build_escalation_policy(
        settings.escalation, settings.weak_decoder
    )
    plan = plan_build.build_plan(settings, escalation_policy)
    code = plan.code
    rows = {
        "layout": collect.json_value(plan.layout),
        "scheme": collect.json_value(plan.scheme),
        "boundary_policy": collect.json_value(plan.boundary_policy),
        "window_interaction": collect.json_value(plan.window_interaction),
        "idle_policy": collect.json_value(plan.idle_policy),
    }
    return {
        "code": collect.json_value(code),
        "commit_rounds": code.commit_rounds(),
        "buffer_rounds": code.buffer_rounds(),
        "rows": rows,
        "run_plan": collect.json_value(plan.run_plan),
    }


def _write_inputs(inputs_dir: pathlib.Path, record) -> None:
    """The workload as files, and each file's sha256 in hashes.json.

    A folder recorded again holds its earlier hashes.json, which is the
    record of the inputs and not one of them.
    """
    workload_files.write_workload(record, inputs_dir)
    hashes_path = inputs_dir / HASHES_FILE
    hashes = {}
    written = inputs_dir.rglob("*")
    for path in sorted(written):
        if not path.is_file() or path == hashes_path:
            continue
        relative = path.relative_to(inputs_dir)
        hashes[str(relative)] = _sha256_of(path)
    write_json(hashes_path, hashes)


def _producer(workload: workload_settings.WorkloadSettings) -> Optional[dict]:
    """A producer row's maker, its arguments and its package's version.

    The name's first dotted word is the package in both forms
    pkgutil.resolve_name reads, module:function and module.function. A
    workload read from files has no maker, so None.
    """
    if workload.kind != "producer":
        return None
    row_settings = workload.row_settings
    function = row_settings.function
    module_name, _, _ = function.partition(":")
    return {
        "function": function,
        "arguments": collect.json_value(row_settings.arguments),
        "version": _package_version(module_name),
    }


def _cells_of(record: dict, paths: list) -> dict:
    """One point's cell at each swept path, the value it ran with.

    The resolved sections hold it: a swept reference as the value it
    names, and a mapping as its child axes changed it.
    """
    sections = record["sections"]
    cells = {}
    for path in paths:
        cells[path] = _resolved_cell(sections, path)
    return cells


def _resolved_cell(sections: Optional[dict], path: str):
    """The cell of a value the point's sections hold, or an empty one."""
    value = sections
    for name in path.split("."):
        if not isinstance(value, Mapping) or name not in value:
            return ""
        value = value[name]
    return _cell_of(value)


def _cell_of(value):
    """A string or a number as itself, any other value as compact json."""
    if isinstance(value, str):
        return value
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if is_number:
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _package_version(module_name: str) -> Optional[str]:
    """The installed version of the package a module belongs to, or None."""
    names = module_name.split(".")
    package = names[0]
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256_of(path: pathlib.Path) -> str:
    contents = path.read_bytes()
    digest = hashlib.sha256(contents)
    return digest.hexdigest()


def _compiled_library_hashes() -> dict:
    """Every library a loader of this process names, each its sha256.

    Keyed by its absolute path, since a loader may take it from outside
    the package. A library is built from tracked C source and is not
    tracked itself, so the commit does not name the bytes a run loaded.
    """
    hashes = {}
    for path in compiled_libraries.paths():
        hashes[str(path)] = _sha256_of(path)
    return hashes


def _untracked_files(checkout: pathlib.Path) -> Optional[list]:
    """The tree's untracked files git does not ignore, which it runs too.

    None when git cannot answer (the container ships none), where
    slurm/slurm_run.sh refuses a tree with untracked files unless told
    to run it anyway.
    """
    listed = _git_output(
        "git",
        "-C",
        str(checkout),
        "ls-files",
        "--others",
        "--exclude-standard",
    )
    if listed is None:
        return None
    return listed.splitlines()


def _untracked_patch(checkout: pathlib.Path, relative: str) -> str:
    """A patch that creates one untracked file; git exits 1 on a difference."""
    arguments = [
        "git",
        "-C",
        str(checkout),
        "diff",
        "--no-index",
        "--binary",
        "--",
        "/dev/null",
        relative,
    ]
    completed = subprocess.run(arguments, capture_output=True, text=True)
    return completed.stdout.rstrip("\n")


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


def _checkout() -> pathlib.Path:
    """The tree this code was imported from, which is the code that ran.

    Not the working directory: a cluster task starts in the folder its
    job was submitted from and may import a checkout pinned at a commit
    somewhere else (docs/how-to/run_a_sweep_on_slurm.md), so a folder
    that named the working directory's commit would name code no part
    of the run read.
    """
    this_file = pathlib.Path(__file__)
    here = this_file.resolve()
    return here.parents[2]


def _git_state() -> dict:
    """The manifest's git block: the one reading this process took.

    gem5 prints its version, its build date, the host and the command
    line at every start, so a result says what produced it
    (gem5 src/python/m5/main.py:524-537), and sinter
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

    Read at the first ask, which every caller takes before its work,
    and reused by every later ask. A run writes its manifest at its
    start and again at its end, and a fold writes one at its end and
    asks at its start (read_the_tree); a tree can move in between. A
    Slurm array running for hours out of a checkout somebody commits
    to would otherwise name, in every folder, whatever the tree held
    when that task finished, which is code no part of the run read. A
    manifest whose commit is not the code's is worse than none.

    Both referents record provenance before the work and not after.
    gem5 prints its version, its build date, its host, its pid and its
    command line at :524-556 of
    gem5 src/python/m5/main.py, then executes the
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


def _commit_from_git_files(checkout: pathlib.Path) -> Optional[str]:
    """The checkout's own HEAD, read without git."""
    git_dir = _git_dir(checkout)
    if git_dir is None:
        return None
    head = _head_of(git_dir)
    if head is None or not head.startswith("ref: "):
        return head
    reference = head[len("ref: ") :]
    common = _common_git_dir(git_dir)
    for directory in (git_dir, common):
        found = _reference_in(directory, reference)
        if found is not None:
            return found
    return None


def _head_of(git_dir: pathlib.Path) -> Optional[str]:
    """What HEAD holds: a commit, or "ref: " and a branch; None if absent."""
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return None
    head_text = head_path.read_text()
    return head_text.strip()


def _reference_in(directory: pathlib.Path, reference: str) -> Optional[str]:
    """One reference in one git directory, loose or packed."""
    reference_path = directory / reference
    if reference_path.exists():
        reference_text = reference_path.read_text()
        return reference_text.strip()
    packed = directory / "packed-refs"
    if not packed.exists():
        return None
    return _packed_reference(packed, reference)


def _git_dir(checkout: pathlib.Path) -> Optional[pathlib.Path]:
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
    return pathlib.Path(text[len(prefix) :])


def _common_git_dir(git_dir: pathlib.Path) -> pathlib.Path:
    """Where a worktree's git dir keeps the refs it shares with the repo."""
    commondir = git_dir / "commondir"
    if not commondir.exists():
        return git_dir
    written = commondir.read_text()
    relative = written.strip()
    common = git_dir / relative
    return common.resolve()


def _packed_reference(packed: pathlib.Path, reference: str) -> Optional[str]:
    packed_text = packed.read_text()
    for line in packed_text.splitlines():
        if line.endswith(reference):
            words = line.split()
            return words[0]
    return None


def _versions() -> dict:
    """Python's version and every installed package's, by name."""
    version_words = sys.version.split()
    python_version = version_words[0]
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        packages[name] = distribution.version
    names = sorted(packages)
    ordered = {}
    for name in names:
        ordered[name] = packages[name]
    return {"python": python_version, "packages": ordered}
