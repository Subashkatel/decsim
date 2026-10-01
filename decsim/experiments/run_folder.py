"""The experiment folder and its run folders: results, config, identity.

results/<utc stamp>-<name>/, or the folder --out names, is an
experiment folder: every piece of every point (pieces/), each point's
values and maker (resolved/) and workload (inputs/), the configurations
it ran (configurations.csv), and one run folder per configuration,
combined/<name>-<id8>/. A run folder holds the config chain, the code
state as a patch, a manifest of the commit, container, host and
packages, and the rows folded from the pieces. gem5 writes m5out/ the
same way, out of the code tree and never overwritten
(src/python/m5/main.py --outdir).
"""

import contextlib
import csv
import datetime
import functools
import hashlib
import importlib.metadata
import io
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterator, Mapping
from typing import Optional

import decsim.build.escalation as escalation_build
import decsim.build.plan as plan_build
import decsim.collect as collect
import decsim.compiled_libraries as compiled_libraries
import decsim.experiments.experiment as experiment
import decsim.frontends.workload_files as workload_files
import decsim.records.program as program_records

RESULTS_DIR = pathlib.Path("results")
RESOLVED_FOLDER = "resolved"
INPUTS_FOLDER = "inputs"
HASHES_FILE = "hashes.json"
COMBINED_FOLDER = "combined"
CONFIGURATIONS_FILE = "configurations.csv"
CONFIGURATION_COLUMNS = ("configuration_id", "name", "config_chain")
# What the launcher saw of the tree it was about to run, for a process
# whose interpreter has no git of its own (a job script exports it):
# "1" dirty, "0" clean, unset means nobody looked.
TREE_DIRTY_VARIABLE = "DECSIM_TREE_DIRTY"
# the modules whose versions every piece names, each with the
# distribution importlib.metadata knows it by. A module's own __version__
# is read first, since a folder of packages can hold a dist-info stale
# against the module beside it; the distribution is read only for one
# that states none (relay_bp). decsim's own version is the commit.
RUN_MODULES = {
    "numpy": "numpy",
    "pymatching": "PyMatching",
    "relay_bp": "relay-bp",
    "scipy": "scipy",
    "sinter": "sinter",
    "stim": "stim",
}
# where Linux names the processor; another system's piece records the
# word platform.processor gives
CPU_INFO_FILE = pathlib.Path("/proc/cpuinfo")


def run_dir_for(config, out_dir=None) -> pathlib.Path:
    """Where this run writes: the folder asked for, or a fresh stamped one.

    gem5's --outdir names the folder and makes it (src/python/m5/main.py:
    102); with no --outdir it writes m5out/. A named folder is reused as
    the caller asked, which is how a collect run again resumes into the
    pieces it saved.
    """
    if out_dir is None:
        return new_run_dir(config)
    run_dir = pathlib.Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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
) -> str:
    """The code state and the manifest, before the first shot; the time.

    config is None for a run no yaml describes (the tools/ examples).
    point_ids are the run's points in task order (write_manifest).
    """
    snapshot_code_state(config, run_dir)
    started_utc = utc_now()
    write_manifest(config, run_dir, point_ids, started_utc)
    return started_utc


def finish_run(
    config: Optional[experiment.ExperimentConfig],
    run_dir: pathlib.Path,
    point_ids: list,
    started_utc: str,
) -> None:
    """The manifest again, with the time the run ended."""
    finished_utc = utc_now()
    write_manifest(config, run_dir, point_ids, started_utc, finished_utc)


def configuration_id(config: experiment.ExperimentConfig) -> str:
    """sha256 of the configuration's sections as json.

    The sweep and the collection are not sections (load_experiment pops
    them), so collecting more points or more shots is the same
    configuration. Two yaml files that resolve to the same sections are
    one configuration, whatever their names, as a point is named by what
    it resolves to (collect.Task.strong_id).
    """
    value = collect.json_value(config.sections)
    text = json.dumps(value, sort_keys=True)
    encoded = text.encode("utf8")
    digest = hashlib.sha256(encoded)
    return digest.hexdigest()


def combined_folder(
    experiment_dir: pathlib.Path, config: experiment.ExperimentConfig
) -> pathlib.Path:
    """The run folder of one recorded configuration: combined/<name>-<id8>/.

    The name is that of the id's first line in configurations.csv
    (recorded_combined_folders), so every yaml of one configuration id,
    whatever its name, folds into one folder.
    """
    identity = configuration_id(config)
    folders = recorded_combined_folders(experiment_dir)
    return folders[identity]


def recorded_combined_folders(experiment_dir: pathlib.Path) -> dict:
    """Each configuration id configurations.csv names, and its run folder.

    The folder takes the name of the id's first line, as a collect of
    that line's yaml names it.
    """
    folders = {}
    for line in _configuration_lines(experiment_dir):
        identity = line["configuration_id"]
        if identity in folders:
            continue
        name = line["name"]
        folders[identity] = _combined_folder_named(
            experiment_dir, name, identity
        )
    return folders


def record_configuration(
    experiment_dir: pathlib.Path, config: experiment.ExperimentConfig
) -> None:
    """The configuration's row in configurations.csv, once.

    A row is its id, its name and its config chain, nearest file first,
    so a status over the experiment finds every configuration it ran.
    The csv module quotes a name or a path that holds a comma or a
    quote, and the chain is a json list, so a path that holds any
    character reads back whole, as sinter writes its json_metadata cell
    (sinter/_data/_csv_out.py:8-13, 34-57).
    """
    path = experiment_dir / CONFIGURATIONS_FILE
    rows = []
    if path.is_file():
        rows = _configuration_lines(experiment_dir)
    identity = configuration_id(config)
    chain = []
    for config_file in config.config_files:
        chain.append(str(config_file))
    chain_text = json.dumps(chain)
    row = {
        "configuration_id": identity,
        "name": config.name,
        "config_chain": chain_text,
    }
    if row in rows:
        return
    rows.append(row)
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=CONFIGURATION_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    written = text.getvalue()
    with staged_replacement(path) as staging:
        staging.write_text(written)


def recorded_configurations(experiment_dir: pathlib.Path) -> dict:
    """Each configuration id configurations.csv names, and its yamls, loaded.

    A yaml is loaded from the nearest file of its chain. One id may have
    several yamls, which split one configuration's sweep between them
    (the sweep is no part of the id).
    """
    files_by_id = {}
    for line in _configuration_lines(experiment_dir):
        chain = json.loads(line["config_chain"])
        files = files_by_id.setdefault(line["configuration_id"], [])
        if chain[0] not in files:
            files.append(chain[0])
    configurations = {}
    for identity, files in files_by_id.items():
        configurations[identity] = [
            experiment.load_experiment(file) for file in files
        ]
    return configurations


def piece_identity() -> dict:
    """What a piece records of the process that ran it: code, host, job.

    Beside the commit, a rerun needs the interpreter and the packages
    that sampled and decoded, and the processor that set the seconds a
    shot; the array job and task name the Slurm task that ran it, which
    the round's plan.csv maps back to its pieces.
    """
    commit, is_dirty = _tree_reading()
    python_version = platform.python_version()
    packages = _run_module_versions()
    cpu_model = _cpu_model()
    return {
        "commit": commit,
        "dirty": is_dirty,
        "python": python_version,
        "packages": packages,
        "host": platform.node(),
        "cpu_model": cpu_model,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


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


def write_manifest(
    config: Optional[experiment.ExperimentConfig],
    run_dir: pathlib.Path,
    point_ids: list,
    started_utc: str,
    finished_utc: Optional[str] = None,
) -> None:
    """Write the run's identity: enough to interpret or reproduce it.

    Sampling is deterministic from (stim version, circuit, distance,
    rounds, p, seed), so the manifest plus seeds are the raw data. config
    is None for a run no yaml describes (tools/deltakit_example.py),
    whose every value is in resolved/. point_ids are the sweep's points
    in task order, each the name of its resolved/ record, which is the
    order a fold writes the rows in.
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

    point_record says what resolved/<id>.json holds; inputs/<id>/ holds
    the workload as the files row reads it (write_point_record).
    Returns the id.
    """
    record = point_record(task, seeds, sections)
    return write_point_record(run_dir, task, record)


def point_record(
    task: collect.Task,
    seeds: Optional[list] = None,
    sections: Optional[Mapping] = None,
    experiment_facts: Optional[Mapping] = None,
) -> dict:
    """What resolved/<id>.json holds of one sweep point, built, not written.

    The metadata, the seed ranges run, the sections the point's yaml
    resolved to (its axes placed and its references resolved, as Hydra
    keeps each job's composed config in .hydra/config.yaml), the maker
    the workload's row says it called with the point's own arguments,
    every setting and the values the build derives; a point a Python
    caller built has no sections. An experiment's point also holds
    experiment_facts, what its fold needs besides its pieces
    (collect_command). Building runs the point's build, so a point the
    build refuses is refused here, before anything is written.
    """
    settings = task.settings
    shot_settings = task.shot_settings()
    maker = settings.workload.maker()
    plan = _plan(shot_settings)
    resolved = {
        "id": task.strong_id(),
        "metadata": collect.json_value(task.metadata),
        "seeds": seeds,
        "sections": collect.json_value(sections),
        "maker": collect.json_value(maker),
        "settings": collect.json_value(settings),
        "built": _built_values(plan),
    }
    resolved["rounds_per_shot"] = _rounds_per_shot(plan)
    if experiment_facts is not None:
        resolved["experiment"] = collect.json_value(experiment_facts)
    return resolved


def write_point_record(
    run_dir: pathlib.Path, task: collect.Task, record: dict
) -> str:
    """A point's record in resolved/<id>.json, and its workload in inputs/.

    inputs/<id>/ holds the workload as the files row reads it, with each
    file's sha256 in hashes.json, so a rerun needs no maker installed.
    Returns the id.
    """
    point_id = record["id"]
    resolved_dir = run_dir / RESOLVED_FOLDER
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = resolved_dir / f"{point_id}.json"
    write_json(resolved_path, record)
    workload_record = task.settings.workload.workload_record
    if workload_record is not None:
        inputs_dir = run_dir / INPUTS_FOLDER / point_id
        _write_inputs(inputs_dir, workload_record)
    return point_id


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
    """Seed ranges as [first, how many], in order, touching ones joined."""
    joined = []
    for first, count in sorted(ranges):
        if joined and sum(joined[-1]) == first:
            joined[-1][1] += count
            continue
        joined.append([first, count])
    return joined


def copy_points_of(
    experiment_dir: pathlib.Path,
    point_ids: list,
    seeds_by_point: dict,
    out_dir: pathlib.Path,
) -> None:
    """These points' resolved/ records and inputs/ into a run folder.

    Each record's seeds are the ranges its pieces hold, so the run
    folder says which shots its rows are.
    """
    target_dir = out_dir / RESOLVED_FOLDER
    target_dir.mkdir(parents=True, exist_ok=True)
    for point_id in point_ids:
        source = experiment_dir / RESOLVED_FOLDER / f"{point_id}.json"
        record_text = source.read_text()
        record = json.loads(record_text)
        record["seeds"] = seeds_by_point.get(point_id, [])
        target = target_dir / source.name
        write_json(target, record)
        _copy_the_inputs(experiment_dir, point_id, out_dir)


def write_json(path: pathlib.Path, value) -> None:
    """A json file of the run folder: indented, ending in a newline."""
    text = json.dumps(value, indent=2)
    lines = text + "\n"
    path.write_text(lines)


def publish_the_fold(staging: pathlib.Path, run_dir: pathlib.Path) -> None:
    """A fold built whole in staging moved into run_dir, the last one out.

    A fold writes the folder's csv files and its points' resolved/ and
    inputs/ copies, all derived from the experiment folder, so the last
    fold's are removed and the new ones moved in. The manifest, the
    code state and the traced shots' files are not a fold's and stay.
    Another folder of the same configuration id, which a collect of a
    second yaml name wrote before one folder was kept per id, holds the
    same points, so its fold is removed too and each point is counted in
    one folder.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    _remove_the_fold(run_dir)
    for path in staging.iterdir():
        target = run_dir / path.name
        path.rename(target)
    _name, identity8 = run_dir.name.rsplit("-", 1)
    for folder in run_dir.parent.glob(f"*-{identity8}"):
        if folder != run_dir:
            _remove_the_fold(folder)


@contextlib.contextmanager
def staged_replacement(path: pathlib.Path) -> Iterator[pathlib.Path]:
    """A hidden name beside path to write, renamed over path at the end.

    Two tasks of one experiment may replace one file at once, so each
    stages its copy under a random name, and neither moves the other's.
    The rename is atomic (rename(2)), so a reader sees the old file or
    the new one whole, never a part.
    """
    identifier = uuid.uuid4()
    token = identifier.hex
    staging = path.with_name(f".{path.name}.{token}.partial")
    yield staging
    os.replace(staging, path)


def utc_now() -> str:
    """This moment as an iso timestamp, for the manifest's times."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.isoformat()


def _combined_folder_named(
    experiment_dir: pathlib.Path, name: str, identity: str
) -> pathlib.Path:
    folder_name = f"{name}-{identity[:8]}"
    return experiment_dir / COMBINED_FOLDER / folder_name


def _remove_the_fold(run_dir: pathlib.Path) -> None:
    """A run folder's csv files, resolved/ and inputs/, removed."""
    for csv_path in run_dir.glob("*.csv"):
        csv_path.unlink()
    for folder_name in (RESOLVED_FOLDER, INPUTS_FOLDER):
        folder = run_dir / folder_name
        shutil.rmtree(folder, ignore_errors=True)


def _configuration_lines(experiment_dir: pathlib.Path) -> list:
    """configurations.csv's lines, each a dict of its columns."""
    path = experiment_dir / CONFIGURATIONS_FILE
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


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


def _copy_the_inputs(
    experiment_dir: pathlib.Path, point_id: str, out_dir: pathlib.Path
) -> None:
    """One point's inputs/ folder, when its workload was written as files."""
    source = experiment_dir / INPUTS_FOLDER / point_id
    if not source.is_dir():
        return
    target = out_dir / INPUTS_FOLDER / point_id
    shutil.copytree(source, target, dirs_exist_ok=True)


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


def _plan(settings) -> plan_build.Plan:
    """The plan the build derives from the settings, before it wires."""
    escalation_policy = escalation_build.build_escalation_policy(
        settings.escalation, settings.weak_decoder
    )
    return plan_build.build_plan(settings, escalation_policy)


def _built_values(plan: plan_build.Plan) -> dict:
    """The values the build derives from the settings, before it wires.

    The code card at the point's distance with the window sizes a null
    commit_rounds or buffer_rounds resolves to, the rows the plan built
    (the boundary and terminal defaults the escalation row names among
    them), and the run plan: every operation's rounds and windows.
    """
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


def _rounds_per_shot(plan: plan_build.Plan) -> int:
    """A shot's QEC rounds as planned: every patch's rounds, added up.

    They size a point's pieces before any shot runs. A live stream also
    idles through its feedback wait, rounds only its run knows, so a
    shot's own count is measured (measure.py executed_rounds).

    A round is one patch's syndrome extraction, whenever its operation
    starts. Tesseract 2503.10988 lines 287-290 set a two-code shot's r
    to the rounds across both codes, so its per-round rate compares
    with one memory; Litinski 1808.02892 Eq. 11 (lines 1408-1414) adds
    qubits times cycles. The QPU fires the workload's operations, and an
    operation with no detector data sends no syndrome
    (qpu/cycle_clock.py _emit_operation_rounds); a decode operation or a
    stream is how the decoder reads those rounds, so it adds none.
    """
    round_counts = {}
    for resolved in plan.run_plan.resolved_operations:
        round_counts[resolved.operation_id] = resolved.round_count
    patch_rounds = 0
    for operation in plan.operations:
        if not operation.emits_detector_data:
            continue
        patches = program_records.patches_of(operation)
        patch_rounds += round_counts[operation.id] * len(patches)
    return patch_rounds


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

    None when git cannot answer (the container ships none).
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
    """What the launcher saw, when it exported it (TREE_DIRTY_VARIABLE)."""
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


def _run_module_versions() -> dict:
    """Each of RUN_MODULES's version, by module name."""
    versions = {}
    for module_name, distribution_name in RUN_MODULES.items():
        version = _module_version(module_name, distribution_name)
        versions[module_name] = version
    return versions


def _module_version(module_name: str, distribution_name: str) -> Optional[str]:
    """The imported module's version; None for one this run never imported."""
    module = sys.modules.get(module_name)
    if module is None:
        return None
    if hasattr(module, "__version__"):
        return module.__version__
    return importlib.metadata.version(distribution_name)


def _cpu_model() -> str:
    """The model name /proc/cpuinfo lists, else the platform's word."""
    cpu_info = ""
    if CPU_INFO_FILE.is_file():
        cpu_info = CPU_INFO_FILE.read_text()
    for line in cpu_info.splitlines():
        label, _separator, value = line.partition(":")
        if label.strip() == "model name":
            return value.strip()
    return platform.processor()


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
