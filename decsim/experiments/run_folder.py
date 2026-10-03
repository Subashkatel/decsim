"""The results folder: one experiment's records, pieces and rows.

results/<date>_<name>/, or the folder --out names, holds one
experiment: its run file copied, the code state as a patch, run.json
(the commit, host, packages and the points in order), each point's
record and workload under points/<name>/ (machine.json, inputs/),
every piece of every point (pieces/), and the rows folded from the
pieces. gem5 writes m5out/ the same way, out of the code tree, with
the config of what ran beside its results (src/python/m5/main.py
--outdir; src/python/m5/simulate.py:95-144 config.json).
"""

import contextlib
import datetime
import functools
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import sys
import uuid
from collections.abc import Iterator, Mapping
from typing import Any, Optional

import decsim.build.escalation as escalation_build
import decsim.build.plan as plan_build
import decsim.collect as collect
import decsim.compiled_libraries as compiled_libraries
import decsim.engine as engine_module
import decsim.experiments.experiment as experiment
import decsim.experiments.refusal as refusal
import decsim.frontends.workload_files as workload_files
import decsim.machine as machine_module
import decsim.records.program as program_records
import decsim.records.results as result_records
import decsim.settings as machine_settings

RESULTS_DIR = pathlib.Path("results")
RUN_FILE = "run.json"
POINTS_FOLDER = "points"
RECORD_FILE = "machine.json"
INPUTS_FOLDER = "inputs"
HASHES_FILE = "hashes.json"
CONFIG_FOLDER = "config"
# What the launcher saw of the tree it was about to run, for a process
# whose interpreter has no git of its own (a job script exports it):
# "1" dirty, "0" clean, unset means nobody looked.
TREE_DIRTY_VARIABLE = "DECSIM_TREE_DIRTY"
# Set, it lets a run go on from a tree git does not vouch for.
ALLOW_DIRTY_VARIABLE = "ALLOW_DIRTY"
# what open gives a new file before the umask filters it
ORDINARY_FILE_MODE = 0o666
# where Linux names the processor; another system's piece records the
# word platform.processor gives
PROCESSOR_INFO_FILE = pathlib.Path("/proc/cpuinfo")
# What git reads as the code: the whole checkout but its results folder.
# A run writes run.json and the run file's copy there, which .gitignore
# keeps tracked as evidence, so they would make every later reading
# dirty; dirty says the code differs from its commit, and results are
# not code.
CODE_PATHSPEC = ("--", ".", f":(exclude){RESULTS_DIR}")


def run_dir_for(name: str, out_dir=None) -> pathlib.Path:
    """Where this run writes: the folder asked for, or a new dated one.

    gem5's --outdir names the folder and makes it (src/python/m5/main.py:
    102). A named folder is reused as the caller asked, which is how a
    run started again resumes into the pieces it saved; a launcher fixes
    the folder once and hands it to every task.
    """
    if out_dir is None:
        return new_run_dir(name)
    run_dir = pathlib.Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def new_run_dir(name: str) -> pathlib.Path:
    """results/<date>_<name>/, never reused; a second one that day gets _2.

    The date is the launcher's local date, so the folder sorts by the day
    the experiment started, as Hydra's outputs/<date>/ does.
    """
    today = datetime.date.today()
    date_text = today.isoformat()
    base = RESULTS_DIR / f"{date_text}_{name}"
    run_dir = base
    suffix = 2
    while True:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            run_dir = base.with_name(f"{base.name}_{suffix}")
            suffix += 1


def start_run(
    run_dir: pathlib.Path, run_file: Optional[pathlib.Path], point_ids: list
) -> str:
    """The code state, the run file and run.json, before the first shot.

    run_file is None for a run no file describes (the tools/ examples).
    point_ids are the run's points in order (write_run_record). Returns
    the start time.
    """
    snapshot_code_state(run_file, run_dir)
    started_utc = utc_now()
    write_run_record(run_dir, run_file, point_ids, started_utc)
    return started_utc


def accept_raised_stop_rules(
    run_dir: pathlib.Path, run_file: pathlib.Path, point_ids: list
) -> None:
    """A changed run file naming the folder's points replaces its copy.

    A point's id hashes its machine and metadata and not its stop rule,
    so a run file whose points have the ids run.json recorded differs
    only in how far they run: a pilot's caps raised for the final run,
    which goes on from the saved pieces. Its copy replaces the folder's;
    any other change is refused when the copy is written (_copy_once).
    A run whose point ids are not content hashes never calls this.
    """
    run_path = run_dir / RUN_FILE
    if not run_path.is_file():
        return
    recorded = read_json(run_path)
    if recorded["points"] != list(point_ids):
        return
    for source, target in _run_file_copies(run_file, run_dir):
        if not target.exists():
            continue
        text = source.read_text()
        with staged_replacement(target) as staging:
            staging.write_text(text)


def finish_run(
    run_dir: pathlib.Path,
    run_file: Optional[pathlib.Path],
    point_ids: list,
    started_utc: str,
) -> None:
    """run.json again, with the time the run ended."""
    finished_utc = utc_now()
    write_run_record(run_dir, run_file, point_ids, started_utc, finished_utc)


def refuse_another_tree(run_dir: pathlib.Path) -> None:
    """A folder whose run.json names another commit, or the same one dirty.

    A folder's rows pool the shots of every run into it, so they must
    have run one simulator; a run asks before it writes anything. A
    dirty flag nobody could read says nothing either way, so only two
    read flags that differ are refused. A new folder from a tree whose
    commit cannot be read is refused unless ALLOW_DIRTY_VARIABLE is set,
    since its results could not say what ran.
    """
    run_path = run_dir / RUN_FILE
    if not run_path.is_file():
        _refuse_an_unread_commit()
        return
    recorded = read_json(run_path)
    recorded_git = recorded["git"]
    this_git = _git_state()
    is_same_commit = recorded_git["commit"] == this_git["commit"]
    flags = (recorded_git["dirty"], this_git["dirty"])
    is_same_flag = None in flags or flags[0] == flags[1]
    if is_same_commit and is_same_flag:
        return
    raise refusal.RefusalError(
        f"{run_dir} holds a run of commit {recorded_git['commit']} dirty "
        f"{recorded_git['dirty']}, and this is commit {this_git['commit']} "
        f"dirty {this_git['dirty']}; a folder holds one tree's results, so "
        "give --out a new folder"
    )


def _refuse_an_unread_commit() -> None:
    """A tree whose commit neither git nor its .git files can name."""
    commit, _is_dirty = _tree_reading()
    if commit is not None or os.environ.get(ALLOW_DIRTY_VARIABLE):
        return
    checkout = _checkout()
    message = (
        f"the commit of the decsim tree at {checkout} cannot be read, so "
        "run.json could not say what ran; run from a git checkout, whose "
        f".git folder holds its HEAD, or set {ALLOW_DIRTY_VARIABLE}=1"
    )
    raise refusal.RefusalError(message)


def piece_identity() -> dict:
    """What a piece records of the process that ran it: code, host, job.

    Beside the commit, a rerun needs the interpreter and the processor
    that set the seconds a shot; the array job and task name the Slurm
    task that ran it, which the batch's plan.csv maps back to its
    pieces. The package versions come from the process that ran the
    shots, with its measurements (collect.imported_module_versions).
    """
    commit, is_dirty = _tree_reading()
    python_version = platform.python_version()
    processor_model = _processor_model()
    return {
        "commit": commit,
        "dirty": is_dirty,
        "python": python_version,
        "host": platform.node(),
        "processor_model": processor_model,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def snapshot_code_state(
    run_file: Optional[pathlib.Path], run_dir: pathlib.Path
) -> None:
    """Copy the run's exact inputs next to its results.

    The run file is copied (_copy_the_run_file), and any uncommitted code
    goes into code_state.patch, an untracked file as a patch that creates
    it (git diff --no-index), so run.json's commit + patch + run file =
    the whole experiment. The patch is the imported tree's, for the
    reason run.json's commit is (_checkout).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    if run_file is not None:
        _copy_the_run_file(run_file, run_dir)
    checkout = _checkout()
    diff = _git_output(
        "git", "-C", str(checkout), "diff", "HEAD", *CODE_PATHSPEC
    )
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


def write_run_record(
    run_dir: pathlib.Path,
    run_file: Optional[pathlib.Path],
    point_ids: list,
    started_utc: str,
    finished_utc: Optional[str] = None,
) -> None:
    """run.json: what ran, where, and with which code and packages.

    Sampling is deterministic from (stim version, circuit, distance,
    rounds, p, seed), so run.json plus the seeds are the raw data.
    run_files are the run file and, for a yaml, the bases its extends
    chain reads, none for a run no file describes (the tools/ examples).
    point_ids are the run's points in order, which is the order a fold
    writes the rows in.
    """
    run_files = []
    if run_file is not None:
        for path in experiment.run_files(run_file):
            run_files.append(str(path))
    record = {"run_files": run_files, "points": point_ids}
    how_it_ran = _how_it_ran()
    record.update(how_it_ran)
    record["started_utc"] = started_utc
    record["finished_utc"] = finished_utc
    run_path = run_dir / RUN_FILE
    with staged_replacement(run_path) as staging:
        write_json(staging, record)


def recorded_run_file(run_dir: pathlib.Path) -> pathlib.Path:
    """The run file run.json names, which a planned task loads again."""
    run_path = run_dir / RUN_FILE
    recorded = read_json(run_path)
    run_files = recorded["run_files"]
    return pathlib.Path(run_files[0])


def recorded_point_ids(run_dir: pathlib.Path, first_ids: list) -> list:
    """Every recorded point's id: first_ids' recorded ones, then the rest.

    The rest are points an earlier run of the folder recorded, which a
    fold still counts, in the order of their names.
    """
    records = point_records(run_dir)
    ordered = []
    for point_id in first_ids:
        if point_id in records:
            ordered.append(point_id)
    others = []
    for point_id, record in records.items():
        if point_id not in ordered:
            others.append((record["name"], point_id))
    for _name, point_id in sorted(others):
        ordered.append(point_id)
    return ordered


def record_point(
    run_dir: pathlib.Path,
    name: str,
    task: collect.Task,
    seeds: Optional[list] = None,
    sections: Optional[Mapping] = None,
) -> str:
    """One point's values and workload, under points/<name>/.

    point_record says what machine.json holds; inputs/ holds the
    workload as the files row reads it (write_point_record). Returns
    the point's id.
    """
    record = point_record(name, task, seeds, sections)
    return write_point_record(run_dir, task, record)


def point_record(
    name: str,
    task: collect.Task,
    seeds: Optional[list] = None,
    sections: Optional[Mapping] = None,
    experiment_facts: Optional[Mapping] = None,
) -> dict:
    """What a point's machine.json holds, built, not written.

    Its id and name, the metadata, the seed ranges run, the sections a
    yaml point resolved to (its axes placed and its references resolved,
    as Hydra keeps each job's composed config in .hydra/config.yaml),
    the maker the workload's row says it called with the point's own
    arguments, every setting and the values the build derives, as gem5's
    config.json holds every parameter (src/python/m5/SimObject.py:1175);
    a point built in Python has no sections. An experiment's point also
    holds experiment_facts, what its fold needs besides its pieces
    (collect_command). Building runs the point's build, so a point the
    build refuses is refused here, before anything is written.
    """
    settings = task.settings
    maker = settings.workload.maker()
    plan = _plan(task)
    record = {
        "id": task.strong_id(),
        "name": name,
        "metadata": collect.json_value(task.metadata),
        "seeds": seeds,
        "sections": collect.json_value(sections),
        "maker": collect.json_value(maker),
        "settings": collect.json_value(settings),
        "built": _built_values(plan),
    }
    record["rounds_per_shot"] = _rounds_per_shot(plan)
    if experiment_facts is not None:
        record["experiment"] = collect.json_value(experiment_facts)
    return record


def write_point_record(
    run_dir: pathlib.Path, task: collect.Task, record: dict
) -> str:
    """A point's machine.json, and its workload in inputs/, in its folder.

    inputs/ holds the workload as the files row reads it, with each
    file's sha256 in hashes.json, so a rerun needs no maker installed.
    Returns the point's id.
    """
    point_dir = run_dir / POINTS_FOLDER / record["name"]
    point_dir.mkdir(parents=True, exist_ok=True)
    record_path = point_dir / RECORD_FILE
    write_json(record_path, record)
    workload_record = task.settings.workload.workload_record
    if workload_record is not None:
        inputs_dir = point_dir / INPUTS_FOLDER
        _write_inputs(inputs_dir, workload_record)
    return record["id"]


def write_shot(
    machine: machine_module.Machine,
    settings: machine_settings.MachineSettings,
    run_dir: pathlib.Path,
    label: str,
    result: result_records.RunResult,
) -> None:
    """One narrated shot's files in its folder, the log and trace named label.

    result.json always, and the log and the trace when the observation
    asks for them. tools/deltakit_example.py and
    tools/live_memory_example.py write their shot through it too.
    """
    _write_the_log_and_trace(machine, settings, run_dir, label)
    _write_result(result, run_dir)


def point_records(run_dir: pathlib.Path) -> dict:
    """Each point's machine.json, keyed by its point id."""
    records = {}
    points_dir = pathlib.Path(run_dir) / POINTS_FOLDER
    paths = points_dir.glob(f"*/{RECORD_FILE}")
    for path in sorted(paths):
        record = read_json(path)
        records[record["id"]] = record
    return records


def inputs_dir_of(run_dir: pathlib.Path, record: Mapping) -> pathlib.Path:
    """Where a recorded point's workload files are."""
    point_dir = run_dir / POINTS_FOLDER / record["name"]
    return point_dir / INPUTS_FOLDER


def record_seeds(run_dir: pathlib.Path, seeds_by_point: dict) -> None:
    """Each point's machine.json given the seed ranges its pieces hold.

    A fold writes them, so the record says which shots the rows count.
    """
    records = point_records(run_dir)
    for point_id, record in records.items():
        record["seeds"] = seeds_by_point.get(point_id, [])
        point_dir = run_dir / POINTS_FOLDER / record["name"]
        record_path = point_dir / RECORD_FILE
        with staged_replacement(record_path) as staging:
            write_json(staging, record)


def swept_values(run_dir: pathlib.Path, point_ids: list) -> dict:
    """Each point's value at every swept name, as csv cells.

    A swept name is a key of a point's metadata. One column per name is
    Wickham's tidy table, each variable a column and each observation a
    row (Tidy Data, J. Stat. Softw. 59(10), 2014, section 2.3); the
    names come in the order the points first set them. A point built in
    Python has its metadata's value there. A yaml point's metadata names
    yaml paths, and its cell is the value its sections resolved to,
    whether its block set the path or not, so a swept reference is the
    value it names; a name the point does not hold is an empty cell. A
    value other than a string or a number is one cell of compact json,
    as sinter writes json_metadata (sinter/_data/_csv_out.py:35-37); the
    typed value is in the record.
    """
    records = point_records(run_dir)
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
    """The named parts of a point's record, each value at its dotted path.

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


def write_json(path: pathlib.Path, value) -> None:
    """A json file of the run folder: indented, ending in a newline."""
    text = json.dumps(value, indent=2)
    lines = text + "\n"
    path.write_text(lines)


def read_json(path: pathlib.Path) -> Any:
    """One json file of the run folder."""
    text = path.read_text()
    return json.loads(text)


def publish_the_fold(staging: pathlib.Path, run_dir: pathlib.Path) -> None:
    """A fold built whole in staging moved into run_dir, the last one out.

    A fold writes the folder's csv files, all derived from the pieces
    and the records, so the last fold's are removed and the new ones
    moved in. run.json, the code state, the records, the pieces and
    the traced shots' files are not a fold's and stay.
    """
    for csv_path in run_dir.glob("*.csv"):
        csv_path.unlink()
    for path in staging.iterdir():
        target = run_dir / path.name
        path.rename(target)


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
    """This moment as an iso timestamp, for run.json's times."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.isoformat()


def _write_result(
    result: result_records.RunResult, run_dir: pathlib.Path
) -> None:
    """result.json: every field of the shot's result record.

    A record's class names a setting's identity and not a result's, so
    result.json holds the fields alone.
    """
    value = collect.json_value(result, record_classes=False)
    result_path = run_dir / "result.json"
    write_json(result_path, value)


def _write_the_log_and_trace(
    machine: machine_module.Machine,
    settings: machine_settings.MachineSettings,
    run_dir: pathlib.Path,
    label: str,
) -> None:
    """The shot's log and trace, each where its knob says."""
    observation = settings.observation
    if observation.writes_log:
        log_dir = run_dir / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        text = "\n".join(machine.observation.log.lines)
        log_path = log_dir / f"{label}.log"
        contents = text + "\n"
        log_path.write_text(contents)
    if not observation.writes_trace:
        return
    trace_path = _trace_path(observation, run_dir, label)
    machine.observation.trace_writer.write(str(trace_path))


def _trace_path(observation, run_dir: pathlib.Path, label: str) -> pathlib.Path:
    """Where this shot's trace goes: the named path, or trace/ in the folder."""
    named = observation.trace_path
    if named is not None:
        return pathlib.Path(named)
    trace_dir = run_dir / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir / f"{label}.trace.json"


def _copy_the_run_file(run_file: pathlib.Path, run_dir: pathlib.Path) -> None:
    """The run file beside the results, or a yaml's whole chain in config/.

    A Python run file is copied under its own name, so a plot script
    loads the copy that made the rows. A yaml goes into config/ with
    every base of its extends chain, each at its place relative to the
    others, so every `extends` still resolves. Each copy is written once
    (_copy_once), so a folder's copy is the file that made its rows.
    """
    for source, target in _run_file_copies(run_file, run_dir):
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_once(source, target)


def _run_file_copies(run_file: pathlib.Path, run_dir: pathlib.Path) -> list:
    """Each file the run reads, paired with where its copy goes."""
    run_files = experiment.run_files(run_file)
    if len(run_files) == 1 and run_file.suffix == ".py":
        target = run_dir / run_file.name
        return [(run_file, target)]
    config_dir = run_dir / CONFIG_FOLDER
    chain_folder = _chain_folder(run_files)
    copies = []
    for config_file in run_files:
        config_path = pathlib.Path(config_file)
        source = config_path.resolve()
        place = source.relative_to(chain_folder)
        target = config_dir / place
        copies.append((source, target))
    return copies


def _copy_once(source: pathlib.Path, target: pathlib.Path) -> None:
    """The source's text at target, or a refusal if target holds another.

    The first run into a folder writes the copy; a later one, a resume
    or another array task, finds it and must bring the same text, since
    a folder's rows come from one run file. The text is staged under a
    random name and hard-linked into place, which fails when the target
    exists, so a task starting beside another never reads half a file.
    """
    text = source.read_text()
    if not target.exists():
        _link_into_place(text, target)
    held_text = target.read_text()
    if held_text == text:
        return
    raise refusal.RefusalError(
        f"{target} holds another {target.name} than {source}; a folder "
        "holds the results of one run file, so give --out a new folder"
    )


def _link_into_place(text: str, target: pathlib.Path) -> None:
    """The text written beside target, then linked to it if still free.

    The staging file is created with mode 0666 filtered by the umask, as
    open creates a file, so a group sharing the folder can read it.
    """
    identifier = uuid.uuid4()
    staging = target.with_name(f".{target.name}.{identifier.hex}.partial")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(staging, flags, ORDINARY_FILE_MODE)
    try:
        with os.fdopen(descriptor, "w") as staging_file:
            staging_file.write(text)
        with contextlib.suppress(FileExistsError):
            os.link(staging, target)
    finally:
        staging.unlink()


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


def _how_it_ran() -> dict:
    """Where and by what this process ran, for run.json."""
    return {
        "git": _git_state(),
        "container": _container(),
        "versions": _versions(),
        "compiled_libraries": _compiled_library_hashes(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "argv": sys.argv,
    }


def rounds_per_shot(task: collect.Task) -> int:
    """A shot's QEC rounds as the point's plan gives them (_rounds_per_shot)."""
    plan = _plan(task)
    return _rounds_per_shot(plan)


def _plan(task: collect.Task) -> plan_build.Plan:
    """The plan the build derives from the task, before it wires."""
    settings = task.settings
    engine = engine_module.Engine()
    switching = escalation_build.build_switching(
        settings.switching,
        settings.weak_decoder,
        engine,
        task.online_threshold,
    )
    return plan_build.build_plan(
        settings.qpu,
        settings.workload,
        settings.windows,
        settings.idle_policy,
        settings.detection_events,
        settings.switching,
        settings.decoder_manager.bulk_strong,
        switching,
    )


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
    """One point's cell at each swept name, the value it ran with.

    A yaml point's resolved sections hold it: a swept reference as the
    value it names, and a mapping as its child axes changed it. A point
    built in Python has no sections, and its metadata holds it.
    """
    sections = record["sections"]
    if sections is None:
        return _metadata_cells(record["metadata"], paths)
    cells = {}
    for path in paths:
        cells[path] = _resolved_cell(sections, path)
    return cells


def _metadata_cells(metadata: Mapping, paths: list) -> dict:
    """A Python point's cells: its metadata, empty where it has none."""
    cells = {}
    for path in paths:
        cells[path] = ""
        if path in metadata:
            cells[path] = _cell_of(metadata[path])
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
        *CODE_PATHSPEC,
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
    """run.json's git block: the one reading this process took.

    gem5 prints its version, its build date, the host and the command
    line at every start, so a result says what produced it
    (gem5 src/python/m5/main.py:524-537), and sinter
    carries the decoder and the task's metadata in every row of its csv
    for the same reason (sinter/_data/_task_stats.py:196-204 through
    _data/_csv_out.py:56-65). decsim's run folder is where that belongs
    here, so run.json names the commit and says whether the tree had
    uncommitted changes.
    """
    commit, is_dirty = _tree_reading()
    return {"commit": commit, "dirty": is_dirty}


@functools.lru_cache(maxsize=1)
def _tree_reading() -> tuple:
    """(commit, dirty) of the tree this code came from, read once.

    Read at the first ask, which every caller takes before its work,
    and reused by every later ask. A run writes run.json at its start
    and again at its end; a tree can move in between. A
    Slurm array running for hours out of a checkout somebody commits
    to would otherwise name, in every folder, whatever the tree held
    when that task finished, which is code no part of the run read. A
    record whose commit is not the code's is worse than none.

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
        is_dirty = _code_is_dirty(checkout)
    return commit, is_dirty


def fresh_tree_reading() -> tuple:
    """(checkout, commit, dirty) of the tree, read now from git itself.

    Not the cached reading and not the launcher's: a launcher and every
    task it starts each look at the tree as they start. commit and dirty
    are None where git could not answer.
    """
    checkout = _checkout()
    commit = _git_output("git", "-C", str(checkout), "rev-parse", "HEAD")
    is_dirty = _code_is_dirty(checkout)
    return checkout, commit, is_dirty


def _code_is_dirty(checkout: pathlib.Path) -> Optional[bool]:
    """Whether the code differs from its commit; None when git is silent."""
    porcelain = _git_output(
        "git", "-C", str(checkout), "status", "--porcelain", *CODE_PATHSPEC
    )
    if porcelain is None:
        return None
    return bool(porcelain)


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


def _processor_model() -> str:
    """The model name /proc/cpuinfo lists, else the platform's word."""
    processor_info = ""
    if PROCESSOR_INFO_FILE.is_file():
        processor_info = PROCESSOR_INFO_FILE.read_text()
    for line in processor_info.splitlines():
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
