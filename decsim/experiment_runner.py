"""One experiment script's points, its command line and its results folder.

An experiment is a Python script that adds its points to an Experiment
and calls main. The command line is gem5 MultiSim's (gem5 v24.0
RELEASE-NOTES.md, "gem5 MultiSim"): `run.py --list` names every point by
id, `run.py <id>` runs one, and `run.py` runs them all, so one Slurm
array task runs one point (slurm/run.sbatch). An offline point is one
sinter task that sinter.collect runs to its stop rule, each point into
its own resume CSV, so array tasks running at once never write one
file and a resubmitted task continues where it stopped. A function
point is a Python function whose rows the runner saves whole, once it
returns, so a resubmitted task runs only the points with no CSV yet.
An experiment's points are all of one kind. The folder keeps the
script and the commit that ran, as gem5 keeps config.ini and ns-3 SEM
the commit with every run.
"""

import argparse
import csv
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import secrets
import sys
import time
from collections.abc import Callable, Mapping
from typing import Optional

import sinter
import stim

import decsim.experiments.run_folder as run_folder

# `run.py combine` folds every point's CSV into stats.csv, or a function
# point's into the results file it names.
COMBINE = "combine"
POINTS_FOLDER = "points"
STATS_FILE = "stats.csv"
COMMIT_FILE = "commit.txt"
ONE_KIND_PER_EXPERIMENT = (
    "an experiment's points are all sinter tasks or all function points, "
    "so its folder holds one kind of results"
)
ONE_RUN_PER_FOLDER = (
    "a results folder belongs to one script and one commit, so give --out "
    "a new folder"
)
# what open gives a new file before the umask filters it
ORDINARY_FILE_MODE = 0o666
# where a folder goes when no --out names one
RESULTS_ROOT_VARIABLE = "DECSIM_RESULTS"
DEFAULT_RESULTS_ROOT = "results"


class Experiment:
    """The points one experiment script adds, run from its command line."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.tasks = []
        self.decoders = []
        self.function_points = []

    def add_offline(
        self,
        circuit: stim.Circuit,
        decoder: str,
        labels: Mapping,
        max_errors: int,
        max_shots: int,
        custom_decoder: Optional[sinter.Decoder] = None,
    ) -> None:
        """One point sinter samples and decodes; its id is its place.

        decoder is the name stats.csv shows. custom_decoder decodes the
        point; None leaves the name to sinter's built-ins. It is kept per
        point because a decoder may be built from its point's circuit,
        which sinter's compile step never sees (sinter.Decoder,
        compile_decoder_for_dem takes the model only). The labels go
        into sinter's json_metadata, so stats.csv carries them beside
        every point's counts.
        """
        _refuse_a_second_kind(self.function_points)
        options = sinter.CollectionOptions(
            max_shots=max_shots, max_errors=max_errors
        )
        metadata = dict(labels)
        task = sinter.Task(
            circuit=circuit,
            decoder=decoder,
            json_metadata=metadata,
            collection_options=options,
        )
        self.tasks.append(task)
        self.decoders.append(custom_decoder)

    def add_point(
        self, function: Callable, labels: Mapping, results_file: str
    ) -> None:
        """One point that function(labels, seed, folder) runs, rows labelled.

        function returns a list of rows, each a dict from column to
        value; the runner writes them after the labels' columns into
        the point's CSV, and combine gathers every point's rows naming
        the same results_file into that file. The seed is a hash of the
        labels, so a point draws the same shots whatever other points
        the grid holds. folder is the results folder, where a point
        reads the saved rows of a point it depends on (point_path). A
        function point runs in one process and leaves --workers to
        sinter's points.
        """
        _refuse_a_second_kind(self.tasks)
        point = _FunctionPoint(function, dict(labels), results_file)
        self.function_points.append(point)

    def main(self, arguments: Optional[list] = None) -> None:
        """Run what the command line asks: list, one point, all, or combine.

        arguments are the command line's, sys.argv's when None.
        """
        parser = _parser()
        parsed = parser.parse_args(arguments)
        if parsed.list:
            self._print_points()
            return
        _refuse_no_workers(parsed.workers)
        folder = _results_folder(self.name, parsed.out)
        _record_the_run(folder)
        if parsed.target == COMBINE:
            self.combine(folder)
            return
        point_ids = self._point_ids(parsed.target)
        for point_id in point_ids:
            self._run_point(point_id, folder, parsed.workers)
        if parsed.target is None:
            self.combine(folder)

    def combine(self, folder: pathlib.Path) -> None:
        """Every saved point's stats into stats.csv, sinter's combine.

        Function points' rows go into the results files they name.
        """
        if self.function_points:
            _combine_rows(folder, self.function_points)
            return
        paths = []
        for point_id in range(len(self.tasks)):
            path = point_path(folder, point_id)
            if path.exists():
                paths.append(path)
        saved_statistics = sinter.read_stats_from_csv_files(*paths)
        lines = [sinter.CSV_HEADER]
        for point_statistics in saved_statistics:
            line = point_statistics.to_csv_line()
            lines.append(line)
        lines.append("")
        text = "\n".join(lines)
        stats_path = folder / STATS_FILE
        stats_path.write_text(text)

    def _print_points(self) -> None:
        for point_id, task in enumerate(self.tasks):
            labels_text = _labels_text(task.json_metadata)
            print(f"{point_id} decoder={task.decoder} {labels_text}")
        for point_id, point in enumerate(self.function_points):
            labels_text = _labels_text(point.labels)
            print(f"{point_id} {point.results_file} {labels_text}")

    def _point_ids(self, target: Optional[str]) -> list:
        """Every point for no target, else the one target names."""
        task_count = len(self.tasks)
        function_point_count = len(self.function_points)
        point_count = task_count + function_point_count
        if target is None:
            return list(range(point_count))
        if not target.isdigit():
            message = (
                f"{target!r} is no point id; give a number --list shows, "
                "or combine"
            )
            raise ValueError(message)
        point_id = int(target)
        if point_id >= point_count:
            last_point_id = point_count - 1
            message = (
                f"point {point_id} is not in this experiment; --list shows "
                f"its {point_count} points, 0 to {last_point_id}"
            )
            raise ValueError(message)
        return [point_id]

    def _run_point(
        self,
        point_id: int,
        folder: pathlib.Path,
        worker_count: int,
    ) -> None:
        """One point collected to its stop rule, resumed from its own CSV."""
        if self.function_points:
            point = self.function_points[point_id]
            _run_function_point(point, point_id, folder)
            return
        task = self.tasks[point_id]
        decoders = {}
        custom_decoder = self.decoders[point_id]
        if custom_decoder is not None:
            decoders[task.decoder] = custom_decoder
        path = point_path(folder, point_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        (point_statistics,) = sinter.collect(
            num_workers=worker_count,
            tasks=[task],
            custom_decoders=decoders,
            save_resume_filepath=path,
        )
        print(
            f"point {point_id}: {point_statistics.shots} shots, "
            f"{point_statistics.errors} errors, "
            f"{point_statistics.seconds:.0f} core seconds"
        )


def point_path(folder: pathlib.Path, point_id: int) -> pathlib.Path:
    """Where a point's rows are saved; the file exists once it is done."""
    return folder / POINTS_FOLDER / f"{point_id}.csv"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an experiment's points (decsim/experiment_runner.py)."
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="a point id from --list, or combine; all points when absent",
    )
    parser.add_argument(
        "--list", action="store_true", help="print every point's id"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="sinter's worker processes; a function point runs in one",
    )
    parser.add_argument(
        "--out", help="the results folder; a dated one when absent"
    )
    return parser


@dataclasses.dataclass(frozen=True)
class _FunctionPoint:
    """A point add_point names: its function, labels and results file."""

    function: Callable
    labels: dict
    results_file: str


def _refuse_a_second_kind(other_kind_points: list) -> None:
    if other_kind_points:
        raise ValueError(ONE_KIND_PER_EXPERIMENT)


def _labels_text(labels: Mapping) -> str:
    label_pairs = []
    for key, value in labels.items():
        label_pairs.append(f"{key}={value}")
    return " ".join(label_pairs)


def _run_function_point(
    point: _FunctionPoint, point_id: int, folder: pathlib.Path
) -> None:
    """The point's rows, labelled, written once its function returns.

    A point whose CSV exists is done: it is written whole or not at
    all, so a task the time limit stopped leaves no CSV to trust.
    """
    path = point_path(folder, point_id)
    if path.exists():
        print(f"point {point_id}: saved already")
        return
    seed = _point_seed(point.labels)
    start = time.perf_counter()
    rows = point.function(point.labels, seed, folder)
    end = time.perf_counter()
    if not rows:
        _refuse_no_rows(point_id)
    elapsed_seconds = end - start
    labelled_rows = []
    for row in rows:
        labelled_row = point.labels | row
        labelled_rows.append(labelled_row)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_rows_once(path, labelled_rows)
    row_count = len(labelled_rows)
    print(f"point {point_id}: {row_count} rows, {elapsed_seconds:.0f} seconds")


def _refuse_no_rows(point_id: int) -> None:
    message = (
        f"point {point_id}'s function returned no rows; a function point "
        "saves at least one, so that its CSV says it ran"
    )
    raise ValueError(message)


def _point_seed(labels: Mapping) -> int:
    """A 64-bit seed read off the labels, the same on every machine."""
    labels_json = json.dumps(labels, sort_keys=True)
    labels_bytes = labels_json.encode()
    hasher = hashlib.sha256(labels_bytes)
    digest = hasher.digest()
    seed_bytes = digest[:8]
    return int.from_bytes(seed_bytes, "big")


def _write_rows_once(path: pathlib.Path, rows: list) -> None:
    """The rows as CSV, staged beside path and renamed over it."""
    random_suffix = secrets.token_hex(8)
    staging = path.with_name(f".{path.name}.{random_suffix}")
    columns = list(rows[0])
    with staging.open("w", newline="") as staging_file:
        writer = csv.DictWriter(
            staging_file, fieldnames=columns, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(staging, path)


def _combine_rows(folder: pathlib.Path, points: list) -> None:
    """Each results file: its saved points' rows, in point order."""
    lines_by_file = {}
    for point_id, point in enumerate(points):
        path = point_path(folder, point_id)
        if not path.exists():
            continue
        point_text = path.read_text()
        point_lines = point_text.splitlines(keepends=True)
        header, *rows = point_lines
        file_lines = lines_by_file.setdefault(point.results_file, [header])
        file_lines.extend(rows)
    for results_file, file_lines in lines_by_file.items():
        results_path = folder / results_file
        text = "".join(file_lines)
        results_path.write_text(text)


def _refuse_no_workers(worker_count: int) -> None:
    """Sinter waits for its workers to answer, so none would hang it."""
    if worker_count < 1:
        message = (
            f"--workers is {worker_count}; sinter needs at least one worker "
            "process"
        )
        raise ValueError(message)


def _results_folder(name: str, requested_folder: Optional[str]) -> pathlib.Path:
    """--out, or <root>/<today>_<name> under DECSIM_RESULTS or ./results."""
    if requested_folder is not None:
        return pathlib.Path(requested_folder)
    root = os.environ.get(RESULTS_ROOT_VARIABLE, DEFAULT_RESULTS_ROOT)
    today = datetime.date.today()
    date_text = today.isoformat()
    return pathlib.Path(root) / f"{date_text}_{name}"


def _record_the_run(folder: pathlib.Path) -> None:
    """A copy of the running script and the commit, or a refusal.

    A folder holds one script's points from one commit, so a different
    script or commit is refused rather than mixed into its results. The
    commit and dirty flag are run_folder's reading of the tree, the one
    every run folder records: the dirty flag is DECSIM_TREE_DIRTY when
    the job exports it, else git's, else None on a node with no git.
    The commit is read from .git itself when there is no git program; a
    tree with no readable commit is refused, since its results could not
    say what ran.
    """
    identity = run_folder.piece_identity()
    _refuse_an_unread_commit(identity["commit"])
    folder.mkdir(parents=True, exist_ok=True)
    script = pathlib.Path(sys.argv[0])
    script_text = script.read_text()
    script_copy = folder / script.name
    _write_once(script_copy, script_text)
    _refuse_another_text(script_copy, script_text)
    commit = str(identity["commit"])
    dirty = str(identity["dirty"])
    commit_path = folder / COMMIT_FILE
    _write_once(commit_path, f"commit {commit}\ndirty {dirty}\n")
    _refuse_another_tree(commit_path, commit, dirty)


def _refuse_an_unread_commit(commit: Optional[str]) -> None:
    if commit is None:
        message = (
            "the commit of the decsim tree running this script cannot be "
            "read, so commit.txt could not say what ran; run from a git "
            "checkout, whose .git folder holds its HEAD"
        )
        raise ValueError(message)


def _write_once(path: pathlib.Path, text: str) -> None:
    """The file written by whichever array task comes first.

    The text is staged in a file created exclusively under a random name
    beside it, since process ids repeat across nodes, and hard-linked
    into place, which fails when the file exists, so a task that starts
    beside another never reads a half-written file. The staging file is
    created with mode 0666 filtered by the umask, as open would create
    the file itself, so a group sharing the folder can read it, and it
    is removed however the write or the link ends.
    """
    if path.exists():
        return
    random_suffix = secrets.token_hex(8)
    staging = path.with_name(f".{path.name}.{random_suffix}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(staging, flags, ORDINARY_FILE_MODE)
    try:
        _publish(descriptor, text, staging, path)
    finally:
        staging.unlink()


def _publish(
    descriptor: int, text: str, staging: pathlib.Path, path: pathlib.Path
) -> None:
    """The text written to the staging file, then linked to path if free."""
    with os.fdopen(descriptor, "w") as staging_file:
        staging_file.write(text)
    try:
        os.link(staging, path)
    except FileExistsError:
        pass


def _refuse_another_text(path: pathlib.Path, text: str) -> None:
    held_text = path.read_text()
    if held_text != text:
        message = (
            f"{path} already holds another {path.name}; {ONE_RUN_PER_FOLDER}"
        )
        raise ValueError(message)


def _refuse_another_tree(path: pathlib.Path, commit: str, dirty: str) -> None:
    """The recorded commit, and the dirty flag where both sides read it.

    A node with no git records dirty None, which says nothing either
    way, so only two known flags that differ are refused.
    """
    held_text = path.read_text()
    commit_line, dirty_line = held_text.splitlines()
    held_commit = commit_line.removeprefix("commit ")
    held_dirty = dirty_line.removeprefix("dirty ")
    is_unread = "None" in (held_dirty, dirty)
    is_same_flag = held_dirty == dirty or is_unread
    if held_commit == commit and is_same_flag:
        return
    message = (
        f"{path} records commit {held_commit} dirty {held_dirty}, and this "
        f"is commit {commit} dirty {dirty}; {ONE_RUN_PER_FOLDER}"
    )
    raise ValueError(message)
