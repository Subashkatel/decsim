"""One experiment script's points, its command line and its results folder.

An experiment is a Python script that adds its points to an Experiment
and calls main. The command line is gem5 MultiSim's (gem5 v24.0
RELEASE-NOTES.md, "gem5 MultiSim"): `run.py --list` names every point by
id, `run.py <id>` runs one, and `run.py` runs them all, so one Slurm
array task runs one point (slurm/run.sbatch). An offline point is one
sinter task that sinter.collect runs to its stop rule, each point into
its own resume CSV, so array tasks running at once never write one
file and a resubmitted task continues where it stopped. The folder
keeps the script and the commit that ran, as gem5 keeps config.ini and
ns-3 SEM the commit with every run.
"""

import argparse
import datetime
import os
import pathlib
import sys
import tempfile
from collections.abc import Mapping
from typing import Optional

import sinter
import stim

import decsim.experiments.run_folder as run_folder

# `run.py combine` folds every point's CSV into stats.csv.
COMBINE = "combine"
POINTS_FOLDER = "points"
STATS_FILE = "stats.csv"
COMMIT_FILE = "commit.txt"
ONE_RUN_PER_FOLDER = (
    "a results folder belongs to one script and one commit, so give --out "
    "a new folder"
)
# where a folder goes when no --out names one
RESULTS_ROOT_VARIABLE = "DECSIM_RESULTS"
DEFAULT_RESULTS_ROOT = "results"


class Experiment:
    """The points one experiment script adds, run from its command line."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.tasks = []

    def add_offline(
        self,
        circuit: stim.Circuit,
        decoder: str,
        labels: Mapping,
        max_errors: int,
        max_shots: int,
    ) -> None:
        """One point sinter samples and decodes; its id is its place.

        The labels go into sinter's json_metadata, so stats.csv carries
        them beside every point's counts.
        """
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

    def main(
        self,
        custom_decoders: Optional[Mapping] = None,
        arguments: Optional[list] = None,
    ) -> None:
        """Run what the command line asks: list, one point, all, or combine.

        custom_decoders maps a decoder name to its sinter.Decoder, None
        for a decoder built into sinter. arguments are the command
        line's, sys.argv's when None.
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
        decoders = _sinter_decoders(custom_decoders)
        for point_id in point_ids:
            self._run_point(point_id, folder, parsed.workers, decoders)
        if parsed.target is None:
            self.combine(folder)

    def combine(self, folder: pathlib.Path) -> None:
        """Every saved point's stats into stats.csv, sinter's combine."""
        paths = []
        for point_id in range(len(self.tasks)):
            path = _point_path(folder, point_id)
            if path.exists():
                paths.append(path)
        saved_stats = sinter.read_stats_from_csv_files(*paths)
        lines = [sinter.CSV_HEADER]
        for point_stats in saved_stats:
            line = point_stats.to_csv_line()
            lines.append(line)
        lines.append("")
        text = "\n".join(lines)
        stats_path = folder / STATS_FILE
        stats_path.write_text(text)

    def _print_points(self) -> None:
        for point_id, task in enumerate(self.tasks):
            label_pairs = []
            for key, value in task.json_metadata.items():
                label_pairs.append(f"{key}={value}")
            labels_text = " ".join(label_pairs)
            print(f"{point_id} decoder={task.decoder} {labels_text}")

    def _point_ids(self, target: Optional[str]) -> list:
        """Every point for no target, else the one target names."""
        point_count = len(self.tasks)
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
        decoders: dict,
    ) -> None:
        """One point collected to its stop rule, resumed from its own CSV."""
        task = self.tasks[point_id]
        path = _point_path(folder, point_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        (point_stats,) = sinter.collect(
            num_workers=worker_count,
            tasks=[task],
            custom_decoders=decoders,
            save_resume_filepath=path,
        )
        print(
            f"point {point_id}: {point_stats.shots} shots, "
            f"{point_stats.errors} errors, "
            f"{point_stats.seconds:.0f} core seconds"
        )


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
        "--workers", type=int, default=1, help="sinter's worker processes"
    )
    parser.add_argument(
        "--out", help="the results folder; a dated one when absent"
    )
    return parser


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
    """
    folder.mkdir(parents=True, exist_ok=True)
    script = pathlib.Path(sys.argv[0])
    script_text = script.read_text()
    script_copy = folder / script.name
    _write_once(script_copy, script_text)
    _refuse_another_text(script_copy, script_text)
    identity = run_folder.piece_identity()
    commit = str(identity["commit"])
    dirty = str(identity["dirty"])
    commit_path = folder / COMMIT_FILE
    _write_once(commit_path, f"commit {commit}\ndirty {dirty}\n")
    _refuse_another_tree(commit_path, commit, dirty)


def _write_once(path: pathlib.Path, text: str) -> None:
    """The file written by whichever array task comes first.

    The text is staged in a file created exclusively under a random name
    beside it, since process ids repeat across nodes, and hard-linked
    into place, which fails when the file exists, so a task that starts
    beside another never reads a half-written file.
    """
    if not path.exists():
        prefix = f".{path.name}."
        descriptor, staging_name = tempfile.mkstemp(
            prefix=prefix, dir=path.parent
        )
        with os.fdopen(descriptor, "w") as staging_file:
            staging_file.write(text)
        staging = pathlib.Path(staging_name)
        try:
            os.link(staging, path)
        except FileExistsError:
            pass
        staging.unlink()


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


def _sinter_decoders(custom_decoders: Optional[Mapping]) -> dict:
    """The decoders sinter is handed: every named one but its built-ins."""
    decoders = {}
    if custom_decoders is None:
        return decoders
    for name, decoder in custom_decoders.items():
        if decoder is not None:
            decoders[name] = decoder
    return decoders


def _point_path(folder: pathlib.Path, point_id: int) -> pathlib.Path:
    return folder / POINTS_FOLDER / f"{point_id}.csv"
