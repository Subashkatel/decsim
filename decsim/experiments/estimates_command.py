"""`decsim estimates`: what --slurm sizes its jobs by, read off a run.

    decsim estimates <run dir>

Prints plan_command.SECONDS_FILE's rows for the run's tasks: each
task's name; its core seconds a shot, the fold's
sim_wall_seconds_per_shot in sweep.csv; and its peak resident memory,
the largest peak_memory_mb its pieces' piece.json files recorded. A
piece's peak is its worker's since the worker started, so it bounds the
task's own from above. Written beside a run file, the file sizes the
next launch's pieces and its jobs' memory from what was measured, as
sinter sizes its batches from measured time (_sampler_ramp_throttled.py).
"""

import argparse
import csv
import pathlib
import sys

import decsim.experiments.fold as fold
import decsim.experiments.pieces as pieces
import decsim.experiments.refusal as refusal
import decsim.experiments.run_folder as run_folder

SWEEP_FILE = "sweep.csv"
COLUMNS = ("task", "core_seconds_per_shot", "peak_memory_mb")


def estimates(argv: list) -> None:
    """Print the seconds file's rows for the run folder's tasks."""
    parser = _parser()
    parsed = parser.parse_args(argv)
    run_dir = pathlib.Path(parsed.run_dir)
    rows = estimate_rows(run_dir)
    writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)


def estimate_rows(run_dir: pathlib.Path) -> list:
    """One row per folded task, in sweep.csv's order."""
    sweep_path = run_dir / SWEEP_FILE
    if not sweep_path.is_file():
        raise refusal.RefusalError(
            f"{run_dir} has no {SWEEP_FILE}; fold it first with "
            f"decsim run --fold --out {run_dir}"
        )
    records = run_folder.task_records(run_dir)
    folders = pieces.every_folder(run_dir)
    rows = []
    for sweep_row in fold.row_stream(sweep_path):
        task_id = sweep_row["task_id"]
        task_folders = pieces.task_folders(folders, task_id)
        row = {
            "task": records[task_id]["name"],
            "core_seconds_per_shot": sweep_row["sim_wall_seconds_per_shot"],
            "peak_memory_mb": _largest_peak(task_folders),
        }
        rows.append(row)
    return rows


def _largest_peak(task_folders: list) -> float:
    peaks = []
    for folder in task_folders:
        piece = pieces.read_piece(folder)
        peaks.append(piece["peak_memory_mb"])
    return max(peaks)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="decsim estimates")
    parser.add_argument("run_dir", help="a folded run's results folder")
    return parser
