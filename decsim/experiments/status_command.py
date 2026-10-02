"""`decsim status`: a results folder's pieces folded, and where each point is.

Every point the folder recorded is folded from its saved pieces, as a
collect of it folds them. What a fold reads is the points' records and
the pieces, never the run file, so a point the run file no longer makes
is still counted. status.csv gets one row per point: its sweep.csv row
whole (its id and swept values, the state of its contiguous prefix, its
counts, and every estimate and exact limit the fold computed for that
prefix, per shot, per round and plan-unbiased, failure_statistics), then
the rounds and core seconds of all its pieces. A point with no piece yet
is a row in state "no data". Status reads and writes nothing a running
task writes, so it can run while a round runs; its files are replaced
whole.
"""

import argparse
import collections
import pathlib
import sys

import decsim.experiments.collect_command as collect_command
import decsim.experiments.pieces as pieces
import decsim.experiments.report as report
import decsim.experiments.run_folder as run_folder

STATUS_FILE = "status.csv"
NO_DATA = "no data"


def main(argv: list) -> None:
    """Fold the experiment and print where status.csv went, and its states."""
    parser = argparse.ArgumentParser(prog="decsim status")
    parser.add_argument("experiment", help="the results folder to fold")
    parsed = parser.parse_args(argv)
    experiment_dir = pathlib.Path(parsed.experiment)
    rows = fold_the_experiment(experiment_dir)
    states = collections.Counter(row["state"] for row in rows)
    state_text = ", ".join(
        f"{count} {state}" for state, count in states.items()
    )
    print(f"{len(rows)} points: {state_text}", file=sys.stderr)
    status_path = experiment_dir / STATUS_FILE
    print(status_path)


def fold_the_experiment(experiment_dir: pathlib.Path) -> list:
    """The folder folded, its points in run.json's order; status.csv's rows."""
    run_path = experiment_dir / run_folder.RUN_FILE
    run_record = run_folder.read_json(run_path)
    point_ids = run_folder.recorded_point_ids(
        experiment_dir, run_record["points"]
    )
    folders = pieces.folders_of(experiment_dir, point_ids)
    sweep_rows = collect_command.fold_the_folder(
        experiment_dir, point_ids, folders
    )
    sweep_by_point = {row["point_id"]: row for row in sweep_rows}
    rows = []
    for point_id in point_ids:
        sweep_row = sweep_by_point.get(point_id)
        row = _status_row(folders, point_id, sweep_row)
        rows.append(row)
    swept = run_folder.swept_values(experiment_dir, point_ids)
    status_path = experiment_dir / STATUS_FILE
    with run_folder.staged_replacement(status_path) as staging:
        report.write_csv(rows, staging, swept)
    return rows


def _status_row(folders: list, point_id: str, sweep_row) -> dict:
    """One point's status row: its id, its sweep row, its rounds and seconds.

    The sweep row is whole, every column the fold gives the point. The
    rounds and seconds are summed over the pieces the fold read, folders,
    so they count the shots the row counts.
    """
    row = {"point_id": point_id}
    if sweep_row is None:
        row["state"] = NO_DATA
    else:
        row.update(sweep_row)
    point_folders = pieces.point_folders(folders, point_id)
    row["rounds"] = _summed(point_folders, "rounds")
    row["core_seconds"] = _summed(point_folders, "core_seconds")
    return row


def _summed(folders: list, name: str):
    """One piece.json line summed over the pieces."""
    total = 0
    for folder in folders:
        piece = pieces.read_piece(folder)
        total += piece[name]
    return total
