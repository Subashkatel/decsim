"""`decsim status`: an experiment's pieces folded, and where each point stands.

Every point the experiment recorded is folded from its saved pieces
into its configuration's run folder, combined/<name>-<id8>/, as a
collect of it folds them. Every configuration's folder is rebuilt, one
left with no point too, so a point that moved to another configuration
is in one run folder. What a fold reads is the points' resolved/
records and the pieces, never the yamls, so a point a yaml no longer
sweeps is still counted, and a point two configurations recorded is
counted once, under the one that recorded it last. status.csv gets one
row per point: its configuration id, then its sweep.csv row whole (its
id and swept values, the state of its contiguous prefix, its counts,
and every estimate and exact limit the fold computed for that prefix,
per shot, per round and plan-unbiased, failure_statistics), then the
rounds and core seconds of all its pieces. A point with no piece yet is
a row in state "no data". Status reads and writes nothing a running task writes,
so it can run while a round runs; its files are replaced whole.
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
    parser.add_argument("experiment", help="the experiment folder to fold")
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
    """Every configuration folded into its run folder; status.csv's rows."""
    records = run_folder.resolved_by_point(experiment_dir)
    point_ids_by_configuration = _point_ids_by_configuration(records)
    combined_folders = run_folder.recorded_combined_folders(experiment_dir)
    rows = []
    swept = {}
    for configuration_id, report_dir in combined_folders.items():
        point_ids = point_ids_by_configuration.get(configuration_id, [])
        started_utc = run_folder.utc_now()
        folders = pieces.folders_of(experiment_dir, point_ids)
        sweep_rows = collect_command.write_the_run_folder(
            experiment_dir, folders, point_ids, report_dir
        )
        run_folder.snapshot_code_state(None, report_dir)
        run_folder.finish_run(None, report_dir, point_ids, started_utc)
        sweep_by_point = {row["point_id"]: row for row in sweep_rows}
        for point_id in point_ids:
            sweep_row = sweep_by_point.get(point_id)
            row = _status_row(folders, configuration_id, point_id, sweep_row)
            rows.append(row)
        point_swept = run_folder.swept_values(experiment_dir, point_ids)
        swept.update(point_swept)
    status_path = experiment_dir / STATUS_FILE
    with run_folder.staged_replacement(status_path) as staging:
        report.write_csv(rows, staging, swept)
    return rows


def _point_ids_by_configuration(records: dict) -> dict:
    """The recorded points of each configuration, as their records say."""
    point_ids_by_configuration = {}
    for point_id, record in records.items():
        configuration_id = record["experiment"]["configuration_id"]
        point_ids = point_ids_by_configuration.setdefault(configuration_id, [])
        point_ids.append(point_id)
    return point_ids_by_configuration


def _status_row(
    folders: list, configuration_id: str, point_id: str, sweep_row
) -> dict:
    """One point's status row: its ids, its sweep row, its rounds and seconds.

    The sweep row is whole, every column the fold gives the point. The
    rounds and seconds are summed over the pieces the fold read, folders,
    so they count the shots the row counts.
    """
    row = {"point_id": point_id, "configuration_id": configuration_id}
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
