"""`decsim status`: an experiment's pieces folded, and where each point stands.

Every configuration configurations.csv names is folded from its saved
pieces into its run folder, combined/<name>-<id8>/, as a collect of it
folds them, and status.csv gets one row per point: its configuration and
point ids, its swept values, the state of its contiguous prefix, its
counts, the estimate and exact limits the fold computed for that prefix
(failure_statistics), and the core seconds of all its pieces. A point
with no piece yet is a row in state "no data". Status reads and writes
nothing a running task writes, so it can run while a round runs; its
files are replaced whole.
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
# The fold's sweep columns a status row carries, in order.
SWEEP_COLUMNS = (
    "state",
    "shots",
    "scored_shots",
    "logical_failures",
    "unscored_shots",
    "prefix_shots",
    "prefix_scored_shots",
    "prefix_failures",
    "logical_error_rate_estimate",
    "logical_error_rate_low",
    "logical_error_rate_high",
)
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
    configurations = run_folder.recorded_configurations(experiment_dir)
    rows = []
    swept = {}
    for configuration_id, configs in configurations.items():
        config = configs[0]
        points = collect_command.saved_points(configs, experiment_dir)
        point_ids = [point.task.strong_id() for point in points]
        report_dir = run_folder.combined_folder(experiment_dir, config)
        started_utc = run_folder.start_run(config, report_dir, point_ids)
        sweep_rows = collect_command.write_the_run_folder(
            experiment_dir, points, point_ids, report_dir
        )
        run_folder.finish_run(config, report_dir, point_ids, started_utc)
        sweep_by_point = {row["point_id"]: row for row in sweep_rows}
        for point_id in point_ids:
            sweep_row = sweep_by_point.get(point_id)
            row = _status_row(
                experiment_dir, configuration_id, point_id, sweep_row
            )
            rows.append(row)
        point_swept = run_folder.swept_values(experiment_dir, point_ids)
        swept.update(point_swept)
    status_path = experiment_dir / STATUS_FILE
    partial = status_path.with_name(f".{STATUS_FILE}.partial")
    report.write_csv(rows, partial, swept)
    partial.replace(status_path)
    return rows


def _status_row(
    experiment_dir: pathlib.Path,
    configuration_id: str,
    point_id: str,
    sweep_row,
) -> dict:
    """One point's status row: its ids, its fold's columns, its seconds."""
    row = {"point_id": point_id, "configuration_id": configuration_id}
    if sweep_row is None:
        row["state"] = NO_DATA
    else:
        for column in SWEEP_COLUMNS:
            row[column] = sweep_row[column]
    row["core_seconds"] = _core_seconds(experiment_dir, point_id)
    return row


def _core_seconds(experiment_dir: pathlib.Path, point_id: str) -> float:
    """The core seconds of every saved piece of the point."""
    core_seconds = 0.0
    for folder in pieces.folders_of(experiment_dir, [point_id]):
        piece = pieces.read_piece(folder)
        core_seconds += piece["core_seconds"]
    return core_seconds
