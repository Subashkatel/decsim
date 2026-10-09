"""The switching-per-window run's comparison table, one row per setting.

`python comparison.py <results folder>` reads the tables tables.py
wrote there and writes comparison.csv beside them: for each distance
and physical error rate, every configuration's shots, failed shots,
logical error rate per round and per d rounds with its 95% interval,
windows decoded per shot and windows wrong alone or in a seam pair;
then switching's escalations and decisions and Relay-BP-5's outcomes
on the escalated windows. Every number is a count from the tables or a
rate computed from those counts.
"""

import csv
import pathlib
import sys

import tables

import decsim.experiments.failure_statistics as failure_statistics

ROUNDS_PER_SHOT = 100
CONFIGURATIONS = (
    "union_find_alone",
    "switching",
    "relay_bp5_alone",
    "tesseract_alone",
)
# switching_summary.csv's columns carried into the comparison, by the
# name they take there
SWITCHING_COLUMNS = {
    "escalated_right": "switching_false_positives",
    "kept_wrong_alone": "switching_false_negatives",
    "kept_wrong_paired": "switching_kept_wrong_paired",
    "kept_wrong_partner_unseen": "switching_kept_wrong_partner_unseen",
    "fixed": "relay_fixed",
    "broke": "relay_broke",
    "both_right": "relay_both_right",
    "both_wrong": "relay_both_wrong",
    "not_converged": "relay_not_converged",
}


def main(results_dir: pathlib.Path) -> None:
    """comparison.csv written from the results folder's tables."""
    tables_by_name = {}
    for name in ("shot_summary", "window_summary", "switching_summary"):
        table_path = results_dir / f"{name}.csv"
        tables_by_name[name] = rows_by_setting(table_path)
    shots = tables_by_name["shot_summary"]
    windows = tables_by_name["window_summary"]
    switching = tables_by_name["switching_summary"]
    rows = []
    for setting in sorted(switching):
        row = comparison_row(setting, shots, windows, switching[setting])
        rows.append(row)
    comparison_path = results_dir / "comparison.csv"
    tables.write_rows(comparison_path, rows)


def comparison_row(
    setting: tuple, shots: dict, windows: dict, switching_row: dict
) -> dict:
    """One setting's columns, configuration by configuration."""
    distance, rate = setting
    row = {"distance": distance, "physical_error_rate": rate}
    for configuration in CONFIGURATIONS:
        key = (configuration, distance, rate)
        shot_part = shot_columns(configuration, shots[key], distance)
        window_part = window_columns(configuration, windows[key])
        row.update(shot_part)
        row.update(window_part)
    switching_part = switching_columns(switching_row)
    row.update(switching_part)
    return row


def shot_columns(configuration: str, shot_row: dict, distance: int) -> dict:
    """Shots, failures, and the logical error rate per round and per d.

    The shot rate and its Wilson bounds are each turned into a rate per
    round over the shot's rounds, then over d rounds, the form Toshio
    et al. 2510.25222 report (line 1915).
    """
    shots = int(shot_row["shots"])
    failed = int(shot_row["failed_shots"])
    decoded = int(shot_row["decoded_windows"])
    low, high = tables.wilson_interval(failed, shots)
    rates = {"": failed / shots, "_low": low, "_high": high}
    columns = {
        f"{configuration}_shots": shots,
        f"{configuration}_failed_shots": failed,
        f"{configuration}_windows_per_shot": decoded / shots,
    }
    per_round = {}
    for suffix, shot_rate in rates.items():
        per_round[suffix] = per_round_rate(shot_rate)
        columns[f"{configuration}_per_round{suffix}"] = per_round[suffix]
    for suffix, round_rate in per_round.items():
        per_d_rounds = per_rounds_rate(round_rate, distance)
        columns[f"{configuration}_per_d_rounds{suffix}"] = per_d_rounds
    return columns


def window_columns(configuration: str, window_row: dict) -> dict:
    """Windows by final answer: total, wrong alone, wrong in a seam pair."""
    windows = 0
    for answer_class in tables.ANSWER_CLASSES:
        windows += int(window_row[answer_class])
    return {
        f"{configuration}_windows": windows,
        f"{configuration}_wrong_alone": int(window_row["wrong_alone"]),
        f"{configuration}_wrong_paired": int(window_row["wrong_paired"]),
    }


def switching_columns(switching_row: dict) -> dict:
    """Switching's decisions and Relay-BP-5's outcomes, as counts."""
    escalated = 0
    decided = 0
    for answer_class in tables.ANSWER_CLASSES:
        escalated += int(switching_row[f"escalated_{answer_class}"])
        decided += int(switching_row[f"kept_{answer_class}"])
    decided += escalated
    correct = escalated - int(switching_row["escalated_right"])
    columns = {
        "switching_decided_windows": decided,
        "switching_escalated": escalated,
        "switching_correct_escalations": correct,
    }
    for column, name in SWITCHING_COLUMNS.items():
        columns[name] = int(switching_row[column])
    return columns


def per_round_rate(shot_rate: float) -> float:
    """A shot's failure rate as a rate per round over its rounds."""
    return failure_statistics.per_round_rate(shot_rate, ROUNDS_PER_SHOT)


def per_rounds_rate(per_round: float, rounds: int) -> float:
    """The chance of an odd number of flips in rounds rounds."""
    survival = (1 - 2 * per_round) ** rounds
    return (1 - survival) / 2


def rows_by_setting(path: pathlib.Path) -> dict:
    """A table's rows keyed by (configuration if any, distance, rate)."""
    with path.open(newline="") as table_file:
        reader = csv.DictReader(table_file)
        table_rows = list(reader)
    rows = {}
    for row in table_rows:
        key = (int(row["distance"]), float(row["physical_error_rate"]))
        if "configuration" in row:
            key = (row["configuration"], *key)
        rows[key] = row
    return rows


if __name__ == "__main__":
    results_folder = pathlib.Path(sys.argv[1])
    main(results_folder)
