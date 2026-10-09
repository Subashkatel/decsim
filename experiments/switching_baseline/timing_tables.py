"""decode_time.csv and stage_means.csv from the folded window_samples.csv.

`python timing_tables.py <run>/combined <results folder>` reads each
configuration's window_samples.csv, as `decsim status` folds it, and
writes the two tables plot.py draws the decode time and the latency
breakdown from: decode_time.csv, the "algorithm" value counts per tier,
and stage_means.csv, each stage's mean time per window per tier.
"""

import csv
import pathlib
import sys

DECODE_TIME_COLUMNS = [
    "configuration",
    "physical_error_rate",
    "distance",
    "round_period_us",
    "tier",
    "value_us",
    "count",
]
STAGE_MEANS_COLUMNS = [
    "configuration",
    "physical_error_rate",
    "distance",
    "tier",
    "stage",
    "windows",
    "mean_us",
]
# the stages plot.py has no use for
LEFT_OUT_STAGES = {
    "backend_queue_wait",
    "dd_per_window",
    "qpu_first_round_to_frame",
    "qpu_last_round_to_frame",
    "service",
}


def main(combined: pathlib.Path, out: pathlib.Path) -> None:
    decode_time_rows = []
    stage_totals = {}
    # a folder starting with a dot is decsim status's staging folder
    for folder in sorted(combined.glob("[!.]*")):
        configuration = folder.name.rsplit("-", 1)[0]
        with open(folder / "window_samples.csv") as handle:
            for row in csv.DictReader(handle):
                add_row(configuration, row, decode_time_rows, stage_totals)
    write(out / "decode_time.csv", DECODE_TIME_COLUMNS, decode_time_rows)
    stage_rows = []
    for key, (windows, total_us) in stage_totals.items():
        mean_us = f"{total_us / windows:.6f}"
        stage_rows.append([*key, windows, mean_us])
    write(out / "stage_means.csv", STAGE_MEANS_COLUMNS, stage_rows)


def add_row(configuration: str, row: dict, decode_time_rows: list, stage_totals: dict) -> None:
    """One value count: a decode time row, and its share of a stage mean."""
    tier = row["tier"]
    if not tier:
        return
    error_rate = row["workload.arguments.physical_error_probability"]
    distance = row["qpu.distance"]
    stage = row["name"]
    count = int(row["count"])
    value_us = float(row["value_us"])
    if stage == "algorithm":
        round_period = row["qpu.round_period_microseconds"]
        decode_time_rows.append(
            [configuration, error_rate, distance, round_period, tier, row["value_us"], count]
        )
    if stage in LEFT_OUT_STAGES:
        return
    key = (configuration, error_rate, distance, tier, stage)
    totals = stage_totals.setdefault(key, [0, 0.0])
    totals[0] += count
    totals[1] += value_us * count


def write(path: pathlib.Path, columns: list, rows: list) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
