"""Raw counts per configuration on the same shots, and the two checks.

    python compare.py RESULTS_FOLDER

For each (signal, distance, physical error rate): the shots and failures
of weak alone, belief matching alone and switching, and the shots
switching escalated. Two checks, each a count that must be 0:
  - same shots: a seed whose sample_digest differs between configurations;
  - switching after the fact: a seed whose switching failure differs from
    belief matching's when it escalated and weak alone's when it did not.
It also counts the shots that are not one window.
"""

import collections
import csv
import pathlib
import sys

HEADER = (
    "signal",
    "d",
    "p",
    "shots",
    "weak_fail",
    "strong_fail",
    "switch_fail",
    "escalated",
    "not_one_window",
    "digest_mismatch",
    "posthoc_mismatch",
)
PROBABILITY_COLUMN = "workload.arguments.physical_error_probability"


def main(results: pathlib.Path) -> None:
    """Print one row of counts per point of the folded run."""
    rows_by_point = _rows_by_point(results)
    header_line = ",".join(HEADER)
    print(header_line)
    for point in sorted(rows_by_point):
        by_decoding = rows_by_point[point]
        counts = _point_counts(point, by_decoding)
        values = [str(value) for value in counts]
        line = ",".join(values)
        print(line)


def _rows_by_point(results: pathlib.Path) -> dict:
    """Each point's rows, by decoding and then by seed."""
    shots_path = results / "shots.csv"
    rows_by_point = collections.defaultdict(dict)
    with shots_path.open() as shots_file:
        for row in csv.DictReader(shots_file):
            distance = int(row["qpu.distance"])
            probability = float(row[PROBABILITY_COLUMN])
            point = (row["signal"], distance, probability)
            by_decoding = rows_by_point[point]
            seeds = by_decoding.setdefault(row["decoding"], {})
            seed = int(row["seed"])
            seeds[seed] = row
    return rows_by_point


def _point_counts(point: tuple, by_decoding: dict) -> tuple:
    """The counts of one point, over the seeds all three decodings ran."""
    weak = by_decoding.get("weak_alone", {})
    strong = by_decoding.get("strong_alone", {})
    switching = by_decoding.get("switching", {})
    shared = set(weak) & set(strong) & set(switching)
    counts = collections.Counter()
    for seed in sorted(shared):
        rows = (weak[seed], strong[seed], switching[seed])
        _count_seed(counts, rows)
    signal, distance, probability = point
    counted = [counts[name] for name in HEADER[4:]]
    return (signal, distance, probability, len(shared), *counted)


def _count_seed(counts: collections.Counter, rows: tuple) -> None:
    """Add one seed's failures and checks to the point's counts."""
    weak_row, strong_row, switch_row = rows
    digests = {row["sample_digest"] for row in rows}
    windows = {row["decoded_windows"] for row in rows}
    weak_failed = weak_row["logical_failure"] == "True"
    strong_failed = strong_row["logical_failure"] == "True"
    switch_failed = switch_row["logical_failure"] == "True"
    escalated_windows = int(switch_row["escalated_windows"])
    escalated = escalated_windows > 0
    expected = weak_failed
    if escalated:
        expected = strong_failed
    counts["digest_mismatch"] += len(digests) > 1
    counts["not_one_window"] += windows != {"1"}
    counts["weak_fail"] += weak_failed
    counts["strong_fail"] += strong_failed
    counts["switch_fail"] += switch_failed
    counts["escalated"] += escalated
    counts["posthoc_mismatch"] += expected != switch_failed


if __name__ == "__main__":
    results_folder = pathlib.Path(sys.argv[1])
    main(results_folder)
