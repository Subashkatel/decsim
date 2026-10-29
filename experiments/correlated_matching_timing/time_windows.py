r"""Correlated matching's decode time on each kept window, on one pinned core.

Reads capture_windows.py's folder. For each window it builds PyMatching's
correlated matching graph from the window's model file
(Matching.from_detector_error_model_file with enable_correlations, the
graph CorrelatedPyMatchingDecoder compiles from the same model) and
times Matching.decode(syndrome, enable_correlations=True), the call the
row makes, on the one core the process is pinned to: one untimed decode
of the window's syndrome, then REPEATS timed back to back, the window's
time their median. Each graph is warmed before its first window, as the
row warms its own (minimum_weight_perfect_matching _warm_up_correlated).
The windows run in a shuffled order, so slow drift of the core's speed
spreads over every noise level alike, and the garbage collector is off
while they run, since a running decoder has none. It needs numpy and
PyMatching only, so the same file runs on any host.

It writes times.csv, each window's row, less its fired rows, with the
median and the fastest of its repeats in nanoseconds, and host.txt: the
node, lscpu, the core and the versions.

Usage
-----

Through Slurm, one core:

```
python time_windows.py --capture DIR --out DIR
```
"""

import argparse
import csv
import gc
import os
import pathlib
import platform
import random
import subprocess
import sys
import time

import numpy
import pymatching

REPEATS = 7
WARM_UP_DECODES = 10
SHUFFLE_SEED = 20261029


def pinned_core() -> int:
    """Pin this process to the lowest core it may run on; that core."""
    allowed = os.sched_getaffinity(0)
    core = min(allowed)
    os.sched_setaffinity(0, {core})
    return core


def timed_window(matching, syndrome) -> list:
    """REPEATS timings of one window's decode, after one untimed decode."""
    matching.decode(syndrome, enable_correlations=True)
    nanoseconds = []
    for _repeat in range(REPEATS):
        started = time.perf_counter_ns()
        matching.decode(syndrome, enable_correlations=True)
        finished = time.perf_counter_ns()
        elapsed = finished - started
        nanoseconds.append(elapsed)
    return nanoseconds


def time_rows(capture: pathlib.Path, rows: list) -> list:
    """Each row with its median and fastest decode, in a shuffled order."""
    order = list(range(len(rows)))
    shuffler = random.Random(SHUFFLE_SEED)
    shuffler.shuffle(order)
    matching_by_model = {}
    timed = []
    gc.disable()
    for index in order:
        row = rows[index]
        matching = matching_by_model.get(row["model"])
        if matching is None:
            matching = _warm_matching(capture, row)
            matching_by_model[row["model"]] = matching
        syndrome = _syndrome(row)
        nanoseconds = timed_window(matching, syndrome)
        median = numpy.median(nanoseconds)
        timed_row = dict(row)
        del timed_row["fired_rows"]
        timed_row["median_nanoseconds"] = int(median)
        timed_row["fastest_nanoseconds"] = min(nanoseconds)
        timed.append(timed_row)
    gc.enable()
    return timed


def host_text(core: int) -> str:
    """The node, its processor, the pinned core and the versions."""
    lscpu = subprocess.run(
        ["lscpu"], capture_output=True, text=True, check=True
    )
    node = platform.node()
    lines = [
        f"node: {node}",
        f"pinned core: {core}",
        f"python: {sys.version}",
        f"pymatching: {pymatching.__version__}",
        f"numpy: {numpy.__version__}",
        f"repeats: {REPEATS}",
        lscpu.stdout,
    ]
    return "\n".join(lines)


def main() -> None:
    """Time every window of the capture folder; write times.csv."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    core = pinned_core()
    capture = pathlib.Path(arguments.capture)
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = _capture_rows(capture)
    timed = time_rows(capture, rows)
    text = host_text(core)
    host_path = out / "host.txt"
    host_path.write_text(text)
    columns = list(timed[0])
    path = out / "times.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(timed)
    print(f"wrote {path}: {len(timed)} windows on core {core}")


def _capture_rows(capture: pathlib.Path) -> list:
    """Every window of every windows file the capture wrote."""
    rows = []
    found = capture.glob("windows_d*.csv")
    paths = sorted(found)
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(reader)
    return rows


def _warm_matching(capture: pathlib.Path, row: dict):
    """The window's correlated graph, warmed on its first window's syndrome."""
    path = capture / "models" / row["model"]
    matching = pymatching.Matching.from_detector_error_model_file(
        path, enable_correlations=True
    )
    syndrome = _syndrome(row)
    for _decode in range(WARM_UP_DECODES):
        matching.decode(syndrome, enable_correlations=True)
    return matching


def _syndrome(row: dict):
    detector_count = int(row["detectors"])
    syndrome = numpy.zeros(detector_count, dtype=numpy.uint8)
    fired_names = row["fired_rows"].split()
    fired = [int(name) for name in fired_names]
    syndrome[fired] = 1
    return syndrome


if __name__ == "__main__":
    main()
