"""Burst detection: how fast the masked regional CUSUM catches a burst.

One trial: a quiet shot of the d, p surface code memory is copy A, and
copy B is copy A XOR one shot of a burst's extra noise alone, which
starts at round 1,000 at a random place on the patch. The detector,
warmed up on one quiet shot, scores both copies at three alarm lines and
the trial saves each line's first alarm in A and in B. Quiet points
count false alarms over 104 s of quiet shots. Each (d, p)'s lines come
from its calibration point, which its other points read, so on Slurm
they run after it (--dependency=afterok).

The command line is gem5 MultiSim's (gem5 v24.0 RELEASE-NOTES.md,
"gem5 MultiSim"): `python run.py --list` prints the points, `python
run.py <id> --out DIR` runs one, `python run.py` runs them all and
combines, and `python run.py combine --out DIR` gathers the saved
points' rows into each results file. A point's seed is a hash of its
labels, so it draws the same shots in any grid, and its rows are saved
whole once its function returns, so a point with a CSV is done and a
resubmitted task runs only the rest.
"""

import argparse
import csv
import dataclasses
import hashlib
import itertools
import json
import pathlib
import sys
import time
from collections.abc import Callable
from typing import Optional

import numpy
import stim

import decsim.detector_error_model.detector_formation as detector_formation
import decsim.experiments.run_folder as run_folder
import decsim.frontends.settings as workload_settings
import decsim.qpu.stim_device as stim_device
from decsim.burst_detectors.masked_regional_cusum import (
    detector as masked_regional_cusum,
)

DISTANCES = [5, 7, 9, 11, 13, 15]
ERROR_RATES = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005]
RATE_POINTS = [
    (distance, rate) for distance in DISTANCES for rate in ERROR_RATES
]
ROUNDS = 2000
ROUND_PERIOD_MICROSECONDS = 1.0
MICROSECONDS_PER_SECOND = 1e6
ONSET_ROUND = 1000
# Willow's large bursts: a 3-round rise, a 400-700 us decay (2408.13687)
RISE_ROUNDS = 3
DECAY_ROUNDS = 600.0
# Disc radii in Stim units, 2 per qubit spacing (2506.18228, 2408.13687)
RADIUS_BY_SIZE = {"small": 3.0, "medium": 4.0, "large": 6.0, "whole": None}
# A hit qubit runs at this multiple of p; saturating adds Stim's cap, 0.75
MULTIPLE_BY_STRENGTH = {"weak": 2.5, "strong": 6.0, "saturating": None}
SATURATING_ADDED_PROBABILITY = 0.75
TRIALS = 200
FALSE_ALARMS_PER_SECOND = (1.0, 0.1, 0.03)
LINE_INDICES = range(len(FALSE_ALARMS_PER_SECOND))
# 13 batches of 1,000 streams of 2,000-round shots: 26 s a part, 104 s a (d, p)
QUIET_PARTS = 4
QUIET_BATCHES = 13
QUIET_STREAMS = 1000
WARM_UP_SHOTS = 1
STIM_SEED_BOUND = 2**63
DETECTOR_SETTINGS = (
    masked_regional_cusum.MaskedRegionalCusumBurstDetector.Settings()
)
NAME = "burst_detection"
_SCRIPT_AS_GIVEN = pathlib.Path(__file__)
SCRIPT = _SCRIPT_AS_GIVEN.resolve()
# `run.py combine` gathers every saved point's rows into its results file
COMBINE = "combine"
POINTS_FOLDER = "points"


@dataclasses.dataclass(frozen=True)
class FunctionPoint:
    """One point: function(labels, seed, folder) returns its rows.

    Each row is a dict from column to value, saved after the labels'
    columns into the point's CSV and gathered with every point naming
    the same results_file into that file.
    """

    function: Callable
    labels: dict
    results_file: str


def circuit(distance: int, error_rate: float) -> stim.Circuit:
    """The Z memory at one rate on all four channels, as the baseline's."""
    return workload_settings.memory_circuit(
        "surface_code:rotated_memory_z", ROUNDS, distance, error_rate
    )


def added_probability(strength: str, error_rate: float) -> float:
    """The burst's probability on top of a hit qubit's own p."""
    multiple = MULTIPLE_BY_STRENGTH[strength]
    if multiple is None:
        return SATURATING_ADDED_PROBABILITY
    added_multiple = multiple - 1
    return added_multiple * error_rate


def calibration_point(labels: dict, _seed: int, _folder: pathlib.Path) -> list:
    """Every line's level per chart group, read off warm quiet streams."""
    quiet_circuit = circuit(labels["d"], labels["p"])
    lines = masked_regional_cusum.AlarmLines.calibrated(
        quiet_circuit,
        ROUNDS,
        DETECTOR_SETTINGS,
        ROUND_PERIOD_MICROSECONDS,
        FALSE_ALARMS_PER_SECOND,
        WARM_UP_SHOTS,
    )
    rows = []
    for rate, levels in zip(FALSE_ALARMS_PER_SECOND, lines.levels, strict=True):
        for group, level in enumerate(levels):
            row = {"alarm_line": rate, "group": group, "level": float(level)}
            rows.append(row)
    return rows


def saved_lines(
    folder: pathlib.Path, distance: int, error_rate: float
) -> masked_regional_cusum.AlarmLines:
    """The lines the (d, p) calibration point saved, or a refusal."""
    calibration_id = RATE_POINTS.index((distance, error_rate))
    path = point_path(folder, calibration_id)
    if not path.exists():
        message = (
            f"no alarm levels at {path}: run calibration point "
            f"{calibration_id} first (on Slurm, --dependency=afterok)"
        )
        raise ValueError(message)
    with path.open(newline="") as levels_file:
        reader = csv.DictReader(levels_file)
        levels = [float(row["level"]) for row in reader]
    level_grid = numpy.reshape(levels, (len(LINE_INDICES), -1))
    quiet_circuit = circuit(distance, error_rate)
    return masked_regional_cusum.AlarmLines.with_levels(
        quiet_circuit, ROUNDS, DETECTOR_SETTINGS, level_grid
    )


def burst_point(labels: dict, seed: int, folder: pathlib.Path) -> list:
    """TRIALS trials: each line's first alarm in copy A and in copy B.

    first_alarm_a is A's first alarm anywhere in the shot, a false alarm;
    first_alarm_b is B's first at or after the onset. Empty is no alarm.
    """
    lines = saved_lines(folder, labels["d"], labels["p"])
    quiet_circuit = circuit(labels["d"], labels["p"])
    generator = numpy.random.default_rng(seed)
    quiet_seed = _stim_seed(generator)
    sampler = quiet_circuit.compile_detector_sampler(seed=quiet_seed)
    stream_count = 2 * TRIALS
    state = lines.new_state(stream_count)
    for _ in range(WARM_UP_SHOTS):
        warm_up = sampler.sample(TRIALS)
        both_warm_ups = numpy.concatenate([warm_up, warm_up])
        lines.score_ratios(state, both_warm_ups)
    copy_a = sampler.sample(TRIALS)
    flips = _burst_flips(quiet_circuit, generator, labels)
    copy_b = copy_a ^ flips
    both_copies = numpy.concatenate([copy_a, copy_b])
    ratios = lines.score_ratios(state, both_copies)
    first_bulk_round = lines.layout.first_bulk_round
    firsts_a = lines.first_alarm_rounds(ratios[:TRIALS], first_bulk_round)
    firsts_b = lines.first_alarm_rounds(ratios[TRIALS:], ONSET_ROUND)
    rows = []
    for trial, line in itertools.product(range(TRIALS), LINE_INDICES):
        first_alarm_a = _alarm_round(firsts_a[trial, line])
        first_alarm_b = _alarm_round(firsts_b[trial, line])
        rate = FALSE_ALARMS_PER_SECOND[line]
        row = {"trial": trial, "alarm_line": rate}
        row |= {"first_alarm_a": first_alarm_a, "first_alarm_b": first_alarm_b}
        rows.append(row)
    return rows


def quiet_point(labels: dict, seed: int, folder: pathlib.Path) -> list:
    """Each line's false alarms over QUIET_BATCHES of quiet shots.

    QUIET_STREAMS streams each score shots one after another on a carried
    state, the warm-up not counted. An alarm is a shot of a stream on
    which a line fires, and the time is whole shots, the calibration's
    units.
    """
    lines = saved_lines(folder, labels["d"], labels["p"])
    quiet_circuit = circuit(labels["d"], labels["p"])
    sampler = quiet_circuit.compile_detector_sampler(seed=seed)
    state = lines.new_state(QUIET_STREAMS)
    for _ in range(WARM_UP_SHOTS):
        warm_up = sampler.sample(QUIET_STREAMS)
        lines.score_ratios(state, warm_up)
    alarm_counts = numpy.zeros(len(LINE_INDICES), dtype=int)
    for _ in range(QUIET_BATCHES):
        events = sampler.sample(QUIET_STREAMS)
        ratios = lines.score_ratios(state, events)
        is_firing = ratios >= 1.0
        is_alarmed = numpy.any(is_firing, axis=1)
        alarm_counts += numpy.sum(is_alarmed, axis=0)
    shot_count = QUIET_BATCHES * QUIET_STREAMS
    round_count = shot_count * ROUNDS
    microseconds = round_count * ROUND_PERIOD_MICROSECONDS
    quiet_seconds = microseconds / MICROSECONDS_PER_SECOND
    rows = []
    for rate, alarms in zip(FALSE_ALARMS_PER_SECOND, alarm_counts, strict=True):
        row = {"alarm_line": rate, "quiet_seconds": quiet_seconds}
        row["alarms"] = int(alarms)
        rows.append(row)
    return rows


def burst_detection() -> list:
    """The calibrations in RATE_POINTS order, then each (d, p)'s points."""
    points = []
    for distance, error_rate in RATE_POINTS:
        rate_labels = {"d": distance, "p": error_rate}
        point = FunctionPoint(
            calibration_point, rate_labels, "alarm_levels.csv"
        )
        points.append(point)
    for distance, error_rate in RATE_POINTS:
        rate_labels = {"d": distance, "p": error_rate}
        classes = itertools.product(RADIUS_BY_SIZE, MULTIPLE_BY_STRENGTH)
        for size, strength in classes:
            labels = rate_labels | {"size": size, "strength": strength}
            point = FunctionPoint(burst_point, labels, "trials.csv")
            points.append(point)
        for part in range(QUIET_PARTS):
            labels = rate_labels | {"part": part}
            point = FunctionPoint(quiet_point, labels, "quiet.csv")
            points.append(point)
    return points


def main(arguments: Optional[list] = None) -> None:
    """Run what the command line asks: list, one point, all, or combine."""
    parser = _parser()
    parsed = parser.parse_args(arguments)
    points = burst_detection()
    if parsed.list:
        _print_points(points)
        return
    point_ids = _point_ids(parser, parsed.target, len(points))
    folder = _results_folder(parser, parsed)
    run_folder.refuse_another_tree(folder)
    every_id = list(range(len(points)))
    started_utc = run_folder.start_run(folder, SCRIPT, every_id)
    for point_id in point_ids:
        _run_point(points[point_id], point_id, folder)
    if parsed.target in (None, COMBINE):
        combine(folder, points)
    run_folder.finish_run(folder, SCRIPT, every_id, started_utc)


def combine(folder: pathlib.Path, points: list) -> None:
    """Each results file: its saved points' rows, in point order.

    The rows are joined as text under the first point's header, so a
    point whose header differs is refused rather than read under the
    wrong columns.
    """
    lines_by_file = {}
    first_path_by_file = {}
    for point_id, point in enumerate(points):
        path = point_path(folder, point_id)
        if not path.exists():
            continue
        point_text = path.read_text()
        header, *rows = point_text.splitlines(keepends=True)
        file_lines = lines_by_file.setdefault(point.results_file, [header])
        first_path = first_path_by_file.setdefault(point.results_file, path)
        if header != file_lines[0]:
            _refuse_a_different_header(first_path, path)
        file_lines.extend(rows)
    for results_file, file_lines in lines_by_file.items():
        results_path = folder / results_file
        text = "".join(file_lines)
        results_path.write_text(text)


def point_path(folder: pathlib.Path, point_id: int) -> pathlib.Path:
    """Where a point's rows are saved; the file exists once it is done."""
    return folder / POINTS_FOLDER / f"{point_id}.csv"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the burst detection experiment's points."
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
        "--out",
        help="the results folder; a new dated one when every point runs",
    )
    return parser


def _print_points(points: list) -> None:
    for point_id, point in enumerate(points):
        label_pairs = []
        for key, value in point.labels.items():
            label_pairs.append(f"{key}={value}")
        labels_text = " ".join(label_pairs)
        print(f"{point_id} {point.results_file} {labels_text}")


def _point_ids(
    parser: argparse.ArgumentParser, target: Optional[str], point_count: int
) -> list:
    """Every point for no target, none for combine, else the one named."""
    if target is None:
        return list(range(point_count))
    if target == COMBINE:
        return []
    if not target.isdigit() or int(target) >= point_count:
        last_point_id = point_count - 1
        parser.error(
            f"{target!r} is no point id; --list shows the {point_count} "
            f"points, 0 to {last_point_id}, or give combine"
        )
    return [int(target)]


def _results_folder(
    parser: argparse.ArgumentParser, parsed: argparse.Namespace
) -> pathlib.Path:
    """--out, or a new dated folder for a run of every point.

    One point and combine name their folder, since every task of an
    array writes the one folder its points share.
    """
    if parsed.out is None and parsed.target is not None:
        parser.error(
            "one point or combine writes the folder its array shares; "
            "name it with --out"
        )
    return run_folder.run_dir_for(NAME, parsed.out)


def _run_point(
    point: FunctionPoint, point_id: int, folder: pathlib.Path
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
        raise ValueError(
            f"point {point_id}'s function returned no rows; a point saves "
            "at least one, so that its CSV says it ran"
        )
    labelled_rows = [point.labels | row for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_rows_once(path, labelled_rows)
    elapsed_seconds = end - start
    row_count = len(labelled_rows)
    print(f"point {point_id}: {row_count} rows, {elapsed_seconds:.0f} seconds")


def _point_seed(labels: dict) -> int:
    """A 64-bit seed read off the labels, the same on every machine."""
    labels_json = json.dumps(labels, sort_keys=True)
    labels_bytes = labels_json.encode()
    hasher = hashlib.sha256(labels_bytes)
    digest = hasher.digest()
    seed_bytes = digest[:8]
    return int.from_bytes(seed_bytes, "big")


def _write_rows_once(path: pathlib.Path, rows: list) -> None:
    """The rows as CSV, staged beside path and renamed over it."""
    columns = list(rows[0])
    with run_folder.staged_replacement(path) as staging:
        with staging.open("w", newline="") as staging_file:
            writer = csv.DictWriter(
                staging_file, fieldnames=columns, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)


def _refuse_a_different_header(
    first_path: pathlib.Path, path: pathlib.Path
) -> None:
    first_header = _header(first_path)
    header = _header(path)
    raise ValueError(
        f"{path} has the columns {header} but {first_path} has "
        f"{first_header}; the rows of one results file are joined under one "
        "header, so its points must save the same columns in the same order"
    )


def _header(path: pathlib.Path) -> str:
    """A saved CSV's first line, without its line end."""
    with path.open() as saved_file:
        first_line = saved_file.readline()
    return first_line.rstrip("\n")


def _burst_flips(
    quiet_circuit: stim.Circuit, generator: numpy.random.Generator, labels
) -> numpy.ndarray:
    """(TRIALS, detectors): one shot of the burst's noise alone per trial.

    Each burst is centred uniformly over the box the qubits span.
    """
    coordinates = quiet_circuit.get_final_qubit_coordinates()
    qubit_positions = coordinates.values()
    position_list = list(qubit_positions)
    positions = numpy.array(position_list)
    low = positions.min(axis=0)
    high = positions.max(axis=0)
    added = added_probability(labels["strength"], labels["p"])
    table = detector_formation.build_formation_table(quiet_circuit, ROUNDS)
    detector_count = quiet_circuit.num_detectors
    flips = numpy.zeros((TRIALS, detector_count), dtype=bool)
    for trial in range(TRIALS):
        centre = generator.uniform(low, high)
        centre_list = centre.tolist()
        settings = stim_device.BurstStimDevice.Settings(
            burst_onset_round=ONSET_ROUND,
            burst_rise_rounds=RISE_ROUNDS,
            burst_decay_rounds=DECAY_ROUNDS,
            burst_radius=RADIUS_BY_SIZE[labels["size"]],
            burst_center=tuple(centre_list),
            burst_error_probability=added,
        )
        noise = stim_device.burst_noise(quiet_circuit, table, settings)
        burst_seed = _stim_seed(generator)
        burst_sampler = noise.compile_detector_sampler(seed=burst_seed)
        (flips[trial],) = burst_sampler.sample(1)
    return flips


def _stim_seed(generator: numpy.random.Generator) -> int:
    # numpy draws below its bound; Stim takes any 64-bit seed
    seed = generator.integers(STIM_SEED_BOUND)
    return int(seed)


def _alarm_round(first_round: int) -> Optional[int]:
    if first_round == masked_regional_cusum.NO_ALARM:
        return None
    return int(first_round)


if __name__ == "__main__":
    main(sys.argv[1:])
