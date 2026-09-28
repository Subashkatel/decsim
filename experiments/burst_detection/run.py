"""Burst detection: how fast the masked regional CUSUM catches a burst.

One trial: a quiet shot of the d, p surface code memory is copy A, and
copy B is copy A XOR one shot of a burst's extra noise alone, which
starts at round 1,000 at a random place on the patch. The detector,
warmed up on one quiet shot, scores both copies at three alarm lines and
the trial saves each line's first alarm in A and in B. Quiet points
count false alarms over 104 s of quiet shots. Each (d, p)'s lines come
from its calibration point, which its other points read, so on Slurm
they run after it (--dependency=afterok).
"""

import csv
import itertools
import pathlib
from typing import Optional

import numpy
import stim

import decsim.detector_error_model.detector_formation as detector_formation
import decsim.experiment_runner as experiment_runner
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
    path = experiment_runner.point_path(folder, calibration_id)
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


def burst_detection() -> experiment_runner.Experiment:
    """The calibrations in RATE_POINTS order, then each (d, p)'s points."""
    experiment = experiment_runner.Experiment("burst_detection")
    for distance, error_rate in RATE_POINTS:
        rate_labels = {"d": distance, "p": error_rate}
        experiment.add_point(calibration_point, rate_labels, "alarm_levels.csv")
    for distance, error_rate in RATE_POINTS:
        rate_labels = {"d": distance, "p": error_rate}
        classes = itertools.product(RADIUS_BY_SIZE, MULTIPLE_BY_STRENGTH)
        for size, strength in classes:
            labels = rate_labels | {"size": size, "strength": strength}
            experiment.add_point(burst_point, labels, "trials.csv")
        for part in range(QUIET_PARTS):
            labels = rate_labels | {"part": part}
            experiment.add_point(quiet_point, labels, "quiet.csv")
    return experiment


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
    experiment = burst_detection()
    experiment.main()
