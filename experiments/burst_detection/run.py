"""Burst detection, step one: one quiet stream twice, a burst added to one.

Stim's rotated surface code memory in the Z basis, 2,000 rounds of 1 us,
one error rate on all four noise channels (decsim.frontends.settings
memory_circuit, the circuit the decoder baseline samples). A trial's
copy A is one quiet shot's detection events; copy B is copy A XOR a
shot of the burst's extra noise alone (decsim.qpu.stim_device
burst_noise), so the two share every quiet error and differ only by
the burst's flips. The burst starts at round 1,000, climbs over 3
rounds and decays over 600, the shape of Willow's large bursts,
centred anywhere on the patch. decsim's
masked regional CUSUM at its defaults scores both copies at three
false alarm rates, its lines read off its own quiet calibration shots,
which take seed 0; every point's seed is a hash of its labels.

Each (d, p) has 12 burst points, a size by a strength, a quiet point
that streams 100 s of chip time through the lines, a levels point and
an example traces point. `python run.py --list` prints them and
`python run.py <id>` runs one (docs/how-to/run_an_experiment.md).
Stage 1a is d = 5 at p = 0.003; stage 1b sets DISTANCES to 5, 7, 9, 11,
13 and 15 and ERROR_RATES to 0.0005, 0.001, 0.002, 0.003, 0.004 and
0.005.
"""

import itertools
import pathlib
from typing import Optional

import numpy
import stim

import decsim.burst_detectors.masked_regional_cusum.detector as cusum
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.experiment_runner as experiment_runner
import decsim.frontends.settings as workload_settings
import decsim.qpu.stim_device as stim_device

DISTANCES = [5]
ERROR_RATES = [0.003]
CODE_TASK = "surface_code:rotated_memory_z"
ROUNDS = 2000
ROUND_PERIOD_MICROSECONDS = 1.0
ONSET_ROUND = 1000
# Willow's large bursts decay over 400 to 700 us (2408.13687 lines
# 2117-2119), and the six largest in the repetition-code data released
# with it peak about 3 rounds after their onset.
RISE_ROUNDS = 3
DECAY_ROUNDS = 600.0
TRIALS = 200
FALSE_ALARMS_PER_SECOND = (1.0, 0.1, 0.03)
# Burst radii in data-qubit spacings, two Stim coordinate units each;
# None is the whole patch. Centred inside the patch the discs hold 13,
# 25 and about 60 qubits, data and measure: Kurilovich's bursts cover a
# median of 9 qubits in their T1 part and 15 in their phase part
# (2506.18228 lines 1114-1116 and 1210-1223), Willow's roughly 30
# (2408.13687 line 391).
RADIUS_BY_SIZE = {"small": 1.5, "medium": 2.0, "large": 3.0, "whole": None}
STIM_UNITS_PER_SPACING = 2.0
# Each strength is the firing of a check whose qubits are all hit, as
# a multiple of its quiet firing; saturating is Stim's largest
# DEPOLARIZE1, 3/4, the noise at its limit.
FIRING_MULTIPLE_BY_STRENGTH = {"weak": 2.5, "strong": 6.0, "saturating": None}
STRONG_FIRING_CAP = 0.45
SATURATING_LEVEL = 0.75
# The level search's circuit: long enough for the rise to peak.
SEARCH_ROUNDS = 12
SEARCH_ONSET_ROUND = 4
SEARCH_STEPS = 30
QUIET_SECONDS = 100.0
QUIET_STREAMS = 1000
MICROSECONDS_PER_SECOND = 1e6
STIM_SEED_BOUND = 2**63
# The example traces' rounds: the onset and the 300 rounds a catch is
# counted in, with a margin on each side.
TRACE_FIRST_ROUND = 900
TRACE_LAST_ROUND = 1400
DETECTOR_SETTINGS = cusum.MaskedRegionalCusumBurstDetector.Settings()


def circuit(distance: int, error_rate: float, rounds: int) -> stim.Circuit:
    """The memory at one rate on all four channels, as the baseline's."""
    return workload_settings.memory_circuit(
        CODE_TASK, rounds, distance, error_rate
    )


def alarm_lines(quiet_circuit: stim.Circuit) -> cusum.AlarmLines:
    """The detector's lines at every rate, from its own calibration."""
    return cusum.AlarmLines.calibrated(
        quiet_circuit,
        ROUNDS,
        DETECTOR_SETTINGS,
        ROUND_PERIOD_MICROSECONDS,
        FALSE_ALARMS_PER_SECOND,
    )


def burst_settings(
    level: float, radius: Optional[float], centre: tuple
) -> stim_device.BurstStimDevice.Settings:
    """One trial's burst; a radius of None is the whole patch."""
    return stim_device.BurstStimDevice.Settings(
        burst_onset_round=ONSET_ROUND,
        burst_rise_rounds=RISE_ROUNDS,
        burst_decay_rounds=DECAY_ROUNDS,
        burst_radius=radius,
        burst_center=centre,
        burst_error_probability=level,
    )


def burst_level(distance: int, error_rate: float, strength: str) -> dict:
    """The burst level whose peak firing meets the strength's target.

    Firing is read exactly off the detector error model of a short
    whole-patch burst, the mean over the checks of their peak round, so
    it is the firing of a check with every qubit in the disc. The level
    is bisected, firing growing with it; a target past the firing at
    3/4 takes 3/4.
    """
    quiet_circuit = circuit(distance, error_rate, SEARCH_ROUNDS)
    quiet_firing = _round_firing(quiet_circuit)
    mean_firing = numpy.mean(quiet_firing)
    usual_firing = float(mean_firing)
    multiple = FIRING_MULTIPLE_BY_STRENGTH[strength]
    level = SATURATING_LEVEL
    if multiple is not None:
        target = _target_firing(usual_firing, multiple, strength)
        level = _bisected_level(quiet_circuit, target)
    firing = _peak_firing(quiet_circuit, level)
    return {"level": level, "firing": firing, "quiet_firing": usual_firing}


def levels_point(labels: dict, _seed: int, _folder: pathlib.Path) -> list:
    """Each strength's level at the point's d and p."""
    rows = []
    for strength in FIRING_MULTIPLE_BY_STRENGTH:
        found = burst_level(labels["d"], labels["p"], strength)
        row = {
            "strength": strength,
            "level": found["level"],
            "firing_reached": found["firing"],
            "quiet_firing": found["quiet_firing"],
        }
        rows.append(row)
    return rows


def burst_point(labels: dict, seed: int, _folder: pathlib.Path) -> list:
    """TRIALS paired trials: each line's first alarm in copy A and in B.

    first_alarm_a is copy A's first alarm anywhere in the shot, a false
    alarm; first_alarm_b is copy B's first at or after the onset, since
    before it B is A. None is no alarm.
    """
    distance = labels["d"]
    quiet_circuit = circuit(distance, labels["p"], ROUNDS)
    lines = alarm_lines(quiet_circuit)
    found = burst_level(distance, labels["p"], labels["strength"])
    generator = numpy.random.default_rng(seed)
    centres = _centres(generator, quiet_circuit, TRIALS)
    radius = _stim_radius(labels["size"])
    events_a, events_b = _paired_events(
        quiet_circuit, generator, found["level"], radius, centres
    )
    ratios = _paired_ratios(lines, events_a, events_b)
    first_bulk_round = lines.layout.first_bulk_round
    firsts_a = lines.first_alarm_rounds(ratios["a"], first_bulk_round)
    firsts_b = lines.first_alarm_rounds(ratios["b"], ONSET_ROUND)
    return _trial_rows(centres, radius, firsts_a, firsts_b)


def quiet_point(labels: dict, seed: int, _folder: pathlib.Path) -> list:
    """QUIET_SECONDS of quiet chip time through the lines, alarms per line.

    QUIET_STREAMS streams run side by side, each a run of quiet shots
    scored one after another on one state, so a chart keeps its score
    and its usual rate across the join; the join drops each shot's first
    and last rounds, which hold no bulk detector, so it is a missing
    round, not a restart. An alarm is a shot of one stream on which a
    line fires, the unit the calibration counts in.
    """
    quiet_circuit = circuit(labels["d"], labels["p"], ROUNDS)
    lines = alarm_lines(quiet_circuit)
    sampler = quiet_circuit.compile_detector_sampler(seed=seed)
    state = lines.new_state(QUIET_STREAMS)
    alarm_counts = numpy.zeros(len(FALSE_ALARMS_PER_SECOND), dtype=int)
    scored_rounds = 0
    while _seconds(scored_rounds) < QUIET_SECONDS:
        events = sampler.sample(QUIET_STREAMS)
        ratios = lines.score_ratios(state, events)
        is_firing = ratios >= 1.0
        is_alarmed = numpy.any(is_firing, axis=1)
        alarm_counts += numpy.sum(is_alarmed, axis=0)
        scored_rounds += ratios.shape[0] * ratios.shape[1]
    quiet_seconds = _seconds(scored_rounds)
    rows = []
    for rate, alarms in zip(FALSE_ALARMS_PER_SECOND, alarm_counts, strict=True):
        row = {
            "alarm_line": rate,
            "quiet_seconds": quiet_seconds,
            "alarms": int(alarms),
        }
        rows.append(row)
    return rows


def traces_point(labels: dict, seed: int, _folder: pathlib.Path) -> list:
    """One example trial per class: each line's score over level by round.

    The score is the largest group score over its level, so a line
    fires where it reaches one; the rounds kept run from
    TRACE_FIRST_ROUND to TRACE_LAST_ROUND.
    """
    distance = labels["d"]
    quiet_circuit = circuit(distance, labels["p"], ROUNDS)
    lines = alarm_lines(quiet_circuit)
    generator = numpy.random.default_rng(seed)
    rows = []
    for size, strength in itertools.product(
        RADIUS_BY_SIZE, FIRING_MULTIPLE_BY_STRENGTH
    ):
        found = burst_level(distance, labels["p"], strength)
        centres = _centres(generator, quiet_circuit, 1)
        radius = _stim_radius(size)
        events_a, events_b = _paired_events(
            quiet_circuit, generator, found["level"], radius, centres
        )
        ratios = _paired_ratios(lines, events_a, events_b)
        class_labels = {"size": size, "strength": strength}
        class_rows = _trace_rows(lines, ratios, class_labels)
        rows.extend(class_rows)
    return rows


def burst_detection() -> experiment_runner.Experiment:
    """Every point, distance by rate: the bursts, then quiet, levels, traces."""
    experiment = experiment_runner.Experiment("burst_detection")
    for distance, error_rate in itertools.product(DISTANCES, ERROR_RATES):
        rate_labels = {"d": distance, "p": error_rate}
        for size, strength in itertools.product(
            RADIUS_BY_SIZE, FIRING_MULTIPLE_BY_STRENGTH
        ):
            class_labels = {"size": size, "strength": strength}
            labels = rate_labels | class_labels
            experiment.add_point(burst_point, labels, "trials.csv")
        experiment.add_point(quiet_point, rate_labels, "quiet.csv")
        experiment.add_point(levels_point, rate_labels, "levels.csv")
        experiment.add_point(traces_point, rate_labels, "traces.csv")
    return experiment


def _round_firing(noisy_circuit: stim.Circuit) -> numpy.ndarray:
    """Each bulk round's mean detection probability over its checks.

    A detector fires on an odd number of its independent mechanisms,
    (1 - prod(1 - 2 p)) / 2 (Tan et al. 2406.18897 lines 956-960). The
    bulk rounds are the Stim times between the first round's, 0, and
    the data readout's.
    """
    model = noisy_circuit.detector_error_model(decompose_errors=False)
    survivals = numpy.ones(model.num_detectors)
    for instruction in model.flattened():
        _multiply_survivals(survivals, instruction)
    flipped = 1.0 - survivals
    detection = flipped / 2.0
    coordinates = noisy_circuit.get_detector_coordinates()
    time_list = [coordinates[index][2] for index in coordinates]
    times = numpy.array(time_list)
    readout_time = times.max()
    bulk_times = numpy.arange(1, readout_time)
    firing = []
    for time in bulk_times:
        is_at_time = times == time
        round_detection = detection[is_at_time]
        round_firing = round_detection.mean()
        firing.append(round_firing)
    return numpy.array(firing)


def _multiply_survivals(
    survivals: numpy.ndarray, instruction: stim.DemInstruction
) -> None:
    """One mechanism's 1 - 2 p on every detector it flips."""
    if instruction.type != "error":
        return
    arguments = instruction.args_copy()
    doubled = 2.0 * arguments[0]
    survival = 1.0 - doubled
    for target in instruction.targets_copy():
        if target.is_relative_detector_id():
            survivals[target.val] *= survival


def _target_firing(
    usual_firing: float, multiple: float, strength: str
) -> float:
    target = usual_firing * multiple
    if strength == "strong":
        return min(target, STRONG_FIRING_CAP)
    return target


def _peak_firing(quiet_circuit: stim.Circuit, level: float) -> float:
    """The mean firing of the whole-patch burst's peak round."""
    table = detector_formation.build_formation_table(
        quiet_circuit, SEARCH_ROUNDS
    )
    settings = stim_device.BurstStimDevice.Settings(
        burst_onset_round=SEARCH_ONSET_ROUND,
        burst_rise_rounds=RISE_ROUNDS,
        burst_decay_rounds=DECAY_ROUNDS,
        burst_error_probability=level,
    )
    burst = stim_device.burst_circuit(quiet_circuit, table, settings)
    firing = _round_firing(burst)
    peak = firing.max()
    return float(peak)


def _bisected_level(quiet_circuit: stim.Circuit, target: float) -> float:
    low = 0.0
    high = SATURATING_LEVEL
    if _peak_firing(quiet_circuit, high) <= target:
        return high
    for _ in range(SEARCH_STEPS):
        bracket_total = low + high
        middle = bracket_total / 2
        if _peak_firing(quiet_circuit, middle) < target:
            low = middle
        else:
            high = middle
    bracket_total = low + high
    return bracket_total / 2


def _stim_radius(size: str) -> Optional[float]:
    """The size's radius in Stim units, or None for the whole patch."""
    radius = RADIUS_BY_SIZE[size]
    if radius is None:
        return None
    return radius * STIM_UNITS_PER_SPACING


def _centres(
    generator: numpy.random.Generator,
    quiet_circuit: stim.Circuit,
    count: int,
) -> numpy.ndarray:
    """(count, 2): centres uniform over the box the qubits span."""
    coordinates = quiet_circuit.get_final_qubit_coordinates()
    qubit_positions = coordinates.values()
    position_list = list(qubit_positions)
    positions = numpy.array(position_list)
    low = positions.min(axis=0)
    high = positions.max(axis=0)
    return generator.uniform(low, high, size=(count, 2))


def _paired_events(
    quiet_circuit: stim.Circuit,
    generator: numpy.random.Generator,
    level: float,
    radius: Optional[float],
    centres: numpy.ndarray,
) -> tuple:
    """Copies A and B, (trials, detectors) each, B = A XOR the burst's.

    Every trial draws its own quiet shot and its own burst shot.
    """
    trial_count = len(centres)
    quiet_seed = _stim_seed(generator)
    sampler = quiet_circuit.compile_detector_sampler(seed=quiet_seed)
    events_a = sampler.sample(trial_count)
    table = detector_formation.build_formation_table(quiet_circuit, ROUNDS)
    flips = numpy.zeros_like(events_a)
    for trial, centre in enumerate(centres):
        centre_pair = (float(centre[0]), float(centre[1]))
        settings = burst_settings(level, radius, centre_pair)
        noise = stim_device.burst_noise(quiet_circuit, table, settings)
        burst_seed = _stim_seed(generator)
        burst_sampler = noise.compile_detector_sampler(seed=burst_seed)
        (flips[trial],) = burst_sampler.sample(1)
    events_b = events_a ^ flips
    return events_a, events_b


def _stim_seed(generator: numpy.random.Generator) -> int:
    # numpy draws below its bound; Stim takes any 64-bit seed
    seed = generator.integers(STIM_SEED_BOUND)
    return int(seed)


def _paired_ratios(
    lines: cusum.AlarmLines,
    events_a: numpy.ndarray,
    events_b: numpy.ndarray,
) -> dict:
    """Both copies scored side by side from new charts, split by copy."""
    trial_count = len(events_a)
    both_copies = numpy.concatenate([events_a, events_b])
    stream_count = len(both_copies)
    state = lines.new_state(stream_count)
    ratios = lines.score_ratios(state, both_copies)
    return {"a": ratios[:trial_count], "b": ratios[trial_count:]}


def _trial_rows(
    centres: numpy.ndarray,
    radius: Optional[float],
    firsts_a: numpy.ndarray,
    firsts_b: numpy.ndarray,
) -> list:
    """One row per trial and line; a whole-patch burst has no centre."""
    rows = []
    for trial, centre in enumerate(centres):
        centre_x, centre_y = _centre_cells(centre, radius)
        for line, rate in enumerate(FALSE_ALARMS_PER_SECOND):
            first_alarm_a = _alarm_cell(firsts_a[trial, line])
            first_alarm_b = _alarm_cell(firsts_b[trial, line])
            row = {
                "trial": trial,
                "centre_x": centre_x,
                "centre_y": centre_y,
                "alarm_line": rate,
                "first_alarm_a": first_alarm_a,
                "first_alarm_b": first_alarm_b,
            }
            rows.append(row)
    return rows


def _centre_cells(centre: numpy.ndarray, radius: Optional[float]) -> tuple:
    if radius is None:
        return None, None
    return float(centre[0]), float(centre[1])


def _alarm_cell(first_round: int) -> Optional[int]:
    if first_round == cusum.NO_ALARM:
        return None
    return int(first_round)


def _trace_rows(
    lines: cusum.AlarmLines, ratios: dict, class_labels: dict
) -> list:
    """The kept rounds' score over level, per copy and line."""
    first_offset = TRACE_FIRST_ROUND - lines.layout.first_bulk_round
    last_offset = TRACE_LAST_ROUND - lines.layout.first_bulk_round
    after_last_offset = last_offset + 1
    rows = []
    for copy, copy_ratios in ratios.items():
        kept = copy_ratios[0, first_offset:after_last_offset]
        for offset, line_ratios in enumerate(kept):
            round_index = TRACE_FIRST_ROUND + offset
            copy_labels = class_labels | {"copy": copy, "round": round_index}
            line_rows = _line_cells(copy_labels, line_ratios)
            rows.extend(line_rows)
    return rows


def _line_cells(copy_labels: dict, line_ratios: numpy.ndarray) -> list:
    rows = []
    for rate, ratio in zip(FALSE_ALARMS_PER_SECOND, line_ratios, strict=True):
        cells = {"alarm_line": rate, "score_over_level": float(ratio)}
        row = copy_labels | cells
        rows.append(row)
    return rows


def _seconds(scored_rounds: int) -> float:
    microseconds = scored_rounds * ROUND_PERIOD_MICROSECONDS
    return microseconds / MICROSECONDS_PER_SECOND


if __name__ == "__main__":
    experiment = burst_detection()
    experiment.main()
