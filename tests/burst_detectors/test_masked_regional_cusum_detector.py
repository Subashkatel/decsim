"""The masked_regional_cusum row against its written rules and Stim."""

import numpy
import pytest

import decsim.burst_detectors.settings as burst_detector_settings
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.engine as engine_module
import decsim.frontends.settings as workload_settings
import decsim.qpu.stim_device as stim_device
import tests.burst_detectors.burst_rounds as burst_rounds
from tests.burst_detectors.burst_rounds import (
    BULK_ROUND_LOUD,
    BULK_ROUND_QUIET,
    CLOCKS,
    CODE_TASK,
    CUSUM,
    DISTANCE,
    GRAPHLIKE,
    PHYSICAL_ERROR_PROBABILITY,
    ROUNDS,
)

# The masked regional CUSUM, held against a reference written straight
# from the method's rules, one function per rule and none of the row's
# code:
#   flag: a check whose repeats (fired this round and the one before),
#     or whose joint firings with its same-basis pair, reach mask_count
#     in the last 64 rounds is flagged that round;
#   mask: a check flagged in the last 100 rounds is left out;
#   regions: a disc of each radius around each check, then the patch;
#   expected count: the region's usual count mu times f, the share of
#     its usual rate still unmasked;
#   step: for each design, c log rho - (rho - 1) mu f, the score kept
#     at zero or above (Page 1954; Lucas 1985 for counts);
#   tracking: mu moves toward c / f over 5000 rounds while f > 0.2;
#   groups: each design's largest score at each radius;
#   alarm: any group at or over its threshold (multichart CUSUM).
REFERENCE_RADII = (0.0, 1.5, 2.3, 3.2)
REFERENCE_DESIGNS = (2.0, 4.0, 20.0)
# a stream long enough for the 64-round mask window and its 100-round
# hold, with a burst loud enough to mask checks
LONG_ROUNDS = 160
LONG_BURST_ONSET = 40
# the rules' windows: flag, mask hold, tracking, and the share floor
REFERENCE_WINDOW = 64
REFERENCE_HOLD = 100
REFERENCE_TRACKING = 5000.0
REFERENCE_SHARE_FLOOR = 0.2
# grid units: neighbouring checks sqrt(2) apart
GRID_UNIT = 0.7071067811865476


def _long_burst_shot(seed):
    """A d = 5 shot of LONG_ROUNDS rounds, bursting from LONG_BURST_ONSET."""
    circuit = workload_settings.memory_circuit(
        CODE_TASK, LONG_ROUNDS, DISTANCE, PHYSICAL_ERROR_PROBABILITY
    )
    table = detector_formation.build_formation_table(circuit, LONG_ROUNDS)
    burst = stim_device.BurstStimDevice.Settings(
        burst_onset_round=LONG_BURST_ONSET, burst_error_probability=0.05
    )
    shot_circuit = stim_device.burst_circuit(circuit, table, burst)
    sampler = shot_circuit.compile_detector_sampler(seed=seed)
    shots = sampler.sample(1)
    return circuit, shots[0]


def _bulk_rows(circuit, sampled, positions):
    """(bulk rounds, checks) of 0/1, read off Stim's detector coordinates.

    Stim's time coordinate 0 is the first round and its last is the
    data readout, so the bulk rows are the times between.
    """
    coordinates = circuit.get_detector_coordinates()
    column_by_position = {}
    for column, position in enumerate(positions):
        column_by_position[position] = column
    times = [coordinates[detector][2] for detector in coordinates]
    readout_time = int(max(times))
    shape = (readout_time - 1, len(positions))
    rows = numpy.zeros(shape, dtype=int)
    for detector, coordinate in coordinates.items():
        time = int(coordinate[2])
        if 1 <= time < readout_time:
            position = (coordinate[0], coordinate[1])
            column = column_by_position[position]
            rows[time - 1, column] = sampled[detector]
    return rows


def _reference_regions(positions, radii):
    """Regions: discs of each radius around each check, then the patch."""
    coordinates = numpy.asarray(positions)
    grid = coordinates * GRID_UNIT
    members = []
    scales = []
    for scale, radius in enumerate(radii):
        for center in grid:
            offsets = grid - center
            distances = numpy.linalg.norm(offsets, axis=1)
            reach = radius + 1e-9
            is_inside = distances <= reach
            disc = numpy.flatnonzero(is_inside)
            members.append(disc)
            scales.append(scale)
    whole = numpy.arange(len(positions))
    members.append(whole)
    scales.append(len(radii))
    return members, scales


def _reference_flags(rows, pairs, mask_count):
    """Flag: each check's flag per round, over the last 64 rounds.

    A repeat at round t is a firing at t and t - 1; a pair's joint
    firing is both its checks at t. mask_count None flags nothing.
    """
    round_count, check_count = rows.shape
    flags = numpy.zeros(rows.shape, dtype=bool)
    if mask_count is None:
        return flags
    silent_row = numpy.zeros((1, check_count), dtype=int)
    previous = numpy.vstack([silent_row, rows[:-1]])
    repeats = rows & previous
    for round_index in range(round_count):
        first = _window_start(round_index)
        window_repeats = repeats[first : round_index + 1].sum(axis=0)
        flags[round_index] = window_repeats >= mask_count
        _flag_joint_pairs(flags, rows, pairs, round_index, mask_count)
    return flags


def _window_start(round_index):
    """The first of the 64 rounds that end at round_index, from zero."""
    start = round_index - REFERENCE_WINDOW + 1
    return max(0, start)


def _flag_joint_pairs(flags, rows, pairs, round_index, mask_count):
    first = _window_start(round_index)
    window = rows[first : round_index + 1]
    for first_check, second_check in pairs:
        joint = window[:, first_check] & window[:, second_check]
        if joint.sum() >= mask_count:
            flags[round_index, first_check] = True
            flags[round_index, second_check] = True


def _reference_kept(flags, round_index):
    """Mask: a check counts unless it was flagged in [t - 100, t]."""
    hold_start = round_index - REFERENCE_HOLD
    first = max(0, hold_start)
    recent = flags[first : round_index + 1]
    return ~recent.any(axis=0)


def _reference_scores(rows, positions, pairs, usual_rates, mask_count=8):
    """Regions to groups, region by region, round by round.

    Returns (rounds, groups) design-major, and the masked check-rounds.
    """
    members, scales = _reference_regions(positions, REFERENCE_RADII)
    flags = _reference_flags(rows, pairs, mask_count)
    usual_counts = []
    for member in members:
        region_rate = usual_rates[member].sum()
        usual_counts.append(region_rate)
    scores = numpy.zeros((len(REFERENCE_DESIGNS), len(members)))
    group_scores = []
    masked = 0
    for round_index, row in enumerate(rows):
        kept = _reference_kept(flags, round_index)
        is_masked = ~kept
        masked += is_masked.sum()
        for region, member in enumerate(members):
            _reference_region_step(
                scores, usual_counts, region, member, row, kept, usual_rates
            )
        groups = _reference_groups(scores, scales)
        group_scores.append(groups)
    return numpy.asarray(group_scores), masked


def _reference_region_step(
    scores, usual_counts, region, member, row, kept, usual_rates
):
    """Expected count, step and tracking for one region, in that order."""
    kept_member = member[kept[member]]
    count = row[kept_member].sum()
    kept_rate = usual_rates[kept_member].sum()
    region_rate = usual_rates[member].sum()
    share = kept_rate / region_rate
    expected = usual_counts[region] * share
    for design, multiplier in enumerate(REFERENCE_DESIGNS):
        fired = (1 - (1 - 2 * usual_rates[member]) ** multiplier) / 2
        ratio = fired.sum() / region_rate
        step = count * numpy.log(ratio) - (ratio - 1) * expected
        score = scores[design, region] + step
        scores[design, region] = max(0.0, score)
    if share > REFERENCE_SHARE_FLOOR:
        error = count / share - usual_counts[region]
        usual_counts[region] += error / REFERENCE_TRACKING


def _reference_groups(scores, scales):
    """Groups: each design's largest region score at each radius."""
    scale_count = max(scales) + 1
    groups = []
    for design_scores in scores:
        for scale in range(scale_count):
            is_at_scale = numpy.asarray(scales) == scale
            at_scale = design_scores[is_at_scale]
            groups.append(max(at_scale))
    return groups


def _bank_inputs(detector):
    """The bank's positions, pairs and floored usual rates."""
    calibration = detector.charts_by_operation[1].calibration
    bank = calibration.bank
    positions = calibration.layout.positions
    pair_rows = bank.pair_incidence
    pairs = []
    for row in pair_rows:
        checks = numpy.flatnonzero(row)
        pairs.append(tuple(checks))
    return positions, pairs, bank.usual_rates


def _bank_scores(detector, rows):
    """The row's own bank scoring the rows from a cold state."""
    calibration = detector.charts_by_operation[1].calibration
    streams = rows[None, :, :]
    bank = calibration.bank
    state = bank.new_state(1, 2)
    group_scores = []
    for offset in range(rows.shape[0]):
        round_index = offset + 2
        round_rows = streams[:, offset]
        round_scores = bank.score_round(state, round_rows, round_index)
        group_scores.append(round_scores[0])
    return numpy.asarray(group_scores)


def test_the_bank_scores_every_group_as_the_written_rules_do():
    """All 15 group scores of every round, on a burst that trips the mask.

    Only the order of the float sums differs from the transcription.
    """
    settings = CUSUM.Settings(calibration_shots=200)
    detector = burst_rounds.cusum_detector(settings, rounds=LONG_ROUNDS)
    positions, pairs, usual_rates = _bank_inputs(detector)
    circuit, sampled = _long_burst_shot(seed=4)
    rows = _bulk_rows(circuit, sampled, positions)

    scores = _bank_scores(detector, rows)
    reference, masked = _reference_scores(rows, positions, pairs, usual_rates)

    assert masked > 0
    numpy.testing.assert_allclose(scores, reference, rtol=1e-9, atol=1e-9)


def test_the_row_fires_on_the_rounds_the_written_rules_alarm():
    """Alarm, at the row's own thresholds, through observe_round."""
    settings = CUSUM.Settings(calibration_shots=2000)
    detector = burst_rounds.cusum_detector(settings, rounds=LONG_ROUNDS)
    positions, pairs, usual_rates = _bank_inputs(detector)
    circuit, sampled = _long_burst_shot(seed=4)
    rows = _bulk_rows(circuit, sampled, positions)
    reference, _ = _reference_scores(rows, positions, pairs, usual_rates)
    charts = detector.charts_by_operation[1]
    thresholds = charts.calibration.thresholds
    reaches = reference >= thresholds
    is_alarm = reaches.any(axis=1)
    alarms = numpy.flatnonzero(is_alarm)

    shot_rounds = burst_rounds.rounds_of_events(circuit, LONG_ROUNDS, sampled)
    burst_rounds.feed(detector, shot_rounds)
    flags = charts.flags.flags
    fired = [index for index, flag in enumerate(flags) if flag is not None]

    assert len(alarms) > 0
    assert fired == list(alarms)


def test_a_null_mask_count_scores_the_unmasked_regional_cusum():
    settings = CUSUM.Settings(mask_count=None, calibration_shots=200)
    detector = burst_rounds.cusum_detector(settings, rounds=LONG_ROUNDS)
    positions, pairs, usual_rates = _bank_inputs(detector)
    circuit, sampled = _long_burst_shot(seed=4)
    rows = _bulk_rows(circuit, sampled, positions)

    scores = _bank_scores(detector, rows)
    reference, masked = _reference_scores(
        rows, positions, pairs, usual_rates, mask_count=None
    )

    assert masked == 0
    numpy.testing.assert_allclose(scores, reference, rtol=1e-9, atol=1e-9)


def test_no_radii_leave_the_whole_patch_the_only_region():
    settings = CUSUM.Settings(region_radii=(), calibration_shots=200)
    detector = burst_rounds.cusum_detector(settings)
    bank = detector.charts_by_operation[1].calibration.bank
    thresholds = detector.charts_by_operation[1].calibration.thresholds

    assert bank.incidence.shape == (24, 1)
    assert len(thresholds) == len(REFERENCE_DESIGNS)


def _reference_pairs(circuit):
    """For each data qubit, its two same-basis checks, the flag's pairs.

    On Stim's rotated surface code data qubits sit at odd coordinates
    and X checks carry an H, and a check beside a data qubit sits one
    unit off in both coordinates.
    """
    coordinates = circuit.get_final_qubit_coordinates()
    x_checks = _hadamard_targets(circuit)
    pairs = set()
    for x, y in coordinates.values():
        if x % 2 == 1 and y % 2 == 1:
            pairs |= _pairs_beside(coordinates, x_checks, x, y)
    return pairs


def _hadamard_targets(circuit):
    targets = set()
    for instruction in circuit.flattened():
        if instruction.name != "H":
            continue
        for target in instruction.targets_copy():
            targets.add(target.value)
    return targets


def _pairs_beside(coordinates, x_checks, x, y):
    by_basis = {True: [], False: []}
    for qubit, (check_x, check_y) in coordinates.items():
        x_offset = check_x - x
        y_offset = check_y - y
        is_beside = abs(x_offset) == 1 and abs(y_offset) == 1
        if is_beside:
            by_basis[qubit in x_checks].append((check_x, check_y))
    pairs = set()
    for checks in by_basis.values():
        if len(checks) == 2:
            pair = frozenset(checks)
            pairs.add(pair)
    return pairs


def test_the_pairs_are_the_checks_one_data_qubit_flip_fires_together():
    settings = CUSUM.Settings(calibration_shots=200)
    detector = burst_rounds.cusum_detector(settings)
    positions, pairs, _ = _bank_inputs(detector)
    circuit = workload_settings.memory_circuit(
        CODE_TASK, ROUNDS, DISTANCE, PHYSICAL_ERROR_PROBABILITY
    )

    bank_pairs = set()
    for first, second in pairs:
        pair = frozenset([positions[first], positions[second]])
        bank_pairs.add(pair)

    assert len(bank_pairs) == 30
    assert bank_pairs == _reference_pairs(circuit)


def _quiet_rows(shot_count, seed, positions):
    circuit = workload_settings.memory_circuit(
        CODE_TASK, ROUNDS, DISTANCE, PHYSICAL_ERROR_PROBABILITY
    )
    sampler = circuit.compile_detector_sampler(seed=seed)
    shots = sampler.sample(shot_count)
    rows = []
    for sampled in shots:
        shot_rows = _bulk_rows(circuit, sampled, positions)
        rows.append(shot_rows)
    return numpy.asarray(rows)


def test_quiet_shots_alarm_at_the_calibrated_rate():
    """0.05 of 30 us shots is 1667 false alarms a second.

    4000 fresh quiet shots (another seed than the calibration's) alarm
    within five binomial standard errors of that share.
    """
    shot_seconds = ROUNDS * 1e-6
    rate = 0.05 / shot_seconds
    settings = CUSUM.Settings(
        false_alarms_per_second=rate, calibration_shots=4000
    )
    detector = burst_rounds.cusum_detector(settings)
    calibration = detector.charts_by_operation[1].calibration
    rows = _quiet_rows(4000, 11, calibration.layout.positions)

    maxima = calibration.bank.block_maxima(rows, 2)

    reaches = maxima >= calibration.thresholds
    is_alarmed = reaches.any(axis=1)
    alarmed_share = numpy.mean(is_alarmed)
    # sqrt(0.05 x 0.95 / 4000), the binomial standard error
    standard_error = 0.003446
    difference = alarmed_share - 0.05
    assert abs(difference) < 5 * standard_error


def test_a_whole_patch_burst_is_flagged_from_its_onset():
    """The study's 0.03 per second; the flag's change point is round 12."""
    settings = CUSUM.Settings()
    detector = burst_rounds.cusum_detector(settings)
    burst = burst_rounds.whole_patch_burst(onset_round=12, probability=0.1)
    rounds = burst_rounds.sampled_rounds(burst, seed=3)
    burst_rounds.feed(detector, rounds[:14])
    burst_window = burst_rounds.window(12, 14)
    before_it = burst_rounds.window(1, 11)

    assert detector.is_burst_window(burst_window)
    assert not detector.is_burst_window(before_it)


def test_a_quiet_shot_is_not_flagged_by_the_cusum():
    settings = CUSUM.Settings()
    detector = burst_rounds.cusum_detector(settings)
    circuit = burst_rounds.memory_circuit()
    rounds = burst_rounds.sampled_rounds(circuit, seed=3)
    burst_rounds.feed(detector, rounds)
    window = burst_rounds.window(1, ROUNDS)

    assert not detector.is_burst_window(window)


def _cusum_flagged_detector(raise_strong_priors):
    """Every check loud from round 12 to 17: the whole patch in burst."""
    settings = CUSUM.Settings(raise_strong_priors=raise_strong_priors)
    detector = burst_rounds.cusum_detector(settings)
    quiet_before = burst_rounds.quiet_rounds(11)
    loud = [BULK_ROUND_LOUD] * 6
    rounds = [*quiet_before, *loud]
    burst_rounds.feed(detector, rounds)
    return detector


def test_cusum_burst_priors_cap_the_region_and_keep_the_graph():
    detector = _cusum_flagged_detector(raise_strong_priors=True)
    window = burst_rounds.window(12, 17)
    model = burst_rounds.window_model(12, 17)
    raised = detector.with_burst_priors(window, model)
    faults = model.require_faults(GRAPHLIKE)
    raised_faults = raised.require_faults(GRAPHLIKE)
    changed_checks = faults.check != raised_faults.check
    is_capped = raised_faults.priors == 0.5

    assert changed_checks.nnz == 0
    assert numpy.all(is_capped)


def test_a_window_before_the_cusum_flag_keeps_its_model():
    detector = _cusum_flagged_detector(raise_strong_priors=True)
    model = burst_rounds.window_model(1, 6)
    window = burst_rounds.window(1, 6)

    kept = detector.with_burst_priors(window, model)

    assert kept is model


def test_without_burst_priors_a_cusum_flagged_window_keeps_its_model():
    detector = _cusum_flagged_detector(raise_strong_priors=False)
    model = burst_rounds.window_model(12, 17)
    window = burst_rounds.window(12, 17)

    kept = detector.with_burst_priors(window, model)

    assert kept is model


@pytest.mark.filterwarnings("error")
def test_a_noiseless_background_fires_the_cusum_on_its_first_event():
    """Every usual rate floors at 1e-6, so one event is far past it.

    The region no fault reaches keeps its priors.
    """
    settings = CUSUM.Settings(raise_strong_priors=True)
    circuit = workload_settings.memory_circuit(CODE_TASK, ROUNDS, DISTANCE, 0.0)
    engine = engine_module.Engine()
    detector = CUSUM(settings, engine, {1: (circuit, ROUNDS)}, 1.0)
    quiet_before = burst_rounds.quiet_rounds(11)
    burst_rounds.feed(detector, quiet_before)
    quiet_window = burst_rounds.window(1, 11)
    was_quiet = detector.is_burst_window(quiet_window)
    one_event = (1,) + BULK_ROUND_QUIET[1:]
    detector.observe_round(1, 12, one_event)
    window = burst_rounds.window(12, 12)
    model = burst_rounds.window_model(12, 12)
    kept = detector.with_burst_priors(window, model)

    assert not was_quiet
    assert detector.is_burst_window(window)
    assert kept is model


def test_a_rate_of_a_false_alarm_a_shot_is_refused():
    settings = CUSUM.Settings(false_alarms_per_second=1e6)

    with pytest.raises(ValueError, match="at least one false alarm a shot"):
        burst_rounds.cusum_detector(settings)


@pytest.mark.parametrize(
    ("key", "value", "sentence"),
    [
        ("mask_count", "8", "whole number of firings"),
        ("mask_window_rounds", 0, "whole number of rounds"),
        ("calibration_shots", 0, "whole number of shots"),
        ("region_radii", [-1.0], "each a number at least zero"),
        ("fault_rate_multipliers", [1.0], "each a number above one"),
        ("fault_rate_multipliers", [], "each a number above one"),
        ("unmasked_share_floor", 1.0, "not including, 1"),
        ("false_alarms_per_second", 0, "a rate above zero"),
        ("datapaths", 2, "datapaths prices the chart bank"),
        ("pipeline_cycles", 3, "pipeline_cycles prices the chart bank"),
    ],
)
def test_a_wrong_cusum_key_is_refused_by_a_sentence(key, value, sentence):
    section = {"kind": "masked_regional_cusum", key: value}

    with pytest.raises(ValueError, match=sentence):
        burst_detector_settings.BurstDetectorSettings.from_yaml(section, CLOCKS)


def test_the_cusum_keys_reach_the_rows_settings():
    section = {
        "kind": "masked_regional_cusum",
        "mask_count": None,
        "region_radii": [],
        "fault_rate_multipliers": [3],
        "clock": "fridge",
        "datapaths": 2,
    }

    detector_section = burst_detector_settings.BurstDetectorSettings.from_yaml(
        section, CLOCKS
    )

    settings = detector_section.row_settings
    assert settings.mask_count is None
    assert settings.region_radii == ()
    assert settings.fault_rate_multipliers == (3.0,)
    assert settings.datapaths == 2
    assert settings.clock == CLOCKS.clock("fridge")
