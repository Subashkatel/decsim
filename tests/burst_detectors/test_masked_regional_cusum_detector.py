"""The masked_regional_cusum row against its written rules and Stim.

The row alarms on the rounds written_rules.py's transcription of the
multichart rule (Zhang et al. 1410.8765 lines 338-350) alarms on, and
its calibrated thresholds hold Stim's quiet shots to the target rate.
Its offline form, AlarmLines, scores a stream as the written rules do
and holds the row's own thresholds at each of its rates.
"""

import dataclasses
import functools

import numpy
import pytest

import decsim.burst_detectors.masked_regional_cusum.detector as detector_module
import decsim.burst_detectors.settings as burst_detector_settings
import decsim.engine as engine_module
import decsim.frontends.settings as workload_settings
import tests.burst_detectors.burst_rounds as burst_rounds
import tests.burst_detectors.written_rules as written_rules


def test_the_row_fires_on_the_rounds_the_written_rules_alarm():
    """Alarm, at the row's own thresholds, through observe_round."""
    settings = burst_rounds.CUSUM.Settings(calibration_shot_count=2000)
    detector = burst_rounds.cusum_detector(
        settings, rounds=burst_rounds.LONG_ROUNDS
    )
    positions, pairs, usual_rates = burst_rounds.bank_inputs(detector)
    circuit, sampled = burst_rounds.long_burst_shot(seed=4)
    rows = burst_rounds.bulk_rows(circuit, sampled, positions)
    reference, _ = written_rules.reference_scores(
        rows, positions, pairs, usual_rates
    )
    charts = detector.charts_by_operation[1]
    thresholds = charts.calibration.thresholds
    reaches = reference >= thresholds
    is_alarm = reaches.any(axis=1)
    alarms = numpy.flatnonzero(is_alarm)

    shot_rounds = burst_rounds.rounds_of_events(
        circuit, burst_rounds.LONG_ROUNDS, sampled
    )
    burst_rounds.feed(detector, shot_rounds)
    fired = _fired_rounds(charts.flags.flags)

    assert len(alarms) > 0
    assert fired == list(alarms)


def test_quiet_shots_alarm_at_the_calibrated_rate():
    """0.05 of 30 us shots is 1667 false alarms a second.

    4000 fresh quiet shots (another seed than the calibration's) alarm
    within five binomial standard errors of that share.
    """
    shot_seconds = burst_rounds.ROUNDS * 1e-6
    rate = 0.05 / shot_seconds
    settings = burst_rounds.CUSUM.Settings(
        false_alarms_per_second=rate, calibration_shot_count=4000
    )
    detector = burst_rounds.cusum_detector(settings)
    calibration = detector.charts_by_operation[1].calibration
    rows = _quiet_rows(4000, 11, calibration.layout.positions)

    state = calibration.bank.new_state(4000, 2)
    maxima = calibration.bank.block_maxima(state, rows, 2)

    reaches = maxima >= calibration.thresholds
    is_alarmed = reaches.any(axis=1)
    alarmed_share = numpy.mean(is_alarmed)
    # sqrt(0.05 x 0.95 / 4000), the binomial standard error
    standard_error = 0.003446
    difference = alarmed_share - 0.05
    bound = 5 * standard_error
    assert abs(difference) < bound


def test_a_whole_patch_burst_is_flagged_from_its_onset():
    """The study's 0.03 per second; the flag's change point is round 12."""
    settings = burst_rounds.CUSUM.Settings()
    detector = burst_rounds.cusum_detector(settings)
    burst = burst_rounds.whole_patch_burst(onset_round=12, probability=0.1)
    rounds = burst_rounds.sampled_rounds(burst, seed=3)
    burst_rounds.feed(detector, rounds[:14])
    burst_window = burst_rounds.window(12, 14)
    before_it = burst_rounds.window(1, 11)

    assert detector.is_burst_window(burst_window)
    assert not detector.is_burst_window(before_it)


def test_a_quiet_shot_is_not_flagged_by_the_cusum():
    settings = burst_rounds.CUSUM.Settings()
    detector = burst_rounds.cusum_detector(settings)
    circuit = burst_rounds.memory_circuit()
    rounds = burst_rounds.sampled_rounds(circuit, seed=3)
    burst_rounds.feed(detector, rounds)
    window = burst_rounds.window(1, burst_rounds.ROUNDS)

    assert not detector.is_burst_window(window)


def test_without_burst_priors_a_cusum_flagged_window_keeps_its_model():
    detector = burst_rounds.flagged_cusum_detector(raise_strong_priors=False)
    model = burst_rounds.window_model(12, 17)
    window = burst_rounds.window(12, 17)

    kept = detector.with_burst_priors(window, model)

    assert kept is model


@pytest.mark.filterwarnings("error")
def test_a_noiseless_background_fires_the_cusum_on_its_first_event():
    """Every usual rate floors at 1e-6, so one event is far past it.

    The region no fault reaches keeps its priors.
    """
    settings = burst_rounds.CUSUM.Settings(raise_strong_priors=True)
    circuit = workload_settings.memory_circuit(
        burst_rounds.CODE_TASK, burst_rounds.ROUNDS, burst_rounds.DISTANCE, 0.0
    )
    engine = engine_module.Engine()
    detector = burst_rounds.CUSUM(
        settings, engine, {1: (circuit, burst_rounds.ROUNDS)}, 1.0
    )
    quiet_before = burst_rounds.quiet_rounds(11)
    burst_rounds.feed(detector, quiet_before)
    quiet_window = burst_rounds.window(1, 11)
    was_quiet = detector.is_burst_window(quiet_window)
    one_event = (1,) + burst_rounds.BULK_ROUND_QUIET[1:]
    detector.observe_round(1, 12, one_event)
    window = burst_rounds.window(12, 12)
    model = burst_rounds.window_model(12, 12)
    kept = detector.with_burst_priors(window, model)

    assert not was_quiet
    assert detector.is_burst_window(window)
    assert kept is model


def test_a_rate_of_a_false_alarm_a_shot_is_refused():
    settings = burst_rounds.CUSUM.Settings(false_alarms_per_second=1e6)

    with pytest.raises(ValueError, match="at least one false alarm a shot"):
        burst_rounds.cusum_detector(settings)


@pytest.mark.parametrize(
    ("key", "value", "sentence"),
    [
        ("mask_count", "8", "whole number of firings"),
        ("mask_window_rounds", 0, "whole number of rounds"),
        ("calibration_shot_count", 0, "whole number of shots"),
        ("region_radii", [-1.0], "each a number at least zero"),
        ("fault_rate_multipliers", [1.0], "each a number above one"),
        ("fault_rate_multipliers", [], "each a number above one"),
        ("unmasked_share_floor", 1.0, "not including, 1"),
        ("false_alarms_per_second", 0, "a rate above zero"),
        ("datapath_count", 2, "datapath_count prices the chart bank"),
        ("pipeline_cycles", 3, "pipeline_cycles prices the chart bank"),
    ],
)
def test_a_wrong_cusum_key_is_refused_by_a_sentence(key, value, sentence):
    section = {"kind": "masked_regional_cusum", key: value}

    with pytest.raises(ValueError, match=sentence):
        burst_detector_settings.BurstDetectorSettings.from_yaml(
            section, burst_rounds.CLOCKS
        )


def test_the_cusum_keys_reach_the_rows_settings():
    section = {
        "kind": "masked_regional_cusum",
        "mask_count": None,
        "region_radii": [],
        "fault_rate_multipliers": [3],
        "clock": "fridge",
        "datapath_count": 2,
    }

    detector_section = burst_detector_settings.BurstDetectorSettings.from_yaml(
        section, burst_rounds.CLOCKS
    )

    settings = detector_section.row_settings
    assert settings.mask_count is None
    assert settings.region_radii == ()
    assert settings.fault_rate_multipliers == (3.0,)
    assert settings.datapath_count == 2
    assert settings.clock == burst_rounds.CLOCKS.clock("fridge")


# Shares 0.1 and 0.02 of the 160-round shots of 1 us each.
LINE_RATES = (625.0, 125.0)
LINE_SETTINGS = burst_rounds.CUSUM.Settings(calibration_shot_count=2000)


def test_lines_with_no_warm_up_and_one_shot_a_stream_are_cold_thresholds():
    """1,000 streams of one shot from cold are the row's 1,000 cold shots."""
    one_shot_settings = dataclasses.replace(
        LINE_SETTINGS, calibration_shot_count=1000
    )
    circuit = long_circuit()
    lines = detector_module.AlarmLines.calibrated(
        circuit,
        burst_rounds.LONG_ROUNDS,
        one_shot_settings,
        1.0,
        LINE_RATES,
        warm_up_shots=0,
    )
    first_settings = dataclasses.replace(
        one_shot_settings, false_alarms_per_second=625.0
    )
    second_settings = dataclasses.replace(
        one_shot_settings, false_alarms_per_second=125.0
    )

    first_row = burst_rounds.cusum_detector(
        first_settings, rounds=burst_rounds.LONG_ROUNDS
    )
    second_row = burst_rounds.cusum_detector(
        second_settings, rounds=burst_rounds.LONG_ROUNDS
    )

    first_calibration = first_row.charts_by_operation[1].calibration
    second_calibration = second_row.charts_by_operation[1].calibration
    assert list(lines.levels[0]) == list(first_calibration.thresholds)
    assert list(lines.levels[1]) == list(second_calibration.thresholds)


@pytest.mark.parametrize("rate", [0.0, -1.0])
def test_lines_at_a_rate_of_zero_or_below_are_refused(rate):
    """Refused before any quiet shot is drawn or any level is read."""
    circuit = long_circuit()

    with pytest.raises(ValueError) as refused:
        detector_module.AlarmLines.calibrated(
            circuit,
            burst_rounds.LONG_ROUNDS,
            LINE_SETTINGS,
            1.0,
            [125.0, rate],
            warm_up_shots=0,
        )

    assert str(refused.value) == (
        "burst_detector.false_alarms_per_second must be a rate above zero "
        f"(got {rate!r})"
    )


def test_two_shots_on_one_state_score_as_one_stream_of_the_written_rules():
    """A quiet shot, then a burst shot; the rules read their rows joined."""
    lines = long_alarm_lines()
    circuit, burst_shot = burst_rounds.long_burst_shot(seed=4)
    sampler = circuit.compile_detector_sampler(seed=5)
    quiet_shots = sampler.sample(1)
    state = lines.new_state(1)

    quiet_ratios = lines.score_ratios(state, quiet_shots)
    burst_shots = burst_shot[None, :]
    burst_ratios = lines.score_ratios(state, burst_shots)

    positions, pairs, usual_rates = line_inputs(lines)
    quiet_rows = burst_rounds.bulk_rows(circuit, quiet_shots[0], positions)
    burst_rows = burst_rounds.bulk_rows(circuit, burst_shot, positions)
    rows = numpy.concatenate([quiet_rows, burst_rows])
    reference, _ = written_rules.reference_scores(
        rows, positions, pairs, usual_rates
    )
    reference_scores = reference[:, None, :]
    reference_ratios = reference_scores / lines.levels
    expected = numpy.max(reference_ratios, axis=2)
    scored = numpy.concatenate([quiet_ratios[0], burst_ratios[0]])
    assert numpy.allclose(scored, expected)
    assert expected.max() >= 1.0


def test_a_first_alarm_is_the_first_round_a_line_fires_from_the_earliest():
    """Offset i is round 2 + i on d = 5; a firing before round 3 is unread."""
    lines = long_alarm_lines()
    ratios = numpy.zeros((2, 5, 2))
    ratios[0, 1, 0] = 1.0
    ratios[0, 3, 0] = 2.0
    ratios[1, 4, 0] = 1.5
    ratios[1, 0, 1] = 3.0
    ratios[1, 2, 1] = 0.99

    first_rounds = lines.first_alarm_rounds(ratios, earliest_round=3)

    no_alarm = detector_module.NO_ALARM
    assert first_rounds.tolist() == [[3, no_alarm], [6, no_alarm]]


@functools.lru_cache(maxsize=1)
def long_alarm_lines() -> detector_module.AlarmLines:
    """The 160-round d = 5 memory's lines at LINE_RATES, built once."""
    circuit = long_circuit()
    return detector_module.AlarmLines.calibrated(
        circuit,
        burst_rounds.LONG_ROUNDS,
        LINE_SETTINGS,
        1.0,
        LINE_RATES,
        warm_up_shots=1,
    )


def long_circuit():
    return workload_settings.memory_circuit(
        burst_rounds.CODE_TASK,
        burst_rounds.LONG_ROUNDS,
        burst_rounds.DISTANCE,
        burst_rounds.PHYSICAL_ERROR_PROBABILITY,
    )


def line_inputs(lines: detector_module.AlarmLines) -> tuple:
    """The bank's positions, pairs and floored usual rates."""
    pairs = []
    for row in lines.bank.pair_incidence:
        checks = numpy.flatnonzero(row)
        pairs.append(tuple(checks))
    return lines.layout.positions, pairs, lines.bank.usual_rates


def _fired_rounds(flags):
    """The offsets of the rounds a flag was recorded on."""
    fired = []
    for index, flag in enumerate(flags):
        if flag is not None:
            fired.append(index)
    return fired


def _quiet_rows(shot_count, seed, positions):
    circuit = workload_settings.memory_circuit(
        burst_rounds.CODE_TASK,
        burst_rounds.ROUNDS,
        burst_rounds.DISTANCE,
        burst_rounds.PHYSICAL_ERROR_PROBABILITY,
    )
    sampler = circuit.compile_detector_sampler(seed=seed)
    shots = sampler.sample(shot_count)
    rows = []
    for sampled in shots:
        shot_rows = burst_rounds.bulk_rows(circuit, sampled, positions)
        rows.append(shot_rows)
    return numpy.asarray(rows)
