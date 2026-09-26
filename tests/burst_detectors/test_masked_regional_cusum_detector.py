"""The masked_regional_cusum row against its written rules and Stim.

The row alarms on the rounds written_rules.py's transcription of the
multichart rule (Zhang et al. 1410.8765 lines 338-350) alarms on, and
its calibrated thresholds hold Stim's quiet shots to the target rate.
"""

import numpy
import pytest

import decsim.burst_detectors.settings as burst_detector_settings
import decsim.engine as engine_module
import decsim.frontends.settings as workload_settings
import tests.burst_detectors.burst_rounds as burst_rounds
import tests.burst_detectors.written_rules as written_rules


def test_the_row_fires_on_the_rounds_the_written_rules_alarm():
    """Alarm, at the row's own thresholds, through observe_round."""
    settings = burst_rounds.CUSUM.Settings(calibration_shots=2000)
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
        "datapaths": 2,
    }

    detector_section = burst_detector_settings.BurstDetectorSettings.from_yaml(
        section, burst_rounds.CLOCKS
    )

    settings = detector_section.row_settings
    assert settings.mask_count is None
    assert settings.region_radii == ()
    assert settings.fault_rate_multipliers == (3.0,)
    assert settings.datapaths == 2
    assert settings.clock == burst_rounds.CLOCKS.clock("fridge")


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
