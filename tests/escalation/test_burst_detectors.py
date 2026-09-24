"""The event_count burst detector against Stim's own samples.

The referent for a threshold is the tail Stim's detector sampler gives
on the same circuit: Stim's surface_code:rotated_memory_z at p = 1e-3,
the circuit decsim's memory_circuit generates. The counters are Q3DE's
(Suzuki et al. 2501.00331 lines 713-727) and Tan et al.'s one-round
Hamming weight widened to W rounds (2406.18897 lines 956-966); the
priors change is IonQ's (2608.25027 lines 334-340).
"""

import numpy
import pytest

import decsim.config as config
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.engine as engine_module
import decsim.escalation.burst_detectors as burst_detectors
import decsim.escalation.settings as escalation_settings
import decsim.frontends.settings as workload_settings
import decsim.qpu.stim_device as stim_device
import decsim.records.windows as window_records

ROUNDS = 30
AFTER_LAST_ROUND = ROUNDS + 1
DISTANCE = 5
PHYSICAL_ERROR_PROBABILITY = 1e-3
CODE_TASK = "surface_code:rotated_memory_z"
# d = 5 rotated_memory_z: round one forms the 12 Z detectors against the
# prepared state, every later round 24 bulk detectors
FIRST_ROUND_QUIET = (0,) * 12
BULK_ROUND_QUIET = (0,) * 24
BULK_ROUND_LOUD = (1,) * 24
CLOCKS = config.ClockSettings({"fridge": 250.0})


def _circuit():
    return workload_settings.memory_circuit(
        CODE_TASK, ROUNDS, DISTANCE, PHYSICAL_ERROR_PROBABILITY
    )


def _detector(settings, engine=None):
    if engine is None:
        engine = engine_module.Engine()
    circuit = _circuit()
    circuits = {1: (circuit, ROUNDS)}
    return burst_detectors.EventCountBurstDetector(settings, engine, circuits)


def _table():
    circuit = _circuit()
    return detector_formation.build_formation_table(circuit, ROUNDS)


def _rounds_of_events(sampled_events):
    """One shot's detection events cut into rounds, in formation order."""
    table = _table()
    rounds = []
    for round_index in range(1, AFTER_LAST_ROUND):
        recipes = table.detectors_of_round(round_index)
        values = []
        for recipe in recipes:
            value = sampled_events[recipe.detector_index]
            values.append(value)
        rounds.append(values)
    return rounds


def _sampled_rounds(circuit, seed):
    sampler = circuit.compile_detector_sampler(seed=seed)
    shots = sampler.sample(1)
    return _rounds_of_events(shots[0])


def _whole_patch_burst(onset_round, probability):
    circuit = _circuit()
    table = _table()
    burst = stim_device.BurstStimDevice.Settings(
        burst_onset_round=onset_round,
        burst_error_probability=probability,
    )
    return stim_device.burst_circuit(circuit, table, burst)


def _feed(detector, rounds):
    for index, events in enumerate(rounds):
        round_index = index + 1
        detector.observe_round(1, round_index, events)


def _sampled_bulk_round_counts(shot_count, seed):
    """Stim's detection events per bulk round, summed over the patch."""
    circuit = _circuit()
    sampler = circuit.compile_detector_sampler(seed=seed)
    samples = sampler.sample(shot_count)
    table = _table()
    bulk = detector_formation.LayerKind.BULK
    shape = (shot_count, ROUNDS)
    per_round = numpy.zeros(shape, dtype=numpy.int64)
    for recipe in table.detectors:
        if recipe.kind is bulk:
            column = recipe.round_index - 1
            per_round[:, column] += samples[:, recipe.detector_index]
    return per_round[:, 1:]


def _quiet_rounds(round_count):
    bulk_round_count = round_count - 1
    bulk_rounds = [BULK_ROUND_QUIET] * bulk_round_count
    return [FIRST_ROUND_QUIET, *bulk_rounds]


def _window(first_round, last_round):
    round_count = last_round - first_round + 1
    return window_records.Window(
        operation_id=1,
        window_index=0,
        commit_lo=first_round,
        commit_hi=last_round,
        buffer_hi=last_round,
        round_count=round_count,
    )


def test_the_patch_count_tail_is_the_tail_stim_samples():
    """W = 1 at d = 5: the law's tail at its 1e-3 threshold, beside Stim's.

    Stim samples 20,000 shots, 29 bulk rounds each; the law's tail at
    the threshold must sit within five standard errors of the sampled
    frequency of the same count.
    """
    settings = burst_detectors.EventCountBurstDetector.Settings(
        patch_window_rounds=1, false_alarms_per_round=2e-3
    )
    detector = _detector(settings)
    counts = detector.counts_by_operation[1]
    law = counts.calibration.patch_law
    threshold = law.threshold(1e-3, 1.0)
    tails = law.tails(1.0)
    predicted = tails[threshold]
    bulk_rounds = _sampled_bulk_round_counts(20_000, seed=5)
    is_at_threshold = bulk_rounds >= threshold
    sampled = numpy.mean(is_at_threshold)
    variance = predicted / bulk_rounds.size
    standard_error = numpy.sqrt(variance)
    difference = predicted - sampled
    assert threshold == 6
    assert abs(difference) < 5 * standard_error


def test_a_whole_patch_burst_is_flagged_within_three_rounds():
    settings = burst_detectors.EventCountBurstDetector.Settings()
    detector = _detector(settings)
    burst = _whole_patch_burst(onset_round=12, probability=0.1)
    rounds = _sampled_rounds(burst, seed=3)
    _feed(detector, rounds[:14])
    window = _window(12, 14)
    assert detector.is_burst_window(window)


def test_a_burst_free_shot_is_not_flagged():
    settings = burst_detectors.EventCountBurstDetector.Settings()
    detector = _detector(settings)
    circuit = _circuit()
    rounds = _sampled_rounds(circuit, seed=3)
    _feed(detector, rounds)
    window = _window(1, ROUNDS)
    assert not detector.is_burst_window(window)


def test_a_flag_reaches_back_one_window_before_the_round_that_fired():
    """Q3DE 2501.00331 lines 727-728: the onset is read off the window."""
    settings = burst_detectors.EventCountBurstDetector.Settings(
        patch_window_rounds=4, detector_window_rounds=20
    )
    detector = _detector(settings)
    rounds = _quiet_rounds(11)
    _feed(detector, rounds)
    detector.observe_round(1, 12, BULK_ROUND_LOUD)
    first_flagged = _window(9, 9)
    before_it = _window(8, 8)
    assert detector.is_burst_window(first_flagged)
    assert not detector.is_burst_window(before_it)


def test_a_flag_is_not_seen_before_the_detector_publishes_it():
    """Five cycles of 1000 ticks a round, the rounds served in order."""
    clock = config.Clock(period_ticks=1000)
    settings = burst_detectors.EventCountBurstDetector.Settings(
        clock=clock, cycles_per_round=5
    )
    engine = engine_module.Engine()
    detector = _detector(settings, engine)
    detector.observe_round(1, 1, FIRST_ROUND_QUIET)
    detector.observe_round(1, 2, BULK_ROUND_LOUD)
    window = _window(2, 2)
    engine.now = 9_999
    assert not detector.is_burst_window(window)
    engine.now = 10_000
    assert detector.is_burst_window(window)


def test_the_row_none_takes_no_keys():
    section = {"kind": "none", "patch_window_rounds": 4}
    with pytest.raises(ValueError, match="burst_detector does not know"):
        escalation_settings.BurstDetectorSettings.from_yaml(section, CLOCKS)


def test_a_priced_count_needs_a_clock():
    section = {"kind": "event_count", "cycles_per_round": 3}
    with pytest.raises(ValueError, match="cycles_per_round needs a clock"):
        escalation_settings.BurstDetectorSettings.from_yaml(section, CLOCKS)


def test_a_false_alarm_rate_is_a_probability():
    section = {"kind": "event_count", "false_alarms_per_round": "1e-6"}
    with pytest.raises(ValueError, match="must be a probability"):
        escalation_settings.BurstDetectorSettings.from_yaml(section, CLOCKS)


def test_a_window_is_a_whole_number_of_rounds():
    section = {"kind": "event_count", "patch_window_rounds": 0}
    with pytest.raises(ValueError, match="whole number of rounds"):
        escalation_settings.BurstDetectorSettings.from_yaml(section, CLOCKS)


def test_raising_priors_is_true_or_false():
    section = {"kind": "event_count", "raise_strong_priors": "yes"}
    with pytest.raises(ValueError, match="must be true or false"):
        escalation_settings.BurstDetectorSettings.from_yaml(section, CLOCKS)


def test_burst_mode_ends_when_the_counts_return_to_their_usual_rate():
    """One loud round fires the patch count for W = 4 rounds, no longer."""
    settings = burst_detectors.EventCountBurstDetector.Settings(
        patch_window_rounds=4, detector_window_rounds=20
    )
    detector = _detector(settings)
    quiet_before = _quiet_rounds(11)
    quiet_after = [BULK_ROUND_QUIET] * 8
    rounds = [*quiet_before, BULK_ROUND_LOUD, *quiet_after]
    _feed(detector, rounds)
    last_firing = _window(15, 15)
    after_it = _window(16, 20)
    assert detector.is_burst_window(last_firing)
    assert not detector.is_burst_window(after_it)
