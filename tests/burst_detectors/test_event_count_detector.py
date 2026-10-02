"""The event_count row against Stim's own samples.

The referent for a threshold is the tail Stim's detector sampler gives
on the same circuit. The counters are Q3DE's (Suzuki et al. 2501.00331
lines 713-727) and Tan et al.'s one-round Hamming weight widened to W
rounds (2406.18897 lines 956-966); the priors change is IonQ's
(2608.25027 lines 334-340).
"""

import numpy
import pytest

import decsim.burst_detectors.event_count.detector as event_count
import decsim.burst_detectors.settings as burst_detector_settings
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.engine as engine_module
import decsim.frontends.settings as workload_settings
import tests.burst_detectors.burst_rounds as burst_rounds


def test_the_patch_count_tail_is_the_tail_stim_samples():
    """W = 1 at d = 5: the law's tail at its 1e-3 threshold, beside Stim's.

    Stim samples 20,000 shots, 29 bulk rounds each; the law's tail at
    the threshold must sit within five standard errors of the sampled
    frequency of the same count.
    """
    settings = event_count.EventCountBurstDetector.Settings(
        patch_window_rounds=1, false_alarms_per_round=2e-3
    )
    detector = burst_rounds.event_count_detector(settings)
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
    bound = 5 * standard_error
    assert abs(difference) < bound


def test_a_whole_patch_burst_is_flagged_within_three_rounds():
    settings = event_count.EventCountBurstDetector.Settings()
    detector = burst_rounds.event_count_detector(settings)
    burst = burst_rounds.whole_patch_burst(onset_round=12, probability=0.1)
    rounds = burst_rounds.sampled_rounds(burst, seed=3)
    burst_rounds.feed(detector, rounds[:14])
    window = burst_rounds.window(12, 14)
    assert detector.is_burst_window(window)


def test_a_burst_free_shot_is_not_flagged():
    settings = event_count.EventCountBurstDetector.Settings()
    detector = burst_rounds.event_count_detector(settings)
    circuit = burst_rounds.memory_circuit()
    rounds = burst_rounds.sampled_rounds(circuit, seed=3)
    burst_rounds.feed(detector, rounds)
    window = burst_rounds.window(1, burst_rounds.ROUNDS)
    assert not detector.is_burst_window(window)


def test_an_operation_shorter_than_the_windows_is_refused():
    """The laws are read off a slab two rounds longer than a window."""
    settings = event_count.EventCountBurstDetector.Settings()
    circuit = workload_settings.memory_circuit(
        burst_rounds.CODE_TASK,
        21,
        burst_rounds.DISTANCE,
        burst_rounds.PHYSICAL_ERROR_PROBABILITY,
    )
    engine = engine_module.Engine()
    circuits = {7: (circuit, 21)}

    with pytest.raises(ValueError, match="operation 7 has 21 rounds"):
        event_count.EventCountBurstDetector(settings, engine, circuits, 1.0)


def test_a_priced_count_needs_a_clock():
    section = {"kind": "event_count", "cycles_per_round": 3}
    with pytest.raises(ValueError, match="cycles_per_round needs a clock"):
        burst_detector_settings.detector_from_yaml(section, burst_rounds.CLOCKS)


def test_a_false_alarm_rate_is_a_probability():
    section = {"kind": "event_count", "false_alarms_per_round": "1e-6"}
    with pytest.raises(ValueError, match="must be a probability"):
        burst_detector_settings.detector_from_yaml(section, burst_rounds.CLOCKS)


def test_a_window_is_a_whole_number_of_rounds():
    section = {"kind": "event_count", "patch_window_rounds": 0}
    with pytest.raises(ValueError, match="whole number of rounds"):
        burst_detector_settings.detector_from_yaml(section, burst_rounds.CLOCKS)


def test_raising_priors_is_true_or_false():
    section = {"kind": "event_count", "raise_strong_priors": "yes"}
    with pytest.raises(ValueError, match="must be true or false"):
        burst_detector_settings.detector_from_yaml(section, burst_rounds.CLOCKS)


def test_without_burst_priors_a_flagged_window_keeps_its_model():
    detector = burst_rounds.flagged_event_count_detector(
        raise_strong_priors=False
    )
    window = burst_rounds.window(12, 17)
    model = burst_rounds.window_model(12, 17)
    kept = detector.with_burst_priors(window, model)
    assert kept is model


@pytest.mark.filterwarnings("error")
def test_a_noiseless_background_fires_on_its_first_event():
    """No fault reaches a detector, so one event is rarer than any budget.

    The law of a count no fault reaches is a count that is always zero,
    and a region no fault reaches keeps its priors.
    """
    settings = event_count.EventCountBurstDetector.Settings(
        raise_strong_priors=True
    )
    detector = _noiseless_detector(settings)
    quiet_before = burst_rounds.quiet_rounds(11)
    one_event = (1,) + burst_rounds.BULK_ROUND_QUIET[1:]
    burst_rounds.feed(detector, quiet_before)
    quiet_window = burst_rounds.window(1, 11)
    was_quiet = detector.is_burst_window(quiet_window)
    detector.observe_round(1, 12, one_event)
    window = burst_rounds.window(12, 12)
    model = burst_rounds.window_model(12, 12)
    kept = detector.with_burst_priors(window, model)

    assert not was_quiet
    assert detector.is_burst_window(window)
    assert kept is model


def _sampled_bulk_round_counts(shot_count, seed):
    """Stim's detection events per bulk round, summed over the patch."""
    circuit = burst_rounds.memory_circuit()
    sampler = circuit.compile_detector_sampler(seed=seed)
    samples = sampler.sample(shot_count)
    table = burst_rounds.formation_table()
    bulk = detector_formation.LayerKind.BULK
    shape = (shot_count, burst_rounds.ROUNDS)
    per_round = numpy.zeros(shape, dtype=numpy.int64)
    for recipe in table.detectors:
        if recipe.kind is bulk:
            column = recipe.round_index - 1
            per_round[:, column] += samples[:, recipe.detector_index]
    return per_round[:, 1:]


def _noiseless_detector(settings):
    """The detector calibrated on the d = 5 memory circuit at p = 0."""
    circuit = workload_settings.memory_circuit(
        burst_rounds.CODE_TASK, burst_rounds.ROUNDS, burst_rounds.DISTANCE, 0.0
    )
    circuits = {1: (circuit, burst_rounds.ROUNDS)}
    engine = engine_module.Engine()
    return event_count.EventCountBurstDetector(settings, engine, circuits, 1.0)
