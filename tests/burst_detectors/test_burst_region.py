"""A flagged region's priors raised with the graph kept.

IonQ 2608.25027 lines 334-340: only the prior vector changes, and no
Tanner graph is rebuilt.
"""

import numpy

import decsim.burst_detectors.event_count.detector as event_count
import tests.burst_detectors.burst_rounds as burst_rounds


def test_burst_priors_raise_the_priors_and_keep_the_graph():
    """IonQ 2608.25027 lines 334-340: only the prior vector changes.

    Every detector fires every round, a rate no prior below one half
    explains, so every column the flagged rows see sits at the cap.
    """
    detector = burst_rounds.flagged_event_count_detector(
        raise_strong_priors=True
    )
    window = burst_rounds.window(12, 17)
    model = burst_rounds.window_model(12, 17)
    raised = detector.with_burst_priors(window, model)
    faults = model.require_faults(burst_rounds.GRAPHLIKE)
    raised_faults = raised.require_faults(burst_rounds.GRAPHLIKE)
    changed_checks = faults.check != raised_faults.check
    assert changed_checks.nnz == 0
    assert raised.detector_ids == model.detector_ids
    assert raised_faults.source_fault_ids == faults.source_fault_ids
    is_capped = raised_faults.priors == 0.5
    was_below_cap = faults.priors < 0.5
    assert numpy.all(is_capped)
    assert numpy.all(was_below_cap)


def test_a_flag_with_no_anomalous_position_keeps_the_model():
    """One loud round fires the patch count, no position's own count."""
    settings = event_count.EventCountBurstDetector.Settings(
        raise_strong_priors=True
    )
    detector = burst_rounds.event_count_detector(settings)
    quiet_before = burst_rounds.quiet_rounds(11)
    rounds = [*quiet_before, burst_rounds.BULK_ROUND_LOUD]
    burst_rounds.feed(detector, rounds)
    window = burst_rounds.window(12, 12)
    model = burst_rounds.window_model(12, 12)
    kept = detector.with_burst_priors(window, model)
    assert detector.is_burst_window(window)
    assert kept is model


def test_cusum_burst_priors_cap_the_region_and_keep_the_graph():
    detector = burst_rounds.flagged_cusum_detector(raise_strong_priors=True)
    window = burst_rounds.window(12, 17)
    model = burst_rounds.window_model(12, 17)
    raised = detector.with_burst_priors(window, model)
    faults = model.require_faults(burst_rounds.GRAPHLIKE)
    raised_faults = raised.require_faults(burst_rounds.GRAPHLIKE)
    changed_checks = faults.check != raised_faults.check
    is_capped = raised_faults.priors == 0.5

    assert changed_checks.nnz == 0
    assert numpy.all(is_capped)
