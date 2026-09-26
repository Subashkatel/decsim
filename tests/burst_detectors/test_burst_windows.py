"""A window a published flag does not meet keeps its model.

The switching policy escalates every window a published flag meets
(IonQ 2608.25027 lines 334-340 raise only such a window's priors).
"""

import tests.burst_detectors.burst_rounds as burst_rounds


def test_a_window_before_the_flag_keeps_its_model():
    detector = burst_rounds.flagged_event_count_detector(
        raise_strong_priors=True
    )
    window = burst_rounds.window(1, 6)
    model = burst_rounds.window_model(1, 6)
    kept = detector.with_burst_priors(window, model)
    assert kept is model


def test_a_window_before_the_cusum_flag_keeps_its_model():
    detector = burst_rounds.flagged_cusum_detector(raise_strong_priors=True)
    model = burst_rounds.window_model(1, 6)
    window = burst_rounds.window(1, 6)

    kept = detector.with_burst_priors(window, model)

    assert kept is model
