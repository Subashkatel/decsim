"""A flag's onset, its end, and when it is published."""

import decsim.burst_detectors.event_count.detector as event_count
import decsim.config as config
import decsim.engine as engine_module
import tests.burst_detectors.burst_rounds as burst_rounds
from tests.burst_detectors.burst_rounds import (
    BULK_ROUND_LOUD,
    BULK_ROUND_QUIET,
    CUSUM,
    FIRST_ROUND_QUIET,
)


def test_a_flag_reaches_back_one_window_before_the_round_that_fired():
    """Q3DE 2501.00331 lines 727-728: the onset is read off the window."""
    settings = event_count.EventCountBurstDetector.Settings(
        patch_window_rounds=4, detector_window_rounds=20
    )
    detector = burst_rounds.event_count_detector(settings)
    rounds = burst_rounds.quiet_rounds(11)
    burst_rounds.feed(detector, rounds)
    detector.observe_round(1, 12, BULK_ROUND_LOUD)
    first_flagged = burst_rounds.window(9, 9)
    before_it = burst_rounds.window(8, 8)
    assert detector.is_burst_window(first_flagged)
    assert not detector.is_burst_window(before_it)


def test_a_flag_is_not_seen_before_the_detector_publishes_it():
    """Five cycles of 1000 ticks a round, the rounds served in order."""
    clock = config.Clock(period_ticks=1000)
    settings = event_count.EventCountBurstDetector.Settings(
        clock=clock, cycles_per_round=5
    )
    engine = engine_module.Engine()
    detector = burst_rounds.event_count_detector(settings, engine)
    detector.observe_round(1, 1, FIRST_ROUND_QUIET)
    detector.observe_round(1, 2, BULK_ROUND_LOUD)
    window = burst_rounds.window(2, 2)
    engine.now = 9_999
    assert not detector.is_burst_window(window)
    engine.now = 10_000
    assert detector.is_burst_window(window)


def test_burst_mode_ends_when_the_counts_return_to_their_usual_rate():
    """One loud round fires the patch count for W = 4 rounds, no longer."""
    settings = event_count.EventCountBurstDetector.Settings(
        patch_window_rounds=4, detector_window_rounds=20
    )
    detector = burst_rounds.event_count_detector(settings)
    quiet_before = burst_rounds.quiet_rounds(11)
    quiet_after = [BULK_ROUND_QUIET] * 8
    rounds = [*quiet_before, BULK_ROUND_LOUD, *quiet_after]
    burst_rounds.feed(detector, rounds)
    last_firing = burst_rounds.window(15, 15)
    after_it = burst_rounds.window(16, 20)
    assert detector.is_burst_window(last_firing)
    assert not detector.is_burst_window(after_it)


def test_a_cusum_flag_is_published_after_the_banks_cycles():
    """ceil(97 regions x 3 designs / 1 datapath) + 30 = 321 cycles."""
    clock = config.Clock(period_ticks=1000)
    settings = CUSUM.Settings(clock=clock)
    engine = engine_module.Engine()
    detector = burst_rounds.cusum_detector(settings, engine=engine)
    detector.observe_round(1, 1, FIRST_ROUND_QUIET)
    detector.observe_round(1, 2, BULK_ROUND_LOUD)
    window = burst_rounds.window(2, 2)
    engine.now = 320_999
    was_published = detector.is_burst_window(window)
    engine.now = 321_000

    assert not was_published
    assert detector.is_burst_window(window)
