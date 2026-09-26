"""The chart bank's thresholds on hand maxima.

Closed forms on the maxima 1 to 100: the observed tail, and past it the
exponential fitted to the largest maxima, u + beta ln(k / (n share)).
The 21st largest of them is 80, and the 20 largest exceed it by 10.5 on
average.
"""

import numpy
import pytest

import decsim.burst_detectors.masked_regional_cusum.thresholds as thresholds


def test_a_measured_level_is_the_smallest_maximum_at_most_its_share():
    """30 of 100 blocks reach 71, and 30 is at least the 20 fitted."""
    hand_maxima = numpy.arange(1.0, 101.0)
    maxima = hand_maxima[:, None]

    levels = thresholds.bank_thresholds(maxima, 0.3)

    assert list(levels) == [71.0]


def test_one_group_alarms_on_at_most_its_target_share_of_blocks():
    """5 percent of 100 blocks: the level passes 95, so 96 to 100 alarm.

    The fitted tail alone gives 80 + 10.5 ln(20 / 5) = 94.56, which
    admits 95 too, six blocks; one group goes through the bank's
    bisection like any other bank.
    """
    hand_maxima = numpy.arange(1.0, 101.0)
    maxima = hand_maxima[:, None]

    levels = thresholds.bank_thresholds(maxima, 0.05)

    reaches = maxima >= levels
    assert levels[0] == pytest.approx(95.0)
    assert numpy.count_nonzero(reaches) == 5


def test_a_shared_level_keeps_the_bank_at_its_target():
    """A group that never alarms leaves the other's level at the target.

    The bisection stays below the target share, so the level is the
    next maximum up: 72.
    """
    silent = numpy.zeros(100)
    hand_maxima = numpy.arange(1.0, 101.0)
    maxima = numpy.stack([hand_maxima, silent], axis=1)

    levels = thresholds.bank_thresholds(maxima, 0.3)

    assert list(levels) == [72.0, 1.0]


def test_an_unmeasured_target_keeps_the_ratio_found_at_five_alarms():
    """One expected alarm in 100 blocks: the ratio at five gives 111.899.

    Five alarms bisect to a group share, the bank-to-group ratio there
    carries down to one alarm, and the fitted tail gives the level.
    """
    silent = numpy.zeros(100)
    hand_maxima = numpy.arange(1.0, 101.0)
    maxima = numpy.stack([hand_maxima, silent], axis=1)

    levels = thresholds.bank_thresholds(maxima, 0.01)

    assert levels[0] == pytest.approx(111.899098, abs=1e-6)


def test_the_bank_alarms_on_at_most_its_target_share_of_its_blocks():
    """Heavy tails fit levels a share of target / groups still passes.

    The lower end of the shared share halves until the bank meets the
    target, so the calibration blocks never alarm above it.
    """
    generator = numpy.random.default_rng(0)
    maxima = generator.pareto(1.5, (91, 4))
    target = 0.10722369416000488

    levels = thresholds.bank_thresholds(maxima, target)

    reaches = maxima >= levels
    is_alarmed = reaches.any(axis=1)
    assert numpy.mean(is_alarmed) <= target
