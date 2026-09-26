"""The chart bank's thresholds, read off quiet shots' block maxima.

Every group takes the level its block maxima pass at one shared tail
share, bisected until the share of shots on which any group alarms
meets the target.
"""

import dataclasses
import math
from typing import Any

import numpy

# The calibration rule's constants: the exponential tail is fitted to
# the largest TAIL_FIT_COUNT block maxima, a bank target is measured
# when the blocks hold MEASURED_ALARMS expected alarms, the shared level
# is bisected this many times, and a fitted tail is never flatter than
# SMALLEST_TAIL_SCALE.
TAIL_FIT_COUNT = 20
MEASURED_ALARMS = 5
SHARE_BISECTION_STEPS = 30
SMALLEST_TAIL_SCALE = 1e-9


def bank_thresholds(maxima, target_share: float):
    """One level per group, sharing one tail share, so the bank meets target.

    maxima is (blocks, groups). The shared share is bisected so the
    share of blocks where any group reaches its level is at most the
    target. A target the blocks hold fewer than MEASURED_ALARMS alarms
    at keeps the bank-to-group ratio found where they hold that many.
    """
    block_count, group_count = maxima.shape
    tails = []
    for group in range(group_count):
        tail = _group_tail(maxima[:, group])
        tails.append(tail)
    expected_alarms = target_share * block_count
    if expected_alarms >= MEASURED_ALARMS:
        group_share = _shared_share(maxima, tails, target_share)
        return _levels(tails, group_share)
    measured_share = MEASURED_ALARMS / block_count
    group_share = _shared_share(maxima, tails, measured_share)
    bank_ratio = measured_share / group_share
    extrapolated_share = target_share / bank_ratio
    return _levels(tails, extrapolated_share)


@dataclasses.dataclass(frozen=True)
class _GroupTail:
    """One group's block maxima: their observed tail and a fitted one.

    level(share) is the smallest observed maximum whose tail is at most
    share while the blocks reach that share (share x blocks at least the
    fitted count); past it, the exponential fitted to the largest
    maxima's excesses over the next one, u + beta ln(k / (n share)).
    """

    values: Any
    tails: Any
    block_count: int
    fitted_count: int
    fit_base: float
    fit_scale: float

    def level(self, share: float) -> float:
        expected_blocks = share * self.block_count
        if expected_blocks >= self.fitted_count:
            return self._observed_level(share)
        reach = self.fitted_count / expected_blocks
        return self.fit_base + self.fit_scale * math.log(reach)

    def _observed_level(self, share: float) -> float:
        negated_tails = -self.tails
        negated_share = -share
        index = numpy.searchsorted(negated_tails, negated_share, side="left")
        if index < len(self.values):
            return float(self.values[index])
        past_largest = self.values[-1] + 1.0
        return float(past_largest)


def _group_tail(maxima) -> _GroupTail:
    values, counts = numpy.unique(maxima, return_counts=True)
    block_count = len(maxima)
    reversed_counts = counts[::-1]
    reversed_tails = numpy.cumsum(reversed_counts)
    tails = reversed_tails[::-1] / block_count
    ascending = numpy.sort(maxima)
    descending = ascending[::-1]
    largest_fitted_count = block_count - 1
    fitted_count = min(TAIL_FIT_COUNT, largest_fitted_count)
    fit_base = descending[fitted_count]
    excesses = descending[:fitted_count] - fit_base
    mean_excess = numpy.mean(excesses)
    fit_scale = max(float(mean_excess), SMALLEST_TAIL_SCALE)
    return _GroupTail(
        values, tails, block_count, fitted_count, fit_base, fit_scale
    )


def _shared_share(maxima, tails: list, bank_share: float) -> float:
    """The largest group share whose bank alarms at most bank_share."""
    group_count = len(tails)
    low = bank_share / group_count
    high = bank_share
    while _bank_alarm_share(maxima, tails, low) > bank_share:
        low /= 2
    for _ in range(SHARE_BISECTION_STEPS):
        product = low * high
        middle = math.sqrt(product)
        if _bank_alarm_share(maxima, tails, middle) <= bank_share:
            low = middle
        else:
            high = middle
    return low


def _bank_alarm_share(maxima, tails: list, group_share: float) -> float:
    levels = _levels(tails, group_share)
    is_reached = maxima >= levels
    is_alarmed = numpy.any(is_reached, axis=1)
    alarmed_share = numpy.mean(is_alarmed)
    return float(alarmed_share)


def _levels(tails: list, group_share: float):
    levels = []
    for tail in tails:
        level = tail.level(group_share)
        levels.append(level)
    return numpy.asarray(levels)
