"""Estimates, exact intervals and paired tests of logical failure rates.

Pure functions over failure counts and failure bits. A point stops by a
truncated inverse binomial rule: at the shot where its scored failures
reach the target, no earlier than a minimum shot count, or at a cap. The
intervals are exact for that rule, and the confidence sequences hold at
every shot, so at any stop (Howard et al., arXiv 1810.08240, Lemma 3).
"""

import dataclasses
import enum
import math
from collections.abc import Sequence
from typing import Optional

import numpy
import scipy.special
import scipy.stats

# The 95 percent intervals put 2.5 percent on each side (NIST/SEMATECH
# e-Handbook 7.2.4.1).
LOWER_QUANTILE = 0.025
UPPER_QUANTILE = 0.975
# Ville's inequality: a nonnegative martingale of mean one reaches 1/alpha
# with chance at most alpha, at any seed (Howard et al. Lemma 3).
MIXTURE_EVIDENCE_LEVEL = 20.0
LOG_MIXTURE_EVIDENCE_LEVEL = math.log(MIXTURE_EVIDENCE_LEVEL)
LOG_HALF = math.log(0.5)


class StopKind(enum.Enum):
    """Why a point's contiguous prefix of shots stopped.

    TARGET: the scored failures reached the target after the minimum shot
    count. The target's failure on the cap's own shot is a TARGET stop,
    since the rule stops on that failure. MINIMUM: the target was reached
    by the minimum shot count, which stopped the point. CAP: a shot or
    time cap stopped the point before its target, so it is incomplete; a
    time cap is exact only when run time does not depend on failure.
    """

    TARGET = "target"
    MINIMUM = "minimum"
    CAP = "cap"


@dataclasses.dataclass(frozen=True)
class Estimate:
    """A failure probability and its exact 95 percent interval.

    None where the stop gives no value: a cap with no failure has an upper
    limit alone, and a point with no scored shot has nothing.
    """

    rate: Optional[float]
    low: Optional[float]
    high: Optional[float]


def estimate(failures: int, scored_shots: int, stop_kind: StopKind) -> Estimate:
    """Failures over scored shots, with the limits exact for the stop.

    A minimum or cap stop fixes the shot count, and its limits are
    Clopper and Pearson's (Biometrika 26, 1934). A target stop fixes the
    failure count r and leaves the shot count N random; Jennison and
    Turnbull (Technometrics 25, 1983) order the outcomes by N, and
    P(N <= n) = P(Binomial(n, p) >= r) makes their limits the beta
    quantiles B(0.025; r, n - r + 1) and B(0.975; r, n - r). Ordering
    minimum stops above target stops above cap stops keeps each row's
    limits exact for the whole truncated rule.
    """
    if scored_shots == 0:
        return Estimate(None, None, None)
    if failures == 0:
        # Only a cap stops with no failure; the upper limit is NIST
        # 7.2.4.1's 1 - 0.025^(1/n), which is B(0.975; 1, n).
        high = _upper_limit(failures, scored_shots, stop_kind)
        return Estimate(None, None, high)
    rate = failures / scored_shots
    lower_second_shape = scored_shots - failures + 1
    low = scipy.stats.beta.ppf(LOWER_QUANTILE, failures, lower_second_shape)
    high = _upper_limit(failures, scored_shots, stop_kind)
    return Estimate(rate, float(low), high)


def plan_unbiased_estimate(
    failures: int, scored_shots: int, stop_kind: StopKind
) -> Optional[float]:
    """The estimate that is unbiased over every outcome of the plan.

    Girshick, Mosteller and Savage (Ann. Math. Statist. 17, 1946),
    Theorem 3: for a closed plan, the paths from (one shot, one failure)
    to the stopping point over the paths from the origin to it. At a
    target stop the last shot is the r-th failure, which gives
    (r - 1)/(N - 1); at a minimum or cap stop every path to (n, x)
    counts, which gives x/n. It is not unbiased among the outcomes that
    reached the target: when a cap can stop the point first, it is
    biased given that the target was reached.
    """
    if scored_shots == 0:
        return None
    if failures == scored_shots:
        # One path of failures only, from either start.
        return 1.0
    if stop_kind is not StopKind.TARGET:
        return failures / scored_shots
    failures_before_the_last = failures - 1
    shots_before_the_last = scored_shots - 1
    return failures_before_the_last / shots_before_the_last


def estimate_counting_unscored(
    failures: int, scored_shots: int, unscored_shots: int
) -> Optional[float]:
    """The failure fraction with every unscored shot counted as a failure.

    An unscored shot got no correction from some window's decoder, so at
    worst it failed; this bounds the rate when unscored shots are not few
    beside the failures.
    """
    shots = scored_shots + unscored_shots
    if shots == 0:
        return None
    counted_failures = failures + unscored_shots
    return counted_failures / shots


def per_round_rate(shot_rate: float, rounds: int) -> float:
    """The per-round failure probability that gives shot_rate over rounds.

    sinter 1.16.0 shot_error_rate_to_piece_error_rate
    (_probability_util.py:465-475): each round flips the observable with
    probability e, independently, so 1 - 2P = (1 - 2e)^R. Above one half
    the rate goes through its complement (:468-469), which keeps the map
    increasing, so an interval's limits map to the per-round limits. A
    shot rate above one half needs a per-round rate above one half, so a
    row there is flagged by its reader.
    """
    if shot_rate > 0.5:
        complement = 1 - shot_rate
        complement_per_round = per_round_rate(complement, rounds)
        return 1 - complement_per_round
    doubled = 2 * shot_rate
    shot_survival = 1 - doubled
    exponent = 1 / rounds
    round_survival = shot_survival**exponent
    round_flip = 1 - round_survival
    return round_flip / 2


def mcnemar_p_value(
    first_only_failures: int, second_only_failures: int
) -> float:
    """The exact two-sided McNemar p-value on a paired count fixed in advance.

    McNemar (Psychometrika 12, 1947): with no difference, each discordant
    seed (one point failed, the other did not) is a fair coin, so the
    smaller count is Binomial(discordant, 1/2); the two-sided value
    doubles its tail, at most one. Valid only when the paired count was
    fixed before the run; any other stop takes the mixture sequence.
    """
    discordant = first_only_failures + second_only_failures
    smaller = min(first_only_failures, second_only_failures)
    tail = scipy.stats.binom.cdf(smaller, discordant, 0.5)
    doubled_tail = 2 * tail
    return float(min(doubled_tail, 1.0))


def is_mixture_difference(
    first_only_failures: int, second_only_failures: int
) -> bool:
    """Whether the discordant seeds show a difference, at any stop.

    The beta-binomial mixture of Robbins (1970), Howard et al. Proposition
    7 with g = h = 1/2 and a uniform prior on theta, the chance that the
    first point is the one failing on a discordant seed:
    M(1/2) = B(a + 1, b + 1) / (B(1, 1) (1/2)^(a + b)), where B(1, 1) is
    one. Under the null, each discordant sign a fair coin given every
    earlier seed, M is a nonnegative martingale of mean one, so it
    reaches 20 at any seed with chance at most 0.05 (Lemma 3).
    """
    discordant = first_only_failures + second_only_failures
    first_shape = first_only_failures + 1
    second_shape = second_only_failures + 1
    log_mixture = scipy.special.betaln(first_shape, second_shape)
    log_fair = discordant * LOG_HALF
    log_evidence = log_mixture - log_fair
    is_difference = log_evidence >= LOG_MIXTURE_EVIDENCE_LEVEL
    return bool(is_difference)


def empirical_bernstein_sequence(values: Sequence[float]) -> tuple:
    """The 95 percent confidence sequence for the mean of values in [0, 1].

    Howard et al. (arXiv 1810.08240) eq. (24), Theorem 4 with the
    polynomial stitched boundary (c = 1, eta = 2, m = 1, h(k) ~ k^1.4):
    mean_t +- [1.7 sqrt(V (log log 2V + 3.8)) + 3.4 log log 2V + 13] / t,
    with V = max(1, sum of (X_i - mean_(i-1))^2). It covers mu_t, the
    average of each value's expectation given every earlier value
    (section 2), with no common mean assumed, at every t and so at any
    stop. On a learning run's failure bits it bounds the average
    conditional failure probability. Returns (low, high).
    """
    outcomes = numpy.asarray(values, dtype=float)
    count = len(outcomes)
    indexes = numpy.arange(count)
    positions = indexes + 1
    running_sums = numpy.cumsum(outcomes)
    running_means = running_sums / positions
    # Theorem 4 takes any prediction in [0, 1] before the first value.
    predictions = numpy.zeros(count)
    predictions[1:] = running_means[:-1]
    deviations = outcomes - predictions
    squared_deviations = deviations * deviations
    deviation_sum = numpy.sum(squared_deviations)
    variance = max(deviation_sum, 1.0)
    doubled_variance = 2 * variance
    log_doubled = math.log(doubled_variance)
    iterated_log = math.log(log_doubled)
    root_argument = variance * (iterated_log + 3.8)
    root = math.sqrt(root_argument)
    numerator = 1.7 * root + 3.4 * iterated_log + 13
    radius = numerator / count
    mean = running_means[-1]
    low = mean - radius
    high = mean + radius
    return (float(low), float(high))


def difference_sequence(
    first_failures: Sequence[bool], second_failures: Sequence[bool]
) -> tuple:
    """The 95 percent sequence for the average difference of two rates.

    Howard et al. Theorem 4 on X_s = fail_first - fail_second in [-1, 1],
    as eq. (24) on (X + 1) / 2 in [0, 1]. It covers the average over the
    seeds of p_first(s | past) - p_second(s | past), the seeds being
    those both points scored, with no common mean assumed. Returns
    (low, high).
    """
    first = numpy.asarray(first_failures, dtype=float)
    second = numpy.asarray(second_failures, dtype=float)
    differences = first - second
    shifted = differences + 1
    rescaled = shifted / 2
    rescaled_low, rescaled_high = empirical_bernstein_sequence(rescaled)
    doubled_low = 2 * rescaled_low
    doubled_high = 2 * rescaled_high
    low = doubled_low - 1
    high = doubled_high - 1
    return (low, high)


def _upper_limit(
    failures: int, scored_shots: int, stop_kind: StopKind
) -> float:
    if failures == scored_shots:
        return 1.0
    successes = scored_shots - failures
    first_shape = failures + 1
    if stop_kind is StopKind.TARGET:
        first_shape = failures
    high = scipy.stats.beta.ppf(UPPER_QUANTILE, first_shape, successes)
    return float(high)
