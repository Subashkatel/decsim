"""Estimates and exact intervals of logical failure rates.

Pure functions over failure counts. A point stops by a truncated inverse
binomial rule: at the shot where its scored failures reach the target,
no earlier than a minimum shot count, or at a cap. The intervals are
exact for that rule.
"""

import dataclasses
import enum
import math
from typing import Optional

import scipy.stats

# The 95 percent intervals put 2.5 percent on each side (NIST/SEMATECH
# e-Handbook 7.2.4.1).
LOWER_QUANTILE = 0.025
UPPER_QUANTILE = 0.975


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
    """An estimate of one failure probability.

    low and high bound its exact 95 percent interval. A field is None
    where the stop gives no value: a cap with no failure has an upper
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
    probability e, independently, so 1 - 2P = (1 - 2e)^R, and
    e = -expm1(log1p(-2P) / R) / 2, the form that keeps a small rate's
    digits where sinter's 1 - (1 - 2P)^(1/R) rounds to zero and falls
    back to P / R. Above one half the rate goes through its complement
    (:468-469), a convention that keeps the map increasing, so an
    interval's limits map to the per-round limits; for an even R no
    round-flip probability gives a shot rate above one half, so a row
    there is flagged by its reader.
    """
    if shot_rate == 0.5:
        # log1p(-1) is minus infinity: every round is a fair coin.
        return 0.5
    if shot_rate > 0.5:
        complement = 1 - shot_rate
        complement_per_round = per_round_rate(complement, rounds)
        return 1 - complement_per_round
    doubled = 2 * shot_rate
    shot_log_survival = math.log1p(-doubled)
    round_log_survival = shot_log_survival / rounds
    negative_round_flip = math.expm1(round_log_survival)
    return -negative_round_flip / 2


def per_output_round_rate(shot_rate: float, outputs: int, rounds: int) -> float:
    """One output's per-round rate, when a shot fails if any output does.

    sinter 1.16.0 shot_error_rate_to_piece_error_rate with values
    (_probability_util.py:463-465), before its last step, which joins
    the outputs again: the outputs are independent or-pieces, each the
    xor of its rounds, so one output survives a shot with probability
    (1 - P)^(1 / outputs), and its rate is converted over its own rounds
    (per_round_rate). Several memory patches are such
    outputs, each drawing its own shot; one entangled circuit is one
    output, its rounds across both codes (Tesseract 2503.10988 eq. 8).
    """
    # log1p(-1) is minus infinity: at a rate of one every output failed,
    # and one output's rate is the shot's
    if outputs == 1 or shot_rate == 1:
        return per_round_rate(shot_rate, rounds)
    output_log_survival = math.log1p(-shot_rate) / outputs
    output_rate = -math.expm1(output_log_survival)
    return per_round_rate(output_rate, rounds)


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
