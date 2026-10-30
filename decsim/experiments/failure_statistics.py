"""Estimates and exact intervals of logical failure rates.

Pure functions over failure counts. A task stops by a truncated inverse
binomial rule: at the shot where its scored failures reach the target,
no earlier than a minimum shot count, or at a cap. The intervals are
exact for that rule. The per-round rate over runs of several lengths
is a fit, with a normal interval (per_round_rate_over_lengths), or over
many streams' checkpoints, with a bootstrap interval
(per_round_rate_over_checkpoints). Two runs of the same shots are
compared by McNemar's exact test (exact_mcnemar).
"""

import dataclasses
import enum
import functools
import math
from collections.abc import Sequence
from typing import Optional

import numpy
import scipy.stats

# The 95 percent intervals put 2.5 percent on each side (NIST/SEMATECH
# e-Handbook 7.2.4.1).
LOWER_QUANTILE = 0.025
UPPER_QUANTILE = 0.975
# The normal quantile of a two-sided 95 percent interval, about 1.96.
NORMAL_UPPER_QUANTILE = scipy.stats.norm.ppf(UPPER_QUANTILE)


class StopKind(enum.Enum):
    """Why a task's contiguous prefix of shots stopped.

    TARGET: the scored failures reached the target after the minimum shot
    count. The target's failure on the cap's own shot is a TARGET stop,
    since the rule stops on that failure. MINIMUM: the target was reached
    by the minimum shot count, which stopped the task. CAP: a shot or
    time cap stopped the task before its target, so it is incomplete; a
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
    limit alone, and a task with no scored shot has nothing.
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
    reached the target: when a cap can stop the task first, it is
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


def latency_effective_rate(
    cycle_rate: float, mean_latency_rounds: float, distance: int
) -> float:
    """A logical cycle's failure rate, the wait for its decode counted.

    Micro Blossom's effective logical error rate (Wu et al. 2502.14787
    Sec. 8.3, p.12): a feed-forward waits L rounds for its decoded
    outcome, and the waiting qubit fails as over L / d more cycles, so a
    cycle of rate p_L becomes p_L (1 + L / d); over a spread of
    latencies only the mean counts. cycle_rate is the failure rate of d
    rounds, one logical cycle, and mean_latency_rounds the mean reaction
    time over the round period. The form is the paper's first order.
    """
    if mean_latency_rounds < 0:
        raise ValueError(
            f"a mean latency is not negative: {mean_latency_rounds} rounds"
        )
    waited_cycles = mean_latency_rounds / distance
    return cycle_rate * (1 + waited_cycles)


@dataclasses.dataclass(frozen=True)
class DecayFit:
    """The per-round failure rate that runs of several lengths give.

    Each round flips the outcome with probability per_round_rate (e),
    and preparation and the final readout flip it once more with
    probability boundary_rate (q), so a run of n rounds fails with a
    probability P(n) where 1 - 2 P(n) = (1 - 2 q)(1 - 2 e)^n (Tan et
    al. 2209.09219 eq. S4). low and high bound e's 95 percent interval.
    """

    per_round_rate: float
    low: float
    high: float
    boundary_rate: float


def per_round_rate_over_lengths(
    rounds: Sequence[int],
    failures: Sequence[int],
    shots: Sequence[int],
    lengths_share_runs: bool = False,
) -> DecayFit:
    """The fit of 1 - 2 P(n) = (1 - 2 q)(1 - 2 e)^n over the lengths.

    Tan et al. eq. S5-S8: a weighted least squares line through
    y = log(1 - 2 P) against n, each length weighted by one over
    Var(y) = (2 / (2 P - 1))^2 P (1 - P) / N. Its slope is log(1 - 2 e)
    and its intercept log(1 - 2 q). Runs drawn apart for each length
    give the slope the usual variance, one over the sum of w x^2, with
    x the length less the weighted mean length. Checkpoints of one run
    share their rounds, so their rates are correlated, and
    lengths_share_runs takes Tan's conservative variance (eq. S9). The
    interval is the slope plus or minus 1.96 standard deviations, as
    in Tan's Fig. S8, mapped to e.
    """
    _check_lengths(rounds, failures, shots)
    rates = numpy.divide(failures, shots)
    doubled = 2 * rates
    logs = numpy.log1p(-doubled)
    weights = _decay_weights(rates, shots)
    line = _fitted_line(rounds, logs, weights, lengths_share_runs)
    slope, intercept, deviation = line
    margin = NORMAL_UPPER_QUANTILE * deviation
    steeper = slope - margin
    shallower = slope + margin
    # e falls as the slope rises, so the shallower slope gives e's low
    # end; a rate is not below zero, where data too few to tell e from
    # zero would put it
    per_round = _flip_rate_of(slope)
    low = _flip_rate_of(shallower)
    high = _flip_rate_of(steeper)
    boundary = _flip_rate_of(intercept)
    floored_low = max(low, 0.0)
    return DecayFit(per_round, floored_low, high, boundary)


def per_round_rate_over_checkpoints(
    rounds: Sequence[int],
    failed_by_stream: Sequence[Sequence[bool]],
    resamples: int = 1000,
    seed: int = 0,
) -> DecayFit:
    """The decay fit over many streams' checkpoints, its interval by bootstrap.

    failed_by_stream holds one row a stream, one entry a checkpoint at
    the matching rounds (observe/checkpoints.py). A stream's checkpoints
    share its rounds, so their failures are correlated, and the streams
    are independent: the fit is per_round_rate_over_lengths over the
    failures summed across streams, and its interval is the 2.5 and
    97.5 percentiles of the same fit over streams drawn again with
    replacement, a whole stream at a time (scipy.stats.bootstrap,
    paired, the percentile method).
    """
    failed = numpy.asarray(failed_by_stream, dtype=int)
    stream_count = failed.shape[0]
    failures = numpy.sum(failed, axis=0)
    shots = [stream_count] * len(rounds)
    fit = per_round_rate_over_lengths(rounds, failures, shots)
    rate_of = functools.partial(_rate_of_columns, rounds)
    columns = tuple(failed.T)
    result = scipy.stats.bootstrap(
        columns,
        rate_of,
        n_resamples=resamples,
        paired=True,
        vectorized=False,
        method="percentile",
        rng=seed,
    )
    interval = result.confidence_interval
    low = float(interval.low)
    high = float(interval.high)
    return DecayFit(fit.per_round_rate, low, high, fit.boundary_rate)


def exact_mcnemar(first_only: int, second_only: int) -> float:
    """McNemar's exact two-sided p of two runs over the same shots.

    first_only and second_only are the shots only one run failed. Under
    the hypothesis that both fail as often, each such shot is a fair
    coin, so the p is the two-sided binomial test of first_only in their
    sum at one half (scipy.stats.binomtest). No such shot gives 1.
    """
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    test = scipy.stats.binomtest(first_only, discordant, 0.5)
    return float(test.pvalue)


def _rate_of_columns(rounds: Sequence[int], *columns: numpy.ndarray) -> float:
    """The fitted rate of one resample, a column a checkpoint."""
    failures = [numpy.sum(column) for column in columns]
    shots = [len(column) for column in columns]
    fit = per_round_rate_over_lengths(rounds, failures, shots)
    return fit.per_round_rate


def _check_lengths(
    rounds: Sequence[int], failures: Sequence[int], shots: Sequence[int]
) -> None:
    if not len(rounds) == len(failures) == len(shots):
        raise ValueError(
            "a decay fit takes one failure count and one shot count for "
            f"each length; got {len(rounds)} lengths, {len(failures)} "
            f"failure counts and {len(shots)} shot counts"
        )
    if len(set(rounds)) < 2:
        raise ValueError(
            "a decay fit needs runs of at least two lengths to draw a "
            f"line; got the lengths {list(rounds)}"
        )
    for length, failed, count in zip(rounds, failures, shots, strict=True):
        _check_rate(length, failed, count)


def _check_rate(length: int, failed: int, count: int) -> None:
    # no failure gives a variance of zero and so an infinite weight; at
    # one half or above, log(1 - 2 P) is not defined
    if failed == 0:
        raise ValueError(
            f"the runs of {length} rounds have no failure in {count} "
            "shots, which gives no variance to weigh them by; run more "
            "shots or leave that length out"
        )
    if 2 * failed >= count:
        raise ValueError(
            f"the runs of {length} rounds fail {failed} times in {count} "
            "shots, at least one half, where log(1 - 2 P) is not "
            "defined; fit shorter lengths"
        )


def _decay_weights(rates: numpy.ndarray, shots: Sequence[int]) -> numpy.ndarray:
    # Tan et al. eq. S7-S8: the binomial variance of P, carried to
    # log(1 - 2 P) by the delta method
    rate_variance = rates * (1 - rates) / numpy.asarray(shots)
    log_derivative = 2 / (2 * rates - 1)
    log_variance = log_derivative**2 * rate_variance
    return 1 / log_variance


def _fitted_line(
    rounds: Sequence[int],
    logs: numpy.ndarray,
    weights: numpy.ndarray,
    lengths_share_runs: bool,
) -> tuple:
    """The line's slope, its intercept and the slope's deviation."""
    lengths = numpy.asarray(rounds, dtype=float)
    mean_length = numpy.average(lengths, weights=weights)
    mean_log = numpy.average(logs, weights=weights)
    offsets = lengths - mean_length
    weighted_offsets = weights * offsets
    spread = numpy.dot(weighted_offsets, offsets)
    slope = numpy.dot(weighted_offsets, logs) / spread
    intercept = mean_log - slope * mean_length
    deviation = _slope_deviation(weights, offsets, spread, lengths_share_runs)
    return float(slope), float(intercept), deviation


def _slope_deviation(
    weights: numpy.ndarray,
    offsets: numpy.ndarray,
    spread: float,
    lengths_share_runs: bool,
) -> float:
    if not lengths_share_runs:
        variance = 1 / spread
        return math.sqrt(variance)
    # Tan et al. eq. S9: the slope's deviation if every length's error
    # moved together, the sum of |x| w sqrt(Var(y)) over the sum of w x^2;
    # w sqrt(Var(y)) is sqrt(w)
    magnitudes = numpy.abs(offsets)
    roots = numpy.sqrt(weights)
    together = numpy.dot(magnitudes, roots)
    deviation = together / spread
    return float(deviation)


def _flip_rate_of(log_survival: float) -> float:
    # 1 - 2 e = exp(log_survival), in the form that keeps a small rate's
    # digits (per_round_rate)
    return -math.expm1(log_survival) / 2


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
