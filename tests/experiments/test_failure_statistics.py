"""The failure statistics against scipy, sinter, mpmath and exact sums.

- Limits: scipy.stats.beta and the NIST/SEMATECH e-Handbook 7.2.4.1
  closed form.
- Coverage and the Girshick, Mosteller and Savage estimator's
  unbiasedness: exact enumeration of one small plan, with scipy's nbinom
  and binom as the outcome probabilities.
- Per-round rate: sinter 1.16.0 shot_error_rate_to_piece_error_rate, and
  the root at 60 digits in mpmath 1.3.0, which the test extra pins.
"""

import mpmath
import pytest
import scipy.stats
import sinter

import decsim.experiments.failure_statistics as failure_statistics

StopKind = failure_statistics.StopKind

# The small plan: stop at the fifth failure, no earlier than the minimum,
# at most 600 shots, with p = 0.02. Its exact coverage, by an enumeration
# written apart from this file, with and without a minimum.
PLAN_PROBABILITY = 0.02
PLAN_TARGET = 5
PLAN_CAP = 600
PLAN_COVERAGE_BY_MINIMUM = {0: 0.9504484671, 100: 0.9595339421}
# The enumeration's coverage is printed to ten places.
COVERAGE_ROUNDING = 1e-9
# The float error of an exact sum over a few hundred outcomes.
SUM_ROUNDING = 1e-12
# The digits mpmath carries for the exact root.
REFERENCE_DIGITS = 60


def test_a_target_stop_takes_the_inverse_sampling_limits():
    result = failure_statistics.estimate(100, 100_000, StopKind.TARGET)

    low = scipy.stats.beta.ppf(0.025, 100, 99_901)
    high = scipy.stats.beta.ppf(0.975, 100, 99_900)
    assert result.rate == 100 / 100_000
    assert result.low == pytest.approx(low, rel=1e-12, abs=0)
    assert result.high == pytest.approx(high, rel=1e-12, abs=0)


@pytest.mark.parametrize("stop_kind", [StopKind.MINIMUM, StopKind.CAP])
def test_a_minimum_or_cap_stop_takes_the_clopper_pearson_limits(stop_kind):
    result = failure_statistics.estimate(3, 200_000, stop_kind)

    low = scipy.stats.beta.ppf(0.025, 3, 199_998)
    high = scipy.stats.beta.ppf(0.975, 4, 199_997)
    assert result.rate == 3 / 200_000
    assert result.low == pytest.approx(low, rel=1e-12, abs=0)
    assert result.high == pytest.approx(high, rel=1e-12, abs=0)


@pytest.mark.parametrize("stop_kind", [StopKind.TARGET, StopKind.MINIMUM])
def test_every_shot_failing_has_upper_limit_one(stop_kind):
    result = failure_statistics.estimate(7, 7, stop_kind)

    low = scipy.stats.beta.ppf(0.025, 7, 1)
    assert result.rate == 1.0
    assert result.low == pytest.approx(low, rel=1e-12, abs=0)
    assert result.high == 1.0


def test_a_cap_with_no_failure_gives_the_upper_limit_alone():
    result = failure_statistics.estimate(0, 300_000, StopKind.CAP)

    nist_high = 1 - 0.025 ** (1 / 300_000)
    beta_high = scipy.stats.beta.ppf(0.975, 1, 300_000)
    assert result.rate is None
    assert result.low is None
    assert result.high == pytest.approx(nist_high, rel=1e-9, abs=0)
    assert result.high == pytest.approx(beta_high, rel=1e-12, abs=0)


def test_no_scored_shot_gives_no_estimate():
    result = failure_statistics.estimate(0, 0, StopKind.CAP)
    plan_unbiased = failure_statistics.plan_unbiased_estimate(
        0, 0, StopKind.CAP
    )

    assert result == failure_statistics.Estimate(None, None, None)
    assert plan_unbiased is None


@pytest.mark.parametrize("minimum", [0, 100])
def test_the_interval_covers_as_the_enumeration_says_property(minimum):
    coverage = 0.0
    outcomes = _plan_outcomes(PLAN_PROBABILITY, PLAN_TARGET, minimum, PLAN_CAP)
    for failures, shots, stop_kind, probability in outcomes:
        result = failure_statistics.estimate(failures, shots, stop_kind)
        low = result.low or 0.0
        is_covered = low <= PLAN_PROBABILITY <= result.high
        coverage += probability * is_covered

    expected = PLAN_COVERAGE_BY_MINIMUM[minimum]
    assert coverage == pytest.approx(expected, abs=COVERAGE_ROUNDING)
    assert coverage >= 0.95


@pytest.mark.parametrize(
    ("probability", "target", "minimum", "cap"),
    [(0.02, 5, 0, 600), (0.02, 5, 100, 600), (0.2, 1, 0, 30)],
)
def test_the_plan_unbiased_estimate_is_unbiased_over_the_plan_property(
    probability, target, minimum, cap
):
    expectation = 0.0
    outcomes = _plan_outcomes(probability, target, minimum, cap)
    for failures, shots, stop_kind, weight in outcomes:
        value = failure_statistics.plan_unbiased_estimate(
            failures, shots, stop_kind
        )
        expectation += weight * value

    assert expectation == pytest.approx(probability, rel=SUM_ROUNDING, abs=0)


def test_unscored_shots_count_as_failures():
    counted = failure_statistics.estimate_counting_unscored(3, 97, 3)
    nothing = failure_statistics.estimate_counting_unscored(0, 0, 0)

    assert counted == 6 / 100
    assert nothing is None


@pytest.mark.parametrize(
    ("shot_rate", "rounds"),
    [
        (1e-12, 1),
        (1e-12, 1_000_000),
        (0.05, 100),
        (0.1, 2),
        (0.5, 10),
        (0.6, 10),
        (0.99, 25),
        (1.0, 5),
    ],
)
def test_the_per_round_rate_is_sinters(shot_rate, rounds):
    per_round = failure_statistics.per_round_rate(shot_rate, rounds)

    sinter_rate = sinter.shot_error_rate_to_piece_error_rate(
        shot_rate, pieces=rounds
    )
    assert per_round == pytest.approx(sinter_rate, rel=1e-9, abs=0)


@pytest.mark.parametrize(
    ("shot_rate", "outputs", "rounds"),
    [
        (0.19, 2, 1),
        (0.1, 4, 30),
        (0.53, 8, 50),
        (0.05, 1, 100),
        (1.0, 4, 30),
    ],
)
def test_one_outputs_per_round_rate_is_sinters_before_it_joins_them(
    shot_rate, outputs, rounds
):
    """With values, sinter returns 1 - (1 - e)^values; e is one output's."""
    per_round = failure_statistics.per_output_round_rate(
        shot_rate, outputs, rounds
    )

    joined = sinter.shot_error_rate_to_piece_error_rate(
        shot_rate, pieces=rounds, values=outputs
    )
    one_output = 1 - (1 - joined) ** (1 / outputs)
    assert per_round == pytest.approx(one_output, rel=1e-9, abs=0)


def test_two_one_round_patches_failing_a_tenth_each_are_a_tenth():
    """0.19 is 1 - 0.9 squared: each patch's own round is 0.1."""
    per_round = failure_statistics.per_output_round_rate(0.19, 2, 1)

    assert per_round == pytest.approx(0.1, rel=1e-12)


@pytest.mark.parametrize(
    ("shot_rate", "rounds"),
    [
        (1e-12, 1),
        (1e-12, 1_000_000),
        (1e-9, 100),
        (0.05, 100),
        (0.3, 1_000_000_000),
    ],
)
def test_the_per_round_rate_is_the_exact_root(shot_rate, rounds):
    """(1 - (1 - 2P)^(1/R)) / 2 at 60 digits, in mpmath."""
    per_round = failure_statistics.per_round_rate(shot_rate, rounds)

    with mpmath.workdps(REFERENCE_DIGITS):
        doubled = 2 * mpmath.mpf(shot_rate)
        survival = 1 - doubled
        root = mpmath.root(survival, rounds)
        exact = (1 - root) / 2
    expected = float(exact)
    assert per_round == pytest.approx(expected, rel=1e-12, abs=0)


def _plan_outcomes(probability, target, minimum, cap):
    """Every stop of the plan: (failures, shots, stop kind, probability)."""
    shots_after_the_minimum = minimum + 1
    for failures in range(target, shots_after_the_minimum):
        weight = scipy.stats.binom.pmf(failures, minimum, probability)
        yield failures, minimum, StopKind.MINIMUM, weight
    first_target_shot = max(target, shots_after_the_minimum)
    shots_after_the_cap = cap + 1
    for shots in range(first_target_shot, shots_after_the_cap):
        # nbinom counts the successes before the target-th failure
        successes = shots - target
        weight = scipy.stats.nbinom.pmf(successes, target, probability)
        yield target, shots, StopKind.TARGET, weight
    for failures in range(target):
        weight = scipy.stats.binom.pmf(failures, cap, probability)
        yield failures, cap, StopKind.CAP, weight
