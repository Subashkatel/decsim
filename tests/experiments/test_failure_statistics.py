"""The failure statistics against scipy, sinter, mpmath and exact sums.

- Limits: scipy.stats.beta and the NIST/SEMATECH e-Handbook 7.2.4.1
  closed form.
- Coverage and the Girshick, Mosteller and Savage estimator's
  unbiasedness: exact enumeration of one small plan, with scipy's nbinom
  and binom as the outcome probabilities.
- Per-round rate: sinter 1.16.0 shot_error_rate_to_piece_error_rate, and
  the root at 60 digits in mpmath 1.3.0 (installed with the test extra,
  through qldpc's sympy).
- McNemar: scipy.stats.binomtest.
- Mixture: the ratio of scipy's betabinom and binom probabilities; at a
  billion seeds, the beta function at 60 digits in mpmath; its crossing
  rate under the null, by a seeded simulation.
- Empirical-Bernstein sequence: Howard et al. (arXiv 1810.08240) eq.
  (24) computed at every prefix at once, and its coverage on seeded
  learning runs.
"""

import mpmath
import numpy
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
# Eq. (24)'s boundary, 1.7 sqrt(V (log log 2V + 3.8)) + 3.4 log log 2V
# + 13, worked by hand: at V = 1, log log 2 = -0.366512920581664 and the
# root is sqrt(3.433487079418336); at V = 2, log log 4 = 0.326634259978281
# and the root is sqrt(8.253268519956562).
BOUNDARY_AT_ONE = 14.903900142653546
BOUNDARY_AT_TWO = 18.99440189739641
# The digits mpmath carries for a referent: log-gamma terms near 1e10
# leave some forty digits past the decision.
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


@pytest.mark.parametrize(
    ("first_only", "second_only"),
    [(12, 3), (3, 12), (7, 7), (0, 6), (40, 22), (1, 0)],
)
def test_mcnemar_is_the_exact_binomial_test(first_only, second_only):
    p_value = failure_statistics.mcnemar_p_value(first_only, second_only)

    discordant_count = first_only + second_only
    binomial = scipy.stats.binomtest(first_only, discordant_count, 0.5)
    assert p_value == pytest.approx(binomial.pvalue, rel=1e-12, abs=0)


@pytest.mark.parametrize(
    ("first_only", "second_only"),
    [
        (34, 66),
        (35, 65),
        (30, 10),
        (29, 11),
        (1, 9),
        (17, 3),
        (16, 4),
        (0, 7),
        (8, 0),
    ],
)
def test_the_mixture_is_the_beta_binomial_likelihood_ratio(
    first_only, second_only
):
    is_difference = failure_statistics.is_mixture_difference(
        first_only, second_only
    )

    discordant_count = first_only + second_only
    mixture = scipy.stats.betabinom.pmf(first_only, discordant_count, 1, 1)
    fair = scipy.stats.binom.pmf(first_only, discordant_count, 0.5)
    evidence = mixture / fair
    assert is_difference == (evidence >= 20)


@pytest.mark.parametrize(
    ("first_only", "second_only"),
    [(388_642_986, 388_500_809), (388_642_987, 388_500_808)],
)
def test_the_mixture_decides_at_a_billion_seeds_as_exact_arithmetic_does(
    first_only, second_only
):
    """Counts whose M(1/2) is 20 less 2e-5, and one step past 20.

    log M = log B(a + 1, b + 1) + n log 2, at 60 digits in mpmath.
    """
    is_difference = failure_statistics.is_mixture_difference(
        first_only, second_only
    )

    discordant_count = first_only + second_only
    first_shape = first_only + 1
    second_shape = second_only + 1
    with mpmath.workdps(REFERENCE_DIGITS):
        beta = mpmath.beta(first_shape, second_shape)
        log_beta = mpmath.log(beta)
        log_two = mpmath.log(2)
        log_level = mpmath.log(20)
        log_evidence = log_beta + discordant_count * log_two
        is_exact_difference = log_evidence >= log_level
    assert is_difference == is_exact_difference


def test_the_mixture_rarely_crosses_under_the_null_property():
    """Fair discordant signs, 1,000 runs of 200 seeds, read at every seed.

    Ville's inequality bounds the chance of ever crossing by 0.05.
    """
    generator = numpy.random.default_rng(7)
    draws = generator.random((1000, 200))
    signs = draws < 0.5
    seeds = numpy.arange(1, 201)
    crossed_runs = 0
    for run in signs:
        first_only = numpy.cumsum(run)
        second_only = seeds - first_only
        pairs = zip(first_only, second_only, strict=True)
        crossings = [
            failure_statistics.is_mixture_difference(first, second)
            for first, second in pairs
        ]
        crossed_runs += any(crossings)

    assert crossed_runs / 1000 <= 0.05


def test_the_sequence_is_equation_24():
    generator = numpy.random.default_rng(3)
    failures = generator.random(5000) < 0.01

    low, high = failure_statistics.empirical_bernstein_sequence(failures)

    reference_low, reference_high = _equation_24(failures)
    assert low == pytest.approx(reference_low[-1], rel=1e-12, abs=0)
    assert high == pytest.approx(reference_high[-1], rel=1e-12, abs=0)


def test_the_difference_sequence_is_equation_24_on_the_shifted_differences():
    generator = numpy.random.default_rng(4)
    uniforms = generator.random(5000)
    first_failures = uniforms < 0.02
    second_failures = uniforms < 0.01

    low, high = failure_statistics.difference_sequence(
        first_failures, second_failures
    )

    differences = first_failures.astype(float) - second_failures
    rescaled = (differences + 1) / 2
    reference_low, reference_high = _equation_24(rescaled)
    expected_low = 2 * reference_low[-1] - 1
    expected_high = 2 * reference_high[-1] - 1
    assert low == pytest.approx(expected_low, rel=1e-12, abs=0)
    assert high == pytest.approx(expected_high, rel=1e-12, abs=0)


@pytest.mark.parametrize(
    ("values", "mean", "boundary"),
    [
        # Every prediction is right: V = 0, floored to one.
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0.0, BOUNDARY_AT_ONE),
        # Only the first prediction, zero, misses: V = 1.
        ([1, 1, 1, 1], 1.0, BOUNDARY_AT_ONE),
        # The first misses by a half: V = 1/4, floored to one.
        ([0.5, 0.5, 0.5, 0.5], 0.5, BOUNDARY_AT_ONE),
        # Predictions 0 then 1 miss by one each: V = 2.
        ([1, 0], 0.5, BOUNDARY_AT_TWO),
    ],
)
def test_the_sequence_floors_the_variance_at_one(values, mean, boundary):
    low, high = failure_statistics.empirical_bernstein_sequence(values)

    radius = boundary / len(values)
    expected_low = mean - radius
    expected_high = mean + radius
    assert low == pytest.approx(expected_low, rel=1e-12, abs=0)
    assert high == pytest.approx(expected_high, rel=1e-12, abs=0)


def test_no_value_gives_no_sequence():
    single = failure_statistics.empirical_bernstein_sequence([])
    paired = failure_statistics.difference_sequence([], [])

    assert single is None
    assert paired is None


def test_the_difference_sequence_refuses_failures_that_do_not_pair():
    with pytest.raises(ValueError) as refusal:
        failure_statistics.difference_sequence([True, False], [True])

    assert str(refusal.value) == (
        "the two points' failures pair seed by seed, but the first has 2 "
        "seeds and the second 1"
    )


def test_the_sequence_covers_a_learning_run_at_its_stop_property():
    """A learner whose failure chance falls with each failure.

    As an online threshold's does; 200 seeded runs to the 20th failure.
    """
    generator = numpy.random.default_rng(11)
    covered_runs = 0
    for _ in range(200):
        failures, chances = _learning_run(generator, 0.05, 0.15)
        low, high = failure_statistics.empirical_bernstein_sequence(failures)
        average_chance = numpy.mean(chances)
        covered_runs += low <= average_chance <= high

    assert covered_runs / 200 >= 0.95


def test_the_difference_sequence_covers_two_learners_property():
    """Two learners on shared uniforms, as two decoders on one sample.

    200 seeded runs, each to the earlier 20th failure.
    """
    generator = numpy.random.default_rng(5)
    covered_runs = 0
    for _ in range(200):
        first, second, difference = _paired_learning_run(generator)
        low, high = failure_statistics.difference_sequence(first, second)
        covered_runs += low <= difference <= high

    assert covered_runs / 200 >= 0.95


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


def _equation_24(values):
    """Howard et al. eq. (24) at every prefix, as whole arrays."""
    values = numpy.asarray(values, dtype=float)
    count = len(values)
    positions = numpy.arange(count) + 1
    sums = numpy.cumsum(values)
    means = sums / positions
    previous_means = numpy.concatenate(([0.0], means[:-1]))
    squared_deviations = (values - previous_means) ** 2
    deviation_sums = numpy.cumsum(squared_deviations)
    variance = numpy.maximum(deviation_sums, 1.0)
    doubled_variance = 2 * variance
    log_doubled = numpy.log(doubled_variance)
    iterated_log = numpy.log(log_doubled)
    root_argument = variance * (iterated_log + 3.8)
    root = numpy.sqrt(root_argument)
    radius = (1.7 * root + 3.4 * iterated_log + 13) / positions
    return means - radius, means + radius


def _learning_run(generator, base, span):
    """Failure bits and each shot's failure chance, to the 20th failure."""
    failures = []
    chances = []
    failure_count = 0
    while failure_count < 20:
        exponent = -failure_count / 5
        decay = numpy.exp(exponent)
        chance = base + span * decay
        failed = generator.random() < chance
        failures.append(failed)
        chances.append(chance)
        failure_count += failed
    return failures, chances


def _paired_learning_run(generator):
    """Two learners' bits on shared seeds and their average difference."""
    first = []
    second = []
    differences = []
    first_count = 0
    second_count = 0
    while max(first_count, second_count) < 20:
        first_exponent = -first_count / 5
        second_exponent = -second_count / 5
        first_decay = numpy.exp(first_exponent)
        second_decay = numpy.exp(second_exponent)
        first_chance = 0.06 + 0.12 * first_decay
        second_chance = 0.04 + 0.08 * second_decay
        uniform = generator.random()
        first_failed = uniform < first_chance
        second_failed = uniform < second_chance
        difference = first_chance - second_chance
        first.append(first_failed)
        second.append(second_failed)
        differences.append(difference)
        first_count += first_failed
        second_count += second_failed
    return first, second, numpy.mean(differences)
