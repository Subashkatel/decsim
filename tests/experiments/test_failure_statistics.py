"""The failure statistics against scipy, sinter, mpmath and exact sums.

- Limits: scipy.stats.beta and the NIST/SEMATECH e-Handbook 7.2.4.1
  closed form.
- Coverage and the Girshick, Mosteller and Savage estimator's
  unbiasedness: exact enumeration of one small plan, with scipy's nbinom
  and binom as the outcome probabilities.
- Per-round rate: sinter 1.16.0 shot_error_rate_to_piece_error_rate, and
  the root at 60 digits in mpmath 1.3.0, which the test extra pins.
- Decay fit over several lengths: Stim's rotated memory circuits decoded
  whole by PyMatching, a long one against short ones; the closed form of
  Tan et al. 2209.09219 eq. S4; and its coverage on counts numpy draws
  from that law, apart and as checkpoints of shared runs.
"""

import math

import mpmath
import numpy
import pymatching
import pytest
import scipy.stats
import sinter
import stim

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
# The decay law the fit's coverage is drawn from: per-round and boundary
# flip probabilities near a distance-3 memory's (Tan et al. eq. S4).
DECAY_PER_ROUND = 0.002
DECAY_BOUNDARY = 0.01
DECAY_LENGTHS = numpy.array([10, 20, 30, 40, 50])
DECAY_DRAWS = 400
# A distance-3 rotated memory at p = 0.005, whose failure rate passes
# 0.3 by 120 rounds, so the short lengths stay far from one half.
MEMORY_ERROR_PROBABILITY = 0.005
MEMORY_SHOTS = 20_000


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


def test_a_fit_over_short_stim_memories_agrees_with_one_long_memory():
    """Whole-circuit PyMatching: the fit over 6 to 30 rounds against 120."""
    short_lengths = [6, 12, 18, 24, 30]
    failures = [_memory_failures(length) for length in short_lengths]
    shots = [MEMORY_SHOTS] * len(short_lengths)
    long_failures = _memory_failures(120)

    fit = failure_statistics.per_round_rate_over_lengths(
        short_lengths, failures, shots
    )
    long = failure_statistics.estimate(
        long_failures, MEMORY_SHOTS, StopKind.CAP
    )
    long_rate = failure_statistics.per_round_rate(long.rate, 120)
    long_low = failure_statistics.per_round_rate(long.low, 120)
    long_high = failure_statistics.per_round_rate(long.high, 120)
    # a two-sided test at 95 percent on the difference of two estimates,
    # each deviation read off its own interval
    quantile = failure_statistics.NORMAL_UPPER_QUANTILE
    fit_deviation = (fit.high - fit.low) / (2 * quantile)
    long_deviation = (long_high - long_low) / (2 * quantile)
    combined = math.hypot(fit_deviation, long_deviation)
    difference = fit.per_round_rate - long_rate
    assert abs(difference) < quantile * combined


def test_a_fit_recovers_the_rates_of_an_exact_decay():
    """Counts of 10^12 shots that follow Tan et al. eq. S4 to the shot."""
    lengths = [6, 12, 18, 24, 30]
    shots = 10**12
    survivals = (1 - 2 * 0.01) * (1 - 2 * 0.002) ** numpy.array(lengths)
    exact_failures = (1 - survivals) / 2 * shots
    failures = numpy.rint(exact_failures)
    shot_counts = [shots] * 5

    fit = failure_statistics.per_round_rate_over_lengths(
        lengths, failures, shot_counts
    )

    assert fit.per_round_rate == pytest.approx(0.002, rel=1e-9, abs=0)
    assert fit.boundary_rate == pytest.approx(0.01, rel=1e-9, abs=0)
    assert fit.low < 0.002 < fit.high


def test_the_interval_holds_the_true_rate_95_times_in_100_property():
    """Lengths drawn apart; 400 draws hold 0.95 within three deviations."""
    generator = numpy.random.default_rng(5)
    probabilities = _decay_probabilities(DECAY_LENGTHS)
    shots = [10_000] * len(DECAY_LENGTHS)
    held = 0
    for _ in range(DECAY_DRAWS):
        failures = generator.binomial(10_000, probabilities)
        fit = failure_statistics.per_round_rate_over_lengths(
            DECAY_LENGTHS, failures, shots
        )
        held += fit.low <= DECAY_PER_ROUND <= fit.high

    assert 0.92 <= held / DECAY_DRAWS <= 0.98


def test_checkpoints_of_shared_runs_hold_the_true_rate_property():
    """Every length a checkpoint of the same 4,000 runs: Tan's eq. S9."""
    generator = numpy.random.default_rng(6)
    shots = [4_000] * len(DECAY_LENGTHS)
    gaps = numpy.diff(DECAY_LENGTHS, prepend=0)
    held = 0
    for _ in range(DECAY_DRAWS):
        flips = generator.binomial(gaps, DECAY_PER_ROUND, size=(4_000, 5))
        parities = numpy.cumsum(flips, axis=1) % 2
        # each checkpoint's own readout flips it once, apart from the rest
        readouts = generator.random((4_000, 5)) < DECAY_BOUNDARY
        wrong = parities ^ readouts
        failures = numpy.sum(wrong, axis=0)
        fit = failure_statistics.per_round_rate_over_lengths(
            DECAY_LENGTHS, failures, shots, lengths_share_runs=True
        )
        held += fit.low <= DECAY_PER_ROUND <= fit.high

    assert held / DECAY_DRAWS >= 0.95


def test_a_fit_refuses_a_length_with_no_failure():
    with pytest.raises(ValueError) as refusal:
        failure_statistics.per_round_rate_over_lengths(
            [6, 12], [0, 3], [1000, 1000]
        )

    message = str(refusal.value)
    assert "the runs of 6 rounds have no failure in 1000 shots" in message


def test_a_fit_refuses_a_rate_at_one_half():
    with pytest.raises(ValueError) as refusal:
        failure_statistics.per_round_rate_over_lengths(
            [6, 12], [3, 500], [1000, 1000]
        )

    message = str(refusal.value)
    assert "the runs of 12 rounds fail 500 times in 1000 shots" in message


def test_a_fit_refuses_one_length():
    with pytest.raises(ValueError) as refusal:
        failure_statistics.per_round_rate_over_lengths(
            [6, 6], [3, 4], [1000, 1000]
        )

    message = str(refusal.value)
    assert "needs runs of at least two lengths" in message


def _memory_failures(rounds):
    """PyMatching's failures on Stim's memory, the seed its length."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=rounds,
        after_clifford_depolarization=MEMORY_ERROR_PROBABILITY,
        before_round_data_depolarization=MEMORY_ERROR_PROBABILITY,
        before_measure_flip_probability=MEMORY_ERROR_PROBABILITY,
        after_reset_flip_probability=MEMORY_ERROR_PROBABILITY,
    )
    sampler = circuit.compile_detector_sampler(seed=rounds)
    detections, observables = sampler.sample(
        MEMORY_SHOTS, separate_observables=True
    )
    model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)
    predictions = matching.decode_batch(detections)
    wrong = predictions != observables
    failures = numpy.sum(wrong)
    return int(failures)


def _decay_probabilities(lengths):
    """P(n) of Tan et al. eq. S4 at the decay law's two rates."""
    round_survival = 1 - 2 * DECAY_PER_ROUND
    boundary_survival = 1 - 2 * DECAY_BOUNDARY
    survivals = boundary_survival * round_survival**lengths
    return (1 - survivals) / 2


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
