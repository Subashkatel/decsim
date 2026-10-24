"""Three runs paired by seed, against hand counts and the basic case.

- McNemar's exact p: the basic case's sum of binomial coefficients
  (experiments/basic_case/window_check_plots.py), written again here.
- The counts: a hand-made set of shots, and the identity that the pair
  fails broken - repaired more times than the fast decoder alone, on
  outcomes numpy draws.
- The rows: shots.csv text read by csv.DictReader.
"""

import csv
import io
import math

import numpy
import pytest

import decsim.experiments.failure_statistics as failure_statistics
import decsim.experiments.paired_shots as paired_shots

# The float error of a sum of a few thousand binomial terms.
P_VALUE_TOLERANCE = 1e-12

SHOTS_CSV = """decoding,seed,is_scored,logical_failure
fast,1,True,True
fast,2,True,False
fast,3,False,True
pair,1,True,False
"""


def test_mcnemars_exact_p_is_the_basic_cases_binomial_sum_property():
    for first_only in range(13):
        for second_only in range(13):
            p_value = failure_statistics.exact_mcnemar(first_only, second_only)
            expected = _binomial_sum_p(first_only, second_only)
            assert p_value == pytest.approx(expected, abs=P_VALUE_TOLERANCE)


def test_shots_with_no_discordant_pair_give_a_p_of_one():
    assert failure_statistics.exact_mcnemar(0, 0) == 1.0


def test_a_hand_made_set_of_shots_counts_each_kind_on_the_shared_seeds():
    fast = {1: True, 2: True, 3: False, 4: False, 5: True, 9: True}
    strong = {1: False, 2: True, 3: False, 4: False, 5: False}
    pair = {1: False, 2: True, 3: True, 4: False, 5: True}
    repairs = paired_shots.paired_repairs(fast, strong, pair)
    assert repairs == paired_shots.PairedRepairs(
        shots=5,
        fast_failures=3,
        strong_failures=1,
        pair_failures=3,
        best_of_two_failures=1,
        repaired=1,
        broken=1,
        missed=2,
        p_value=1.0,
    )


def test_the_best_of_two_line_is_under_each_decoder_alone_property():
    generator = numpy.random.default_rng(seed=11)
    for _ in range(200):
        fast = _random_outcomes(generator)
        strong = _random_outcomes(generator)
        pair = _random_outcomes(generator)
        repairs = paired_shots.paired_repairs(fast, strong, pair)
        fewer_alone = min(repairs.fast_failures, repairs.strong_failures)
        assert repairs.best_of_two_failures <= fewer_alone


def test_the_pair_fails_broken_minus_repaired_more_than_fast_alone_property():
    generator = numpy.random.default_rng(seed=7)
    for _ in range(200):
        fast = _random_outcomes(generator)
        strong = _random_outcomes(generator)
        pair = _random_outcomes(generator)
        repairs = paired_shots.paired_repairs(fast, strong, pair)
        change = repairs.pair_failures - repairs.fast_failures
        assert change == repairs.broken - repairs.repaired


def test_a_runs_failures_leave_out_unscored_shots_and_other_runs():
    text = io.StringIO(SHOTS_CSV)
    reader = csv.DictReader(text)
    rows = list(reader)
    failures = paired_shots.failures_by_seed(rows, {"decoding": "fast"})
    assert failures == {1: True, 2: False}


def _binomial_sum_p(first_only: int, second_only: int) -> float:
    """The basic case's McNemar p, as window_check_plots.py writes it.

    With no discordant pair the one term doubles to 2, so the min gives 1.
    """
    discordant = first_only + second_only
    fewer = min(first_only, second_only)
    last = fewer + 1
    counts = range(last)
    tail = sum(math.comb(discordant, count) for count in counts)
    both_sides = 2 * tail / 2**discordant
    return min(1.0, both_sides)


def _random_outcomes(generator: numpy.random.Generator) -> dict:
    """{seed: failed} on 40 seeds, about one in three failed."""
    failed = generator.random(40) < 0.3
    outcomes = failed.tolist()
    return dict(enumerate(outcomes))
