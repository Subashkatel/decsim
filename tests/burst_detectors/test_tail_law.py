"""The count's tail law against exact tails.

The odd-number law of Tan et al. (2406.18897 lines 956-960): the law's
tail sits within its own sampling error of the exact Poisson binomial,
and of the enumerated tail where faults cancel on a shared detector.
"""

import numpy
import pytest
import scipy.stats

import decsim.burst_detectors.event_count.tail_law as tail_law


def _exact_independent_tails(priors):
    """P(count >= k) for independent detectors: the Poisson binomial.

    Each fault flips its own detector, so the count's distribution is
    the convolution of the faults' Bernoulli terms.
    """
    distribution = numpy.array([1.0])
    for prior in priors:
        bernoulli = [1.0 - prior, prior]
        distribution = numpy.convolve(distribution, bernoulli)
    reversed_distribution = distribution[::-1]
    reversed_tails = numpy.cumsum(reversed_distribution)
    return reversed_tails[::-1]


def _exact_chain_tails(priors):
    """P(count >= k) when fault i flips detectors i and i + 1, enumerated.

    Two neighbouring faults cancel on the detector they share, as Stim
    XORs them, which is the case a sum of priors gets wrong.
    """
    fault_count = len(priors)
    detector_count = fault_count + 1
    incidence = _chain_incidence(fault_count)
    count_values = detector_count + 1
    distribution = numpy.zeros(count_values)
    mask_count = 1 << fault_count
    misses = 1.0 - priors
    for fired_mask in range(mask_count):
        fired = _fired_faults(fired_mask, fault_count)
        chances = numpy.where(fired, priors, misses)
        chance = numpy.prod(chances)
        flipped = incidence[fired].sum(axis=0) % 2
        flipped_count = flipped.sum()
        distribution[flipped_count] += chance
    reversed_distribution = distribution[::-1]
    reversed_tails = numpy.cumsum(reversed_distribution)
    return reversed_tails[::-1]


def _fired_faults(fired_mask: int, fault_count: int):
    shifts = numpy.arange(fault_count)
    shifted = fired_mask >> shifts
    bits = shifted & 1
    return bits.astype(bool)


def _chain_incidence(fault_count: int):
    shape = (fault_count, fault_count + 1)
    incidence = numpy.zeros(shape, dtype=numpy.uint8)
    for fault in range(fault_count):
        incidence[fault, fault] = 1
        incidence[fault, fault + 1] = 1
    return incidence


def _drawn_standard_errors(law):
    """Each tail's sampling error: the drawn conditional tails' spread."""
    fault_counts = numpy.arange(len(law.conditional_tails))
    weights = scipy.stats.poisson.pmf(fault_counts, law.fault_rate)
    conditional = law.conditional_tails
    spread = conditional * (1.0 - conditional)
    squared_weights = weights**2
    variance = squared_weights @ spread
    draws = tail_law.DRAWS_PER_FAULT_COUNT
    variance_of_mean = variance / draws
    return numpy.sqrt(variance_of_mean)


# The priors the exact grid runs, from a quiet chip to ten times Google's
# burst floor, and the fault counts each shape runs at.
GRID_PRIORS = [1e-3, 1e-2, 3e-2, 0.1]
INDEPENDENT_FAULT_COUNTS = [5, 12, 40]
CHAIN_FAULT_COUNTS = [5, 10]
# The smallest tail the grid compares, a tenth of the smallest budget a
# row is likely to ask for; the draws do not resolve smaller ones.
SMALLEST_COMPARED_TAIL = 1e-9
# The float rounding a tail of one carries, where no draw varies.
TAIL_ROUNDING = 1e-12


def _assert_the_law_is_the_exact_tail(priors, incidence, exact_tails):
    law = tail_law.TailLaw.from_priors(priors, incidence, 1e-8)
    tails = law.tails(1.0)
    standard_errors = _drawn_standard_errors(law)
    is_compared = exact_tails >= SMALLEST_COMPARED_TAIL
    compared_counts = numpy.flatnonzero(is_compared)
    law_tails = tails[compared_counts]
    exact = exact_tails[compared_counts]
    errors = standard_errors[compared_counts]
    signed_differences = law_tails - exact
    differences = numpy.abs(signed_differences)
    allowed = 5 * errors + TAIL_ROUNDING
    is_within = differences <= allowed
    assert numpy.all(is_within)


@pytest.mark.parametrize("prior", GRID_PRIORS)
@pytest.mark.parametrize("fault_count", INDEPENDENT_FAULT_COUNTS)
@pytest.mark.parametrize("is_mixed", [False, True])
def test_the_tail_law_is_the_poisson_binomial_tail(
    prior, fault_count, is_mixed
):
    """Independent faults on their own detectors, equal or mixed priors.

    Within five of the law's own sampling errors of the exact tail at
    every count whose tail is at least 1e-9; a law that fired each fault
    at a Poisson rate equal to its prior sat as much as 75 errors low.
    """
    priors = numpy.full(fault_count, prior)
    if is_mixed:
        generator = numpy.random.default_rng(fault_count)
        spread = generator.uniform(0.2, 1.8, fault_count)
        priors = priors * spread
    incidence = numpy.eye(fault_count, dtype=numpy.uint8)
    exact_tails = _exact_independent_tails(priors)

    _assert_the_law_is_the_exact_tail(priors, incidence, exact_tails)


@pytest.mark.parametrize("prior", GRID_PRIORS)
@pytest.mark.parametrize("fault_count", CHAIN_FAULT_COUNTS)
def test_the_tail_law_is_the_exact_tail_where_faults_cancel(prior, fault_count):
    priors = numpy.full(fault_count, prior)
    incidence = _chain_incidence(fault_count)
    exact_tails = _exact_chain_tails(priors)

    _assert_the_law_is_the_exact_tail(priors, incidence, exact_tails)


def test_a_budget_just_above_the_exact_tail_takes_the_exact_threshold():
    """20 faults at 1e-2: count 4 has tail 4.26e-5, over a 4.2e-5 budget."""
    priors = numpy.full(20, 1e-2)
    incidence = numpy.eye(20, dtype=numpy.uint8)
    exact_tails = _exact_independent_tails(priors)
    law = tail_law.TailLaw.from_priors(priors, incidence, 4.2e-5)

    threshold = law.threshold(4.2e-5, 1.0)

    assert exact_tails[4] > 4.2e-5
    assert threshold == 5


def test_a_count_no_fault_reaches_is_always_zero():
    priors = numpy.zeros(3)
    incidence = numpy.eye(3, dtype=numpy.uint8)
    law = tail_law.TailLaw.from_priors(priors, incidence, 1e-6)

    tails = law.tails(1.0)
    threshold = law.threshold(1e-6, 1.0)

    assert list(tails) == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert threshold == 1


def test_the_tail_law_hands_bincount_counts_it_casts_safely(monkeypatch):
    """Before 2.2, numpy casts bincount's input to intp only when safe.

    The stand-in holds numpy 2.2's bincount to that older rule, so the
    law is checked on every numpy: uint64 counts, which numpy.sum gives
    for uint8 bits, raised TypeError there.
    """
    bincount = numpy.bincount

    def safe_casting_bincount(counts, minlength=0):
        assert numpy.can_cast(counts.dtype, numpy.intp), counts.dtype
        return bincount(counts, minlength=minlength)

    monkeypatch.setattr(numpy, "bincount", safe_casting_bincount)
    priors = numpy.full(3, 1e-2)
    incidence = numpy.eye(3, dtype=numpy.uint8)

    law = tail_law.TailLaw.from_priors(priors, incidence, 1e-6)

    tails = law.tails(1.0)
    assert tails[0] == 1.0
