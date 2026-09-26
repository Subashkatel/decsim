"""The tail of one count of detection events under the usual rates."""

import dataclasses

import numpy
import scipy.special
import scipy.stats

# The draws per fault count that integrate a count's tail given that
# many faults; tests/burst_detectors/ holds the tail against Stim's
# sampled one and against exact tails.
DRAWS_PER_FAULT_COUNT = 4000
# The draws are a Monte Carlo integral over the circuit's own model, so
# they take one fixed seed and every shot of a run gets the same
# thresholds.
CALIBRATION_SEED = 0
# The fault counts the law integrates cover a rate twice the calibrated
# one; past that the truncated remainder is counted as a firing, so the
# truncation can only raise the tail.
RATE_HEADROOM = 2.0
# The truncated remainder is kept this far below the smallest
# false-alarm rate the law answers for.
TRUNCATION_SHARE = 0.01


@dataclasses.dataclass(frozen=True)
class TailLaw:
    """The tail of one count under the usual rates.

    Each fault of prior p fires a Poisson number of times of rate
    -ln(1 - 2p) / 2, whose chance of being odd, (1 - e^(-2 rate)) / 2, is
    p: only a fault's parity reaches the detectors, so the counted
    detectors flip exactly as under Stim's independent faults (the
    odd-number law of Tan et al. 2406.18897 lines 956-960). The firings
    are one Poisson process of rate fault_rate, the rates' sum, and
    conditional_tails[n][k] is the chance that n firings, picked in
    proportion to the rates and XOR-ed onto the counted detectors, leave
    at least k of them flipped, so two faults that cancel on a shared
    detector count as Stim counts them. The tail is sum_n Poisson(n;
    rate) conditional_tails[n][k] plus the truncated remainder counted
    as a firing. It is exact but for two things: conditional_tails is
    drawn, DRAWS_PER_FAULT_COUNT draws per fault count, so the tail is
    an estimate with that sampling error, and the truncation raises it
    by at most the remainder. A law with no faults is a count that is
    always zero.
    """

    fault_rate: float
    conditional_tails: numpy.ndarray

    @classmethod
    def from_priors(
        cls,
        priors: numpy.ndarray,
        incidence: numpy.ndarray,
        smallest_false_alarms: float,
    ) -> "TailLaw":
        """The tail law of one count, integrated by fault count.

        priors holds each fault's probability and incidence its row of
        0/1 over the counted detectors. The fault counts cover
        RATE_HEADROOM times the calibrated rate until the Poisson
        remainder is below TRUNCATION_SHARE of the smallest false-alarm
        rate asked for.
        """
        fault_rates = _parity_rates(priors)
        fault_rate = numpy.sum(fault_rates)
        if fault_rate == 0:
            return _faultless_law(incidence)
        headroom_rate = fault_rate * RATE_HEADROOM
        remainder_bound = smallest_false_alarms * TRUNCATION_SHARE
        largest_fault_count = _fault_count_bound(headroom_rate, remainder_bound)
        generator = numpy.random.default_rng(CALIBRATION_SEED)
        conditional_tails = _conditional_tails(
            fault_rates, incidence, largest_fault_count, generator
        )
        return cls(fault_rate, conditional_tails)

    def tails(self, rate_scale: float) -> numpy.ndarray:
        """P(count >= k) for every k, with every fault's rate scaled.

        A scale s on the rates gives a fault of prior p the odd chance
        (1 - (1 - 2p)^s) / 2, the law of s copies of the fault; s = 1 is
        the calibrated law.
        """
        rate = self.fault_rate * rate_scale
        if rate == 0:
            return self.conditional_tails[0]
        fault_count_limit = len(self.conditional_tails)
        fault_counts = numpy.arange(fault_count_limit)
        # Poisson(n; rate) = exp(n log rate - rate - log n!), the form
        # that costs one vector of logs rather than scipy's per-call setup
        counts_plus_one = fault_counts + 1
        log_factorials = scipy.special.gammaln(counts_plus_one)
        log_rate = numpy.log(rate)
        log_weights = fault_counts * log_rate - rate - log_factorials
        weights = numpy.exp(log_weights)
        weight_total = numpy.sum(weights)
        # the weights past the last fault count, never below zero when
        # the sum rounds past one
        uncovered = 1.0 - weight_total
        remainder = max(uncovered, 0.0)
        drawn = weights @ self.conditional_tails
        return drawn + remainder

    def threshold(self, false_alarms: float, rate_scale: float) -> int:
        """The smallest count whose tail is at most false_alarms.

        One past the largest count when none is, a count no round
        reaches, so the statistic never fires.
        """
        tails = self.tails(rate_scale)
        is_rare_enough = tails <= false_alarms
        rare_counts = numpy.flatnonzero(is_rare_enough)
        if len(rare_counts) == 0:
            return len(tails)
        return int(rare_counts[0])


def _fault_count_bound(rate: float, remainder_bound: float) -> int:
    """The fewest fault counts whose Poisson remainder is below the bound."""
    largest_fault_count = 1
    while scipy.stats.poisson.sf(largest_fault_count, rate) > remainder_bound:
        largest_fault_count += 1
    return largest_fault_count


def _parity_rates(priors):
    """Each fault's Poisson rate whose odd chance is its prior.

    A Poisson count of rate r is odd with chance (1 - e^(-2r)) / 2, which
    is p at r = -ln(1 - 2p) / 2.
    """
    survivals = 1.0 - 2.0 * priors
    log_survivals = numpy.log(survivals)
    return -log_survivals / 2.0


def _faultless_law(incidence) -> TailLaw:
    """The law of a count no fault reaches: zero, with tail one at k = 0."""
    column_count = incidence.shape[1]
    tail_width = column_count + 2
    tails = numpy.zeros((1, tail_width))
    tails[0, 0] = 1.0
    return TailLaw(0.0, tails)


def _conditional_tails(
    fault_rates, incidence, largest_fault_count: int, generator
):
    """P(count >= k | n firings) for n up to the bound, by drawing."""
    column_count = incidence.shape[1]
    packed = numpy.packbits(incidence, axis=1)
    choice_weights = fault_rates / numpy.sum(fault_rates)
    row_count = largest_fault_count + 1
    tail_width = column_count + 2
    tails = numpy.zeros((row_count, tail_width))
    tails[0, 0] = 1.0
    for fault_count in range(1, row_count):
        tails[fault_count] = _drawn_tail(
            packed, choice_weights, fault_count, tail_width, generator
        )
    return tails


def _drawn_tail(
    packed, choice_weights, fault_count: int, tail_width, generator
):
    """The tail of the flipped count when fault_count faults fire."""
    fault_total = len(choice_weights)
    draw_shape = (DRAWS_PER_FAULT_COUNT, fault_count)
    picks = generator.choice(fault_total, size=draw_shape, p=choice_weights)
    picked_rows = packed[picks]
    parity = numpy.bitwise_xor.reduce(picked_rows, axis=1)
    flipped_bits = numpy.unpackbits(parity, axis=1)
    # bincount converts its input to intp (numpy
    # _core/src/multiarray/compiled_base.c, arr_bincount); a sum of uint8
    # is uint64, which numpy 2.0 and 2.1 refuse to cast to intp.
    flipped_counts = flipped_bits.sum(axis=1, dtype=numpy.intp)
    histogram = numpy.bincount(flipped_counts, minlength=tail_width)
    reversed_histogram = histogram[::-1]
    reversed_tails = numpy.cumsum(reversed_histogram)
    at_least = reversed_tails[::-1]
    return at_least / DRAWS_PER_FAULT_COUNT
