"""The complementary gap against the published calibration law.

Gidney et al. 2312.04522 (Fig. 10) and Toshio et al. 2510.25222 (Fig. 4)
report that MWPM's complementary gap is a calibrated confidence: the
probability that the decoder's class is wrong at a gap of g decibels is
fitted by f(g) = 1 / (1 + 10^(0.09 g)) on 10d-round memory experiments
at p = 1e-3 (the sandbox harness rowD6's law). Here 400 shots of a d=3,
30-round memory are binned by nearest decibel and f(g) must lie inside
each populated bin's Wilson 95 percent interval; the two forced-class
solves of the pair must agree with the plain solve on the minimum
weight (min(w_forced) == w_plain, the whole-window consistency of
Sec. III A).
"""

import math

import numpy
import pytest
import stim

import decsim.confidence.complementary as complementary
import decsim.detector_error_model.fault_model_contracts as fault_models
import tests.decoders.windows as windows

DECIBELS_PER_NAT = 10.0 * math.log10(math.e)
ROUNDS = 30
SHOTS = 400
MINIMUM_BIN_SAMPLES = 30


def _fitted_failure_probability(gap_decibels: float) -> float:
    exponent = 0.09 * gap_decibels
    return 1.0 / (1.0 + 10.0**exponent)


def _wilson_interval(failures: int, count: int) -> tuple:
    z = 1.96
    proportion = failures / count
    z_squared = z * z
    denominator = 1.0 + z_squared / count
    center = (proportion + z_squared / (2 * count)) / denominator
    spread = proportion * (1.0 - proportion) / count
    spread += z_squared / (4 * count * count)
    half_width = z * math.sqrt(spread) / denominator
    return center - half_width, center + half_width


def _gaps_and_failures() -> tuple:
    """Per shot: the gap in decibels, the class's wrongness, both w_min.

    The plain solve's minimum weight and the paired solves' minimum
    weight are returned as two lists.
    """
    circuit = windows.memory_circuit(3, ROUNDS, 0.001)
    requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    model = windows.whole_circuit_window(circuit, ROUNDS, requirement)
    signal = complementary.ComplementaryGap()
    metric = signal.metric_for(model)
    events, observables = windows.sampled_shots(circuit, SHOTS, 1)
    gap_decibels = []
    failures = []
    plain_minimum_weights = []
    paired_minimum_weights = []
    for shot, observable in zip(events, observables):
        syndrome = windows.row_syndrome(model, shot)
        plain = metric.evaluate(syndrome)
        paired = metric.paired_evaluate(syndrome)
        truth = int(observable[0])
        decibels = plain.gap * DECIBELS_PER_NAT
        gap_decibels.append(decibels)
        failed = paired.predicted_class != truth
        failures.append(failed)
        plain_minimum_weights.append(plain.w_min)
        paired_minimum_weights.append(paired.soft_output.w_min)
    return gap_decibels, failures, plain_minimum_weights, paired_minimum_weights


def test_the_gap_is_calibrated_to_the_published_law_in_every_populated_bin():
    """A referent test over 400 sampled shots and their decibel bins."""
    gap_decibels, failures, _plain, _paired = _gaps_and_failures()
    counts = {}
    failure_counts = {}
    for gap, failed in zip(gap_decibels, failures):
        rounded = round(gap)
        bin_decibels = int(rounded)
        counts[bin_decibels] = counts.get(bin_decibels, 0) + 1
        failure_counts[bin_decibels] = failure_counts.get(bin_decibels, 0)
        failure_counts[bin_decibels] += int(failed)
    populated_bins = 0
    for bin_decibels, count in counts.items():
        if count < MINIMUM_BIN_SAMPLES:
            continue
        populated_bins += 1
        low, high = _wilson_interval(failure_counts[bin_decibels], count)
        law = _fitted_failure_probability(bin_decibels)
        assert low <= law <= high, (bin_decibels, count, low, law, high)
    assert populated_bins >= 3


def test_the_paired_solves_minimum_weight_is_the_plain_minimum_weight():
    """A referent test: min(w_forced) == w_plain on every sampled shot."""
    _gaps, _failures, plain, paired = _gaps_and_failures()
    assert paired == pytest.approx(plain)


def test_the_signal_has_no_metric_for_a_model_without_an_observable():
    circuit = windows.memory_circuit(3, 6, 0.001)
    stripped = stim.Circuit()
    for instruction in circuit:
        if instruction.name != "OBSERVABLE_INCLUDE":
            stripped.append(instruction)
    requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    model = windows.whole_circuit_window(stripped, 6, requirement)
    signal = complementary.ComplementaryGap()
    assert signal.metric_for(model) is None
    assert signal.source is complementary.COMPLEMENTARY_GAP_SOURCE
    assert signal.fault_model_requirement is requirement


def test_the_gap_is_the_weight_difference_of_the_two_classes():
    """One shot: g = |w_comp - w_min|, both weights on the result."""
    circuit = windows.memory_circuit(3, ROUNDS, 0.001)
    requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    model = windows.whole_circuit_window(circuit, ROUNDS, requirement)
    signal = complementary.ComplementaryGap()
    metric = signal.metric_for(model)
    events, _observables = windows.sampled_shots(circuit, 1, 3)
    syndrome = windows.row_syndrome(model, events[0])
    soft_output = metric.evaluate(syndrome)
    difference = soft_output.w_comp - soft_output.w_min
    assert soft_output.gap == pytest.approx(abs(difference))
    assert soft_output.w_comp >= soft_output.w_min
    assert numpy.isfinite(soft_output.gap)
