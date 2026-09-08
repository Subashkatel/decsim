"""The complementary gap against the published calibration law.

Gidney et al. 2312.04522 (Fig. 10) and Toshio et al. 2510.25222 (Fig. 4)
report that MWPM's complementary gap is a calibrated confidence: the
probability that the decoder's class is wrong at a gap of g decibels is
fitted by f(g) = 1 / (1 + 10^(0.09 g)) on 10d-round memory experiments
at p = 1e-3. Here 400 shots of a d=3,
30-round memory are binned by nearest decibel and f(g) must lie inside
each populated bin's Wilson 95 percent interval; the two forced-class
solves of the pair must agree with the plain solve on the minimum
weight (min(w_forced) == w_plain, the whole-window consistency of
Sec. III A). The hand-written graph at the end pins the gap to the
window decoder's own minimum, which no sampled circuit can isolate.
"""

import math

import numpy
import pytest
import stim

import decsim.confidence.complementary as complementary
import decsim.decoders.minimum_weight_perfect_matching.decoder as adapter
import decsim.decoders.minimum_weight_perfect_matching.weights as weights_module
import decsim.detector_error_model.fault_model_contracts as fault_models
import tests.decoders.windows as windows

DECIBELS_PER_NAT = 10.0 * math.log10(math.e)
ROUNDS = 30
SHOTS = 400
MINIMUM_BIN_SAMPLES = 30
GRAPHLIKE = fault_models.FaultRepresentation.GRAPHLIKE


def placed_faults(check, priors, observables):
    """One window's graphlike faults from a hand-written check matrix."""
    matrix = numpy.asarray(check, dtype=numpy.uint8)
    prior_array = numpy.asarray(priors, dtype=float)
    observable_matrix = numpy.asarray(observables, dtype=numpy.uint8)
    fault_count = matrix.shape[1]
    owned = numpy.ones(fault_count, dtype=bool)
    source_fault_ids = tuple(range(fault_count))
    return fault_models.PlacedFaultModel(
        representation=GRAPHLIKE,
        check=matrix,
        priors=prior_array,
        observables=observable_matrix,
        owned=owned,
        source_fault_ids=source_fault_ids,
        boundary_flips={},
    )


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
    """Per shot: the gap in decibels, the class's wrongness, both weights.

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
        plain_minimum_weights.append(plain.decoded_class_weight)
        paired_minimum_weights.append(paired.soft_output.decoded_class_weight)
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
    """One shot: g is the two class weights' difference, both reported."""
    circuit = windows.memory_circuit(3, ROUNDS, 0.001)
    requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    model = windows.whole_circuit_window(circuit, ROUNDS, requirement)
    signal = complementary.ComplementaryGap()
    metric = signal.metric_for(model)
    events, _observables = windows.sampled_shots(circuit, 1, 3)
    syndrome = windows.row_syndrome(model, events[0])
    soft_output = metric.evaluate(syndrome)
    difference = (
        soft_output.complementary_class_weight
        - soft_output.decoded_class_weight
    )
    assert soft_output.gap == pytest.approx(abs(difference))
    assert (
        soft_output.complementary_class_weight
        >= soft_output.decoded_class_weight
    )
    assert numpy.isfinite(soft_output.gap)


def test_the_minimum_weight_is_the_window_decoders_own_matching_weight():
    """The gap is measured from the decoder's minimum, not another graph's.

    The metric's base matching merges parallel faults as independent
    errors, p1(1 - p2) + p2(1 - p1), exactly as the PyMatching row's
    graph does (the convention of Stim's detector error models and of
    PyMatching's model loader), so decoded_class_weight is the weight
    that row reports for the same solve and the gap is that decoder's
    own confidence. Two parallel 0.05 columns on detector 0 combine to
    0.095, and the other logical class runs through detector 1's two
    0.2 edges. Faults that flip the observable are boundary faults
    here, as they are for a logical operator supported on the code
    boundary (Stim's memory circuits), which the augmented-detector
    solve requires.
    """
    check = numpy.asarray([[1, 1, 1, 0], [0, 0, 1, 1]], dtype=numpy.uint8)
    priors = numpy.asarray([0.05, 0.05, 0.2, 0.2])
    observables = numpy.asarray([[1, 1, 0, 0]], dtype=numpy.uint8)
    edge_weights = weights_module.matching_weights(priors)
    metric = complementary.ComplementaryGapMetric(
        check, observables, edge_weights
    )
    faults = placed_faults(check, priors, observables)
    row = adapter.PyMatchingDecoder()
    row_graphs = row.compile(faults)
    row_matching = row_graphs.plain
    syndrome = numpy.asarray([1, 0], dtype=numpy.uint8)
    _correction, row_weight = row_matching.decode(syndrome, return_weight=True)
    soft_output = metric.evaluate(syndrome)
    # ln((1 - 0.095) / 0.095) and 2 ln(0.8 / 0.2)
    assert row_weight == pytest.approx(2.2540580520993854, abs=1e-6)
    assert soft_output.decoded_class_weight == pytest.approx(
        2.2540580520993854, abs=1e-6
    )
    assert soft_output.complementary_class_weight == pytest.approx(
        2.772588722239781, abs=1e-6
    )
    assert soft_output.gap == pytest.approx(0.5185306701404, abs=1e-6)
