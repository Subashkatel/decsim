"""The complementary gap against the published calibration law.

Gidney et al. 2312.04522 (Fig. 10) and Toshio et al. 2510.25222 (Fig. 4)
report that MWPM's complementary gap is a calibrated confidence: the
probability that the decoder's class is wrong at a gap of g decibels is
fitted by f(g) = 1 / (1 + 10^(0.09 g)) on 10d-round memory experiments
at p = 1e-3. Here 400 shots of a d=3, 30-round memory are binned by
nearest decibel and f(g) must lie inside each populated bin's Wilson 95
percent interval. Both class weights come from the pymatching row's own
pinned graph, the way the machine gets them: two forced-class jobs of
one window, subtracted by this signal. The hand-written graph at the end
pins the two weights to that graph's edges, which no sampled circuit can
isolate.
"""

import math

import numpy
import pytest
import stim

import decsim.confidence.complementary as complementary
import decsim.decoders.minimum_weight_perfect_matching.decoder as adapter
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


def _forced_class_solves(row, model, shot) -> list:
    """The two forced-class solves of one shot, the row's own."""
    solves = []
    for forced_class in complementary.FORCED_LOGICAL_CLASSES:
        job = windows.job_for(model, shot)
        job.forced_logical_class = forced_class
        solve = row.decode(job)
        solves.append(solve)
    return solves


def _class_weights(solves) -> list:
    """The weight each solve reported, in class order."""
    weights = []
    for solve in solves:
        weights.append(solve.forced_class_weight)
    return weights


def _gaps_and_failures() -> tuple:
    """Per shot: the gap in decibels and whether the lighter class is wrong."""
    circuit = windows.memory_circuit(3, ROUNDS, 0.001)
    requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    model = windows.whole_circuit_window(circuit, ROUNDS, requirement)
    signal = complementary.ComplementaryGap()
    row = adapter.PyMatchingDecoder()
    events, observables = windows.sampled_shots(circuit, SHOTS, 1)
    gap_decibels = []
    failures = []
    for shot, observable in zip(events, observables):
        solves = _forced_class_solves(row, model, shot)
        computation = signal.compute(solves)
        soft_output = computation.soft_output
        decibels = soft_output.gap * DECIBELS_PER_NAT
        gap_decibels.append(decibels)
        weights = _class_weights(solves)
        is_odd_class_lighter = weights[1] < weights[0]
        lighter_class = int(is_odd_class_lighter)
        truth = int(observable[0])
        failed = lighter_class != truth
        failures.append(failed)
    return gap_decibels, failures


def test_the_gap_is_calibrated_to_the_published_law_in_every_populated_bin():
    """A referent test over 400 sampled shots and their decibel bins."""
    gap_decibels, failures = _gaps_and_failures()
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


def test_a_window_without_an_observable_reports_no_weight_and_no_gap():
    """The fail-safe: no observable to pin, no forced solve, no gap."""
    circuit = windows.memory_circuit(3, 6, 0.001)
    stripped = stim.Circuit()
    for instruction in circuit:
        if instruction.name != "OBSERVABLE_INCLUDE":
            stripped.append(instruction)
    requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    model = windows.whole_circuit_window(stripped, 6, requirement)
    row = adapter.PyMatchingDecoder()
    events, _observables = windows.sampled_shots(stripped, 1, 3)
    solves = _forced_class_solves(row, model, events[0])
    assert _class_weights(solves) == [None, None]
    signal = complementary.ComplementaryGap()
    computation = signal.compute(solves)
    assert computation.soft_output is None
    assert computation.ticks == 0
    assert signal.source is complementary.COMPLEMENTARY_GAP_SOURCE
    assert signal.fault_model_requirement is requirement


def test_the_gap_is_the_weight_difference_of_the_two_classes():
    """One shot: g is the two class weights' difference, both reported."""
    circuit = windows.memory_circuit(3, ROUNDS, 0.001)
    requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    model = windows.whole_circuit_window(circuit, ROUNDS, requirement)
    row = adapter.PyMatchingDecoder()
    events, _observables = windows.sampled_shots(circuit, 1, 3)
    solves = _forced_class_solves(row, model, events[0])
    signal = complementary.ComplementaryGap()
    computation = signal.compute(solves)
    soft_output = computation.soft_output
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


def test_the_two_class_weights_are_the_pinned_graphs_own_edges():
    """The gap is measured on the graph the forced solves run on.

    Appending the observable row as one more check separates faults
    that the plain graph merges when they share detector endpoints but
    differ in their logical effect, which is what a class-constrained
    solve needs. Here two parallel 0.05 columns both flip detector 0
    and the observable, so they still share endpoints on the pinned
    graph and merge at p1(1 - p2) + p2(1 - p1) = 0.095 (weight 2.254);
    the even class runs through detector 1's two 0.2 edges (weight
    2.773). Faults that flip the observable are boundary faults here,
    as they are for a logical operator supported on the code boundary
    (Stim's memory circuits), which the pinned solve requires.
    """
    check = numpy.asarray([[1, 1, 1, 0], [0, 0, 1, 1]], dtype=numpy.uint8)
    priors = numpy.asarray([0.05, 0.05, 0.2, 0.2])
    observables = numpy.asarray([[1, 1, 0, 0]], dtype=numpy.uint8)
    faults = placed_faults(check, priors, observables)
    row = adapter.PyMatchingDecoder()
    graphs = row.compile(faults)
    syndrome = numpy.asarray([1, 0], dtype=numpy.uint8)
    solves = []
    for forced_class in complementary.FORCED_LOGICAL_CLASSES:
        answer = row.decode_forced_window(
            graphs, None, faults, syndrome, forced_class
        )
        solves.append(answer)
    weights = _class_weights(solves)
    # 2 ln(0.8 / 0.2) for the even class, ln((1 - 0.095) / 0.095) for the odd
    assert weights[0] == pytest.approx(2.772588722239781, abs=1e-6)
    assert weights[1] == pytest.approx(2.2540580520993854, abs=1e-6)
    signal = complementary.ComplementaryGap()
    computation = signal.compute(solves)
    soft_output = computation.soft_output
    assert soft_output.decoded_class_weight == pytest.approx(
        2.2540580520993854, abs=1e-6
    )
    assert soft_output.gap == pytest.approx(0.5185306701404, abs=1e-6)
