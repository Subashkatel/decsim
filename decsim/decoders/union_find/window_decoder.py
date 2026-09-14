"""Prior-weighted graphlike Union-Find: the graph and one decode on it.

Delfosse and Nickerson 1709.06218:
every odd cluster grows by one half-edge per round (Algorithm 1, step
4), clusters that meet fuse, and the peeling decoder reads the
correction off a spanning forest of the grown erasure (Algorithm 2).
The weighted variant is Huang, Newman and Brown 2004.04693 (same
folder): an edge of log-odds weight w has the integer length
round(w / weight_step), never below one tick, so a likelier fault is
crossed sooner; that quantization is _quantize_weight_ticks.

This module builds the graph of one placed model and turns one
syndrome into the evidence a decode returns. The growth, the forest
and the peeling run in the compiled decoder, reached through
compiled_decoder. What a decode leaves behind is a record a confidence
signal reads, so the graph, its edges, the open and closed intervals
and the weight step live in decsim/records/decoder_evidence.py.
"""

import math
from typing import Optional

import numpy

import decsim.decoders.union_find.compiled_decoder as compiled_decoder
import decsim.records.decoder_evidence as evidence_records

BOUNDARY = -1


def graph_from_model(
    faults, *, location: str, weight_step: float
) -> evidence_records.UnionFindGraph:
    """The weighted graph of one placed graphlike model.

    Faults likelier than one half are the baseline correction; every
    other column becomes an edge whose length is its residual log-odds
    in weight ticks.
    """
    check = faults.check
    raw_priors = numpy.asarray(faults.priors)
    fault_count = check.shape[1]
    _check_priors(raw_priors, fault_count, location)
    # observables are few rows; dense per-fault columns are cheap to read
    observables = faults.observables.toarray()
    observables = observables.astype(numpy.uint8, copy=False)
    priors = raw_priors.astype(float, copy=False)
    likely = priors > 0.5
    baseline = likely.astype(numpy.uint8)
    baseline_syndrome = _parity_product(check, baseline)
    edges = []
    for fault_index in range(fault_count):
        edge = _edge_of(
            check, priors, baseline, observables, fault_index, weight_step
        )
        if edge is not None:
            edges.append(edge)
    logical_columns = _logical_columns(observables, fault_count)
    baseline_faults = _int_tuple(baseline)
    baseline_syndrome = _int_tuple(baseline_syndrome)
    return evidence_records.UnionFindGraph(
        detector_count=check.shape[0],
        fault_count=fault_count,
        edges=tuple(edges),
        baseline_faults=baseline_faults,
        baseline_syndrome=baseline_syndrome,
        logical_observables_by_fault=logical_columns,
        logical_observable_count=observables.shape[0],
        weight_step=weight_step,
    )


def decode_graph(
    graph: evidence_records.UnionFindGraph, syndrome
) -> evidence_records.UnionFindHardEvidence:
    """Decode one syndrome on the graph: grow, take the forest, peel, report."""
    syndrome_array = _checked_syndrome(graph, syndrome)
    baseline_syndrome = numpy.asarray(
        graph.baseline_syndrome, dtype=numpy.uint8
    )
    syndrome_bits = _int_tuple(syndrome_array)
    residual_syndrome = syndrome_array ^ baseline_syndrome
    residual_bits = _int_tuple(residual_syndrome)
    outcome = compiled_decoder.decode(graph, residual_syndrome)
    selected_edges = outcome.selected_edges
    selected_faults = _selected_faults(graph, selected_edges)
    reproduced = _reproduced_syndrome(graph, baseline_syndrome, selected_edges)
    mismatch = reproduced ^ syndrome_array
    unmatched = numpy.nonzero(mismatch)
    unmatched_detectors = _int_tuple(unmatched[0])
    logical_observables = _logical_observables(graph, selected_faults)
    contact_faults = _fault_indices(graph, outcome.contact_edges)
    erasure_forest_faults = _fault_indices(graph, outcome.forest_edges)
    return evidence_records.UnionFindHardEvidence(
        graph=graph,
        syndrome=syndrome_bits,
        residual_syndrome=residual_bits,
        selected_faults=tuple(selected_faults),
        contact_faults=contact_faults,
        edge_intervals=outcome.edge_intervals,
        erasure_forest_faults=erasure_forest_faults,
        logical_observables=logical_observables,
        unmatched_detectors=unmatched_detectors,
    )


def _natural_log_odds(residual_probability: float) -> float:
    if residual_probability == 0.5:
        return 0.0
    if residual_probability >= 0.25:
        ratio = (1.0 - 2.0 * residual_probability) / residual_probability
        return math.log1p(ratio)
    log_survival = math.log1p(-residual_probability)
    log_probability = math.log(residual_probability)
    return log_survival - log_probability


def _quantize_weight_ticks(weight: float, weight_step: float) -> int:
    """The edge length in weight ticks: round(weight / weight_step), at least 1.

    Exact in rational arithmetic, so equal weights get equal lengths on
    every platform (Huang, Newman and Brown 2004.04693: integer edge
    lengths from the log-odds).
    """
    if weight == 0.0:
        return 0
    weight_numerator, weight_denominator = weight.as_integer_ratio()
    step_numerator, step_denominator = weight_step.as_integer_ratio()
    numerator = weight_numerator * step_denominator
    denominator = weight_denominator * step_numerator
    whole, remainder = divmod(numerator, denominator)
    rounds_up = 2 * remainder >= denominator
    ticks = whole + int(rounds_up)
    return max(1, ticks)


def _check_priors(raw_priors, fault_count: int, location: str) -> None:
    if raw_priors.ndim != 1 or raw_priors.size != fault_count:
        raise ValueError(
            f"{location} priors must have one entry per fault column"
        )
    is_finite = numpy.isfinite(raw_priors)
    if not numpy.all(is_finite):
        raise ValueError(f"{location} priors must be finite")
    at_least_zero = raw_priors >= 0.0
    at_most_one = raw_priors <= 1.0
    inside = at_least_zero & at_most_one
    if not numpy.all(inside):
        raise ValueError(f"{location} priors must lie in [0, 1]")


def _parity_product(matrix, vector):
    """The matrix times the vector over GF(2), as a flat array."""
    matrix_integers = matrix.astype(numpy.int64)
    vector_integers = vector.astype(numpy.int64)
    product = matrix_integers @ vector_integers
    product = numpy.asarray(product)
    flat = product.ravel()
    return flat % 2


def _edge_of(
    check, priors, baseline, observables, fault_index: int, weight_step: float
) -> Optional[evidence_records.UnionFindEdge]:
    """The edge of one fault column; None when its residual is certain."""
    probability = float(priors[fault_index])
    residual_probability = probability
    if baseline[fault_index]:
        residual_probability = 1.0 - probability
    if residual_probability == 0.0:
        return None
    detector_a, detector_b = _endpoints(check, fault_index)
    log_odds = _natural_log_odds(residual_probability)
    weight_ticks = _quantize_weight_ticks(log_odds, weight_step)
    column = observables[:, fault_index]
    logical_observables = _int_tuple(column)
    length_half_ticks = 2 * weight_ticks
    return evidence_records.UnionFindEdge(
        fault_index=fault_index,
        detector_a=detector_a,
        detector_b=detector_b,
        logical_observables=logical_observables,
        length_half_ticks=length_half_ticks,
    )


def _endpoints(check, fault_index: int) -> tuple:
    """The two detectors of a column; a missing one is the boundary."""
    start = check.indptr[fault_index]
    end = check.indptr[fault_index + 1]
    detectors = _int_tuple(check.indices[start:end])
    if len(detectors) == 0:
        return BOUNDARY, BOUNDARY
    if len(detectors) == 1:
        return detectors[0], BOUNDARY
    detector_a, detector_b = detectors
    return detector_a, detector_b


def _logical_columns(observables, fault_count: int) -> tuple:
    """Each fault's logical-observable column as a tuple of bits."""
    columns = []
    for fault_index in range(fault_count):
        column = observables[:, fault_index]
        bits = _int_tuple(column)
        columns.append(bits)
    return tuple(columns)


def _int_tuple(values) -> tuple:
    integers = []
    for value in values:
        integers.append(int(value))
    return tuple(integers)


def _fault_indices(
    graph: evidence_records.UnionFindGraph, edge_indices
) -> tuple:
    fault_indices = []
    for edge_index in edge_indices:
        fault_indices.append(graph.edges[edge_index].fault_index)
    return tuple(fault_indices)


def _checked_syndrome(graph: evidence_records.UnionFindGraph, syndrome):
    raw_syndrome = numpy.asarray(syndrome)
    if raw_syndrome.ndim != 1 or raw_syndrome.size != graph.detector_count:
        raise ValueError(
            "Union-Find syndrome must be a one-dimensional detector vector "
            f"of length {graph.detector_count}"
        )
    is_zero = raw_syndrome == 0
    is_one = raw_syndrome == 1
    is_bit = is_zero | is_one
    if not numpy.all(is_bit):
        raise ValueError("Union-Find syndrome must contain only binary values")
    return raw_syndrome.astype(numpy.uint8, copy=False)


def _selected_faults(
    graph: evidence_records.UnionFindGraph, edge_indices: tuple
) -> list:
    selected = list(graph.baseline_faults)
    for edge_index in edge_indices:
        fault_index = graph.edges[edge_index].fault_index
        selected[fault_index] ^= 1
    return selected


def _reproduced_syndrome(
    graph: evidence_records.UnionFindGraph, baseline_syndrome, edge_indices
):
    reproduced = baseline_syndrome.copy()
    for edge_index in edge_indices:
        edge = graph.edges[edge_index]
        if edge.detector_a != BOUNDARY:
            reproduced[edge.detector_a] ^= 1
        if edge.detector_b != BOUNDARY:
            reproduced[edge.detector_b] ^= 1
    return reproduced


def _logical_observables(
    graph: evidence_records.UnionFindGraph, selected_faults: list
) -> tuple:
    parities = [0] * graph.logical_observable_count
    for fault_index in range(graph.fault_count):
        if not selected_faults[fault_index]:
            continue
        column = graph.logical_observables_by_fault[fault_index]
        for logical_index, flips in enumerate(column):
            parities[logical_index] ^= flips
    return tuple(parities)
