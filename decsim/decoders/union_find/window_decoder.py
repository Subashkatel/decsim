"""Prior-weighted graphlike Union-Find: the graph and one decode on it.

Delfosse and Nickerson 1709.06218: every odd cluster grows by one
half-edge per round (Algorithm 1, step 4), clusters that meet fuse, and
the peeling decoder reads the correction off a spanning forest of the
grown erasure (step 8). Edge lengths follow Huang, Newman and Brown
2004.04693: an edge of log-odds weight w has the integer length round(w
/ weight_step), at least one tick when w is not zero, so a likelier
fault is crossed sooner. A fault of probability one half has length
zero, an edge the growth starts closed.

The growth, the forest and the peeling run in the compiled decoder
(compiled_decoder.py); the graph and the evidence a decode leaves behind
are records in decsim/records/decoder_evidence.py.
"""

import math
from typing import Optional

import numpy

import decsim.decoders.decoder as decoder_module
import decsim.decoders.union_find.compiled_decoder as compiled_decoder
import decsim.detector_error_model.basis_split as basis_split
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoder_evidence as evidence_records

BOUNDARY = -1
# the compiled decoder counts lengths, ticks and walk distances in int64
_LARGEST_HALF_TICK_COUNT = 2**63 - 1


def graph_from_model(
    faults: fault_models.PlacedFaultModel,
    *,
    location: str,
    weight_step: float,
) -> evidence_records.UnionFindGraph:
    """The weighted graph of one placed graphlike model.

    Faults likelier than one half are the baseline correction; every
    other column becomes an edge whose length is its residual log-odds
    in weight ticks.
    """
    check = faults.check
    raw_priors = numpy.asarray(faults.priors)
    fault_count = check.shape[1]
    _check_priors(raw_priors, location)
    # observables are few rows; dense per-fault columns are cheap to read
    observables = faults.observables.toarray()
    observables = observables.astype(numpy.uint8, copy=False)
    priors = raw_priors.astype(float, copy=False)
    likely = priors > 0.5
    baseline = likely.astype(numpy.uint8)
    baseline_syndrome = decoder_module.parity_product(check, baseline)
    edges = _edges(check, priors, baseline, observables, weight_step)
    _refuse_lengths_past_the_counters(edges, weight_step, location)
    logical_columns = _logical_columns(observables, fault_count)
    baseline_faults = decoder_module.int_tuple(baseline)
    baseline_syndrome = decoder_module.int_tuple(baseline_syndrome)
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
    graph: evidence_records.UnionFindGraph, syndrome: numpy.ndarray
) -> evidence_records.UnionFindHardEvidence:
    """Decode one syndrome on the graph: grow, take the forest, peel, report."""
    syndrome_array = _checked_syndrome(graph, syndrome)
    baseline_syndrome = numpy.asarray(
        graph.baseline_syndrome, dtype=numpy.uint8
    )
    syndrome_bits = decoder_module.int_tuple(syndrome_array)
    residual_syndrome = syndrome_array ^ baseline_syndrome
    residual_bits = decoder_module.int_tuple(residual_syndrome)
    outcome = compiled_decoder.decode(graph, residual_syndrome)
    selected_edges = outcome.selected_edges
    selected_faults = _selected_faults(graph, selected_edges)
    reproduced = _reproduced_syndrome(graph, baseline_syndrome, selected_edges)
    mismatch = reproduced ^ syndrome_array
    unmatched = numpy.nonzero(mismatch)
    unmatched_detectors = decoder_module.int_tuple(unmatched[0])
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
        growth_steps=outcome.growth_steps,
        forest_depth=outcome.forest_depth,
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
    """The edge length in weight ticks: round(weight / weight_step).

    At least one tick for a nonzero weight, and zero for weight zero,
    which is a fault of probability one half.

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


def _edges(check, priors, baseline, observables, weight_step: float) -> list:
    """The edge of every fault column whose residual is not certain."""
    edges = []
    fault_count = check.shape[1]
    for fault_index in range(fault_count):
        edge = _edge_of(
            check, priors, baseline, observables, fault_index, weight_step
        )
        if edge is not None:
            edges.append(edge)
    return edges


def _refuse_lengths_past_the_counters(
    edges: list, weight_step: float, location: str
) -> None:
    """A weight_step so fine that the compiled sums could wrap is refused.

    The cluster gap's shortest odd walk crosses each edge at most once in
    each parity layer and relaxes one edge past that, so three times the
    edges' total length is an upper bound on every sum the compiled
    decoder forms; a signed 64-bit sum past its largest value wraps
    negative without a word (C11 6.5p5 leaves it undefined).
    """
    total_half_ticks = 0
    for edge in edges:
        total_half_ticks += edge.length_half_ticks
    if 3 * total_half_ticks <= _LARGEST_HALF_TICK_COUNT:
        return
    raise ValueError(
        f"{location}: at weight_step {weight_step} the graph's edges are "
        f"{total_half_ticks} half ticks long together, and the compiled "
        "decoder's 64-bit sums can reach up to three times that; raise "
        "the decoder's weight_step"
    )


def _check_priors(raw_priors, location: str) -> None:
    is_finite = numpy.isfinite(raw_priors)
    if not numpy.all(is_finite):
        raise ValueError(f"{location} priors must be finite")
    at_least_zero = raw_priors >= 0.0
    at_most_one = raw_priors <= 1.0
    inside = at_least_zero & at_most_one
    if not numpy.all(inside):
        raise ValueError(f"{location} priors must lie in [0, 1]")


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
    logical_observables = decoder_module.int_tuple(column)
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
    detectors = basis_split.column_rows(check, fault_index)
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
        bits = decoder_module.int_tuple(column)
        columns.append(bits)
    return tuple(columns)


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
