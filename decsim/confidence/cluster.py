"""The cluster gap: the confidence of one weighted Union-Find window decode.

Meister et al. 2405.07433 Definition 9 and Algorithm 2 (lines 518-536):
the weighted edge intervals the hard decode grew are quotiented into a
graph whose shortest closed walk of odd logical parity is the gap, in
the growth's half-tick units, reported as natural-log weight, which is
the unit the switching threshold is held in. The hard decode is Delfosse
and Nickerson 1709.06218 (union_find/window_decoder.py), and this row
reads the growth that decode already returned on its result
(DecodeResult.cluster_evidence), so the confidence is the decoder's own
and no second decode is run. The signal needs one ordinary decode of the
window, not a forced-class pair, so its forced_logical_classes is empty.

The exact likelihood-ratio reading of the gap holds only in the uniform
repetition-code setting of Meister's Theorem 10; on a surface code it is
a confidence, not a calibrated failure probability. The gap needs one
nonzero logical-observable row, and thresholds are calibrated per
weight step.
"""

import heapq
import itertools
import math
from fractions import Fraction
from typing import Optional, Union

import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoding as decoding_records


def union_find_cluster_gap_source(
    weight_step: float = window_decoder.DEFAULT_WEIGHT_STEP,
) -> decoding_records.SoftOutputSource:
    """The source of a cluster gap at one absolute natural-log weight step."""
    normalized_step = window_decoder.normalized_weight_step(weight_step)
    return decoding_records.SoftOutputSource(
        method="cluster_gap",
        cluster_origin="union_find_decoder",
        growth_schedule="weighted_global_fair",
        gap_units="log_likelihood_weight",
        correction="none",
        weight_step_natural_log=normalized_step,
        references=("cluster-gap method",),
    )


class ClusterGap:
    """The signal row: the gap of one cluster-based decode's own growth."""

    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    decoder_evidence_requirement = decoding_records.CLUSTER_GROWTH_EVIDENCE
    # the window is decoded once; the gap is read off that decode
    forced_logical_classes = ()

    def __init__(
        self, weight_step: float = window_decoder.DEFAULT_WEIGHT_STEP
    ) -> None:
        self.weight_step = window_decoder.normalized_weight_step(weight_step)
        self.source = union_find_cluster_gap_source(self.weight_step)

    def soft_output_for(
        self, solves: tuple
    ) -> Optional[decoding_records.SoftOutput]:
        """The gap of the window's one decode, in natural-log weight.

        None when the decode carried no growth: a job without a window
        model runs no growth, and the escalation policy escalates a
        window without a soft output (escalation/policies.py).
        """
        evidence = solves[0].cluster_evidence
        if evidence is None:
            return None
        graph = evidence.graph
        _require_one_logical_row(graph)
        _require_weight_step(graph, self.weight_step)
        gap = _cluster_gap(evidence, self.weight_step)
        return decoding_records.SoftOutput(gap=gap, source=self.source)


def _require_one_logical_row(graph) -> None:
    """The gap is defined for exactly one nonzero logical-observable row."""
    row_count = graph.logical_observable_count
    if row_count != 1:
        raise ValueError(
            "Union-Find cluster confidence requires exactly one logical "
            f"observable, got {row_count}"
        )
    for edge in graph.edges:
        if edge.logical_observables[0]:
            return
    raise ValueError(
        "Union-Find cluster confidence requires one nonzero logical "
        "observable row"
    )


def _require_weight_step(graph, weight_step: float) -> None:
    """The gap is read off the ticks the growth actually used."""
    if graph.weight_step == weight_step:
        return
    raise RuntimeError(
        "the cluster gap reads the decode's own ticks: the decoder grew "
        f"at weight step {graph.weight_step} and the signal reports "
        f"{weight_step}"
    )


def _cluster_gap(
    hard_evidence: window_decoder.UnionFindHardEvidence, weight_step: float
) -> float:
    gap_half_ticks = _quotient_cluster_gap(
        hard_evidence.graph, hard_evidence.edge_intervals
    )
    return _gap_half_ticks_to_natural_log_weight(gap_half_ticks, weight_step)


def _quotient_cluster_gap(
    graph: window_decoder.UnionFindGraph, edge_intervals: tuple
) -> Union[int, float]:
    """The shortest odd-logical quotient walk, in integer half ticks.

    Every edge becomes a path of segments split at its interval's
    ticks; a segment the growth covered costs nothing, an uncovered one
    its length, and the first segment carries the edge's logical
    parity. The walk is Dijkstra over (node, parity) from each node
    back to itself with parity one (Meister et al. Algorithm 2).
    """
    adjacency = {}
    edges = zip(graph.edges, edge_intervals)
    for edge_index, (edge, interval) in enumerate(edges):
        _add_edge_segments(adjacency, edge_index, edge, interval)
    shortest = math.inf
    sequence = itertools.count()
    for reference_node in adjacency:
        shortest = _shortest_odd_walk_from(
            adjacency, reference_node, shortest, sequence
        )
    return shortest


def _add_edge_segments(
    adjacency: dict, edge_index: int, edge, interval
) -> None:
    coordinates = _split_coordinates(edge, interval)
    path_nodes = [edge.detector_a]
    split_count = len(coordinates) - 1
    for split_index in range(1, split_count):
        path_nodes.append(("union_find_edge", edge_index, split_index))
    path_nodes.append(edge.detector_b)
    segments = zip(coordinates, coordinates[1:])
    for segment_index, (lower, upper) in enumerate(segments):
        weight = _segment_weight(interval, lower, upper)
        parity = 0
        if segment_index == 0:
            parity = edge.logical_observables[0]
        next_index = segment_index + 1
        left = path_nodes[segment_index]
        right = path_nodes[next_index]
        _add_segment(adjacency, left, right, weight, parity)


def _split_coordinates(edge, interval) -> tuple:
    """The edge's tick coordinates where its segments meet, ascending."""
    if isinstance(interval, window_decoder.Closed):
        return (0, edge.length_half_ticks)
    ticks = {
        0,
        interval.lower_tick,
        interval.upper_tick,
        edge.length_half_ticks,
    }
    ordered = sorted(ticks)
    return tuple(ordered)


def _segment_weight(interval, lower: int, upper: int) -> int:
    """The segment's cost: zero where the growth covered it."""
    if isinstance(interval, window_decoder.Closed):
        return 0
    if upper <= interval.lower_tick:
        return 0
    if lower >= interval.upper_tick:
        return 0
    return upper - lower


def _add_segment(
    adjacency: dict, left, right, weight: int, parity: int
) -> None:
    left_neighbors = adjacency.setdefault(left, [])
    left_neighbors.append((right, weight, parity))
    right_neighbors = adjacency.setdefault(right, [])
    right_neighbors.append((left, weight, parity))


def _shortest_odd_walk_from(
    adjacency: dict, reference_node, shortest, sequence
) -> Union[int, float]:
    """The shortest closed odd walk from the node, if under `shortest`."""
    source_state = (reference_node, 0)
    target_state = (reference_node, 1)
    distances = {source_state: 0}
    first_order = next(sequence)
    frontier = [(0, first_order, source_state)]
    while frontier:
        distance, _order, state = heapq.heappop(frontier)
        if distance != distances.get(state, math.inf):
            continue
        if distance >= shortest:
            return shortest
        if state == target_state:
            return distance
        _relax_neighbors(
            adjacency, state, distance, distances, frontier, sequence
        )
    return shortest


def _relax_neighbors(
    adjacency: dict,
    state: tuple,
    distance: int,
    distances: dict,
    frontier: list,
    sequence,
) -> None:
    graph_node, logical_parity = state
    for neighbor, weight, edge_parity in adjacency.get(graph_node, ()):
        parity = logical_parity ^ edge_parity
        neighbor_state = (neighbor, parity)
        candidate = distance + weight
        known = distances.get(neighbor_state, math.inf)
        if candidate < known:
            distances[neighbor_state] = candidate
            order = next(sequence)
            heapq.heappush(frontier, (candidate, order, neighbor_state))


def _gap_half_ticks_to_natural_log_weight(
    gap_half_ticks: Union[int, float], weight_step: float
) -> float:
    """Half ticks of the growth as natural-log weight, exactly, rounded once.

    Every signal reports its gap in the units the switching threshold is
    held in (escalation.gap_threshold_db is converted once, at the yaml
    boundary, by decoders/settings.py decibels_to_nats), so a threshold
    means the same thing whichever signal a run names.
    """
    if gap_half_ticks == math.inf:
        return math.inf
    half_ticks = Fraction(gap_half_ticks, 2)
    step = Fraction.from_float(weight_step)
    exact_gap_nats = half_ticks * step
    try:
        return float(exact_gap_nats)
    except OverflowError:
        return math.inf
