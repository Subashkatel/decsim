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

The walk is not free. Algorithm 2 is a Dijkstra over the decode's own
edge intervals, and this Python implementation of it takes milliseconds
where the union-find decode takes hundreds of microseconds, so the row
reports what it cost: measured on the host clock the way a measured
decoder is, or the declared number of a tier that is a card. The
evidence and its reader are the same hardware, so the time is charged on
the unit that produced the growth (decision D8; Toshio et al.
2510.25222 lines 152-160 compute the soft output on the weak decoder).

The exact likelihood-ratio reading of the gap holds only in the uniform
repetition-code setting of Meister's Theorem 10; on a surface code it is
a confidence, not a calibrated failure probability. The gap needs one
nonzero logical-observable row, and thresholds are calibrated per
weight step.
"""

import heapq
import itertools
import math
import time
from fractions import Fraction
from typing import Optional, Union

import decsim.config as config
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoder_evidence as evidence_records
import decsim.records.decoding as decoding_records


def union_find_cluster_gap_source(
    weight_step: float = evidence_records.DEFAULT_WEIGHT_STEP,
) -> decoding_records.SoftOutputSource:
    """The source of a cluster gap at one absolute natural-log weight step."""
    normalized_step = evidence_records.normalized_weight_step(weight_step)
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
    evidence_refusal = (
        "the cluster gap walks the radii a growth left behind, and "
        "PyMatching's API reports no regions, blossoms or radii "
        "(pymatching 2.4.0 Matching), while belief propagation and "
        "search decoders grow no clusters at all; use "
        "weak_decoder.kind union_find, or escalation.confidence "
        "complementary_gap"
    )

    def __init__(
        self,
        weight_step: float = evidence_records.DEFAULT_WEIGHT_STEP,
        walk_microseconds: Optional[float] = None,
    ) -> None:
        self.weight_step = evidence_records.normalized_weight_step(weight_step)
        self.source = union_find_cluster_gap_source(self.weight_step)
        # what the walk costs on the weak tier's clock: a card's declared
        # number, or None to measure the call as a measured decoder is
        self.walk_microseconds = walk_microseconds

    def compute(self, solves: tuple) -> decoding_records.SoftOutputComputation:
        """The gap of the window's one decode, and what the walk cost.

        The gap is in natural-log weight, and None when the decode
        carried no growth: a job without a window model runs no growth,
        and the escalation policy escalates a window without a soft
        output (escalation/policies.py). A window with no growth walks
        nothing and charges nothing.
        """
        evidence = solves[0].cluster_evidence
        if evidence is None:
            return decoding_records.SoftOutputComputation(None, 0)
        graph = evidence.graph
        _require_one_logical_row(graph)
        _require_weight_step(graph, self.weight_step)
        gap, ticks = self._walk(evidence)
        soft_output = decoding_records.SoftOutput(gap=gap, source=self.source)
        return decoding_records.SoftOutputComputation(soft_output, ticks)

    def _walk(self, evidence) -> tuple:
        """The gap and the ticks the walk cost, declared or measured."""
        if self.walk_microseconds is not None:
            gap = _cluster_gap(evidence, self.weight_step)
            ticks = config.microseconds_to_ticks(self.walk_microseconds)
            return gap, ticks
        started_ns = time.perf_counter_ns()
        gap = _cluster_gap(evidence, self.weight_step)
        finished_ns = time.perf_counter_ns()
        elapsed_ns = finished_ns - started_ns
        elapsed_microseconds = elapsed_ns / 1000.0
        ticks = config.microseconds_to_ticks(elapsed_microseconds)
        return gap, ticks


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
    hard_evidence: evidence_records.UnionFindHardEvidence, weight_step: float
) -> float:
    gap_half_ticks = _quotient_cluster_gap(
        hard_evidence.graph, hard_evidence.edge_intervals
    )
    return _gap_half_ticks_to_natural_log_weight(gap_half_ticks, weight_step)


def _quotient_cluster_gap(
    graph: evidence_records.UnionFindGraph, edge_intervals: tuple
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
    if isinstance(interval, evidence_records.Closed):
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
    if isinstance(interval, evidence_records.Closed):
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
    boundary, by escalation/settings.py decibels_to_nats), so a threshold
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
