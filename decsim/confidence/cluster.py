"""The cluster gap: the confidence of one weighted Union-Find window decode.

Meister et al. 2405.07433 Definition 9 and Algorithm 2 (the PDF under
the sandbox tmp/papers): the weighted edge intervals the hard decode
grew are quotiented into a graph whose shortest closed walk of odd
logical parity is the gap, in the growth's half-tick units and reported
in decibels. The hard decode is Delfosse and Nickerson 1709.06218
(union_find/window_decoder.py); UnionFindClusterGapDecoder is a Decoder
over it, built in Python and given to a tier as its decoder, beside the
confidence wrappers (decoder.py). The gap reads the hard decode's
intervals, not the syndrome, so it cannot be a ConfidenceSignal on the
port, and it is not a tier kind of the root's table either: it reports
its own soft output, in decibels at its weight step, where the
switching policy decides on the complementary gap in nats, and no yaml
key selects another signal yet.

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

import numpy

import decsim.decoders.decoder as decoder_module
import decsim.decoders.union_find.decoder as union_find_decoder
import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoding as decoding_records

_BINARY64_LN_TEN = float.fromhex("0x1.26bb1bbb55516p+1")


def union_find_cluster_gap_source(
    weight_step: float = 0.1,
) -> decoding_records.SoftOutputSource:
    """The source of a cluster gap at one absolute natural-log weight step."""
    normalized_step = window_decoder.normalized_weight_step(weight_step)
    return decoding_records.SoftOutputSource(
        method="cluster_gap",
        cluster_origin="union_find_decoder",
        growth_schedule="weighted_global_fair",
        gap_units="decibels",
        correction="none",
        weight_step_natural_log=normalized_step,
        references=("cluster-gap method",),
    )


class UnionFindClusterGapDecoder(decoder_module.DecoderBase):
    """The hard Union-Find decode with its one-logical cluster gap attached.

    Latency, occupancy, pipeline depth and cancel are the base decoder's,
    as in the confidence wrappers (decoder.py); on the measured path the
    unit's time is the base's growth-and-peeling call plus the timed gap
    walk, never the base's untimed setup. The gap reads the immutable
    intervals of the same hard decode.
    """

    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED

    def __init__(self, base: union_find_decoder.UnionFindDecoder) -> None:
        if not isinstance(base, union_find_decoder.UnionFindDecoder):
            raise TypeError(
                "UnionFindClusterGapDecoder requires a UnionFindDecoder base"
            )
        self.base = base

    def run_seed_children(self) -> tuple:
        """The hard decoder's seed paths; the gap draws nothing."""
        return self.base.run_seed_children()

    def latency(self, job: decoding_records.DecodeJob) -> int:
        """The base decoder's timing; the gap adds no latency."""
        return self.base.latency(job)

    def occupancy(self, job: decoding_records.DecodeJob) -> Optional[int]:
        """The base decoder's occupancy; None when it is measured."""
        return self.base.occupancy(job)

    def pipeline_depth(self, job: decoding_records.DecodeJob) -> int:
        """The base decoder's pipeline depth."""
        return self.base.pipeline_depth(job)

    def cancel(self, job: decoding_records.DecodeJob) -> None:
        """Stop the base decoder's job."""
        self.base.cancel(job)

    def decode(
        self, job: decoding_records.DecodeJob
    ) -> decoding_records.DecodeResult:
        """Decode once, compute the cluster gap, attach the confidence."""
        result, _elapsed_nanoseconds = self.decode_timed(job)
        return result

    def decode_timed(self, job: decoding_records.DecodeJob) -> tuple:
        """(the result with its confidence, nanoseconds of decode and gap)."""
        model = job.dem
        if model is None:
            return self.base.decode_timed(job)
        _require_one_logical_row(model)
        decoded_window = self.base.decode_with_growth_evidence(job)
        started = time.perf_counter_ns()
        gap = _cluster_gap(decoded_window.hard_evidence, self.base.weight_step)
        finished = time.perf_counter_ns()
        source = union_find_cluster_gap_source(self.base.weight_step)
        decoded_window.hard_result.soft_output = decoding_records.SoftOutput(
            gap=gap, source=source
        )
        gap_nanoseconds = finished - started
        return (
            decoded_window.hard_result,
            decoded_window.backend_ns + gap_nanoseconds,
        )


def _require_one_logical_row(model) -> None:
    """The gap is defined for exactly one nonzero logical-observable row."""
    faults = model.require_faults(fault_models.FaultRepresentation.GRAPHLIKE)
    observables = faults.observables.toarray()
    row_count = observables.shape[0]
    if row_count != 1:
        raise ValueError(
            "Union-Find cluster confidence requires exactly one logical "
            f"observable, got {row_count}"
        )
    if not numpy.any(observables[0]):
        raise ValueError(
            "Union-Find cluster confidence requires one nonzero logical "
            "observable row"
        )


def _cluster_gap(
    hard_evidence: window_decoder.UnionFindHardEvidence, weight_step: float
) -> float:
    gap_half_ticks = _quotient_cluster_gap(
        hard_evidence.graph, hard_evidence.edge_intervals
    )
    return _gap_half_ticks_to_decibels(gap_half_ticks, weight_step)


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


def _gap_half_ticks_to_decibels(
    gap_half_ticks: Union[int, float], weight_step: float
) -> float:
    """Half ticks of the growth as decibels, exactly, then rounded once."""
    if gap_half_ticks == math.inf:
        return math.inf
    half_ticks = Fraction(gap_half_ticks, 2)
    step = Fraction.from_float(weight_step)
    ln_ten = Fraction.from_float(_BINARY64_LN_TEN)
    nats = half_ticks * step
    scaled = nats * 10
    exact_gap_decibels = scaled / ln_ten
    try:
        return float(exact_gap_decibels)
    except OverflowError:
        return math.inf
