"""The Python cluster gap walk decsim ran before it ran in C.

The row under test is decsim/decoders/union_find/cluster_gap.c through
decsim/decoders/union_find/compiled_decoder.py, and this module is the
walk it replaced, moved out of decsim/confidence/cluster.py unchanged in
logic and in name. It is one of the two oracles the compiled walk is
held against; the other, independent_cluster_gap.py, is an independent
reading of Meister et al. 2405.07433 that runs a different algorithm.
"""

import heapq
import itertools
import math
from typing import Union

import decsim.records.decoder_evidence as evidence_records


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
