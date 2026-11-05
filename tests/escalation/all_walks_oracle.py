"""The Python all_walks walk decsim ran before it ran in C.

The row under test is the all_walks doubt of
decsim/escalation/doubtful_box.py, whose walks
decsim/decoders/union_find/cluster_gap.c now marks, and this module is
the walk it replaced, moved out of doubtful_box.py unchanged in logic
and in name. Its searches pop ties in (length, node, parity) order,
the order the compiled walk keeps when it marks every walk.
"""

import heapq
from typing import Optional, Union

import numpy

import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.records.decoder_evidence as evidence_records


def all_walk_rows(
    evidence: evidence_records.UnionFindHardEvidence,
    roots: numpy.ndarray,
    event_rows: numpy.ndarray,
    limit_half_ticks: float,
) -> tuple:
    """The events of the clusters that odd closed walks under the limit reach.

    The walks are the cluster gap's (union_find/cluster_gap.c): each
    edge a chain of segments split at its interval's ticks, a covered
    segment free and an uncovered one its length, the logical parity on
    the first segment. From each node an odd segment leaves, a Dijkstra
    on the parity-doubled nodes finds the shortest odd closed walk
    through it (odd_walks); each walk shorter than the limit marks the
    clusters with a detection event that its edges reach, as
    cluster_record does for the shortest walk alone.
    """
    event_roots = {int(root) for root in roots[event_rows]}
    found = set()
    walks = odd_walks(evidence.graph, evidence.edge_intervals)
    for length, edges in walks:
        if length >= limit_half_ticks:
            continue
        reached = _walk_roots(evidence.graph, edges, roots)
        doubtful = reached & event_roots
        found.update(doubtful)
    rows = [int(row) for row in event_rows if int(roots[row]) in found]
    return tuple(sorted(rows))


def odd_walks(
    graph: evidence_records.UnionFindGraph, edge_intervals: tuple
) -> list:
    """(length, edges) of the shortest odd closed walk from each source.

    A source is the node an odd segment leaves. The shortest of the
    walks is the cluster gap, in half ticks.
    """
    adjacency, sources = _segment_graph(graph, edge_intervals)
    walks = []
    for source in sources:
        found = _odd_walk_from(adjacency, source)
        if found is not None:
            walks.append(found)
    return walks


def _walk_roots(
    graph: evidence_records.UnionFindGraph, edges: tuple, roots: numpy.ndarray
) -> set:
    """The clusters the walk's edges reach; the boundary is none."""
    reached = set()
    for edge_index in edges:
        edge = graph.edges[edge_index]
        ends = (edge.detector_a, edge.detector_b)
        detectors = [end for end in ends if end != window_decoder.BOUNDARY]
        for end in detectors:
            reached.add(int(roots[end]))
    return reached


def _segment_graph(
    graph: evidence_records.UnionFindGraph, edge_intervals: tuple
) -> tuple:
    """(adjacency, sources): the cluster gap's segment graph.

    adjacency maps a node to its (neighbour, weight, parity, edge)
    slots, in the order the segments are laid; sources holds each node
    an odd segment leaves, in that order too.
    """
    adjacency: dict = {}
    sources: dict = {}
    segments = _segments(graph, edge_intervals)
    for left, right, weight, parity, edge_index in segments:
        left_slots = adjacency.setdefault(left, [])
        left_slots.append((right, weight, parity, edge_index))
        right_slots = adjacency.setdefault(right, [])
        right_slots.append((left, weight, parity, edge_index))
        if parity:
            sources[left] = True
    return adjacency, sources


def _segments(
    graph: evidence_records.UnionFindGraph, edge_intervals: tuple
) -> list:
    """(left, right, weight, parity, edge) of every segment, edge by edge.

    The boundary is one node, numbered detector_count, and an edge's
    inner nodes are numbered past it, edge by edge.
    """
    endpoint_a, endpoint_b, lengths = graph.edge_arrays
    parities = graph.logical_parities
    boundary = graph.detector_count
    next_node = boundary + 1
    segments = []
    for edge_index, interval in enumerate(edge_intervals):
        length = int(lengths[edge_index])
        cuts = _segment_cuts(interval, length)
        first = _node(int(endpoint_a[edge_index]), boundary)
        last = _node(int(endpoint_b[edge_index]), boundary)
        inner_count = len(cuts) - 2
        inner_end = next_node + inner_count
        inner = range(next_node, inner_end)
        chain = [first, *inner, last]
        next_node = inner_end
        parity = int(parities[edge_index])
        edge_segments = _edge_segments(chain, cuts, interval, parity)
        for left, right, weight, segment_parity in edge_segments:
            segments.append((left, right, weight, segment_parity, edge_index))
    return segments


def _segment_cuts(
    interval: Union[evidence_records.Open, evidence_records.Closed], length: int
) -> list:
    """The ticks an edge splits at: its ends, and an open interval's two."""
    if isinstance(interval, evidence_records.Closed):
        return [0, length]
    ticks = {0, int(interval.lower_tick), int(interval.upper_tick), length}
    return sorted(ticks)


def _node(end: int, boundary: int) -> int:
    """A detector's own node; the boundary's one node for the boundary."""
    if end == window_decoder.BOUNDARY:
        return boundary
    return end


def _edge_segments(
    chain: list,
    cuts: list,
    interval: Union[evidence_records.Open, evidence_records.Closed],
    parity: int,
) -> list:
    """(left, right, weight, parity) of each segment along one edge.

    An uncovered segment weighs its length and a covered one nothing;
    the edge's logical parity rides its first segment.
    """
    segments = []
    segment_count = len(cuts) - 1
    for segment in range(segment_count):
        lower = cuts[segment]
        upper = cuts[segment + 1]
        weight = _segment_weight(interval, lower, upper)
        segment_parity = 0
        if segment == 0:
            segment_parity = parity
        left = chain[segment]
        right = chain[segment + 1]
        segments.append((left, right, weight, segment_parity))
    return segments


def _segment_weight(
    interval: Union[evidence_records.Open, evidence_records.Closed],
    lower: int,
    upper: int,
) -> int:
    """The segment's length when the growth left it uncovered; else zero."""
    if isinstance(interval, evidence_records.Closed):
        return 0
    is_uncovered = lower < interval.upper_tick and upper > interval.lower_tick
    if is_uncovered:
        return upper - lower
    return 0


def _odd_walk_from(adjacency: dict, source: int) -> Optional[tuple]:
    """(length, edges) of the shortest walk from (source, 0) to (source, 1).

    The second number is the walk's parity so far: Dijkstra's search on
    the parity-doubled nodes, as cluster_gap.c runs it.
    """
    start = (source, 0)
    goal = (source, 1)
    distance = {start: 0}
    previous = {}
    heap = [(0, source, 0)]
    while heap:
        length, node, parity = heapq.heappop(heap)
        state = (node, parity)
        if length > distance[state]:
            continue
        if state == goal:
            edges = _walk_edges(previous, start, goal)
            return length, edges
        slots = adjacency.get(node, ())
        _relax_neighbours(slots, state, length, distance, previous, heap)
    return None


def _relax_neighbours(
    slots: tuple,
    state: tuple,
    length: int,
    distance: dict,
    previous: dict,
    heap: list,
) -> None:
    """Push each neighbour state this state reaches more cheaply."""
    _node_index, parity = state
    for neighbour, weight, edge_parity, edge_index in slots:
        following_parity = parity ^ edge_parity
        following = (neighbour, following_parity)
        candidate = length + weight
        is_shorter = (
            following not in distance or candidate < distance[following]
        )
        if not is_shorter:
            continue
        distance[following] = candidate
        previous[following] = (state, edge_index)
        heapq.heappush(heap, (candidate, neighbour, following_parity))


def _walk_edges(previous: dict, start: tuple, goal: tuple) -> tuple:
    """The edges of the walk read back from the goal, ascending."""
    edges = set()
    state = goal
    while state != start:
        state, edge_index = previous[state]
        edges.add(edge_index)
    return tuple(sorted(edges))
