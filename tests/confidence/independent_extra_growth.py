"""An independent reading of Kishi et al. 2602.03336 Algorithm 1.

Written from the paper text (lines 478 to 491 of the arXiv text) and
not from decsim's code: "Increase the radii of all clusters
and boundary nodes by delta_epsilon / 2", "Merge colliding clusters",
"if a single cluster connects boundaries b1 and b2 then return
epsilon". The clusters are the ones the decode left (lines 430 to 434,
"the resulting clusters are grown additionally"), so a front advances
only from a node that stands in a cluster holding a defect, or in the
boundary's; a node no defect reached is passed over once a front
covers the edge to it, and grows from then on. That is what makes
Definition 3 (lines 500 to 507) sum the edges between two consecutive
clusters: the ball rolls over the bare nodes between them. A tick grows
every such front one half tick, the graph's own unit (Meister
2405.07433 Algorithm 1 line 6, "r_i <- r_i + w/2"), and the boundaries
are joined when the closed edges hold a closed walk of odd logical
parity, which is what a path from b1 to b2 is once the boundary is
split (Kishi lines 244-258).

The join test here is a breadth-first two-colouring of the closed-edge
graph, re-run from scratch after every tick, and the cluster
membership a flood over the closed edges, which is a different
algorithm from decsim's union-find with walk parities: that is the
point, the two must agree on the tick.
"""

import collections
from typing import Optional

import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.records.decoder_evidence as evidence_records


def joined_at_tick(
    graph, edge_intervals: tuple, residual_syndrome, growth_limit_ticks: int
) -> Optional[int]:
    """The tick the boundaries join at, None when the limit comes first."""
    boundary = graph.detector_count
    fronts = _fronts(edge_intervals)
    seeds = {boundary}
    for detector, bit in enumerate(residual_syndrome):
        if bit:
            seeds.add(detector)
    for tick in range(growth_limit_ticks + 1):
        closed = _closed_edges(fronts)
        if _holds_an_odd_closed_walk(graph, closed):
            return tick
        if tick == growth_limit_ticks:
            break
        growing = _nodes_in_growing_clusters(graph, closed, seeds)
        _advance(graph, fronts, growing)
    return None


def _fronts(edge_intervals: tuple) -> list:
    """[lower, upper] of every edge when the decode stopped; None is closed."""
    fronts = []
    for interval in edge_intervals:
        if isinstance(interval, evidence_records.Closed):
            fronts.append(None)
            continue
        fronts.append([interval.lower_tick, interval.upper_tick])
    return fronts


def _closed_edges(fronts: list) -> list:
    """Every edge whose two fronts have met."""
    closed = []
    for edge_index, front in enumerate(fronts):
        if front is None or front[0] >= front[1]:
            closed.append(edge_index)
    return closed


def _nodes_in_growing_clusters(graph, closed: list, seeds: set) -> set:
    """Every node a closed-edge walk joins to a defect or to the boundary."""
    boundary = graph.detector_count
    neighbours = collections.defaultdict(list)
    for edge_index in closed:
        edge = graph.edges[edge_index]
        node_a = _node(edge.detector_a, boundary)
        node_b = _node(edge.detector_b, boundary)
        neighbours[node_a].append(node_b)
        neighbours[node_b].append(node_a)
    growing = set()
    queue = collections.deque(seeds)
    while queue:
        node = queue.popleft()
        if node in growing:
            continue
        growing.add(node)
        queue.extend(neighbours[node])
    return growing


def _advance(graph, fronts: list, growing: set) -> None:
    """One tick: every front on a growing node moves one half tick."""
    boundary = graph.detector_count
    for edge_index, front in enumerate(fronts):
        if front is None or front[0] >= front[1]:
            continue
        edge = graph.edges[edge_index]
        if _node(edge.detector_a, boundary) in growing:
            front[0] += 1
        if _node(edge.detector_b, boundary) in growing:
            front[1] -= 1


def _holds_an_odd_closed_walk(graph, closed: list) -> bool:
    """Whether the closed edges fail to two-colour by logical parity."""
    boundary = graph.detector_count
    neighbours = collections.defaultdict(list)
    for edge_index in closed:
        edge = graph.edges[edge_index]
        node_a = _node(edge.detector_a, boundary)
        node_b = _node(edge.detector_b, boundary)
        parity = edge.logical_observables[0]
        neighbours[node_a].append((node_b, parity))
        neighbours[node_b].append((node_a, parity))
    colour = {}
    for start in list(neighbours):
        if start in colour:
            continue
        if _odd_from(start, neighbours, colour):
            return True
    return False


def _odd_from(start, neighbours, colour: dict) -> bool:
    """Colour one component from start; True when a colour conflicts."""
    colour[start] = 0
    queue = collections.deque([start])
    while queue:
        node = queue.popleft()
        if _colours_conflict(node, neighbours, colour, queue):
            return True
    return False


def _colours_conflict(node, neighbours, colour: dict, queue) -> bool:
    """Colour node's neighbours; True when one wears the other colour."""
    for other, parity in neighbours[node]:
        wanted = colour[node] ^ parity
        if other not in colour:
            colour[other] = wanted
            queue.append(other)
            continue
        if colour[other] != wanted:
            return True
    return False


def _node(detector: int, boundary: int) -> int:
    if detector == window_decoder.BOUNDARY:
        return boundary
    return detector
