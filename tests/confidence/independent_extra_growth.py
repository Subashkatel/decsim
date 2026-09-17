"""An independent reading of Kishi et al. 2602.03336 Algorithm 1.

Written from the paper text (lines 478 to 491 of the arXiv text) and
not from decsim's code: "Increase the radii of all clusters
and boundary nodes by delta_epsilon / 2", "Merge colliding clusters",
"if a single cluster connects boundaries b1 and b2 then return
epsilon". A tick grows every front one half tick, the graph's own unit
(Meister 2405.07433 Algorithm 1 line 6, "r_i <- r_i + w/2"), so an edge
loses two half ticks a tick whichever cluster each end is in, and the
boundaries are joined when the closed edges hold a closed walk of odd
logical parity, which is what a path from b1 to b2 is once the boundary
is split (Kishi lines 244-258).

The join test here is a breadth-first two-colouring of the closed-edge
graph, re-run from scratch after every tick, which is a different
algorithm from decsim's union-find with walk parities: that is the
point, the two must agree on the tick.
"""

import collections
from typing import Optional

import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.records.decoder_evidence as evidence_records


def joined_at_tick(
    graph, edge_intervals: tuple, growth_limit_ticks: int
) -> Optional[int]:
    """The tick the boundaries join at, None when the limit comes first."""
    uncovered = _uncovered_half_ticks(edge_intervals)
    tick_count = growth_limit_ticks + 1
    for tick in range(tick_count):
        closed = _closed_edges(uncovered, tick)
        if _holds_an_odd_closed_walk(graph, closed):
            return tick
    return None


def _uncovered_half_ticks(edge_intervals: tuple) -> list:
    """What is left of every edge when the decode stopped."""
    uncovered = []
    for interval in edge_intervals:
        if isinstance(interval, evidence_records.Closed):
            uncovered.append(0)
            continue
        left = interval.upper_tick - interval.lower_tick
        uncovered.append(left)
    return uncovered


def _closed_edges(uncovered: list, tick: int) -> list:
    """Every edge whose two fronts have met after this many ticks."""
    closed = []
    for edge_index, left in enumerate(uncovered):
        if left <= 2 * tick:
            closed.append(edge_index)
    return closed


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
