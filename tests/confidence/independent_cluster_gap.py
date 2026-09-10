"""An independent reading of Meister et al. 2405.07433 Definition 9.

Written for round V1 from the paper text alone, not from decsim's code.

Definition 1 (line 209 of tmp/papers/txt/2405.07433.txt): the cluster set
is the union of balls B_{r_v}(v) in the metric space X_G, so a ball may
cover part of an edge, not only whole edges.

Definition 9 (lines 511-520): "Define a new metric space (X'_GD, d'_GD)
that is the quotient of (X_GD, d_GD) by each C_i i.e. identify each C_i
with a point. Define phi(C) to be the length of the shortest path that
covers a logical operator in X'_GD."

Algorithm 1 line 6 (line 447): "r_i <- r_i + w/2", so a growing front
advances half a weight step at a time; decsim's edge lengths and this
module's coordinates are counted in those half ticks.

The realization used here: every edge of length L half ticks becomes a
chain of segments split at the growth's coordinates; a segment's cost is
the measure of its intersection with the uncovered part of the edge, so a
covered stretch costs nothing (which is the quotient, since the covered
stretches of all edges around one vertex meet at that vertex at cost
zero). "Covers a logical operator" is a closed walk of odd logical
parity, so the search runs on the parity-doubled node set and the answer
is min over nodes v of d((v, 0), (v, 1)).

The shortest paths come from scipy.sparse.csgraph.dijkstra over the whole
doubled graph from every (v, 0) source at once. That is a different
algorithm from decsim's per-node Dijkstra with a running cutoff, which is
the point: the two must agree.
"""

import math

import numpy
import scipy.sparse
import scipy.sparse.csgraph


def segment_graph(graph, edge_intervals):
    """Every (left node, right node, half-tick cost, logical parity)."""
    segments = []
    for edge_index in range(len(graph.edges)):
        edge = graph.edges[edge_index]
        interval = edge_intervals[edge_index]
        built = _segments_of(edge_index, edge, interval)
        segments.extend(built)
    return segments


def _segments_of(edge_index, edge, interval):
    """One edge as a chain of segments, the parity on the first."""
    length = edge.length_half_ticks
    lower, upper = _uncovered_span(interval)
    coordinates = sorted({0, lower, upper, length})
    node_count = len(coordinates)
    nodes = [edge.detector_a]
    inner_count = node_count - 1
    for split in range(1, inner_count):
        nodes.append(("segment", edge_index, split))
    nodes.append(edge.detector_b)
    parity = edge.logical_observables[0]
    built = []
    segment_count = node_count - 1
    for index in range(segment_count):
        left_coordinate = coordinates[index]
        right_coordinate = coordinates[index + 1]
        cost = _uncovered_measure(
            left_coordinate, right_coordinate, lower, upper
        )
        segment_parity = 0
        if index == 0:
            segment_parity = parity
        left_node = nodes[index]
        right_node = nodes[index + 1]
        segment = (left_node, right_node, cost, segment_parity)
        built.append(segment)
    return built


def _uncovered_span(interval):
    """The part of the edge no ball reached, as [lower, upper] half ticks."""
    interval_type = type(interval)
    is_closed = interval_type.__name__ == "Closed"
    if is_closed:
        return 0, 0
    return interval.lower_tick, interval.upper_tick


def _uncovered_measure(left, right, lower, upper):
    """The measure of [left, right] outside every ball."""
    overlap_low = max(left, lower)
    overlap_high = min(right, upper)
    measure = overlap_high - overlap_low
    if measure < 0:
        return 0
    return measure


def shortest_odd_closed_walk(graph, edge_intervals):
    """Min over v of the shortest closed walk from v of odd logical parity.

    Costs are integers in half ticks. scipy's sparse matrices drop an
    explicit zero, so every segment is stored at cost + EPSILON and the
    answer is rounded back to the integer it must be; a walk of at most a
    few thousand segments moves the total by less than 1e-5.
    """
    segments = segment_graph(graph, edge_intervals)
    names = _node_names(segments)
    node_count = len(names)
    if node_count == 0:
        return math.inf
    matrix = _doubled_matrix(segments, names, node_count)
    sources = numpy.arange(node_count)
    distances = scipy.sparse.csgraph.dijkstra(
        matrix, directed=True, indices=sources
    )
    best = _shortest_odd_distance(distances, node_count)
    if not math.isfinite(best):
        return math.inf
    rounded = round(best)
    error = best - rounded
    drift = abs(error)
    if drift > 1e-4:
        raise AssertionError(
            f"walk cost {best} is not an integer of half ticks"
        )
    return rounded


def _node_names(segments):
    """One index per node of the segment graph, in first-seen order."""
    names = {}
    for left, right, _cost, _parity in segments:
        _name_node(names, left)
        _name_node(names, right)
    return names


def _name_node(names, node):
    """Give the node the next free index, if it has none."""
    if node in names:
        return
    next_index = len(names)
    names[node] = next_index


def _doubled_matrix(segments, names, node_count):
    """The parity-doubled arc matrix scipy takes."""
    rows = []
    columns = []
    costs = []
    for left, right, cost, parity in segments:
        left_index = names[left]
        right_index = names[right]
        _append_both_directions(
            rows,
            columns,
            costs,
            left_index,
            right_index,
            cost,
            parity,
            node_count,
        )
    size = 2 * node_count
    data = numpy.array(costs, dtype=float)
    coordinates = (rows, columns)
    return scipy.sparse.csr_matrix((data, coordinates), shape=(size, size))


def _append_both_directions(
    rows, columns, costs, left, right, cost, parity, node_count
):
    """One segment as four arcs: both directions, both incoming parities."""
    for parity_in in (0, 1):
        parity_out = parity_in ^ parity
        _append_arc(
            rows,
            columns,
            costs,
            left,
            parity_in,
            right,
            parity_out,
            cost,
            node_count,
        )
        _append_arc(
            rows,
            columns,
            costs,
            right,
            parity_in,
            left,
            parity_out,
            cost,
            node_count,
        )


def _shortest_odd_distance(distances, node_count):
    """The smallest distance from (v, 0) to (v, 1) over every node v."""
    best = math.inf
    for source in range(node_count):
        target = source + node_count
        reached = distances[source][target]
        if reached < best:
            best = reached
    return best


EPSILON = 1e-9


def _append_arc(
    rows, columns, costs, left, parity_in, right, parity_out, cost, node_count
):
    row = left + node_count * parity_in
    column = right + node_count * parity_out
    lifted_cost = cost + EPSILON
    rows.append(row)
    columns.append(column)
    costs.append(lifted_cost)


def half_ticks_to_nats(half_ticks, weight_step):
    """The paper's growth counted in w/2 steps, read back as weight."""
    if half_ticks == math.inf:
        return math.inf
    return (half_ticks / 2.0) * weight_step
