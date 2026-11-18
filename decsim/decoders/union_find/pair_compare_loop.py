"""A serial union-find chip's loop that compares every pair of defects.

Barber et al. 2309.05558 (Sec. II A, Sec. III, Fig. 2) keep one entry a
defect in a stack, with the defect's growth radius, and repeat three
stages until no cluster grows. Grow visits every entry, finds the root
of its cluster and grows the radius of an entry whose cluster holds an
odd number of defects and touches no boundary. Match compares every
pair of entries at every growth step: the distance of the two defects
against the sum of their radii. Union joins the pairs that collide.
The chip decodes one of the two disjoint decoding graphs (their
Fig. 5), so a decode has a chip a connected part of its graph.

passes_of runs that loop and notes what each pass did; it prices
nothing (serial_chip_time.PairCompareCount does). A pass grows a radius
by one unit, half a weight tick, the unit of decsim's edge lengths
(length_half_ticks); two entries collide when their radii together
reach the shortest path between the two defects, and an entry touches
the boundary when its radius reaches its shortest path there. The
clusters are then those of decsim's union-find on the same graph
(tests/decoders/test_union_find_pair_compare_loop.py). The paper's chip
has no weights: every edge is two units, the reading of its growth that
gives its published logical error rates. On a weighted graph the loop
is the same and a longer edge takes more passes, as the weighted
hardware of Liyanage et al. 2406.08491 Sec. VI and Wu et al. 2502.14787
Sec. 8.1 grows, one weight unit a step.
"""

import collections
import dataclasses

import numpy
import scipy.sparse
import scipy.sparse.csgraph

import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.records.decoder_evidence as evidence_records

# a radius never reaches a boundary the chip's graph does not have
_INT64_LIMITS = numpy.iinfo(numpy.int64)
NO_BOUNDARY = _INT64_LIMITS.max
# chips_of's chips by their graph's identity, the least recently used
# dropped past this many; the graph is kept beside its chips so that
# its identity stays its own
_KEPT_GRAPH_LIMIT = 64
_CHIPS_BY_GRAPH: collections.OrderedDict = collections.OrderedDict()


@dataclasses.dataclass(frozen=True)
class Chip:
    """One connected part of a decode's graph, as its chip holds it.

    detectors are the part's detectors, ascending. between[i, j] is the
    shortest path from detectors[i] to detectors[j] in half ticks, and
    to_boundary[i] the shortest from detectors[i] to the boundary.
    """

    detectors: numpy.ndarray
    between: numpy.ndarray
    to_boundary: numpy.ndarray


@dataclasses.dataclass(frozen=True)
class Pass:
    """What one Grow pass and its Match did.

    entry_count entries were visited over root_step_count parent-table
    steps to their roots; grown_entry_count of them grew and
    boundary_hit_count reached the boundary. Match compared
    compared_pair_count pairs, first_collision_count of which met for
    the first time. The pass that finds no entry to grow ends the loop:
    it visits the entries and compares nothing.
    """

    entry_count: int
    root_step_count: int
    grown_entry_count: int
    boundary_hit_count: int
    compared_pair_count: int
    first_collision_count: int


def chips_of(graph: evidence_records.UnionFindGraph) -> tuple:
    """The chips of a graph, laid out once while the graph is in use."""
    key = id(graph)
    kept = _CHIPS_BY_GRAPH.get(key)
    if kept is not None:
        _CHIPS_BY_GRAPH.move_to_end(key)
        return kept[1]
    chips = _lay_out_chips(graph)
    _CHIPS_BY_GRAPH[key] = (graph, chips)
    if len(_CHIPS_BY_GRAPH) > _KEPT_GRAPH_LIMIT:
        _CHIPS_BY_GRAPH.popitem(last=False)
    return chips


def passes_of(chip: Chip, defects: numpy.ndarray) -> tuple:
    """The passes of one decode of a chip, and the root of each defect.

    defects are positions in chip.detectors. Returns the passes and,
    for each defect, the defect that is its cluster's root at the end:
    the loop's clusters, which a test holds against the union-find's.
    """
    square = numpy.ix_(defects, defects)
    pair_reach = chip.between[square]
    boundary_reach = chip.to_boundary[defects]
    entry_count = len(defects)
    stack = _new_stack(entry_count)
    passes = []
    while True:
        one_pass = _run_pass(stack, pair_reach, boundary_reach)
        passes.append(one_pass)
        if one_pass.grown_entry_count == 0:
            return passes, stack.root


@dataclasses.dataclass
class _Stack:
    """The loop's state, an entry a defect; a pass changes it in place.

    root[i] is the entry at the root of entry i's cluster and depth[i]
    the parent-table steps from entry i to it. is_odd and touches are
    read at a root: whether its cluster holds an odd number of defects,
    and whether it has reached the boundary. radius[i] is how far entry
    i has grown, and has_collided[i, j] whether entries i and j have
    met.
    """

    root: numpy.ndarray
    depth: numpy.ndarray
    is_odd: numpy.ndarray
    touches: numpy.ndarray
    radius: numpy.ndarray
    has_collided: numpy.ndarray


def _new_stack(entry_count: int) -> _Stack:
    """Every defect a cluster of its own, not grown."""
    root = numpy.arange(entry_count)
    depth = numpy.zeros(entry_count, dtype=numpy.int64)
    is_odd = numpy.ones(entry_count, dtype=bool)
    touches = numpy.zeros(entry_count, dtype=bool)
    radius = numpy.zeros(entry_count, dtype=numpy.int64)
    has_collided = numpy.zeros((entry_count, entry_count), dtype=bool)
    return _Stack(
        root=root,
        depth=depth,
        is_odd=is_odd,
        touches=touches,
        radius=radius,
        has_collided=has_collided,
    )


def _run_pass(stack: _Stack, pair_reach, boundary_reach) -> Pass:
    """Grow once, then Match with Union beside it."""
    entry_count = len(stack.root)
    grows = stack.is_odd[stack.root] & ~stack.touches[stack.root]
    grown_entry_count = numpy.count_nonzero(grows)
    depth_total = stack.depth.sum()
    root_step_count = int(depth_total)
    if grown_entry_count == 0:
        return Pass(
            entry_count=entry_count,
            root_step_count=root_step_count,
            grown_entry_count=0,
            boundary_hit_count=0,
            compared_pair_count=0,
            first_collision_count=0,
        )
    stack.radius[grows] += 1
    reaches = grows & (stack.radius >= boundary_reach)
    stack.touches[stack.root[reaches]] = True
    firsts, seconds = _first_collisions(stack, pair_reach)
    for first, second in zip(firsts, seconds, strict=True):
        _join(stack, first, second)
    boundary_hit_count = numpy.count_nonzero(reaches)
    pair_count = entry_count * (entry_count - 1) // 2
    return Pass(
        entry_count=entry_count,
        root_step_count=root_step_count,
        grown_entry_count=grown_entry_count,
        boundary_hit_count=boundary_hit_count,
        compared_pair_count=pair_count,
        first_collision_count=len(firsts),
    )


def _first_collisions(stack: _Stack, pair_reach) -> tuple:
    """Match: the pairs whose radii meet now and did not meet before.

    Every pair is compared, the first entry before the second;
    has_collided takes in the pairs that meet.
    """
    reach_sum = stack.radius[:, None] + stack.radius[None, :]
    meets = reach_sum >= pair_reach
    collides = numpy.triu(meets, 1)
    is_first = collides & ~stack.has_collided
    stack.has_collided |= collides
    firsts, seconds = numpy.nonzero(is_first)
    return firsts.tolist(), seconds.tolist()


def _join(stack: _Stack, first: int, second: int) -> None:
    """Union: the second entry's cluster goes under the first's root.

    The paper leaves the choice of the root open ("arbitrary"). The
    parent table has no shortcuts, so every entry of the cluster that
    goes under is one table step further from its root.
    """
    kept = stack.root[first]
    dropped = stack.root[second]
    if kept == dropped:
        return
    moved = stack.root == dropped
    stack.depth[moved] += 1
    stack.root[moved] = kept
    stack.is_odd[kept] ^= stack.is_odd[dropped]
    stack.touches[kept] |= stack.touches[dropped]


def _lay_out_chips(graph: evidence_records.UnionFindGraph) -> tuple:
    endpoint_a, endpoint_b, lengths = graph.edge_arrays[:3]
    count = graph.detector_count
    joins_two = endpoint_b != window_decoder.BOUNDARY
    shortest = _shortest_edges(
        endpoint_a[joins_two], endpoint_b[joins_two], lengths[joins_two], count
    )
    boundary_edge = numpy.full(count, numpy.inf)
    numpy.minimum.at(boundary_edge, endpoint_a[~joins_two], lengths[~joins_two])
    part_count, part = scipy.sparse.csgraph.connected_components(
        shortest, directed=False
    )
    chips = []
    for one_part in range(part_count):
        is_part = part == one_part
        detectors = numpy.flatnonzero(is_part)
        chip = _chip_of(shortest, boundary_edge, detectors)
        chips.append(chip)
    return tuple(chips)


def _shortest_edges(detectors_a, detectors_b, lengths, count: int):
    """The graph as a matrix, the shortest of the edges two detectors share."""
    low = numpy.minimum(detectors_a, detectors_b)
    high = numpy.maximum(detectors_a, detectors_b)
    low = low.astype(numpy.int64)
    high = high.astype(numpy.int64)
    pair = low * count + high
    order = numpy.lexsort((lengths, pair))
    pair = pair[order]
    is_shortest = numpy.ones(len(pair), dtype=bool)
    is_shortest[1:] = pair[1:] != pair[:-1]
    rows = low[order][is_shortest]
    columns = high[order][is_shortest]
    values = lengths[order][is_shortest].astype(numpy.float64)
    return scipy.sparse.csr_matrix(
        (values, (rows, columns)), shape=(count, count)
    )


def _chip_of(shortest, boundary_edge, detectors: numpy.ndarray) -> Chip:
    part = shortest[detectors][:, detectors]
    paths = scipy.sparse.csgraph.dijkstra(part, directed=False)
    through = paths + boundary_edge[detectors][None, :]
    nearest = through.min(axis=1)
    to_boundary = numpy.full(len(detectors), NO_BOUNDARY, dtype=numpy.int64)
    has_boundary = numpy.isfinite(nearest)
    to_boundary[has_boundary] = nearest[has_boundary].astype(numpy.int64)
    between = paths.astype(numpy.int32)
    return Chip(detectors=detectors, between=between, to_boundary=to_boundary)
