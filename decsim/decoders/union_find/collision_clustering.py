"""The Collision Clustering chip: its loop on one decode, and its cycles.

Barber et al. 2309.05558 (Sec. II A, Sec. III, Fig. 2) keep one entry a
defect in a Cluster Growth Stack, with the defect's growth radius, and
repeat three stages until no cluster grows. Grow visits every entry,
finds the root of its cluster and grows the radius of an entry whose
cluster holds an odd number of defects and touches no boundary. Match
compares every pair of entries at every growth step: the distance of
the two defects against the sum of their radii. Union joins the pairs
that collide. Init loads the defects first. The chip decodes one of the
two disjoint decoding graphs (their Fig. 5), so a decode has a chip a
connected part of its graph.

passes_of runs that loop and notes what each pass did. A pass grows a
radius by one unit, half a weight tick, the unit of decsim's edge
lengths (length_half_ticks); two entries collide when their radii
together reach the shortest path between the two defects, and an entry
touches the boundary when its radius reaches its shortest path there.
The clusters are then those of decsim's union-find on the same graph
(tests/decoders/test_union_find_collision_clustering.py). The paper's
chip has no weights: every edge is two units, the reading of its growth
that gives its published logical error rates. On a weighted graph the
loop is the same and a longer edge takes more passes, as the weighted
hardware of Liyanage et al. 2406.08491 Sec. VI and Wu et al. 2502.14787
Sec. 8.1 grows, one weight unit a step. The paper leaves the weighted
distance out (Appendix E), so its cost here is the cost without
weights.

cycles_of prices the passes. The paper prints no cycles of an
operation. Its data record (Zenodo 10.5281/zenodo.11621877,
fpga_accuracy_data.csv) gives the mean cycles of Init, Grow and Merge
for 77 cases, distance 3 to 23 at error rates 0.001 to 0.009, with the
circuits. The costs below are the least squares of this loop's
operations on those circuits against those stage cycles, one set for
all cases. They give the record's total cycles within 1.3% in the
typical case and 4.8% in the worst; the Match stage alone is off by up
to 10%, with a pattern in the distance that is not explained.
"""

import dataclasses

import numpy
import scipy.sparse
import scipy.sparse.csgraph

import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.records.decoder_evidence as evidence_records

# Init: a decode, a 32-bit word of the chip's syndrome, a defect
INIT_CYCLES = 8.756
INIT_CYCLES_PER_WORD = 1.055
INIT_CYCLES_PER_DEFECT = 22.14
SYNDROME_WORD_BITS = 32
# Grow, each pass: the pass, an entry, an entry it grows, a parent-table
# step to a root, an entry that reaches the boundary
GROW_CYCLES_PER_PASS = 0.0986
GROW_CYCLES_PER_ENTRY = 7.316
GROW_CYCLES_PER_GROWN_ENTRY = 0.4987
GROW_CYCLES_PER_ROOT_STEP = 1.114
GROW_CYCLES_PER_BOUNDARY_HIT = 5.579
# Match, each growth step: the step, a pair compared, a pair that
# collides for the first time, an entry that reaches the boundary
MATCH_CYCLES_PER_STEP = 5.185
MATCH_CYCLES_PER_PAIR = 0.9959
MATCH_CYCLES_PER_FIRST_COLLISION = 12.83
MATCH_CYCLES_PER_BOUNDARY_HIT = 8.225
# Merge is Match with Union beside it; the record's Merge is Match's
# cycles and this much a growth step
MERGE_CYCLES_PAST_MATCH = 1.422
# a radius never reaches a boundary the chip's graph does not have
_INT64_LIMITS = numpy.iinfo(numpy.int64)
NO_BOUNDARY = _INT64_LIMITS.max


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

    The pass that finds no entry to grow ends the loop: it visits the
    entries and compares nothing.
    """

    entries: int
    grown_entries: int
    root_steps: int
    boundary_hits: int
    compared_pairs: int
    first_collisions: int


# the chips of the graphs decoded so far, by the graph's identity; the
# graph is kept beside them so that its identity stays its own
_CHIPS_BY_GRAPH: dict = {}


def chips_of(graph: evidence_records.UnionFindGraph) -> tuple:
    """The chips of a graph, laid out once per graph."""
    key = id(graph)
    kept = _CHIPS_BY_GRAPH.get(key)
    if kept is not None:
        return kept[1]
    chips = _lay_out_chips(graph)
    _CHIPS_BY_GRAPH[key] = (graph, chips)
    return chips


def passes_of(chip: Chip, defects: numpy.ndarray) -> tuple:
    """The passes of one decode of a chip, and the root of each defect.

    defects are positions in chip.detectors. Returns the passes and,
    for each defect, the defect that is its cluster's root at the end.
    """
    count = len(defects)
    root = numpy.arange(count)
    if count == 0:
        quiet_pass = Pass(0, 0, 0, 0, 0, 0)
        return [quiet_pass], root
    square = numpy.ix_(defects, defects)
    pair_reach = chip.between[square]
    boundary_reach = chip.to_boundary[defects]
    depth = numpy.zeros(count, dtype=numpy.int64)
    is_odd = numpy.ones(count, dtype=bool)
    touches = numpy.zeros(count, dtype=bool)
    radius = numpy.zeros(count, dtype=numpy.int64)
    has_collided = numpy.zeros((count, count), dtype=bool)
    pair_count = count * (count - 1) // 2
    passes = []
    while True:
        grows = is_odd[root] & ~touches[root]
        grown = numpy.count_nonzero(grows)
        depth_total = depth.sum()
        root_steps = int(depth_total)
        if grown == 0:
            last_pass = Pass(count, 0, root_steps, 0, 0, 0)
            passes.append(last_pass)
            return passes, root
        radius[grows] += 1
        reaches = grows & (radius >= boundary_reach)
        touches[root[reaches]] = True
        firsts, seconds = _first_collisions(radius, pair_reach, has_collided)
        for first, second in zip(firsts, seconds, strict=True):
            _join(root, depth, is_odd, touches, first, second)
        hits = numpy.count_nonzero(reaches)
        collisions = len(firsts)
        one_pass = Pass(count, grown, root_steps, hits, pair_count, collisions)
        passes.append(one_pass)


def cycles_of(passes: list, detector_count: int) -> float:
    """The chip's clock cycles for one decode: Init, Grow and Merge."""
    defects = passes[0].entries
    word_count = -(-detector_count // SYNDROME_WORD_BITS)
    cycles = INIT_CYCLES
    cycles += INIT_CYCLES_PER_WORD * word_count
    cycles += INIT_CYCLES_PER_DEFECT * defects
    for one_pass in passes:
        cycles += GROW_CYCLES_PER_PASS
        cycles += GROW_CYCLES_PER_ENTRY * one_pass.entries
        cycles += GROW_CYCLES_PER_GROWN_ENTRY * one_pass.grown_entries
        cycles += GROW_CYCLES_PER_ROOT_STEP * one_pass.root_steps
        cycles += GROW_CYCLES_PER_BOUNDARY_HIT * one_pass.boundary_hits
        if one_pass.grown_entries == 0:
            continue
        cycles += MATCH_CYCLES_PER_STEP + MERGE_CYCLES_PAST_MATCH
        cycles += MATCH_CYCLES_PER_PAIR * one_pass.compared_pairs
        first_collisions = one_pass.first_collisions
        cycles += MATCH_CYCLES_PER_FIRST_COLLISION * first_collisions
        cycles += MATCH_CYCLES_PER_BOUNDARY_HIT * one_pass.boundary_hits
    return cycles


def _first_collisions(radius, pair_reach, has_collided) -> tuple:
    """Match: the pairs whose radii meet now and did not meet before.

    Every pair is compared, the first entry before the second;
    has_collided takes in the pairs that meet.
    """
    reach_sum = radius[:, None] + radius[None, :]
    meets = reach_sum >= pair_reach
    collides = numpy.triu(meets, 1)
    is_first = collides & ~has_collided
    has_collided |= collides
    firsts, seconds = numpy.nonzero(is_first)
    return firsts.tolist(), seconds.tolist()


def _join(root, depth, is_odd, touches, first: int, second: int) -> None:
    """Union: the second entry's cluster goes under the first's root.

    The paper leaves the choice of the root open ("arbitrary"). The
    parent table has no shortcuts, so every entry of the cluster that
    goes under is one table step further from its root.
    """
    kept = root[first]
    dropped = root[second]
    if kept == dropped:
        return
    moved = root == dropped
    depth[moved] += 1
    root[moved] = kept
    is_odd[kept] ^= is_odd[dropped]
    touches[kept] |= touches[dropped]


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
