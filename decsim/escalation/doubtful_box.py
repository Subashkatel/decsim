"""The doubtful clusters' box, packed on the chip that grew them.

A window escalates when its cluster gap is small, and the gap's shortest
odd walk names the doubtful clusters (decoders/cluster_record.py). The
chip that grew the window's union-find clusters packs what the strong
side needs, and nothing more: the box of the doubtful clusters'
detection events in Stim's detector coordinates, widened by margin
rounds and margin stabilizer columns (2 x margin in Stim's x and y),
with every cluster that has an event inside it taken whole. The request
carries the box's detection events, and union-find's correction outside
the box, which the host joins the strong answer to; the frame holds no
weak answer of an escalated window (windows/window_commits.py
_commit_on_a_guess). The strong side reads no round, grows no cluster
and pins no face: the chip's events hold the earlier window's committed
boundary already, since a window escalates only once its input is final
(windows/input_checks.py). Correlated matching decodes the window's
graph given only the box's events. Each cluster's correction explains
exactly that cluster's events, so the joined correction explains the
window's syndrome.

The chip reads each of the window's detection events twice: once to
mark the clusters with an event in the box, once to list the box's
events and the corrections of the clusters outside it. The packing is
priced at one cycle an event a pass (PACKING_PASSES), on the switching
clock (windows/window_commits.py _form_verdict). The walk that named
the doubtful clusters is the confidence's, priced there
(confidence/cluster.py walk_microseconds). On the wire an event is its
window detector row and a correction its fault column, each in the
fewest bits that name every row or column.

Delfosse's lazy decoder (2108.10790) and pre-decoders hand the strong
decoder the whole syndrome or what is left of it; Bombin et al.
(2303.04846) give each region a buffer of about d. The margin is that
buffer here, and a margin of 0 is the bare doubtful clusters.

The box's shape (BoxShape) can differ from that one box, to find the
fewest bits the strong side needs:

    margins     the rounds (time_margin) and the stabilizer columns
                (space_margin) the box is widened by, each on its own
    form        bounding: one box around every doubtful event;
                per_cluster: one box around each doubtful cluster, the
                shape the union of them; graph: no box, the detectors
                within space_margin edges of a doubtful event on the
                window's graph (the boundary joins nothing)
    doubt       walk: the clusters the gap's shortest odd walk reaches;
                all_walks: the clusters that every odd closed walk
                lighter than doubt_nats reaches, one shortest walk from
                each node an odd edge leaves (all_walk_rows)

Every form takes each cluster with an event inside it whole.
"""

import dataclasses
import heapq
import weakref
from typing import Optional, Union

import numpy

import decsim.decoders.cluster_record as cluster_record
import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.records.decoder_evidence as evidence_records
import decsim.records.decoding as decoding_records

# Stim's rotated memory: two coordinate steps between stabilizers of one
# type in x and y, one step a round in t
COORDINATE_STEP = numpy.asarray([2.0, 2.0, 1.0])
# the chip reads the window's events once to mark the box's clusters,
# and once to list the box's events and the outside's corrections
PACKING_PASSES = 2
FORMS = ("bounding", "per_cluster", "graph")
DOUBTS = ("walk", "all_walks")


@dataclasses.dataclass(frozen=True)
class BoxShape:
    """How the chip widens the doubtful clusters into what it sends.

    time_margin is in rounds and space_margin in stabilizer columns (2
    Stim steps each in x and y); a graph form reads space_margin as the
    edges it walks out and needs the two margins equal. doubt_nats is
    the walk weight all_walks keeps, in the gap's nats (the switching
    threshold's 20 dB is 4.605 nats).
    """

    time_margin: int
    space_margin: int
    form: str = "bounding"
    doubt: str = "walk"
    doubt_nats: Optional[float] = None

    def __post_init__(self) -> None:
        check_margin(self.time_margin)
        check_margin(self.space_margin)
        if self.form not in FORMS:
            raise ValueError(
                f"box form must be one of {FORMS} (got {self.form!r})"
            )
        if self.doubt not in DOUBTS:
            raise ValueError(
                f"box doubt must be one of {DOUBTS} (got {self.doubt!r})"
            )
        if self.form == "graph" and self.time_margin != self.space_margin:
            raise ValueError(
                "a graph box walks one number of edges out; give equal "
                f"margins (got {self.time_margin} rounds, "
                f"{self.space_margin} columns)"
            )
        if (self.doubt == "all_walks") != (self.doubt_nats is not None):
            raise ValueError(
                "doubt_nats is the walk weight all_walks keeps, and only "
                "all_walks reads it"
            )


@dataclasses.dataclass(frozen=True)
class DoubtfulBox:
    """What the chip sends the strong side for one escalated window.

    event_rows are the box's detection events, as window detector rows;
    outside is union-find's correction with the box's clusters taken
    out, one uint8 a fault column. Both go on the wire: the rows, and
    the columns outside selects. last_rounds_read is (stream, last
    round) of each stream the chip's decode read: the strong decode
    reads no round, and its answer rests on those (records/decoding.py
    DecoderStageRecord.last_rounds_read). volume_detectors (the window's
    detectors the shape covers) and doubtful_event_count are for the
    records alone.
    """

    event_rows: numpy.ndarray
    outside: numpy.ndarray
    cluster_count: int
    window_event_count: int
    detector_count: int
    last_rounds_read: tuple = ()
    volume_detectors: int = 0
    doubtful_event_count: int = 0

    def syndrome(self) -> numpy.ndarray:
        """The window's syndrome with only the box's events set."""
        syndrome = numpy.zeros(self.detector_count, dtype=numpy.uint8)
        syndrome[self.event_rows] = 1
        return syndrome

    def wire_bits(self) -> int:
        """The box's events and the outside's corrections, by their names."""
        row_bits = _bits_to_name(self.detector_count)
        column_bits = _bits_to_name(len(self.outside))
        selected_count = numpy.count_nonzero(self.outside)
        outside_count = int(selected_count)
        return len(self.event_rows) * row_bits + outside_count * column_bits

    def packing_cycles(self) -> int:
        """The chip's two passes over the window's detection events."""
        return PACKING_PASSES * self.window_event_count

    def numbers(self) -> dict:
        """The box's size, as the strong result's evidence records it."""
        wire_bits = self.wire_bits()
        outside_count = numpy.count_nonzero(self.outside)
        return {
            "box_detection_events": float(len(self.event_rows)),
            "box_clusters": float(self.cluster_count),
            "window_detection_events": float(self.window_event_count),
            "box_wire_bits": float(wire_bits),
            "box_outside_corrections": float(outside_count),
            "box_volume_detectors": float(self.volume_detectors),
            "box_doubtful_events": float(self.doubtful_event_count),
            "window_detectors": float(self.detector_count),
        }


class BoxPacker:
    """Packs an escalated window's doubtful box from the chip's own growth."""

    def __init__(self, margin: int, shape: Optional[BoxShape] = None) -> None:
        if shape is None:
            shape = BoxShape(time_margin=margin, space_margin=margin)
        self.margin = margin
        self.shape = shape

    def pack(
        self,
        weak_job: decoding_records.DecodeJob,
        weak_result: decoding_records.DecodeResult,
    ) -> DoubtfulBox:
        """The box of the weak decode's doubtful clusters, and the rest."""
        evidence = _union_find_evidence(weak_result)
        coordinates = _detector_coordinates(weak_job)
        graph = evidence.graph
        root_list = cluster_record.detector_roots(
            graph, evidence.edge_intervals
        )
        roots = numpy.asarray(root_list)
        syndrome = numpy.asarray(evidence.syndrome, dtype=numpy.uint8)
        event_rows = numpy.flatnonzero(syndrome)
        walk_rows = doubtful_rows(evidence, roots, event_rows, self.shape)
        chosen, volume = shaped_roots(
            graph, coordinates, roots, event_rows, walk_rows, self.shape
        )
        is_box_row = numpy.isin(roots[event_rows], chosen)
        box_rows = event_rows[is_box_row]
        weak = numpy.asarray(evidence.selected_faults, dtype=numpy.uint8)
        inside = _box_columns(graph, roots, chosen)
        outside = weak & ~inside
        last_rounds_read = _last_rounds_read(weak_job)
        return DoubtfulBox(
            box_rows,
            outside,
            len(chosen),
            len(event_rows),
            graph.detector_count,
            last_rounds_read,
            volume_detectors=volume,
            doubtful_event_count=len(walk_rows),
        )


def doubtful_rows(
    evidence: evidence_records.UnionFindHardEvidence,
    roots: numpy.ndarray,
    event_rows: numpy.ndarray,
    shape: BoxShape,
) -> tuple:
    """The doubtful clusters' detection events, as window rows, ascending."""
    if shape.doubt == "walk":
        record = cluster_record.cluster_record(evidence)
        return record.walk_detection_events
    # the gap's half ticks to nats: nats = half ticks x weight step / 2
    # (confidence/cluster.py gap_half_ticks_to_natural_log_weight)
    limit_half_ticks = 2.0 * shape.doubt_nats / evidence.graph.weight_step
    return all_walk_rows(evidence, roots, event_rows, limit_half_ticks)


def box_roots(
    coordinates: numpy.ndarray,
    roots: numpy.ndarray,
    event_rows: numpy.ndarray,
    walk_rows: tuple,
    margin: int,
) -> numpy.ndarray:
    """The clusters with an event in the doubtful clusters' widened box."""
    if not walk_rows:
        return numpy.zeros(0, dtype=roots.dtype)
    shape = BoxShape(time_margin=margin, space_margin=margin)
    chosen, _volume = _bounding_roots(
        coordinates, roots, event_rows, [walk_rows], shape
    )
    return chosen


def shaped_roots(
    graph: evidence_records.UnionFindGraph,
    coordinates: numpy.ndarray,
    roots: numpy.ndarray,
    event_rows: numpy.ndarray,
    walk_rows: tuple,
    shape: BoxShape,
) -> tuple:
    """(the chosen clusters' roots, the window detectors the shape covers)."""
    if not walk_rows:
        return numpy.zeros(0, dtype=roots.dtype), 0
    if shape.form == "graph":
        reached = _within_edges(graph, walk_rows, shape.space_margin)
        is_reached = numpy.zeros(graph.detector_count, dtype=bool)
        is_reached[reached] = True
        inside_roots = roots[event_rows[is_reached[event_rows]]]
        chosen = numpy.unique(inside_roots)
        reached_count = is_reached.sum()
        return chosen, int(reached_count)
    boxes = [walk_rows]
    if shape.form == "per_cluster":
        boxes = _rows_by_cluster(roots, walk_rows)
    return _bounding_roots(coordinates, roots, event_rows, boxes, shape)


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


def check_margin(margin: object) -> None:
    """A margin is a whole number of rounds, 0 or more."""
    is_whole = isinstance(margin, int) and not isinstance(margin, bool)
    if is_whole and margin >= 0:
        return
    raise ValueError(
        "box_margin must be a whole number of rounds, 0 or more "
        f"(got {margin!r})"
    )


def _rows_by_cluster(roots: numpy.ndarray, walk_rows: tuple) -> list:
    """The doubtful rows grouped by their cluster, a list a cluster."""
    groups = {}
    for row in walk_rows:
        root = int(roots[row])
        group = groups.setdefault(root, [])
        group.append(row)
    grouped = groups.values()
    return list(grouped)


def _bounding_roots(
    coordinates: numpy.ndarray,
    roots: numpy.ndarray,
    event_rows: numpy.ndarray,
    boxes: list,
    shape: BoxShape,
) -> tuple:
    """The clusters with an event in any widened box, and its detectors.

    The detectors are those any of the boxes covers.
    """
    margins = [shape.space_margin, shape.space_margin, shape.time_margin]
    widening = numpy.asarray(margins, dtype=float) * COORDINATE_STEP
    is_covered = numpy.zeros(len(coordinates), dtype=bool)
    for rows in boxes:
        box_coordinates = coordinates[numpy.asarray(rows)]
        low = box_coordinates.min(axis=0) - widening
        high = box_coordinates.max(axis=0) + widening
        is_within = (coordinates >= low) & (coordinates <= high)
        is_covered |= numpy.all(is_within, axis=1)
    inside_roots = roots[event_rows[is_covered[event_rows]]]
    chosen = numpy.unique(inside_roots)
    covered_count = is_covered.sum()
    return chosen, int(covered_count)


def _within_edges(
    graph: evidence_records.UnionFindGraph, start_rows: tuple, hops: int
) -> list:
    """The detectors at most hops edges from a start row.

    The boundary joins no two detectors.
    """
    neighbours = _neighbours(graph)
    reached = {int(row) for row in start_rows}
    frontier = list(reached)
    for _ in range(hops):
        frontier = _next_frontier(frontier, neighbours, reached)
    return sorted(reached)


def _next_frontier(frontier: list, neighbours: list, reached: set) -> list:
    """The frontier's neighbours not reached yet, each now marked reached."""
    candidates = [
        neighbour for row in frontier for neighbour in neighbours[row]
    ]
    following = []
    for neighbour in candidates:
        if neighbour in reached:
            continue
        reached.add(neighbour)
        following.append(neighbour)
    return following


# graph identity -> (graph reference, each detector's neighbours)
_NEIGHBOURS: dict = {}


def _neighbours(graph: evidence_records.UnionFindGraph) -> list:
    """Each detector's neighbours on the window's graph, cached a graph."""
    identity = id(graph)
    entry = _NEIGHBOURS.get(identity)
    if entry is not None and entry[0]() is graph:
        return entry[1]
    neighbours = _neighbour_lists(graph)
    reference = weakref.ref(
        graph, lambda _dead: _NEIGHBOURS.pop(identity, None)
    )
    _NEIGHBOURS[identity] = (reference, neighbours)
    return neighbours


def _neighbour_lists(graph: evidence_records.UnionFindGraph) -> list:
    """Each detector's neighbours, ascending; the boundary joins none."""
    found = [set() for _ in range(graph.detector_count)]
    for edge in graph.edges:
        ends = (edge.detector_a, edge.detector_b)
        if window_decoder.BOUNDARY in ends:
            continue
        found[ends[0]].add(ends[1])
        found[ends[1]].add(ends[0])
    return [tuple(sorted(each)) for each in found]


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


def _box_columns(
    graph: evidence_records.UnionFindGraph, roots: numpy.ndarray, chosen
) -> numpy.ndarray:
    """1 for each fault column inside a chosen cluster, as uint8."""
    fault_rows = _fault_rows(graph)
    has_row = fault_rows >= 0
    column_roots = roots[fault_rows]
    in_box = numpy.isin(column_roots, chosen) & has_row
    return in_box.astype(numpy.uint8)


# graph identity -> (graph reference, each column's detector row): a
# stream's windows share a few graphs (window_decoder.shared_graph)
_FAULT_ROWS: dict = {}


def _fault_rows(graph: evidence_records.UnionFindGraph) -> numpy.ndarray:
    """One window detector row each fault column touches; -1 for no edge.

    A selected fault's cluster is that row's cluster.
    """
    identity = id(graph)
    entry = _FAULT_ROWS.get(identity)
    if entry is not None and entry[0]() is graph:
        return entry[1]
    rows = numpy.full(graph.fault_count, -1, dtype=numpy.int64)
    for edge in graph.edges:
        ends = (edge.detector_a, edge.detector_b)
        touched = [row for row in ends if row != window_decoder.BOUNDARY]
        rows[edge.fault_index] = touched[0]
    rows.setflags(write=False)
    reference = weakref.ref(
        graph, lambda _dead: _FAULT_ROWS.pop(identity, None)
    )
    _FAULT_ROWS[identity] = (reference, rows)
    return rows


def _last_rounds_read(weak_job: decoding_records.DecodeJob) -> tuple:
    """(stream, last round) of each stream the weak decode read."""
    decoder_input = weak_job.decoder_input
    if decoder_input is None:
        fragments = list(weak_job.payloads)
    else:
        fragments = decoder_input.fragments()
    return decoding_records.last_round_by_stream(fragments)


def _bits_to_name(count: int) -> int:
    """The fewest bits that name each of count things; at least one."""
    last_name = count - 1
    bits = last_name.bit_length()
    return max(1, bits)


def _union_find_evidence(
    weak_result: decoding_records.DecodeResult,
) -> evidence_records.UnionFindHardEvidence:
    evidence = weak_result.cluster_evidence
    # union-find is the one row whose result carries cluster evidence
    if evidence is not None:
        return evidence
    carried = type(evidence)
    raise ValueError(
        "the doubtful box is packed from the weak decode's union-find "
        f"growth; this result carries {carried.__name__}"
    )


def _detector_coordinates(
    weak_job: decoding_records.DecodeJob,
) -> numpy.ndarray:
    model = weak_job.detector_error_model
    coordinates = None
    if model is not None:
        coordinates = model.detector_coordinates
    if coordinates is None:
        raise ValueError(
            "the doubtful box needs the window's detector coordinates; "
            f"{weak_job.label} has none"
        )
    places = [place[:3] for place in coordinates]
    return numpy.asarray(places, dtype=float)
