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
"""

import dataclasses
import weakref

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


@dataclasses.dataclass(frozen=True)
class DoubtfulBox:
    """What the chip sends the strong side for one escalated window.

    event_rows are the box's detection events, as window detector rows;
    outside is union-find's correction with the box's clusters taken
    out, one uint8 a fault column. Both go on the wire: the rows, and
    the columns outside selects. last_rounds_read is (stream, last
    round) of each stream the chip's decode read: the strong decode
    reads no round, and its answer rests on those (records/decoding.py
    DecoderStageRecord.last_rounds_read).
    """

    event_rows: numpy.ndarray
    outside: numpy.ndarray
    cluster_count: int
    window_event_count: int
    detector_count: int
    last_rounds_read: tuple = ()

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
        return {
            "box_detection_events": float(len(self.event_rows)),
            "box_clusters": float(self.cluster_count),
            "window_detection_events": float(self.window_event_count),
        }


class BoxPacker:
    """Packs an escalated window's doubtful box from the chip's own growth."""

    def __init__(self, margin: int) -> None:
        self.margin = margin

    def pack(
        self,
        weak_job: decoding_records.DecodeJob,
        weak_result: decoding_records.DecodeResult,
    ) -> DoubtfulBox:
        """The box of the weak decode's doubtful clusters, and the rest."""
        evidence = _union_find_evidence(weak_result)
        coordinates = _detector_coordinates(weak_job)
        graph = evidence.graph
        record = cluster_record.cluster_record(evidence)
        walk_rows = record.walk_detection_events
        root_list = cluster_record.detector_roots(
            graph, evidence.edge_intervals
        )
        roots = numpy.asarray(root_list)
        syndrome = numpy.asarray(evidence.syndrome, dtype=numpy.uint8)
        event_rows = numpy.flatnonzero(syndrome)
        chosen = box_roots(
            coordinates, roots, event_rows, walk_rows, self.margin
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
        )


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
    walk_coordinates = coordinates[numpy.asarray(walk_rows)]
    widening = margin * COORDINATE_STEP
    low = walk_coordinates.min(axis=0) - widening
    high = walk_coordinates.max(axis=0) + widening
    event_coordinates = coordinates[event_rows]
    is_above_low = event_coordinates >= low
    is_below_high = event_coordinates <= high
    is_within = is_above_low & is_below_high
    inside = numpy.all(is_within, axis=1)
    inside_roots = roots[event_rows[inside]]
    return numpy.unique(inside_roots)


def check_margin(margin: object) -> None:
    """A margin is a whole number of rounds, 0 or more."""
    is_whole = isinstance(margin, int) and not isinstance(margin, bool)
    if is_whole and margin >= 0:
        return
    raise ValueError(
        "box_margin must be a whole number of rounds, 0 or more "
        f"(got {margin!r})"
    )


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
