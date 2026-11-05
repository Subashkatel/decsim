"""The cluster record beside the walk and the clusters it is held to.

The row under test is decsim/decoders/cluster_record.py with
the entry point's walk that marks its edges (union_find_cluster_gap
in decsim/decoders/union_find/cluster_gap.c). The gap alone is the
oracle for the walk's length; the marked edges must cost exactly that
length; and scipy's connected components, a second reading of which
detectors the covered edges join, is the oracle for the clusters.

The corpus is test_compiled_cluster_gap.py's: Stim's rotated surface
code at its points, and its random small graphs.
"""

import math
import random

import numpy
import scipy.sparse
import scipy.sparse.csgraph

import decsim.decoders.cluster_record as cluster_record
import decsim.decoders.union_find.compiled_decoder as compiled_decoder
import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.records.decoder_evidence as evidence_records
import tests.confidence.test_compiled_cluster_gap as walk_corpus
import tests.decoders.test_union_find_compiled_decoder as decoder_corpus
import tests.decoders.windows as windows

# the compiled walk affords many more shots than the Python one
RECORD_POINTS = (
    (3, 0.005, 200),
    (5, 0.003, 200),
    (5, 0.005, 200),
    (7, 0.003, 60),
    (7, 0.005, 60),
)


def corpus_growths():
    """Every growth of the Stim points and the random graphs, decoded once."""
    for distance, probability, shots in RECORD_POINTS:
        circuit = windows.memory_circuit(distance, distance, probability)
        model = windows.whole_circuit_window(
            circuit, distance, walk_corpus.REQUIREMENT
        )
        graph = window_decoder.graph_from_model(
            model.graphlike_faults,
            location="record corpus window",
            weight_step=walk_corpus.WEIGHT_STEP,
        )
        sampled, _observables = windows.sampled_shots(
            circuit, shots, walk_corpus.CIRCUIT_SEED
        )
        for shot in sampled:
            syndrome = windows.row_syndrome(model, shot)
            yield window_decoder.decode_graph(graph, syndrome)
    rng = random.Random(walk_corpus.RANDOM_GRAPH_SEED)
    for _ in range(walk_corpus.RANDOM_GRAPH_COUNT):
        graph = decoder_corpus.random_graph(rng)
        bits = decoder_corpus.random_bits(rng, graph.detector_count)
        syndrome = numpy.asarray(bits, dtype=numpy.uint8)
        yield window_decoder.decode_graph(graph, syndrome)


def uncovered_half_ticks(interval) -> int:
    if isinstance(interval, evidence_records.Closed):
        return 0
    return interval.upper_tick - interval.lower_tick


def marked_cost(evidence, walk_edges: tuple) -> int:
    """The walk's length read off its marked edges, each crossed once.

    A shortest odd walk is one odd cycle and at most some free stretches
    crossed twice, so each marked edge's uncovered span counts once.
    """
    cost = 0
    for edge_index in walk_edges:
        interval = evidence.edge_intervals[edge_index]
        cost += uncovered_half_ticks(interval)
    return cost


def scipy_clusters(evidence) -> tuple:
    """The cluster count and the largest cluster's events, by scipy."""
    graph = evidence.graph
    rows = []
    columns = []
    edges = zip(graph.edges, evidence.edge_intervals, strict=True)
    for edge, interval in edges:
        if not isinstance(interval, evidence_records.Closed):
            continue
        if window_decoder.BOUNDARY in (edge.detector_a, edge.detector_b):
            continue
        rows.append(edge.detector_a)
        columns.append(edge.detector_b)
    ones = numpy.ones(len(rows), dtype=numpy.int8)
    shape = (graph.detector_count, graph.detector_count)
    adjacency = scipy.sparse.coo_matrix((ones, (rows, columns)), shape=shape)
    _count, labels = scipy.sparse.csgraph.connected_components(
        adjacency, directed=False
    )
    events = numpy.asarray(evidence.residual_syndrome, dtype=bool)
    event_labels = labels[events]
    if len(event_labels) == 0:
        return 0, 0
    _labels, sizes = numpy.unique(event_labels, return_counts=True)
    largest = sizes.max()
    return len(sizes), int(largest)


def test_the_walk_entry_point_gives_the_gap_entry_point_s_gap():
    for evidence in corpus_growths():
        gap = compiled_decoder.cluster_gap(
            evidence.graph, evidence.edge_intervals
        )
        walk_gap, _walk_edges = compiled_decoder.cluster_gap_walk(
            evidence.graph, evidence.edge_intervals
        )
        assert walk_gap == gap


def test_the_marked_edges_cost_exactly_the_gap():
    checked = 0
    for evidence in corpus_growths():
        gap, walk_edges = compiled_decoder.cluster_gap_walk(
            evidence.graph, evidence.edge_intervals
        )
        if gap == math.inf:
            assert walk_edges == ()
            continue
        assert len(walk_edges) > 0
        assert marked_cost(evidence, walk_edges) == gap
        checked += 1
    assert checked > 0


def test_the_clusters_match_scipy_s_connected_components():
    for evidence in corpus_growths():
        record = cluster_record.cluster_record(evidence)
        count, largest = scipy_clusters(evidence)
        assert record.cluster_count == count
        assert record.largest_cluster_events == largest


def test_the_doubtful_clusters_are_clusters_of_the_window():
    found_doubt = 0
    for evidence in corpus_growths():
        record = cluster_record.cluster_record(evidence)
        event_count = sum(evidence.residual_syndrome)
        assert record.walk_cluster_count <= record.cluster_count
        assert record.walk_event_count == len(record.walk_detection_events)
        assert record.walk_event_count <= event_count
        for detector in record.walk_detection_events:
            assert evidence.residual_syndrome[detector] == 1
        if record.walk_cluster_count > 0:
            found_doubt += 1
    assert found_doubt > 0


def test_a_quiet_ring_walks_all_three_edges_and_holds_no_cluster():
    """test_compiled_cluster_gap.py's ring: the walk is the ring itself."""
    graph = decoder_corpus.graph_of(
        walk_corpus.RING_CHECK,
        walk_corpus.RING_PRIORS,
        walk_corpus.RING_OBSERVABLES,
    )
    quiet = numpy.zeros(graph.detector_count, dtype=numpy.uint8)
    evidence = window_decoder.decode_graph(graph, quiet)
    record = cluster_record.cluster_record(evidence)
    _gap, walk_edges = compiled_decoder.cluster_gap_walk(
        graph, evidence.edge_intervals
    )
    assert walk_edges == (0, 1, 2)
    assert record.walk_half_ticks == 3 * walk_corpus.RING_EDGE_HALF_TICKS
    assert record.cluster_count == 0
    assert record.walk_cluster_count == 0
    assert not record.is_walk_through_boundary
