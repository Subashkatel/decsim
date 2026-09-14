"""The compiled cluster gap walk beside the walks it is held to.

The row under test is decsim/decoders/union_find/cluster_gap.c through
decsim/decoders/union_find/compiled_decoder.py. The gap it must return
is the one decsim returned before, so cluster_gap_oracle.py, the walk
that left decsim/confidence/cluster.py, is the oracle on every input.
independent_cluster_gap.py, an independent reading of Meister et al.
2405.07433 that runs one all-sources scipy Dijkstra where the other two
run a per-node search with a running cutoff, is the second oracle on the
Stim windows.

The corpus is the decoder's own
(tests/decoders/test_union_find_compiled_decoder.py): its Stim points,
its circuit seed, its random graphs and its graph seed, every window
decoded once and then walked. The shot counts are this module's,
because a Python walk of one distance nine window costs seconds where
the decode costs milliseconds.

The random graphs carry shapes a surface code never produces: a column
on the boundary at both ends, two columns with the same pair of
detectors, and a prior of exactly one half, whose edge is zero half
ticks long. The scipy realization cannot read those: it stores one arc
per segment in a sparse matrix, which sums the entries that share a row
and a column, so two segments between one pair of nodes cost their sum
rather than the smaller of them; and it splits an edge on the set of
its coordinates, which is one coordinate for a zero-length edge, so
such an edge carries no segment and its free logical crossing is lost.
Both shapes are absent from every Stim window here, which is checked
where the walks are compared.
"""

import ctypes.util
import math
import os
import pathlib
import random
import shutil
import subprocess
import sys

import numpy
import pytest

import decsim.decoders.union_find.compiled_decoder as compiled_decoder
import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.records.decoder_evidence as evidence_records
import tests.confidence.cluster_gap_oracle as cluster_gap_oracle
import tests.confidence.independent_cluster_gap as independent_cluster_gap
import tests.decoders.test_union_find_compiled_decoder as decoder_corpus
import tests.decoders.windows as windows

TESTS_FILE = pathlib.Path(__file__)
TESTS_PATH = TESTS_FILE.resolve()
CHECKOUT = TESTS_PATH.parent.parent.parent
REQUIREMENT = decoder_corpus.REQUIREMENT
WEIGHT_STEP = decoder_corpus.WEIGHT_STEP
CIRCUIT_SEED = decoder_corpus.CIRCUIT_SEED
RANDOM_GRAPH_SEED = decoder_corpus.RANDOM_GRAPH_SEED
RANDOM_GRAPH_COUNT = decoder_corpus.RANDOM_GRAPH_COUNT

# the decoder corpus's points, with the shots a Python walk of each
# point affords: one distance nine window costs the oracle seconds
CORPUS_POINTS = (
    (3, 0.001, 40),
    (3, 0.005, 40),
    (3, 0.01, 40),
    (5, 0.001, 40),
    (5, 0.005, 40),
    (5, 0.01, 40),
    (7, 0.005, 8),
    (9, 0.005, 8),
)

SANITIZED_GRAPH_COUNT = 40
SANITIZER_RUNTIME = ctypes.util.find_library("asan")
COMPILER = shutil.which("gcc")
SANITIZERS_MISSING = SANITIZER_RUNTIME is None or COMPILER is None
SANITIZED_LIBRARY = "decsim/decoders/union_find/union_find_sanitized.so"
CHILD_PROGRAM = (
    "import tests.confidence.test_compiled_cluster_gap as corpus;"
    f"corpus.walk_random_graphs({SANITIZED_GRAPH_COUNT},"
    f" {RANDOM_GRAPH_SEED})"
)

# a triangle of detectors with no column on the boundary at all
RING_CHECK = ((1, 0, 1), (1, 1, 0), (0, 1, 1))
RING_PRIORS = (0.1, 0.1, 0.1)
RING_OBSERVABLES = ((1, 0, 0),)
RING_EDGE_HALF_TICKS = 44

# one detector and two columns of probability one half, one of them
# logical: crossing out on one and back on the other flips the
# observable at no weight at all
FREE_CHECK = ((1, 1),)
FREE_PRIORS = (0.5, 0.5)
FREE_OBSERVABLES = ((1, 0),)

# one column with no detector at all: both its ends are the boundary
LOOP_CHECK = ((0, 1),)
LOOP_PRIORS = (0.1, 0.1)
LOOP_OBSERVABLES = ((1, 0),)
LOOP_EDGE_HALF_TICKS = 44


def compare_one_walk(graph, edge_intervals) -> None:
    """One growth's gap, compiled, against the walk it replaced."""
    compiled_gap = compiled_decoder.cluster_gap(graph, edge_intervals)
    moved = cluster_gap_oracle._quotient_cluster_gap(graph, edge_intervals)
    assert compiled_gap == moved


def compare_one_walk_three_ways(graph, edge_intervals) -> None:
    """The same, with the paper's own reading of the walk beside it."""
    compare_one_walk(graph, edge_intervals)
    compiled_gap = compiled_decoder.cluster_gap(graph, edge_intervals)
    paper = independent_cluster_gap.shortest_odd_closed_walk(
        graph, edge_intervals
    )
    assert compiled_gap == paper


def walk_random_graphs(count: int, seed: int) -> None:
    """One decode and one walk on each of count random graphs."""
    rng = random.Random(seed)
    for _ in range(count):
        graph = decoder_corpus.random_graph(rng)
        bits = decoder_corpus.random_bits(rng, graph.detector_count)
        syndrome = numpy.asarray(bits, dtype=numpy.uint8)
        result = window_decoder.decode_graph(graph, syndrome)
        compare_one_walk(graph, result.edge_intervals)


def walk_corpus_point(distance: int, probability: float, shots: int) -> None:
    """Every sampled shot of one point, decoded once and walked."""
    circuit = windows.memory_circuit(distance, distance, probability)
    model = windows.whole_circuit_window(circuit, distance, REQUIREMENT)
    graph = window_decoder.graph_from_model(
        model.graphlike_faults,
        location="corpus window",
        weight_step=WEIGHT_STEP,
    )
    _require_the_scipy_reading_can_read(graph)
    sampled, _observables = windows.sampled_shots(circuit, shots, CIRCUIT_SEED)
    for shot in sampled:
        syndrome = windows.row_syndrome(model, shot)
        result = window_decoder.decode_graph(graph, syndrome)
        compare_one_walk_three_ways(graph, result.edge_intervals)


def test_the_property_corpus_of_surface_code_windows_matches_both_walks():
    """Eight points of Stim's rotated surface code, walk for walk."""
    for distance, probability, shots in CORPUS_POINTS:
        walk_corpus_point(distance, probability, shots)


def test_the_property_corpus_of_random_small_graphs_matches_the_moved_walk():
    """Two hundred random graphs, one growth each, walk for walk."""
    walk_random_graphs(RANDOM_GRAPH_COUNT, RANDOM_GRAPH_SEED)


def test_the_property_corpus_carries_every_shape_the_walk_must_cross():
    """The random graphs reach the shapes a surface code never produces.

    A growth with no odd walk at all, an edge of length zero, an edge
    the growth closed, and an interval whose bound falls on zero or on
    the edge's length: each is a different count of segments, and the
    first segment of an edge is the one that carries the logical
    parity, so an edge with no segment at all is the case that breaks a
    walk which assumes a chain.
    """
    shapes = _random_corpus_shapes(RANDOM_GRAPH_COUNT, RANDOM_GRAPH_SEED)
    assert shapes["unreachable"] > 0
    assert shapes["zero_length"] > 0
    assert shapes["closed"] > 0
    assert shapes["bound_at_zero"] > 0
    assert shapes["bound_at_length"] > 0


def test_the_shortest_odd_walk_need_not_pass_through_the_boundary():
    """Three detectors in a ring, one logical column, no boundary column.

    The walk closes on the ring itself, so its length is the three
    edges together and the boundary node carries no segment at all. A
    walk out along the logical column and back would cross the parity
    twice, which is even, so the ring is the only way round.
    """
    graph = decoder_corpus.graph_of(RING_CHECK, RING_PRIORS, RING_OBSERVABLES)
    first, second, third = graph.edges
    assert (first.detector_a, first.detector_b) == (0, 1)
    assert (second.detector_a, second.detector_b) == (1, 2)
    assert (third.detector_a, third.detector_b) == (0, 2)
    assert first.length_half_ticks == RING_EDGE_HALF_TICKS
    assert second.length_half_ticks == RING_EDGE_HALF_TICKS
    assert third.length_half_ticks == RING_EDGE_HALF_TICKS
    quiet = numpy.zeros(graph.detector_count, dtype=numpy.uint8)
    result = window_decoder.decode_graph(graph, quiet)
    gap = compiled_decoder.cluster_gap(graph, result.edge_intervals)
    assert gap == 3 * RING_EDGE_HALF_TICKS
    compare_one_walk_three_ways(graph, result.edge_intervals)


def test_a_growth_that_crosses_no_logical_column_has_no_walk():
    """A gap of infinity is what a growth with no odd closed walk leaves."""
    graph = decoder_corpus.graph_of(RING_CHECK, RING_PRIORS, ((0, 0, 0),))
    quiet = numpy.zeros(graph.detector_count, dtype=numpy.uint8)
    result = window_decoder.decode_graph(graph, quiet)
    gap = compiled_decoder.cluster_gap(graph, result.edge_intervals)
    assert gap == math.inf
    compare_one_walk_three_ways(graph, result.edge_intervals)


def test_a_logical_column_of_probability_one_half_leaves_no_gap_at_all():
    """A prior of one half is an edge of zero half ticks, closed at once.

    Crossing out on the logical column and back on the other flips the
    observable at no weight, so the gap is zero: the growth carries no
    confidence. The edge still holds one segment, of zero cost.
    """
    graph = decoder_corpus.graph_of(FREE_CHECK, FREE_PRIORS, FREE_OBSERVABLES)
    logical, quiet_column = graph.edges
    assert logical.length_half_ticks == 0
    assert quiet_column.length_half_ticks == 0
    quiet = numpy.zeros(graph.detector_count, dtype=numpy.uint8)
    result = window_decoder.decode_graph(graph, quiet)
    assert result.edge_intervals == (
        evidence_records.Closed(),
        evidence_records.Closed(),
    )
    gap = compiled_decoder.cluster_gap(graph, result.edge_intervals)
    assert gap == 0
    compare_one_walk(graph, result.edge_intervals)


def test_a_column_with_no_detector_closes_its_walk_on_the_boundary():
    """A column with neither end on a detector is a loop at the boundary.

    Its two path nodes are the same node, so one crossing is already a
    closed walk, and an odd one when the column is logical.
    """
    graph = decoder_corpus.graph_of(LOOP_CHECK, LOOP_PRIORS, LOOP_OBSERVABLES)
    loop, detector_column = graph.edges
    assert (loop.detector_a, loop.detector_b) == (-1, -1)
    assert (detector_column.detector_a, detector_column.detector_b) == (0, -1)
    assert loop.length_half_ticks == LOOP_EDGE_HALF_TICKS
    quiet = numpy.zeros(graph.detector_count, dtype=numpy.uint8)
    result = window_decoder.decode_graph(graph, quiet)
    gap = compiled_decoder.cluster_gap(graph, result.edge_intervals)
    assert gap == LOOP_EDGE_HALF_TICKS
    compare_one_walk(graph, result.edge_intervals)


def test_a_growth_with_one_interval_per_edge_missing_is_refused():
    """The C sizes nothing itself, so the count is checked before the call."""
    graph = decoder_corpus.graph_of(RING_CHECK, RING_PRIORS, RING_OBSERVABLES)
    one_interval = evidence_records.Open(0, 1)
    short = (one_interval,)

    with pytest.raises(ValueError) as refusal:
        compiled_decoder.cluster_gap(graph, short)

    assert "one interval per edge" in str(refusal.value)
    assert "3 edges" in str(refusal.value)


@pytest.mark.skipif(
    SANITIZERS_MISSING,
    reason="no sanitizer runtime this interpreter can preload",
)
def test_the_compiled_walk_stays_inside_its_buffers_on_the_corpus():
    """The random graphs again in a child process under the sanitizers.

    The address sanitizer has to be the first library a process loads,
    so the corpus runs in a child with the runtime preloaded and the
    sanitized library named. Anything the sanitizers find stops that
    child, and the leak check is off because the interpreter itself
    holds allocations at exit.
    """
    build_command = CHECKOUT / compiled_decoder.BUILD_COMMAND
    subprocess.run([str(build_command), "--sanitize"], cwd=CHECKOUT, check=True)
    library = CHECKOUT / SANITIZED_LIBRARY
    environment = dict(os.environ)
    environment["LD_PRELOAD"] = SANITIZER_RUNTIME
    environment["ASAN_OPTIONS"] = "detect_leaks=0"
    environment["PYTHONPATH"] = str(CHECKOUT)
    environment[compiled_decoder.LIBRARY_VARIABLE] = str(library)
    child = subprocess.run(
        [sys.executable, "-c", CHILD_PROGRAM],
        cwd=CHECKOUT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr
    assert "AddressSanitizer" not in child.stderr


def _require_the_scipy_reading_can_read(graph) -> None:
    """The two shapes the all-sources reading cannot represent.

    A zero-length edge collapses to one split coordinate and carries no
    segment there, and two segments between one pair of nodes are
    summed rather than compared by the sparse matrix that holds them.
    A window of a Stim circuit has neither, which is why it is the
    corpus that walk carries.
    """
    pairs = set()
    for edge in graph.edges:
        assert edge.length_half_ticks > 0
        ends = (edge.detector_a, edge.detector_b)
        assert ends not in pairs
        pairs.add(ends)


def _random_corpus_shapes(count: int, seed: int) -> dict:
    """How many of each shape the random growths leave behind."""
    shapes = {
        "unreachable": 0,
        "zero_length": 0,
        "closed": 0,
        "bound_at_zero": 0,
        "bound_at_length": 0,
    }
    rng = random.Random(seed)
    for _ in range(count):
        graph = decoder_corpus.random_graph(rng)
        bits = decoder_corpus.random_bits(rng, graph.detector_count)
        syndrome = numpy.asarray(bits, dtype=numpy.uint8)
        result = window_decoder.decode_graph(graph, syndrome)
        _count_graph_shapes(shapes, graph, result.edge_intervals)
    return shapes


def _count_graph_shapes(shapes: dict, graph, edge_intervals) -> None:
    """One growth's shapes, added to the running counts."""
    gap = compiled_decoder.cluster_gap(graph, edge_intervals)
    if gap == math.inf:
        shapes["unreachable"] += 1
    for edge_index, edge in enumerate(graph.edges):
        interval = edge_intervals[edge_index]
        _count_edge_shapes(shapes, edge, interval)


def _count_edge_shapes(shapes: dict, edge, interval) -> None:
    if edge.length_half_ticks == 0:
        shapes["zero_length"] += 1
    is_closed = isinstance(interval, evidence_records.Closed)
    if is_closed:
        shapes["closed"] += 1
        return
    if interval.lower_tick == 0:
        shapes["bound_at_zero"] += 1
    if interval.upper_tick == edge.length_half_ticks:
        shapes["bound_at_length"] += 1
