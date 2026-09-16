"""The compiled Union-Find beside the Python it replaced, on one corpus.

The oracle is tests/decoders/union_find_oracle.py, the growth, the
forest and the peeling decsim ran in Python, unchanged in logic; the row
under test is decsim/decoders/union_find/union_find.c through
decsim/decoders/union_find/compiled_decoder.py. Timing comes from the
latency card and never from how long a decoder runs, so the compiled
row has to make the same decisions and not merely a correct one: the
same selected faults, the same intervals, the same contacts in the same
order, the same forest, the same unmatched detectors.

The corpus is Stim's rotated surface code memory circuits, which is what
an experiment decodes, and random small graphs, which reach the shapes a
surface code never produces: a boundary to boundary column, a column
with no detector at all, a prior of exactly one half, priors above one
half, duplicate columns and ties in edge length.

The growth steps the C reports beside its decisions, one per step with
the boundary edges the step advanced and the deepest flood over grown
edges from a fused cluster's root, not through the boundary (Helios
2301.08419 lines 623-629), are checked by hand on the twelve-detector
graph of test_union_find_decoder.py: a chain with a boundary edge at
each end and two four-cycles.
"""

import ctypes.util
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
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoder_evidence as evidence_records
import tests.decoders.test_union_find_decoder as hand_graph
import tests.decoders.union_find_oracle as union_find_oracle
import tests.decoders.windows as windows

TESTS_FILE = pathlib.Path(__file__)
TESTS_PATH = TESTS_FILE.resolve()
CHECKOUT = TESTS_PATH.parent.parent.parent
GRAPHLIKE = fault_models.FaultRepresentation.GRAPHLIKE
REQUIREMENT = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
WEIGHT_STEP = 0.1

# the four distances the 2026-09 grid runs below eleven, each with as
# many rounds as its distance, at three of the grid's error rates
CORPUS_DISTANCES = (3, 5, 7, 9)
CORPUS_PROBABILITIES = (0.001, 0.005, 0.01)
CORPUS_SHOTS = 300
CIRCUIT_SEED = 5

# the two distances above that, at the rate the experiment's largest
# points run, with fewer shots because the oracle costs a second a
# decode there
LARGE_DISTANCES = (11, 13)
LARGE_PROBABILITY = 0.005
LARGE_SHOTS = 30

RANDOM_GRAPH_COUNT = 200
RANDOM_GRAPH_SEED = 11
SANITIZED_GRAPH_COUNT = 40

# a column takes two detectors, one and the boundary, or neither
ENDPOINT_COUNTS = (0, 1, 2, 2, 2)
# one half is an ordinary zero-weight fault and anything above it is a
# baseline fault, so both sides of the half are drawn; zero and one are
# the only priors that drop a column, which is the only way an edge
# index parts from its fault index
PRIOR_CHOICES = (
    0.0,
    0.5,
    1.0,
    0.001,
    0.05,
    0.1,
    0.1,
    0.25,
    0.3,
    0.7,
    0.9,
    0.999,
)
DUPLICATE_COLUMN_RATE = 0.25
OBSERVABLE_ROWS = 2

SANITIZER_RUNTIME = ctypes.util.find_library("asan")
COMPILER = shutil.which("gcc")
SANITIZERS_MISSING = SANITIZER_RUNTIME is None or COMPILER is None
SANITIZED_LIBRARY = "decsim/decoders/union_find/union_find_sanitized.so"
CHILD_PROGRAM = (
    "import tests.decoders.test_union_find_compiled_decoder as corpus;"
    f"corpus.compare_random_graphs({SANITIZED_GRAPH_COUNT},"
    f" {RANDOM_GRAPH_SEED})"
)


def graph_of(check, priors, observables):
    """The weighted graph of one hand-built or random check matrix."""
    matrix = numpy.asarray(check, dtype=numpy.uint8)
    prior_array = numpy.asarray(priors, dtype=float)
    observable_array = numpy.asarray(observables, dtype=numpy.uint8)
    fault_count = matrix.shape[1]
    owned = numpy.ones(fault_count, dtype=bool)
    source_fault_ids = tuple(range(fault_count))
    placed = fault_models.PlacedFaultModel(
        representation=GRAPHLIKE,
        check=matrix,
        priors=prior_array,
        observables=observable_array,
        owned=owned,
        source_fault_ids=source_fault_ids,
        boundary_flips={},
    )
    return window_decoder.graph_from_model(
        placed, location="corpus window", weight_step=WEIGHT_STEP
    )


def compare_one_decode(graph, syndrome) -> None:
    """Every field of one decode beside the oracle's, element for element."""
    compiled = window_decoder.decode_graph(graph, syndrome)
    expected = union_find_oracle.decode_graph(graph, syndrome)
    assert compiled.selected_faults == expected.selected_faults
    assert compiled.edge_intervals == expected.edge_intervals
    assert compiled.contact_faults == expected.contact_faults
    assert compiled.erasure_forest_faults == expected.erasure_forest_faults
    assert compiled.unmatched_detectors == expected.unmatched_detectors
    assert compiled.logical_observables == expected.logical_observables


def random_column(rng, detector_count: int) -> list:
    column = [0] * detector_count
    endpoint_count = rng.choice(ENDPOINT_COUNTS)
    taken = min(endpoint_count, detector_count)
    rows = rng.sample(range(detector_count), taken)
    for row in rows:
        column[row] = 1
    return column


def random_columns(rng, detector_count: int, fault_count: int) -> list:
    """Fault columns, a quarter of them a copy of an earlier column."""
    columns = []
    for fault_index in range(fault_count):
        column = random_column(rng, detector_count)
        draw = rng.random()
        repeats_earlier = draw < DUPLICATE_COLUMN_RATE
        if repeats_earlier and fault_index > 0:
            earlier = rng.randrange(fault_index)
            column = list(columns[earlier])
        columns.append(column)
    return columns


def random_observables(rng, fault_count: int) -> list:
    rows = []
    for _ in range(OBSERVABLE_ROWS):
        row = random_bits(rng, fault_count)
        rows.append(row)
    return rows


def random_bits(rng, count: int) -> list:
    bits = []
    for _ in range(count):
        bit = rng.randint(0, 1)
        bits.append(bit)
    return bits


def random_graph(rng):
    """A small graph with duplicate columns, ties and priors above a half."""
    detector_count = rng.randint(1, 9)
    fault_count = rng.randint(1, 14)
    columns = random_columns(rng, detector_count, fault_count)
    rows = numpy.asarray(columns, dtype=numpy.uint8)
    check = rows.transpose()
    priors = []
    for _ in range(fault_count):
        prior = rng.choice(PRIOR_CHOICES)
        priors.append(prior)
    observables = random_observables(rng, fault_count)
    return graph_of(check, priors, observables)


def compare_random_graphs(count: int, seed: int) -> None:
    """One syndrome on each of count random graphs, against the oracle."""
    rng = random.Random(seed)
    for _ in range(count):
        graph = random_graph(rng)
        bits = random_bits(rng, graph.detector_count)
        syndrome = numpy.asarray(bits, dtype=numpy.uint8)
        compare_one_decode(graph, syndrome)


def compare_corpus_point(distance: int, probability: float, shots: int) -> None:
    """Every sampled shot of one distance and error rate, decode for decode."""
    circuit = windows.memory_circuit(distance, distance, probability)
    model = windows.whole_circuit_window(circuit, distance, REQUIREMENT)
    graph = window_decoder.graph_from_model(
        model.graphlike_faults,
        location="corpus window",
        weight_step=WEIGHT_STEP,
    )
    sampled, _observables = windows.sampled_shots(circuit, shots, CIRCUIT_SEED)
    for shot in sampled:
        syndrome = windows.row_syndrome(model, shot)
        compare_one_decode(graph, syndrome)


def test_the_property_corpus_of_surface_code_shots_matches_the_oracle():
    """Twelve points of three hundred sampled syndromes, decode for decode."""
    for distance in CORPUS_DISTANCES:
        for probability in CORPUS_PROBABILITIES:
            compare_corpus_point(distance, probability, CORPUS_SHOTS)


def test_the_property_corpus_of_the_two_largest_distances_matches_the_oracle():
    """Distances eleven and thirteen, thirty sampled syndromes each."""
    for distance in LARGE_DISTANCES:
        compare_corpus_point(distance, LARGE_PROBABILITY, LARGE_SHOTS)


def test_the_property_corpus_of_random_small_graphs_matches_the_oracle():
    """Two hundred random graphs, one syndrome each, decode for decode."""
    compare_random_graphs(RANDOM_GRAPH_COUNT, RANDOM_GRAPH_SEED)


def hand_graph_of():
    model = hand_graph._model()
    faults = model.require_faults(hand_graph.GRAPHLIKE)
    return window_decoder.graph_from_model(
        faults, location="test", weight_step=WEIGHT_STEP
    )


def steps_of(defect_rows) -> tuple:
    syndrome = numpy.zeros(hand_graph.DETECTOR_COUNT, dtype=numpy.uint8)
    syndrome[list(defect_rows)] = 1
    placed = hand_graph_of()
    evidence = window_decoder.decode_graph(placed, syndrome)
    return evidence.growth_steps


def test_two_adjacent_defects_take_one_step_over_three_edges_one_hop():
    """Rows 8 and 9 each grow along two edges, sharing the one between."""
    step = evidence_records.GrowthStep(edge_count=3, hop_count=1)
    assert steps_of([8, 9]) == (step,)


def test_a_lone_defect_grows_twice_and_its_flood_deepens_each_step():
    """Row 10 reaches rows 9 and 11, then row 8 and the boundary.

    The second flood runs from row 8, the cluster's root, to row 11,
    three hops; the boundary is one hop further and is not counted.
    """
    first = evidence_records.GrowthStep(edge_count=2, hop_count=2)
    second = evidence_records.GrowthStep(edge_count=2, hop_count=3)
    assert steps_of([10]) == (first, second)


def test_the_flood_does_not_pass_through_the_boundary():
    """Rows 8 and 11 both reach the boundary in one step.

    They fuse into one cluster through the boundary node, and the
    flood from row 8 stops there, so the deepest hop is one.
    """
    step = evidence_records.GrowthStep(edge_count=4, hop_count=1)
    assert steps_of([8, 11]) == (step,)


def test_opposite_corners_of_a_four_cycle_fuse_at_depth_two():
    step = evidence_records.GrowthStep(edge_count=4, hop_count=2)
    assert steps_of([0, 2]) == (step,)


def test_an_empty_syndrome_takes_no_step():
    assert steps_of([]) == ()


def test_the_graph_emits_its_edges_in_increasing_fault_order():
    """The contract union_find.h states: an edge index is a fault order.

    The C never sees a fault index. It sorts the closing batch, the
    Kruskal ties and every node's forest neighbours by the edge index
    instead, which is the same order only while the graph emits one
    edge per fault column in increasing fault index. A column whose
    residual is certain is dropped, so the two indices part there, and
    the priors below drop three of six.
    """
    check = [[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]]
    priors = [0.0, 0.1, 1.0, 0.2, 0.0, 0.3]
    observables = [[0, 0, 0, 0, 0, 0]]
    graph = graph_of(check, priors, observables)
    edges = graph.edges
    assert len(edges) == 3
    assert edges[0].fault_index == 1
    assert edges[1].fault_index == 3
    assert edges[2].fault_index == 5


def test_a_syndrome_of_the_wrong_length_is_refused():
    graph = graph_of([[1, 0], [0, 1]], [0.1, 0.1], [[0, 0]])
    syndrome = numpy.asarray([1, 0, 1], dtype=numpy.uint8)
    with pytest.raises(ValueError) as refusal:
        window_decoder.decode_graph(graph, syndrome)
    assert "one-dimensional detector vector of length 2" in str(refusal.value)


def test_a_syndrome_that_is_not_binary_is_refused():
    graph = graph_of([[1, 0], [0, 1]], [0.1, 0.1], [[0, 0]])
    syndrome = numpy.asarray([2, 0], dtype=numpy.uint8)
    with pytest.raises(ValueError) as refusal:
        window_decoder.decode_graph(graph, syndrome)
    assert "must contain only binary values" in str(refusal.value)


def test_a_missing_compiled_library_is_refused_by_its_build_command(
    monkeypatch,
):
    monkeypatch.setenv(compiled_decoder.LIBRARY_VARIABLE, "/no/such/library")
    compiled_decoder.entry_point.cache_clear()
    with pytest.raises(RuntimeError) as refusal:
        compiled_decoder.entry_point()
    assert compiled_decoder.BUILD_COMMAND in str(refusal.value)


@pytest.mark.skipif(
    SANITIZERS_MISSING,
    reason="no sanitizer runtime this interpreter can preload",
)
def test_the_compiled_decoder_stays_inside_its_buffers_on_the_corpus():
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
