"""A serial union-find chip's time, on hand cases and on the paper's case.

ClusterReadCount is the read count of Das et al. 2001.06598 Sec. III C,
Eq. 5 and 6, at the paper's cost of four cycles a read on a 4 GHz clock.
The paper's own case is the planar code of distance 11 over a logical
cycle of 11 rounds under phenomenological noise of 0.001, one error
type, with a graph open at its left and right sides, its bottom and
its top (Sec. I C, Fig. 3); its published mean is 42 ns for a single logical
qubit (Das et al., HPCA 2022, Sec. IV E). The property test decodes
random errors of that case with decsim's own union-find and compares
the mean. The hand cases are the two small machines of
test_union_find_cycle_count.py, placed on Stim's coordinates.

PairCompareCount prices the loop of Barber et al. 2309.05558. Its hand
cases are the same machines at costs chosen for the sums. The paper's
data record (Zenodo 10.5281/zenodo.11621877, CC BY 4.0) gives the mean
cycles of a decode on its FPGA for each of its circuits; the property
test decodes one of them, tests/data/rotated_planar_code_d5_p0.003.stim,
at the costs fitted to the record and compares the mean.
"""

import dataclasses
import itertools
import pathlib

import numpy
import pytest
import stim

import decsim.config as config
import decsim.decoders.union_find.decoder as union_find
import decsim.decoders.union_find.serial_chip_time as serial_chip_time
import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.records.fault_model_contracts as fault_models
import tests.decoders.test_union_find_cycle_count as traced
import tests.decoders.windows as windows

# 2001.06598 Sec. III C: four cycles a memory access at 4 GHz
PAPER_CLOCK = config.Clock.from_megahertz(4000.0)
PAPER_READ_COUNT = serial_chip_time.ClusterReadCount(
    clock=PAPER_CLOCK, cycles_per_read=4
)
READ_TICKS = 4 * PAPER_CLOCK.period_ticks
# two detectors of one type, one grid step apart in one round
ADJACENT_PAIR_PLACES = ((0.0, 0.0, 0.0), (2.0, 2.0, 0.0))
# a detector, its neighbour one grid step away, and itself a round later
QUIET_NEIGHBOUR_PLACES = ((2.0, 2.0, 0.0), (4.0, 4.0, 0.0), (2.0, 2.0, 1.0))
# detectors 0 and 1 share fault 0, detectors 2 and 3 share fault 1, and
# each detector has a boundary edge of its own
TWO_PAIR_CHECK = (
    (1, 0, 1, 0, 0, 0),
    (1, 0, 0, 1, 0, 0),
    (0, 1, 0, 0, 1, 0),
    (0, 1, 0, 0, 0, 1),
)
# the first pair on one type's grid, the second on the other's
TWO_PAIR_PLACES = (
    (0.0, 0.0, 0.0),
    (2.0, 2.0, 0.0),
    (2.0, 0.0, 0.0),
    (4.0, 2.0, 0.0),
)
PAPER_DISTANCE = 11
PAPER_ERROR_RATE = 0.001
PAPER_MEAN_NANOSECONDS = 42.0
SAMPLE_COUNT = 4000
# costs for the hand cases, each its own number so that a sum shows
# every term; they are no chip's
HAND_CLOCK = config.Clock.from_megahertz(400.0)
HAND_PAIR_COUNT = serial_chip_time.PairCompareCount(
    clock=HAND_CLOCK,
    syndrome_word_bits=32,
    load_cycles=9.0,
    load_cycles_per_word=2.0,
    load_cycles_per_defect=22.0,
    grow_cycles_per_pass=0.25,
    grow_cycles_per_entry=7.0,
    grow_cycles_per_grown_entry=0.5,
    grow_cycles_per_root_step=1.5,
    grow_cycles_per_boundary_hit=5.5,
    match_cycles_per_step=5.0,
    match_cycles_per_pair=1.0,
    match_cycles_per_first_collision=13.0,
    match_cycles_per_boundary_hit=8.0,
    union_cycles_per_step=1.75,
)
# 2309.05558's data record: its circuit of distance 5 at p = 0.003 and
# the mean cycles of a decode of it (fpga_accuracy_data.csv,
# total_cycles), on its FPGA's clock (Sec. III)
THIS_FILE = pathlib.Path(__file__)
DATA_DIRECTORY = THIS_FILE.parents[1] / "data"
RECORD_CIRCUIT = DATA_DIRECTORY / "rotated_planar_code_d5_p0.003.stim"
RECORD_ROUND_COUNT = 5
RECORD_MEAN_CYCLES = 108.812
RECORD_CLOCK = config.Clock.from_megahertz(400.0)
# the record gives the mean cycles of Init, Grow and Merge for 77 cases,
# distance 3 to 23 at p = 0.001 to 0.009, and no cycles of an operation;
# these are the least squares of the loop's operations on its circuits
# against those stage cycles, one set for all cases, with the words of
# 32 bits that fit its Init
RECORD_PAIR_COUNT = serial_chip_time.PairCompareCount(
    clock=RECORD_CLOCK,
    syndrome_word_bits=32,
    load_cycles=8.756,
    load_cycles_per_word=1.055,
    load_cycles_per_defect=22.14,
    grow_cycles_per_pass=0.0986,
    grow_cycles_per_entry=7.316,
    grow_cycles_per_grown_entry=0.4987,
    grow_cycles_per_root_step=1.114,
    grow_cycles_per_boundary_hit=5.579,
    match_cycles_per_step=5.185,
    match_cycles_per_pair=0.9959,
    match_cycles_per_first_collision=12.83,
    match_cycles_per_boundary_hit=8.225,
    union_cycles_per_step=1.422,
)
# a step above every edge's weight makes each edge one tick: the paper's
# chip has no weights
NO_WEIGHT_STEP = 100.0
RECORD_SAMPLE_COUNT = 4000


def placed_evidence(rows, syndrome, places):
    """One decode of a traced machine, with its detectors' coordinates."""
    graph = traced.traced_graph(rows)
    bits = numpy.asarray(syndrome, dtype=numpy.uint8)
    evidence = window_decoder.decode_graph(graph, bits)
    return dataclasses.replace(evidence, detector_coordinates=places)


def traced_evidence(rows, syndrome):
    """One decode of a traced machine, every edge two half ticks long."""
    graph = traced.traced_graph(rows)
    bits = numpy.asarray(syndrome, dtype=numpy.uint8)
    return window_decoder.decode_graph(graph, bits)


def test_two_adjacent_defects_cost_nine_reads():
    """One cluster of two detectors and diameter two: (1 + 4) + 2 x 2."""
    evidence = placed_evidence(
        traced.ADJACENT_PAIR_CHECK, [1, 1], ADJACENT_PAIR_PLACES
    )

    assert PAPER_READ_COUNT.reads(evidence) == [9, 0]
    assert PAPER_READ_COUNT.decode_ticks(evidence, 0, 0) == 9 * READ_TICKS


def test_a_lone_defect_takes_in_its_quiet_neighbours_and_costs_eleven_reads():
    """Three detectors one step across in space and in time: 5 + 2 x 3."""
    evidence = placed_evidence(
        traced.QUIET_NEIGHBOUR_CHECK, [1, 0, 0], QUIET_NEIGHBOUR_PLACES
    )

    assert PAPER_READ_COUNT.reads(evidence) == [11, 0]


def test_the_two_types_chips_work_side_by_side():
    """A pair on each type's grid is nine reads a chip, held for nine."""
    evidence = placed_evidence(TWO_PAIR_CHECK, [1, 1, 1, 1], TWO_PAIR_PLACES)

    assert PAPER_READ_COUNT.reads(evidence) == [9, 9]
    assert PAPER_READ_COUNT.decode_ticks(evidence, 0, 0) == 9 * READ_TICKS


def test_a_decode_with_no_detection_event_takes_no_time_by_its_reads():
    evidence = placed_evidence(
        traced.ADJACENT_PAIR_CHECK, [0, 0], ADJACENT_PAIR_PLACES
    )

    assert PAPER_READ_COUNT.reads(evidence) == [0, 0]
    assert PAPER_READ_COUNT.decode_ticks(evidence, 0, 0) == 0


def test_a_decode_whose_window_has_no_coordinates_is_refused():
    graph = traced.traced_graph(traced.ADJACENT_PAIR_CHECK)
    syndrome = numpy.asarray([1, 1], dtype=numpy.uint8)
    evidence = window_decoder.decode_graph(graph, syndrome)

    with pytest.raises(ValueError, match="carries no detector coordinates"):
        PAPER_READ_COUNT.decode_ticks(evidence, 0, 0)


def test_negative_cycles_a_read_are_refused():
    with pytest.raises(ValueError, match="must not be negative"):
        serial_chip_time.ClusterReadCount(clock=PAPER_CLOCK, cycles_per_read=-1)


def test_the_read_count_refuses_to_price_growth_past_the_decode():
    with pytest.raises(ValueError, match="prices the decode only"):
        PAPER_READ_COUNT.extra_growth_ticks((), 0)


def test_the_union_find_row_hands_its_windows_coordinates_to_the_law():
    settings = union_find.UnionFindDecoder.Settings(
        weight_step=traced.ONE_TICK_WEIGHT_STEP, timing=PAPER_READ_COUNT
    )
    decoder = settings.build()
    model = pair_window_model()
    backend = decoder.compile(model.graphlike_faults, model)
    syndrome = numpy.asarray([1, 1], dtype=numpy.uint8)

    decode = decoder.decode_window(
        backend, model, model.graphlike_faults, syndrome
    )

    evidence = decode.cluster_evidence
    assert evidence.detector_coordinates == ADJACENT_PAIR_PLACES
    assert PAPER_READ_COUNT.reads(evidence) == [9, 0]


def test_a_decode_on_matrices_alone_carries_no_coordinates():
    """A caller with no window model, as the machine tests' referee is."""
    settings = union_find.UnionFindDecoder.Settings(
        weight_step=traced.ONE_TICK_WEIGHT_STEP, timing=PAPER_READ_COUNT
    )
    decoder = settings.build()
    model = pair_window_model()
    faults = model.graphlike_faults
    backend = decoder.compile(faults, None)
    syndrome = numpy.asarray([1, 1], dtype=numpy.uint8)

    decode = decoder.decode_window(backend, None, faults, syndrome)

    evidence = decode.cluster_evidence
    assert evidence.detector_coordinates is None


def test_the_read_count_gives_the_papers_mean_time_on_its_own_case_property():
    """42 ns for d = 11, p = 0.001 (HPCA 2022 Sec. IV E), within 1.5 ns.

    The mean of 4,000 decodes has a standard error near 0.4 ns.
    """
    check, places = planar_case(PAPER_DISTANCE)
    graph = traced.traced_graph(check)
    generator = numpy.random.default_rng(20261117)
    total_ticks = 0
    for _ in range(SAMPLE_COUNT):
        draws = generator.random(check.shape[1])
        errors = draws < PAPER_ERROR_RATE
        syndrome = (check @ errors) % 2
        evidence = window_decoder.decode_graph(graph, syndrome)
        placed = dataclasses.replace(evidence, detector_coordinates=places)
        total_ticks += PAPER_READ_COUNT.decode_ticks(placed, 0, 0)

    mean_ticks = total_ticks / SAMPLE_COUNT
    mean_nanoseconds = 1000 * config.ticks_to_microseconds(mean_ticks)
    assert mean_nanoseconds == pytest.approx(PAPER_MEAN_NANOSECONDS, abs=1.5)


def test_two_adjacent_defects_cost_their_load_two_grows_and_one_match():
    """55 to load, 15.25 and 20.75 for the pass that meets, 15.75 to end.

    The load is 9 + 2 + 2 x 22. The pass in which the two meet grows
    for 0.25 + 2 x 7 + 2 x 0.5 and matches for 5 + 1.75 + 1 + 13. The
    last pass finds one entry a step from its root: 0.25 + 2 x 7 + 1.5.
    """
    evidence = traced_evidence(traced.ADJACENT_PAIR_CHECK, [1, 1])

    assert HAND_PAIR_COUNT.cycles(evidence) == [106.75]
    whole_cycle_ticks = 107 * HAND_CLOCK.period_ticks
    assert HAND_PAIR_COUNT.decode_ticks(evidence, 0, 0) == whole_cycle_ticks


def test_a_defect_that_reaches_the_boundary_pays_in_its_grow_and_its_match():
    """33 to load, two passes of 7.75 and 6.75, 13.5 at the boundary, 7.25.

    The load is 9 + 2 + 22. Two passes grow the lone defect, each
    0.25 + 7 + 0.5 and 5 + 1.75; the second reaches the boundary, 5.5
    in its grow and 8 in its match. The last pass is 0.25 + 7.
    """
    evidence = traced_evidence(traced.QUIET_NEIGHBOUR_CHECK, [1, 0, 0])

    assert HAND_PAIR_COUNT.cycles(evidence) == [82.75]


def test_a_decode_with_no_defect_costs_its_load_and_one_grow_pass():
    """9 + 2 for the load of one word and 0.25 for the pass."""
    evidence = traced_evidence(traced.ADJACENT_PAIR_CHECK, [0, 0])

    assert HAND_PAIR_COUNT.cycles(evidence) == [11.25]


def test_a_syndrome_is_loaded_a_word_at_a_time():
    """Three detectors in words of two bits: 9 + 2 x 2, and 0.25."""
    two_bit_words = dataclasses.replace(HAND_PAIR_COUNT, syndrome_word_bits=2)
    evidence = traced_evidence(traced.QUIET_NEIGHBOUR_CHECK, [0, 0, 0])

    assert two_bit_words.cycles(evidence) == [13.25]


def test_each_connected_part_has_a_chip_and_the_slower_holds_the_decode():
    """A pair in one part, 106.75, and a lone defect in the other, 82.75."""
    evidence = traced_evidence(TWO_PAIR_CHECK, [1, 1, 1, 0])

    assert HAND_PAIR_COUNT.cycles(evidence) == [106.75, 82.75]
    whole_cycle_ticks = 107 * HAND_CLOCK.period_ticks
    assert HAND_PAIR_COUNT.decode_ticks(evidence, 0, 0) == whole_cycle_ticks


def test_a_negative_cost_of_an_operation_is_refused():
    with pytest.raises(ValueError, match="finite nonnegative number"):
        dataclasses.replace(HAND_PAIR_COUNT, match_cycles_per_pair=-1.0)


def test_a_syndrome_word_of_no_bits_is_refused():
    with pytest.raises(ValueError, match="a whole number of bits"):
        dataclasses.replace(HAND_PAIR_COUNT, syndrome_word_bits=0)


def test_the_pair_count_refuses_to_price_growth_past_the_decode():
    with pytest.raises(ValueError, match="prices the decode only"):
        HAND_PAIR_COUNT.extra_growth_ticks((), 0)


def test_the_pair_count_gives_the_records_mean_cycles_on_its_case_property():
    """108.8 cycles for d = 5, p = 0.003, within 5 cycles.

    The costs are fitted to the record, this case among its 77, so the
    test holds the loop and those costs to the record together; it does
    not show the costs on a case outside it. The mean of 4,000 decodes
    has a standard error near 1.6 cycles.
    """
    circuit = stim.Circuit.from_file(RECORD_CIRCUIT)
    model = windows.whole_circuit_window(
        circuit, RECORD_ROUND_COUNT, fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    )
    graph = window_decoder.graph_from_model(
        model.graphlike_faults,
        location="the record's circuit",
        weight_step=NO_WEIGHT_STEP,
    )
    shots, _ = windows.sampled_shots(circuit, RECORD_SAMPLE_COUNT, 20261118)
    total_cycles = 0.0
    for shot in shots:
        syndrome = windows.row_syndrome(model, shot)
        evidence = window_decoder.decode_graph(graph, syndrome)
        chip_cycles = RECORD_PAIR_COUNT.cycles(evidence)
        total_cycles += max(chip_cycles)

    mean_cycles = total_cycles / RECORD_SAMPLE_COUNT
    assert mean_cycles == pytest.approx(RECORD_MEAN_CYCLES, abs=5.0)


def pair_window_model() -> fault_models.WindowErrorModel:
    """The adjacent pair as a window model with its coordinates."""
    check = numpy.asarray(traced.ADJACENT_PAIR_CHECK, dtype=numpy.uint8)
    fault_count = check.shape[1]
    priors = numpy.full(fault_count, traced.TRACED_PRIOR)
    observables = numpy.zeros((1, fault_count), dtype=numpy.uint8)
    owned = numpy.ones(fault_count, dtype=bool)
    placed = fault_models.PlacedFaultModel(
        representation=traced.GRAPHLIKE,
        check=check,
        priors=priors,
        observables=observables,
        owned=owned,
        source_fault_ids=tuple(range(fault_count)),
        boundary_flips={},
    )
    return fault_models.WindowErrorModel(
        detector_ids=(0, 1),
        detector_coordinates=ADJACENT_PAIR_PLACES,
        defect_positions={0: (1, 0), 1: (1, 1)},
        first_commit_round=1,
        graphlike_faults=placed,
        physical_faults=None,
    )


def planar_case(distance: int) -> tuple:
    """The paper's graph as a check matrix, and its detectors' coordinates.

    Vertex (column, row, layer) has distance - 1 columns, distance rows
    and distance layers. A vertex sits at Stim's (2 (column + row),
    2 (column - row), layer), so that a step along a row or a column is
    one step of the grid.
    """
    vertices = planar_vertices(distance)
    edges = planar_edges(vertices)
    check = numpy.zeros((len(vertices), len(edges)), dtype=numpy.uint8)
    for fault, ends in enumerate(edges):
        detectors = [end for end in ends if end is not None]
        check[detectors, fault] = 1
    places = []
    for column, row, layer in vertices:
        diagonal = column + row
        other_diagonal = column - row
        place = (2.0 * diagonal, 2.0 * other_diagonal, float(layer))
        places.append(place)
    return check, tuple(places)


def planar_vertices(distance: int) -> dict:
    """Each vertex's row in the check matrix, by (column, row, layer)."""
    column_count = distance - 1
    layers = range(distance)
    rows = range(distance)
    columns = range(column_count)
    vertices = {}
    for layer, row, column in itertools.product(layers, rows, columns):
        vertices[column, row, layer] = len(vertices)
    return vertices


def planar_edges(vertices: dict) -> list:
    """Each edge's two ends; None is the boundary.

    A layer has an edge from each vertex along its row, the two outer
    ones to the boundary, and one along its column where a next row
    exists. Every vertex has an edge to the layer above and one to the
    layer below, to the boundary at the top and at the bottom.
    """
    edges = []
    for column, row, layer in vertices:
        here = vertices[column, row, layer]
        next_column = column + 1
        next_row = row + 1
        next_layer = layer + 1
        along_row = vertices.get((next_column, row, layer))
        along_column = vertices.get((column, next_row, layer))
        above = vertices.get((column, row, next_layer))
        edges.append((here, along_row))
        edges.append((here, above))
        if along_column is not None:
            edges.append((here, along_column))
        if column == 0:
            edges.append((here, None))
        if layer == 0:
            edges.append((here, None))
    return edges
