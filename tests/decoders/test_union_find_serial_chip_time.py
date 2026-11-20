"""A serial union-find chip's time, on hand cases and on the paper's case.

PairCompareCount prices the loop of Barber et al. 2309.05558. Its hand
cases are the two small machines of test_union_find_cycle_count.py and
a machine of two pairs, at costs chosen for the sums. The paper's data
record (Zenodo 10.5281/zenodo.11621877, CC BY 4.0) gives the mean
cycles of a decode on its FPGA for each of its circuits; the property
test decodes one of them, tests/data/rotated_planar_code_d5_p0.003.stim,
at the costs fitted to the record and compares the mean.
"""

import dataclasses
import pathlib

import numpy
import pytest
import stim

import decsim.config as config
import decsim.decoders.union_find.serial_chip_time as serial_chip_time
import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.records.fault_model_contracts as fault_models
import tests.decoders.test_union_find_cycle_count as traced
import tests.decoders.windows as windows

# detectors 0 and 1 share fault 0, detectors 2 and 3 share fault 1, and
# each detector has a boundary edge of its own
TWO_PAIR_CHECK = (
    (1, 0, 1, 0, 0, 0),
    (1, 0, 0, 1, 0, 0),
    (0, 1, 0, 0, 1, 0),
    (0, 1, 0, 0, 0, 1),
)
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


def traced_evidence(rows, syndrome):
    """One decode of a traced machine, every edge two half ticks long."""
    graph = traced.traced_graph(rows)
    bits = numpy.asarray(syndrome, dtype=numpy.uint8)
    return window_decoder.decode_graph(graph, bits)


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
