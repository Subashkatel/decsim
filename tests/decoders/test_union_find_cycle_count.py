"""The Union-Find cycle count: Helios's traced law on the steps it reads.

The law is the one line of cycle_count.py, traced through Helios's
register transfer level (github.com/yale-paragon/Helios_scalable_QEC at
2dda998, the design behind 2301.08419v2) and checked here on the three
machine states that trace gives a number for: an empty syndrome, two
adjacent defects, and a lone defect beside the boundary. A unit under
the count is held for exactly those cycles to the clock's edge, the way
a measured unit is held for its host time (decoder.py,
DecoderBase._start_measured), and a real decode on the twelve-detector
graph of test_union_find_decoder.py lands on the tick the count names.
The steps themselves are checked in test_union_find_compiled_decoder.py.
"""

import math

import numpy
import pytest

import decsim.config as config
import decsim.decoders.staged_decoder as staged_decoder
import decsim.decoders.union_find.cycle_count as cycle_count_module
import decsim.decoders.union_find.decoder as union_find
import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.engine as engine_module
import decsim.records.decoder_evidence as evidence_records
import tests.decoders.test_union_find_decoder as hand_graph

MEGAHERTZ = 100.0
PERIOD_MICROSECONDS = 1 / MEGAHERTZ
CYCLE_TICKS = config.microseconds_to_ticks(PERIOD_MICROSECONDS)
CLOCK = config.Clock(CYCLE_TICKS)
WEIGHT_STEP = 0.1
# Helios's row: three registers between an element's change and the
# controller's check (Helios_single_FPGA_core.v line 76)
HELIOS = cycle_count_module.CycleCount(CLOCK, delay_cycles=3)
# every edge of the twelve-detector graph is this long in half ticks,
# and both ends of the one between two defects grow
HAND_GRAPH_TICKS = 22

GRAPHLIKE = fault_models.FaultRepresentation.GRAPHLIKE
# detector 0 carries the defect; faults 0 and 1 reach the quiet
# detectors 1 and 2, fault 2 is the boundary edge
QUIET_NEIGHBOUR_CHECK = ((1, 1, 1), (1, 0, 0), (0, 1, 0))
QUIET_NEIGHBOUR_PRIORS = (0.1, 0.1, 0.1)
# the log-odds of that prior, so round(weight / weight_step) is one
ONE_TICK_WEIGHT_STEP = 2.0


def lone_defect_graph() -> evidence_records.UnionFindGraph:
    """Detector 0 with two quiet neighbours and a boundary edge.

    Every prior is the same, and the weight step is that prior's
    log-odds, so every edge is one weight tick, two half ticks, which is
    the weight Helios gives every edge of its grid
    (single_FPGA_FIFO_verification_test_rsc.sv, WEIGHT_X and WEIGHT_Z).
    """
    check = numpy.asarray(QUIET_NEIGHBOUR_CHECK, dtype=numpy.uint8)
    priors = numpy.asarray(QUIET_NEIGHBOUR_PRIORS, dtype=float)
    fault_count = len(QUIET_NEIGHBOUR_PRIORS)
    observables = numpy.zeros((1, fault_count), dtype=numpy.uint8)
    owned = numpy.ones(fault_count, dtype=bool)
    placed = fault_models.PlacedFaultModel(
        representation=GRAPHLIKE,
        check=check,
        priors=priors,
        observables=observables,
        owned=owned,
        source_fault_ids=tuple(range(fault_count)),
        boundary_flips={},
    )
    return window_decoder.graph_from_model(
        placed, location="one defect", weight_step=ONE_TICK_WEIGHT_STEP
    )


def evidence_with(steps, detector_count=5, edge_count=4, forest_depth=0):
    """A hand-built evidence carrying only what the count reads."""
    edges = []
    for index in range(edge_count):
        next_detector = index + 1
        edge = evidence_records.UnionFindEdge(
            fault_index=index,
            detector_a=index,
            detector_b=next_detector,
            logical_observables=(),
            length_half_ticks=2,
        )
        edges.append(edge)
    baseline_syndrome = (0,) * detector_count
    placed = evidence_records.UnionFindGraph(
        detector_count=detector_count,
        fault_count=edge_count,
        edges=tuple(edges),
        baseline_faults=(),
        baseline_syndrome=baseline_syndrome,
    )
    return evidence_records.UnionFindHardEvidence(
        graph=placed,
        syndrome=(),
        residual_syndrome=(),
        selected_faults=(),
        contact_faults=(),
        edge_intervals=(),
        erasure_forest_faults=(),
        logical_observables=(),
        growth_steps=tuple(steps),
        forest_depth=forest_depth,
    )


def test_an_empty_syndrome_costs_the_quiet_machines_eleven_cycles():
    """One quiet iteration and a peel that finds nothing to peel.

    1 + (2 + 3) + (3 + 2), the trace of Helios's controller from the
    cycle its counter starts at to the cycle it leaves PEELING.
    """
    evidence = evidence_with([])
    assert HELIOS.cycles(evidence) == 11


def test_two_adjacent_defects_cost_sixteen_cycles():
    """One iteration whose fusion moves a parity, then a peel of depth one.

    The edge between them is two half ticks and both ends grow, so the
    step is one growth tick: 1 + (2 + 3) + 3 x 1 + (3 + 2 + 2).
    """
    step = evidence_records.GrowthStep(
        edge_count=3, hop_count=1, growth_ticks=1, fusion="parity"
    )
    evidence = evidence_with([step], forest_depth=1)
    assert HELIOS.cycles(evidence) == 16


def test_a_lone_defect_beside_the_boundary_costs_nineteen_cycles():
    """The third traced machine, decoded: two ticks, a quiet join, a peel.

    One defect with two quiet neighbours and a boundary edge, every edge
    two half ticks. The edges grow from the defect's end alone, so the
    step spans two iterations; the neighbours and the boundary it takes
    in move roots and the touching flag only; the peel walks the one
    level below the boundary: 1 + 2 x (2 + 3) + 1 + (3 + 2 + 2).
    """
    graph = lone_defect_graph()
    syndrome = numpy.asarray([1, 0, 0], dtype=numpy.uint8)
    evidence = window_decoder.decode_graph(graph, syndrome)

    lengths = set()
    for edge in graph.edges:
        lengths.add(edge.length_half_ticks)
    assert lengths == {2}
    assert evidence.forest_depth == 1
    assert HELIOS.cycles(evidence) == 19


def test_a_steps_changes_are_its_fusion_kind_over_its_deepest_flood():
    """Nothing, a flood of roots, or a root and two cycles a level.

    A step that unites nothing changes nothing; one that only moves
    roots and the touching flag settles a level a cycle; one that moves
    a parity moves the root once and then climbs the parity while the
    odd flag descends, 1 + 2 x the flood's depth.
    """
    quiet = evidence_records.GrowthStep(
        edge_count=1, hop_count=4, growth_ticks=1, fusion="none"
    )
    roots = evidence_records.GrowthStep(
        edge_count=1, hop_count=4, growth_ticks=1, fusion="roots"
    )
    parity = evidence_records.GrowthStep(
        edge_count=1, hop_count=4, growth_ticks=1, fusion="parity"
    )
    floor = 1 + 5 + 5
    quiet_evidence = evidence_with([quiet])
    roots_evidence = evidence_with([roots])
    parity_evidence = evidence_with([parity])

    assert HELIOS.cycles(quiet_evidence) == floor
    assert HELIOS.cycles(roots_evidence) == floor + 4
    assert HELIOS.cycles(parity_evidence) == floor + 9


def test_a_unit_that_walks_its_edges_pays_its_port_instead_of_its_changes():
    """The step costs the larger of its changes and its work per edge.

    Three edges advanced over one growth tick at four cycles an edge is
    twelve, past the three cycles the fusion's parity takes.
    """
    step = evidence_records.GrowthStep(
        edge_count=3, hop_count=1, growth_ticks=1, fusion="parity"
    )
    evidence = evidence_with([step], forest_depth=1)
    memory_bound = cycle_count_module.CycleCount(
        CLOCK, delay_cycles=3, cycles_per_edge=4
    )
    assert memory_bound.cycles(evidence) == 1 + 5 + 12 + 7


def test_a_decode_with_no_steps_pays_its_setup_and_one_quiet_iteration():
    """With no delay: the counter's first cycle, an iteration and a peel."""
    evidence = evidence_with([], detector_count=5, edge_count=4)
    laid_out = cycle_count_module.CycleCount(
        CLOCK,
        setup_cycles=1,
        setup_cycles_per_vertex=2,
        setup_cycles_per_edge=3,
    )
    setup = 1 + 2 * 5 + 3 * 4
    assert laid_out.cycles(evidence) == 1 + 2 + 2 + setup
    assert HELIOS.cycles(None) == 0


def test_the_count_ends_on_the_edge_of_its_own_clock():
    """The ticks run to the edge the cycles end on, from where the unit is.

    A unit standing mid-cycle first reaches the next edge, as every
    stage does (config.Clock.edge, after gem5's clockEdge).
    """
    one_step = evidence_records.GrowthStep(
        edge_count=3, hop_count=1, growth_ticks=1, fusion="parity"
    )
    evidence = evidence_with([one_step], forest_depth=1)
    no_step = evidence_with([])
    assert HELIOS.ticks(evidence, 0) == 16 * CYCLE_TICKS
    assert HELIOS.ticks(evidence, 1) == 17 * CYCLE_TICKS - 1
    assert HELIOS.ticks(no_step, 1) == 12 * CYCLE_TICKS - 1
    assert HELIOS.ticks(None, 7) == 0


def test_a_counted_unit_is_held_for_the_counted_cycles():
    """The decode runs at the start and the unit frees when the count ends.

    Two adjacent defects of the hand graph fuse over an edge of 44 half
    ticks, so the step spans 22 growth ticks and Helios's row holds the
    unit 1 + 22 x 5 + 3 + 7 cycles of its 100 MHz clock; the unit
    cannot say so in advance, as a measured unit cannot.
    """
    engine = engine_module.Engine()
    decoder = union_find.UnionFindDecoder(cycle_count=HELIOS)
    timing = staged_decoder.UnitTiming((), (), CLOCK)
    unit = staged_decoder.StagedDecoder(decoder, timing)
    model = hand_graph._model()
    syndrome = numpy.zeros(hand_graph.DETECTOR_COUNT, dtype=numpy.uint8)
    syndrome[[8, 9]] = 1
    job = hand_graph._job(model, syndrome)
    delivered = []

    def on_result(result):
        delivered.append((engine.now, result))

    unit.start(job, engine, on_result)
    engine.run()
    ((tick, result),) = delivered
    selected = result.cluster_evidence.selected_faults
    iteration_cycles = 5 * HAND_GRAPH_TICKS
    held_cycles = 1 + iteration_cycles + 3 + 7
    assert tick == held_cycles * CYCLE_TICKS
    assert selected[9] == 1
    assert sum(selected) == 1
    assert unit.occupancy(job) is None


def test_a_negative_field_is_refused_by_name():
    with pytest.raises(ValueError, match="cycle_count.setup_cycles"):
        cycle_count_module.CycleCount(CLOCK, setup_cycles=-1)
    with pytest.raises(ValueError, match="cycle_count.delay_cycles"):
        cycle_count_module.CycleCount(CLOCK, delay_cycles=-1)
    with pytest.raises(ValueError, match="cycles_per_edge"):
        cycle_count_module.CycleCount(CLOCK, cycles_per_edge=-0.5)
    with pytest.raises(ValueError, match="finite"):
        cycle_count_module.CycleCount(CLOCK, cycles_per_edge=math.nan)


def test_a_key_the_block_does_not_have_is_refused_by_name():
    clocks = config.ClockSettings({"helios": MEGAHERTZ})
    block = {"clock": "helios", "cycles_per_edeg": 4}
    with pytest.raises(ValueError, match="cycles_per_edeg"):
        cycle_count_module.CycleCount.from_yaml(block, clocks)


def test_the_yaml_block_resolves_its_clock_and_defaults_the_rest_to_zero():
    clocks = config.ClockSettings({"helios": MEGAHERTZ})
    block = {"clock": "helios", "delay_cycles": 3, "setup_cycles": 4}
    count = cycle_count_module.CycleCount.from_yaml(block, clocks)
    assert count.clock == CLOCK
    assert count.delay_cycles == 3
    assert count.setup_cycles == 4
    assert count.setup_cycles_per_edge == 0
    assert count.cycles_per_edge == 0.0
    assert count.setup_cycles_per_vertex == 0
