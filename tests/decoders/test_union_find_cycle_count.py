"""The Union-Find cycle count: the price of the steps a decode reports.

The price is the one line of cycle_count.py, checked on hand-built
steps; a unit under the count is held for exactly those cycles to the
clock's edge, the way a measured unit is held for its host time
(decoder.py, DecoderBase._start_measured), and a real decode on the
twelve-detector graph of test_union_find_decoder.py lands on the tick
the count names. The steps themselves are checked in
test_union_find_compiled_decoder.py.
"""

import math

import numpy
import pytest

import decsim.config as config
import decsim.decoders.staged_decoder as staged_decoder
import decsim.decoders.union_find.cycle_count as cycle_count_module
import decsim.decoders.union_find.decoder as union_find
import decsim.engine as engine_module
import decsim.records.decoder_evidence as evidence_records
import tests.decoders.test_union_find_decoder as hand_graph

MEGAHERTZ = 100.0
PERIOD_MICROSECONDS = 1 / MEGAHERTZ
CYCLE_TICKS = config.microseconds_to_ticks(PERIOD_MICROSECONDS)
CLOCK = config.Clock(CYCLE_TICKS)
WEIGHT_STEP = 0.1
# Helios's row: grow 1, wait 2, decide 1 per step (2301.08419 lines
# 694-699, 921, 923-929); three floods per unit of depth (lines
# 876-889); a load and readout of 11 cycles fitted to the d = 13 point
HELIOS = cycle_count_module.CycleCount(
    CLOCK, setup_cycles=11, cycles_per_step=4, cycles_per_hop=3
)


def evidence_with(steps, detector_count=5, edge_count=4):
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
    )


def test_the_count_is_setup_plus_the_steps_plus_the_drain():
    """Each step is the larger of its critical path and its spread work."""
    steps = [
        evidence_records.GrowthStep(
            edge_count=3, hop_count=1, growth_ticks=1, odd_fusion=True
        ),
        evidence_records.GrowthStep(
            edge_count=10, hop_count=2, growth_ticks=1, odd_fusion=True
        ),
    ]
    evidence = evidence_with(steps, detector_count=5, edge_count=4)
    assert HELIOS.cycles(evidence) == 11 + (4 + 3) + (4 + 6)
    memory_bound = cycle_count_module.CycleCount(
        CLOCK,
        setup_cycles=1,
        setup_cycles_per_vertex=2,
        setup_cycles_per_edge=3,
        cycles_per_step=4,
        cycles_per_hop=3,
        cycles_per_edge=4,
        drain_cycles=5,
    )
    setup = 1 + 2 * 5 + 3 * 4
    assert memory_bound.cycles(evidence) == setup + 12 + 40 + 5
    half_a_cycle_an_edge = cycle_count_module.CycleCount(
        CLOCK, cycles_per_edge=0.5
    )
    assert half_a_cycle_an_edge.cycles(evidence) == 2 + 5
    assert HELIOS.cycles(None) == 0


def test_the_count_ends_on_the_edge_of_its_own_clock():
    """The ticks run to the edge the cycles end on, from where the unit is.

    A unit standing mid-cycle first reaches the next edge, as every
    stage does (config.Clock.edge, after gem5's clockEdge).
    """
    one_step = evidence_records.GrowthStep(
        edge_count=3, hop_count=1, growth_ticks=1, odd_fusion=True
    )
    evidence = evidence_with([one_step])
    no_step = evidence_with([])
    assert HELIOS.ticks(evidence, 0) == 18 * CYCLE_TICKS
    assert HELIOS.ticks(evidence, 1) == 19 * CYCLE_TICKS - 1
    assert HELIOS.ticks(no_step, 1) == 12 * CYCLE_TICKS - 1
    assert HELIOS.ticks(None, 7) == 0


def test_a_counted_unit_is_held_for_the_counted_cycles():
    """The decode runs at the start and the unit frees when the count ends.

    Two adjacent defects take one step of one hop, so Helios's row
    holds the unit 11 + 4 + 3 cycles of its 100 MHz clock; the unit
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
    assert tick == 18 * CYCLE_TICKS
    assert selected[9] == 1
    assert sum(selected) == 1
    assert unit.occupancy(job) is None


def test_a_negative_field_is_refused_by_name():
    with pytest.raises(ValueError, match="cycle_count.cycles_per_hop"):
        cycle_count_module.CycleCount(CLOCK, cycles_per_hop=-1)
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
    block = {"clock": "helios", "cycles_per_step": 4, "cycles_per_hop": 3}
    count = cycle_count_module.CycleCount.from_yaml(block, clocks)
    assert count.clock == CLOCK
    assert count.cycles_per_step == 4
    assert count.cycles_per_hop == 3
    assert count.setup_cycles == 0
    assert count.cycles_per_edge == 0.0
    assert count.drain_cycles == 0
