"""The Collision Clustering chip's loop and time, on hand cases.

The loop is that of Barber et al. 2309.05558 Sec. II A and III, and the
cycles of each operation are those collision_clustering.py reads from
the paper's data record. The hand cases are the two small machines of
test_union_find_cycle_count.py, every edge two half ticks long, and the
same pair with edges of four. The property test grows random detection
events on a grid of random weights and compares the loop's clusters
with those of decsim's own union-find.
"""

import collections
import math

import numpy
import pytest

import decsim.config as config
import decsim.decoders.cluster_record as cluster_record
import decsim.decoders.union_find.collision_clustering as chip_loop
import decsim.decoders.union_find.serial_chip_time as serial_chip_time
import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.records.fault_model_contracts as fault_models
import tests.decoders.test_union_find_cycle_count as traced

# 2309.05558 Table II: the chip's clock on a 12 nm process
PAPER_CLOCK = config.Clock.from_megahertz(2000.0)
CHIP_COUNT = serial_chip_time.CollisionClusteringCount(clock=PAPER_CLOCK)
# detectors 0 and 1 share fault 0, detectors 2 and 3 share fault 1, and
# each detector has a boundary edge of its own
TWO_PAIR_CHECK = (
    (1, 0, 1, 0, 0, 0),
    (1, 0, 0, 1, 0, 0),
    (0, 1, 0, 0, 1, 0),
    (0, 1, 0, 0, 0, 1),
)
# detectors 0 and 1 share two faults and no boundary edge
DOUBLE_EDGE_CHECK = ((1, 1), (1, 1))
# Init of a chip whose syndrome fits one word, before its defects
QUIET_INIT_CYCLES = chip_loop.INIT_CYCLES + chip_loop.INIT_CYCLES_PER_WORD
# a growth step's Merge before its pairs
QUIET_MERGE_CYCLES = (
    chip_loop.MATCH_CYCLES_PER_STEP + chip_loop.MERGE_CYCLES_PAST_MATCH
)
GRID_SIDE = 5
GRID_DETECTOR_COUNT = GRID_SIDE * GRID_SIDE
GRID_WEIGHT_STEP = 0.5
SAMPLE_COUNT = 300
EVENT_CHANCE = 0.15


def graph_of(rows, priors, weight_step):
    """A graph of the faults in rows, with the given prior a fault."""
    check = numpy.asarray(rows, dtype=numpy.uint8)
    fault_count = check.shape[1]
    observables = numpy.zeros((1, fault_count), dtype=numpy.uint8)
    owned = numpy.ones(fault_count, dtype=bool)
    prior_array = numpy.asarray(priors, dtype=float)
    placed = fault_models.PlacedFaultModel(
        representation=traced.GRAPHLIKE,
        check=check,
        priors=prior_array,
        observables=observables,
        owned=owned,
        source_fault_ids=tuple(range(fault_count)),
        boundary_flips={},
    )
    return window_decoder.graph_from_model(
        placed, location="collision clustering", weight_step=weight_step
    )


def decode_of(rows, syndrome):
    """One decode of a traced machine, every edge two half ticks long."""
    graph = traced.traced_graph(rows)
    bits = numpy.asarray(syndrome, dtype=numpy.uint8)
    return window_decoder.decode_graph(graph, bits)


def adjacent_pair_cycles() -> float:
    """Two defects one edge of two half ticks apart, by hand.

    The first pass grows both by one and they meet; the second finds
    the second entry one table step from its root and grows nothing.
    """
    init = QUIET_INIT_CYCLES + 2 * chip_loop.INIT_CYCLES_PER_DEFECT
    visit = chip_loop.GROW_CYCLES_PER_PASS + 2 * chip_loop.GROW_CYCLES_PER_ENTRY
    first_grow = visit + 2 * chip_loop.GROW_CYCLES_PER_GROWN_ENTRY
    first_merge = QUIET_MERGE_CYCLES + chip_loop.MATCH_CYCLES_PER_PAIR
    first_merge += chip_loop.MATCH_CYCLES_PER_FIRST_COLLISION
    last_grow = visit + chip_loop.GROW_CYCLES_PER_ROOT_STEP
    return init + first_grow + first_merge + last_grow


def test_two_adjacent_defects_meet_in_one_pass():
    graph = traced.traced_graph(traced.ADJACENT_PAIR_CHECK)
    (chip,) = chip_loop.chips_of(graph)
    defects = numpy.asarray([0, 1])

    passes, root = chip_loop.passes_of(chip, defects)

    assert passes == [
        chip_loop.Pass(2, 2, 0, 0, 1, 1),
        chip_loop.Pass(2, 0, 1, 0, 0, 0),
    ]
    assert root.tolist() == [0, 0]


def test_two_adjacent_defects_cost_their_init_two_grows_and_one_merge():
    evidence = decode_of(traced.ADJACENT_PAIR_CHECK, [1, 1])

    cycles = CHIP_COUNT.cycles(evidence)

    expected = adjacent_pair_cycles()
    assert cycles == [pytest.approx(expected)]
    whole_cycles = math.ceil(expected)
    expected_ticks = whole_cycles * PAPER_CLOCK.period_ticks
    assert CHIP_COUNT.decode_ticks(evidence, 0, 0) == expected_ticks


def test_a_lone_defect_grows_until_it_reaches_the_boundary():
    """Its boundary edge is two half ticks: the second pass reaches it."""
    graph = traced.traced_graph(traced.QUIET_NEIGHBOUR_CHECK)
    (chip,) = chip_loop.chips_of(graph)
    defects = numpy.asarray([0])

    passes, _ = chip_loop.passes_of(chip, defects)

    assert passes == [
        chip_loop.Pass(1, 1, 0, 0, 0, 0),
        chip_loop.Pass(1, 1, 0, 1, 0, 0),
        chip_loop.Pass(1, 0, 0, 0, 0, 0),
    ]


def test_a_boundary_hit_costs_its_grow_and_its_match():
    evidence = decode_of(traced.QUIET_NEIGHBOUR_CHECK, [1, 0, 0])

    cycles = CHIP_COUNT.cycles(evidence)

    init = QUIET_INIT_CYCLES + chip_loop.INIT_CYCLES_PER_DEFECT
    visit = chip_loop.GROW_CYCLES_PER_PASS + chip_loop.GROW_CYCLES_PER_ENTRY
    growing = visit + chip_loop.GROW_CYCLES_PER_GROWN_ENTRY
    hit = chip_loop.GROW_CYCLES_PER_BOUNDARY_HIT
    hit += chip_loop.MATCH_CYCLES_PER_BOUNDARY_HIT
    expected = init + 2 * growing + 2 * QUIET_MERGE_CYCLES + hit + visit
    assert cycles == [pytest.approx(expected)]


def test_a_decode_with_no_defect_costs_its_init_and_one_grow_pass():
    evidence = decode_of(traced.ADJACENT_PAIR_CHECK, [0, 0])

    cycles = CHIP_COUNT.cycles(evidence)

    expected = QUIET_INIT_CYCLES + chip_loop.GROW_CYCLES_PER_PASS
    assert cycles == [pytest.approx(expected)]


def test_each_connected_part_has_a_chip_and_the_slower_holds_the_decode():
    """A pair in one part and a lone defect in the other."""
    evidence = decode_of(TWO_PAIR_CHECK, [1, 1, 1, 0])

    cycles = CHIP_COUNT.cycles(evidence)

    assert len(cycles) == 2
    pair_cycles = adjacent_pair_cycles()
    assert cycles[0] == pytest.approx(pair_cycles)
    slowest = max(cycles)
    whole_cycles = math.ceil(slowest)
    expected_ticks = whole_cycles * PAPER_CLOCK.period_ticks
    assert CHIP_COUNT.decode_ticks(evidence, 0, 0) == expected_ticks


def test_an_edge_twice_as_long_takes_twice_the_growth_passes():
    """Edges of four half ticks: the pair meets in the second pass."""
    priors = [traced.TRACED_PRIOR] * 3
    half_step = traced.ONE_TICK_WEIGHT_STEP / 2
    graph = graph_of(traced.ADJACENT_PAIR_CHECK, priors, half_step)
    (chip,) = chip_loop.chips_of(graph)
    defects = numpy.asarray([0, 1])

    passes, _ = chip_loop.passes_of(chip, defects)

    assert chip.between[0, 1] == 4
    assert passes == [
        chip_loop.Pass(2, 2, 0, 0, 1, 0),
        chip_loop.Pass(2, 2, 0, 0, 1, 1),
        chip_loop.Pass(2, 0, 1, 0, 0, 0),
    ]


def test_two_detectors_are_as_far_apart_as_their_shortest_edge():
    """Two faults on the same two detectors, one likelier and shorter."""
    graph = graph_of(DOUBLE_EDGE_CHECK, [0.1, 0.01], 1.0)

    (chip,) = chip_loop.chips_of(graph)

    lengths = [edge.length_half_ticks for edge in graph.edges]
    assert chip.between[0, 1] == min(lengths)
    assert min(lengths) < max(lengths)


def test_a_part_with_no_boundary_edge_is_never_at_the_boundary():
    graph = graph_of(DOUBLE_EDGE_CHECK, [0.1, 0.01], 1.0)

    (chip,) = chip_loop.chips_of(graph)

    assert chip.to_boundary.tolist() == [chip_loop.NO_BOUNDARY] * 2


def test_the_chips_of_a_graph_are_laid_out_once():
    graph = traced.traced_graph(TWO_PAIR_CHECK)

    first = chip_loop.chips_of(graph)
    second = chip_loop.chips_of(graph)

    assert first is second
    parts = [chip.detectors.tolist() for chip in first]
    assert parts == [[0, 1], [2, 3]]


def test_the_chip_count_refuses_to_price_growth_past_the_decode():
    with pytest.raises(ValueError, match="prices the decode only"):
        CHIP_COUNT.extra_growth_ticks((), 0)


def grid_faults() -> list:
    """A square grid's faults: a fault each neighbour pair, and the rim's."""
    index = numpy.arange(GRID_DETECTOR_COUNT)
    grid = index.reshape(GRID_SIDE, GRID_SIDE)
    left = grid[:, :-1].ravel()
    right = grid[:, 1:].ravel()
    upper = grid[:-1, :].ravel()
    lower = grid[1:, :].ravel()
    across = numpy.stack([left, right], axis=1)
    down = numpy.stack([upper, lower], axis=1)
    rim = numpy.concatenate([grid[:, 0], grid[:, -1]])
    lone = [[detector] for detector in rim.tolist()]
    return across.tolist() + down.tolist() + lone


def grid_rows():
    """The grid's check matrix, a column a fault."""
    faults = grid_faults()
    shape = (GRID_DETECTOR_COUNT, len(faults))
    rows = numpy.zeros(shape, dtype=numpy.uint8)
    for fault, detectors in enumerate(faults):
        rows[list(detectors), fault] = 1
    return rows


def groups_of(labels, events: list) -> set:
    """The detection events grouped by their label."""
    groups = collections.defaultdict(set)
    for event in events:
        label = labels[event]
        groups[label].add(event)
    return {frozenset(group) for group in groups.values()}


def test_the_loops_clusters_are_the_union_finds_on_random_weights_property():
    random = numpy.random.default_rng(20261118)
    rows = grid_rows()
    priors = random.uniform(0.001, 0.2, size=rows.shape[1])
    graph = graph_of(rows, priors, GRID_WEIGHT_STEP)
    (chip,) = chip_loop.chips_of(graph)
    lengths = {edge.length_half_ticks for edge in graph.edges}
    assert len(lengths) > 3
    for _ in range(SAMPLE_COUNT):
        fired = random.random(rows.shape[0]) < EVENT_CHANCE
        syndrome = fired.astype(numpy.uint8)
        events = numpy.flatnonzero(fired)

        evidence = window_decoder.decode_graph(graph, syndrome)
        _, root = chip_loop.passes_of(chip, events)

        decode_roots = cluster_record.detector_roots(
            graph, evidence.edge_intervals
        )
        event_list = events.tolist()
        root_list = root.tolist()
        labelled = zip(event_list, root_list, strict=True)
        loop_labels = dict(labelled)
        loop_groups = groups_of(loop_labels, event_list)
        assert loop_groups == groups_of(decode_roots, event_list)
