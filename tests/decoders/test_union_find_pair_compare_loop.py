"""The loop that compares every pair of defects, on hand cases.

The loop is that of Barber et al. 2309.05558 Sec. II A and III. The
hand cases are the two small machines of test_union_find_cycle_count.py,
every edge two half ticks long, and the same pair with edges of four.
The property test grows random detection events on a grid of random
weights and compares the loop's clusters with those of decsim's own
union-find.
"""

import collections

import numpy

import decsim.decoders.cluster_record as cluster_record
import decsim.decoders.union_find.pair_compare_loop as pair_compare_loop
import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.records.fault_model_contracts as fault_models
import tests.decoders.test_union_find_cycle_count as traced

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
        placed, location="pair compare loop", weight_step=weight_step
    )


def test_two_adjacent_defects_meet_in_one_pass():
    """The second pass finds the second entry one step from its root."""
    graph = traced.traced_graph(traced.ADJACENT_PAIR_CHECK)
    (chip,) = pair_compare_loop.chips_of(graph)
    defects = numpy.asarray([0, 1])

    passes, root = pair_compare_loop.passes_of(chip, defects)

    meeting_pass = pair_compare_loop.Pass(
        entry_count=2,
        root_step_count=0,
        grown_entry_count=2,
        boundary_hit_count=0,
        compared_pair_count=1,
        first_collision_count=1,
    )
    last_pass = pair_compare_loop.Pass(
        entry_count=2,
        root_step_count=1,
        grown_entry_count=0,
        boundary_hit_count=0,
        compared_pair_count=0,
        first_collision_count=0,
    )
    assert passes == [meeting_pass, last_pass]
    assert root.tolist() == [0, 0]


def test_a_lone_defect_grows_until_it_reaches_the_boundary():
    """Its boundary edge is two half ticks: the second pass reaches it."""
    graph = traced.traced_graph(traced.QUIET_NEIGHBOUR_CHECK)
    (chip,) = pair_compare_loop.chips_of(graph)
    defects = numpy.asarray([0])

    passes, _ = pair_compare_loop.passes_of(chip, defects)

    grown_counts = [one_pass.grown_entry_count for one_pass in passes]
    hit_counts = [one_pass.boundary_hit_count for one_pass in passes]
    assert grown_counts == [1, 1, 0]
    assert hit_counts == [0, 1, 0]


def test_no_defect_takes_one_pass_that_visits_nothing():
    graph = traced.traced_graph(traced.ADJACENT_PAIR_CHECK)
    (chip,) = pair_compare_loop.chips_of(graph)
    defects = numpy.asarray([], dtype=int)

    passes, _ = pair_compare_loop.passes_of(chip, defects)

    empty_pass = pair_compare_loop.Pass(
        entry_count=0,
        root_step_count=0,
        grown_entry_count=0,
        boundary_hit_count=0,
        compared_pair_count=0,
        first_collision_count=0,
    )
    assert passes == [empty_pass]


def test_an_edge_twice_as_long_takes_twice_the_growth_passes():
    """Edges of four half ticks: the pair meets in the second pass."""
    priors = [traced.TRACED_PRIOR] * 3
    half_step = traced.ONE_TICK_WEIGHT_STEP / 2
    graph = graph_of(traced.ADJACENT_PAIR_CHECK, priors, half_step)
    (chip,) = pair_compare_loop.chips_of(graph)
    defects = numpy.asarray([0, 1])

    passes, _ = pair_compare_loop.passes_of(chip, defects)

    assert chip.between[0, 1] == 4
    grown_counts = [one_pass.grown_entry_count for one_pass in passes]
    collision_counts = [one_pass.first_collision_count for one_pass in passes]
    assert grown_counts == [2, 2, 0]
    assert collision_counts == [0, 1, 0]


def test_two_detectors_are_as_far_apart_as_their_shortest_edge():
    """Two faults on the same two detectors, one likelier and shorter."""
    graph = graph_of(DOUBLE_EDGE_CHECK, [0.1, 0.01], 1.0)

    (chip,) = pair_compare_loop.chips_of(graph)

    lengths = [edge.length_half_ticks for edge in graph.edges]
    assert chip.between[0, 1] == min(lengths)
    assert min(lengths) < max(lengths)


def test_a_part_with_no_boundary_edge_is_never_at_the_boundary():
    graph = graph_of(DOUBLE_EDGE_CHECK, [0.1, 0.01], 1.0)

    (chip,) = pair_compare_loop.chips_of(graph)

    assert chip.to_boundary.tolist() == [pair_compare_loop.NO_BOUNDARY] * 2


def test_each_connected_part_of_a_graph_is_a_chip_laid_out_once():
    graph = traced.traced_graph(TWO_PAIR_CHECK)

    first = pair_compare_loop.chips_of(graph)
    second = pair_compare_loop.chips_of(graph)

    assert first is second
    parts = [chip.detectors.tolist() for chip in first]
    assert parts == [[0, 1], [2, 3]]


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
    (chip,) = pair_compare_loop.chips_of(graph)
    lengths = {edge.length_half_ticks for edge in graph.edges}
    assert len(lengths) > 3
    for _ in range(SAMPLE_COUNT):
        fired = random.random(rows.shape[0]) < EVENT_CHANCE
        syndrome = fired.astype(numpy.uint8)
        events = numpy.flatnonzero(fired)

        evidence = window_decoder.decode_graph(graph, syndrome)
        _, root = pair_compare_loop.passes_of(chip, events)

        decode_roots = cluster_record.detector_roots(
            graph, evidence.edge_intervals
        )
        event_list = events.tolist()
        root_list = root.tolist()
        labelled = zip(event_list, root_list, strict=True)
        loop_labels = dict(labelled)
        loop_groups = groups_of(loop_labels, event_list)
        assert loop_groups == groups_of(decode_roots, event_list)
