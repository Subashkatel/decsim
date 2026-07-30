"""Union-Find hard correction and same-call cluster-gap confidence."""

import gc
import importlib
import itertools
import math
import pathlib
import sys
from decimal import Decimal, localcontext
from fractions import Fraction

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

np = pytest.importorskip("numpy")

from decsim.detector_error_model import (
    FaultRepresentation,
    PlacedFaultModel,
    WindowErrorModel,
)
from decsim.message import SoftOutput


def _unit_window_model(
    detector_count,
    edge_endpoints,
    logical_edges=(),
    logical_rows=None,
    priors=None,
):
    check = np.zeros((detector_count, len(edge_endpoints)), dtype=np.uint8)
    if logical_rows is None:
        logical_rows = (tuple(logical_edges),)
    observables = np.zeros(
        (len(logical_rows), len(edge_endpoints)),
        dtype=np.uint8,
    )
    for fault_index, (detector_a, detector_b) in enumerate(edge_endpoints):
        check[detector_a, fault_index] = 1
        if detector_b is not None:
            check[detector_b, fault_index] = 1
    for logical_index, row_edges in enumerate(logical_rows):
        observables[logical_index, list(row_edges)] = 1
    if priors is None:
        priors = np.full(len(edge_endpoints), 0.1)
    graphlike_faults = PlacedFaultModel(
        representation=FaultRepresentation.GRAPHLIKE,
        check=check,
        priors=np.asarray(priors, dtype=float),
        observables=observables,
        owned=np.ones(len(edge_endpoints), dtype=bool),
        future_flips={},
        source_fault_ids=tuple(range(len(edge_endpoints))),
    )
    return WindowErrorModel(
        detector_ids=tuple(range(detector_count)),
        detector_coordinates=None,
        commit_hi=1,
        defect_positions={},
        graphlike_faults=graphlike_faults,
        physical_faults=None,
    )


def _decode_job(model, syndrome):
    from decsim.message import DecodeJob, SyndromePayload

    return DecodeJob(
        op_id=4,
        window_id=2,
        n_rounds=1,
        dem=model,
        payloads=[
            SyndromePayload(
                operation_id=4,
                patch_id=4,
                round_index=1,
                bits=np.asarray(syndrome, dtype=np.uint8),
            )
        ],
        label="union-find W2",
    )


class _FixedLatency:
    def latency(self, job):
        return 7


def _edge_decibels(probability=0.1):
    return math.log((1.0 - probability) / probability) * 10.0 / math.log(10.0)


def _exhaustive_minimum_odd_eulerian(vertices, edges):
    """Independent bounded oracle: enumerate even-degree odd-logical sets."""
    best = math.inf
    for mask in range(1 << len(edges)):
        degree_parity = {vertex: 0 for vertex in vertices}
        logical_parity = 0
        weight = 0.0
        for edge_index, edge in enumerate(edges):
            if not (mask >> edge_index) & 1:
                continue
            left, right, edge_weight, edge_parity = edge
            degree_parity[left] ^= 1
            degree_parity[right] ^= 1
            logical_parity ^= edge_parity
            weight += edge_weight
        if logical_parity and not any(degree_parity.values()):
            best = min(best, weight)
    return best


def _fraction_contact_schedule(detector_count, edge_endpoints, weights, syndrome):
    """Independent small-graph oracle using sets and exact rational time."""
    boundary = detector_count
    endpoints = tuple(
        (
            boundary if left is None else left,
            boundary if right is None else right,
        )
        for left, right in edge_endpoints
    )
    components = tuple(frozenset({node}) for node in range(detector_count + 1))
    intervals = [
        None if weight == 0 else [Fraction(0), weight]
        for weight in weights
    ]
    contacts = []

    def component_index_by_node():
        return {
            node: component_index
            for component_index, component in enumerate(components)
            for node in component
        }

    def active(component):
        parity = sum(syndrome[node] for node in component if node != boundary) % 2
        return boundary not in component and parity == 1

    def merge(selected):
        nonlocal components
        owners = component_index_by_node()
        connected = [set((index,)) for index in range(len(components))]
        for edge_index in selected:
            left, right = endpoints[edge_index]
            left_component = owners[left]
            right_component = owners[right]
            connected[left_component].add(right_component)
            connected[right_component].add(left_component)
        merged = []
        unseen = set(range(len(components)))
        while unseen:
            seed = min(unseen)
            group = set()
            pending = [seed]
            while pending:
                index = pending.pop()
                if index in group:
                    continue
                group.add(index)
                pending.extend(connected[index])
            unseen.difference_update(group)
            merged.append(
                frozenset().union(*(components[index] for index in group))
            )
        components = tuple(merged)

    owners = component_index_by_node()
    zero_batch = tuple(
        edge_index
        for edge_index, interval in enumerate(intervals)
        if interval is None
        and owners[endpoints[edge_index][0]] != owners[endpoints[edge_index][1]]
    )
    contacts.extend(zero_batch)
    merge(zero_batch)

    while any(active(component) for component in components):
        owners = component_index_by_node()
        active_components = tuple(active(component) for component in components)
        candidates = []
        for edge_index, interval in enumerate(intervals):
            if interval is None:
                continue
            left, right = endpoints[edge_index]
            left_component = owners[left]
            right_component = owners[right]
            if left_component == right_component:
                continue
            rate = int(active_components[left_component]) + int(
                active_components[right_component]
            )
            if rate:
                candidates.append(
                    ((interval[1] - interval[0]) / rate, edge_index)
                )
        if not candidates:
            raise RuntimeError("unreachable syndrome")
        elapsed = min(candidate for candidate, _edge_index in candidates)
        selected = tuple(
            edge_index
            for candidate, edge_index in candidates
            if candidate == elapsed
        )
        proposals = []
        for edge_index, interval in enumerate(intervals):
            if interval is None:
                proposals.append(None)
                continue
            left, right = endpoints[edge_index]
            left_component = owners[left]
            right_component = owners[right]
            left_active = active_components[left_component]
            right_active = active_components[right_component]
            if left_component != right_component and edge_index in selected:
                proposals.append(None)
                continue
            if left_component == right_component:
                if not left_active:
                    proposals.append(interval)
                    continue
                remaining = interval[1] - interval[0]
                if 2 * elapsed >= remaining:
                    proposals.append(None)
                    continue
                left_active = right_active = True
            proposals.append(
                [
                    interval[0] + elapsed if left_active else interval[0],
                    interval[1] - elapsed if right_active else interval[1],
                ]
            )
        intervals = proposals
        contacts.extend(selected)
        merge(selected)
    return tuple(contacts)


def _literal_weight_graph(detector_count, edge_endpoints, weights):
    from decsim.union_find_decoder.window_decoder import UnionFindEdge, UnionFindGraph

    edges = tuple(
        UnionFindEdge(
            fault_index=edge_index,
            detector_a=-1 if left is None else left,
            detector_b=-1 if right is None else right,
            logical_observables=(0,),
            weight=float(weight),
        )
        for edge_index, ((left, right), weight) in enumerate(
            zip(edge_endpoints, weights)
        )
    )
    return UnionFindGraph(
        detector_count=detector_count,
        fault_count=len(edges),
        edges=edges,
        adjacency=(),
        baseline_faults=(0,) * len(edges),
        baseline_syndrome=(0,) * detector_count,
        logical_observables_by_fault=((0,),) * len(edges),
    )


def test_union_find_grows_half_edges_and_peels_only_completed_faults():
    from decsim.union_find_decoder import decode_union_find_model

    model = _unit_window_model(3, ((0, 1), (1, 2)))
    evidence = decode_union_find_model(
        model,
        np.array([1, 0, 1], dtype=np.uint8),
    )

    assert evidence.syndrome == (1, 0, 1)
    assert evidence.selected_faults == (1, 1)
    assert evidence.contact_faults == (0, 1)
    assert evidence.erasure_forest_faults == (0, 1)
    assert all(type(interval).__name__ == "Closed" for interval in evidence.edge_intervals)
    assert np.array_equal(
        (
            model.require_faults(FaultRepresentation.GRAPHLIKE).check
            @ np.asarray(evidence.selected_faults)
        ) % 2,
        np.array([1, 0, 1], dtype=np.uint8),
    )


def test_union_find_hard_evidence_is_deeply_immutable():
    from dataclasses import FrozenInstanceError

    from decsim.union_find_decoder import decode_union_find_model

    evidence = decode_union_find_model(
        _unit_window_model(2, ((0, 1),)),
        np.array([1, 1], dtype=np.uint8),
    )

    with pytest.raises(FrozenInstanceError):
        evidence.syndrome = (0, 0)
    with pytest.raises(TypeError):
        evidence.graph.adjacency[0] = ()
    assert isinstance(evidence.graph.edges[0].logical_observables, tuple)


def test_union_find_boundary_neutralizes_one_odd_cluster():
    from decsim.union_find_decoder import decode_union_find_model

    evidence = decode_union_find_model(
        _unit_window_model(1, ((0, None),)),
        np.array([1], dtype=np.uint8),
    )

    assert evidence.selected_faults == (1,)
    assert evidence.contact_faults == (0,)
    assert type(evidence.edge_intervals[0]).__name__ == "Closed"


def test_union_find_internal_chord_does_not_enter_hard_erasure():
    from decsim.union_find_decoder import decode_union_find_model

    model = _unit_window_model(
        3,
        (
            (1, 2),
            (0, 1),
            (0, 2),
            (2, None),
        ),
        logical_edges=(0,),
    )
    evidence = decode_union_find_model(
        model,
        np.array([1, 0, 0], dtype=np.uint8),
    )

    assert evidence.contact_faults == (1, 2, 3)
    assert evidence.erasure_forest_faults == (1, 2, 3)
    assert evidence.selected_faults == (0, 0, 1, 1)
    assert type(evidence.edge_intervals[0]).__name__ == "Closed"


def test_union_find_fair_sweeps_grow_all_snapshot_odd_clusters():
    from decsim.union_find_decoder import decode_union_find_model

    model = _unit_window_model(
        6,
        tuple((detector, detector + 1) for detector in range(5)),
    )
    evidence = decode_union_find_model(
        model,
        np.array([1, 0, 1, 1, 0, 1], dtype=np.uint8),
    )

    assert set(evidence.contact_faults) == {0, 1, 2, 3, 4}
    assert evidence.selected_faults == (1, 1, 0, 1, 1)


def test_union_find_records_every_initial_zero_weight_cycle_contact():
    from decsim.union_find_decoder import decode_union_find_model

    model = _unit_window_model(
        3,
        ((0, 1), (1, 2), (0, 2), (0, None)),
        priors=(0.5, 0.5, 0.5, 0.1),
    )

    evidence = decode_union_find_model(
        model,
        np.ones(3, dtype=np.uint8),
    )

    assert evidence.contact_faults[:3] == (0, 1, 2)


def test_union_find_parallel_edge_tie_uses_lowest_fault_index():
    from decsim.union_find_decoder import decode_union_find_model

    evidence = decode_union_find_model(
        _unit_window_model(2, ((0, 1), (0, 1))),
        np.array([1, 1], dtype=np.uint8),
    )

    assert evidence.contact_faults == (0, 1)
    assert evidence.erasure_forest_faults == (0,)
    assert evidence.selected_faults == (1, 0)


def test_union_find_rejects_an_unreachable_odd_cluster():
    from decsim.union_find_decoder import decode_union_find_model

    model = _unit_window_model(1, ())
    with pytest.raises(RuntimeError, match="no outward graph edge"):
        decode_union_find_model(model, np.array([1], dtype=np.uint8))


def test_hard_union_find_preserves_every_logical_observable_row():
    from decsim.union_find_decoder import (
        UnionFindDecoder,
        decode_union_find_model,
    )

    model = _unit_window_model(
        2,
        ((0, 1), (0, 1)),
        logical_rows=((0,), (1,)),
    )
    result = UnionFindDecoder(_FixedLatency()).decode(
        _decode_job(model, (1, 1))
    )
    evidence = decode_union_find_model(model, np.array([1, 1]))

    assert result.correction.tolist() == [1, 0]
    assert result.logical_observables == (1, 0)
    assert result.soft_output is None
    assert tuple(
        edge.logical_observables for edge in evidence.graph.edges
    ) == ((1, 0), (0, 1))


def test_hard_evidence_preserves_logical_rows_when_fault_catalog_is_empty():
    from decsim.union_find_decoder import decode_union_find_model

    model = _unit_window_model(
        0,
        (),
        logical_rows=((), ()),
    )

    evidence = decode_union_find_model(model, np.zeros(0, dtype=np.uint8))

    assert evidence.selected_faults == ()
    assert evidence.logical_observables == (0, 0)


@pytest.mark.parametrize(
    "edge_endpoints",
    [
        ((0, 1), (1, 2), (1, 2)),
        ((0, None), (0, 1), (1, 2), (1, 2)),
        ((0, None), (1, 2), (1, 2)),
        ((0, 3), (1, 2), (1, 2)),
    ],
)
def test_union_find_gap_searches_every_signed_component(edge_endpoints):
    from decsim.soft_output.cluster import _cluster_gap
    from decsim.union_find_decoder import decode_union_find_model

    model = _unit_window_model(
        4,
        edge_endpoints,
        logical_edges=(len(edge_endpoints) - 1,),
    )
    evidence = decode_union_find_model(
        model,
        np.zeros(4, dtype=np.uint8),
    )

    assert _cluster_gap(evidence) == pytest.approx(
        2.0 * _edge_decibels(), abs=1e-12
    )


def test_cluster_gap_matches_bounded_odd_eulerian_oracle():
    from decsim.soft_output.cluster import _cluster_gap
    from decsim.union_find_decoder import decode_union_find_model

    graph_families = (
        (
            3,
            ((0, 1), (1, 2), (2, 0)),
        ),
        (
            4,
            ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2)),
        ),
    )
    checked_assignments = 0
    for detector_count, edge_endpoints in graph_families:
        for parity_bits in itertools.product(
            (0, 1),
            repeat=len(edge_endpoints),
        ):
            logical_edges = tuple(
                index for index, parity in enumerate(parity_bits) if parity
            )
            model = _unit_window_model(
                detector_count,
                edge_endpoints,
                logical_edges=logical_edges,
            )
            evidence = decode_union_find_model(
                model,
                np.zeros(detector_count, dtype=np.uint8),
            )
            literal_edges = tuple(
                (*endpoints, 1.0, parity_bits[index])
                for index, endpoints in enumerate(edge_endpoints)
            )
            expected = _exhaustive_minimum_odd_eulerian(
                tuple(range(detector_count)),
                literal_edges,
            )
            assert _cluster_gap(evidence) == pytest.approx(
                expected * _edge_decibels(), abs=1e-12
            )
            checked_assignments += 1

    assert checked_assignments == 40


def test_cluster_gap_matches_explicit_planar_boundary_path():
    from decsim.soft_output.cluster import _cluster_gap
    from decsim.union_find_decoder import decode_union_find_model

    model = _unit_window_model(
        3,
        ((0, None), (0, 1), (1, 2), (2, None)),
        logical_edges=(3,),
    )
    evidence = decode_union_find_model(
        model,
        np.zeros(3, dtype=np.uint8),
    )

    assert _cluster_gap(evidence) == pytest.approx(
        4.0 * _edge_decibels(), abs=1e-12
    )


def test_cluster_gap_preserves_partial_edge_lengths():
    from decsim.soft_output.cluster import _quotient_cluster_gap
    from decsim.union_find_decoder import decode_union_find_model
    from decsim.union_find_decoder.window_decoder import Open

    model = _unit_window_model(
        3,
        ((0, 1), (1, 2), (2, 0)),
        logical_edges=(0,),
    )
    evidence = decode_union_find_model(
        model,
        np.zeros(3, dtype=np.uint8),
    )
    edge_weight = evidence.graph.edges[0].weight
    edge_intervals = (
        Open(0.5 * edge_weight, edge_weight),
        Open(0.0, edge_weight),
        Open(0.0, edge_weight),
    )

    assert _quotient_cluster_gap(
        evidence.graph,
        edge_intervals,
    ) == pytest.approx(2.5 * edge_weight, abs=1e-12)


def test_union_find_repetition_gap_matches_theorem_ten_exhaustively():
    from decsim.soft_output.cluster import _cluster_gap
    from decsim.union_find_decoder import decode_union_find_model

    bit_count = 5
    model = _unit_window_model(
        bit_count,
        tuple(
            (edge_index, (edge_index + 1) % bit_count)
            for edge_index in range(bit_count)
        ),
        logical_edges=(0,),
    )
    for error_bits in itertools.product((0, 1), repeat=bit_count):
        error = np.asarray(error_bits, dtype=np.uint8)
        graphlike_faults = model.require_faults(FaultRepresentation.GRAPHLIKE)
        syndrome = (graphlike_faults.check @ error) % 2
        evidence = decode_union_find_model(model, syndrome)
        correction_weight = sum(evidence.selected_faults)
        assert correction_weight <= bit_count // 2
        assert _cluster_gap(evidence) == pytest.approx(
            (bit_count - 2 * correction_weight) * _edge_decibels(),
            abs=1e-12,
        )


def test_cluster_wrapper_attaches_confidence_after_one_exact_hard_call():
    from decsim.soft_output import (
        UNION_FIND_CLUSTER_GAP_SOURCE,
        UnionFindClusterGapDecoder,
    )
    from decsim.union_find_decoder import UnionFindDecoder

    class CapturingUnionFind(UnionFindDecoder):
        def decode_with_growth_evidence(self, job):
            self.call_count = getattr(self, "call_count", 0) + 1
            decoded_window = super().decode_with_growth_evidence(job)
            self.last_decoded_window = decoded_window
            return decoded_window

    model = _unit_window_model(
        1,
        ((0, None), (0, None)),
        logical_edges=(1,),
    )
    base = CapturingUnionFind(_FixedLatency())
    decoder = UnionFindClusterGapDecoder(base)
    job = _decode_job(model, (1,))

    result = decoder.decode(job)

    assert base.call_count == 1
    assert result is base.last_decoded_window.hard_result
    assert base.last_decoded_window.hard_evidence.syndrome == (1,)
    assert decoder.latency(job) == 7
    assert result.correction.tolist() == [1, 0]
    assert result.logical_observables == (0,)
    assert result.soft_output == SoftOutput(
        gap=0.0,
        source=UNION_FIND_CLUSTER_GAP_SOURCE,
    )


def test_cluster_wrapper_rejects_multiple_logicals_before_hard_decode():
    from decsim.soft_output import UnionFindClusterGapDecoder
    from decsim.union_find_decoder import UnionFindDecoder

    class CountingUnionFind(UnionFindDecoder):
        call_count = 0

        def decode_with_growth_evidence(self, job):
            self.call_count += 1
            return super().decode_with_growth_evidence(job)

    model = _unit_window_model(
        2,
        ((0, 1), (0, 1)),
        logical_rows=((0,), (1,)),
    )
    base = CountingUnionFind(_FixedLatency())
    decoder = UnionFindClusterGapDecoder(base)

    with pytest.raises(ValueError, match="exactly one logical observable"):
        decoder.decode(_decode_job(model, (1, 1)))
    assert base.call_count == 0


def test_cluster_wrapper_publishes_infinite_unreachable_logical_gap():
    from decsim.soft_output import UnionFindClusterGapDecoder
    from decsim.union_find_decoder import UnionFindDecoder

    class CapturingUnionFind(UnionFindDecoder):
        def decode_with_growth_evidence(self, job):
            decoded_window = super().decode_with_growth_evidence(job)
            self.last_decoded_window = decoded_window
            return decoded_window

    model = _unit_window_model(1, ((0, None),), logical_edges=(0,))
    base = CapturingUnionFind(_FixedLatency())
    decoder = UnionFindClusterGapDecoder(base)

    result = decoder.decode(_decode_job(model, (1,)))
    assert math.isinf(result.soft_output.gap)
    assert result is base.last_decoded_window.hard_result


def test_cluster_wrapper_preserves_hard_decoder_seed_path_and_derived_seed():
    from decsim.message import RunSeedPathSegment, RunSeedReservation
    from decsim.seeding import bind_run_seed
    from decsim.soft_output import UnionFindClusterGapDecoder
    from decsim.union_find_decoder import UnionFindDecoder

    class CapturingLatency:
        def __init__(self):
            self.seed = None

        def latency(self, job):
            return 0

        def reserve_run_seed(self, seed):
            return RunSeedReservation("derived", seed, seed)

        def cancel_run_seed(self, reservation):
            pass

        def commit_run_seed(self, reservation):
            self.seed = reservation.proposed_seed

    root_path = (RunSeedPathSegment("field", "decoder"),)
    hard_latency = CapturingLatency()
    wrapped_latency = CapturingLatency()
    hard = UnionFindDecoder(hard_latency)
    wrapped = UnionFindClusterGapDecoder(UnionFindDecoder(wrapped_latency))

    assert wrapped.run_seed_children()[0].relative_path == (
        hard.run_seed_children()[0].relative_path
    )
    bind_run_seed(9917, ((root_path, hard),))
    bind_run_seed(9917, ((root_path, wrapped),))

    assert hard_latency.seed == wrapped_latency.seed


def test_union_find_graph_cache_releases_dead_placed_models():
    from decsim.union_find_decoder import UnionFindDecoder

    decoder = UnionFindDecoder(_FixedLatency())

    def populate_transient_models():
        for detector_count in range(1, 257):
            model = _unit_window_model(
                detector_count,
                ((0, None),),
            )
            graphlike_faults = model.require_faults(
                FaultRepresentation.GRAPHLIKE
            )
            decoder._graph_for_model(
                graphlike_faults,
                "transient cache probe",
            )

    populate_transient_models()
    gc.collect()

    assert decoder._graphs == {}


def test_old_soft_output_hard_decoder_module_is_deleted():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("decsim.soft_output.union_find_decoder")


def test_weighted_union_find_chooses_the_lighter_parallel_fault():
    from decsim.union_find_decoder import decode_union_find_model

    heavy_probability = 1.0 / (1.0 + math.exp(10.0))
    light_probability = 1.0 / (1.0 + math.exp(3.0))
    model = _unit_window_model(
        2,
        ((0, 1), (0, 1)),
        priors=(heavy_probability, light_probability),
    )

    evidence = decode_union_find_model(model, np.array([1, 1], dtype=np.uint8))

    assert evidence.selected_faults == (0, 1)
    assert evidence.contact_faults == (1,)
    assert evidence.erasure_forest_faults == (1,)


def test_weight_conversion_matches_high_precision_log_odds_within_two_ulps():
    from decsim.union_find_decoder.window_decoder import _natural_log_odds

    probabilities = {
        math.nextafter(0.0, 1.0),
        math.nextafter(0.25, 0.0),
        0.25,
        math.nextafter(0.25, 0.5),
        math.nextafter(0.5, 0.0),
        0.5,
    }
    probabilities.update(float(value) for value in np.geomspace(1e-300, 0.25, 200))
    probabilities.update(float(value) for value in np.linspace(0.25, 0.5, 200))

    with localcontext() as context:
        context.prec = 200
        for probability in sorted(probabilities):
            decimal_probability = Decimal.from_float(probability)
            exact = (
                (Decimal(1) - decimal_probability).ln()
                - decimal_probability.ln()
            )
            expected = float(exact)
            actual = _natural_log_odds(probability)
            assert abs(actual - expected) <= 2.0 * math.ulp(expected)


def test_complete_tied_contacts_feed_a_minimum_weight_forest():
    from decsim.union_find_decoder.window_decoder import (
        UnionFindEdge,
        UnionFindGraph,
        _minimum_weight_contact_forest,
    )

    edges = tuple(
        UnionFindEdge(fault_index, *endpoints, (), weight)
        for fault_index, (endpoints, weight) in enumerate(
            (((0, 1), 10.0), ((1, 2), 4.0), ((0, 2), 3.0))
        )
    )
    graph = UnionFindGraph(
        detector_count=3,
        fault_count=3,
        edges=edges,
        adjacency=(),
        baseline_faults=(0, 0, 0),
        baseline_syndrome=(0, 0, 0),
    )

    forest = _minimum_weight_contact_forest(graph, (0, 1, 2))

    assert tuple(graph.edges[index].fault_index for index in forest) == (2, 1)
    assert sum(graph.edges[index].weight for index in forest) == 7.0


def test_contact_forest_matches_exhaustive_acyclic_subset_oracle():
    from decsim.union_find_decoder.window_decoder import (
        _minimum_weight_contact_forest,
    )

    endpoints = ((0, 1), (1, 2), (2, None), (0, None), (0, 2))
    weights = tuple(map(Fraction, (4, 1, 3, 2, 5)))
    graph = _literal_weight_graph(3, endpoints, weights)

    def partition(edge_indices):
        components = [{node} for node in range(4)]
        for edge_index in edge_indices:
            left, right = endpoints[edge_index]
            left = 3 if left is None else left
            right = 3 if right is None else right
            left_component = next(group for group in components if left in group)
            right_component = next(group for group in components if right in group)
            if left_component is right_component:
                return None
            left_component.update(right_component)
            components.remove(right_component)
        return frozenset(frozenset(group) for group in components)

    target_partition = frozenset((frozenset(range(4)),))
    candidates = []
    for edge_count in range(len(endpoints) + 1):
        for selected in itertools.combinations(range(len(endpoints)), edge_count):
            if partition(selected) == target_partition:
                candidates.append((sum(weights[index] for index in selected), selected))
    expected_weight = min(weight for weight, _selected in candidates)

    forest = _minimum_weight_contact_forest(graph, tuple(range(len(endpoints))))

    assert sum(weights[index] for index in forest) == expected_weight


def test_weighted_events_match_exact_fraction_schedule_exhaustively():
    from decsim.union_find_decoder.window_decoder import _decode_graph

    graph_families = (
        (
            3,
            ((0, 1), (1, 2), (0, 2), (0, None)),
            tuple(map(Fraction, (2, 4, 6, 8))),
        ),
        (
            3,
            ((0, 1), (1, 2), (0, None), (2, None)),
            tuple(map(Fraction, (1, 2, 4, 3))),
        ),
    )
    checked = 0
    for detector_count, endpoints, weights in graph_families:
        graph = _literal_weight_graph(detector_count, endpoints, weights)
        for syndrome in itertools.product((0, 1), repeat=detector_count):
            expected_contacts = _fraction_contact_schedule(
                detector_count,
                endpoints,
                weights,
                syndrome,
            )
            evidence = _decode_graph(graph, np.asarray(syndrome, dtype=np.uint8))
            assert evidence.contact_faults == expected_contacts
            selected = np.asarray(evidence.selected_faults, dtype=np.uint8)
            reproduced = np.zeros(detector_count, dtype=np.uint8)
            for edge_index, bit in enumerate(selected):
                if not bit:
                    continue
                left, right = endpoints[edge_index]
                if left is not None:
                    reproduced[left] ^= 1
                if right is not None:
                    reproduced[right] ^= 1
            assert tuple(reproduced) == syndrome
            checked += 1
    assert checked == 16


def test_internal_edge_order_loss_is_rejected_without_mutation():
    from decsim.union_find_decoder.window_decoder import Open, _advance_open_interval

    lower = 1.0
    one_ulp = math.ulp(lower)
    upper = lower + 2.0 * one_ulp
    elapsed = math.nextafter(one_ulp, 0.0)
    interval = Open(lower, upper)
    assert elapsed < (upper - lower) / 2.0

    with pytest.raises(RuntimeError, match="represented interval order"):
        _advance_open_interval(
            interval,
            upper,
            elapsed,
            True,
            True,
        )
    assert interval == Open(lower, upper)


def test_cross_root_edge_order_loss_is_rejected_without_mutation():
    from decsim.union_find_decoder.window_decoder import Open, _advance_open_interval

    lower = 1.0
    upper = math.nextafter(lower, math.inf)
    elapsed = math.nextafter(upper - lower, 0.0)
    interval = Open(lower, upper)
    assert elapsed < upper - lower

    with pytest.raises(RuntimeError, match="represented interval order"):
        _advance_open_interval(
            interval,
            upper,
            elapsed,
            True,
            False,
        )
    assert interval == Open(lower, upper)


def test_closed_weighted_interval_has_no_one_ulp_quotient_remainder():
    from decsim.soft_output.cluster import _quotient_cluster_gap
    from decsim.union_find_decoder.window_decoder import (
        Closed,
        UnionFindEdge,
        UnionFindGraph,
    )

    weight = float.fromhex("0x1.ad7a872cedaa1p+1")
    partial_growth = float.fromhex("0x1.525e284196b65p+0")
    assert partial_growth + (weight - partial_growth) < weight
    graph = UnionFindGraph(
        detector_count=0,
        fault_count=1,
        edges=(UnionFindEdge(0, -1, -1, (1,), weight),),
        adjacency=(),
        baseline_faults=(0,),
        baseline_syndrome=(),
    )

    natural_gap = _quotient_cluster_gap(graph, (Closed(partial_growth),))

    assert natural_gap == 0.0


def test_majority_baseline_restores_detector_and_all_logical_rows():
    from decsim.union_find_decoder import decode_union_find_model

    model = _unit_window_model(
        2,
        ((0, None), (1, None), (0, 1), (0, 1)),
        logical_rows=((0, 2), (1, 3)),
        priors=(0.0, 0.5, 0.75, 1.0),
    )
    faults = model.require_faults(FaultRepresentation.GRAPHLIKE)
    expected_baseline = np.asarray((0, 0, 1, 1), dtype=np.uint8)
    expected_baseline_syndrome = (faults.check @ expected_baseline) % 2

    for fault_bits in itertools.product((0, 1), repeat=4):
        fault_vector = np.asarray(fault_bits, dtype=np.uint8)
        syndrome = (faults.check @ fault_vector) % 2
        evidence = decode_union_find_model(model, syndrome)
        selected = np.asarray(evidence.selected_faults, dtype=np.uint8)
        assert evidence.baseline_faults == tuple(expected_baseline)
        assert evidence.residual_syndrome == tuple(
            int(value) for value in syndrome ^ expected_baseline_syndrome
        )
        assert np.array_equal((faults.check @ selected) % 2, syndrome)
        expected_logicals = tuple(
            int(value) for value in (faults.observables @ selected) % 2
        )
        assert evidence.logical_observables == expected_logicals


def test_cluster_wrapper_publishes_weighted_decibel_source_and_infinity():
    from decsim.soft_output import (
        UNION_FIND_CLUSTER_GAP_SOURCE,
        UnionFindClusterGapDecoder,
    )
    from decsim.union_find_decoder import UnionFindDecoder

    assert UNION_FIND_CLUSTER_GAP_SOURCE.growth_schedule == "weighted_global_fair"
    assert UNION_FIND_CLUSTER_GAP_SOURCE.gap_units == "decibels"
    model = _unit_window_model(
        1,
        ((0, None),),
        logical_edges=(0,),
        priors=(0.1,),
    )

    result = UnionFindClusterGapDecoder(UnionFindDecoder(_FixedLatency())).decode(
        _decode_job(model, (1,))
    )

    assert math.isinf(result.soft_output.gap)
