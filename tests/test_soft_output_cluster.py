"""Union-Find hard correction and same-call cluster-gap confidence."""

import gc
import importlib
import itertools
import math
import pathlib
import sys

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
    graphlike_faults = PlacedFaultModel(
        representation=FaultRepresentation.GRAPHLIKE,
        check=check,
        priors=np.full(len(edge_endpoints), 0.1),
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


def test_union_find_grows_half_edges_and_peels_only_completed_faults():
    from decsim.union_find_decoder import decode_union_find_model

    model = _unit_window_model(3, ((0, 1), (1, 2)))
    evidence = decode_union_find_model(
        model,
        np.array([1, 0, 1], dtype=np.uint8),
    )

    assert evidence.syndrome == (1, 0, 1)
    assert evidence.selected_faults == (1, 1)
    assert evidence.completed_growth_faults == (0, 1)
    assert evidence.erasure_forest_faults == (0, 1)
    assert evidence.radius_by_syndrome_center == ((0, 1.0), (2, 1.0))
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
    assert evidence.completed_growth_faults == (0,)
    assert evidence.radius_by_syndrome_center == ((0, 1.0),)


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

    assert evidence.radius_by_syndrome_center == ((0, 2.0),)
    assert evidence.completed_growth_faults == (1, 2, 3)
    assert evidence.erasure_forest_faults == (1, 2, 3)
    assert evidence.selected_faults == (0, 0, 1, 1)


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

    assert evidence.radius_by_syndrome_center == (
        (0, 1.5),
        (2, 0.5),
        (3, 0.5),
        (5, 1.5),
    )
    assert evidence.completed_growth_faults == (0, 1, 2, 3, 4)
    assert evidence.selected_faults == (1, 1, 0, 1, 1)


def test_union_find_fused_odd_cluster_does_not_regrow_in_one_sweep():
    from decsim.union_find_decoder.window_decoder import (
        _DisjointSet,
        _graph_from_model,
        _grow_one_fair_sweep,
    )

    model = _unit_window_model(3, ((0, 1), (0, 2)))
    graphlike_faults = model.require_faults(FaultRepresentation.GRAPHLIKE)
    graph = _graph_from_model(
        graphlike_faults,
        location="asymmetric sweep oracle",
    )
    disjoint_set = _DisjointSet(
        detector_count=3,
        syndrome=np.ones(3, dtype=np.uint8),
    )
    radii = {0: 0.5, 1: 0.0, 2: 0.0}
    edge_growth_units = [1, 1]

    visit_count = _grow_one_fair_sweep(
        graph,
        disjoint_set,
        radii,
        edge_growth_units,
    )

    assert visit_count == 1
    assert radii == {0: 1.0, 1: 0.0, 2: 0.0}
    assert edge_growth_units == [2, 2]


def test_union_find_parallel_edge_tie_uses_lowest_fault_index():
    from decsim.union_find_decoder import decode_union_find_model

    evidence = decode_union_find_model(
        _unit_window_model(2, ((0, 1), (0, 1))),
        np.array([1, 1], dtype=np.uint8),
    )

    assert evidence.completed_growth_faults == (0, 1)
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

    assert _cluster_gap(evidence) == pytest.approx(2.0, abs=1e-12)


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
            assert _cluster_gap(evidence) == expected
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

    assert _cluster_gap(evidence) == 4.0


def test_cluster_gap_preserves_partial_edge_lengths():
    from decsim.soft_output.cluster import _quotient_cluster_gap
    from decsim.union_find_decoder import decode_union_find_model

    model = _unit_window_model(
        3,
        ((0, 1), (1, 2), (2, 0)),
        logical_edges=(0,),
    )
    evidence = decode_union_find_model(
        model,
        np.zeros(3, dtype=np.uint8),
    )
    intervals_by_edge = (
        ((0, 0.0, 0.5),),
        (),
        (),
    )

    assert _quotient_cluster_gap(
        evidence.graph,
        intervals_by_edge,
    ) == pytest.approx(2.5, abs=1e-12)


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
            bit_count - 2 * correction_weight,
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


def test_cluster_wrapper_rejects_unreachable_logical_without_publication():
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

    with pytest.raises(ValueError, match="odd logical cycle"):
        decoder.decode(_decode_job(model, (1,)))
    assert base.last_decoded_window.hard_result.soft_output is None


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
