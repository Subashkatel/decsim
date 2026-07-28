"""Union-Find hard correction and actual-cluster confidence exact cases."""
import gc
import itertools
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

np = pytest.importorskip("numpy")

from decsim.detector_error_model import WindowErrorModel
from decsim.message import SoftOutput


def _unit_window_model(detector_count, edge_endpoints, logical_edges=()):
    check = np.zeros((detector_count, len(edge_endpoints)), dtype=np.uint8)
    obs = np.zeros((1, len(edge_endpoints)), dtype=np.uint8)
    for fault_index, (detector_a, detector_b) in enumerate(edge_endpoints):
        check[detector_a, fault_index] = 1
        if detector_b is not None:
            check[detector_b, fault_index] = 1
    obs[0, list(logical_edges)] = 1
    return WindowErrorModel(
        detector_ids=tuple(range(detector_count)),
        commit_hi=1,
        check=check,
        priors=np.full(len(edge_endpoints), 0.1),
        obs=obs,
        owned=np.ones(len(edge_endpoints), dtype=bool),
        future_flips={},
    )


def test_union_find_grows_half_edges_and_peels_only_completed_edges():
    from decsim.soft_output.union_find_decoder import decode_union_find_model

    model = _unit_window_model(3, ((0, 1), (1, 2)))
    outcome = decode_union_find_model(model, np.array([1, 0, 1], dtype=np.uint8))

    assert outcome.selected_faults == (1, 1)
    assert outcome.completed_growth_edges == (0, 1)
    assert outcome.erasure_forest_edges == (0, 1)
    assert outcome.radius_by_syndrome_center == ((0, 1.0), (2, 1.0))
    assert np.array_equal(
        (model.check @ np.asarray(outcome.selected_faults)) % 2,
        np.array([1, 0, 1], dtype=np.uint8),
    )


def test_union_find_boundary_neutralizes_one_odd_cluster():
    from decsim.soft_output.union_find_decoder import decode_union_find_model

    model = _unit_window_model(1, ((0, None),))
    outcome = decode_union_find_model(model, np.array([1], dtype=np.uint8))

    assert outcome.selected_faults == (1,)
    assert outcome.completed_growth_edges == (0,)
    assert outcome.radius_by_syndrome_center == ((0, 1.0),)
    assert outcome.covered_edge_intervals == (
        ((0, 0.0, 1.0),),
    )


def test_union_find_internal_chord_changes_confidence_not_hard_erasure():
    from decsim.soft_output.union_find_decoder import decode_union_find_model

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
    outcome = decode_union_find_model(model, np.array([1, 0, 0], dtype=np.uint8))

    assert outcome.radius_by_syndrome_center == ((0, 2.0),)
    assert outcome.covered_edge_intervals[0] == ((0, 0.0, 1.0),)
    assert outcome.completed_growth_edges == (1, 2, 3)
    assert outcome.erasure_forest_edges == (1, 2, 3)
    assert outcome.selected_faults == (0, 0, 1, 1)
    assert outcome.cluster_gap == pytest.approx(0.0, abs=1e-12)


def test_union_find_fair_sweeps_grow_all_snapshot_odd_clusters():
    from decsim.soft_output.union_find_decoder import decode_union_find_model

    model = _unit_window_model(
        6,
        tuple((detector, detector + 1) for detector in range(5)),
    )
    outcome = decode_union_find_model(
        model,
        np.array([1, 0, 1, 1, 0, 1], dtype=np.uint8),
    )

    assert outcome.radius_by_syndrome_center == (
        (0, 1.5),
        (2, 0.5),
        (3, 0.5),
        (5, 1.5),
    )
    assert outcome.completed_growth_edges == (0, 1, 2, 3, 4)
    assert outcome.selected_faults == (1, 1, 0, 1, 1)


def test_union_find_fused_odd_cluster_does_not_regrow_in_one_sweep():
    from decsim.soft_output.union_find_decoder import (
        _DisjointSet,
        _graph_from_model,
        _grow_one_fair_sweep,
    )

    model = _unit_window_model(3, ((0, 1), (0, 2)))
    graph = _graph_from_model(model, location="asymmetric sweep oracle")
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
    from decsim.soft_output.union_find_decoder import decode_union_find_model

    model = _unit_window_model(2, ((0, 1), (0, 1)))
    outcome = decode_union_find_model(
        model,
        np.array([1, 1], dtype=np.uint8),
    )

    assert outcome.completed_growth_edges == (0, 1)
    assert outcome.erasure_forest_edges == (0,)
    assert outcome.selected_faults == (1, 0)


def test_union_find_rejects_an_unreachable_odd_cluster():
    from decsim.soft_output.union_find_decoder import decode_union_find_model

    model = _unit_window_model(1, ())
    with pytest.raises(RuntimeError, match="no outward graph edge"):
        decode_union_find_model(model, np.array([1], dtype=np.uint8))


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
    from decsim.soft_output.union_find_decoder import decode_union_find_model

    model = _unit_window_model(
        4,
        edge_endpoints,
        logical_edges=(len(edge_endpoints) - 1,),
    )
    outcome = decode_union_find_model(
        model,
        np.zeros(4, dtype=np.uint8),
    )

    assert outcome.cluster_gap == pytest.approx(2.0, abs=1e-12)


def test_union_find_repetition_gap_matches_theorem_ten_exhaustively():
    from decsim.soft_output.union_find_decoder import decode_union_find_model

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
        syndrome = (model.check @ error) % 2
        outcome = decode_union_find_model(model, syndrome)
        correction_weight = sum(outcome.selected_faults)
        assert correction_weight <= bit_count // 2
        assert outcome.cluster_gap == pytest.approx(
            bit_count - 2 * correction_weight,
            abs=1e-12,
        )


def test_union_find_decoder_publishes_actual_cluster_source_and_manifest():
    from decsim.message import DecodeJob, SyndromePayload
    from decsim.soft_output import (
        UNION_FIND_CLUSTER_GAP_SOURCE,
        UnionFindDecoder,
    )

    class FixedLatency:
        def latency(self, job):
            return 7

    model = _unit_window_model(
        1,
        ((0, None), (0, None)),
        logical_edges=(1,),
    )
    decoder = UnionFindDecoder(FixedLatency())
    job = DecodeJob(
        op_id=4,
        window_id=2,
        n_rounds=1,
        dem=model,
        payloads=[
            SyndromePayload(
                operation_id=4,
                patch_id=4,
                round_index=1,
                bits=np.array([1], dtype=np.uint8),
            )
        ],
        label="union-find W2",
    )

    result = decoder.decode(job)

    assert decoder.latency(job) == 7
    assert result.correction.tolist() == [1, 0]
    assert result.logical_observables == (0,)
    assert result.soft_output == SoftOutput(
        gap=0.0,
        source=UNION_FIND_CLUSTER_GAP_SOURCE,
    )
    assert decoder.run_manifest_config() == {
        "algorithm": "union_find",
        "growth_schedule": "meister_uniform_fair",
        "edge_geometry": "unit_graph_edges",
        "cluster_origin": "union_find_decoder",
        "correction": "completed_growth_edge_peeling",
        "confidence_method": "cluster_gap",
        "gap_units": "graph_edges",
        "graph_domain": "one_or_two_detectors",
        "confidence_observable_count": 1,
        "logical_search": "global_odd_parity_closed_walk",
    }
    child = decoder.run_seed_children()
    assert len(child) == 1
    assert child[0].relative_path[0].value == "latency_model"
    assert child[0].child is decoder.latency_model


def test_union_find_graph_cache_releases_dead_window_models():
    from decsim.soft_output import UnionFindDecoder

    class FixedLatency:
        def latency(self, job):
            return 1

    decoder = UnionFindDecoder(FixedLatency())

    def populate_transient_models():
        for detector_count in range(1, 257):
            model = _unit_window_model(
                detector_count,
                ((0, None),),
            )
            decoder._graph_for_model(model, "transient cache probe")

    populate_transient_models()
    gc.collect()

    assert decoder._graphs == {}
