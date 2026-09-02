"""One fault, one owner, decided by the window dependency graph.

Sources: Tan et al. 2209.09219 (the sandwich decoder: type-1 windows
decode first, a type-2 seam between two of them waits for both) and
Skoric et al. 2209.08552, section III (layer A windows commit first,
layer B windows are decoded once their neighbours have committed). The
owner of a fault is the shallowest window whose commit rounds it touches;
two candidates of the same depth leave the fault without a causal owner.
"""

import pytest
import stim

import decsim.message as message
from decsim.detector_error_model import (
    fault_model_contracts,
    window_model_builders,
    window_ownership_dag,
)

GRAPHLIKE = fault_model_contracts.FaultRepresentation.GRAPHLIKE


def surface_code_circuit(rounds):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=rounds,
        after_clifford_depolarization=0.001,
    )


def owned_faults(model):
    faults = model.require_faults(GRAPHLIKE)
    return [
        fault_index
        for fault_index, is_owned in zip(faults.source_fault_ids, faults.owned)
        if is_owned
    ]


def test_a_window_that_waits_for_nothing_has_depth_zero_and_a_seam_depth_one():
    depths = window_ownership_dag.dependency_depths(3, ((0, 1), (2, 1)))
    assert depths == (0, 1, 0)


def test_a_chain_of_dependencies_deepens_one_step_per_edge():
    depths = window_ownership_dag.dependency_depths(3, ((0, 1), (1, 2)))
    assert depths == (0, 1, 2)


def test_a_cycle_is_refused():
    with pytest.raises(ValueError, match="acyclic"):
        window_ownership_dag.dependency_depths(2, ((0, 1), (1, 0)))


def test_a_negative_window_index_is_refused():
    with pytest.raises(ValueError, match="nonnegative"):
        window_ownership_dag.dependency_depths(2, ((-1, 0),))


def test_an_edge_to_a_window_outside_the_plan_is_refused():
    with pytest.raises(
        ValueError, match=r"edge \(0, 7\) names a window outside"
    ):
        window_ownership_dag.dependency_depths(2, ((0, 7),))


def test_a_plan_whose_edge_names_a_missing_window_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(
        ValueError, match=r"edge \(5, 1\) names a window outside"
    ):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 1, 2, 2), (3, 3, 4, 4)],
            round_count=4,
            fault_model_requirement=fault_model_contracts.GRAPHLIKE_FAULT_MODEL_REQUIRED,
            fault_exclusion_ranges=(),
            dependency_edges=((5, 1),),
        )


def test_ancestors_include_indirect_predecessors():
    depths = window_ownership_dag.dependency_depths(3, ((0, 1), (1, 2)))
    ancestors = window_ownership_dag.dependency_ancestors(
        3, ((0, 1), (1, 2)), depths
    )
    assert ancestors == (frozenset(), frozenset({0}), frozenset({0, 1}))


def test_no_fault_is_owned_twice_in_a_sandwich_plan():
    circuit = surface_code_circuit(5)
    models = window_model_builders.build_window_error_models(
        circuit,
        [(1, 1, 2, 3), (3, 3, 3, 3), (3, 4, 5, 5)],
        round_count=5,
        fault_model_requirement=fault_model_contracts.GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
        dependency_edges=((0, 1), (2, 1)),
        closed_temporal_boundary_windows=(1,),
        window_protocol=message.WindowProtocol.TAN_ZERO_SEAM_GRAPHLIKE,
    )
    owners = [owned_faults(model) for model in models]
    every_owned = sum(owners, [])
    assert len(every_owned) == len(set(every_owned))
    first_faults = models[0].require_faults(GRAPHLIKE)
    assert first_faults.check.shape[0] == 20
    assert sorted(every_owned) == list(range(len(set(every_owned))))


def test_the_seam_leaves_out_what_its_neighbours_own():
    circuit = surface_code_circuit(5)
    models = window_model_builders.build_window_error_models(
        circuit,
        [(1, 1, 2, 3), (3, 3, 3, 3), (3, 4, 5, 5)],
        round_count=5,
        fault_model_requirement=fault_model_contracts.GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
        dependency_edges=((0, 1), (2, 1)),
        closed_temporal_boundary_windows=(1,),
        window_protocol=message.WindowProtocol.TAN_ZERO_SEAM_GRAPHLIKE,
    )
    seam_faults = models[1].require_faults(GRAPHLIKE)
    before_owns = owned_faults(models[0])
    after_owns = owned_faults(models[2])
    neighbours_own = set(before_owns) | set(after_owns)
    assert neighbours_own & set(seam_faults.source_fault_ids) == set()
    assert seam_faults.owned.all()
    assert len(models[1].detector_ids) == 8


def test_a_fault_between_two_windows_of_the_same_depth_has_no_owner():
    circuit = surface_code_circuit(4)
    with pytest.raises(
        ValueError, match="straddles independent commit regions"
    ):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 1, 2, 2), (3, 3, 4, 4)],
            round_count=4,
            fault_model_requirement=fault_model_contracts.GRAPHLIKE_FAULT_MODEL_REQUIRED,
            fault_exclusion_ranges=(),
            dependency_edges=(),
        )
