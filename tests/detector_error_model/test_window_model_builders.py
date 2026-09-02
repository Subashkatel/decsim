"""A plan is sliced in order, and a plan that cannot be decoded is refused.

Sources: qLDPC's SlidingWindowDecoder (windows in time order, each
committing the region the next one starts after; the last window commits
everything it holds) and Skoric et al. 2209.08552 (a fault the decoder may
use to explain the syndrome but may not commit stays outside the commit
region).
"""

import pytest
import stim

from decsim.detector_error_model import (
    fault_model_contracts,
    window_model_builders,
)

GRAPHLIKE = fault_model_contracts.FaultRepresentation.GRAPHLIKE
GRAPHLIKE_REQUIRED = fault_model_contracts.GRAPHLIKE_FAULT_MODEL_REQUIRED


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


def test_a_sliding_plan_owns_every_fault_exactly_once():
    circuit = surface_code_circuit(6)
    models = window_model_builders.build_window_error_models(
        circuit,
        [(1, 2, 3), (3, 4, 5), (5, 6, 6)],
        round_count=6,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=(),
    )
    owners = [owned_faults(model) for model in models]
    every_owned = sorted(sum(owners, []))
    assert len(models) == 3
    assert len(every_owned) == len(set(every_owned))
    assert every_owned[0] == 0
    assert len(models[2].detector_ids) == 20


def test_a_gap_between_commit_regions_is_refused():
    circuit = surface_code_circuit(6)
    with pytest.raises(ValueError, match="contiguous"):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 2, 3), (4, 5, 6)],
            round_count=6,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=(),
        )


def test_a_commit_region_past_the_last_round_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(ValueError, match="exceeds round_count"):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 2, 3), (3, 5, 5)],
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=(),
        )


def test_a_window_bound_that_is_not_a_built_in_int_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(TypeError, match="built-in ints"):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 2.0, 3)],
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=(),
        )


def test_a_single_window_keeps_excluded_faults_but_does_not_own_them():
    circuit = surface_code_circuit(4)
    plain = window_model_builders.build_single_window_error_model(
        circuit,
        (1, 1, 2, 3),
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
    )
    excluded = window_model_builders.build_single_window_error_model(
        circuit,
        (1, 1, 2, 3),
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        exclude_faults_touching=(1, 1),
    )
    plain_faults = plain.require_faults(GRAPHLIKE)
    excluded_faults = excluded.require_faults(GRAPHLIKE)
    assert excluded_faults.source_fault_ids == plain_faults.source_fault_ids
    assert excluded_faults.owned.sum() < plain_faults.owned.sum()
    assert excluded_faults.owned.sum() > 0


def test_exclusion_ranges_that_cover_the_window_leave_nothing_owned():
    circuit = surface_code_circuit(4)
    builder = (
        window_model_builders.build_single_window_error_model_with_exclusions
    )
    excluded = builder(
        circuit,
        (1, 1, 2, 3),
        round_count=4,
        fault_model_requirement=GRAPHLIKE_REQUIRED,
        fault_exclusion_ranges=((1, 1), (2, 3)),
    )
    faults = excluded.require_faults(GRAPHLIKE)
    assert not faults.owned.any()
    assert faults.boundary_flips == {}
