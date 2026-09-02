"""The protocol policy refuses every plan that is not the construction it names.

Source: Tan et al. 2209.09219, the sandwich decoder: every type-2 seam is
one detector layer, depends on the type-1 window before it and the one
after it, and is a closed (smooth) time boundary, so no fault may be cut
at its edge into an artificial boundary edge. The construction is
validated for the graphlike matching representation only.
"""

import pytest
import stim

import decsim.message as message
from decsim.detector_error_model import (
    fault_model_contracts,
    window_model_builders,
    window_protocol_policy,
)

TAN = message.WindowProtocol.TAN_ZERO_SEAM_GRAPHLIKE
GENERIC = message.WindowProtocol.GENERIC
GRAPHLIKE_REQUIRED = fault_model_contracts.GRAPHLIKE_FAULT_MODEL_REQUIRED
PHYSICAL_REQUIRED = fault_model_contracts.PHYSICAL_FAULT_MODEL_REQUIRED
SANDWICH = ((1, 1, 2, 3), (3, 3, 3, 3), (3, 4, 5, 5))
SANDWICH_EDGES = ((0, 1), (2, 1))


def test_the_generic_protocol_accepts_any_plan():
    window_protocol_policy.validate_window_protocol(
        SANDWICH, GENERIC, None, (), PHYSICAL_REQUIRED
    )


def test_a_protocol_that_is_not_a_member_is_refused():
    with pytest.raises(ValueError, match="unsupported window protocol"):
        window_protocol_policy.validate_window_protocol(
            SANDWICH, "sandwich", SANDWICH_EDGES, (1,), GRAPHLIKE_REQUIRED
        )


def test_a_correct_sandwich_plan_is_accepted():
    window_protocol_policy.validate_window_protocol(
        SANDWICH, TAN, SANDWICH_EDGES, (1,), GRAPHLIKE_REQUIRED
    )


def test_a_sandwich_plan_needs_the_graphlike_representation():
    with pytest.raises(ValueError, match="graphlike"):
        window_protocol_policy.validate_window_protocol(
            SANDWICH, TAN, SANDWICH_EDGES, (1,), PHYSICAL_REQUIRED
        )


def test_every_seam_and_only_a_seam_is_closed():
    with pytest.raises(
        ValueError, match="only a seam, must be temporally closed"
    ):
        window_protocol_policy.validate_window_protocol(
            SANDWICH, TAN, SANDWICH_EDGES, (0, 1), GRAPHLIKE_REQUIRED
        )


def test_a_seam_wider_than_one_layer_is_refused():
    wide_seam = ((1, 1, 2, 3), (3, 3, 4, 4), (4, 5, 6, 6))
    with pytest.raises(ValueError, match="one detector layer"):
        window_protocol_policy.validate_window_protocol(
            wide_seam, TAN, SANDWICH_EDGES, (1,), GRAPHLIKE_REQUIRED
        )


def test_a_sandwich_with_an_even_number_of_windows_is_refused():
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=4,
        after_clifford_depolarization=0.001,
    )
    with pytest.raises(ValueError, match="window 2 is outside the plan"):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 1, 3, 4), (4, 4, 4, 4)],
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=(),
            dependency_edges=((0, 1), (2, 1)),
            closed_temporal_boundary_windows=(1,),
            window_protocol=TAN,
        )


def test_a_seam_must_depend_on_exactly_its_two_neighbours():
    with pytest.raises(ValueError, match="two adjacent type-1 tasks"):
        window_protocol_policy.validate_window_protocol(
            SANDWICH, TAN, ((0, 1),), (1,), GRAPHLIKE_REQUIRED
        )


def test_a_closed_window_that_cuts_a_fault_of_the_circuit_is_refused():
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=5,
        after_clifford_depolarization=0.001,
    )
    with pytest.raises(ValueError, match="window 1 truncates global fault"):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 1, 2, 2), (3, 3, 4, 4)],
            round_count=5,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=(),
            dependency_edges=((0, 1),),
            closed_temporal_boundary_windows=(1,),
        )


def test_a_closed_window_must_be_a_dependency_destination():
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=4,
        after_clifford_depolarization=0.001,
    )
    with pytest.raises(ValueError, match="must be a dependency destination"):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 1, 2, 2), (3, 3, 4, 4)],
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=(),
            dependency_edges=((0, 1),),
            closed_temporal_boundary_windows=(0,),
        )


def test_a_closed_window_with_no_dependency_edges_is_refused_with_a_sentence():
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=4,
        after_clifford_depolarization=0.001,
    )
    with pytest.raises(ValueError, match="must be a dependency destination"):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 1, 2, 2), (3, 3, 4, 4)],
            round_count=4,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=(),
            closed_temporal_boundary_windows=(1,),
        )


def test_a_range_that_cuts_at_a_closed_seam_is_refused():
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=30,
        after_clifford_depolarization=0.001,
    )
    # The first three windows of the runtime's thirty-round sandwich with
    # step 2 and buffer 1, as of this commit (nothing here reads the
    # runtime's plan): the seam is round 3 and depends on both neighbours.
    # The fault-cut check of a closed window runs for every protocol, so
    # none is named. An excluded fault is owned by nobody and stays a
    # column of every window that sees it, so fault 22, which touches
    # rounds 3 and 4, is a column of the seam and cut at its edge.
    with pytest.raises(ValueError, match="window 1 truncates global fault 22"):
        window_model_builders.build_window_error_models(
            circuit,
            [(1, 1, 2, 4), (3, 3, 3, 3), (3, 4, 4, 6)],
            round_count=30,
            fault_model_requirement=GRAPHLIKE_REQUIRED,
            fault_exclusion_ranges=((3, 4),),
            dependency_edges=((0, 1), (2, 1)),
            closed_temporal_boundary_windows=(1,),
        )


def test_a_sandwich_plan_with_no_closed_window_is_refused():
    with pytest.raises(ValueError, match="must be temporally closed"):
        window_protocol_policy.validate_window_protocol(
            SANDWICH, TAN, SANDWICH_EDGES, (), GRAPHLIKE_REQUIRED
        )


def test_a_seam_with_buffer_rounds_around_its_one_layer_is_refused():
    buffered_seam = ((1, 1, 2, 3), (2, 3, 3, 4), (3, 4, 5, 5))
    with pytest.raises(ValueError, match="one detector layer"):
        window_protocol_policy.validate_window_protocol(
            buffered_seam, TAN, SANDWICH_EDGES, (1,), GRAPHLIKE_REQUIRED
        )
