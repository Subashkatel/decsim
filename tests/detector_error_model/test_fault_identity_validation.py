"""Fault identities reduce modulo two and decoder matrices are checked.

Source for the reduction: Stim, doc/file_format_dem_detector_error_model.md
(a target listed twice in one error cancels, and an error that flips an
observable but no detector cannot be corrected). The matrix checks are
this module's own contract, exercised here on hand-written matrices: one
column per fault, at most two detectors per graphlike column, and a
physical column equal to the parity of the graphlike components its map
names, with one finite prior in [0, 1] per physical column.
"""

import pytest

from decsim.detector_error_model import fault_identity_validation


def test_ids_listed_an_even_number_of_times_cancel():
    reduced = fault_identity_validation.xor_target_ids([3, 1, 3, 2, 2, 2])
    assert reduced == (1, 2)


def test_a_fault_that_flips_nothing_reduces_to_none():
    identity = fault_identity_validation.validate_fault_identity(
        [5, 5], [], location="error 0"
    )
    assert identity is None


def test_a_fault_that_flips_an_observable_but_no_detector_is_refused():
    with pytest.raises(ValueError, match="error 3 is a detectorless"):
        fault_identity_validation.validate_fault_identity(
            [5, 5], [0], location="error 3"
        )


def test_a_graphlike_fault_with_two_detectors_is_reduced_and_sorted():
    identity = fault_identity_validation.validate_graphlike_fault(
        [2, 1], [0], location="error 0"
    )
    assert identity == ((1, 2), (0,))


def test_a_graphlike_fault_with_three_detectors_is_refused():
    with pytest.raises(ValueError, match="hyperedge"):
        fault_identity_validation.validate_graphlike_fault(
            [0, 1, 2], [], location="error 1"
        )


def test_a_placed_matrix_column_with_no_detector_and_an_observable_is_refused():
    check = [[1, 0], [0, 0]]
    observables = [[0, 1]]
    with pytest.raises(ValueError, match="column 1 is a detectorless"):
        fault_identity_validation.validate_placed_fault_matrices(
            check, observables, location="window"
        )


def test_a_graphlike_matrix_column_with_three_detectors_is_refused():
    check = [[1, 1], [1, 0], [1, 0]]
    observables = [[0, 0]]
    with pytest.raises(ValueError, match="column 0 is a detector hyperedge"):
        fault_identity_validation.validate_graphlike_matrices(
            check, observables, location="window"
        )


def test_matrices_with_different_column_counts_are_refused():
    with pytest.raises(ValueError, match="different fault counts"):
        fault_identity_validation.validate_placed_fault_matrices(
            [[1, 0]], [[0]], location="window"
        )


def test_a_matrix_with_a_value_other_than_zero_or_one_is_refused():
    with pytest.raises(ValueError, match="only binary values"):
        fault_identity_validation.validate_placed_fault_matrices(
            [[2]], [[0]], location="window"
        )


def test_a_physical_column_equal_to_the_parity_of_its_components_passes():
    fault_identity_validation.validate_belief_matching_matrices(
        [[1, 0], [1, 1], [0, 1]],
        [[1, 0]],
        [[1], [0], [1]],
        [0.1],
        [[1], [1]],
        location="window",
    )


def test_a_physical_column_not_the_parity_of_its_components_is_refused():
    with pytest.raises(ValueError, match="physical column 0 detector"):
        fault_identity_validation.validate_belief_matching_matrices(
            [[1, 0], [1, 1], [0, 1]],
            [[1, 0]],
            [[1], [1], [1]],
            [0.1],
            [[1], [1]],
            location="window",
        )


def test_a_physical_column_that_flips_nothing_is_refused_as_inert():
    with pytest.raises(ValueError, match="physical column 0 is inert"):
        fault_identity_validation.validate_belief_matching_matrices(
            [[1, 0], [0, 1]],
            [[0, 0]],
            [[0], [0]],
            [0.1],
            [[0], [0]],
            location="window",
        )


def test_checks_with_different_detector_counts_are_refused():
    with pytest.raises(ValueError, match="different detector counts"):
        fault_identity_validation.validate_belief_matching_matrices(
            [[1, 0], [1, 1], [0, 1]],
            [[1, 0]],
            [[1], [0]],
            [0.1],
            [[1], [1]],
            location="window",
        )


def test_physical_priors_must_be_a_rank_one_array():
    with pytest.raises(ValueError, match="priors must be rank 1"):
        fault_identity_validation.validate_belief_matching_matrices(
            [[1, 0], [1, 1], [0, 1]],
            [[1, 0]],
            [[1], [0], [1]],
            [[0.1]],
            [[1], [1]],
            location="window",
        )


def test_one_physical_prior_per_physical_column_is_required():
    with pytest.raises(
        ValueError, match="2 physical priors for 1 physical fault columns"
    ):
        fault_identity_validation.validate_belief_matching_matrices(
            [[1, 0], [1, 1], [0, 1]],
            [[1, 0]],
            [[1], [0], [1]],
            [0.1, 0.2],
            [[1], [1]],
            location="window",
        )


def test_a_physical_prior_that_is_not_finite_is_refused():
    with pytest.raises(ValueError, match="priors must be finite"):
        fault_identity_validation.validate_belief_matching_matrices(
            [[1, 0], [1, 1], [0, 1]],
            [[1, 0]],
            [[1], [0], [1]],
            [float("nan")],
            [[1], [1]],
            location="window",
        )


def test_a_physical_prior_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError, match=r"inclusive range \[0, 1\]"):
        fault_identity_validation.validate_belief_matching_matrices(
            [[1, 0], [1, 1], [0, 1]],
            [[1, 0]],
            [[1], [0], [1]],
            [1.5],
            [[1], [1]],
            location="window",
        )


def test_a_physical_to_component_map_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="physical-to-component map has shape"):
        fault_identity_validation.validate_belief_matching_matrices(
            [[1, 0], [1, 1], [0, 1]],
            [[1, 0]],
            [[1], [0], [1]],
            [0.1],
            [[1]],
            location="window",
        )


def test_a_matrix_that_is_not_rank_two_is_refused():
    with pytest.raises(
        ValueError, match="window check must be a rank-2 matrix"
    ):
        fault_identity_validation.validate_placed_fault_matrices(
            [1, 0], [[0, 1]], location="window"
        )
