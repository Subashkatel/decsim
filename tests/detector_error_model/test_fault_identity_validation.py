"""Fault identities reduce modulo two.

Source for the reduction: Stim, doc/file_format_dem_detector_error_model.md
(a target listed twice in one error cancels, and an error that flips an
observable but no detector cannot be corrected).
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
