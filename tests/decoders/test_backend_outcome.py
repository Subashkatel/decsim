"""One decode-failure policy for every backend.

decsim/decoders/backend_outcome.py's own contract: a backend that
produced a correction has it committed as it stands, best effort or
not, with its status on the result; a backend that produced no
correction at all commits an empty one marked with its reason, sinter's
discard (sinter/_decoding/_decoding.py:123-125). The dispositions are
the ones relay-bp's detailed API reports (Maurer et al. 2510.21600),
which names a nonconverged solution and an upstream error apart.
"""

import numpy

import decsim.decoders.backend_outcome as backend_outcome
import decsim.decoders.decoder as decoder_module
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoding as decoding_records
from tests.decoders import windows

GRAPHLIKE = fault_models.FaultRepresentation.GRAPHLIKE


def placed_faults(check, priors, observables):
    """One window's graphlike faults from a hand-written check matrix."""
    matrix = numpy.asarray(check, dtype=numpy.uint8)
    prior_array = numpy.asarray(priors, dtype=float)
    observable_matrix = numpy.asarray(observables, dtype=numpy.uint8)
    fault_count = matrix.shape[1]
    owned = numpy.ones(fault_count, dtype=bool)
    source_fault_ids = tuple(range(fault_count))
    return fault_models.PlacedFaultModel(
        representation=GRAPHLIKE,
        check=matrix,
        priors=prior_array,
        observables=observable_matrix,
        owned=owned,
        source_fault_ids=source_fault_ids,
        boundary_flips={},
    )


def window_of(faults, detector_count):
    """A one-window model over the detector rows in row order."""
    detector_ids = tuple(range(detector_count))
    defect_positions = {}
    for detector_id in detector_ids:
        defect_positions[detector_id] = (1, detector_id)
    return fault_models.WindowErrorModel(
        detector_ids=detector_ids,
        detector_coordinates=None,
        defect_positions=defect_positions,
        first_commit_round=1,
        graphlike_faults=faults,
        physical_faults=None,
        physical_to_graphlike_detector_projection=None,
    )


def outcome_of(status, failure_reason, physical_correction):
    """One backend call's outcome, diagnostics left out."""
    return backend_outcome.BackendDecodeOutcome(
        status=status,
        failure_reason=failure_reason,
        physical_correction=physical_correction,
        component_correction=None,
        reconstructed_syndrome=None,
        iterations=None,
        iteration_limit=None,
        posterior_log_likelihood_ratios=None,
    )


def test_an_outcome_that_carries_a_correction_is_committed_with_its_status():
    """A best-effort correction is a result, not a stop.

    relay-bp returns a decoding even when no relay leg converged
    (Maurer et al. 2510.21600), and that decoding is the best answer
    the machine has for the window, so the row commits it and carries
    NONCONVERGED on the result for the frame and the reports to read.
    """
    nonconverged = decoding_records.BackendDecodeStatus.NONCONVERGED
    reasons = decoding_records.BackendFailureReason
    reason = reasons.NO_CONVERGED_RELAY_SOLUTION
    outcome = outcome_of(nonconverged, reason, (1, 0))
    answer = backend_outcome.window_decode_of(outcome, 2)
    selected = answer.selected_faults
    status = answer.decode_status
    assert selected == (1, 0)
    assert status is nonconverged
    # two boundaryless edges, (0, 2) and (1, 3), so two fault columns
    check = [[1, 0], [0, 1], [1, 0], [0, 1]]
    faults = placed_faults(check, [0.1, 0.1], [[0, 0]])
    model = window_of(faults, 4)
    syndrome = numpy.asarray([1, 0, 1, 0], dtype=numpy.uint8)
    job = windows.job_for(model, syndrome)
    result = decoder_module.result_from_selected_faults(
        job, model, faults, selected, decode_status=status
    )
    assert result.decode_status is nonconverged
    assert result.correction.tolist() == [1, 0]


def test_an_outcome_with_no_correction_commits_an_empty_one_and_its_reason():
    """Nothing to commit is an unscored window, not a stopped run.

    A backend that raised upstream produced no vector at all, so there
    is no best effort to carry forward: the row commits the empty
    correction of the model's width and names the backend's own reason,
    which is what makes the shot unscored.
    """
    backend_error = decoding_records.BackendDecodeStatus.BACKEND_ERROR
    reasons = decoding_records.BackendFailureReason
    reason = reasons.UPSTREAM_EXCEPTION
    outcome = outcome_of(backend_error, reason, None)
    answer = backend_outcome.window_decode_of(outcome, 3)
    assert answer.selected_faults.tolist() == [0, 0, 0]
    assert answer.decode_status is backend_error
    assert answer.no_correction_reason is reason
