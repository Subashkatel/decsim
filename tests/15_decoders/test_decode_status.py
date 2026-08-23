"""One decode-failure policy: a backend that produced a correction has it
committed with a status on the result; only a backend that produced no
correction at all stops the run."""

import numpy as np
import pytest

from decsim.decoders.decoders import PresetLatencyDecoder
from decsim.decoders.mwpm.decoder import PyMatchingDecoder
from decsim.decoders.union_find.decoder import UnionFindDecoder
from decsim.decoders.window_decode_results import (
    BackendDecodeOutcome,
    BackendDecodeStatus,
    BackendFailureReason,
    result_from_backend_outcome,
)
from decsim.detector_error_model.fault_model_contracts import (
    FaultRepresentation,
    PlacedFaultModel,
    WindowErrorModel,
)
from decsim.message import DecodeJob, SyndromePayload


def _window(check):
    """A one-window model whose graph is two boundaryless edges (0,2) and (1,3)."""
    check = np.asarray(check, dtype=np.uint8)
    column_count = check.shape[1]
    placed = PlacedFaultModel(
        representation=FaultRepresentation.GRAPHLIKE,
        check=check,
        priors=np.full(column_count, 0.1),
        observables=np.zeros((1, column_count), dtype=np.uint8),
        owned=np.ones(column_count, dtype=bool),
        source_fault_ids=tuple(range(column_count)),
        boundary_flips={column: tuple(int(r) for r in np.nonzero(check[:, column])[0])
                        for column in range(column_count)},
    )
    rows = tuple(range(check.shape[0]))
    return WindowErrorModel(
        detector_ids=rows,
        detector_coordinates=None,
        defect_positions={row: (1, row) for row in rows},
        graphlike_faults=placed,
        physical_faults=None,
        physical_to_graphlike_detector_projection=None,
    )


def _job(model, syndrome):
    payload = SyndromePayload(operation_id=1, patch_id=0, round_index=1,
                              bits=tuple(int(b) for b in syndrome), code=None,
                              n_fragments=1, fragment_index=0, size_bits=len(syndrome))
    return DecodeJob(op_id=1, window_id=0, n_rounds=1, dem=model, payloads=[payload], label="test W0")


BOUNDARYLESS = [[1, 0], [0, 1], [1, 0], [0, 1]]


def test_mwpm_reports_an_unmatchable_syndrome_instead_of_raising():
    """One defect per boundaryless component: PyMatching has no perfect
    matching; the run gets an empty correction marked INVALID_CORRECTION."""
    result = PyMatchingDecoder(PresetLatencyDecoder(0.0)).decode(_job(_window(BOUNDARYLESS), [1, 1, 0, 0]))
    assert result.decode_status is BackendDecodeStatus.INVALID_CORRECTION
    assert result.correction.tolist() == [0, 0]


def test_mwpm_satisfiable_syndrome_has_no_status():
    result = PyMatchingDecoder(PresetLatencyDecoder(0.0)).decode(_job(_window(BOUNDARYLESS), [1, 0, 1, 0]))
    assert result.decode_status is None
    assert result.correction.tolist() == [1, 0]


def test_union_find_marks_its_best_effort_correction():
    result = UnionFindDecoder(PresetLatencyDecoder(0.0)).decode(_job(_window(BOUNDARYLESS), [1, 1, 0, 0]))
    assert result.decode_status is BackendDecodeStatus.INVALID_CORRECTION
    result = UnionFindDecoder(PresetLatencyDecoder(0.0)).decode(_job(_window(BOUNDARYLESS), [1, 0, 1, 0]))
    assert result.decode_status is None


def _outcome(status, reason, correction, fingerprint):
    return BackendDecodeOutcome(
        status=status, failure_reason=reason, physical_correction=correction,
        component_correction=None, reconstructed_syndrome=None, iterations=None,
        iteration_limit=None, posterior_log_likelihood_ratios=None,
        fault_model_fingerprint=fingerprint, decoder_configuration_fingerprint=fingerprint)


def test_backend_outcome_with_a_correction_is_committed_with_its_status():
    from decsim.decoders.window_decode_results import fault_model_fingerprint
    model = _window(BOUNDARYLESS)
    faults = model.require_faults(FaultRepresentation.GRAPHLIKE)
    fingerprint = fault_model_fingerprint(faults)
    job = _job(model, [1, 0, 1, 0])
    nonconverged = _outcome(BackendDecodeStatus.NONCONVERGED,
                            BackendFailureReason.NO_CONVERGED_RELAY_SOLUTION, (1, 0), fingerprint)
    result = result_from_backend_outcome(job, model, faults, nonconverged)
    assert result.decode_status is BackendDecodeStatus.NONCONVERGED
    assert result.correction.tolist() == [1, 0]


def test_backend_outcome_without_a_correction_is_structural():
    from decsim.decoders.window_decode_results import fault_model_fingerprint
    model = _window(BOUNDARYLESS)
    faults = model.require_faults(FaultRepresentation.GRAPHLIKE)
    fingerprint = fault_model_fingerprint(faults)
    broken = _outcome(BackendDecodeStatus.BACKEND_ERROR,
                      BackendFailureReason.UPSTREAM_EXCEPTION, None, fingerprint)
    with pytest.raises(RuntimeError, match="produced no correction"):
        result_from_backend_outcome(_job(model, [1, 0, 1, 0]), model, faults, broken)
