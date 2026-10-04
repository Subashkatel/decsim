"""The backend outcome record Tesseract and Relay-BP return.

One backend call's correction, its disposition and its diagnostics;
window_decode_of turns it into the correction the row commits and the
status the result carries. A backend that produced a correction has it
committed as it stands, best effort or not. A backend that produced none
commits an empty correction marked with its reason, and its shot is
unscored: sinter's discard, counted apart and never as an error
(sinter/_decoding/_decoding.py:123-125).
"""

import dataclasses
from typing import Optional

import numpy

import decsim.records.decoding as decoding_records

_Status = decoding_records.BackendDecodeStatus
_Reason = decoding_records.BackendFailureReason


@dataclasses.dataclass(frozen=True)
class BackendDecodeOutcome:
    """Immutable correction, disposition and diagnostics of one backend call."""

    status: _Status
    failure_reason: Optional[_Reason]
    physical_correction: Optional[tuple[int, ...]]
    iterations: Optional[int]

    def __post_init__(self) -> None:
        # a user's decoder may build this record, and a success with no
        # correction would be scored as the empty one
        if self.physical_correction is not None:
            return
        if self.succeeded:
            raise ValueError(
                "a successful outcome needs its physical_correction"
            )

    @property
    def succeeded(self) -> bool:
        """Whether the backend committed a correction it stands behind."""
        return self.status is _Status.SUCCEEDED


def empty_fault_model_outcome(syndrome) -> BackendDecodeOutcome:
    """Resolve a no-fault physical model without invoking a backend.

    A nonzero syndrome on a model with no fault columns is unsatisfiable.
    """
    if numpy.any(syndrome):
        return _empty_model_outcome(
            _Status.EMPTY_MODEL_UNSATISFIABLE,
            _Reason.NONZERO_SYNDROME_WITHOUT_FAULTS,
            None,
        )
    return _empty_model_outcome(_Status.SUCCEEDED, None, ())


def window_decode_of(
    outcome: BackendDecodeOutcome, fault_count: int
) -> decoding_records.WindowDecode:
    """The window's answer from a backend outcome: one policy for every row.

    A decode that produced a correction is committed as it stands, best
    effort or not, with its status on the result (nonconverged, low
    confidence, does not reproduce the syndrome). An outcome with no
    correction, which the record allows only off success (an upstream
    exception, a malformed vector, an unsatisfiable empty model), commits
    an empty correction of the model's fault_count columns and carries
    the backend's own reason, which makes its shot unscored.
    """
    if outcome.physical_correction is None:
        return no_correction_decode(
            outcome.status, outcome.failure_reason, fault_count
        )
    decode_status = None
    if not outcome.succeeded:
        decode_status = outcome.status
    return decoding_records.WindowDecode(
        outcome.physical_correction,
        decode_status,
        iterations=outcome.iterations,
    )


def no_correction_decode(
    status: _Status,
    reason: _Reason,
    fault_count: int,
) -> decoding_records.WindowDecode:
    """The answer of a backend that produced no correction: empty, marked."""
    empty = numpy.zeros(fault_count, dtype=numpy.uint8)
    return decoding_records.WindowDecode(
        empty, status, no_correction_reason=reason
    )


def _empty_model_outcome(
    status: _Status,
    reason: Optional[_Reason],
    physical_correction,
) -> BackendDecodeOutcome:
    return BackendDecodeOutcome(
        status=status,
        failure_reason=reason,
        physical_correction=physical_correction,
        iterations=None,
    )
