"""The backend outcome record Tesseract and Relay-BP return.

One backend call's correction, its disposition and its diagnostics,
normalized once at construction; window_decode_of turns it into the
correction the row commits and the status the result carries. A backend
that produced a correction has it committed as it stands, best effort
or not. A backend that produced none commits an empty correction marked
with its reason, and the shot it belongs to is unscored: sinter's
discard, counted apart and never as an error
(sinter/_decoding/_decoding.py:123-125).
"""

import dataclasses
import numbers
from typing import Optional

import numpy

import decsim.records.decoding as decoding_records

_Status = decoding_records.BackendDecodeStatus
_Reason = decoding_records.BackendFailureReason
_STATUS_REASONS = {
    _Status.LOW_CONFIDENCE: frozenset({_Reason.SEARCH_LIMIT_EXHAUSTED}),
    _Status.NONCONVERGED: frozenset({_Reason.NO_CONVERGED_RELAY_SOLUTION}),
    _Status.INVALID_CORRECTION: frozenset(
        {
            _Reason.CORRECTION_NOT_BINARY,
            _Reason.CORRECTION_WRONG_ARITY,
            _Reason.CORRECTION_DOES_NOT_MATCH_SYNDROME,
        }
    ),
    _Status.EMPTY_MODEL_UNSATISFIABLE: frozenset(
        {_Reason.NONZERO_SYNDROME_WITHOUT_FAULTS}
    ),
    _Status.BACKEND_ERROR: frozenset({_Reason.UPSTREAM_EXCEPTION}),
}


@dataclasses.dataclass(frozen=True)
class BackendDecodeOutcome:
    """Immutable correction, disposition and diagnostics of one backend call."""

    status: _Status
    failure_reason: Optional[_Reason]
    physical_correction: Optional[tuple[int, ...]]
    iterations: Optional[int]

    def __post_init__(self) -> None:
        _check_status_reason(self.status, self.failure_reason)
        may_lack_correction = self.status is not _Status.SUCCEEDED
        physical_correction = _binary_tuple(
            self.physical_correction,
            name="physical_correction",
            allow_none=may_lack_correction,
        )
        iterations = _nonnegative_integer(self.iterations, name="iterations")
        object.__setattr__(self, "physical_correction", physical_correction)
        object.__setattr__(self, "iterations", iterations)

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


def _check_status_reason(
    status: _Status,
    reason: Optional[_Reason],
) -> None:
    """A failed outcome names a reason of its status; a success names none."""
    if status is _Status.SUCCEEDED:
        if reason is not None:
            raise ValueError(
                "a successful outcome cannot have a failure reason"
            )
        return
    if reason is None:
        raise ValueError("a failed outcome requires a typed failure reason")
    if reason not in _STATUS_REASONS[status]:
        raise ValueError(
            f"{reason.value} is not valid for status {status.value}"
        )


def _binary_tuple(value, *, name: str, allow_none: bool):
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} is required")
    values = _one_dimensional(value, name)
    normalized = []
    for index, bit in enumerate(values):
        if not _is_bit(bit):
            raise ValueError(f"{name}[{index}] must be binary")
        normalized.append(int(bit))
    return tuple(normalized)


def _is_bit(bit) -> bool:
    if not isinstance(bit, numbers.Integral):
        return False
    return int(bit) in (0, 1)


def _one_dimensional(value, name: str) -> tuple:
    try:
        return tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a one-dimensional iterable") from error


def _nonnegative_integer(value, *, name: str):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer or None")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


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
