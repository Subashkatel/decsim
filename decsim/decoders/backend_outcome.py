"""The backend outcome record Tesseract and Relay-BP return.

One backend call's correction, its disposition and its diagnostics,
normalized once at construction; window_decode_of turns it into the
correction the row commits and the status the result carries. A backend
that produced a correction has it committed as it stands, best effort
or not; only a backend that produced no correction at all is a
structural failure and stops the run.
"""

import dataclasses
import enum
import math
from numbers import Integral, Real
from typing import Optional

import numpy

import decsim.decoders.decoder as decoder_module
import decsim.records.decoding as decoding_records

BackendDecodeStatus = decoder_module.BackendDecodeStatus


class BackendFailureReason(enum.Enum):
    """Typed reason a backend attempt could not be committed."""

    SEARCH_LIMIT_EXHAUSTED = "search_limit_exhausted"
    NO_CONVERGED_RELAY_SOLUTION = "no_converged_relay_solution"
    CORRECTION_NOT_BINARY = "correction_not_binary"
    CORRECTION_WRONG_ARITY = "correction_wrong_arity"
    CORRECTION_DOES_NOT_MATCH_SYNDROME = "correction_does_not_match_syndrome"
    NONZERO_SYNDROME_WITHOUT_FAULTS = "nonzero_syndrome_without_faults"
    UPSTREAM_EXCEPTION = "upstream_exception"


_STATUS_REASONS = {
    BackendDecodeStatus.LOW_CONFIDENCE: frozenset(
        {BackendFailureReason.SEARCH_LIMIT_EXHAUSTED}
    ),
    BackendDecodeStatus.NONCONVERGED: frozenset(
        {BackendFailureReason.NO_CONVERGED_RELAY_SOLUTION}
    ),
    BackendDecodeStatus.INVALID_CORRECTION: frozenset(
        {
            BackendFailureReason.CORRECTION_NOT_BINARY,
            BackendFailureReason.CORRECTION_WRONG_ARITY,
            BackendFailureReason.CORRECTION_DOES_NOT_MATCH_SYNDROME,
        }
    ),
    BackendDecodeStatus.EMPTY_MODEL_UNSATISFIABLE: frozenset(
        {BackendFailureReason.NONZERO_SYNDROME_WITHOUT_FAULTS}
    ),
    BackendDecodeStatus.BACKEND_ERROR: frozenset(
        {BackendFailureReason.UPSTREAM_EXCEPTION}
    ),
}


@dataclasses.dataclass(frozen=True)
class BackendDecodeOutcome:
    """Immutable correction, disposition and diagnostics of one backend call."""

    status: BackendDecodeStatus
    failure_reason: Optional[BackendFailureReason]
    physical_correction: Optional[tuple[int, ...]]
    component_correction: Optional[tuple[int, ...]]
    reconstructed_syndrome: Optional[tuple[int, ...]]
    iterations: Optional[int]
    iteration_limit: Optional[int]
    posterior_log_likelihood_ratios: Optional[tuple[float, ...]]

    def __post_init__(self) -> None:
        _check_status_reason(self.status, self.failure_reason)
        may_lack_correction = self.status is not BackendDecodeStatus.SUCCEEDED
        physical_correction = _binary_tuple(
            self.physical_correction,
            name="physical_correction",
            allow_none=may_lack_correction,
        )
        reconstructed_syndrome = _binary_tuple(
            self.reconstructed_syndrome,
            name="reconstructed_syndrome",
            allow_none=may_lack_correction,
        )
        component_correction = _binary_tuple(
            self.component_correction, name="component_correction"
        )
        iterations = _nonnegative_integer(self.iterations, name="iterations")
        iteration_limit = _nonnegative_integer(
            self.iteration_limit, name="iteration_limit"
        )
        posterior = _float_tuple(
            self.posterior_log_likelihood_ratios,
            name="posterior_log_likelihood_ratios",
        )
        object.__setattr__(self, "physical_correction", physical_correction)
        object.__setattr__(self, "component_correction", component_correction)
        object.__setattr__(
            self, "reconstructed_syndrome", reconstructed_syndrome
        )
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "iteration_limit", iteration_limit)
        object.__setattr__(self, "posterior_log_likelihood_ratios", posterior)

    @property
    def succeeded(self) -> bool:
        """Whether the backend committed a correction it stands behind."""
        return self.status is BackendDecodeStatus.SUCCEEDED


def empty_fault_model_outcome(syndrome) -> BackendDecodeOutcome:
    """Resolve a no-fault physical model without invoking a backend.

    A nonzero syndrome on a model with no fault columns is unsatisfiable.
    """
    detector_count = syndrome.shape[0]
    reconstructed = (0,) * detector_count
    if numpy.any(syndrome):
        return _empty_model_outcome(
            BackendDecodeStatus.EMPTY_MODEL_UNSATISFIABLE,
            BackendFailureReason.NONZERO_SYNDROME_WITHOUT_FAULTS,
            None,
            reconstructed,
        )
    return _empty_model_outcome(
        BackendDecodeStatus.SUCCEEDED, None, (), reconstructed
    )


def window_decode_of(
    outcome: BackendDecodeOutcome,
) -> decoding_records.WindowDecode:
    """The window's answer from a backend outcome: one policy for every row.

    A decode that produced a correction is committed as it stands, best
    effort or not, with its status on the result (nonconverged, low
    confidence, does not reproduce the syndrome); only a backend that
    produced no correction at all (an upstream exception, a malformed
    vector) is a structural failure and stops the run.
    """
    if outcome.physical_correction is None:
        raise RuntimeError(
            "decoder backend produced no correction: "
            f"{outcome.status.value}/{outcome.failure_reason.value}"
        )
    decode_status = None
    if not outcome.succeeded:
        decode_status = outcome.status
    return decoding_records.WindowDecode(
        outcome.physical_correction, decode_status
    )


def _check_status_reason(
    status: BackendDecodeStatus, reason: Optional[BackendFailureReason]
) -> None:
    """A failed outcome names a reason of its status; a success names none."""
    if status is BackendDecodeStatus.SUCCEEDED:
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


def _binary_tuple(value, *, name: str, allow_none: bool = True):
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
    if not isinstance(bit, Integral):
        return False
    return int(bit) in (0, 1)


def _float_tuple(value, *, name: str):
    if value is None:
        return None
    values = _one_dimensional(value, name)
    normalized = []
    for index, item in enumerate(values):
        if not _is_real(item):
            raise TypeError(f"{name}[{index}] must be a real number")
        number = float(item)
        if math.isnan(number):
            raise ValueError(f"{name}[{index}] cannot be NaN")
        normalized.append(number)
    return tuple(normalized)


def _is_real(item) -> bool:
    if isinstance(item, bool):
        return False
    return isinstance(item, Real)


def _one_dimensional(value, name: str) -> tuple:
    try:
        return tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a one-dimensional iterable") from error


def _nonnegative_integer(value, *, name: str):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer or None")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


def _empty_model_outcome(
    status: BackendDecodeStatus,
    reason: Optional[BackendFailureReason],
    physical_correction,
    reconstructed: tuple,
) -> BackendDecodeOutcome:
    return BackendDecodeOutcome(
        status=status,
        failure_reason=reason,
        physical_correction=physical_correction,
        component_correction=None,
        reconstructed_syndrome=reconstructed,
        iterations=None,
        iteration_limit=None,
        posterior_log_likelihood_ratios=None,
    )
