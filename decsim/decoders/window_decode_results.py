"""Turns a window decoder's selection into a DecodeResult.

The helpers every decoder shares (the payload syndrome, the size check,
the result built from the selected faults), and the backend outcome
record that Tesseract and Relay-BP return: its correction, its
disposition and its diagnostics, normalized once at construction.
"""

import dataclasses
import enum
import math
from numbers import Integral, Real
from typing import Optional

import numpy

import decsim.message as message


class BackendDecodeStatus(enum.Enum):
    """Backend-neutral disposition of one window decode attempt."""

    SUCCEEDED = "succeeded"
    LOW_CONFIDENCE = "low_confidence"
    NONCONVERGED = "nonconverged"
    INVALID_CORRECTION = "invalid_correction"
    EMPTY_MODEL_UNSATISFIABLE = "empty_model_unsatisfiable"
    BACKEND_ERROR = "backend_error"


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


def parity_product(matrix, vector):
    """The matrix times the vector over GF(2), as a flat array."""
    matrix_integers = matrix.astype(numpy.int64)
    vector_integers = vector.astype(numpy.int64)
    product = matrix_integers @ vector_integers
    product = numpy.asarray(product)
    flat = product.ravel()
    return flat % 2


def bit_tuple(array) -> tuple:
    """The array's entries as a tuple of ints."""
    bits = []
    for bit in array:
        bits.append(int(bit))
    return tuple(bits)


def payload_syndrome(job: message.DecodeJob):
    """Concatenate payload bits into one syndrome vector."""
    if not job.payloads:
        return numpy.zeros(0, dtype=numpy.uint8)
    bit_arrays = []
    for payload in job.payloads:
        if payload.bits is None:
            continue
        bits = numpy.asarray(payload.bits, dtype=numpy.uint8)
        bit_arrays.append(bits)
    return numpy.concatenate(bit_arrays)


def check_syndrome_size(
    job: message.DecodeJob, syndrome, placed_faults
) -> None:
    """Fail when payload bits and detector rows do not line up."""
    detector_count = placed_faults.check.shape[0]
    if syndrome.size == detector_count:
        return
    raise ValueError(
        f"{job.label}: payload bits ({syndrome.size}) do not match the window "
        f"error model's detectors ({detector_count}). The device and the "
        "cluster's model build must use the same folded-round convention."
    )


def selected_faults_of(outcome: BackendDecodeOutcome) -> tuple:
    """(correction, status) of a backend outcome: one policy for every backend.

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
    return outcome.physical_correction, decode_status


def result_from_selected_faults(
    job: message.DecodeJob, model, placed_faults, selected, decode_status=None
) -> message.DecodeResult:
    """Keep the owned selected faults and convert them into a DecodeResult.

    ``decode_status`` marks a best-effort correction (None = succeeded).
    """
    selected = numpy.asarray(selected, dtype=numpy.uint8)
    fault_count = placed_faults.check.shape[1]
    if selected.ndim != 1 or selected.shape[0] != fault_count:
        raise ValueError(
            f"{job.label}: selected correction has shape {selected.shape}; "
            f"expected ({fault_count},)"
        )
    selected_faults = selected.astype(bool)
    committed = selected_faults & placed_faults.owned
    observable_flips = parity_product(placed_faults.observables, committed)
    residual_detector_ids = _detector_ids_from_columns(
        placed_faults.boundary_flips, committed
    )
    defects = _defects_from_detector_ids(model, residual_detector_ids)
    correction = committed.astype(numpy.uint8)
    logical_observables = bit_tuple(observable_flips)
    boundary_data = message.DependencyResidual(
        detector_ids=residual_detector_ids, defects=defects
    )
    return message.DecodeResult(
        job.op_id,
        job.window_id,
        correction=correction,
        logical_observables=logical_observables,
        boundary_data=boundary_data,
        decode_status=decode_status,
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


def _detector_ids_from_columns(detector_flips, committed) -> tuple[int, ...]:
    """XOR complete global detector identities across selected columns."""
    detector_ids = set()
    nonzero = numpy.nonzero(committed)
    for column_index in nonzero[0]:
        flips = detector_flips.get(int(column_index), ())
        detector_ids.symmetric_difference_update(flips)
    return tuple(sorted(detector_ids))


def _defects_from_detector_ids(model, detector_ids) -> Optional[dict]:
    defects: dict = {}
    for detector_id in detector_ids:
        round_index, position = model.defect_positions[detector_id]
        mask = defects.setdefault(round_index, [])
        length = position + 1
        _extend_to(mask, length)
        mask[position] ^= 1
    if not defects:
        return None
    return defects


def _extend_to(mask: list, length: int) -> None:
    missing = length - len(mask)
    if missing > 0:
        padding = [0] * missing
        mask.extend(padding)
