"""Turns a window decoder's selection into a DecodeResult.

The helpers every decoder shares (the payload syndrome, the size check,
the result built from the selected faults), and the backend outcome
record that Tesseract and Relay-BP return with its checks against the
model it came from.
"""

import dataclasses
import enum
import hashlib
import math
import operator
import struct
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Optional

import numpy
import scipy.sparse

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
    fault_model_fingerprint: str
    decoder_configuration_fingerprint: str

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
        _validate_fingerprint(
            self.fault_model_fingerprint, name="fault_model_fingerprint"
        )
        _validate_fingerprint(
            self.decoder_configuration_fingerprint,
            name="decoder_configuration_fingerprint",
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


def fault_model_fingerprint(placed_faults) -> str:
    """Fingerprint every column-aligned input and boundary field of one view."""
    return _sha256_fingerprint(
        {
            "representation": placed_faults.representation,
            "check": placed_faults.check,
            "priors": placed_faults.priors,
            "observables": placed_faults.observables,
            "owned": placed_faults.owned,
            "boundary_flips": placed_faults.boundary_flips,
            "source_fault_ids": placed_faults.source_fault_ids,
        }
    )


def decoder_configuration_fingerprint(configuration) -> str:
    """Fingerprint one frozen backend profile using deterministic bytes."""
    return _sha256_fingerprint(configuration)


def validate_backend_outcome(
    outcome: BackendDecodeOutcome, model, placed_faults, syndrome
) -> None:
    """Validate a same-model outcome before any correction can be committed."""
    if not isinstance(outcome, BackendDecodeOutcome):
        raise TypeError("backend must return a BackendDecodeOutcome")
    expected_fingerprint = fault_model_fingerprint(placed_faults)
    if outcome.fault_model_fingerprint != expected_fingerprint:
        raise ValueError("backend outcome belongs to a different fault model")
    syndrome = _checked_syndrome(
        syndrome, placed_faults, "the placed fault model"
    )
    correction, reconstructed = _checked_correction(outcome, placed_faults)
    if outcome.component_correction is not None:
        _check_component_correction(outcome, model, correction)
    if not outcome.succeeded:
        return
    expected = syndrome.astype(numpy.uint8)
    if not numpy.array_equal(reconstructed, expected):
        raise ValueError("successful correction does not match the syndrome")


def empty_fault_model_outcome(
    placed_faults, syndrome, *, decoder_configuration_fingerprint: str
) -> BackendDecodeOutcome:
    """Resolve a no-fault physical model without invoking a backend."""
    if placed_faults.check.shape[1] != 0:
        raise ValueError(
            "empty-fault outcome requires a model with zero columns"
        )
    syndrome = _checked_syndrome(
        syndrome, placed_faults, "the empty fault model"
    )
    model_fingerprint = fault_model_fingerprint(placed_faults)
    detector_count = syndrome.shape[0]
    reconstructed = (0,) * detector_count
    if numpy.any(syndrome):
        return _empty_model_outcome(
            BackendDecodeStatus.EMPTY_MODEL_UNSATISFIABLE,
            BackendFailureReason.NONZERO_SYNDROME_WITHOUT_FAULTS,
            None,
            reconstructed,
            model_fingerprint,
            decoder_configuration_fingerprint,
        )
    return _empty_model_outcome(
        BackendDecodeStatus.SUCCEEDED,
        None,
        (),
        reconstructed,
        model_fingerprint,
        decoder_configuration_fingerprint,
    )


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


def result_from_backend_outcome(
    job: message.DecodeJob, model, placed_faults, outcome: BackendDecodeOutcome
) -> message.DecodeResult:
    """One policy for every backend: a produced correction is committed.

    Best effort or not, with its status on the result (nonconverged, low
    confidence, does not reproduce the syndrome); only a backend that
    produced no correction at all (an upstream exception, a malformed
    vector) is a structural failure and stops the run.
    """
    if outcome.physical_correction is None:
        raise RuntimeError(
            f"{job.label}: decoder backend produced no correction: "
            f"{outcome.status.value}/{outcome.failure_reason.value}"
        )
    decode_status = None
    if not outcome.succeeded:
        decode_status = outcome.status
    return result_from_selected_faults(
        job,
        model,
        placed_faults,
        outcome.physical_correction,
        decode_status=decode_status,
    )


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
    observable_flips = _parity_product(placed_faults.observables, committed)
    residual_detector_ids = _detector_ids_from_columns(
        placed_faults.boundary_flips, committed
    )
    defects = _defects_from_detector_ids(model, residual_detector_ids)
    correction = committed.astype(numpy.uint8)
    logical_observables = _bit_tuple(observable_flips)
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


def _validate_fingerprint(value: str, *, name: str) -> None:
    text = f"{name} must be a lowercase SHA-256 hexadecimal digest"
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(text)
    if value != value.lower():
        raise ValueError(text)
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(text) from error


def _sha256_fingerprint(value) -> str:
    encoded = _canonical_bytes(value)
    digest = hashlib.sha256(encoded)
    return digest.hexdigest()


def _canonical_bytes(value) -> bytes:
    """Encode supported scientific values without process-local identities."""
    scalar = _scalar_bytes(value)
    if scalar is not None:
        return scalar
    if scipy.sparse.issparse(value):
        return _sparse_bytes(value)
    if isinstance(value, numpy.ndarray):
        return _array_bytes(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _dataclass_bytes(value)
    if isinstance(value, Mapping):
        return _mapping_bytes(value)
    if isinstance(value, tuple):
        return _tuple_bytes(value)
    if isinstance(value, list):
        as_tuple = tuple(value)
        return b"l" + _canonical_bytes(as_tuple)
    kind = type(value)
    raise TypeError(f"cannot fingerprint value of type {kind.__qualname__}")


def _scalar_bytes(value):
    """The bytes of a scalar or string; None for anything else."""
    if value is None:
        return b"n"
    if isinstance(value, enum.Enum):
        class_bytes = _canonical_bytes(value.__class__.__qualname__)
        value_bytes = _canonical_bytes(value.value)
        return b"e" + class_bytes + value_bytes
    if isinstance(value, bool):
        if value:
            return b"b1"
        return b"b0"
    if isinstance(value, Integral):
        digits = str(int(value))
        return b"i" + digits.encode("ascii") + b";"
    if isinstance(value, Real):
        number = float(value)
        if math.isnan(number):
            raise ValueError("fingerprinted values cannot contain NaN")
        return b"f" + struct.pack(">d", number)
    if isinstance(value, str):
        encoded = value.encode("utf8")
        return _length_prefixed(b"s", encoded, len(encoded))
    if isinstance(value, bytes):
        return _length_prefixed(b"y", value, len(value))
    return None


def _length_prefixed(tag: bytes, payload: bytes, count: int) -> bytes:
    digits = str(count)
    return tag + digits.encode("ascii") + b":" + payload


def _sparse_bytes(value) -> bytes:
    matrix = value.tocsc()
    matrix = matrix.copy()
    matrix.sum_duplicates()
    matrix.sort_indices()
    shape_bytes = _canonical_bytes(tuple(matrix.shape))
    indptr = numpy.ascontiguousarray(matrix.indptr)
    indptr_bytes = _canonical_bytes(indptr)
    indices = numpy.ascontiguousarray(matrix.indices)
    indices_bytes = _canonical_bytes(indices)
    data = numpy.ascontiguousarray(matrix.data)
    data_bytes = _canonical_bytes(data)
    return b"c" + shape_bytes + indptr_bytes + indices_bytes + data_bytes


def _array_bytes(value: numpy.ndarray) -> bytes:
    if value.dtype.hasobject:
        raise TypeError("object arrays cannot be fingerprinted")
    array = numpy.ascontiguousarray(value)
    dtype_bytes = _canonical_bytes(array.dtype.str)
    shape_bytes = _canonical_bytes(tuple(array.shape))
    raw = array.tobytes()
    data_bytes = _canonical_bytes(raw)
    return b"a" + dtype_bytes + shape_bytes + data_bytes


def _dataclass_bytes(value) -> bytes:
    class_bytes = _canonical_bytes(value.__class__.__qualname__)
    named_values = []
    for field in dataclasses.fields(value):
        field_value = getattr(value, field.name)
        named_values.append((field.name, field_value))
    fields_bytes = _canonical_bytes(tuple(named_values))
    return b"d" + class_bytes + fields_bytes


def _mapping_bytes(value: Mapping) -> bytes:
    items = []
    for key, item in value.items():
        key_bytes = _canonical_bytes(key)
        item_bytes = _canonical_bytes(item)
        items.append((key_bytes, item_bytes))
    by_key = operator.itemgetter(0)
    items.sort(key=by_key)
    return b"m" + _canonical_bytes(tuple(items))


def _tuple_bytes(value: tuple) -> bytes:
    pieces = []
    for item in value:
        item_bytes = _canonical_bytes(item)
        pieces.append(item_bytes)
    encoded = b"".join(pieces)
    return _length_prefixed(b"t", encoded, len(value))


def _checked_syndrome(syndrome, placed_faults, model_name: str):
    """The syndrome as an array of the model's arity, bits only."""
    syndrome = numpy.asarray(syndrome)
    detector_count = placed_faults.check.shape[0]
    if syndrome.ndim != 1 or syndrome.shape[0] != detector_count:
        raise ValueError(f"syndrome arity does not match {model_name}")
    if not _is_binary(syndrome):
        raise ValueError("syndrome must contain only binary values")
    return syndrome


def _is_binary(array) -> bool:
    is_zero = array == 0
    is_one = array == 1
    is_bit = is_zero | is_one
    all_bits = numpy.all(is_bit)
    return bool(all_bits)


def _checked_correction(outcome: BackendDecodeOutcome, placed_faults) -> tuple:
    """(correction, reconstructed syndrome) arrays, or (None, None)."""
    if outcome.physical_correction is None:
        return None, None
    correction = numpy.asarray(outcome.physical_correction, dtype=numpy.uint8)
    fault_count = placed_faults.check.shape[1]
    if correction.shape != (fault_count,):
        raise ValueError("backend correction has the wrong fault-model arity")
    reconstructed = _parity_product(placed_faults.check, correction)
    if outcome.reconstructed_syndrome is not None:
        reported = _bit_tuple(reconstructed)
        if reported != outcome.reconstructed_syndrome:
            raise ValueError(
                "backend reconstructed syndrome does not match its correction"
            )
    return correction, reconstructed


def _check_component_correction(
    outcome: BackendDecodeOutcome, model, correction
) -> None:
    projection = model.physical_to_graphlike_detector_projection
    if projection is None or correction is None:
        raise ValueError(
            "component correction requires a physical correction and valid "
            "local projection"
        )
    component = _parity_product(projection, correction)
    reported = _bit_tuple(component)
    if reported != outcome.component_correction:
        raise ValueError(
            "component correction does not match the local projection"
        )


def _parity_product(matrix, vector):
    """The matrix times the vector over GF(2), as a flat array."""
    matrix_integers = matrix.astype(numpy.int64)
    vector_integers = vector.astype(numpy.int64)
    product = matrix_integers @ vector_integers
    product = numpy.asarray(product)
    flat = product.ravel()
    return flat % 2


def _bit_tuple(array) -> tuple:
    bits = []
    for bit in array:
        bits.append(int(bit))
    return tuple(bits)


def _empty_model_outcome(
    status: BackendDecodeStatus,
    reason: Optional[BackendFailureReason],
    physical_correction,
    reconstructed: tuple,
    model_fingerprint: str,
    configuration_fingerprint: str,
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
        fault_model_fingerprint=model_fingerprint,
        decoder_configuration_fingerprint=configuration_fingerprint,
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
