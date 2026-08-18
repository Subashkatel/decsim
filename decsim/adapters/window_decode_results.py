"""Shared helpers for turning window decoder selections into DecodeResult objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import math
from numbers import Integral, Real
import struct

from ..message import DecodeJob, DecodeResult, DependencyResidual


class BackendDecodeStatus(Enum):
    """Backend-neutral disposition of one window decode attempt."""

    SUCCEEDED = "succeeded"
    LOW_CONFIDENCE = "low_confidence"
    NONCONVERGED = "nonconverged"
    INVALID_CORRECTION = "invalid_correction"
    EMPTY_MODEL_UNSATISFIABLE = "empty_model_unsatisfiable"
    BACKEND_ERROR = "backend_error"


class BackendFailureReason(Enum):
    """Typed reason a backend attempt could not be committed."""

    SEARCH_LIMIT_EXHAUSTED = "search_limit_exhausted"
    NO_CONVERGED_RELAY_SOLUTION = "no_converged_relay_solution"
    CORRECTION_NOT_BINARY = "correction_not_binary"
    CORRECTION_WRONG_ARITY = "correction_wrong_arity"
    CORRECTION_DOES_NOT_MATCH_SYNDROME = \
        "correction_does_not_match_syndrome"
    NONZERO_SYNDROME_WITHOUT_FAULTS = \
        "nonzero_syndrome_without_faults"
    UPSTREAM_EXCEPTION = "upstream_exception"


_STATUS_REASONS = {
    BackendDecodeStatus.LOW_CONFIDENCE: frozenset({
        BackendFailureReason.SEARCH_LIMIT_EXHAUSTED,
    }),
    BackendDecodeStatus.NONCONVERGED: frozenset({
        BackendFailureReason.NO_CONVERGED_RELAY_SOLUTION,
    }),
    BackendDecodeStatus.INVALID_CORRECTION: frozenset({
        BackendFailureReason.CORRECTION_NOT_BINARY,
        BackendFailureReason.CORRECTION_WRONG_ARITY,
        BackendFailureReason.CORRECTION_DOES_NOT_MATCH_SYNDROME,
    }),
    BackendDecodeStatus.EMPTY_MODEL_UNSATISFIABLE: frozenset({
        BackendFailureReason.NONZERO_SYNDROME_WITHOUT_FAULTS,
    }),
    BackendDecodeStatus.BACKEND_ERROR: frozenset({
        BackendFailureReason.UPSTREAM_EXCEPTION,
    }),
}


def _binary_tuple(value, *, name: str, allow_none: bool = True):
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} is required")
    try:
        values = tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a one-dimensional iterable") from error
    normalized = []
    for index, bit in enumerate(values):
        if not isinstance(bit, Integral) or int(bit) not in (0, 1):
            raise ValueError(f"{name}[{index}] must be binary")
        normalized.append(int(bit))
    return tuple(normalized)


def _float_tuple(value, *, name: str):
    if value is None:
        return None
    try:
        values = tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a one-dimensional iterable") from error
    normalized = []
    for index, item in enumerate(values):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(f"{name}[{index}] must be a real number")
        number = float(item)
        if math.isnan(number):
            raise ValueError(f"{name}[{index}] cannot be NaN")
        normalized.append(number)
    return tuple(normalized)


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
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 hexadecimal digest")
    if value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256 hexadecimal digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(
            f"{name} must be a lowercase SHA-256 hexadecimal digest"
        ) from error


@dataclass(frozen=True)
class BackendDecodeOutcome:
    """Immutable correction, disposition, and diagnostics from one backend call."""

    status: BackendDecodeStatus
    failure_reason: BackendFailureReason | None
    physical_correction: tuple[int, ...] | None
    component_correction: tuple[int, ...] | None
    reconstructed_syndrome: tuple[int, ...] | None
    iterations: int | None
    iteration_limit: int | None
    posterior_log_likelihood_ratios: tuple[float, ...] | None
    fault_model_fingerprint: str
    decoder_configuration_fingerprint: str

    def __post_init__(self) -> None:
        if self.status is BackendDecodeStatus.SUCCEEDED:
            if self.failure_reason is not None:
                raise ValueError("a successful outcome cannot have a failure reason")
        else:
            if self.failure_reason is None:
                raise ValueError("a failed outcome requires a typed failure reason")
            if self.failure_reason not in _STATUS_REASONS[self.status]:
                raise ValueError(
                    f"{self.failure_reason.value} is not valid for "
                    f"status {self.status.value}"
                )

        physical_correction = _binary_tuple(
            self.physical_correction,
            name="physical_correction",
            allow_none=self.status is not BackendDecodeStatus.SUCCEEDED,
        )
        reconstructed_syndrome = _binary_tuple(
            self.reconstructed_syndrome,
            name="reconstructed_syndrome",
            allow_none=self.status is not BackendDecodeStatus.SUCCEEDED,
        )
        object.__setattr__(self, "physical_correction", physical_correction)
        object.__setattr__(
            self,
            "component_correction",
            _binary_tuple(self.component_correction, name="component_correction"),
        )
        object.__setattr__(
            self,
            "reconstructed_syndrome",
            reconstructed_syndrome,
        )
        object.__setattr__(
            self,
            "iterations",
            _nonnegative_integer(self.iterations, name="iterations"),
        )
        object.__setattr__(
            self,
            "iteration_limit",
            _nonnegative_integer(self.iteration_limit, name="iteration_limit"),
        )
        object.__setattr__(
            self,
            "posterior_log_likelihood_ratios",
            _float_tuple(
                self.posterior_log_likelihood_ratios,
                name="posterior_log_likelihood_ratios",
            ),
        )
        _validate_fingerprint(
            self.fault_model_fingerprint,
            name="fault_model_fingerprint",
        )
        _validate_fingerprint(
            self.decoder_configuration_fingerprint,
            name="decoder_configuration_fingerprint",
        )

    @property
    def succeeded(self) -> bool:
        return self.status is BackendDecodeStatus.SUCCEEDED


class DecoderAttemptFailed(RuntimeError):
    """A runtime decoder attempt failed with same-job immutable evidence."""

    def __init__(self, job: DecodeJob, outcome: BackendDecodeOutcome):
        if not isinstance(outcome, BackendDecodeOutcome):
            raise TypeError("outcome must be a BackendDecodeOutcome")
        if outcome.succeeded:
            raise ValueError("a successful outcome cannot fail a decoder attempt")
        self.job_identity = (job.op_id, job.window_id, job.attempt)
        self.outcome = outcome
        super().__init__(
            f"decoder attempt {self.job_identity} failed: "
            f"{outcome.status.value}/{outcome.failure_reason.value}"
        )


def _canonical_bytes(value) -> bytes:
    """Encode supported scientific values without process-local identities."""
    import numpy as np

    if value is None:
        return b"n"
    if isinstance(value, Enum):
        return b"e" + _canonical_bytes(value.__class__.__qualname__) + \
            _canonical_bytes(value.value)
    if isinstance(value, bool):
        return b"b1" if value else b"b0"
    if isinstance(value, Integral):
        return b"i" + str(int(value)).encode("ascii") + b";"
    if isinstance(value, Real):
        number = float(value)
        if math.isnan(number):
            raise ValueError("fingerprinted values cannot contain NaN")
        return b"f" + struct.pack(">d", number)
    if isinstance(value, str):
        encoded = value.encode("utf8")
        return b"s" + str(len(encoded)).encode("ascii") + b":" + encoded
    if isinstance(value, bytes):
        return b"y" + str(len(value)).encode("ascii") + b":" + value
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays cannot be fingerprinted")
        array = np.ascontiguousarray(value)
        return b"a" + _canonical_bytes(array.dtype.str) + \
            _canonical_bytes(tuple(array.shape)) + _canonical_bytes(array.tobytes())
    if is_dataclass(value) and not isinstance(value, type):
        return b"d" + _canonical_bytes(value.__class__.__qualname__) + \
            _canonical_bytes(tuple(
                (field.name, getattr(value, field.name))
                for field in fields(value)
            ))
    if isinstance(value, Mapping):
        items = sorted(
            ((_canonical_bytes(key), _canonical_bytes(item))
             for key, item in value.items()),
            key=lambda pair: pair[0],
        )
        return b"m" + _canonical_bytes(tuple(items))
    if isinstance(value, tuple):
        encoded = b"".join(_canonical_bytes(item) for item in value)
        return b"t" + str(len(value)).encode("ascii") + b":" + encoded
    if isinstance(value, list):
        return b"l" + _canonical_bytes(tuple(value))
    raise TypeError(
        f"cannot fingerprint value of type {type(value).__qualname__}"
    )


def _sha256_fingerprint(value) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def fault_model_fingerprint(placed_faults) -> str:
    """Fingerprint every column-aligned input and boundary field of one view."""
    return _sha256_fingerprint({
        "representation": placed_faults.representation,
        "check": placed_faults.check,
        "priors": placed_faults.priors,
        "observables": placed_faults.observables,
        "owned": placed_faults.owned,
        "future_flips": placed_faults.future_flips,
        "boundary_flips": placed_faults.boundary_flips,
        "source_fault_ids": placed_faults.source_fault_ids,
    })


def decoder_configuration_fingerprint(configuration) -> str:
    """Fingerprint one frozen backend profile using deterministic bytes."""
    return _sha256_fingerprint(configuration)


def validate_backend_outcome(
    outcome: BackendDecodeOutcome,
    model,
    placed_faults,
    syndrome,
) -> None:
    """Validate a same-model outcome before any correction can be committed."""
    import numpy as np

    if not isinstance(outcome, BackendDecodeOutcome):
        raise TypeError("backend must return a BackendDecodeOutcome")
    expected_fingerprint = fault_model_fingerprint(placed_faults)
    if outcome.fault_model_fingerprint != expected_fingerprint:
        raise ValueError("backend outcome belongs to a different fault model")

    syndrome = np.asarray(syndrome)
    if syndrome.ndim != 1 or syndrome.shape[0] != placed_faults.check.shape[0]:
        raise ValueError("syndrome arity does not match the placed fault model")
    if not np.all((syndrome == 0) | (syndrome == 1)):
        raise ValueError("syndrome must contain only binary values")
    correction = None
    reconstructed = None
    if outcome.physical_correction is not None:
        correction = np.asarray(outcome.physical_correction, dtype=np.uint8)
        if correction.shape != (placed_faults.check.shape[1],):
            raise ValueError("backend correction has the wrong fault-model arity")
        reconstructed = (
            np.asarray(placed_faults.check, dtype=np.uint64)
            @ correction.astype(np.uint64)
        ) % 2
        if outcome.reconstructed_syndrome is not None and tuple(
            int(bit) for bit in reconstructed
        ) != outcome.reconstructed_syndrome:
            raise ValueError(
                "backend reconstructed syndrome does not match its correction"
            )

    if outcome.component_correction is not None:
        projection = model.physical_to_graphlike_detector_projection
        if projection is None or correction is None:
            raise ValueError(
                "component correction requires a physical correction and valid "
                "local projection"
            )
        component = (
            np.asarray(projection, dtype=np.uint64)
            @ correction.astype(np.uint64)
        ) % 2
        if tuple(int(bit) for bit in component) != outcome.component_correction:
            raise ValueError("component correction does not match the local projection")

    if outcome.succeeded and not np.array_equal(
        reconstructed,
        syndrome.astype(np.uint8),
    ):
        raise ValueError("successful correction does not match the syndrome")


def empty_fault_model_outcome(
    placed_faults,
    syndrome,
    *,
    decoder_configuration_fingerprint: str,
) -> BackendDecodeOutcome:
    """Resolve a no-fault physical model without invoking an external backend."""
    import numpy as np

    if placed_faults.check.shape[1] != 0:
        raise ValueError("empty-fault outcome requires a model with zero columns")
    syndrome = np.asarray(syndrome)
    if syndrome.ndim != 1 or syndrome.shape[0] != placed_faults.check.shape[0]:
        raise ValueError("syndrome arity does not match the empty fault model")
    if not np.all((syndrome == 0) | (syndrome == 1)):
        raise ValueError("syndrome must contain only binary values")
    model_fingerprint = fault_model_fingerprint(placed_faults)
    reconstructed = tuple(0 for _ in range(syndrome.shape[0]))
    if np.any(syndrome):
        return BackendDecodeOutcome(
            status=BackendDecodeStatus.EMPTY_MODEL_UNSATISFIABLE,
            failure_reason=
                BackendFailureReason.NONZERO_SYNDROME_WITHOUT_FAULTS,
            physical_correction=None,
            component_correction=None,
            reconstructed_syndrome=reconstructed,
            iterations=None,
            iteration_limit=None,
            posterior_log_likelihood_ratios=None,
            fault_model_fingerprint=model_fingerprint,
            decoder_configuration_fingerprint=
                decoder_configuration_fingerprint,
        )
    return BackendDecodeOutcome(
        status=BackendDecodeStatus.SUCCEEDED,
        failure_reason=None,
        physical_correction=(),
        component_correction=None,
        reconstructed_syndrome=reconstructed,
        iterations=None,
        iteration_limit=None,
        posterior_log_likelihood_ratios=None,
        fault_model_fingerprint=model_fingerprint,
        decoder_configuration_fingerprint=decoder_configuration_fingerprint,
    )


def payload_syndrome(job: DecodeJob):
    """Concatenate payload bits into one syndrome vector."""
    import numpy as np

    if not job.payloads:
        return np.zeros(0, dtype=np.uint8)
    return np.concatenate([
        np.asarray(payload.bits, dtype=np.uint8)
        for payload in job.payloads
        if payload.bits is not None
    ])


def check_syndrome_size(job: DecodeJob, syndrome, placed_faults) -> None:
    """Fail when payload bits and detector rows do not line up."""
    if syndrome.size == placed_faults.check.shape[0]:
        return
    raise ValueError(
        f"{job.label}: payload bits ({syndrome.size}) do not match the window "
        f"error model's detectors ({placed_faults.check.shape[0]}). The device and "
        "the cluster's model build must use the same folded-round convention."
    )


def result_from_selected_faults(
    job: DecodeJob,
    model,
    placed_faults,
    selected,
) -> DecodeResult:
    """Keep owned selected faults and convert them into a DecodeResult."""
    import numpy as np

    selected = np.asarray(selected, dtype=np.uint8)
    if selected.ndim != 1 or selected.shape[0] != placed_faults.check.shape[1]:
        raise ValueError(
            f"{job.label}: selected correction has shape {selected.shape}; "
            f"expected ({placed_faults.check.shape[1]},)"
        )
    committed = selected.astype(bool) & placed_faults.owned
    observable_flips = (
        placed_faults.observables @ committed.astype(np.uint8)
    ) % 2
    residual_detector_ids = _detector_ids_from_columns(
        placed_faults.boundary_flips,
        committed,
    )
    return DecodeResult(
        job.op_id,
        job.window_id,
        correction=committed.astype(np.uint8),
        logical_observables=tuple(int(bit) for bit in observable_flips),
        boundary_defects=_defects_from_columns(
            model,
            placed_faults.future_flips,
            committed,
        ),
        boundary_data=DependencyResidual(
            detector_ids=residual_detector_ids,
            defects=_defects_from_detector_ids(model, residual_detector_ids),
        ),
    )


def _detector_ids_from_columns(detector_flips, committed) -> tuple[int, ...]:
    """XOR complete global detector identities across selected columns."""
    import numpy as np

    detector_ids = set()
    for column_index in np.nonzero(committed)[0]:
        detector_ids.symmetric_difference_update(
            detector_flips.get(int(column_index), ())
        )
    return tuple(sorted(detector_ids))


def _defects_from_detector_ids(model, detector_ids) -> dict | None:
    defects: dict = {}
    for detector_id in detector_ids:
        round_index, position = model.defect_positions[detector_id]
        mask = defects.setdefault(round_index, [])
        if len(mask) <= position:
            mask.extend([0] * (position + 1 - len(mask)))
        mask[position] ^= 1
    return defects or None


def _defects_from_columns(model, detector_flips, committed) -> dict | None:
    """Convert selected correction columns into round-indexed detector masks."""
    return _defects_from_detector_ids(
        model,
        _detector_ids_from_columns(detector_flips, committed),
    )
