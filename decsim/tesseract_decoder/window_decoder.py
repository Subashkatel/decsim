"""Tesseract decoding over one explicitly physical window fault model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
import os
import secrets
import threading
from typing import Optional

from ..adapters.window_decode_results import (
    BackendDecodeOutcome,
    BackendDecodeStatus,
    BackendFailureReason,
    decoder_configuration_fingerprint,
    empty_fault_model_outcome,
    fault_model_fingerprint,
)
from ..detector_error_model import (
    FaultRepresentation,
    validate_placed_fault_matrices,
)
from ..message import RunSeedReservation


_DETECTOR_ORDER_METHODS = frozenset({
    "index",
    "breadth_first",
    "coordinate",
})


@dataclass(frozen=True)
class TesseractDecoderConfig:
    """Deterministic search profile for the official Tesseract backend.

    The search limits and ensemble size default to the official short-beam
    profile. This adapter disables backend merging to retain one-to-one
    physical column identity. A ``RunSpec`` supplies the detector-order seed;
    direct offline callers should set one for reproducible results.
    """

    detector_beam: int = 15
    beam_climbing: bool = True
    no_revisit_detectors: bool = True
    priority_queue_limit: int = 200_000
    detector_order_method: str = "index"
    detector_order_count: int = 16
    detector_order_seed: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("detector_beam", "priority_queue_limit",
                     "detector_order_count"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact built-in int")
        if self.detector_beam < 0:
            raise ValueError("detector_beam must be nonnegative")
        if self.priority_queue_limit < 1:
            raise ValueError("priority_queue_limit must be positive")
        if self.detector_order_count < 1:
            raise ValueError("detector_order_count must be positive")
        for name in ("beam_climbing", "no_revisit_detectors"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact built-in bool")
        if (
            type(self.detector_order_method) is not str
            or self.detector_order_method not in _DETECTOR_ORDER_METHODS
        ):
            methods = ", ".join(sorted(_DETECTOR_ORDER_METHODS))
            raise ValueError(
                f"detector_order_method must be one of: {methods}"
            )
        seed = self.detector_order_seed
        if seed is not None and (
            type(seed) is not int or not 0 <= seed < (1 << 64)
        ):
            raise ValueError(
                "detector_order_seed must be an unsigned 64-bit built-in "
                "integer or None"
            )


@dataclass(frozen=True)
class _CompiledTesseract:
    backend_decoder: object
    configuration_fingerprint: str


class _BackendConstructionFailed(RuntimeError):
    """The optional backend rejected a locally validated configuration."""


def _load_tesseract_backend():
    try:
        import tesseract_decoder
    except ImportError as error:
        raise ImportError(
            "Tesseract decoding requires the optional "
            "'tesseract-decoder' package"
        ) from error
    return tesseract_decoder


def _normalized_coordinates(model, detector_count: int):
    coordinates = model.detector_coordinates
    if coordinates is None:
        return tuple(() for _ in range(detector_count))
    if len(coordinates) != detector_count:
        raise ValueError(
            "detector coordinate count does not match physical detector rows"
        )
    normalized = []
    for detector_index, coordinate in enumerate(coordinates):
        try:
            values = tuple(coordinate)
        except TypeError as error:
            raise TypeError(
                f"detector coordinate {detector_index} must be an iterable"
            ) from error
        row = []
        for coordinate_index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(
                    f"detector coordinate {detector_index}[{coordinate_index}] "
                    "must be a real number"
                )
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(
                    f"detector coordinate {detector_index}[{coordinate_index}] "
                    "must be finite"
                )
            row.append(number)
        normalized.append(tuple(row))
    return tuple(normalized)


def _validate_coordinate_order(coordinates) -> None:
    if not coordinates:
        return
    coordinate_dimension = len(coordinates[0])
    if coordinate_dimension == 0:
        raise ValueError(
            "coordinate detector order requires coordinates for every detector"
        )
    if any(len(coordinate) != coordinate_dimension
           for coordinate in coordinates):
        raise ValueError(
            "coordinate detector order requires equal coordinate dimensions"
        )


def _validated_priors(priors, fault_count: int):
    import numpy as np

    values = np.asarray(priors)
    if values.ndim != 1 or values.shape[0] != fault_count:
        raise ValueError(
            "physical priors must have one entry per physical fault column"
        )
    normalized = []
    for fault_index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"physical prior at column {fault_index} must be real"
            )
        probability = float(value)
        if not math.isfinite(probability) or not 0 < probability <= 0.5:
            raise ValueError(
                f"physical prior at column {fault_index} must be finite and "
                f"satisfy 0 < p <= 0.5; got {probability!r}"
            )
        normalized.append(probability)
    return tuple(normalized)


def _build_detector_error_model(model, physical_faults):
    import numpy as np
    import stim

    validate_placed_fault_matrices(
        physical_faults.check,
        physical_faults.observables,
        location="Tesseract physical window model",
    )
    check = np.asarray(physical_faults.check, dtype=np.uint8)
    observables = np.asarray(
        physical_faults.observables,
        dtype=np.uint8,
    )
    detector_count, fault_count = check.shape
    if observables.shape[1] != fault_count:
        raise ValueError(
            "physical check and observable matrices have different fault counts"
        )
    if len(model.detector_ids) != detector_count:
        raise ValueError(
            "window detector identities do not match physical detector rows"
        )
    priors = _validated_priors(physical_faults.priors, fault_count)
    coordinates = _normalized_coordinates(model, detector_count)

    detector_error_model = stim.DetectorErrorModel()
    for fault_index, probability in enumerate(priors):
        targets = [
            stim.DemTarget.relative_detector_id(int(detector_index))
            for detector_index in np.nonzero(check[:, fault_index])[0]
        ]
        targets.extend(
            stim.DemTarget.logical_observable_id(int(observable_index))
            for observable_index in np.nonzero(
                observables[:, fault_index]
            )[0]
        )
        detector_error_model.append(
            stim.DemInstruction("error", [probability], targets)
        )

    for detector_index, coordinate in enumerate(coordinates):
        detector_error_model.append(
            stim.DemInstruction(
                "detector",
                list(coordinate),
                [stim.DemTarget.relative_detector_id(detector_index)],
            )
        )
    for observable_index in range(observables.shape[0]):
        detector_error_model.append(
            stim.DemInstruction(
                "logical_observable",
                [],
                [stim.DemTarget.logical_observable_id(observable_index)],
            )
        )

    if detector_error_model.num_errors != fault_count:
        raise ValueError("synthetic Tesseract model changed physical fault arity")
    if detector_error_model.num_detectors != detector_count:
        raise ValueError(
            "synthetic Tesseract model changed physical detector arity"
        )
    if detector_error_model.num_observables != observables.shape[0]:
        raise ValueError(
            "synthetic Tesseract model changed logical-observable arity"
        )
    round_trip_coordinates = detector_error_model.get_detector_coordinates()
    recovered_coordinates = tuple(
        tuple(float(value) for value in round_trip_coordinates[index])
        for index in range(detector_count)
    )
    if recovered_coordinates != coordinates:
        raise ValueError(
            "synthetic Tesseract model changed detector coordinates"
        )
    return detector_error_model, coordinates


def _failed_outcome(
    *,
    status: BackendDecodeStatus,
    reason: BackendFailureReason,
    physical_faults,
    configuration_fingerprint: str,
    physical_correction=None,
    reconstructed_syndrome=None,
) -> BackendDecodeOutcome:
    return BackendDecodeOutcome(
        status=status,
        failure_reason=reason,
        physical_correction=physical_correction,
        component_correction=None,
        reconstructed_syndrome=reconstructed_syndrome,
        iterations=None,
        iteration_limit=None,
        posterior_log_likelihood_ratios=None,
        fault_model_fingerprint=fault_model_fingerprint(physical_faults),
        decoder_configuration_fingerprint=configuration_fingerprint,
    )


class TesseractWindowDecoder:
    """Decode one physical fault view with the official Tesseract backend."""

    def __init__(
        self,
        configuration: Optional[TesseractDecoderConfig] = None,
    ) -> None:
        self.configuration = (
            TesseractDecoderConfig()
            if configuration is None
            else configuration
        )
        if not isinstance(self.configuration, TesseractDecoderConfig):
            raise TypeError(
                "configuration must be a TesseractDecoderConfig"
            )
        self._explicit_seed = self.configuration.detector_order_seed
        self._effective_seed = self._explicit_seed
        self._seed_lock = threading.Lock()
        self._pending_run_seed = None
        self._run_seed_claimed = False
        self._stochastic_use_started = False
        self._compiled_decoders: dict = {}
        self._worker_process_id = os.getpid()
        self._worker_thread_id = None

    def reserve_run_seed(self, seed: Optional[int]) -> RunSeedReservation:
        """Prepare deterministic detector-order state for one built run."""
        if seed is not None and (
            type(seed) is not int or not 0 <= seed < (1 << 64)
        ):
            raise TypeError(
                "TesseractWindowDecoder run root must be an unsigned 64-bit "
                "built-in integer or None"
            )
        with self._seed_lock:
            if self._stochastic_use_started:
                raise ValueError(
                    "TesseractWindowDecoder was already used and cannot be rebound"
                )
            if self._run_seed_claimed:
                raise ValueError(
                    "TesseractWindowDecoder is already claimed by a built run"
                )
            if self._pending_run_seed is not None:
                raise ValueError(
                    "TesseractWindowDecoder already has a pending run seed"
                )
            if seed is not None and self._explicit_seed is not None:
                raise ValueError(
                    "TesseractWindowDecoder has an explicit seed that conflicts "
                    "with the numeric run root"
                )
            if seed is not None:
                source = "derived"
                effective_seed = seed
            elif self._explicit_seed is not None:
                source = "explicit_local"
                effective_seed = self._explicit_seed
            else:
                source = "entropy"
                effective_seed = secrets.randbits(64)
            reported_seed = (
                None if source == "entropy" else effective_seed
            )
            reservation = RunSeedReservation(
                proposed_seed_source=source,
                proposed_seed=reported_seed,
                prepared_state=effective_seed,
            )
            self._pending_run_seed = reservation
            return reservation

    def cancel_run_seed(self, reservation: RunSeedReservation) -> None:
        with self._seed_lock:
            if self._pending_run_seed is reservation:
                self._pending_run_seed = None

    def commit_run_seed(self, reservation: RunSeedReservation) -> None:
        with self._seed_lock:
            if self._pending_run_seed is not reservation:
                raise ValueError(
                    "TesseractWindowDecoder can commit only its pending run seed"
                )
            self._effective_seed = reservation.prepared_state
            self._pending_run_seed = None
            self._run_seed_claimed = True
            self._compiled_decoders.clear()

    def decode(self, model, syndrome) -> BackendDecodeOutcome:
        """Return an immutable, parity-validated outcome from one backend call."""
        import numpy as np

        physical_faults = model.require_faults(FaultRepresentation.PHYSICAL)
        syndrome_array = np.asarray(syndrome)
        if (
            syndrome_array.ndim != 1
            or syndrome_array.shape[0] != physical_faults.check.shape[0]
        ):
            raise ValueError(
                "syndrome arity does not match the physical fault model"
            )
        if not np.all((syndrome_array == 0) | (syndrome_array == 1)):
            raise ValueError("syndrome must contain only binary values")
        syndrome_array = syndrome_array.astype(np.uint8, copy=False)

        if physical_faults.check.shape[1] == 0:
            coordinates = _normalized_coordinates(
                model,
                physical_faults.check.shape[0],
            )
            if self.configuration.detector_order_method == "coordinate":
                _validate_coordinate_order(coordinates)
            configuration_fingerprint = self._configuration_fingerprint(
                coordinates,
                self._effective_seed,
            )
            return empty_fault_model_outcome(
                physical_faults,
                syndrome_array,
                decoder_configuration_fingerprint=
                    configuration_fingerprint,
            )

        try:
            compiled = self._compiled_decoder(model, physical_faults)
        except ImportError:
            raise
        except _BackendConstructionFailed:
            coordinates = _normalized_coordinates(
                model,
                physical_faults.check.shape[0],
            )
            return _failed_outcome(
                status=BackendDecodeStatus.BACKEND_ERROR,
                reason=BackendFailureReason.UPSTREAM_EXCEPTION,
                physical_faults=physical_faults,
                configuration_fingerprint=self._configuration_fingerprint(
                    coordinates,
                    self._effective_seed,
                ),
            )

        try:
            selected_error_indices = tuple(
                compiled.backend_decoder.decode_to_errors(
                    syndrome_array.astype(bool, copy=False)
                )
            )
            low_confidence = bool(
                compiled.backend_decoder.low_confidence_flag
            )
        except Exception:
            return _failed_outcome(
                status=BackendDecodeStatus.BACKEND_ERROR,
                reason=BackendFailureReason.UPSTREAM_EXCEPTION,
                physical_faults=physical_faults,
                configuration_fingerprint=
                    compiled.configuration_fingerprint,
            )

        correction, invalid_reason = self._correction_from_error_indices(
            selected_error_indices,
            physical_faults.check.shape[1],
        )
        if invalid_reason is not None:
            outcome = _failed_outcome(
                status=BackendDecodeStatus.INVALID_CORRECTION,
                reason=invalid_reason,
                physical_faults=physical_faults,
                configuration_fingerprint=
                    compiled.configuration_fingerprint,
            )
        else:
            reconstructed = (
                np.asarray(physical_faults.check, dtype=np.uint64)
                @ correction.astype(np.uint64)
            ) % 2
            correction_tuple = tuple(int(bit) for bit in correction)
            reconstructed_tuple = tuple(int(bit) for bit in reconstructed)
            if low_confidence:
                outcome = _failed_outcome(
                    status=BackendDecodeStatus.LOW_CONFIDENCE,
                    reason=BackendFailureReason.SEARCH_LIMIT_EXHAUSTED,
                    physical_faults=physical_faults,
                    configuration_fingerprint=
                        compiled.configuration_fingerprint,
                    physical_correction=correction_tuple,
                    reconstructed_syndrome=reconstructed_tuple,
                )
            elif not np.array_equal(reconstructed, syndrome_array):
                parity_failure = (
                    BackendFailureReason.CORRECTION_DOES_NOT_MATCH_SYNDROME
                )
                outcome = _failed_outcome(
                    status=BackendDecodeStatus.INVALID_CORRECTION,
                    reason=parity_failure,
                    physical_faults=physical_faults,
                    configuration_fingerprint=
                        compiled.configuration_fingerprint,
                    physical_correction=correction_tuple,
                    reconstructed_syndrome=reconstructed_tuple,
                )
            else:
                outcome = BackendDecodeOutcome(
                    status=BackendDecodeStatus.SUCCEEDED,
                    failure_reason=None,
                    physical_correction=correction_tuple,
                    component_correction=None,
                    reconstructed_syndrome=reconstructed_tuple,
                    iterations=None,
                    iteration_limit=None,
                    posterior_log_likelihood_ratios=None,
                    fault_model_fingerprint=
                        fault_model_fingerprint(physical_faults),
                    decoder_configuration_fingerprint=
                        compiled.configuration_fingerprint,
                )
        return outcome

    def _resolved_detector_order_seed(self) -> int:
        with self._seed_lock:
            if self._effective_seed is None:
                self._effective_seed = secrets.randbits(64)
            self._stochastic_use_started = True
            return self._effective_seed

    def _claim_worker(self) -> None:
        process_id = os.getpid()
        thread_id = threading.get_ident()
        if process_id != self._worker_process_id:
            self._compiled_decoders.clear()
            self._worker_process_id = process_id
            self._worker_thread_id = thread_id
            return
        if self._worker_thread_id is None:
            self._worker_thread_id = thread_id
        elif self._worker_thread_id != thread_id:
            raise RuntimeError(
                "one TesseractWindowDecoder cannot be shared across threads"
            )

    def _configuration_fingerprint(self, coordinates, seed) -> str:
        return decoder_configuration_fingerprint({
            "backend": "tesseract_decoder",
            "configuration": self.configuration,
            "resolved_detector_order_seed": seed,
            "detector_coordinates": coordinates,
            "merge_errors": False,
            "detector_penalty": 0.0,
            "create_visualization": False,
            "sparsify_errors": False,
        })

    def _compiled_decoder(self, model, physical_faults) -> _CompiledTesseract:
        import weakref

        self._claim_worker()
        model_identity = id(model)
        entry = self._compiled_decoders.get(model_identity)
        if entry is not None and entry[0]() is model:
            return entry[1]

        detector_error_model, coordinates = _build_detector_error_model(
            model,
            physical_faults,
        )
        if self.configuration.detector_order_method == "coordinate":
            _validate_coordinate_order(coordinates)
        seed = self._resolved_detector_order_seed()
        backend = _load_tesseract_backend()
        methods = {
            "index": backend.utils.DetOrder.DetIndex,
            "breadth_first": backend.utils.DetOrder.DetBFS,
            "coordinate": backend.utils.DetOrder.DetCoordinate,
        }
        try:
            detector_orders = backend.utils.build_det_orders(
                detector_error_model,
                self.configuration.detector_order_count,
                methods[self.configuration.detector_order_method],
                seed,
            )
            upstream_configuration = backend.tesseract.TesseractConfig(
                dem=detector_error_model,
                det_beam=self.configuration.detector_beam,
                beam_climbing=self.configuration.beam_climbing,
                no_revisit_dets=self.configuration.no_revisit_detectors,
                verbose=False,
                merge_errors=False,
                pqlimit=self.configuration.priority_queue_limit,
                det_orders=detector_orders,
                det_penalty=0.0,
                create_visualization=False,
                sparsify_errors=False,
                sparsify_base_degree=-1,
                sparsify_max_degree=-1,
                sparsify_reactivate_limit=-1,
            )
            backend_decoder = upstream_configuration.compile_decoder()
        except ImportError:
            raise
        except Exception as error:
            raise _BackendConstructionFailed from error
        compiled = _CompiledTesseract(
            backend_decoder=backend_decoder,
            configuration_fingerprint=self._configuration_fingerprint(
                coordinates,
                seed,
            ),
        )

        def discard_dead_model(reference) -> None:
            current = self._compiled_decoders.get(model_identity)
            if current is not None and current[0] is reference:
                del self._compiled_decoders[model_identity]

        model_reference = weakref.ref(model, discard_dead_model)
        self._compiled_decoders[model_identity] = (
            model_reference,
            compiled,
        )
        return compiled

    @staticmethod
    def _correction_from_error_indices(indices, fault_count: int):
        import numpy as np

        correction = np.zeros(fault_count, dtype=np.uint8)
        seen = set()
        for value in indices:
            if isinstance(value, bool) or not isinstance(value, Integral):
                return None, BackendFailureReason.CORRECTION_NOT_BINARY
            fault_index = int(value)
            if (
                not 0 <= fault_index < fault_count
                or fault_index in seen
            ):
                return None, BackendFailureReason.CORRECTION_WRONG_ARITY
            seen.add(fault_index)
            correction[fault_index] = 1
        return correction, None
