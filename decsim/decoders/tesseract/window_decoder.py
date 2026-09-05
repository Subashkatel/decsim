"""Tesseract decoding over one explicitly physical window fault model.

The official tesseract_decoder package (Google Quantum AI's Tesseract, a
search-based most-likely-error decoder; its paper is not on disk) is
compiled once per live window model from a Stim detector error model
rebuilt out of the window's physical check, observables, priors and
detector coordinates; one decode is one decode_to_errors call. Backend
merging is off so the physical columns keep their one-to-one identity.
"""

import dataclasses
import math
import os
import secrets
import threading
import weakref
from numbers import Integral, Real
from typing import Optional

import numpy
import stim

import decsim.decoders.backend_outcome as backend_outcome
import decsim.decoders.decoder as decoder_module
import decsim.detector_error_model.fault_identity_validation as fault_identity
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.seeding as seeding

_DETECTOR_ORDER_METHODS = frozenset({"index", "breadth_first", "coordinate"})
_SEED_LIMIT = 1 << 64
_Status = backend_outcome.BackendDecodeStatus
_Reason = backend_outcome.BackendFailureReason


@dataclasses.dataclass(frozen=True)
class TesseractDecoderConfig:
    """Deterministic search profile for the official Tesseract backend.

    The search limits and ensemble size default to the official
    short-beam profile. This adapter disables backend merging to retain
    one-to-one physical column identity. The run supplies the detector
    order seed; direct offline callers should set one for reproducible
    results.
    """

    detector_beam: int = 15
    beam_climbing: bool = True
    no_revisit_detectors: bool = True
    priority_queue_limit: int = 200_000
    detector_order_method: str = "index"
    detector_order_count: int = 16
    detector_order_seed: Optional[int] = None

    def __post_init__(self) -> None:
        integer_fields = (
            "detector_beam",
            "priority_queue_limit",
            "detector_order_count",
        )
        for name in integer_fields:
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
            value = getattr(self, name)
            if type(value) is not bool:
                raise TypeError(f"{name} must be an exact built-in bool")
        _check_detector_order_method(self.detector_order_method)
        _check_detector_order_seed(self.detector_order_seed)


class TesseractWindowDecoder(seeding._AtomicRunSeedConsumer):
    """Decode one physical fault view with the official Tesseract backend."""

    def __init__(
        self, configuration: Optional[TesseractDecoderConfig] = None
    ) -> None:
        if configuration is None:
            configuration = TesseractDecoderConfig()
        if not isinstance(configuration, TesseractDecoderConfig):
            raise TypeError("configuration must be a TesseractDecoderConfig")
        self.configuration = configuration
        self._initialize_run_seed_binding(configuration.detector_order_seed)
        self._effective_seed = self._explicit_seed
        self.compiled_by_model: dict = {}
        self._worker_process_id = os.getpid()
        self._worker_thread_id = None

    def decode(self, model, syndrome) -> backend_outcome.BackendDecodeOutcome:
        """An immutable, parity-validated outcome from one backend call."""
        physical_faults = model.require_faults(
            fault_models.FaultRepresentation.PHYSICAL
        )
        syndrome_array = _checked_syndrome(syndrome, physical_faults)
        if physical_faults.check.shape[1] == 0:
            return backend_outcome.empty_fault_model_outcome(syndrome_array)
        try:
            backend_decoder = self._compiled_decoder(model, physical_faults)
        except _BackendConstructionError:
            return _failed_outcome(
                status=_Status.BACKEND_ERROR, reason=_Reason.UPSTREAM_EXCEPTION
            )
        try:
            bits = syndrome_array.astype(bool, copy=False)
            error_indices = backend_decoder.decode_to_errors(bits)
            selected_error_indices = tuple(error_indices)
            low_confidence = bool(backend_decoder.low_confidence_flag)
        except Exception:
            return _failed_outcome(
                status=_Status.BACKEND_ERROR, reason=_Reason.UPSTREAM_EXCEPTION
            )
        return _outcome_of(
            selected_error_indices,
            low_confidence,
            physical_faults,
            syndrome_array,
        )

    def _entropy_seed(self):
        return secrets.randbits(64)

    def _install_run_seed_state(self, prepared_state) -> None:
        self._effective_seed = prepared_state
        self.compiled_by_model.clear()

    def _resolved_detector_order_seed(self) -> int:
        with self._run_seed_lock:
            if self._pending_run_seed is not None:
                raise RuntimeError(
                    "TesseractWindowDecoder cannot draw while a run-seed "
                    "reservation is pending"
                )
            if self._effective_seed is None:
                self._effective_seed = secrets.randbits(64)
            self._stochastic_use_started = True
            return self._effective_seed

    def _claim_worker(self) -> None:
        """One decoder serves one thread; a new process starts afresh."""
        process_id = os.getpid()
        thread_id = threading.get_ident()
        if process_id != self._worker_process_id:
            self.compiled_by_model.clear()
            self._worker_process_id = process_id
            self._worker_thread_id = thread_id
            return
        if self._worker_thread_id is None:
            self._worker_thread_id = thread_id
        elif self._worker_thread_id != thread_id:
            raise RuntimeError(
                "one TesseractWindowDecoder cannot be shared across threads"
            )

    def _compiled_decoder(self, model, physical_faults):
        """The backend compiled for one model, kept while the model lives."""
        self._claim_worker()
        model_identity = id(model)
        entry = self.compiled_by_model.get(model_identity)
        if entry is not None:
            reference, compiled = entry
            if reference() is model:
                return compiled
        detector_error_model, coordinates = detector_error_model_of(
            model, physical_faults
        )
        if self.configuration.detector_order_method == "coordinate":
            _validate_coordinate_order(coordinates)
        seed = self._resolved_detector_order_seed()
        compiled = _compile_backend(
            self.configuration, detector_error_model, seed
        )

        def discard_dead_model(reference) -> None:
            current = self.compiled_by_model.get(model_identity)
            if current is not None and current[0] is reference:
                del self.compiled_by_model[model_identity]

        model_reference = weakref.ref(model, discard_dead_model)
        self.compiled_by_model[model_identity] = (model_reference, compiled)
        return compiled


class _BackendConstructionError(RuntimeError):
    """The optional backend rejected a locally validated configuration."""


def _check_detector_order_method(method) -> None:
    if type(method) is not str or method not in _DETECTOR_ORDER_METHODS:
        methods = ", ".join(sorted(_DETECTOR_ORDER_METHODS))
        raise ValueError(f"detector_order_method must be one of: {methods}")


def _check_detector_order_seed(seed) -> None:
    if seed is None:
        return
    if type(seed) is not int or not 0 <= seed < _SEED_LIMIT:
        raise ValueError(
            "detector_order_seed must be an unsigned 64-bit built-in "
            "integer or None"
        )


def _load_tesseract_backend():
    try:
        import tesseract_decoder
    except ImportError as error:
        raise ImportError(
            "Tesseract decoding requires the optional "
            "'tesseract-decoder' package"
        ) from error
    return tesseract_decoder


def _compile_backend(
    configuration: TesseractDecoderConfig, detector_error_model, seed: int
):
    """The backend decoder for one model; construction failures are typed."""
    backend = _load_tesseract_backend()
    methods = {
        "index": backend.utils.DetOrder.DetIndex,
        "breadth_first": backend.utils.DetOrder.DetBFS,
        "coordinate": backend.utils.DetOrder.DetCoordinate,
    }
    method = methods[configuration.detector_order_method]
    try:
        detector_orders = backend.utils.build_det_orders(
            detector_error_model,
            configuration.detector_order_count,
            method,
            seed,
        )
        upstream_configuration = backend.tesseract.TesseractConfig(
            dem=detector_error_model,
            det_beam=configuration.detector_beam,
            beam_climbing=configuration.beam_climbing,
            no_revisit_dets=configuration.no_revisit_detectors,
            verbose=False,
            merge_errors=False,
            pqlimit=configuration.priority_queue_limit,
            det_orders=detector_orders,
            det_penalty=0.0,
            create_visualization=False,
            sparsify_errors=False,
            sparsify_base_degree=-1,
            sparsify_max_degree=-1,
            sparsify_reactivate_limit=-1,
        )
        return upstream_configuration.compile_decoder()
    except Exception as error:
        raise _BackendConstructionError from error


def _checked_syndrome(syndrome, physical_faults):
    syndrome_array = numpy.asarray(syndrome)
    detector_count = physical_faults.check.shape[0]
    if syndrome_array.ndim != 1 or syndrome_array.shape[0] != detector_count:
        raise ValueError(
            "syndrome arity does not match the physical fault model"
        )
    is_zero = syndrome_array == 0
    is_one = syndrome_array == 1
    is_bit = is_zero | is_one
    if not numpy.all(is_bit):
        raise ValueError("syndrome must contain only binary values")
    return syndrome_array.astype(numpy.uint8, copy=False)


def _normalized_coordinates(model, detector_count: int) -> tuple:
    coordinates = model.detector_coordinates
    if coordinates is None:
        return ((),) * detector_count
    if len(coordinates) != detector_count:
        raise ValueError(
            "detector coordinate count does not match physical detector rows"
        )
    normalized = []
    for detector_index, coordinate in enumerate(coordinates):
        row = _normalized_coordinate(detector_index, coordinate)
        normalized.append(row)
    return tuple(normalized)


def _normalized_coordinate(detector_index: int, coordinate) -> tuple:
    try:
        values = tuple(coordinate)
    except TypeError as error:
        raise TypeError(
            f"detector coordinate {detector_index} must be an iterable"
        ) from error
    row = []
    for coordinate_index, value in enumerate(values):
        number = _finite_coordinate(detector_index, coordinate_index, value)
        row.append(number)
    return tuple(row)


def _finite_coordinate(
    detector_index: int, coordinate_index: int, value
) -> float:
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
    return number


def _validate_coordinate_order(coordinates) -> None:
    if not coordinates:
        return
    coordinate_dimension = len(coordinates[0])
    if coordinate_dimension == 0:
        raise ValueError(
            "coordinate detector order requires coordinates for every detector"
        )
    for coordinate in coordinates:
        if len(coordinate) != coordinate_dimension:
            raise ValueError(
                "coordinate detector order requires equal coordinate dimensions"
            )


def _validated_priors(priors, fault_count: int) -> tuple:
    values = numpy.asarray(priors)
    if values.ndim != 1 or values.shape[0] != fault_count:
        raise ValueError(
            "physical priors must have one entry per physical fault column"
        )
    normalized = []
    for fault_index, value in enumerate(values):
        probability = _validated_prior(fault_index, value)
        normalized.append(probability)
    return tuple(normalized)


def _validated_prior(fault_index: int, value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"physical prior at column {fault_index} must be real")
    probability = float(value)
    if not math.isfinite(probability) or not 0 < probability <= 0.5:
        raise ValueError(
            f"physical prior at column {fault_index} must be finite and "
            f"satisfy 0 < p <= 0.5; got {probability!r}"
        )
    return probability


def detector_error_model_of(model, physical_faults) -> tuple:
    """(Stim detector error model, coordinates) of one physical view."""
    fault_identity.validate_placed_fault_matrices(
        physical_faults.check,
        physical_faults.observables,
        location="Tesseract physical window model",
    )
    check = physical_faults.check
    # observables are few rows; dense per-fault columns are cheap to read
    observables = physical_faults.observables.toarray()
    observables = observables.astype(numpy.uint8, copy=False)
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
    _append_errors(detector_error_model, check, observables, priors)
    _append_detectors(detector_error_model, coordinates)
    _append_observables(detector_error_model, observables.shape[0])
    _check_round_trip(
        detector_error_model,
        fault_count,
        detector_count,
        observables.shape[0],
        coordinates,
    )
    return detector_error_model, coordinates


def _append_errors(detector_error_model, check, observables, priors) -> None:
    for fault_index, probability in enumerate(priors):
        targets = _error_targets(check, observables, fault_index)
        instruction = stim.DemInstruction("error", [probability], targets)
        detector_error_model.append(instruction)


def _error_targets(check, observables, fault_index: int) -> list:
    start = check.indptr[fault_index]
    end = check.indptr[fault_index + 1]
    targets = []
    for detector_index in check.indices[start:end]:
        detector = int(detector_index)
        target = stim.DemTarget.relative_detector_id(detector)
        targets.append(target)
    column = observables[:, fault_index]
    flipped = numpy.nonzero(column)
    for observable_index in flipped[0]:
        observable = int(observable_index)
        target = stim.DemTarget.logical_observable_id(observable)
        targets.append(target)
    return targets


def _append_detectors(detector_error_model, coordinates) -> None:
    for detector_index, coordinate in enumerate(coordinates):
        target = stim.DemTarget.relative_detector_id(detector_index)
        instruction = stim.DemInstruction(
            "detector", list(coordinate), [target]
        )
        detector_error_model.append(instruction)


def _append_observables(detector_error_model, observable_count: int) -> None:
    for observable_index in range(observable_count):
        target = stim.DemTarget.logical_observable_id(observable_index)
        instruction = stim.DemInstruction("logical_observable", [], [target])
        detector_error_model.append(instruction)


def _check_round_trip(
    detector_error_model,
    fault_count: int,
    detector_count: int,
    observable_count: int,
    coordinates,
) -> None:
    if detector_error_model.num_errors != fault_count:
        raise ValueError(
            "synthetic Tesseract model changed physical fault arity"
        )
    if detector_error_model.num_detectors != detector_count:
        raise ValueError(
            "synthetic Tesseract model changed physical detector arity"
        )
    if detector_error_model.num_observables != observable_count:
        raise ValueError(
            "synthetic Tesseract model changed logical-observable arity"
        )
    round_trip = detector_error_model.get_detector_coordinates()
    recovered = []
    for index in range(detector_count):
        coordinate = _float_tuple(round_trip[index])
        recovered.append(coordinate)
    if tuple(recovered) != coordinates:
        raise ValueError(
            "synthetic Tesseract model changed detector coordinates"
        )


def _float_tuple(values) -> tuple:
    numbers = []
    for value in values:
        numbers.append(float(value))
    return tuple(numbers)


def _correction_from_error_indices(indices, fault_count: int) -> tuple:
    """(correction, None), or (None, why the indices are no correction)."""
    correction = numpy.zeros(fault_count, dtype=numpy.uint8)
    seen = set()
    for value in indices:
        if isinstance(value, bool) or not isinstance(value, Integral):
            reason = _Reason.CORRECTION_NOT_BINARY
            return None, reason
        fault_index = int(value)
        if not 0 <= fault_index < fault_count or fault_index in seen:
            reason = _Reason.CORRECTION_WRONG_ARITY
            return None, reason
        seen.add(fault_index)
        correction[fault_index] = 1
    return correction, None


def _outcome_of(
    selected_error_indices: tuple,
    low_confidence: bool,
    physical_faults,
    syndrome_array,
) -> backend_outcome.BackendDecodeOutcome:
    """The outcome of one backend answer: its status and its correction."""
    fault_count = physical_faults.check.shape[1]
    correction, invalid_reason = _correction_from_error_indices(
        selected_error_indices, fault_count
    )
    if invalid_reason is not None:
        return _failed_outcome(
            status=_Status.INVALID_CORRECTION, reason=invalid_reason
        )
    reconstructed = decoder_module.parity_product(
        physical_faults.check, correction
    )
    correction_tuple = decoder_module.bit_tuple(correction)
    reconstructed_tuple = decoder_module.bit_tuple(reconstructed)
    if low_confidence:
        return _failed_outcome(
            status=_Status.LOW_CONFIDENCE,
            reason=_Reason.SEARCH_LIMIT_EXHAUSTED,
            physical_correction=correction_tuple,
            reconstructed_syndrome=reconstructed_tuple,
        )
    if not numpy.array_equal(reconstructed, syndrome_array):
        return _failed_outcome(
            status=_Status.INVALID_CORRECTION,
            reason=_Reason.CORRECTION_DOES_NOT_MATCH_SYNDROME,
            physical_correction=correction_tuple,
            reconstructed_syndrome=reconstructed_tuple,
        )
    return backend_outcome.BackendDecodeOutcome(
        status=_Status.SUCCEEDED,
        failure_reason=None,
        physical_correction=correction_tuple,
        component_correction=None,
        reconstructed_syndrome=reconstructed_tuple,
        iterations=None,
        iteration_limit=None,
        posterior_log_likelihood_ratios=None,
    )


def _failed_outcome(
    *,
    status: backend_outcome.BackendDecodeStatus,
    reason: backend_outcome.BackendFailureReason,
    physical_correction=None,
    reconstructed_syndrome=None,
) -> backend_outcome.BackendDecodeOutcome:
    return backend_outcome.BackendDecodeOutcome(
        status=status,
        failure_reason=reason,
        physical_correction=physical_correction,
        component_correction=None,
        reconstructed_syndrome=reconstructed_syndrome,
        iterations=None,
        iteration_limit=None,
        posterior_log_likelihood_ratios=None,
    )
