"""Relay-BP over one placed physical window model.

The official relay-bp package (Maurer et al. 2510.21600, the qLDPC
real-time baseline; the bb-decoders extra) is compiled once per
distinct model with a fixed gamma table drawn from the run seed, and
decode_detailed is called once per syndrome. The paper assumes
0 < p < 1/2; this adapter also accepts exactly p = 1/2 as a tested
software-profile extension with a zero prior log ratio. Decided,
majority-one and non-finite priors are refused instead of silently
transformed. Backend wall time is diagnostic only and never becomes
simulated time.
"""

import dataclasses
import hashlib
import math
import os
import secrets
import threading
from typing import TYPE_CHECKING

import numpy
import scipy.sparse

import decsim.decoders.backend_outcome as backend_outcome
import decsim.decoders.decoder as decoder_module
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoding as decoding_records
import decsim.seeding as seeding

if TYPE_CHECKING:
    import decsim.decoders.relay_belief_propagation.decoder as relay_decoder

_Status = decoding_records.BackendDecodeStatus
_Reason = decoding_records.BackendFailureReason


class RelayBeliefPropagationWindowDecoder(seeding._AtomicRunSeedConsumer):
    """Decode physical fault columns with one fixed-gamma Relay-BP profile.

    Inputs are original physical fault columns, not graph
    decompositions. Gamma tables are fixed per live model and drawn from
    the run seed; native per-shot resampling is not exposed.
    """

    def __init__(
        self, settings: "relay_decoder.RelayBeliefPropagationDecoder.Settings"
    ) -> None:
        self.settings = settings
        self._initialize_run_seed_binding(None)
        self._effective_gamma_table_seed = None
        self._thread_state = threading.local()

    def decode(
        self, window_model, syndrome
    ) -> backend_outcome.BackendDecodeOutcome:
        """Call the official detailed API once and snapshot its evidence."""
        faults = window_model.require_faults(
            fault_models.FaultRepresentation.PHYSICAL
        )
        syndrome = _validated_syndrome(syndrome)
        compiled = self.compiled_model(faults)
        if faults.check.shape[1] == 0:
            return backend_outcome.empty_fault_model_outcome(syndrome)
        if compiled.backend_construction_failed:
            return _backend_error_outcome()
        return _decode_once(compiled.backend, faults, syndrome)

    def _entropy_seed(self):
        return secrets.randbits(64)

    def _install_run_seed_state(self, prepared_state) -> None:
        self._effective_gamma_table_seed = prepared_state

    def compiled_model(self, faults) -> "_CompiledRelayModel":
        """The backend for one check matrix and priors, built once.

        A shot builds each window afresh, but its windows repeat a few
        shapes, so the backend is kept by the model's content, not by
        the object. With the fixed gamma table a kept backend answers
        each syndrome as a fresh one does.
        """
        cache = self._thread_cache()
        key = _model_key(faults)
        compiled = cache.get(key)
        if compiled is None:
            compiled = self._compile(faults)
            if len(cache) == _KEPT_MODELS:
                oldest = next(iter(cache))
                del cache[oldest]
            cache[key] = compiled
        return compiled

    def _compile(self, faults) -> "_CompiledRelayModel":
        check, priors = _validated_model(faults)
        seed = self._gamma_seed()
        column_count = check.shape[1]
        gamma_table = _gamma_table(self.settings, seed, column_count)
        if column_count == 0:
            return _CompiledRelayModel(
                backend=None, backend_construction_failed=False
            )
        backend = _construct_backend(
            self.settings, check, priors, gamma_table, seed
        )
        construction_failed = backend is None
        return _CompiledRelayModel(
            backend=backend, backend_construction_failed=construction_failed
        )

    def _gamma_seed(self) -> int:
        with self._run_seed_lock:
            if self._pending_run_seed is not None:
                raise RuntimeError(
                    "RelayBeliefPropagationWindowDecoder cannot compile while "
                    "a run-seed reservation is pending"
                )
            self._stochastic_use_started = True
            if self._effective_gamma_table_seed is None:
                self._effective_gamma_table_seed = secrets.randbits(64)
            return self._effective_gamma_table_seed

    def _thread_cache(self) -> dict:
        """This thread's compiled models; a new process starts afresh."""
        process_id = os.getpid()
        cached_process_id = getattr(self._thread_state, "process_id", None)
        if cached_process_id != process_id:
            self._thread_state.process_id = process_id
            self._thread_state.compiled_models = {}
        return self._thread_state.compiled_models


# the backends one thread keeps; a run's windows take a few shapes, and
# strong regions that each differ are let go oldest first
_KEPT_MODELS = 64


def _model_key(faults) -> bytes:
    """A digest of the check matrix and priors a backend is built from."""
    check = scipy.sparse.csr_matrix(faults.check)
    priors = numpy.asarray(faults.priors, dtype=float)
    digest = hashlib.sha256()
    shape = numpy.asarray(check.shape)
    for array in (shape, check.indptr, check.indices, priors):
        array_bytes = array.tobytes()
        digest.update(array_bytes)
    return digest.digest()


def _decode_once(backend, faults, syndrome):
    """One decode_detailed call, its correction checked, its evidence read."""
    try:
        detailed = backend.decode_detailed(syndrome)
        correction = _binary_vector(
            detailed.decoding, expected_size=faults.check.shape[1]
        )
    except _WrongCorrectionArityError:
        return _invalid_outcome(_Reason.CORRECTION_WRONG_ARITY)
    except _NonbinaryCorrectionError:
        return _invalid_outcome(_Reason.CORRECTION_NOT_BINARY)
    except Exception:
        return _backend_error_outcome()
    try:
        evidence = _detailed_evidence(detailed, faults)
    except Exception:
        return _backend_error_outcome()
    return _outcome_of(faults, syndrome, correction, evidence)


@dataclasses.dataclass(frozen=True)
class _CompiledRelayModel:
    backend: object
    backend_construction_failed: bool


@dataclasses.dataclass(frozen=True)
class _DetailedEvidence:
    """What one decode_detailed answer says beside its correction."""

    decoded_detectors: tuple
    iterations: int
    succeeded: bool


class _WrongCorrectionArityError(ValueError):
    pass


class _NonbinaryCorrectionError(ValueError):
    pass


def _load_relay_decoder_type():
    import relay_bp

    return relay_bp.RelayDecoderF32


def _gamma_table(
    settings: "relay_decoder.RelayBeliefPropagationDecoder.Settings",
    seed: int,
    column_count: int,
):
    """The fixed gamma table of one model, drawn from the seed."""
    bit_generator = numpy.random.PCG64(seed)
    generator = numpy.random.Generator(bit_generator)
    gamma_low, gamma_high = settings.gamma_interval
    shape = (settings.relay_set_count, column_count)
    gamma_table = generator.uniform(gamma_low, gamma_high, size=shape)
    gamma_table = gamma_table.astype(numpy.float64, copy=False)
    return numpy.ascontiguousarray(gamma_table)


def _construct_backend(
    settings: "relay_decoder.RelayBeliefPropagationDecoder.Settings",
    check,
    priors,
    gamma_table,
    seed: int,
):
    """The backend decoder, or None when the backend refused the model."""
    decoder_type = _load_relay_decoder_type()
    sparse_check = scipy.sparse.csr_matrix(check)
    try:
        return decoder_type(
            sparse_check,
            priors,
            alpha=settings.alpha,
            alpha_iteration_scaling_factor=(
                settings.alpha_iteration_scaling_factor
            ),
            gamma0=settings.gamma0,
            pre_iter=settings.pre_iterations,
            num_sets=settings.relay_set_count,
            set_max_iter=settings.iterations_per_set,
            gamma_dist_interval=settings.gamma_interval,
            explicit_gammas=gamma_table,
            stop_nconv=settings.converged_solution_count,
            stopping_criterion="nconv",
            logging=False,
            seed=seed,
        )
    except Exception:
        return None


def _validated_model(faults) -> tuple:
    """(check, priors) of a model the paper's assumptions hold for."""
    check = faults.check
    priors = numpy.asarray(faults.priors, dtype=float)
    is_zero = check.data == 0
    is_one = check.data == 1
    is_bit = is_zero | is_one
    if not numpy.all(is_bit):
        raise ValueError("Relay check matrix must be binary")
    for column_index, probability in enumerate(priors):
        if not math.isfinite(probability) or not 0 < probability <= 0.5:
            raise ValueError(
                "Relay prior at physical column "
                f"{column_index} must satisfy finite 0 < p <= 0.5"
            )
    return check, priors.astype(numpy.float64)


def _validated_syndrome(syndrome):
    syndrome = numpy.asarray(syndrome)
    is_zero = syndrome == 0
    is_one = syndrome == 1
    is_bit = is_zero | is_one
    if not numpy.all(is_bit):
        raise ValueError("Relay syndrome must be binary")
    return syndrome.astype(numpy.uint8)


def _binary_vector(value, *, expected_size: int) -> tuple:
    vector = numpy.asarray(value)
    if vector.ndim != 1 or vector.shape[0] != expected_size:
        raise _WrongCorrectionArityError
    is_zero = vector == 0
    is_one = vector == 1
    is_bit = is_zero | is_one
    if not numpy.all(is_bit):
        raise _NonbinaryCorrectionError
    return decoder_module.int_tuple(vector)


def _reconstruct(check, correction) -> tuple:
    correction_array = numpy.asarray(correction)
    parity = decoder_module.parity_product(check, correction_array)
    return decoder_module.int_tuple(parity)


def _detailed_evidence(detailed, faults) -> _DetailedEvidence:
    """The answer's evidence, its decoded detectors checked for arity."""
    decoded_detectors = _binary_vector(
        detailed.decoded_detectors, expected_size=faults.check.shape[0]
    )
    iterations = int(detailed.iterations)
    succeeded = bool(detailed.success)
    return _DetailedEvidence(
        decoded_detectors=decoded_detectors,
        iterations=iterations,
        succeeded=succeeded,
    )


def _outcome_of(
    faults, syndrome, correction: tuple, evidence: _DetailedEvidence
) -> backend_outcome.BackendDecodeOutcome:
    """The outcome of one answer: inconsistent, nonconverged or succeeded."""
    reconstructed = _reconstruct(faults.check, correction)
    syndrome_bits = decoder_module.int_tuple(syndrome)
    is_inconsistent = evidence.decoded_detectors != reconstructed
    if evidence.succeeded and reconstructed != syndrome_bits:
        is_inconsistent = True
    if is_inconsistent:
        return _detailed_outcome(
            correction,
            evidence,
            _Status.INVALID_CORRECTION,
            _Reason.CORRECTION_DOES_NOT_MATCH_SYNDROME,
        )
    if not evidence.succeeded:
        return _detailed_outcome(
            correction,
            evidence,
            _Status.NONCONVERGED,
            _Reason.NO_CONVERGED_RELAY_SOLUTION,
        )
    return _detailed_outcome(correction, evidence, _Status.SUCCEEDED, None)


def _detailed_outcome(
    correction: tuple, evidence: _DetailedEvidence, status, reason
) -> backend_outcome.BackendDecodeOutcome:
    return backend_outcome.BackendDecodeOutcome(
        status=status,
        failure_reason=reason,
        physical_correction=correction,
        iterations=evidence.iterations,
    )


def _invalid_outcome(reason) -> backend_outcome.BackendDecodeOutcome:
    return backend_outcome.BackendDecodeOutcome(
        status=_Status.INVALID_CORRECTION,
        failure_reason=reason,
        physical_correction=None,
        iterations=None,
    )


def _backend_error_outcome() -> backend_outcome.BackendDecodeOutcome:
    return backend_outcome.BackendDecodeOutcome(
        status=_Status.BACKEND_ERROR,
        failure_reason=_Reason.UPSTREAM_EXCEPTION,
        physical_correction=None,
        iterations=None,
    )
