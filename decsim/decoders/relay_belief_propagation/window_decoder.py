"""Relay-BP over one placed physical window model.

The official relay-bp package (Maurer et al. 2510.21600, the qLDPC
real-time baseline; the Rust source is at tmp/reference-decoders/relay-bp,
the wheel is not installed here) is compiled once per live model with a
fixed gamma table drawn from the run seed, and decode_detailed is called
once per syndrome. The paper assumes 0 < p < 1/2; this adapter also
accepts exactly p = 1/2 as a tested software-profile extension with a
zero prior log ratio. Decided, majority-one and non-finite priors are
refused instead of silently transformed. Backend wall time is
diagnostic only and never becomes simulated time.
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
import scipy.sparse

import decsim.decoders.backend_outcome as backend_outcome
import decsim.decoders.decoder as decoder_module
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.seeding as seeding

_Status = backend_outcome.BackendDecodeStatus
_Reason = backend_outcome.BackendFailureReason
_SEED_LIMIT = 2**64


class RelayBeliefPropagationWindowDecoder(seeding._AtomicRunSeedConsumer):
    """Decode physical fault columns with one fixed-gamma Relay-BP profile.

    Inputs are original physical fault columns, not graph
    decompositions. Gamma tables are fixed per live model; native
    per-shot resampling is not exposed.
    """

    _explicit_seed_label = "gamma-table seed"

    def __init__(
        self,
        *,
        alpha: Optional[float] = None,
        alpha_iteration_scaling_factor: float = 1.0,
        gamma0: Optional[float] = 0.1,
        pre_iterations: int = 80,
        relay_set_count: int = 300,
        iterations_per_set: int = 60,
        gamma_interval: tuple[float, float] = (-0.24, 0.66),
        converged_solution_count: int = 1,
        gamma_table_seed: Optional[int] = None,
    ) -> None:
        self.profile = _relay_profile(
            alpha,
            alpha_iteration_scaling_factor,
            gamma0,
            pre_iterations,
            relay_set_count,
            iterations_per_set,
            gamma_interval,
            converged_solution_count,
        )
        self._explicit_gamma_table_seed = _validated_seed(
            gamma_table_seed, "gamma_table_seed"
        )
        self._initialize_run_seed_binding(self._explicit_gamma_table_seed)
        self._effective_gamma_table_seed = self._explicit_seed
        self._thread_state = threading.local()

    def decode(
        self, window_model, syndrome
    ) -> backend_outcome.BackendDecodeOutcome:
        """Call the official detailed API once and snapshot its evidence."""
        faults = window_model.require_faults(
            fault_models.FaultRepresentation.PHYSICAL
        )
        detector_count = faults.check.shape[0]
        syndrome = _validated_syndrome(syndrome, detector_count)
        compiled = self._compiled_model(faults)
        if faults.check.shape[1] == 0:
            return backend_outcome.empty_fault_model_outcome(syndrome)
        if compiled.backend_construction_failed:
            return _backend_error_outcome()
        try:
            detailed = compiled.backend.decode_detailed(syndrome)
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

    def _entropy_seed(self):
        return secrets.randbits(64)

    def _install_run_seed_state(self, prepared_state) -> None:
        self._effective_gamma_table_seed = prepared_state

    def _compiled_model(self, faults) -> "_CompiledRelayModel":
        """The backend compiled for one model, kept while the model lives."""
        cache = self._thread_cache()
        identity = id(faults)
        entry = cache.get(identity)
        if entry is not None:
            reference, compiled = entry
            if reference() is faults:
                return compiled
        compiled = self._compile(faults)

        def discard(reference) -> None:
            current = cache.get(identity)
            if current is not None and current[0] is reference:
                del cache[identity]

        reference = weakref.ref(faults, discard)
        cache[identity] = (reference, compiled)
        return compiled

    def _compile(self, faults) -> "_CompiledRelayModel":
        check, priors = _validated_model(faults)
        seed = self._gamma_seed()
        column_count = check.shape[1]
        gamma_table = _gamma_table(self.profile, seed, column_count)
        if column_count == 0:
            return _CompiledRelayModel(
                backend=None, backend_construction_failed=False
            )
        backend = _construct_backend(
            self.profile, check, priors, gamma_table, seed
        )
        construction_failed = backend is None
        return _CompiledRelayModel(
            backend=backend, backend_construction_failed=construction_failed
        )

    def _gamma_seed(self) -> int:
        with self._run_seed_lock:
            if self._pending_run_seed is not None:
                raise RuntimeError(
                    "RelayBeliefPropagationWindowDecoder cannot compile while a run-seed "
                    "reservation is pending"
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


@dataclasses.dataclass(frozen=True)
class _RelayProfile:
    """One fixed-gamma Relay-BP profile, in relay-bp's own argument names."""

    alpha: Optional[float]
    alpha_iteration_scaling_factor: float
    gamma0: Optional[float]
    pre_iterations: int
    relay_set_count: int
    iterations_per_set: int
    gamma_interval: tuple[float, float]
    converged_solution_count: int


@dataclasses.dataclass(frozen=True)
class _CompiledRelayModel:
    backend: object
    backend_construction_failed: bool


@dataclasses.dataclass(frozen=True)
class _DetailedEvidence:
    """What one decode_detailed answer says beside its correction."""

    decoded_detectors: tuple
    iterations: int
    iteration_limit: int
    posterior_log_likelihood_ratios: tuple
    succeeded: bool


class _WrongCorrectionArityError(ValueError):
    pass


class _NonbinaryCorrectionError(ValueError):
    pass


def _finite_real(value, name: str, *, allow_none: bool = False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _nonnegative_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


def _validated_seed(value, name: str):
    if value is None:
        return None
    if type(value) is not int or not 0 <= value < _SEED_LIMIT:
        raise TypeError(f"{name} must be an unsigned 64-bit integer or None")
    return value


def _relay_profile(
    alpha,
    alpha_iteration_scaling_factor,
    gamma0,
    pre_iterations,
    relay_set_count,
    iterations_per_set,
    gamma_interval,
    converged_solution_count,
) -> _RelayProfile:
    """The profile with every argument checked once."""
    if type(gamma_interval) is not tuple or len(gamma_interval) != 2:
        raise TypeError("gamma_interval must be an exact pair")
    gamma_low = _finite_real(gamma_interval[0], "gamma_interval[0]")
    gamma_high = _finite_real(gamma_interval[1], "gamma_interval[1]")
    if gamma_low > gamma_high:
        raise ValueError("gamma_interval must be ordered low to high")
    converged_solution_count = _nonnegative_integer(
        converged_solution_count, "converged_solution_count"
    )
    if converged_solution_count == 0:
        raise ValueError("converged_solution_count must be positive")
    pre_iterations = _nonnegative_integer(pre_iterations, "pre_iterations")
    if pre_iterations == 0:
        # relay-bp runs its first leg for pre_iter iterations and keeps the
        # previous call's decoding when that loop never runs
        # (relay.rs decode_inner), so a zero first leg returns stale state
        raise ValueError("pre_iterations must be positive")
    alpha = _finite_real(alpha, "alpha", allow_none=True)
    scaling = _finite_real(
        alpha_iteration_scaling_factor, "alpha_iteration_scaling_factor"
    )
    gamma0 = _finite_real(gamma0, "gamma0", allow_none=True)
    relay_set_count = _nonnegative_integer(relay_set_count, "relay_set_count")
    iterations_per_set = _nonnegative_integer(
        iterations_per_set, "iterations_per_set"
    )
    return _RelayProfile(
        alpha=alpha,
        alpha_iteration_scaling_factor=scaling,
        gamma0=gamma0,
        pre_iterations=pre_iterations,
        relay_set_count=relay_set_count,
        iterations_per_set=iterations_per_set,
        gamma_interval=(gamma_low, gamma_high),
        converged_solution_count=converged_solution_count,
    )


def _load_relay_decoder_type():
    try:
        from relay_bp import RelayDecoderF32
    except ImportError as error:
        raise ImportError(
            "Relay-BP decoding requires the optional official "
            "dependency `relay-bp`; install that package before selecting "
            "RelayBeliefPropagationWindowDecoder"
        ) from error
    return RelayDecoderF32


def _gamma_table(profile: _RelayProfile, seed: int, column_count: int):
    """The fixed gamma table of one model, drawn from the seed."""
    bit_generator = numpy.random.PCG64(seed)
    generator = numpy.random.Generator(bit_generator)
    gamma_low, gamma_high = profile.gamma_interval
    shape = (profile.relay_set_count, column_count)
    gamma_table = generator.uniform(gamma_low, gamma_high, size=shape)
    gamma_table = gamma_table.astype(numpy.float64, copy=False)
    return numpy.ascontiguousarray(gamma_table)


def _construct_backend(
    profile: _RelayProfile, check, priors, gamma_table, seed: int
):
    """The backend decoder, or None when the backend refused the model."""
    decoder_type = _load_relay_decoder_type()
    sparse_check = scipy.sparse.csr_matrix(check)
    try:
        return decoder_type(
            sparse_check,
            priors,
            alpha=profile.alpha,
            alpha_iteration_scaling_factor=(
                profile.alpha_iteration_scaling_factor
            ),
            gamma0=profile.gamma0,
            pre_iter=profile.pre_iterations,
            num_sets=profile.relay_set_count,
            set_max_iter=profile.iterations_per_set,
            gamma_dist_interval=profile.gamma_interval,
            explicit_gammas=gamma_table,
            stop_nconv=profile.converged_solution_count,
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
    if len(check.shape) != 2:
        raise ValueError("Relay check matrix must be two-dimensional")
    is_zero = check.data == 0
    is_one = check.data == 1
    is_bit = is_zero | is_one
    if not numpy.all(is_bit):
        raise ValueError("Relay check matrix must be binary")
    if priors.ndim != 1 or priors.shape[0] != check.shape[1]:
        raise ValueError("Relay priors must align with physical fault columns")
    for column_index, probability in enumerate(priors):
        if not math.isfinite(probability) or not 0 < probability <= 0.5:
            raise ValueError(
                "Relay prior at physical column "
                f"{column_index} must satisfy finite 0 < p <= 0.5"
            )
    return check, priors.astype(numpy.float64)


def _validated_syndrome(syndrome, detector_count: int):
    syndrome = numpy.asarray(syndrome)
    if syndrome.ndim != 1 or syndrome.shape[0] != detector_count:
        raise ValueError("Relay syndrome arity does not match detector rows")
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
    return decoder_module.bit_tuple(vector)


def _float_tuple(values) -> tuple:
    numbers = []
    for value in values:
        numbers.append(float(value))
    return tuple(numbers)


def _reconstruct(check, correction) -> tuple:
    correction_array = numpy.asarray(correction)
    parity = decoder_module.parity_product(check, correction_array)
    return decoder_module.bit_tuple(parity)


def _detailed_evidence(detailed, faults) -> _DetailedEvidence:
    """The answer's diagnostics, each checked for its arity."""
    decoded_detectors = _binary_vector(
        detailed.decoded_detectors, expected_size=faults.check.shape[0]
    )
    iterations = int(detailed.iterations)
    iteration_limit = int(detailed.max_iter)
    posterior = numpy.asarray(detailed.posterior_ratios)
    if posterior.ndim != 1 or posterior.shape[0] != faults.check.shape[1]:
        raise ValueError("Relay posterior ratios have the wrong arity")
    as_float = posterior.astype(float)
    is_nan = numpy.isnan(as_float)
    if numpy.any(is_nan):
        raise ValueError("Relay posterior ratios cannot contain NaN")
    succeeded = bool(detailed.success)
    posterior_log_likelihood_ratios = _float_tuple(posterior)
    return _DetailedEvidence(
        decoded_detectors=decoded_detectors,
        iterations=iterations,
        iteration_limit=iteration_limit,
        posterior_log_likelihood_ratios=posterior_log_likelihood_ratios,
        succeeded=succeeded,
    )


def _outcome_of(
    faults, syndrome, correction: tuple, evidence: _DetailedEvidence
) -> backend_outcome.BackendDecodeOutcome:
    """The outcome of one answer: inconsistent, nonconverged or succeeded."""
    reconstructed = _reconstruct(faults.check, correction)
    syndrome_bits = decoder_module.bit_tuple(syndrome)
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
        component_correction=None,
        reconstructed_syndrome=evidence.decoded_detectors,
        iterations=evidence.iterations,
        iteration_limit=evidence.iteration_limit,
        posterior_log_likelihood_ratios=(
            evidence.posterior_log_likelihood_ratios
        ),
    )


def _invalid_outcome(reason) -> backend_outcome.BackendDecodeOutcome:
    return backend_outcome.BackendDecodeOutcome(
        status=_Status.INVALID_CORRECTION,
        failure_reason=reason,
        physical_correction=None,
        component_correction=None,
        reconstructed_syndrome=None,
        iterations=None,
        iteration_limit=None,
        posterior_log_likelihood_ratios=None,
    )


def _backend_error_outcome() -> backend_outcome.BackendDecodeOutcome:
    return backend_outcome.BackendDecodeOutcome(
        status=_Status.BACKEND_ERROR,
        failure_reason=_Reason.UPSTREAM_EXCEPTION,
        physical_correction=None,
        component_correction=None,
        reconstructed_syndrome=None,
        iterations=None,
        iteration_limit=None,
        posterior_log_likelihood_ratios=None,
    )
