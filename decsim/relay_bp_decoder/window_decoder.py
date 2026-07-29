"""Relay-BP adapter for one placed physical window model."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Integral, Real
import os
import secrets
import threading
from typing import Optional
import weakref

from ..adapters.window_decode_results import (
    BackendDecodeOutcome,
    BackendDecodeStatus,
    BackendFailureReason,
    decoder_configuration_fingerprint,
    empty_fault_model_outcome,
    fault_model_fingerprint,
)
from ..detector_error_model import FaultRepresentation
from ..message import RunSeedReservation


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


@dataclass(frozen=True)
class _RelayProfile:
    alpha: Optional[float]
    alpha_iteration_scaling_factor: float
    gamma0: Optional[float]
    pre_iterations: int
    relay_set_count: int
    iterations_per_set: int
    gamma_interval: tuple[float, float]
    converged_solution_count: int


@dataclass(frozen=True)
class _CompiledRelayModel:
    backend: object
    backend_construction_failed: bool
    fault_model_fingerprint: str
    configuration_fingerprint: str


class RelayBpWindowDecoder:
    """Decode physical fault columns with one fixed-gamma Relay-BP profile.

    SCOPE:
    - Inputs are original physical fault columns, not graph decompositions.
    - The official ``decode_detailed`` API is called once per syndrome.
    - Gamma tables are fixed per live model; native per-shot resampling is not
      exposed.
    - Backend wall time is diagnostic only and never becomes simulated time.

    The paper assumes ``0 < p < 1/2``. This adapter also accepts exactly
    ``p=1/2`` as a tested software-profile extension with a zero prior log
    ratio. Decided, majority-one, and non-finite priors are rejected instead
    of being silently transformed. The algorithm follows Relay-BP-S from
    Müller et al., arXiv:2506.01779v2, Algorithm 1.
    """

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
        if type(gamma_interval) is not tuple or len(gamma_interval) != 2:
            raise TypeError("gamma_interval must be an exact pair")
        gamma_low = _finite_real(gamma_interval[0], "gamma_interval[0]")
        gamma_high = _finite_real(gamma_interval[1], "gamma_interval[1]")
        if gamma_low > gamma_high:
            raise ValueError("gamma_interval must be ordered low to high")
        converged_solution_count = _nonnegative_integer(
            converged_solution_count,
            "converged_solution_count",
        )
        if converged_solution_count == 0:
            raise ValueError("converged_solution_count must be positive")
        self._profile = _RelayProfile(
            alpha=_finite_real(alpha, "alpha", allow_none=True),
            alpha_iteration_scaling_factor=_finite_real(
                alpha_iteration_scaling_factor,
                "alpha_iteration_scaling_factor",
            ),
            gamma0=_finite_real(gamma0, "gamma0", allow_none=True),
            pre_iterations=_nonnegative_integer(
                pre_iterations,
                "pre_iterations",
            ),
            relay_set_count=_nonnegative_integer(
                relay_set_count,
                "relay_set_count",
            ),
            iterations_per_set=_nonnegative_integer(
                iterations_per_set,
                "iterations_per_set",
            ),
            gamma_interval=(gamma_low, gamma_high),
            converged_solution_count=converged_solution_count,
        )
        self._explicit_gamma_table_seed = self._validate_seed(
            gamma_table_seed,
            "gamma_table_seed",
        )
        self._effective_gamma_table_seed = self._explicit_gamma_table_seed
        self._seed_lock = threading.Lock()
        self._pending_run_seed = None
        self._run_seed_claimed = False
        self._stochastic_use_started = False
        self._thread_state = threading.local()

    @staticmethod
    def _validate_seed(value, name: str):
        if value is None:
            return None
        if type(value) is not int or not 0 <= value < 2**64:
            raise TypeError(f"{name} must be an unsigned 64-bit integer or None")
        return value

    def reserve_run_seed(self, seed: Optional[int]) -> RunSeedReservation:
        """Prepare one semantic run-root-derived gamma-table seed."""
        seed = self._validate_seed(seed, "RelayBpWindowDecoder run root")
        with self._seed_lock:
            if self._stochastic_use_started:
                raise ValueError(
                    "RelayBpWindowDecoder was already used and cannot be rebound"
                )
            if self._run_seed_claimed:
                raise ValueError(
                    "RelayBpWindowDecoder is already claimed by a built run"
                )
            if self._pending_run_seed is not None:
                raise ValueError(
                    "RelayBpWindowDecoder already has a pending run-seed reservation"
                )
            if seed is not None and self._explicit_gamma_table_seed is not None:
                raise ValueError(
                    "RelayBpWindowDecoder has an explicit gamma-table seed that "
                    "conflicts with the numeric run root"
                )
            if seed is not None:
                source = "derived"
                effective_seed = seed
                reported_seed = seed
            elif self._explicit_gamma_table_seed is not None:
                source = "explicit_local"
                effective_seed = self._explicit_gamma_table_seed
                reported_seed = effective_seed
            else:
                source = "entropy"
                effective_seed = secrets.randbits(64)
                reported_seed = None
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
                    "RelayBpWindowDecoder can commit only its exact pending "
                    "run-seed reservation"
                )
            self._effective_gamma_table_seed = reservation.prepared_state
            self._pending_run_seed = None
            self._run_seed_claimed = True

    def decode(self, window_model, syndrome) -> BackendDecodeOutcome:
        """Call the official detailed API once and snapshot its evidence."""
        import numpy as np

        faults = window_model.require_faults(FaultRepresentation.PHYSICAL)
        syndrome = self._validated_syndrome(syndrome, faults.check.shape[0])
        compiled = self._compiled_model(faults)
        if faults.check.shape[1] == 0:
            return empty_fault_model_outcome(
                faults,
                syndrome,
                decoder_configuration_fingerprint=
                    compiled.configuration_fingerprint,
            )
        if compiled.backend_construction_failed:
            return self._backend_error_outcome(compiled)
        try:
            detailed = compiled.backend.decode_detailed(syndrome)
            correction = self._binary_vector(
                detailed.decoding,
                expected_size=faults.check.shape[1],
            )
        except _WrongCorrectionArity:
            return self._invalid_outcome(
                compiled,
                BackendFailureReason.CORRECTION_WRONG_ARITY,
            )
        except _NonbinaryCorrection:
            return self._invalid_outcome(
                compiled,
                BackendFailureReason.CORRECTION_NOT_BINARY,
            )
        except Exception:
            return self._backend_error_outcome(compiled)
        try:
            decoded_detectors = self._binary_vector(
                detailed.decoded_detectors,
                expected_size=faults.check.shape[0],
            )
            iterations = int(detailed.iterations)
            iteration_limit = int(detailed.max_iter)
            posterior = np.asarray(detailed.posterior_ratios)
            if posterior.ndim != 1 or posterior.shape[0] != faults.check.shape[1]:
                raise ValueError("Relay posterior ratios have the wrong arity")
            if np.any(np.isnan(posterior.astype(float))):
                raise ValueError("Relay posterior ratios cannot contain NaN")
            succeeded = bool(detailed.success)
        except Exception:
            return self._backend_error_outcome(compiled)

        reconstructed = self._reconstruct(faults.check, correction)
        if decoded_detectors != reconstructed or (
            succeeded
            and reconstructed != tuple(int(bit) for bit in syndrome)
        ):
            return BackendDecodeOutcome(
                status=BackendDecodeStatus.INVALID_CORRECTION,
                failure_reason=
                    BackendFailureReason.CORRECTION_DOES_NOT_MATCH_SYNDROME,
                physical_correction=None,
                component_correction=None,
                reconstructed_syndrome=decoded_detectors,
                iterations=iterations,
                iteration_limit=iteration_limit,
                posterior_log_likelihood_ratios=tuple(
                    float(value) for value in posterior
                ),
                fault_model_fingerprint=compiled.fault_model_fingerprint,
                decoder_configuration_fingerprint=
                    compiled.configuration_fingerprint,
            )
        common = dict(
            physical_correction=correction,
            component_correction=None,
            reconstructed_syndrome=decoded_detectors,
            iterations=iterations,
            iteration_limit=iteration_limit,
            posterior_log_likelihood_ratios=tuple(
                float(value) for value in posterior
            ),
            fault_model_fingerprint=compiled.fault_model_fingerprint,
            decoder_configuration_fingerprint=
                compiled.configuration_fingerprint,
        )
        if not succeeded:
            return BackendDecodeOutcome(
                status=BackendDecodeStatus.NONCONVERGED,
                failure_reason=
                    BackendFailureReason.NO_CONVERGED_RELAY_SOLUTION,
                **common,
            )
        return BackendDecodeOutcome(
            status=BackendDecodeStatus.SUCCEEDED,
            failure_reason=None,
            **common,
        )

    def _compiled_model(self, faults) -> _CompiledRelayModel:
        cache = self._thread_cache()
        identity = id(faults)
        entry = cache.get(identity)
        if entry is not None and entry[0]() is faults:
            return entry[1]
        compiled = self._compile(faults)

        def discard(reference) -> None:
            current = cache.get(identity)
            if current is not None and current[0] is reference:
                del cache[identity]

        reference = weakref.ref(faults, discard)
        cache[identity] = (reference, compiled)
        return compiled

    def _compile(self, faults) -> _CompiledRelayModel:
        import numpy as np

        check, priors = self._validated_model(faults)
        seed = self._gamma_seed()
        generator = np.random.Generator(np.random.PCG64(seed))
        gamma_table = generator.uniform(
            self._profile.gamma_interval[0],
            self._profile.gamma_interval[1],
            size=(self._profile.relay_set_count, check.shape[1]),
        ).astype(np.float64, copy=False)
        gamma_table = np.ascontiguousarray(gamma_table)
        gamma_table_sha256 = hashlib.sha256(
            gamma_table.tobytes(order="C")
        ).hexdigest()
        configuration_fingerprint = decoder_configuration_fingerprint({
            "backend": "relay_bp.RelayDecoderF32",
            "profile": self._profile,
            "gamma_table_seed": seed,
            "gamma_table_shape": tuple(gamma_table.shape),
            "gamma_table_sha256": gamma_table_sha256,
            "stopping_criterion": "nconv",
        })
        model_fingerprint = fault_model_fingerprint(faults)
        if check.shape[1] == 0:
            return _CompiledRelayModel(
                backend=None,
                backend_construction_failed=False,
                fault_model_fingerprint=model_fingerprint,
                configuration_fingerprint=configuration_fingerprint,
            )
        decoder_type = self._load_relay_decoder_type()
        try:
            from scipy.sparse import csr_matrix

            backend = decoder_type(
                csr_matrix(check),
                priors,
                alpha=self._profile.alpha,
                alpha_iteration_scaling_factor=
                    self._profile.alpha_iteration_scaling_factor,
                gamma0=self._profile.gamma0,
                pre_iter=self._profile.pre_iterations,
                num_sets=self._profile.relay_set_count,
                set_max_iter=self._profile.iterations_per_set,
                gamma_dist_interval=self._profile.gamma_interval,
                explicit_gammas=gamma_table,
                stop_nconv=self._profile.converged_solution_count,
                stopping_criterion="nconv",
                logging=False,
                seed=seed,
            )
        except Exception:
            return _CompiledRelayModel(
                backend=None,
                backend_construction_failed=True,
                fault_model_fingerprint=model_fingerprint,
                configuration_fingerprint=configuration_fingerprint,
            )
        return _CompiledRelayModel(
            backend=backend,
            backend_construction_failed=False,
            fault_model_fingerprint=model_fingerprint,
            configuration_fingerprint=configuration_fingerprint,
        )

    @staticmethod
    def _load_relay_decoder_type():
        try:
            from relay_bp import RelayDecoderF32
        except ImportError as error:
            raise ImportError(
                "Relay-BP decoding requires the optional official "
                "dependency `relay-bp`; install that package before selecting "
                "RelayBpWindowDecoder"
            ) from error
        return RelayDecoderF32

    def _gamma_seed(self) -> int:
        with self._seed_lock:
            if self._pending_run_seed is not None:
                raise RuntimeError(
                    "RelayBpWindowDecoder cannot compile while a run-seed "
                    "reservation is pending"
                )
            self._stochastic_use_started = True
            if self._effective_gamma_table_seed is None:
                self._effective_gamma_table_seed = secrets.randbits(64)
            return self._effective_gamma_table_seed

    def _thread_cache(self) -> dict:
        process_id = os.getpid()
        if getattr(self._thread_state, "process_id", None) != process_id:
            self._thread_state.process_id = process_id
            self._thread_state.compiled_models = {}
        return self._thread_state.compiled_models

    @staticmethod
    def _validated_model(faults):
        import numpy as np

        check = np.asarray(faults.check)
        priors = np.asarray(faults.priors, dtype=float)
        if check.ndim != 2:
            raise ValueError("Relay check matrix must be two-dimensional")
        if not np.all((check == 0) | (check == 1)):
            raise ValueError("Relay check matrix must be binary")
        if priors.ndim != 1 or priors.shape[0] != check.shape[1]:
            raise ValueError("Relay priors must align with physical fault columns")
        for column_index, probability in enumerate(priors):
            if not math.isfinite(float(probability)) or not 0 < probability <= 0.5:
                raise ValueError(
                    "Relay prior at physical column "
                    f"{column_index} must satisfy finite 0 < p <= 0.5"
                )
        return np.asarray(check, dtype=np.uint8), priors.astype(np.float64)

    @staticmethod
    def _validated_syndrome(syndrome, detector_count: int):
        import numpy as np

        syndrome = np.asarray(syndrome)
        if syndrome.ndim != 1 or syndrome.shape[0] != detector_count:
            raise ValueError("Relay syndrome arity does not match detector rows")
        if not np.all((syndrome == 0) | (syndrome == 1)):
            raise ValueError("Relay syndrome must be binary")
        return syndrome.astype(np.uint8)

    @staticmethod
    def _binary_vector(value, *, expected_size: int) -> tuple[int, ...]:
        import numpy as np

        vector = np.asarray(value)
        if vector.ndim != 1 or vector.shape[0] != expected_size:
            raise _WrongCorrectionArity
        if not np.all((vector == 0) | (vector == 1)):
            raise _NonbinaryCorrection
        return tuple(int(bit) for bit in vector)

    @staticmethod
    def _reconstruct(check, correction) -> tuple[int, ...]:
        import numpy as np

        reconstructed = (
            np.asarray(check, dtype=np.uint64)
            @ np.asarray(correction, dtype=np.uint64)
        ) % 2
        return tuple(int(bit) for bit in reconstructed)

    @staticmethod
    def _invalid_outcome(compiled, reason) -> BackendDecodeOutcome:
        return BackendDecodeOutcome(
            status=BackendDecodeStatus.INVALID_CORRECTION,
            failure_reason=reason,
            physical_correction=None,
            component_correction=None,
            reconstructed_syndrome=None,
            iterations=None,
            iteration_limit=None,
            posterior_log_likelihood_ratios=None,
            fault_model_fingerprint=compiled.fault_model_fingerprint,
            decoder_configuration_fingerprint=
                compiled.configuration_fingerprint,
        )

    @staticmethod
    def _backend_error_outcome(compiled) -> BackendDecodeOutcome:
        return BackendDecodeOutcome(
            status=BackendDecodeStatus.BACKEND_ERROR,
            failure_reason=BackendFailureReason.UPSTREAM_EXCEPTION,
            physical_correction=None,
            component_correction=None,
            reconstructed_syndrome=None,
            iterations=None,
            iteration_limit=None,
            posterior_log_likelihood_ratios=None,
            fault_model_fingerprint=compiled.fault_model_fingerprint,
            decoder_configuration_fingerprint=
                compiled.configuration_fingerprint,
        )


class _WrongCorrectionArity(ValueError):
    pass


class _NonbinaryCorrection(ValueError):
    pass
