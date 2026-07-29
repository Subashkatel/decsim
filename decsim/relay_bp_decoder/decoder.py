"""Runtime Relay-BP decoder adapter."""

from __future__ import annotations

from typing import Optional

from ..adapters.window_decode_results import (
    DecoderAttemptFailed,
    check_syndrome_size,
    payload_syndrome,
    result_from_selected_faults,
    validate_backend_outcome,
)
from ..detector_error_model import (
    FaultRepresentation,
    PHYSICAL_FAULT_MODEL_REQUIRED,
)
from ..message import (
    DecodeJob,
    DecodeResult,
    RunSeedChild,
    RunSeedPathSegment,
)
from .window_decoder import RelayBpWindowDecoder


class RelayBpDecoder:
    """Use Relay-BP for corrections and an injected model for service time."""

    fault_model_requirement = PHYSICAL_FAULT_MODEL_REQUIRED

    def __init__(
        self,
        latency_model,
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
        self.latency_model = latency_model
        self.window_decoder = RelayBpWindowDecoder(
            alpha=alpha,
            alpha_iteration_scaling_factor=
                alpha_iteration_scaling_factor,
            gamma0=gamma0,
            pre_iterations=pre_iterations,
            relay_set_count=relay_set_count,
            iterations_per_set=iterations_per_set,
            gamma_interval=gamma_interval,
            converged_solution_count=converged_solution_count,
            gamma_table_seed=gamma_table_seed,
        )

    def run_seed_children(self):
        """Expose timing and fixed-gamma owners at stable semantic paths."""
        return (
            RunSeedChild(
                (RunSeedPathSegment("field", "latency_model"),),
                self.latency_model,
            ),
            RunSeedChild(
                (RunSeedPathSegment("field", "window_decoder"),),
                self.window_decoder,
            ),
        )

    def latency(self, job: DecodeJob) -> int:
        """Simulation time comes only from the injected latency model."""
        return self.latency_model.latency(job)

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Decode and commit only a successful same-model physical outcome."""
        model = job.dem
        if model is None:
            return DecodeResult(job.op_id, job.window_id)
        faults = model.require_faults(FaultRepresentation.PHYSICAL)
        syndrome = payload_syndrome(job)
        check_syndrome_size(job, syndrome, faults)
        outcome = self.window_decoder.decode(model, syndrome)
        validate_backend_outcome(outcome, model, faults, syndrome)
        if not outcome.succeeded:
            raise DecoderAttemptFailed(job, outcome)
        return result_from_selected_faults(
            job,
            model,
            faults,
            outcome.physical_correction,
        )
