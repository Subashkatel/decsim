"""The Relay-BP adapter: corrections from relay-bp, time from a model."""

from typing import Optional

import decsim.decoders.relay_bp.window_decoder as window_decoder
import decsim.decoders.window_decode_results as window_decode_results
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.message as message


class RelayBpDecoder:
    """Use Relay-BP for corrections and an injected model for service time."""

    fault_model_requirement = fault_models.PHYSICAL_FAULT_MODEL_REQUIRED

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
        self.window_decoder = window_decoder.RelayBpWindowDecoder(
            alpha=alpha,
            alpha_iteration_scaling_factor=alpha_iteration_scaling_factor,
            gamma0=gamma0,
            pre_iterations=pre_iterations,
            relay_set_count=relay_set_count,
            iterations_per_set=iterations_per_set,
            gamma_interval=gamma_interval,
            converged_solution_count=converged_solution_count,
            gamma_table_seed=gamma_table_seed,
        )

    def run_seed_children(self) -> tuple:
        """The timing and fixed-gamma owners at stable semantic paths."""
        latency_path = (message.RunSeedPathSegment("field", "latency_model"),)
        decoder_path = (message.RunSeedPathSegment("field", "window_decoder"),)
        return (
            message.RunSeedChild(latency_path, self.latency_model),
            message.RunSeedChild(decoder_path, self.window_decoder),
        )

    def latency(self, job: message.DecodeJob) -> int:
        """Simulation time comes only from the injected latency model."""
        return self.latency_model.latency(job)

    def decode(self, job: message.DecodeJob) -> message.DecodeResult:
        """The same-model physical outcome, committed best effort or not."""
        model = job.dem
        if model is None:
            return message.DecodeResult(job.op_id, job.window_id)
        faults = model.require_faults(fault_models.FaultRepresentation.PHYSICAL)
        syndrome = window_decode_results.payload_syndrome(job)
        window_decode_results.check_syndrome_size(job, syndrome, faults)
        outcome = self.window_decoder.decode(model, syndrome)
        window_decode_results.validate_backend_outcome(
            outcome, model, faults, syndrome
        )
        return window_decode_results.result_from_backend_outcome(
            job, model, faults, outcome
        )
