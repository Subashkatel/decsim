"""The Relay-BP adapter: corrections from relay-bp, time from a model."""

from typing import Optional

import decsim.decoders.decoder as decoder_module
import decsim.decoders.relay_belief_propagation.window_decoder as window_decoder
import decsim.decoders.window_decode_results as window_decode_results
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.message as message


class RelayBeliefPropagationDecoder(decoder_module.WindowDecoderBase):
    """Use Relay-BP for corrections and an injected model for service time."""

    fault_model_requirement = fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    fault_representation = fault_models.FaultRepresentation.PHYSICAL

    def __init__(
        self,
        latency_model: Optional[decoder_module.DecoderBase] = None,
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
        decoder_module.WindowDecoderBase.__init__(self, latency_model)
        self.window_decoder = (
            window_decoder.RelayBeliefPropagationWindowDecoder(
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
        )

    def run_seed_children(self) -> tuple:
        """The timing and fixed-gamma owners at stable semantic paths."""
        latency_path = (message.RunSeedPathSegment("field", "latency_model"),)
        decoder_path = (message.RunSeedPathSegment("field", "window_decoder"),)
        return (
            message.RunSeedChild(latency_path, self.latency_model),
            message.RunSeedChild(decoder_path, self.window_decoder),
        )

    def compile(self, faults, model):
        """The window decoder, which compiles the backend per model itself."""
        del faults
        del model
        return self.window_decoder

    def decode_window(self, backend, model, faults, syndrome) -> tuple:
        """One backend call; a produced correction is committed as it stands."""
        del faults
        outcome = backend.decode(model, syndrome)
        return window_decode_results.selected_faults_of(outcome)
