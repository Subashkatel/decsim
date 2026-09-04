"""The Tesseract adapter: the referee's decoder as a tier of its own.

Physical fault mechanisms decoded by the official backend, priced by an
injected latency model.
"""

from typing import Optional

import decsim.decoders.tesseract.window_decoder as window_decoder
import decsim.decoders.window_decode_results as window_decode_results
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.message as message


class TesseractDecoder:
    """Decode physical fault mechanisms with injected simulated latency."""

    fault_model_requirement = fault_models.PHYSICAL_FAULT_MODEL_REQUIRED

    def __init__(
        self,
        latency_model,
        configuration: Optional[window_decoder.TesseractDecoderConfig] = None,
    ) -> None:
        self.latency_model = latency_model
        self.window_decoder = window_decoder.TesseractWindowDecoder(
            configuration
        )

    def run_seed_children(self) -> tuple:
        """The timing and detector-order seed owners by semantic role."""
        latency_path = (message.RunSeedPathSegment("field", "latency_model"),)
        decoder_path = (message.RunSeedPathSegment("field", "window_decoder"),)
        return (
            message.RunSeedChild(latency_path, self.latency_model),
            message.RunSeedChild(decoder_path, self.window_decoder),
        )

    def latency(self, job: message.DecodeJob) -> int:
        """Only the configured simulated service time."""
        return self.latency_model.latency(job)

    def decode(self, job: message.DecodeJob) -> message.DecodeResult:
        """The backend's correction, validated, committed best effort or not."""
        model = job.dem
        if model is None:
            return message.DecodeResult(job.op_id, job.window_id)
        physical_faults = model.require_faults(
            fault_models.FaultRepresentation.PHYSICAL
        )
        syndrome = window_decode_results.payload_syndrome(job)
        window_decode_results.check_syndrome_size(
            job, syndrome, physical_faults
        )
        outcome = self.window_decoder.decode(model, syndrome)
        window_decode_results.validate_backend_outcome(
            outcome, model, physical_faults, syndrome
        )
        return window_decode_results.result_from_backend_outcome(
            job, model, physical_faults, outcome
        )
