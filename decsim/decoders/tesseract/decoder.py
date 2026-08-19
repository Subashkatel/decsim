"""Runtime adapter for the official Tesseract decoder."""

from __future__ import annotations

from typing import Optional

from ..window_decode_results import (
    DecoderAttemptFailed,
    check_syndrome_size,
    payload_syndrome,
    result_from_selected_faults,
    validate_backend_outcome,
)
from ...detector_error_model.fault_model_contracts import (
    FaultRepresentation,
    PHYSICAL_FAULT_MODEL_REQUIRED,
)
from ...message import (
    DecodeJob,
    DecodeResult,
    RunSeedChild,
    RunSeedPathSegment,
)
from .window_decoder import (
    TesseractDecoderConfig,
    TesseractWindowDecoder,
)


class TesseractDecoder:
    """Decode physical fault mechanisms with injected simulated latency."""

    fault_model_requirement = PHYSICAL_FAULT_MODEL_REQUIRED

    def __init__(
        self,
        latency_model,
        configuration: Optional[TesseractDecoderConfig] = None,
    ) -> None:
        self.latency_model = latency_model
        self.window_decoder = TesseractWindowDecoder(configuration)

    def run_seed_children(self):
        """Expose timing and detector-order seed owners by semantic role."""
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
        """Return only the configured simulated service time."""
        return self.latency_model.latency(job)

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Commit a same-model correction only after backend validation."""
        model = job.dem
        if model is None:
            return DecodeResult(job.op_id, job.window_id)
        physical_faults = model.require_faults(
            FaultRepresentation.PHYSICAL
        )
        syndrome = payload_syndrome(job)
        check_syndrome_size(job, syndrome, physical_faults)
        outcome = self.window_decoder.decode(model, syndrome)
        validate_backend_outcome(
            outcome,
            model,
            physical_faults,
            syndrome,
        )
        if not outcome.succeeded:
            raise DecoderAttemptFailed(job, outcome)
        return result_from_selected_faults(
            job,
            model,
            physical_faults,
            outcome.physical_correction,
        )
