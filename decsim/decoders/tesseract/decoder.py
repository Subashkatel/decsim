"""The Tesseract adapter: the referee's decoder as a tier of its own.

Physical fault mechanisms decoded by the official backend; the window
decoder module is the backend this row compiles.
"""

from typing import Optional

import decsim.decoders.backend_outcome as backend_outcome
import decsim.decoders.decoder as decoder_module
import decsim.decoders.tesseract.window_decoder as window_decoder
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.message as message


class TesseractDecoder(decoder_module.WindowDecoderBase):
    """Decode physical fault mechanisms with the official Tesseract backend."""

    fault_model_requirement = fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    fault_representation = fault_models.FaultRepresentation.PHYSICAL

    def __init__(
        self,
        latency_model: Optional[decoder_module.DecoderBase] = None,
        configuration: Optional[window_decoder.TesseractDecoderConfig] = None,
    ) -> None:
        decoder_module.WindowDecoderBase.__init__(self, latency_model)
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

    def compile(self, faults, model):
        """The window decoder, which compiles the backend per model itself."""
        del faults
        del model
        return self.window_decoder

    def decode_window(self, backend, model, faults, syndrome) -> tuple:
        """One backend call; a produced correction is committed as it stands."""
        del faults
        outcome = backend.decode(model, syndrome)
        return backend_outcome.selected_faults_of(outcome)
