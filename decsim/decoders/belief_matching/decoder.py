"""Runtime belief-matching decoder adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..window_decode_results import (
    BackendDecodeStatus,
    check_syndrome_size,
    payload_syndrome,
    result_from_selected_faults,
)
from ...message import DecodeResult, RunSeedChild, RunSeedPathSegment
from ...detector_error_model.fault_model_contracts import (
    FaultRepresentation,
    LINKED_FAULT_MODELS_REQUIRED,
)
from .window_decoder import belief_matching_window_decoder

if TYPE_CHECKING:
    from ...message import DecodeJob
    from ...protocols import Decoder


class BeliefMatchingDecoder:
    """Decode one hyperedge-bearing window with belief matching."""

    fault_model_requirement = LINKED_FAULT_MODELS_REQUIRED

    def __init__(self, latency_model: "Decoder", max_iter: int = 30,
                 bp_method: str = "product_sum"):
        self.latency_model = latency_model
        self.max_iter = max_iter
        self.bp_method = bp_method
        self._inner = belief_matching_window_decoder(max_iter=max_iter, bp_method=bp_method)

    def run_seed_children(self):
        """Expose the latency model that controls simulated service time."""
        return (
            RunSeedChild(
                (RunSeedPathSegment("field", "latency_model"),),
                self.latency_model,
            ),
        )

    def latency(self, job: "DecodeJob") -> int:
        """Timing comes from the wrapped latency model."""
        return self.latency_model.latency(job)

    def decode(self, job: "DecodeJob") -> DecodeResult:
        """Run real windowed belief-matching on the job's hyperedge-bearing window model."""
        model = job.dem
        if model is None:
            return DecodeResult(job.op_id, job.window_id)
        faults = model.require_faults(FaultRepresentation.GRAPHLIKE)
        model.require_faults(FaultRepresentation.PHYSICAL)
        if model.physical_to_graphlike_detector_projection is None:
            raise ValueError(
                f"{job.label}: BeliefMatchingDecoder needs the physical-to-"
                "graphlike link"
            )
        syndrome = payload_syndrome(job)
        check_syndrome_size(job, syndrome, faults)
        decode_status = None
        try:
            selected = self._inner(model, syndrome)
        except ValueError as error:
            # PyMatching: odd parity in a boundaryless component, see mwpm
            if "perfect matching" not in str(error):
                raise
            import numpy as np
            selected = np.zeros(faults.check.shape[1], dtype=np.uint8)
            decode_status = BackendDecodeStatus.INVALID_CORRECTION
        return result_from_selected_faults(job, model, faults, selected,
                                           decode_status=decode_status)
