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
    """Decode one hyperedge-bearing window with belief matching; simulated
    latency comes from a latency model, or, with ``latency_model=None``, from
    the measured wall clock of the BP-plus-matching call itself (software
    decoder on this host, the PyMatchingDecoder pattern)."""

    fault_model_requirement = LINKED_FAULT_MODELS_REQUIRED

    def __init__(self, latency_model: "Optional[Decoder]" = None,
                 max_iter: int = 30, bp_method: str = "product_sum"):
        self.latency_model = latency_model
        self.measures_wall_clock = latency_model is None
        self.last_decode_ns = None
        self._warmed_models: set = set()
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
        if self.measures_wall_clock:
            raise RuntimeError("measured wall-clock timing needs the DecoderEngine: "
                               "it decodes first and charges the measured time")
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
        import time
        if self.measures_wall_clock and id(model) not in self._warmed_models:
            # measured mode times the BP-plus-matching call only, never the
            # one-time window-model construction (validation, BP build), the
            # same contract as PyMatchingDecoder's warm-up
            import numpy as np
            self._inner(model, np.zeros_like(np.asarray(syndrome)))
            self._warmed_models.add(id(model))
        started_ns = time.perf_counter_ns()
        try:
            selected = self._inner(model, syndrome)
            self.last_decode_ns = time.perf_counter_ns() - started_ns
        except ValueError as error:
            self.last_decode_ns = time.perf_counter_ns() - started_ns
            # PyMatching: odd parity in a boundaryless component, see mwpm
            if "perfect matching" not in str(error):
                raise
            import numpy as np
            selected = np.zeros(faults.check.shape[1], dtype=np.uint8)
            decode_status = BackendDecodeStatus.INVALID_CORRECTION
        return result_from_selected_faults(job, model, faults, selected,
                                           decode_status=decode_status)
