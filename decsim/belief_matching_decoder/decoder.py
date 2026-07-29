"""Runtime belief-matching decoder adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..adapters.window_decode_results import (
    check_syndrome_size,
    payload_syndrome,
    result_from_selected_faults,
)
from ..message import DecodeResult, RunSeedChild, RunSeedPathSegment
from .window_decoder import belief_matching_window_decoder

if TYPE_CHECKING:
    from ..message import DecodeJob
    from ..protocols import Decoder


class BeliefMatchingDecoder:
    """Decode one hyperedge-bearing window with belief matching."""

    needs_hyperedges = True

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
        if model.h_check is None:
            raise ValueError(
                f"{job.label}: BeliefMatchingDecoder needs hyperedge DEMs. The cluster must "
                "build window models with belief_matching=True (it does so automatically when a "
                "routed decoder exposes needs_hyperedges=True)")
        syndrome = payload_syndrome(job)
        check_syndrome_size(job, syndrome, model)
        selected = self._inner(model, syndrome)
        return result_from_selected_faults(job, model, selected)
