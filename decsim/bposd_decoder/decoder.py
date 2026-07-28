"""Runtime BP-OSD decoder adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..adapters.window_decode_results import (
    check_syndrome_size,
    payload_syndrome,
    result_from_selected_faults,
)
from ..message import DecodeResult, RunSeedChild, RunSeedPathSegment
from .window_decoder import bposd_window_decoder

if TYPE_CHECKING:
    from ..message import DecodeJob
    from ..protocols import Decoder


class BPOSDDecoder:
    """Decode one window with BP-OSD and report simulated latency separately."""

    def __init__(self, latency_model: "Decoder", max_iter: int = 2, osd_order: int = 0,
                 bp_method: str = "product_sum", schedule: str = "serial",
                 osd_method: str = "osd_cs"):
        self.latency_model = latency_model
        self.max_iter = max_iter
        self.osd_order = osd_order
        self.bp_method = bp_method
        self.schedule = schedule
        self.osd_method = osd_method
        self._inner = bposd_window_decoder(max_iter=max_iter, osd_order=osd_order,
                                           bp_method=bp_method, schedule=schedule,
                                           osd_method=osd_method)

    def run_manifest_config(self):
        return {
            "kind": "bposd",
            "max_iter": self.max_iter,
            "osd_order": self.osd_order,
            "bp_method": self.bp_method,
            "schedule": self.schedule,
            "osd_method": self.osd_method,
        }

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
        """Run real windowed BP-OSD on the job's window model."""
        model = job.dem
        if model is None:
            return DecodeResult(job.op_id, job.window_id)
        syndrome = payload_syndrome(job)
        check_syndrome_size(job, syndrome, model)
        selected = self._inner(model, syndrome)
        return result_from_selected_faults(job, model, selected)
