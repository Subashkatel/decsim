"""Decoder wrapper that attaches a soft-output confidence ``g`` to each committed window."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..adapters.window_decode_results import payload_syndrome
from ..message import (
    DecodeJob,
    DecodeResult,
    RunSeedChild,
    RunSeedPathSegment,
)

if TYPE_CHECKING:
    from ..protocols import Decoder


class SoftOutputDecoder:
    """Wrap a base decoder and attach a soft output ``g`` for observable-bearing windows only."""

    def __init__(self, base: "Decoder", metric_cls):
        self.base = base
        self.metric_cls = metric_cls
        self._metrics: dict = {}

    def run_seed_children(self):
        """Expose the base decoder and confidence metric constructor."""
        return (
            RunSeedChild(
                (RunSeedPathSegment("field", "base"),),
                self.base,
            ),
            RunSeedChild(
                (RunSeedPathSegment("field", "metric_cls"),),
                self.metric_cls,
            ),
        )

    def latency(self, job: DecodeJob) -> int:
        """Timing is the base decoder's; the soft output adds no modelled latency."""
        return self.base.latency(job)

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Run the base decode, then attach the soft output when available."""
        result = self.base.decode(job)
        metric = self._metric_for(job.dem)
        if metric is not None:
            result.soft_output = metric.evaluate(payload_syndrome(job)).gap
        return result

    def _metric_for(self, model):
        """Return a cached metric for this window model, or None without observable."""
        import weakref

        import numpy as np

        if model is None or getattr(model, "obs", None) is None:
            return None
        obs = np.asarray(model.obs)
        if obs.shape[0] != 1 or not obs.any():
            return None
        entry = self._metrics.get(id(model))
        metric = entry[1] if entry is not None and entry[0]() is model else None
        if metric is None:
            metric = self.metric_cls.from_window_model(model)
            self._metrics[id(model)] = (weakref.ref(model), metric)
        return metric
