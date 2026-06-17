from __future__ import annotations

from typing import TYPE_CHECKING

from ..message import DecodeResult
from .window_decoder import belief_matching_window_decoder

if TYPE_CHECKING:
    from ..message import DecodeJob
    from ..protocols import Decoder

# =====================================================================================
# BELIEF-MATCHING RUNTIME DECODER
# =====================================================================================

class BeliefMatchingDecoder:
    """Runtime DES belief-matching decoder -- the belief-matching analogue of
    PyMatchingDecoder, and the *strong decoder* of the decoder-switching paper
    (arXiv:2510.25222). Real decoding for correctness; latency from a wrapped latency model.

    Decodes job.dem (a WindowErrorModel that must carry the hyperedge BP matrices, i.e. built
    with belief_matching=True) by running BP on the undecomposed hypergraph then reweighted
    MWPM -- delegated to the validated belief_matching_window_decoder inner callable -- then
    keeps only the OWNED columns (this window's commit), turns their observable flips into
    DecodeResult.logical_value and their beyond-commit detector flips into
    DecodeResult.boundary_defects (artificial defects, arXiv:2209.08552 Sec I.B), EXACTLY as
    PyMatchingDecoder does. So it drops into the same cluster/scheme machinery: pick any
    windowing scheme, route to this decoder, done.

    `needs_hyperedges = True` is the seam the cluster reads to build hyperedge DEMs (otherwise
    job.dem has no h_check and this decoder cannot run).

    Scope, same as PyMatchingDecoder: single-patch ops, one logical observable per op. Real
    DEMs reach this decoder under sliding/naive schemes; the parallel A/B scheme's leading-
    buffer windows are still timing-only at runtime FOR EVERY decoder
    (cluster._build_window_error_models) -- a pre-existing limit, not belief-matching-specific."""

    needs_hyperedges = True

    def __init__(self, latency_model: "Decoder", max_iter: int = 30,
                 bp_method: str = "product_sum"):
        """Reuse a latency model for timing; the BP+MWPM inner callable does the real work."""
        self.latency_model = latency_model
        self._inner = belief_matching_window_decoder(max_iter=max_iter, bp_method=bp_method)

    def latency(self, job: "DecodeJob") -> int:
        """Timing comes from the wrapped latency model (decode WORK is done in decode())."""
        return self.latency_model.latency(job)

    def decode(self, job: "DecodeJob") -> DecodeResult:
        """Run real windowed belief-matching on the job's hyperedge-bearing window model."""
        import numpy as np
        model = job.dem
        if model is None:                        # timing-only job: no real data to decode
            return DecodeResult(job.op_id, job.window_id)
        if model.h_check is None:
            raise ValueError(
                f"{job.label}: BeliefMatchingDecoder needs hyperedge DEMs -- the cluster must "
                "build window models with belief_matching=True (it does so automatically when a "
                "routed decoder exposes needs_hyperedges=True)")
        syndrome = np.concatenate(
            [np.asarray(p.bits, dtype=np.uint8) for p in job.payloads
             if p.bits is not None]) if job.payloads else np.zeros(0, dtype=np.uint8)
        if syndrome.size != model.check.shape[0]:
            raise ValueError(
                f"{job.label}: payload bits ({syndrome.size}) do not match the window "
                f"error model's detectors ({model.check.shape[0]})")
        selected = np.asarray(self._inner(model, syndrome), dtype=np.uint8)
        committed = selected.astype(bool) & model.owned
        obs_flips = (model.obs @ committed.astype(np.uint8)) % 2
        defects: dict = {}
        for col in np.nonzero(committed)[0]:
            for det in model.future_flips.get(int(col), ()):
                r, pos = model.defect_positions[det]
                mask = defects.setdefault(r, [])
                if len(mask) <= pos:
                    mask.extend([0] * (pos + 1 - len(mask)))
                mask[pos] ^= 1
        return DecodeResult(job.op_id, job.window_id,
                            correction=committed.astype(np.uint8),
                            logical_value=int(obs_flips[0]) if obs_flips.size else 0,
                            boundary_defects=defects or None)
