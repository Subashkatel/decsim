from __future__ import annotations

from typing import TYPE_CHECKING

from ..message import DecodeResult
from .window_decoder import bposd_window_decoder

if TYPE_CHECKING:
    from ..message import DecodeJob
    from ..protocols import Decoder

# =====================================================================================
# BP-OSD RUNTIME DECODER
# =====================================================================================

class BPOSDDecoder:
    """Runtime DES BP-OSD decoder (Roffe's `ldpc`), the BP-OSD analogue of PyMatchingDecoder.
    Real decoding for correctness; latency from a wrapped latency model.

    Decodes job.dem.check with BP-OSD (delegated to the bposd_window_decoder inner callable),
    keeps the OWNED columns, and turns them into DecodeResult.logical_value +
    boundary_defects EXACTLY as PyMatchingDecoder / BeliefMatchingDecoder do. Unlike belief-
    matching it needs no hyperedge fields -- BP-OSD runs straight on the window's check matrix
    -- so it drops into the cluster/scheme machinery with zero extra wiring: pick any windowing
    scheme, route to this decoder.

    Scope: single-patch ops, one logical observable per op (logical_value reads observable 0),
    same as the other runtime decoders. CAVEAT: BP-OSD's natural target is qLDPC / BB codes,
    whose DEM is non-graphlike (decompose_errors=False) and (for QUITS-built circuits)
    coordinate-less -- but the cluster's engine-side DEM build is surface/graphlike +
    coordinate-based today, so BB codes do not yet flow through the full DES (a pre-existing
    cluster limitation, not this decoder's). This decoder runs end-to-end under the engine for
    the surface-code DEMs the cluster builds; for full-DEM qLDPC BP-OSD use the OFFLINE path
    build_window_error_models(decompose_errors=False) + bposd_window_decoder (+ decode_windowed)."""

    def __init__(self, latency_model: "Decoder", max_iter: int = 2, osd_order: int = 0,
                 bp_method: str = "product_sum", schedule: str = "serial",
                 osd_method: str = "osd_cs"):
        """Reuse a latency model for timing; the BP-OSD inner callable does the real work."""
        self.latency_model = latency_model
        self._inner = bposd_window_decoder(max_iter=max_iter, osd_order=osd_order,
                                           bp_method=bp_method, schedule=schedule,
                                           osd_method=osd_method)

    def latency(self, job: "DecodeJob") -> int:
        """Timing comes from the wrapped latency model (decode WORK is done in decode())."""
        return self.latency_model.latency(job)

    def decode(self, job: "DecodeJob") -> DecodeResult:
        """Run real windowed BP-OSD on the job's window model."""
        import numpy as np
        model = job.dem
        if model is None:                        # timing-only job: no real data to decode
            return DecodeResult(job.op_id, job.window_id)
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
