from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..adapters.window_error_models import WindowErrorModel


def bposd_window_decoder(max_iter: int = 2, osd_order: int = 0,
                         bp_method: str = "product_sum", schedule: str = "serial",
                         osd_method: str = "osd_cs"):
    """A BP-OSD inner decoder for ``decode_windowed`` -- BB / qLDPC windows, whose faults
    may flip > 2 detectors (build the models with decompose_errors=False; matching does not
    apply). Defaults follow QUITS's sliding_window_bposd_* functions. Caches one
    ``ldpc.BpOsdDecoder`` per WindowErrorModel (matrices are shot-independent; only the
    syndrome changes per shot)."""
    cache: dict = {}

    def decode(model: "WindowErrorModel", syndrome):
        d = cache.get(id(model))
        if d is None:
            from ldpc import BpOsdDecoder
            from scipy.sparse import csr_matrix
            d = BpOsdDecoder(csr_matrix(model.check),
                             error_channel=list(model.priors),
                             max_iter=max_iter, bp_method=bp_method,
                             schedule=schedule, osd_method=osd_method,
                             osd_order=osd_order)
            cache[id(model)] = d
        return d.decode(syndrome)

    return decode
