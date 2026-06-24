"""Inner BP-OSD decoder used by decode_windowed tests and references."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..adapters.window_error_models import WindowErrorModel


def bposd_window_decoder(max_iter: int = 2, osd_order: int = 0,
                         bp_method: str = "product_sum", schedule: str = "serial",
                         osd_method: str = "osd_cs"):
    """Build a cached BP-OSD callable for WindowErrorModel inputs."""
    cache: dict = {}

    def decode(model: "WindowErrorModel", syndrome):
        decoder = cache.get(id(model))
        if decoder is None:
            from ldpc import BpOsdDecoder
            from scipy.sparse import csr_matrix
            decoder = BpOsdDecoder(csr_matrix(model.check),
                                   error_channel=list(model.priors),
                                   max_iter=max_iter, bp_method=bp_method,
                                   schedule=schedule, osd_method=osd_method,
                                   osd_order=osd_order)
            cache[id(model)] = decoder
        return decoder.decode(syndrome)

    return decode
