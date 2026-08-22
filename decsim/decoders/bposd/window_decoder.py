"""Inner BP-OSD decoder over one WindowErrorModel."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...detector_error_model.fault_model_contracts import WindowErrorModel


def bposd_window_decoder(max_iter: int = 2, osd_order: int = 0,
                         bp_method: str = "product_sum", schedule: str = "serial",
                         osd_method: str = "osd_cs"):
    """Build a cached BP-OSD callable for WindowErrorModel inputs."""
    cache: dict = {}

    def decode(model: "WindowErrorModel", syndrome):
        from ...detector_error_model.fault_model_contracts import FaultRepresentation

        faults = model.require_faults(FaultRepresentation.PHYSICAL)
        decoder = cache.get(id(faults))
        if decoder is None:
            import weakref

            from ...detector_error_model.fault_identity_validation import (
                validate_placed_fault_matrices,
            )
            from ldpc import BpOsdDecoder
            from scipy.sparse import csr_matrix

            validate_placed_fault_matrices(
                faults.check,
                faults.observables,
                location="BP-OSD window model",
            )
            # ldpc's osd_cs indexes candidate strings by osd_order without a
            # bound check (osd.hpp:90-99) and overruns past n - m; stimbposd
            # clamps the order to the window's own n - m, so do we.
            row_count, column_count = faults.check.shape
            window_osd_order = max(0, min(osd_order, column_count - row_count))
            decoder = BpOsdDecoder(csr_matrix(faults.check),
                                   error_channel=list(faults.priors),
                                   max_iter=max_iter, bp_method=bp_method,
                                   schedule=schedule, osd_method=osd_method,
                                   osd_order=window_osd_order)
            cache[id(faults)] = decoder
            # id() values are recycled by CPython; evict on GC so a fresh model
            # cannot alias a dead one's key and receive a stale decoder (mirrors
            # the MWPM inner decoder's guard).
            weakref.finalize(faults, cache.pop, id(faults), None)
        return decoder.decode(syndrome)

    return decode
