"""BP-OSD over one window: ldpc's BpOsdDecoder on the physical faults.

qLDPC's default decoder (qldpc/decoders/retrieval.py:132 builds the
same BpOsdDecoder); one decoder per live window model. ldpc's osd_cs
indexes candidate strings by osd_order without a bound check
(osd.hpp:90-99) and overruns past n - m, so the order is clamped to the
window's own n - m, as stimbposd clamps it (bp_osd.py:62-68).
"""

import weakref

import ldpc
import scipy.sparse

import decsim.detector_error_model.fault_identity_validation as fault_identity
import decsim.detector_error_model.fault_model_contracts as fault_models


def bposd_window_decoder(
    max_iterations: int = 2,
    osd_order: int = 0,
    belief_propagation_method: str = "product_sum",
    schedule: str = "serial",
    osd_method: str = "osd_cs",
):
    """A BP-OSD callable over WindowErrorModel inputs, one decoder per model."""
    cache: dict = {}

    def decode(model, syndrome):
        faults = model.require_faults(fault_models.FaultRepresentation.PHYSICAL)
        model_identity = id(faults)
        decoder = cache.get(model_identity)
        if decoder is None:
            decoder = _build_decoder(
                faults,
                max_iterations,
                osd_order,
                belief_propagation_method,
                schedule,
                osd_method,
            )
            cache[model_identity] = decoder
            # id() values are recycled by CPython; evict on GC so a fresh
            # model cannot alias a dead one's key and receive a stale decoder
            weakref.finalize(faults, cache.pop, model_identity, None)
        return decoder.decode(syndrome)

    return decode


def _build_decoder(
    faults,
    max_iterations: int,
    osd_order: int,
    belief_propagation_method: str,
    schedule: str,
    osd_method: str,
):
    fault_identity.validate_placed_fault_matrices(
        faults.check, faults.observables, location="BP-OSD window model"
    )
    row_count, column_count = faults.check.shape
    window_rank = column_count - row_count
    window_osd_order = max(0, min(osd_order, window_rank))
    check = scipy.sparse.csr_matrix(faults.check)
    error_channel = list(faults.priors)
    return ldpc.BpOsdDecoder(
        check,
        error_channel=error_channel,
        max_iter=max_iterations,
        bp_method=belief_propagation_method,
        schedule=schedule,
        osd_method=osd_method,
        osd_order=window_osd_order,
    )
