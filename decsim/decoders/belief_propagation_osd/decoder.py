"""The BP-OSD adapter: ldpc's BpOsdDecoder on the window's physical faults.

qLDPC's default decoder (qldpc/decoders/retrieval.py:132 builds the same
BpOsdDecoder), one decoder per live window model. ldpc's osd_cs indexes
candidate strings by osd_order without a bound check (osd.hpp:90-99) and
overruns past n - m, so the order is clamped to the window's own n - m,
as stimbposd clamps it (bp_osd.py:62-68).
"""

from typing import Optional

import ldpc
import scipy.sparse

import decsim.decoders.decoder as decoder_module
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoding as decoding_records


class BeliefPropagationOsdDecoder(decoder_module.WindowDecoderBase):
    """Decode one window with BP-OSD; ldpc's argument names are kept."""

    fault_model_requirement = fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    fault_representation = fault_models.FaultRepresentation.PHYSICAL

    def __init__(
        self,
        latency_model: Optional[decoder_module.DecoderBase] = None,
        max_iterations: int = 2,
        osd_order: int = 0,
        belief_propagation_method: str = "product_sum",
        schedule: str = "serial",
        osd_method: str = "osd_cs",
    ):
        decoder_module.WindowDecoderBase.__init__(self, latency_model)
        self.max_iterations = max_iterations
        self.osd_order = osd_order
        self.belief_propagation_method = belief_propagation_method
        self.schedule = schedule
        self.osd_method = osd_method

    def compile(self, faults, model=None):
        """Ldpc's BP-OSD decoder over the window's physical check."""
        del model
        row_count, column_count = faults.check.shape
        window_rank = column_count - row_count
        window_osd_order = max(0, min(self.osd_order, window_rank))
        check = scipy.sparse.csr_matrix(faults.check)
        error_channel = list(faults.priors)
        return ldpc.BpOsdDecoder(
            check,
            error_channel=error_channel,
            max_iter=self.max_iterations,
            bp_method=self.belief_propagation_method,
            schedule=self.schedule,
            osd_method=self.osd_method,
            osd_order=window_osd_order,
        )

    def decode_window(self, backend, model, faults, syndrome):
        """One BP-OSD call; ldpc always returns a correction."""
        del model
        del faults
        selected = backend.decode(syndrome)
        return decoding_records.WindowDecode(selected)
