"""The Union-Find adapter: decsim's own weighted growth and peeling.

The growth evidence it returns on every result feeds the cluster gap
(confidence/cluster.py) and is what the ASIC cycle model prices
(AFS 2001.06598, Helios 2301.08419).
"""

from typing import Optional

import decsim.decoders.decoder as decoder_module
import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoder_evidence as evidence_records
import decsim.records.decoding as decoding_records


class UnionFindDecoder(decoder_module.WindowDecoderBase):
    """Prior-weighted graphlike Union-Find hard decoder.

    Faults must be graphlike; detector hyperedges are rejected. Initial
    erasure side information is not implemented. Growth rounds natural
    log-odds to the configured weight step; a probability of one half is
    an ordinary zero-log-odds fault, not erasure. Every logical
    observable row is kept in the hard result. Host runtime is not
    simulated service latency, and this Python implementation does not
    claim the paper's complexity bound.
    """

    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    fault_representation = fault_models.FaultRepresentation.GRAPHLIKE
    decoder_evidence = decoding_records.CLUSTER_GROWTH_EVIDENCE

    def __init__(
        self,
        latency_model: Optional[decoder_module.DecoderBase] = None,
        weight_step=evidence_records.DEFAULT_WEIGHT_STEP,
    ) -> None:
        decoder_module.WindowDecoderBase.__init__(self, latency_model)
        # absolute natural-log units represented by one weight tick
        self.weight_step = evidence_records.normalized_weight_step(weight_step)

    def compile(self, faults, model=None) -> evidence_records.UnionFindGraph:
        """The immutable weighted graph of one placed model."""
        del model
        return window_decoder.graph_from_model(
            faults,
            location="Union-Find window model",
            weight_step=self.weight_step,
        )

    def decode_window(self, backend, model, faults, syndrome):
        """The hard correction and the growth that produced it.

        The immutable growth evidence rides on the result, so a cluster
        gap reads the intervals of this exact decode (Meister et al.
        2405.07433 Algorithm 2 lines 518-525).
        """
        del model
        del faults
        evidence = window_decoder.decode_graph(backend, syndrome)
        decode_status = _status_of(evidence)
        return decoding_records.WindowDecode(
            evidence.selected_faults,
            decode_status,
            cluster_evidence=evidence,
        )


def _status_of(evidence: evidence_records.UnionFindHardEvidence):
    if evidence.unmatched_detectors:
        return decoder_module.BackendDecodeStatus.INVALID_CORRECTION
    return None
