"""The Union-Find adapter: decsim's own weighted growth and peeling.

The growth evidence it returns beside the hard result feeds the cluster
gap (confidence/cluster.py) and is what the ASIC cycle model prices
(tmp/uf-decoder-research/REPORT.md, finding 5).
"""

import dataclasses
from typing import Optional

import decsim.decoders.decoder as decoder_module
import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.decoders.window_decode_results as window_decode_results
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.message as message


@dataclasses.dataclass(frozen=True)
class UnionFindDecodedWindow:
    """One hard result paired with evidence from that exact decode call."""

    hard_result: message.DecodeResult
    hard_evidence: window_decoder.UnionFindHardEvidence


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

    def __init__(
        self,
        latency_model: Optional[decoder_module.DecoderBase] = None,
        weight_step=0.1,
    ) -> None:
        decoder_module.WindowDecoderBase.__init__(self, latency_model)
        # absolute natural-log units represented by one weight tick
        self.weight_step = window_decoder.normalized_weight_step(weight_step)

    def compile(self, faults, model=None) -> window_decoder.UnionFindGraph:
        """The immutable weighted graph of one placed model."""
        del model
        return window_decoder.graph_from_model(
            faults,
            location="Union-Find window model",
            weight_step=self.weight_step,
        )

    def decode_window(self, backend, model, faults, syndrome) -> tuple:
        """The hard correction; unexplained detectors mark it invalid."""
        del model
        del faults
        evidence = window_decoder.decode_graph(backend, syndrome)
        return evidence.selected_faults, _status_of(evidence)

    def decode_with_growth_evidence(
        self, job: message.DecodeJob
    ) -> UnionFindDecodedWindow:
        """One hard result and immutable evidence from the same call."""
        model = job.dem
        if model is None:
            raise ValueError(
                "Union-Find growth evidence requires a window error model"
            )
        faults = model.require_faults(self.fault_representation)
        syndrome = window_decode_results.payload_syndrome(job)
        window_decode_results.check_syndrome_size(job, syndrome, faults)
        graph = self.compiled_for(faults, model)
        hard_evidence = window_decoder.decode_graph(graph, syndrome)
        decode_status = _status_of(hard_evidence)
        hard_result = window_decode_results.result_from_selected_faults(
            job,
            model,
            faults,
            hard_evidence.selected_faults,
            decode_status=decode_status,
        )
        return UnionFindDecodedWindow(hard_result, hard_evidence)


def _status_of(evidence: window_decoder.UnionFindHardEvidence):
    if evidence.unmatched_detectors:
        return window_decode_results.BackendDecodeStatus.INVALID_CORRECTION
    return None
