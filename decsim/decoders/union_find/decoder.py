"""The Union-Find adapter: decsim's own weighted growth and peeling.

The growth evidence it returns on every result feeds the cluster gap
(confidence/cluster.py) and is what the ASIC cycle model prices
(AFS 2001.06598, Helios 2301.08419).
"""

import dataclasses
from typing import Optional

import numpy

import decsim.decoders.decoder as decoder_module
import decsim.decoders.union_find.cycle_count as cycle_count_module
import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.records.decoder_evidence as evidence_records
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models


class UnionFindDecoder(decoder_module.WindowDecoderBase):
    """Prior-weighted graphlike Union-Find hard decoder.

    Faults must be graphlike, and initial erasure side information is
    not implemented. A probability of one half has zero log-odds, so its
    edge starts closed, as an erased edge does (Delfosse and Nickerson
    1709.06218). Every logical observable row is kept in the hard
    result. The decoder does not claim the paper's almost-linear bound:
    the cycle count's flood lays the closed edges out again at every
    growth step. The row is priced by its own cycle count
    (cycle_count.py), from the growth steps and the peel depth of the
    decode just run, or by the host's measured time.
    """

    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    fault_representation = fault_models.FaultRepresentation.GRAPHLIKE
    decoder_evidence = decoding_records.CLUSTER_GROWTH_EVIDENCE
    missing_evidence_reasons = {
        decoding_records.DecoderEvidence.SPLIT_FAULT_PARTNERS: (
            "union-find carries the split-fault pairs only with "
            "split_fault_partners on, which asks each window model for "
            "Stim's physical-to-graphlike link"
        ),
    }

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The union-find row's settings.

        weight_step is the growth resolution, the natural-log weight one
        tick of edge length is (Huang, Newman and Brown 2004.04693): the
        smaller it is, the longer every edge and the more growth
        iterations a decode spans. timing prices the decode: a cycle
        count (cycle_count.CycleCount) or
        cycle_count.HostMeasuredTime(). A run names one; none is
        refused, since the host's time is no hardware's and is never
        assumed. split_fault_partners builds each graph with the edges
        every piece of a decomposed fault shortens
        (window_decoder.split_fault_partners), which the correlated
        cluster gap reads; the decode itself never does.
        """

        weight_step: float = evidence_records.DEFAULT_WEIGHT_STEP
        timing: Optional[cycle_count_module.Timing] = None
        split_fault_partners: bool = False
        # the word the reports name this row by
        name = "union_find"

        def __post_init__(self) -> None:
            weight_step = evidence_records.normalized_weight_step(
                self.weight_step, "weight_step"
            )
            object.__setattr__(self, "weight_step", weight_step)
            if self.timing is None:
                raise ValueError(
                    "timing must be a cycle count (cycle_count.CycleCount) "
                    "or the host's measured time "
                    "(cycle_count.HostMeasuredTime()); none is given"
                )

        def build(self) -> "UnionFindDecoder":
            """A fresh decoder of these settings."""
            return UnionFindDecoder(self)

    def __init__(self, settings: "UnionFindDecoder.Settings") -> None:
        decoder_module.WindowDecoderBase.__init__(self)
        self.compile_key = (UnionFindDecoder, settings)
        # absolute natural-log units represented by one weight tick
        self.weight_step = settings.weight_step
        self.timing = settings.timing
        self.split_fault_partners = settings.split_fault_partners
        if settings.split_fault_partners:
            self.fault_model_requirement = (
                fault_models.LINKED_FAULT_MODELS_REQUIRED
            )
            self.decoder_evidence = decoding_records.CORRELATED_CLUSTER_EVIDENCE

    def ticks_after_decode(
        self,
        result: Optional[decoding_records.DecodeResult],
        elapsed_nanoseconds: int,
        now: int,
    ) -> int:
        """The ticks the row's timing holds the unit for this decode."""
        evidence = result.cluster_evidence
        return self.timing.decode_ticks(evidence, elapsed_nanoseconds, now)

    def compile(
        self,
        faults: fault_models.PlacedFaultModel,
        model: fault_models.WindowErrorModel,
    ) -> evidence_records.UnionFindGraph:
        """The immutable weighted graph of one placed model."""
        graph = window_decoder.graph_from_model(
            faults,
            location="Union-Find window model",
            weight_step=self.weight_step,
        )
        if not self.split_fault_partners:
            return graph
        partners = window_decoder.split_fault_partners(graph, model)
        return dataclasses.replace(graph, split_fault_partners=partners)

    def decode_window(
        self,
        backend: evidence_records.UnionFindGraph,
        model: fault_models.WindowErrorModel,
        faults: fault_models.PlacedFaultModel,
        syndrome: numpy.ndarray,
    ) -> decoding_records.WindowDecode:
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
        return decoding_records.BackendDecodeStatus.INVALID_CORRECTION
    return None
