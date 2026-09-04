"""The Union-Find adapter: decsim's own weighted growth and peeling.

The growth evidence it returns beside the hard result feeds the cluster
gap (confidence/cluster.py) and is what the ASIC cycle model prices
(tmp/uf-decoder-research/REPORT.md, finding 5).
"""

import dataclasses
import weakref

import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.decoders.window_decode_results as window_decode_results
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.message as message


@dataclasses.dataclass(frozen=True)
class UnionFindDecodedWindow:
    """One hard result paired with evidence from that exact decode call."""

    hard_result: message.DecodeResult
    hard_evidence: window_decoder.UnionFindHardEvidence


class UnionFindDecoder:
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

    def __init__(self, latency_model, weight_step=0.1) -> None:
        self.latency_model = latency_model
        # absolute natural-log units represented by one weight tick
        self.weight_step = window_decoder.normalized_weight_step(weight_step)
        self.graph_by_model: dict = {}

    def run_seed_children(self) -> tuple:
        """The latency model at its semantic decoder-child path."""
        path = (message.RunSeedPathSegment("field", "latency_model"),)
        child = message.RunSeedChild(path, self.latency_model)
        return (child,)

    def latency(self, job: message.DecodeJob) -> int:
        """Timing comes from the configured latency model."""
        return self.latency_model.latency(job)

    def decode(self, job: message.DecodeJob) -> message.DecodeResult:
        """Decode one job without constructing confidence."""
        if job.dem is None:
            return message.DecodeResult(job.op_id, job.window_id)
        decoded = self.decode_with_growth_evidence(job)
        return decoded.hard_result

    def decode_with_growth_evidence(
        self, job: message.DecodeJob
    ) -> UnionFindDecodedWindow:
        """One hard result and immutable evidence from the same call."""
        model = job.dem
        if model is None:
            raise ValueError(
                "Union-Find growth evidence requires a window error model"
            )
        faults = model.require_faults(
            fault_models.FaultRepresentation.GRAPHLIKE
        )
        syndrome = window_decode_results.payload_syndrome(job)
        window_decode_results.check_syndrome_size(job, syndrome, faults)
        graph = self.compile(faults, job.label)
        hard_evidence = window_decoder.decode_graph(graph, syndrome)
        decode_status = None
        if hard_evidence.unmatched_detectors:
            decode_status = (
                window_decode_results.BackendDecodeStatus.INVALID_CORRECTION
            )
        hard_result = window_decode_results.result_from_selected_faults(
            job,
            model,
            faults,
            hard_evidence.selected_faults,
            decode_status=decode_status,
        )
        return UnionFindDecodedWindow(hard_result, hard_evidence)

    def compile(
        self, faults, job_label: str = ""
    ) -> window_decoder.UnionFindGraph:
        """The immutable graph of one placed model, kept while it lives."""
        model_identity = id(faults)
        entry = self.graph_by_model.get(model_identity)
        if entry is not None:
            reference, graph = entry
            if reference() is faults:
                return graph
        location = "Union-Find window model"
        if job_label:
            location = f"{job_label} Union-Find window model"
        graph = window_decoder.graph_from_model(
            faults, location=location, weight_step=self.weight_step
        )

        def discard_dead_model(reference) -> None:
            current = self.graph_by_model.get(model_identity)
            if current is not None and current[0] is reference:
                del self.graph_by_model[model_identity]

        reference = weakref.ref(faults, discard_dead_model)
        self.graph_by_model[model_identity] = (reference, graph)
        return graph
