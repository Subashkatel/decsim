"""Runtime adapter for the hard Union-Find decoder."""

from __future__ import annotations

from dataclasses import dataclass

from ..adapters.window_decode_results import (
    check_syndrome_size,
    payload_syndrome,
    result_from_selected_faults,
)
from ..detector_error_model import (
    FaultRepresentation,
    GRAPHLIKE_FAULT_MODEL_REQUIRED,
)
from ..message import (
    DecodeJob,
    DecodeResult,
    RunSeedChild,
    RunSeedPathSegment,
)
from .window_decoder import (
    UnionFindGraph,
    UnionFindHardEvidence,
    _decode_graph,
    _graph_from_model,
)


@dataclass(frozen=True)
class UnionFindDecodedWindow:
    """One hard result paired with evidence from that exact decode call."""

    hard_result: DecodeResult
    hard_evidence: UnionFindHardEvidence


class UnionFindDecoder:
    """Uniform graphlike Union-Find hard decoder.

    SCOPE:
    - Faults must be graphlike/stringlike; detector hyperedges are rejected.
    - Initial erasures and prior-weighted growth are not implemented.
    - Growth uses uniform fair half-edge sweeps with stable fault-index ties.
    - Every logical-observable row is retained in the hard result.
    - Host runtime is not simulated decoder service latency.
    - This Python implementation does not claim the paper's complexity bound.

    The hard algorithm follows Delfosse--Nickerson arXiv:1709.06218v3,
    Algorithms 1--2.
    """

    fault_model_requirement = GRAPHLIKE_FAULT_MODEL_REQUIRED

    def __init__(self, latency_model) -> None:
        self.latency_model = latency_model
        self._graphs: dict = {}

    def run_seed_children(self):
        """Expose the latency model at its semantic decoder-child path."""
        return (
            RunSeedChild(
                (RunSeedPathSegment("field", "latency_model"),),
                self.latency_model,
            ),
        )

    def latency(self, job: DecodeJob) -> int:
        """Timing comes from the configured latency model."""
        return self.latency_model.latency(job)

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Decode one job without constructing confidence."""
        if job.dem is None:
            return DecodeResult(job.op_id, job.window_id)
        return self.decode_with_growth_evidence(job).hard_result

    def decode_with_growth_evidence(
        self,
        job: DecodeJob,
    ) -> UnionFindDecodedWindow:
        """Return one hard result and immutable evidence from the same call."""
        model = job.dem
        if model is None:
            raise ValueError(
                "Union-Find growth evidence requires a window error model"
            )
        faults = model.require_faults(FaultRepresentation.GRAPHLIKE)
        syndrome = payload_syndrome(job)
        check_syndrome_size(job, syndrome, faults)
        graph = self._graph_for_model(faults, job.label)
        hard_evidence = _decode_graph(graph, syndrome)
        hard_result = result_from_selected_faults(
            job,
            model,
            faults,
            hard_evidence.selected_faults,
        )
        return UnionFindDecodedWindow(hard_result, hard_evidence)

    def _graph_for_model(
        self,
        faults,
        job_label: str,
    ) -> UnionFindGraph:
        """Return the immutable graph owned by one live placed model."""
        import weakref

        model_identity = id(faults)
        entry = self._graphs.get(model_identity)
        graph = entry[1] if entry is not None and entry[0]() is faults else None
        if graph is None:
            location = (
                f"{job_label} Union-Find window model"
                if job_label
                else "Union-Find window model"
            )
            graph = _graph_from_model(faults, location=location)

            def discard_dead_model(reference) -> None:
                current = self._graphs.get(model_identity)
                if current is not None and current[0] is reference:
                    del self._graphs[model_identity]

            reference = weakref.ref(faults, discard_dead_model)
            self._graphs[model_identity] = (reference, graph)
        return graph
