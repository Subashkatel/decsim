"""Belief-find: union-find grown on belief propagation's posteriors.

Higgott et al. 2203.04948 (Sec. III.A) run belief propagation on the
circuit's hyperedges and then weighted union-find on the posterior
graph, in place of belief matching's matching, and call both stages
amenable to hardware (p. 6). The posteriors come as the belief_matching
row gets them (belief_matching/decoder.py): ldpc's BpDecoder on one
column per distinct detector set, mapped onto the graphlike faults by
the window's projection. The union_find row's growth then runs with
each edge at the length its posterior gives.

The paper returns a converged BP's own correction (Sec. I); this row
grows every time, one fixed pipeline, so every decode carries the
growth evidence the cluster gap and the cycle count read. A converged
BP's posteriors sit near zero or one, and the growth on them gives BP's
logical class on every converged shot of the row's test (tests/decoders/
test_union_find_belief_find.py: 200 shots of a d = 3, 3-round memory at
p = 0.005); no wider sweep is in the tree.

The unit spends the BP iterations at a stated time each, then the
growth's own cycles. Relay-BP's FPGA runs one iteration in 24 ns
(Maurer et al. 2510.21600 Fig. 5: two 12 ns decoder cycles).
"""

import dataclasses
from typing import Optional

import ldpc
import numpy
import scipy.sparse

import decsim.config as config
import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.decoders.union_find.decoder as union_find
import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.records.decoder_evidence as evidence_records
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models


@dataclasses.dataclass(frozen=True)
class BeliefFindBackend:
    """One window's graph, its BP decoder, and the hyperedge-to-edge map."""

    graph: evidence_records.UnionFindGraph
    belief_propagation: ldpc.BpDecoder
    edge_from_hyperedge: scipy.sparse.csr_matrix


class BeliefFindDecoder(union_find.UnionFindDecoder):
    """BP on the window's hyperedges, then the union_find row's growth.

    BP reads the physical faults and the growth the graphlike ones, so
    the row needs both models.
    """

    fault_model_requirement = fault_models.LINKED_FAULT_MODELS_REQUIRED

    @dataclasses.dataclass(frozen=True)
    class Settings(union_find.UnionFindDecoder.Settings):
        """The union_find row's settings and the BP before the growth.

        max_iterations and belief_propagation_method are ldpc
        BpDecoder's max_iter and bp_method; Higgott et al. stop BP at
        30 iterations (2203.04948 App. A), as the belief_matching row
        does.
        iteration_microseconds is the stated time of one BP iteration on
        the unit. With the host's measured time as the growth's timing,
        the host's clock already spans BP, and this is then zero.
        """

        max_iterations: int = 30
        belief_propagation_method: str = "product_sum"
        iteration_microseconds: float = 0.024
        # the word the reports name this row by
        name = "belief_find"

        def __post_init__(self) -> None:
            union_find.UnionFindDecoder.Settings.__post_init__(self)
            config.check_microseconds(
                "iteration_microseconds", self.iteration_microseconds
            )

        def build(self) -> "BeliefFindDecoder":
            """A fresh decoder of these settings."""
            return BeliefFindDecoder(self)

    def __init__(self, settings: "BeliefFindDecoder.Settings") -> None:
        union_find.UnionFindDecoder.__init__(self, settings)
        self.max_iterations = settings.max_iterations
        self.belief_propagation_method = settings.belief_propagation_method
        self.iteration_microseconds = settings.iteration_microseconds

    def ticks_after_decode(
        self,
        result: Optional[decoding_records.DecodeResult],
        elapsed_nanoseconds: int,
        now: int,
    ) -> int:
        """BP's iterations at their stated time, then the growth's cycles."""
        iterations = result.iterations or 0
        microseconds = self.iteration_microseconds * iterations
        belief_ticks = config.microseconds_to_ticks(microseconds)
        growth_start = now + belief_ticks
        growth_ticks = union_find.UnionFindDecoder.ticks_after_decode(
            self, result, elapsed_nanoseconds, growth_start
        )
        return belief_ticks + growth_ticks

    def compile(
        self,
        faults: fault_models.PlacedFaultModel,
        model: fault_models.WindowErrorModel,
    ) -> BeliefFindBackend:
        """The window's shared graph and BP decoder over its hyperedges."""
        graph = union_find.UnionFindDecoder.compile(self, faults, model)
        belief = belief_matching.hyperedge_belief_propagation(
            model, self.max_iterations, self.belief_propagation_method
        )
        belief_propagation, edge_from_hyperedge = belief
        return BeliefFindBackend(graph, belief_propagation, edge_from_hyperedge)

    def decode_window(
        self,
        backend: BeliefFindBackend,
        model: fault_models.WindowErrorModel,
        faults: fault_models.PlacedFaultModel,
        syndrome: numpy.ndarray,
    ) -> decoding_records.WindowDecode:
        """The growth's correction on the posterior lengths, and BP's steps."""
        del model
        del faults
        iterations, _graph, evidence = grow_on_posteriors(backend, syndrome)
        decode = union_find.window_decode_of(evidence)
        return dataclasses.replace(decode, iterations=iterations)


def grow_on_posteriors(
    backend: BeliefFindBackend, syndrome: numpy.ndarray
) -> tuple:
    """(BP's iterations, the posterior graph, the growth on it)."""
    belief_propagation = backend.belief_propagation
    belief_propagation.decode(syndrome)
    iterations = int(belief_propagation.iter)
    posteriors = belief_matching.edge_posteriors(
        belief_propagation, backend.edge_from_hyperedge
    )
    graph = posterior_graph(backend.graph, posteriors)
    evidence = window_decoder.decode_graph(graph, syndrome)
    return iterations, graph, evidence


def posterior_graph(
    graph: evidence_records.UnionFindGraph, posteriors: numpy.ndarray
) -> evidence_records.UnionFindGraph:
    """The graph with each edge at its fault's posterior length.

    A posterior of one half or more gives the shortest length the
    window's quantization has (window_decoder.length_half_ticks).
    """
    weight_step = graph.weight_step
    edges = []
    for edge in graph.edges:
        posterior = float(posteriors[edge.fault_index])
        length = window_decoder.length_half_ticks(posterior, weight_step)
        reweighted = dataclasses.replace(edge, length_half_ticks=length)
        edges.append(reweighted)
    edge_tuple = tuple(edges)
    return dataclasses.replace(graph, edges=edge_tuple)
