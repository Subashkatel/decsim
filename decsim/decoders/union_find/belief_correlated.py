"""Belief two-pass: correlated_union_find's two passes on BP's posteriors.

Pass one is belief_find's: BP on the window's hyperedges, then the
union_find row's growth with each edge at its posterior's length
(Higgott et al. 2203.04948 Sec. III.A). Pass two is correlated
matching's reweighting (Fowler 1310.0863), as correlated_union_find
makes it: each partner of a fault pass one selected takes the length
P(partner | selected) gives it, and the growth runs again. The sources
reweight a prior graph, and here the graph holds posteriors, so pass
two takes each edge at the shortest of its posterior, prior and partner
lengths: an edge BP made less likely than its prior goes back to its
prior, and with posteriors equal to the priors the row is
correlated_union_find.

The unit spends BP's iterations, then both growths' cycles, as
belief_find and correlated_union_find charge theirs.
"""

import dataclasses

import numpy

import decsim.decoders.union_find.belief_find as belief_find
import decsim.decoders.union_find.correlated_decoder as correlated_decoder
import decsim.decoders.union_find.decoder as union_find
import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.records.decoder_evidence as evidence_records
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models


@dataclasses.dataclass(frozen=True)
class BeliefCorrelatedBackend:
    """belief_find's graph and BP decoder, and the window's partner table."""

    belief: belief_find.BeliefFindBackend
    partners: correlated_decoder.CorrelatedGraph


class BeliefCorrelatedDecoder(belief_find.BeliefFindDecoder):
    """belief_find's growth, then a second growth with partners likelier."""

    @dataclasses.dataclass(frozen=True)
    class Settings(belief_find.BeliefFindDecoder.Settings):
        """belief_find's settings; BP and both growths run on one unit."""

        # the word the reports name this row by
        name = "belief_correlated_union_find"

        def build(self) -> "BeliefCorrelatedDecoder":
            """A fresh decoder of these settings."""
            return BeliefCorrelatedDecoder(self)

    def compile(
        self,
        faults: fault_models.PlacedFaultModel,
        model: fault_models.WindowErrorModel,
    ) -> BeliefCorrelatedBackend:
        """belief_find's backend, and the partner table of its graph."""
        belief = belief_find.BeliefFindDecoder.compile(self, faults, model)
        lengths_by_fault = correlated_decoder.partner_lengths(
            belief.graph, faults, model
        )
        partners = correlated_decoder.CorrelatedGraph(
            belief.graph, lengths_by_fault
        )
        return BeliefCorrelatedBackend(belief, partners)

    def decode_window(
        self,
        backend: BeliefCorrelatedBackend,
        model: fault_models.WindowErrorModel,
        faults: fault_models.PlacedFaultModel,
        syndrome: numpy.ndarray,
    ) -> decoding_records.WindowDecode:
        """Pass two's correction, its evidence holding pass one.

        A pass one that selects no fault with a partner leaves nothing
        to reweight, and its answer stands.
        """
        del model
        del faults
        belief = backend.belief
        first_pass = belief_find.grow_on_posteriors(belief, syndrome)
        iterations, posterior, first = first_pass
        partner_lengths = correlated_decoder.second_pass_lengths(
            backend.partners, first.selected_faults
        )
        if not partner_lengths:
            decode = union_find.window_decode_of(first)
            return dataclasses.replace(decode, iterations=iterations)
        lengths = _shortest_lengths(posterior, belief.graph, partner_lengths)
        second_graph = correlated_decoder.with_lengths(posterior, lengths)
        second = window_decoder.decode_graph(second_graph, syndrome)
        evidence = dataclasses.replace(second, earlier_passes=(first,))
        decode = union_find.window_decode_of(evidence)
        return dataclasses.replace(decode, iterations=iterations)


def _shortest_lengths(
    posterior: evidence_records.UnionFindGraph,
    prior: evidence_records.UnionFindGraph,
    partner_lengths: dict,
) -> dict:
    """Each edge's shortest of its posterior, prior and partner lengths."""
    lengths = {}
    pairs = zip(posterior.edges, prior.edges, strict=True)
    for edge_index, (posterior_edge, prior_edge) in enumerate(pairs):
        posterior_length = posterior_edge.length_half_ticks
        prior_length = prior_edge.length_half_ticks
        partner_length = partner_lengths.get(edge_index, prior_length)
        shortest = min(posterior_length, prior_length, partner_length)
        lengths[edge_index] = shortest
    return lengths
