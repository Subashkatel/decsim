"""Correlated Union-Find: two passes, as correlated matching matches twice.

Fowler 1310.0863 matches, reweights and matches again. Pass one
decodes the window as the union_find row does. Each fault it selects
makes its partners likelier: a partner is another graphlike component
of a physical fault that holds it, as a Y error is an X part and a Z
part. A partner's probability becomes the larger of its prior
and P(partner | selected) = P(both) / P(selected), and pass two decodes
on the lengths those give. PyMatching 2.4.0's correlated matching reads
the same joint probabilities from a decomposed Stim model; Riverlane's
correlated clustering decoder (2610.26785 p. 2) runs the same rule on
its own clustering. The unit spends both passes' cycles
(cycle_count.py), and the cluster gap reads the second pass.
"""

import collections
import dataclasses
import itertools

import numpy

import decsim.decoders.union_find.decoder as union_find
import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.detector_error_model.basis_split as basis_split
import decsim.records.decoder_evidence as evidence_records
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models

# a partner is made no likelier than one half, an edge the growth starts
# closed, as PyMatching caps an implied weight at zero
LARGEST_CONDITIONAL = 0.5


@dataclasses.dataclass(frozen=True)
class CorrelatedGraph:
    """One window's graph and the lengths each fault gives its partners.

    partner_lengths maps a fault to ((edge, length in half ticks), ...),
    the partners its selection makes shorter.
    """

    graph: evidence_records.UnionFindGraph
    partner_lengths: dict


class CorrelatedUnionFindDecoder(union_find.UnionFindDecoder):
    """Two passes of the union_find row, the second on reweighted lengths.

    The partners come from the window's link between its physical and
    graphlike faults, so the row needs both models.
    """

    fault_model_requirement = fault_models.LINKED_FAULT_MODELS_REQUIRED

    @dataclasses.dataclass(frozen=True)
    class Settings(union_find.UnionFindDecoder.Settings):
        """The union_find row's settings; both passes run on one unit."""

        # the word the reports name this row by
        name = "correlated_union_find"

        def build(self) -> "CorrelatedUnionFindDecoder":
            """A fresh decoder of these settings."""
            return CorrelatedUnionFindDecoder(self)

    def compile(
        self,
        faults: fault_models.PlacedFaultModel,
        model: fault_models.WindowErrorModel,
    ) -> CorrelatedGraph:
        """The window's shared graph and its partner table."""
        graph = union_find.UnionFindDecoder.compile(self, faults, model)
        partner_lengths = _partner_lengths(graph, faults, model)
        return CorrelatedGraph(graph, partner_lengths)

    def decode_window(
        self,
        backend: CorrelatedGraph,
        model: fault_models.WindowErrorModel,
        faults: fault_models.PlacedFaultModel,
        syndrome: numpy.ndarray,
    ) -> decoding_records.WindowDecode:
        """Pass two's correction, its evidence holding pass one.

        A first pass that selects no fault with a partner leaves nothing
        to reweight, and its answer stands.
        """
        del model
        del faults
        first = window_decoder.decode_graph(backend.graph, syndrome)
        lengths = _second_pass_lengths(backend, first.selected_faults)
        if not lengths:
            return union_find.window_decode_of(first)
        second_graph = _with_lengths(backend.graph, lengths)
        second = window_decoder.decode_graph(second_graph, syndrome)
        evidence = dataclasses.replace(second, earlier_passes=(first,))
        return union_find.window_decode_of(evidence)


def _partner_lengths(
    graph: evidence_records.UnionFindGraph,
    faults: fault_models.PlacedFaultModel,
    model: fault_models.WindowErrorModel,
) -> dict:
    """Each fault's partners, at the length its selection gives them.

    A partner keeps its own length when its prior is already as likely.
    """
    joint = _joint_probabilities(model)
    priors = numpy.asarray(faults.priors)
    edge_of_fault = {}
    for edge_index, edge in enumerate(graph.edges):
        edge_of_fault[edge.fault_index] = edge_index
    weight_step = graph.weight_step
    partner_lengths = collections.defaultdict(list)
    for (selected, partner), both in joint.items():
        conditional = both / priors[selected]
        conditional = min(conditional, LARGEST_CONDITIONAL)
        partner_edge = edge_of_fault.get(partner)
        if partner_edge is None or conditional <= priors[partner]:
            continue
        length = window_decoder.length_half_ticks(conditional, weight_step)
        partner_lengths[selected].append((partner_edge, length))
    return dict(partner_lengths)


def _joint_probabilities(model: fault_models.WindowErrorModel) -> dict:
    """P(both) of each ordered pair of graphlike parts of one physical fault.

    The pairs of several physical faults add, as PyMatching adds the
    hyperedges two edges share.
    """
    physical = model.require_faults(fault_models.FaultRepresentation.PHYSICAL)
    link = model.physical_to_graphlike_detector_projection
    components_by_fault = basis_split.every_column_rows(link)
    physical_priors = numpy.asarray(physical.priors)
    joint = collections.defaultdict(float)
    for physical_fault, components in enumerate(components_by_fault):
        prior = physical_priors[physical_fault]
        for pair in itertools.permutations(components, 2):
            joint[pair] += prior
    return joint


def _second_pass_lengths(backend: CorrelatedGraph, selected_faults) -> dict:
    """The shortest length each partner of a selected fault is given."""
    lengths = {}
    chosen = numpy.flatnonzero(selected_faults)
    for fault_index in chosen.tolist():
        partners = backend.partner_lengths.get(fault_index, ())
        for edge_index, length in partners:
            current = lengths.get(edge_index, length)
            lengths[edge_index] = min(current, length)
    return lengths


def _with_lengths(
    graph: evidence_records.UnionFindGraph, lengths: dict
) -> evidence_records.UnionFindGraph:
    """The graph with these edges at these lengths."""
    edges = list(graph.edges)
    for edge_index, length in lengths.items():
        edge = edges[edge_index]
        edges[edge_index] = dataclasses.replace(edge, length_half_ticks=length)
    edge_tuple = tuple(edges)
    return dataclasses.replace(graph, edges=edge_tuple)
