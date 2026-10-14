"""The correlated cluster gap: the cluster gap with split-fault pairs shortened.

The referent for the pairs is PyMatching 2.4.0's correlated-matching rule
(user_graph.cc, populate_implied_edge_weights; graph.h, apply_reweights)
applied here to the very Stim model decsim's correlated matching row
hands PyMatching (minimum_weight_perfect_matching _correlated_model),
read error by error. The referent for the gap is the Python walk decsim
ran before the compiled one (cluster_gap_oracle.py), on a graph rebuilt
from the shortened lengths, each edge's uncovered part its new length
less what the growth covered.
"""

import collections
import dataclasses
import itertools
import math

import numpy

import decsim.confidence.cluster as cluster
import decsim.confidence.correlated_cluster as correlated_cluster
import decsim.decoders.union_find.cycle_count as cycle_count_module
import decsim.decoders.union_find.decoder as union_find
import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.records.decoder_evidence as evidence_records
import decsim.records.fault_model_contracts as fault_models
import tests.confidence.cluster_gap_oracle as cluster_gap_oracle
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)
from tests.decoders import windows

ROUNDS = 4
SHOTS = 40
SEED = 7
HOST_TIME = cycle_count_module.HostMeasuredTime()
WITH_PAIRS = union_find.UnionFindDecoder.Settings(
    timing=HOST_TIME, split_fault_partners=True
)
WITHOUT_PAIRS = union_find.UnionFindDecoder.Settings(timing=HOST_TIME)
LINKED = fault_models.LINKED_FAULT_MODELS_REQUIRED


def _window(distance: int = 3):
    circuit = windows.memory_circuit(distance, ROUNDS, 0.005)
    model = windows.whole_circuit_window(circuit, ROUNDS, LINKED)
    return circuit, model


def test_the_pairs_are_pymatchings_rule_on_the_model_it_is_handed():
    """Every column's pairs, against the rule read off the Stim model.

    _correlated_model gives each graphlike column its own observable, so
    a piece of an error is its column; the joint and the marginal
    XOR-merge every error holding the pair and the piece, q is min(1/2,
    joint / marginal), and a pair is kept when its length is shorter.
    """
    _circuit, model = _window()
    faults = model.require_faults(fault_models.FaultRepresentation.GRAPHLIKE)
    row = union_find.UnionFindDecoder(WITH_PAIRS)
    graph = row.compile(faults, model)
    stim_model = minimum_weight_perfect_matching._correlated_model(
        faults, model
    )
    expected = _pairs_by_rule(stim_model, graph)
    assert graph.split_fault_partners == expected
    pair_count = sum(len(pairs) for pairs in expected)
    assert pair_count > 0


def test_the_pairs_change_no_correction_property():
    """The decode never reads the pairs: the same faults, shot for shot."""
    circuit, model = _window()
    with_pairs = union_find.UnionFindDecoder(WITH_PAIRS)
    without_pairs = union_find.UnionFindDecoder(WITHOUT_PAIRS)
    events, _observables = windows.sampled_shots(circuit, SHOTS, SEED)
    for shot in events:
        paired_job = windows.job_for(model, shot)
        plain_job = windows.job_for(model, shot)
        paired = with_pairs.decode(paired_job)
        plain = without_pairs.decode(plain_job)
        paired_faults = paired.cluster_evidence.selected_faults
        plain_faults = plain.cluster_evidence.selected_faults
        assert paired_faults == plain_faults


def test_the_correlated_gap_is_the_oracle_walk_and_never_above_property():
    """The gap on the shortened lengths, walked by the Python oracle.

    Shortening only lowers lengths, so the gap is never above the
    cluster gap of the same decode; on some shots of a d = 5 window it
    is below.
    """
    circuit, model = _window(distance=5)
    row = union_find.UnionFindDecoder(WITH_PAIRS)
    correlated = correlated_cluster.CorrelatedClusterGap()
    plain = cluster.ClusterGap()
    events, _observables = windows.sampled_shots(circuit, SHOTS, SEED)
    below = 0
    for shot in events:
        job = windows.job_for(model, shot)
        result = row.decode(job)
        correlated_computation = correlated.compute((result,))
        plain_computation = plain.compute((result,))
        correlated_gap = correlated_computation.soft_output.gap
        plain_gap = plain_computation.soft_output.gap
        oracle_gap = _oracle_gap(result.cluster_evidence)
        assert _same_gap(correlated_gap, oracle_gap)
        assert correlated_gap <= plain_gap
        below += correlated_gap < plain_gap
    assert below > 0


def _pairs_by_rule(stim_model, graph) -> tuple:
    """PyMatching's pairs, keyed and kept as the graph keys and keeps them."""
    joint = collections.defaultdict(float)
    marginal = collections.defaultdict(float)
    for instruction in stim_model.flattened():
        if instruction.type == "error":
            _merge_error(instruction, joint, marginal)
    return _kept_pairs(joint, marginal, graph)


def _kept_pairs(joint, marginal, graph) -> tuple:
    """Each pair at its conditional length, kept where it is shorter."""
    edge_by_fault = {
        edge.fault_index: index for index, edge in enumerate(graph.edges)
    }
    pairs = [[] for _ in range(graph.fault_count)]
    for (cause, partner), probability in joint.items():
        ratio = probability / marginal[cause]
        q = min(0.5, ratio)
        odds = (1 - q) / q
        weight = math.log(odds)
        ticks = window_decoder._quantize_weight_ticks(weight, graph.weight_step)
        half_ticks = 2 * ticks
        edge_index = edge_by_fault[partner]
        if half_ticks < graph.edges[edge_index].length_half_ticks:
            pair = (edge_index, half_ticks)
            pairs[cause].append(pair)
    return tuple(tuple(sorted(column_pairs)) for column_pairs in pairs)


def _merge_error(instruction, joint, marginal) -> None:
    """One error's probability, XOR-merged into its pieces and pairs."""
    arguments = instruction.args_copy()
    probability = arguments[0]
    pieces = _pieces(instruction)
    for piece in pieces:
        marginal[piece] = _xor(marginal[piece], probability)
    for pair in itertools.permutations(pieces, 2):
        joint[pair] = _xor(joint[pair], probability)


def _pieces(instruction) -> list:
    """An error's pieces: the observable each one carries is its column."""
    pieces = []
    for target in instruction.targets_copy():
        if target.is_logical_observable_id():
            pieces.append(target.val)
    return pieces


def _xor(first: float, second: float) -> float:
    return first * (1 - second) + second * (1 - first)


def _oracle_gap(evidence) -> float:
    """The Python walk on the graph with every pair of the correction."""
    graph = evidence.graph
    lengths = _shortened_lengths(evidence)
    edges = []
    intervals = []
    for edge, interval, length in zip(
        graph.edges, evidence.edge_intervals, lengths, strict=True
    ):
        shortened_edge = dataclasses.replace(edge, length_half_ticks=length)
        uncovered = _uncovered_interval(edge, interval, length)
        edges.append(shortened_edge)
        intervals.append(uncovered)
    shortened = dataclasses.replace(graph, edges=tuple(edges))
    half_ticks = cluster_gap_oracle._quotient_cluster_gap(
        shortened, tuple(intervals)
    )
    return cluster.gap_half_ticks_to_natural_log_weight(
        half_ticks, graph.weight_step
    )


def _shortened_lengths(evidence) -> list:
    """Each edge at the shortest length a selected fault gives it."""
    graph = evidence.graph
    lengths = [edge.length_half_ticks for edge in graph.edges]
    selected = numpy.nonzero(evidence.selected_faults)
    for fault in selected[0]:
        for edge_index, half_ticks in graph.split_fault_partners[fault]:
            lengths[edge_index] = min(lengths[edge_index], half_ticks)
    return lengths


def _uncovered_interval(edge, interval, length: int):
    """The new length less what the growth covered, as one open span."""
    if isinstance(interval, evidence_records.Closed):
        return interval
    uncovered = interval.upper_tick - interval.lower_tick
    covered = edge.length_half_ticks - uncovered
    remaining = length - covered
    still_uncovered = max(0, remaining)
    if still_uncovered == 0:
        return evidence_records.Closed()
    return evidence_records.Open(0, still_uncovered)


def _same_gap(inside: float, outside: float) -> bool:
    if math.isinf(inside):
        return math.isinf(outside)
    return math.isclose(inside, outside, abs_tol=1e-9)
