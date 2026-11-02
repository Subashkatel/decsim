"""belief_correlated_union_find against belief_find and correlated_union_find.

Pass one is belief_find's growth (Higgott et al. 2203.04948 Sec. III.A),
so on every shot it must select what the belief_find row selects on the
same window; that row is checked against beliefmatching's posteriors.
Pass two is correlated matching's reweighting (Fowler 1310.0863) on the
posterior graph: each edge at the shortest of its posterior length, its
prior length and the length correlated_union_find's partner table gives
it for pass one's selection; that table is checked against Stim's
decomposition.
"""

import numpy

import decsim.config as config
import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.decoders.union_find.belief_correlated as belief_correlated
import decsim.decoders.union_find.belief_find as belief_find
import decsim.decoders.union_find.correlated_decoder as correlated
import decsim.records.fault_model_contracts as fault_models
import tests.decoders.test_union_find_cycle_count as cycle_counts
from tests.decoders import windows

LINKED = fault_models.LINKED_FAULT_MODELS_REQUIRED
GRAPHLIKE = fault_models.FaultRepresentation.GRAPHLIKE
ROUNDS = 3
SHOTS = 200
TIMING = cycle_counts.TRACED_CYCLE_COUNT


def test_pass_one_selects_what_belief_find_selects_property():
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(circuit, ROUNDS, LINKED)
    two_pass_settings = belief_correlated.BeliefCorrelatedDecoder.Settings(
        timing=TIMING
    )
    two_pass = two_pass_settings.build()
    one_pass_settings = belief_find.BeliefFindDecoder.Settings(timing=TIMING)
    one_pass = one_pass_settings.build()
    detection_events, _ = windows.sampled_shots(circuit, SHOTS, 7)
    found = []
    expected = []
    second_passes = 0
    for shot in detection_events:
        job = windows.job_for(model, shot)
        result = two_pass.decode(job)
        evidence = result.cluster_evidence
        passes = (*evidence.earlier_passes, evidence)
        found.append(passes[0].selected_faults)
        reference = one_pass.decode(job)
        expected.append(reference.cluster_evidence.selected_faults)
        second_passes += len(evidence.earlier_passes)

    assert found == expected
    # pass two runs on some shots and not on all of them
    assert 0 < second_passes < SHOTS


def test_pass_two_takes_each_edges_shortest_length():
    """Posterior, prior, or partner length, whichever is shortest."""
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(circuit, ROUNDS, LINKED)
    faults = model.require_faults(GRAPHLIKE)
    settings = belief_correlated.BeliefCorrelatedDecoder.Settings(timing=TIMING)
    row = settings.build()
    backend = row.compile(faults, model)
    shot, result = _a_shot_with_two_passes(row, model, circuit)
    second = result.cluster_evidence
    first = second.earlier_passes[0]
    syndrome = windows.row_syndrome(model, shot)
    belief = backend.belief
    belief.belief_propagation.decode(syndrome)
    posteriors = belief_matching.edge_posteriors(
        belief.belief_propagation, belief.edge_from_hyperedge
    )
    posterior = belief_find.posterior_graph(belief.graph, posteriors)
    partner_lengths = correlated.second_pass_lengths(
        backend.partners, first.selected_faults
    )
    prior_lengths = _lengths(belief.graph)
    partnered = correlated.with_lengths(belief.graph, partner_lengths)

    found = _lengths(second.graph)

    posterior_lengths = _lengths(posterior)
    partner_lengths_or_prior = _lengths(partnered)
    expected = numpy.minimum.reduce(
        [posterior_lengths, prior_lengths, partner_lengths_or_prior]
    )
    numpy.testing.assert_array_equal(found, expected)


def test_a_decode_is_charged_its_iterations_then_both_growths():
    """One microsecond an iteration, then both passes' traced cycles."""
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(circuit, ROUNDS, LINKED)
    settings = belief_correlated.BeliefCorrelatedDecoder.Settings(
        timing=TIMING, iteration_microseconds=1.0
    )
    row = settings.build()
    _, result = _a_shot_with_two_passes(row, model, circuit)

    ticks = row.ticks_after_decode(result, 0, 0)

    belief_ticks = config.microseconds_to_ticks(result.iterations)
    evidence = result.cluster_evidence
    growth_ticks = TIMING.decode_ticks(evidence, 0, belief_ticks)
    first = evidence.earlier_passes[0]
    one_growth_ticks = TIMING.decode_ticks(first, 0, belief_ticks)
    assert ticks == belief_ticks + growth_ticks
    assert growth_ticks > one_growth_ticks


def _a_shot_with_two_passes(row, model, circuit) -> tuple:
    """(The first sampled shot whose pass two ran, the row's result on it)."""
    detection_events, _ = windows.sampled_shots(circuit, SHOTS, 7)
    for shot in detection_events:
        job = windows.job_for(model, shot)
        result = row.decode(job)
        if result.cluster_evidence.earlier_passes:
            return shot, result
    raise AssertionError("no sampled shot ran pass two")


def _lengths(graph) -> numpy.ndarray:
    lengths = [edge.length_half_ticks for edge in graph.edges]
    return numpy.asarray(lengths)
