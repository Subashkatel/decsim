"""The belief_find row against beliefmatching's belief propagation.

beliefmatching.BeliefMatching (.pydeps/beliefmatching/belief_matching.py
lines 344-358) runs BP on its hyperedge matrices, maps each hyperedge's
posterior onto the edges of its decomposition (hyperedge_to_edge_matrix)
and clamps them at 1e-14; Higgott et al. 2203.04948 Sec. III.A grow
weighted union-find on those edge posteriors to make belief-find. Each
of the row's edges must take the posterior the library gives the edge
of the same detectors. Where BP converges, the library returns BP's own
correction (Sec. I), and the row's growth must give the same logical.
"""

import beliefmatching
import numpy
import pytest

import decsim.config as config
import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.decoders.union_find.belief_find as belief_find
import decsim.decoders.union_find.window_decoder as window_decoder
import decsim.records.fault_model_contracts as fault_models
import tests.decoders.test_union_find_cycle_count as cycle_counts
from tests.decoders import windows

LINKED = fault_models.LINKED_FAULT_MODELS_REQUIRED
GRAPHLIKE = fault_models.FaultRepresentation.GRAPHLIKE
ROUNDS = 3
SHOTS = 200
MAX_ITERATIONS = 30
# beliefmatching's posterior clamp; the row clamps at 1e-15
LIBRARY_FLOOR = 1e-14
LIBRARY_CEILING = 1 - LIBRARY_FLOOR
SETTINGS = belief_find.BeliefFindDecoder.Settings(
    timing=cycle_counts.TRACED_CYCLE_COUNT, max_iterations=MAX_ITERATIONS
)


def test_each_edge_takes_the_posterior_beliefmatching_gives_it_property():
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(circuit, ROUNDS, LINKED)
    faults = model.require_faults(GRAPHLIKE)
    row = SETTINGS.build()
    backend = row.compile(faults, model)
    referee = _referee(circuit)
    library_columns = _library_columns_of_edges(referee, backend.graph, model)
    fault_indices = [edge.fault_index for edge in backend.graph.edges]
    detection_events, _ = windows.sampled_shots(circuit, SHOTS, 7)
    found = []
    expected = []
    for shot in detection_events:
        syndrome = windows.row_syndrome(model, shot)
        backend.belief_propagation.decode(syndrome)
        posteriors = belief_matching.edge_posteriors(
            backend.belief_propagation, backend.edge_from_hyperedge
        )
        found.append(posteriors[fault_indices])
        library_posteriors = _library_edge_posteriors(referee, shot)
        expected.append(library_posteriors[library_columns])
    found_clamped = numpy.clip(found, LIBRARY_FLOOR, LIBRARY_CEILING)
    numpy.testing.assert_allclose(found_clamped, expected, rtol=1e-6)


def test_where_bp_converges_the_growth_gives_bps_own_logical_property():
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(circuit, ROUNDS, LINKED)
    referee = _referee(circuit)
    row = SETTINGS.build()
    detection_events, _ = windows.sampled_shots(circuit, SHOTS, 7)
    found = []
    expected = []
    converged = []
    for shot in detection_events:
        syndrome = numpy.asarray(shot, dtype=numpy.uint8)
        prediction = referee.decode(syndrome)
        expected.append((int(prediction[0]),))
        converged.append(bool(referee._bpd.converge))
        job = windows.job_for(model, shot)
        result = row.decode(job)
        found.append(result.logical_observables)
    pairs = zip(found, expected, converged, strict=True)
    on_converged = [(row, library) for row, library, is_on in pairs if is_on]
    row_answers = [answer for answer, _ in on_converged]
    library_answers = [answer for _, answer in on_converged]
    assert row_answers == library_answers
    # BP converges on most of these shots and not on all of them
    assert sum(converged) > SHOTS // 2
    assert not all(converged)


def test_a_decode_is_charged_its_iterations_then_its_growth():
    """One microsecond an iteration, then the growth's traced cycles."""
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(circuit, ROUNDS, LINKED)
    timed = belief_find.BeliefFindDecoder.Settings(
        timing=cycle_counts.TRACED_CYCLE_COUNT, iteration_microseconds=1.0
    )
    row = timed.build()
    detection_events, _ = windows.sampled_shots(circuit, SHOTS, 11)
    has_events = detection_events.any(axis=1)
    first_with_events = numpy.argmax(has_events)
    job = windows.job_for(model, detection_events[first_with_events])
    result = row.decode(job)

    ticks = row.ticks_after_decode(result, 0, 0)

    belief_ticks = config.microseconds_to_ticks(result.iterations)
    growth_ticks = cycle_counts.TRACED_CYCLE_COUNT.decode_ticks(
        result.cluster_evidence, 0, belief_ticks
    )
    assert result.iterations > 0
    assert ticks == belief_ticks + growth_ticks


def test_a_negative_iteration_time_is_refused():
    with pytest.raises(ValueError) as refused:
        belief_find.BeliefFindDecoder.Settings(
            timing=cycle_counts.TRACED_CYCLE_COUNT,
            iteration_microseconds=-1.0,
        )

    message = str(refused.value)
    assert "iteration_microseconds must be a finite nonnegative" in message


def _referee(circuit) -> beliefmatching.BeliefMatching:
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    return beliefmatching.BeliefMatching(
        detector_error_model,
        max_bp_iters=MAX_ITERATIONS,
        bp_method="product_sum",
    )


def _library_edge_posteriors(referee, shot) -> numpy.ndarray:
    """The library's edge posteriors for one shot (lines 348-353)."""
    syndrome = numpy.asarray(shot, dtype=numpy.uint8)
    referee._bpd.decode(syndrome)
    hyperedge_posteriors = 1 / (1 + numpy.exp(referee._bpd.log_prob_ratios))
    matrices = referee._matrices
    edge_posteriors = matrices.hyperedge_to_edge_matrix @ hyperedge_posteriors
    return numpy.clip(edge_posteriors, LIBRARY_FLOOR, LIBRARY_CEILING)


def _library_columns_of_edges(referee, graph, model) -> list:
    """The library's edge column of each of the graph's edges, by detectors."""
    edge_check = referee._matrices.edge_check_matrix.tocsc()
    column_by_detectors = {}
    for column in range(edge_check.shape[1]):
        start = edge_check.indptr[column]
        end = edge_check.indptr[column + 1]
        detectors = edge_check.indices[start:end]
        detector_list = detectors.tolist()
        detector_set = frozenset(detector_list)
        column_by_detectors[detector_set] = column
    detector_ids = list(model.detector_ids)
    columns = []
    for edge in graph.edges:
        rows = {edge.detector_a, edge.detector_b} - {window_decoder.BOUNDARY}
        ends = frozenset(detector_ids[row] for row in rows)
        columns.append(column_by_detectors[ends])
    return columns
