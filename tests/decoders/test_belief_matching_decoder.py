"""The belief_matching row against beliefmatching's matching branch.

beliefmatching.BeliefMatching.decode
(.pydeps/beliefmatching/belief_matching.py:344-358) runs BP on the
hyperedge check matrix, projects the posteriors onto the matching edges
and matches with -log(posterior) weights; when BP converges it returns
BP's own correction instead. This test runs its matching branch on its
own matrices for every shot and asks decsim's row for the same
observable; the posterior clamp is the one line that may differ (1e-15
here, 1e-14 there). Its matrices hold one column per distinct hyperedge
(lines 104-136), and 300 shots reach a syndrome where that merging
decides the answer (shot 238).

With a stated time, a decode is charged its own BP iterations, the
count beliefmatching's BP reaches on the same syndrome, at the time per
iteration, plus the matching's time.
"""

import math

import beliefmatching
import numpy
import pymatching
import pytest

import decsim.config as config
import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.decoders.decoder as decoder_module
import decsim.records.fault_model_contracts as fault_models
from tests.decoders import windows

ROUNDS = 3
SHOTS = 300
CLAMP = 1e-14
TIMED_SHOTS = 50
MAX_ITERATIONS = 20


def _reference_prediction(referee: beliefmatching.BeliefMatching, syndrome):
    """Beliefmatching's matching branch, verbatim but for the clamp."""
    matrices = referee._matrices
    referee._bpd.decode(syndrome)
    llrs = referee._bpd.log_prob_ratios
    hyperedge_posteriors = 1 / (1 + numpy.exp(llrs))
    edge_posteriors = matrices.hyperedge_to_edge_matrix @ hyperedge_posteriors
    edge_posteriors[edge_posteriors > 1 - CLAMP] = 1 - CLAMP
    edge_posteriors[edge_posteriors < CLAMP] = CLAMP
    edge_weights = -numpy.log(edge_posteriors)
    matching = pymatching.Matching.from_check_matrix(
        matrices.edge_check_matrix,
        weights=edge_weights,
        faults_matrix=matrices.edge_observables_matrix,
        use_virtual_boundary_node=True,
    )
    return matching.decode(syndrome)


def test_the_row_predicts_what_beliefmatchings_matching_branch_property():
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.LINKED_FAULT_MODELS_REQUIRED
    )
    detection_events, _ = windows.sampled_shots(circuit, SHOTS, 7)
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    referee = beliefmatching.BeliefMatching(
        detector_error_model, max_bp_iters=30, bp_method="product_sum"
    )
    row = belief_matching.BeliefMatchingDecoder(
        max_iterations=30, belief_propagation_method="product_sum"
    )
    expected_observables = []
    decoded_observables = []
    for shot in detection_events:
        syndrome = numpy.asarray(shot, dtype=numpy.uint8)
        expected = _reference_prediction(referee, syndrome)
        prediction = int(expected[0])
        expected_observables.append((prediction,))
        job = windows.job_for(model, shot)
        result = row.decode(job)
        decoded_observables.append(result.logical_observables)
    assert decoded_observables == expected_observables


def test_a_stated_time_charges_each_decodes_own_iterations_property():
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.LINKED_FAULT_MODELS_REQUIRED
    )
    detection_events, _ = windows.sampled_shots(circuit, TIMED_SHOTS, 7)
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    referee = beliefmatching.BeliefMatching(
        detector_error_model,
        max_bp_iters=MAX_ITERATIONS,
        bp_method="product_sum",
    )
    timing = belief_matching.IterationTime(
        iteration_microseconds=1.0, matching_microseconds=1.0
    )
    settings = belief_matching.BeliefMatchingDecoder.Settings(
        max_iterations=MAX_ITERATIONS, timing=timing
    )
    row = settings.build()
    expected_ticks = []
    charged_ticks = []
    for shot in detection_events:
        syndrome = numpy.asarray(shot, dtype=numpy.uint8)
        referee._bpd.decode(syndrome)
        microseconds = referee._bpd.iter * 1.0 + 1.0
        expected = config.microseconds_to_ticks(microseconds)
        expected_ticks.append(expected)
        job = windows.job_for(model, shot)
        _result, ticks = decoder_module.timed_decode(row, job, 0)
        charged_ticks.append(ticks)
    assert charged_ticks == expected_ticks


@pytest.mark.parametrize("microseconds", [-1.0, math.inf, math.nan, True])
def test_a_stated_time_that_is_no_duration_is_refused(microseconds):
    with pytest.raises(ValueError, match="matching_microseconds"):
        belief_matching.IterationTime(
            iteration_microseconds=1.0, matching_microseconds=microseconds
        )
