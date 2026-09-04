"""The belief_matching row against beliefmatching's matching branch.

beliefmatching.BeliefMatching.decode
(.pydeps/beliefmatching/belief_matching.py:344-358) runs BP on the
hyperedge check matrix, projects the posteriors onto the matching edges
and matches with -log(posterior) weights; when BP converges it returns
BP's own correction instead. This test runs its matching branch on its
own matrices for every shot and asks decsim's row for the same
observable; the posterior clamp is the one line that may differ (1e-15
here, 1e-14 there).
"""

import numpy
import pymatching
from beliefmatching import BeliefMatching

import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.detector_error_model.fault_model_contracts as fault_models
from tests.decoders import windows

ROUNDS = 3
SHOTS = 100


def _reference_prediction(referee: BeliefMatching, syndrome):
    """Beliefmatching's matching branch, verbatim but for the clamp."""
    matrices = referee._matrices
    referee._bpd.decode(syndrome)
    llrs = referee._bpd.log_prob_ratios
    hyperedge_posteriors = 1 / (1 + numpy.exp(llrs))
    edge_posteriors = matrices.hyperedge_to_edge_matrix @ hyperedge_posteriors
    eps = 1e-14
    edge_posteriors[edge_posteriors > 1 - eps] = 1 - eps
    edge_posteriors[edge_posteriors < eps] = eps
    matching = pymatching.Matching.from_check_matrix(
        matrices.edge_check_matrix,
        weights=-numpy.log(edge_posteriors),
        faults_matrix=matrices.edge_observables_matrix,
        use_virtual_boundary_node=True,
    )
    return matching.decode(syndrome)


def test_the_row_predicts_what_beliefmatchings_matching_branch_predicts():
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.LINKED_FAULT_MODELS_REQUIRED
    )
    detection_events, _ = windows.sampled_shots(circuit, SHOTS, 7)
    dem = circuit.detector_error_model(decompose_errors=True)
    referee = BeliefMatching(dem, max_bp_iters=30, bp_method="product_sum")
    row = belief_matching.BeliefMatchingDecoder(
        max_iterations=30, belief_propagation_method="product_sum"
    )
    disagreements = []
    for index, shot in enumerate(detection_events):
        syndrome = numpy.asarray(shot, dtype=numpy.uint8)
        expected = _reference_prediction(referee, syndrome)
        result = row.decode(windows.job_for(model, shot))
        if result.logical_observables != (int(expected[0]),):
            disagreements.append(index)
    assert disagreements == []
