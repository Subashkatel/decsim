"""The cluster gap, read off the Union-Find decode's own growth.

Meister et al. 2405.07433 Algorithm 2 (lines 518-536) takes the weighted
decoding graph and the radii the growth reached and returns the shortest
closed walk of odd logical parity. The row here reads exactly that off
the result the Union-Find decoder already returned, so the confidence is
the decoder's own and no second decode runs. Meister's Theorem 10 says
the walk is the complementary gap in the uniform repetition setting, so
the calibration test holds the two signals against each other on the
same shots: they agree to within the growth's weight step.
"""

import statistics

import decsim.confidence.cluster as cluster
import decsim.confidence.complementary as complementary
import decsim.decoders.minimum_weight_perfect_matching.decoder as mwpm
import decsim.decoders.union_find.decoder as union_find
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.ports as ports
import decsim.records.decoder_evidence as evidence_records
import decsim.records.decoding as decoding_records
from tests.decoders import windows

ROUNDS = 4
SHOTS = 30
SEED = 7


def _window_model():
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    model = windows.whole_circuit_window(circuit, ROUNDS, requirement)
    return circuit, model


def test_the_row_declares_one_decode_and_the_growth_it_reads():
    """The signal's requirement and the Union-Find row's declaration meet."""
    signal = cluster.ClusterGap()
    assert isinstance(signal, ports.ConfidenceSignal)
    assert signal.forced_logical_classes == ()
    required = signal.decoder_evidence_requirement
    assert required == decoding_records.CLUSTER_GROWTH_EVIDENCE
    row = union_find.UnionFindDecoder()
    assert not required - row.decoder_evidence
    matching = mwpm.PyMatchingDecoder()
    assert required - matching.decoder_evidence == required


def test_a_decode_without_growth_reports_no_gap():
    """The fail-safe: a job with no window model grows nothing."""
    signal = cluster.ClusterGap()
    solve = decoding_records.DecodeResult(1, 0)
    computation = signal.compute((solve,))
    assert computation.soft_output is None
    assert computation.ticks == 0


def test_the_gap_is_read_off_the_decode_that_produced_the_correction():
    """One decode answers the window and carries the growth the gap walks."""
    _built_circuit, model = _window_model()
    row = union_find.UnionFindDecoder()
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    events, _observables = windows.sampled_shots(circuit, 1, SEED)
    job = windows.job_for(model, events[0])
    result = row.decode(job)
    evidence = result.cluster_evidence
    selected = list(evidence.selected_faults)
    assert selected == list(result.correction)
    graph = evidence.graph
    assert graph.weight_step == evidence_records.DEFAULT_WEIGHT_STEP
    signal = cluster.ClusterGap()
    computation = signal.compute((result,))
    soft_output = computation.soft_output
    assert soft_output.source is signal.source
    assert soft_output.source.gap_units == "log_likelihood_weight"
    assert soft_output.gap > 0.0


def test_the_cluster_gap_and_the_complementary_gap_agree_on_one_window():
    """Meister's Theorem 10, on the same shots and the same window.

    The cluster gap walks Union-Find's own growth and the complementary
    gap subtracts two pinned matchings, so the two numbers come from
    different decoders; the growth is quantized to the weight step, and
    that is the size of the disagreement.
    """
    circuit, model = _window_model()
    hard_row = union_find.UnionFindDecoder()
    matching_row = mwpm.PyMatchingDecoder()
    cluster_signal = cluster.ClusterGap()
    complementary_signal = complementary.ComplementaryGap()
    events, _observables = windows.sampled_shots(circuit, SHOTS, SEED)
    differences = []
    for shot in events:
        hard_job = windows.job_for(model, shot)
        hard_result = hard_row.decode(hard_job)
        cluster_computation = cluster_signal.compute((hard_result,))
        cluster_output = cluster_computation.soft_output
        solves = []
        for forced_class in complementary_signal.forced_logical_classes:
            forced_job = windows.job_for(model, shot)
            forced_job.forced_logical_class = forced_class
            solve = matching_row.decode(forced_job)
            solves.append(solve)
        matching_computation = complementary_signal.compute(tuple(solves))
        matching_output = matching_computation.soft_output
        difference = cluster_output.gap - matching_output.gap
        differences.append(abs(difference))
    weight_step = evidence_records.DEFAULT_WEIGHT_STEP
    assert statistics.median(differences) <= weight_step
    assert max(differences) <= 5 * weight_step
