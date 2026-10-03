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

import math
import statistics

import decsim.confidence.cluster as cluster
import decsim.confidence.complementary as complementary
import decsim.decoders.union_find.cycle_count as cycle_count_module
import decsim.decoders.union_find.decoder as union_find
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoder_evidence as evidence_records
import decsim.records.decoding as decoding_records
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)
from tests.confidence import independent_cluster_gap
from tests.decoders import windows

ROUNDS = 4
SHOTS = 30
SEED = 7
HOST_TIME = cycle_count_module.HostMeasuredTime()
HOST_TIMED_UNION_FIND = union_find.UnionFindDecoder.Settings(timing=HOST_TIME)


def _window_model():
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    model = windows.whole_circuit_window(circuit, ROUNDS, requirement)
    return circuit, model


def test_a_decode_without_growth_reports_no_gap():
    """The fail-safe: a job with no window model grows nothing."""
    signal = cluster.ClusterGap()
    solve = decoding_records.DecodeResult(1, 0)
    computation = signal.compute((solve,))
    assert computation.soft_output is None
    assert computation.ticks == 0


def test_the_cluster_and_complementary_gaps_agree_on_one_window_property():
    """Meister's Theorem 10, on the same shots and the same window.

    The cluster gap walks Union-Find's own growth and the complementary
    gap subtracts two pinned matchings, so the two numbers come from
    different decoders; the growth is quantized to the weight step, and
    that is the size of the disagreement.
    """
    circuit, model = _window_model()
    hard_row = union_find.UnionFindDecoder(HOST_TIMED_UNION_FIND)
    matching_row = minimum_weight_perfect_matching.PyMatchingDecoder()
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


def test_the_gap_walks_against_an_oracle_built_from_the_paper_alone_property():
    """The per-node Dijkstra with a cutoff decsim runs, against scipy.

    independent_cluster_gap.py is an independent reading of Meister et
    al. 2405.07433 Definitions 1 and 9 and Algorithm 1 line 6, written
    from the paper text and not from decsim's code, and it runs a
    different algorithm: one all-sources scipy Dijkstra over the whole
    parity-doubled graph, against decsim's per-node search with a
    running cutoff. The two must reach the same walk on every shot.
    Slow: it decodes and walks a handful of real shots.
    """
    circuit, model = _window_model()
    row = union_find.UnionFindDecoder(HOST_TIMED_UNION_FIND)
    signal = cluster.ClusterGap()
    events, _observables = windows.sampled_shots(circuit, 8, SEED)
    walked = 0
    for shot in events:
        job = windows.job_for(model, shot)
        result = row.decode(job)
        evidence = result.cluster_evidence
        computation = signal.compute((result,))
        outside = _oracle_gap(evidence)
        inside = computation.soft_output.gap
        assert _same_gap(inside, outside), shot
        walked += 1
    assert walked == 8


def _oracle_gap(evidence) -> float:
    """The independent oracle's gap for one decode's growth, in nats."""
    half_ticks = independent_cluster_gap.shortest_odd_closed_walk(
        evidence.graph, evidence.edge_intervals
    )
    step = evidence.graph.weight_step
    return independent_cluster_gap.half_ticks_to_nats(half_ticks, step)


def _same_gap(inside: float, outside: float) -> bool:
    """The two walks agree, both finite or both unreachable."""
    if math.isinf(inside):
        return math.isinf(outside)
    return math.isclose(inside, outside, abs_tol=1e-9)
