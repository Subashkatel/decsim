"""Windowed belief-matching (Higgott & Gidney, arXiv:2203.04948) as a decode_windowed
inner decoder. Validated three ways:
  1. turning belief_matching on only ADDS hyperedge fields (edge model byte-identical);
  2. our single-window belief-matching agrees with the canonical `beliefmatching` package
     (faithfulness to the reference algorithm);
  3. windowed belief-matching agrees with single-window belief-matching (it survives
     windowing -- the artificial-defect boundary handoff doesn't break BP).

Asserts use per-shot AGREEMENT (low variance) rather than the logical error rate (noisy at
small N), so the tests are fast and deterministic-ish at d=3. The fuller d=3/5/7 LER study
lives in experiments/windowed-belief-matching/.
"""
import math

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pytest.importorskip("pymatching")
pytest.importorskip("ldpc")

from decsim.detector_error_model import (  # noqa: E402
    build_window_error_models, decode_windowed)
from decsim.belief_matching_decoder import belief_matching_window_decoder  # noqa: E402
from decsim.run_spec import RunSpec, simulate


def _memory_circuit(d, rounds, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=p, before_round_data_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p)


def _sliding_plan(n_rounds, commit, buffer):
    window_count = max(1, math.ceil(n_rounds / commit))
    plan = []
    for k in range(window_count):
        hi = min((k + 1) * commit, n_rounds)
        plan.append((k * commit + 1, hi, min(hi + buffer, n_rounds)))
    return plan


def _windowed_preds(circ, plan, shots):
    models = build_window_error_models(circ, plan, belief_matching=True)
    dec = belief_matching_window_decoder()
    return np.array([decode_windowed(models, shots[i], dec)[0] for i in range(len(shots))],
                    dtype=np.uint8)


def test_bm_flag_leaves_edge_model_identical():
    """belief_matching=True must ONLY add the hyperedge fields; the decomposed edge model
    (check / owned / obs / future_flips) is byte-identical to the default path."""
    circ = _memory_circuit(3, 9, 3e-3)
    plan = _sliding_plan(9, 3, 3)
    base = build_window_error_models(circ, plan)
    bm = build_window_error_models(circ, plan, belief_matching=True)
    assert len(base) == len(bm)
    for a, b in zip(base, bm):
        assert a.detector_ids == b.detector_ids
        assert np.array_equal(a.check, b.check)
        assert np.array_equal(a.owned, b.owned)
        assert np.array_equal(a.obs, b.obs)
        assert a.future_flips == b.future_flips
        assert a.h_check is None and b.h_check is not None     # only the BM build has them


def test_decsim_bm_faithful_to_package():
    """Single-window decsim belief-matching agrees with the canonical beliefmatching
    package on the vast majority of shots (the few-% residual is the converge early-exit
    variant, which leaves the LER unchanged within error bars)."""
    bmpkg = pytest.importorskip("beliefmatching")
    d, R, p, N = 3, 9, 3e-3, 600
    circ = _memory_circuit(d, R, p)
    shots, obs = circ.compile_detector_sampler().sample(N, separate_observables=True)
    shots = shots.astype(np.uint8)
    ours = _windowed_preds(circ, [(1, R, R)], shots)
    pkg = bmpkg.BeliefMatching(circ.detector_error_model(decompose_errors=True),
                               max_bp_iters=30).decode_batch(shots)[:, 0].astype(np.uint8)
    disagree = int((ours != pkg).sum())
    assert disagree / N < 0.03, f"{disagree}/{N} disagreements with the beliefmatching package"


def test_windowed_bm_tracks_global():
    """Windowed belief-matching agrees with single-window (global) belief-matching on the
    vast majority of shots -- belief-matching survives windowing."""
    d, R, p, N = 3, 12, 3e-3, 600
    circ = _memory_circuit(d, R, p)
    shots, obs = circ.compile_detector_sampler().sample(N, separate_observables=True)
    shots = shots.astype(np.uint8)
    glob = _windowed_preds(circ, [(1, R, R)], shots)
    win = _windowed_preds(circ, _sliding_plan(R, d, d), shots)
    disagree = int((glob != win).sum())
    assert disagree / N < 0.03, f"{disagree}/{N} windowed-vs-global disagreements"


def test_engine_belief_matching_matches_offline():
    """Full stack: belief-matching runs THROUGH the DES (StimDevice -> cluster windows ->
    BeliefMatchingDecoder -> orchestrator) under the sliding scheme, and the engine's decoded
    logical value equals the offline windowed-belief-matching reference per shot. Proves the
    runtime decoder drops into the scheme machinery: pick a scheme, route to belief-matching,
    end to end -- the cluster auto-builds hyperedge DEMs via needs_hyperedges."""
    from decsim.message import Operation
    from decsim.controllers import ModularController, LinkModel
    from decsim.adapters.stim_device import StimDevice
    from decsim.detector_error_model import build_window_error_models
    from decsim.belief_matching_decoder import (BeliefMatchingDecoder,
                                                belief_matching_window_decoder)
    from decsim.schemes import SlidingWindowScheme
    from decsim.codes import SurfaceCodeModel
    from decsim.planner import FixedRounds

    D, R, P = 3, 12, 3e-3
    circuit = _memory_circuit(D, R, P)

    class _ZeroLat:
        def latency(self, job):
            return 1

    def _zero_links(engine):
        return ModularController(engine, links=LinkModel(qc=0, cd=0, dd=0, do=0, oc=0, cq=0), log_syndromes=False)

    coords = circuit.get_detector_coordinates()
    folded = {det: min(int(c[-1]) + 1, R) for det, c in coords.items()}
    plan = [(lo, hi, min(b, R)) for lo, hi, b in
            SlidingWindowScheme().plan_windows(0, R, SurfaceCodeModel(d=D))]
    ref_models = build_window_error_models(circuit, plan, detector_rounds=folded,
                                           belief_matching=True)
    ref_inner = belief_matching_window_decoder()

    for s in range(30):
        device = StimDevice()
        op = Operation(id=1, name="memory", qubits=(0,), clifford=True, circuit=circuit)
        res = simulate(RunSpec(
                  ops=[op],
                  num_units=4,
                  rounds_policy=FixedRounds(R),
                  code=SurfaceCodeModel(d=D),
                  scheme=SlidingWindowScheme(),
                  device=device,
                  decoder=BeliefMatchingDecoder(_ZeroLat()),
                  make_controller=_zero_links,
                  seed=17 + s,
              ), verbose=False)
        pred_engine = res.cluster.op_results[1]
        pred_offline = int(decode_windowed(ref_models, device._dets[1], ref_inner)[0])
        assert pred_engine == (pred_offline,), \
            f"shot {s}: engine {pred_engine} != offline {pred_offline}"
