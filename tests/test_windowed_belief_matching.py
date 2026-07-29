"""Windowed belief-matching (Higgott & Gidney, arXiv:2203.04948) as a decode_windowed
inner decoder. Validated three ways:
  1. requesting linked models only ADDS physical faults and their graphlike link;
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
    FaultRepresentation,
    GRAPHLIKE_FAULT_MODEL_REQUIRED,
    LINKED_FAULT_MODELS_REQUIRED,
    build_window_error_models,
    decode_windowed,
)
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
    models = build_window_error_models(
        circ,
        plan,
        fault_model_requirement=LINKED_FAULT_MODELS_REQUIRED,
        fault_exclusion_ranges=(),
    )
    dec = belief_matching_window_decoder()
    return np.array(
        [
            decode_windowed(
                models,
                shots[i],
                dec,
                selected_fault_representation=FaultRepresentation.GRAPHLIKE,
            )[0]
            for i in range(len(shots))
        ],
        dtype=np.uint8,
    )


def test_linked_requirement_leaves_graphlike_model_identical():
    """Requesting belief-matching inputs only adds physical faults and their link."""
    circ = _memory_circuit(3, 9, 3e-3)
    plan = _sliding_plan(9, 3, 3)
    base = build_window_error_models(
        circ,
        plan,
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
    )
    bm = build_window_error_models(
        circ,
        plan,
        fault_model_requirement=LINKED_FAULT_MODELS_REQUIRED,
        fault_exclusion_ranges=(),
    )
    assert len(base) == len(bm)
    for a, b in zip(base, bm):
        a_graphlike = a.require_faults(FaultRepresentation.GRAPHLIKE)
        b_graphlike = b.require_faults(FaultRepresentation.GRAPHLIKE)
        assert a.detector_ids == b.detector_ids
        assert np.array_equal(a_graphlike.check, b_graphlike.check)
        assert np.array_equal(a_graphlike.owned, b_graphlike.owned)
        assert np.array_equal(
            a_graphlike.observables,
            b_graphlike.observables,
        )
        assert a_graphlike.future_flips == b_graphlike.future_flips
        assert a.physical_faults is None
        assert a.physical_to_graphlike_detector_projection is None
        assert b.physical_faults is not None
        assert b.physical_to_graphlike_detector_projection is not None


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
    end to end -- the cluster builds both fault domains and their link explicitly."""
    from decsim.message import Operation
    from conftest import fixed_latency_link_config
    from decsim.controllers import ModularController
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

    def _zero_links(engine, links):
        return ModularController(engine, links=links, log_syndromes=False)

    coords = circuit.get_detector_coordinates()
    folded = {det: min(int(c[-1]) + 1, R) for det, c in coords.items()}
    plan = [
        (window.commit_lo, window.commit_hi, min(window.buffer_hi, R))
        for window in SlidingWindowScheme().plan_operation(
            0,
            R,
            commit_round_count=D,
            buffer_round_count=D,
        ).windows
    ]
    ref_models = build_window_error_models(
        circuit,
        plan,
        detector_rounds=folded,
        fault_model_requirement=LINKED_FAULT_MODELS_REQUIRED,
        fault_exclusion_ranges=(),
    )
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
                  links=fixed_latency_link_config(),
                  make_controller=_zero_links,
                  seed=17 + s,
              ), verbose=False)
        pred_engine = res.window_manager.op_results[1]
        pred_offline = int(
            decode_windowed(
                ref_models,
                device._dets[1],
                ref_inner,
                selected_fault_representation=FaultRepresentation.GRAPHLIKE,
            )[0]
        )
        assert pred_engine == (pred_offline,), \
            f"shot {s}: engine {pred_engine} != offline {pred_offline}"
