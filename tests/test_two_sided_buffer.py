"""Two-sided-buffer (parallel A/B) real decoding -- issue #3a.

Before the fix, a window with a LEADING buffer (look-ahead rounds BEFORE its commit region)
had its leading-buffer rows stripped of any boundary edge, so a handed-forward artificial
defect landing there had nowhere to match -> pymatching raised on ~40% of shots. The fix
(decsim/adapters/window_error_models.py): rows start at buffer_lo, ownership is a commit-range
test, and already-committed faults reaching a leading-buffer row are added as UNOWNED boundary
edges (Skoric arXiv:2209.08552: an A window's past time boundary is rough).

Acceptance:
- ownership is exactly-once across the A/B plan (every fault committed by one window);
- a leading-buffer window genuinely has an open past boundary (a weight-1 boundary column on a
  leading-buffer row);
- A/B decodes through the FULL engine with NO crashes and TRACKS the global decode (the engine,
  not the single-forward-pass decode_windowed, is the A/B reference: it hands each B window both
  neighbouring A windows' defects);
- the certified sliding (trailing-only) path stays BIT-IDENTICAL to global -- no regression.

R=30 at d=3 is deliberate: it yields genuine leading-buffer windows. R=12 degenerates to one A
window + tail (no leading buffer) and would falsely pass. Requires stim + pymatching."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pymatching = pytest.importorskip("pymatching")

from decsim.message import Operation
from decsim.adapters.stim_device import StimDevice
from decsim.adapters.window_error_models import (build_window_error_models, decode_windowed,
                                                 detector_error_model_to_faults)
from decsim.mwpm_decoder import PyMatchingDecoder, matching_window_decoder
from decsim.schemes import ParallelWindowScheme, SlidingWindowScheme
from decsim.codes import SurfaceCodeModel
from decsim.planner import FixedRounds
from decsim.sampling import logical_error_rate

D, R, P = 3, 30, 0.003


class _ZeroLatency:
    def latency(self, job):
        return 1


def _circuit():
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=D, rounds=R,
        after_clifford_depolarization=P, after_reset_flip_probability=P,
        before_measure_flip_probability=P, before_round_data_depolarization=P)


def _ab_plan():
    raw = ParallelWindowScheme().plan_windows(0, R, SurfaceCodeModel(d=D))
    return [(bl, cl, ch, min(bh, R)) for (bl, cl, ch, bh) in raw]      # clamp like the cluster


def _folded(circuit):
    return {det: min(int(c[-1]) + 1, R) for det, c in circuit.get_detector_coordinates().items()}


def test_ab_plan_has_genuine_leading_buffers():
    plan = _ab_plan()
    leading = [w for w in plan if w[0] < w[1]]                          # buffer_lo < commit_lo
    assert leading, f"R={R} produced no leading-buffer windows -- test would be vacuous"


def test_ab_ownership_exactly_once_and_full_coverage():
    circ = _circuit()
    models = build_window_error_models(circ, _ab_plan(), detector_rounds=_folded(circ))
    n_faults = len(detector_error_model_to_faults(
        circ.detector_error_model(decompose_errors=True))[0])
    total_owned = sum(int(m.owned.sum()) for m in models)
    assert total_owned == n_faults                                     # each fault owned once
    # decode_windowed raises "artificial defects were never consumed" if the plan does not
    # cover the full stream -- run a few shots to exercise it
    inner = matching_window_decoder()
    dets, _ = circ.compile_detector_sampler(seed=1).sample(20, separate_observables=True)
    for s in range(20):
        decode_windowed(models, dets[s], inner)                        # must not raise


def test_ab_leading_buffer_window_has_open_past_boundary():
    circ = _circuit()
    plan = _ab_plan()
    folded = _folded(circ)
    models = build_window_error_models(circ, plan, detector_rounds=folded)
    found = False
    for (buffer_lo, commit_lo, _ch, _bh), m in zip(plan, models):
        if buffer_lo >= commit_lo:
            continue                                                   # no leading buffer
        lead_rows = {i for i, det in enumerate(m.detector_ids) if folded[det] < commit_lo}
        # a rough past-boundary edge = a weight-1 column whose single detector is a leading row
        for j in range(m.check.shape[1]):
            nz = np.nonzero(m.check[:, j])[0]
            if len(nz) == 1 and nz[0] in lead_rows:
                found = True
                break
    assert found, "no leading-buffer window has an open (weight-1) past-boundary edge"


def test_ab_engine_tracks_global_no_crash():
    circ = _circuit()
    global_m = pymatching.Matching.from_detector_error_model(
        circ.detector_error_model(decompose_errors=True))
    ops = [Operation(id=1, name="mem", qubits=(0,), clifford=True, circuit=circ)]
    g = {"agree": 0}

    def on_shot(s, cluster, dev):
        pe = int(cluster.op_results[1])
        pg = int(global_m.decode(dev._dets[1])[0])
        g["agree"] += int(pe == pg)

    shots = 300
    out = logical_error_rate(ops, shots=shots, device=StimDevice(seed=7), on_shot=on_shot,
                             num_units=4, d=D, rounds_policy=FixedRounds(R),
                             code=SurfaceCodeModel(d=D), scheme=ParallelWindowScheme(),
                             decoder=PyMatchingDecoder(_ZeroLatency()))
    # no crash over every shot (the pre-fix failure mode), and the engine tracks global closely
    assert out["shots"] == shots
    assert g["agree"] / shots >= 0.94, g["agree"] / shots


def test_sliding_still_bit_identical_to_global():
    """Regression guard: the builder change must leave the certified trailing-only path exactly
    equal to global (it owns no leading-buffer columns)."""
    circ = _circuit()
    global_m = pymatching.Matching.from_detector_error_model(
        circ.detector_error_model(decompose_errors=True))
    ops = [Operation(id=1, name="mem", qubits=(0,), clifford=True, circuit=circ)]
    mismatches = {"n": 0}

    def on_shot(s, cluster, dev):
        pe = int(cluster.op_results[1])
        pg = int(global_m.decode(dev._dets[1])[0])
        mismatches["n"] += int(pe != pg)

    logical_error_rate(ops, shots=200, device=StimDevice(seed=7), on_shot=on_shot,
                       num_units=4, d=D, rounds_policy=FixedRounds(R), code=SurfaceCodeModel(d=D),
                       scheme=SlidingWindowScheme(), decoder=PyMatchingDecoder(_ZeroLatency()))
    assert mismatches["n"] == 0                                        # bit-identical to global
