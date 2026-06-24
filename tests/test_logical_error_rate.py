"""Logical error rate: the finished LogicalErrorRate metric (per-shot verdict) + the
multi-shot harness decsim.sampling.logical_error_rate with a Wilson confidence interval.

Acceptance (issues #2 / #4):
- the metric reports a correct per-shot pass/fail against the true observable;
- the Wilson interval is well-behaved at small error counts (stays in [0,1], positive lower
  bound once errors > 0);
- the SEED-ROBUST anchor (Skoric App C / test_full_stack): at d=3/R=12 the windowed (engine)
  decode is bit-identical to the global decode per shot, so the harness's engine LER EQUALS
  the global LER and agreement is 100% -- the right pin, not a fragile absolute LER number.

Requires stim + pymatching."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pymatching = pytest.importorskip("pymatching")

from decsim.stimcircuits import NoiseModel
from decsim.message import Operation
from decsim.wiring import build_and_run
from decsim.adapters.stim_device import StimDevice
from decsim.mwpm_decoder import PyMatchingDecoder
from decsim.schemes import SlidingWindowScheme
from decsim.codes import SurfaceCodeModel
from decsim.planner import FixedRounds
from decsim.sampling import logical_error_rate, wilson_interval
from decsim.metrics import LogicalErrorRate

D, R, P = 3, 12, 0.003


class _ZeroLatency:
    def latency(self, job):
        return 1


def _circuit():
    return NoiseModel.circuit_level(P).circuit(distance=D, rounds=R)


def _ops(circuit):
    return [Operation(id=1, name="mem", qubits=(0,), clifford=True, circuit=circuit)]


def _common():
    return dict(num_units=4, d=D, rounds_policy=FixedRounds(R), code=SurfaceCodeModel(d=D),
                scheme=SlidingWindowScheme(), decoder=PyMatchingDecoder(_ZeroLatency()))


def test_wilson_interval_small_counts():
    p, lo, hi = wilson_interval(0, 100)
    assert p == 0.0 and lo == 0.0 and 0.0 < hi < 0.06       # one-sided, never leaves [0,1]
    p, lo, hi = wilson_interval(3, 300)
    assert 0.0 < lo < p < hi < 1.0                          # contains estimate, positive lower
    # Wilson is asymmetric near 0, unlike the naive symmetric bar
    assert (hi - p) > (p - lo)


def test_metric_verdict_matches_truth():
    circ = _circuit()
    device = StimDevice(seed=5)
    res = build_and_run(ops=_ops(circ), device=device, verbose=False, **_common())
    v = LogicalErrorRate(res["cluster"], device).verdicts()
    assert set(v) == {1}
    truth = int(device._truth[1][0])
    pred = int(res["cluster"].op_results[1])
    assert v[1] == {"predicted": pred, "truth": truth, "error": int(pred != truth)}


def test_timing_only_op_has_no_verdict():
    """An op with no circuit decodes timing-only -> no logical value -> no verdict (not a
    false 'pass')."""
    op = Operation(id=1, name="timing", qubits=(0,), clifford=True)
    res = build_and_run(ops=[op], device=None, verbose=False, **_common())
    assert LogicalErrorRate(res["cluster"], StimDevice()).verdicts() == {}


def test_harness_engine_ler_equals_global_per_shot():
    circ = _circuit()
    device = StimDevice(seed=11)
    matcher = pymatching.Matching.from_detector_error_model(
        circ.detector_error_model(decompose_errors=True))
    g = {"errors": 0, "agree": 0}

    def on_shot(s, cluster, dev):
        pe = int(cluster.op_results[1])
        truth = int(dev._truth[1][0])
        pg = int(matcher.decode(dev._dets[1])[0])
        g["errors"] += int(pg != truth)
        g["agree"] += int(pe == pg)

    shots = 250
    out = logical_error_rate(_ops(circ), shots=shots, device=device, on_shot=on_shot,
                             **_common())
    # windowed == global, bit-identical per shot (the seed-robust anchor, not a fixed number)
    assert g["agree"] == shots
    assert out["errors"] == g["errors"]
    assert out["shots"] == shots and out["score_op"] == 1
    assert out["ci_low"] <= out["ler"] <= out["ci_high"]
    assert out["ci_low"] <= g["errors"] / shots <= out["ci_high"]
