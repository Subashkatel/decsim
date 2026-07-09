"""Gate 7 P2: BacklogEarlyWarning focused tests.

Predeclaration: docs/validation/2026-07-03-gate7-p2-predeclaration.md.
Unit level: bin/slope arithmetic, k-consecutive latching (a single
spike must NOT warn), threshold edge, patch attribution (W5).
Engine level: stable regime -> no warning; unstable regime -> warning
(build_and_run + the metrics seam, QLX-stall-formula harness).
Synthetic service times only (no latency claims).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from decsim.codes import SurfaceCodeModel
from decsim.config import us
from decsim.controllers import ModularController, LinkModel
from decsim.message import DecodeResult, Operation
from decsim.metrics import BacklogEarlyWarning, DecodeBacklog
from decsim.schemes import SlidingWindowScheme
from decsim.run_spec import RunSpec, simulate
from decsim.planner import FixedRounds


# ------------------------------------------------------------- unit level

class FakeOp:
    def __init__(self, patches):
        self.patches = patches
        self.qubits = patches


class FakeCluster:
    """Just enough surface for views.backlog_view."""

    def __init__(self, per_op_patch: dict):
        self.ready = []
        self.pool_ready = {}
        self.ops = {op: FakeOp((patch,))
                    for op, patch in per_op_patch.items()}
        self.rounds_arrived = {op: 0 for op in per_op_patch}
        self.windows = {}
        self.committed_windows = set()

    def set_backlog(self, per_op: dict):
        self.rounds_arrived.update(per_op)


class FakeEngine:
    def __init__(self):
        self.now = 0


def drive(warn: BacklogEarlyWarning, cluster, engine, samples):
    """samples: (tick, {op: backlog_rounds}) — one observe per event."""
    for tick, per_op in samples:
        engine.now = tick
        cluster.set_backlog(per_op)
        warn.observe(engine)


def make(cluster, **kw):
    args = dict(round_ticks=us(1), window_ticks=us(10),
                threshold_f=0.1, consecutive=2)
    args.update(kw)
    return BacklogEarlyWarning(cluster, **args)


def test_slope_arithmetic_and_no_warning_when_flat():
    cluster = FakeCluster({0: "p0"})
    eng = FakeEngine()
    warn = make(cluster)
    drive(warn, cluster, eng, [(us(t), {0: 5}) for t in range(0, 45, 5)])
    res = warn.result()
    assert res["bins_evaluated"] >= 3
    assert all(abs(b["slope_f"]) < 1e-12 for b in res["slopes"])
    assert not res["warned"]


def test_slope_matches_known_growth_rate():
    """backlog += 2 rounds per 10us bin -> slope 0.2 (round_ticks=1us)."""
    cluster = FakeCluster({0: "p0"})
    eng = FakeEngine()
    warn = make(cluster, threshold_f=10.0)     # measure only, never warn
    drive(warn, cluster, eng,
          [(us(10 * k), {0: 2 * k}) for k in range(0, 6)])
    res = warn.result()
    inner = [b["slope_f"] for b in res["slopes"]][1:]  # first bin startup
    assert inner and all(abs(s - 0.2) < 1e-9 for s in inner), res["slopes"]


def test_single_spike_does_not_warn_with_k2():
    cluster = FakeCluster({0: "p0"})
    eng = FakeEngine()
    warn = make(cluster)
    # one hot bin (slope 0.5) surrounded by flat bins
    profile = [0, 0, 5, 5, 5, 5]                # rounds at bin boundaries
    drive(warn, cluster, eng,
          [(us(10 * k), {0: v}) for k, v in enumerate(profile)])
    res = warn.result()
    assert max(b["slope_f"] for b in res["slopes"]) >= 0.5 - 1e-9
    assert not res["warned"]


def test_two_consecutive_hot_bins_warn_and_latch():
    cluster = FakeCluster({0: "p0"})
    eng = FakeEngine()
    warn = make(cluster)
    profile = [0, 2, 4, 4, 4]                   # two bins at slope 0.2
    drive(warn, cluster, eng,
          [(us(10 * k), {0: v}) for k, v in enumerate(profile)])
    res = warn.result()
    assert res["warned"]
    assert res["t_warn_ticks"] == us(20)        # end of the 2nd hot bin
    later = warn.t_warn
    drive(warn, cluster, eng, [(us(50), {0: 50}), (us(60), {0: 80})])
    assert warn.t_warn == later                 # latched


def test_threshold_edge_below_does_not_warn():
    cluster = FakeCluster({0: "p0"})
    eng = FakeEngine()
    warn = make(cluster, threshold_f=0.1)
    # slope 0.09 forever
    drive(warn, cluster, eng,
          [(us(100 * k), {0: int(9 * k)}) for k in range(0, 8)])
    assert not warn.result()["warned"]


def test_slope_exactly_at_threshold_never_warns():
    """Codex P2 review finding 4: the quantization step (1 round per
    10-round bin) equals the 0.1 threshold exactly; strict > must
    exclude an indefinite run of exact-threshold bins."""
    cluster = FakeCluster({0: "p0"})
    eng = FakeEngine()
    warn = make(cluster)                        # threshold 0.1, k=2
    drive(warn, cluster, eng,
          [(us(10 * k), {0: k}) for k in range(0, 12)])  # slope == 0.1
    res = warn.result()
    inner = [b["slope_f"] for b in res["slopes"]][1:]
    assert inner and all(abs(s - 0.1) < 1e-12 for s in inner)
    assert not res["warned"]


def test_attribution_names_the_diverging_patch():
    """W5: two patches, only p1 diverges."""
    cluster = FakeCluster({0: "p0", 1: "p1"})
    eng = FakeEngine()
    warn = make(cluster)
    samples = [(us(10 * k), {0: 3, 1: 3 * k}) for k in range(0, 5)]
    drive(warn, cluster, eng, samples)
    res = warn.result()
    assert res["warned"]
    assert res["attribution"] == ["p1"], res["attribution"]


# ------------------------------------------------------------ engine level

class _FixedLatencyDecoder:
    def __init__(self, latency_us):
        self._t = us(latency_us)

    def latency(self, job):
        return self._t

    def decode(self, job):
        return DecodeResult(job.op_id, job.window_id, logical_value=0)


def _run_regime(tau_w_us, rounds=120):
    op = Operation(0, "mem", (0,), clifford=True, patches=(0,))
    res = simulate(RunSpec(
              ops=[op],
              num_units=4,
              d=3,
              rounds_policy=FixedRounds(rounds),
              round_us=1.0,
              decoder=_FixedLatencyDecoder(tau_w_us),
              scheme=SlidingWindowScheme(),
              code=SurfaceCodeModel(d=3),
              make_controller=lambda e: ModularController(
            e, links=LinkModel(qc=0, cd=0, dd=0, do=0, oc=0, cq=0),
            log_syndromes=False),
              make_metrics=lambda e, cl, ch, f: [
            DecodeBacklog(cl),
            BacklogEarlyWarning(cl, round_ticks=us(1),
                                window_ticks=us(10))],
          ), verbose=False)
    return res["metrics"]["backlog_early_warning"]


def test_engine_stable_regime_never_warns():
    out = _run_regime(1.5)                      # f = -0.5
    assert not out["warned"], out


def test_engine_unstable_regime_warns_early():
    out = _run_regime(6.0)                      # f = +1.0
    assert out["warned"], out
    assert out["t_warn_ticks"] <= 0.3 * us(120), out["t_warn_ticks"]
