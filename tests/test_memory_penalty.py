"""Analytic idle-memory logical-error penalty -- issue #5 (analytic path).

SurfaceCodeModel.memory_error(r) = mu*d*r*Lambda^(-(d+1)/2) (arXiv:2511.10633 Eq. 5), and the
MemoryErrorPenalty metric sums it over the idle (memory) rounds the simulator already tracks
while a feedback-blocked gate stalls. This is the closed-form model the papers use for their error
budgets (no stim sampling of the idle stretch).

Acceptance:
- memory_error reproduces the paper's per-logical-cycle table Pmem(d) = mu*d^2*Lambda^(-(d+1)/2);
- it is exactly linear in idle rounds (the per-round penalty property);
- the constants are configurable (the surface lattice-surgery fit);
- the metric sums Pmem over a REAL stalling run's tracked idle rounds, and the timing trace is
  unchanged (the metric is read-only)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from decsim.codes import SurfaceCodeModel
from decsim.metrics import MemoryErrorPenalty
from decsim.decoders import PresetLatencyDecoder
from decsim.frontends.circuit import CircuitFrontend
from decsim.message import Operation
from decsim.wiring import build_and_run


def _t_then_blocked_t():
    """A non-Clifford op whose successor is blocked on its decode, making the patch idle and emit
    memory rounds until the correction returns."""
    return CircuitFrontend([
        Operation(0, "A:T(q0)", (0,), clifford=False),
        Operation(1, "B:T(q0)", (0,), clifford=False, blocked_by=0),
    ]).build()


def test_memory_error_reproduces_paper_table():
    # Pmem(d) = memory_error(rounds=d) with the default fit (mu=0.019, Lambda=9.3)
    table = {3: 1.977e-3, 5: 5.905e-4, 7: 1.245e-4, 9: 2.212e-5, 11: 3.553e-6}
    for d, val in table.items():
        assert SurfaceCodeModel(d=d).memory_error(d) == pytest.approx(val, rel=2e-3)


def test_memory_error_linear_in_rounds():
    c = SurfaceCodeModel(d=5)
    assert c.memory_error(50) == pytest.approx(10 * c.memory_error(5))
    assert c.memory_error(0) == 0.0


def test_constants_configurable_surgery_fit():
    # the surface lattice-surgery fit (mu=0.021, Lambda=10.7): Pmem(5,5)=4.286e-4
    c = SurfaceCodeModel(d=5, mu_mem=0.021, lam_mem=10.7)
    assert c.memory_error(5) == pytest.approx(4.286e-4, rel=2e-3)
    # different constants give a different (here smaller) penalty than the default fit
    assert c.memory_error(5) != SurfaceCodeModel(d=5).memory_error(5)


def test_metric_sums_pmem_over_real_stalling_run():
    res = build_and_run(_t_then_blocked_t(), num_units=1, d=3, rounds_per_op=11,
                        decoder=PresetLatencyDecoder(5.0),    # slow enough to emit idle rounds
                        verbose=False)
    cluster = res["cluster"]
    idle = {op: r for op, r in cluster.memory_rounds.items() if r > 0}
    assert idle, "the blocked chain emitted no idle rounds -- test would be vacuous"

    out = MemoryErrorPenalty(cluster).result()
    code = SurfaceCodeModel(d=3)
    expected = sum(code.memory_error(r) for r in idle.values())
    assert out["total"] == pytest.approx(expected)
    assert out["total"] > 0
    # per-op breakdown carries the idle-round count and distance it used
    for op_id, r in idle.items():
        assert out["per_op"][op_id]["idle_rounds"] == r
        assert out["per_op"][op_id]["d"] == 3
        assert out["per_op"][op_id]["penalty"] == pytest.approx(code.memory_error(r))


def test_metric_is_read_only_timing_unchanged():
    """Reading the penalty must not perturb the run: same finish time with and without it."""
    common = dict(num_units=1, d=3, rounds_per_op=11,
                  decoder=PresetLatencyDecoder(5.0), verbose=False)
    a = build_and_run(_t_then_blocked_t(), **common)
    MemoryErrorPenalty(a["cluster"]).result()
    b = build_and_run(_t_then_blocked_t(), **common)
    assert a["fully_done"] == b["fully_done"]
