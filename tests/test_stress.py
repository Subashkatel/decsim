"""Stress tests: hammer the decoder cluster under high concurrency and high switching load,
and assert the core invariants hold THROUGHOUT the run -- not just at the end.

These do not test a paper claim; they guard the cluster's unit accounting, window-commit
bookkeeping, and syndrome-RAM accounting against regressions under load: many patches
competing for few units, a high escalation rate hammering the strong pool, and the parallel
scheme running many windows at once. The InvariantGuard metric runs after EVERY engine event,
so a violation is caught at the instant it happens, with the time it happened.

Invariants checked at every event:
  - 0 <= pool_free[pool] <= unit_totals[pool]        (never over-dispatch, never leak a unit)
  - committed windows per op never exceed its plan    (no window commits twice)
  - payloads_held >= 0                                (syndrome-RAM accounting never negative)
And at the end of every scenario:
  - all units returned (pool_free == unit_totals),
  - every window committed exactly once,
  - syndrome RAM fully freed (payloads_held == 0, payload_store empty).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from decsim.config import us
from decsim.decoders import SampledSoftOutputDecoder, SwitchingRouter
from decsim.message import DecodeResult, Operation
from decsim.schedulers import EarliestDeadlineScheduler
from decsim.schemes import DoubleWindowScheme, ParallelWindowScheme, SlidingWindowScheme
from decsim.wiring import build_and_run

TAU = 1.0


class PerRoundDecoder:
    """Timing-only decoder: latency proportional to the window's rounds."""
    def __init__(self, tau_us):
        self.tau_us = tau_us

    def latency(self, job):
        return us(job.n_rounds * self.tau_us)

    def decode(self, job):
        return DecodeResult(job.op_id, job.window_id)


class InvariantGuard:
    """Runs after every engine event and records any invariant violation with its timestamp.
    Also tracks the peak simultaneously-busy units per pool, so a scenario can assert it
    really did stress the units (and not, say, trivially serialize)."""
    name = "invariants"

    def __init__(self, cluster):
        self.cluster = cluster
        self.violations = []
        self.checks = 0
        self.peak_busy = {}

    def observe(self, engine):
        cluster = self.cluster
        now = engine.now
        for pool, total in cluster.unit_totals.items():
            free = cluster.pool_free[pool]
            if not 0 <= free <= total:
                self.violations.append(f"t={now}: pool {pool!r} free={free} outside [0,{total}]")
            busy = total - free
            self.peak_busy[pool] = max(self.peak_busy.get(pool, 0), busy)
        if cluster.payloads_held < 0:
            self.violations.append(f"t={now}: payloads_held={cluster.payloads_held} < 0")
        for op_id, committed in cluster._committed_per_op.items():
            planned = cluster.nwin.get(op_id, 0)
            if committed > planned:
                self.violations.append(
                    f"t={now}: op {op_id} committed {committed} windows > planned {planned} "
                    f"(a window committed more than once)")
        self.checks += 1

    def result(self):
        return {"violations": self.violations, "checks": self.checks,
                "peak_busy": dict(self.peak_busy)}


def _assert_clean(result, guard):
    """End-of-run invariants common to every stress scenario."""
    cluster = result["cluster"]
    assert guard.violations == [], guard.violations[:5]
    assert guard.checks > 50, "the guard barely ran -- scenario was not a real stress"
    assert cluster.pool_free == cluster.unit_totals, "a unit was leaked or double-freed"
    assert len(cluster.committed_windows) == cluster.total_windows, "not every window committed"
    assert sum(cluster._committed_per_op.values()) == cluster.total_windows, "double commit"
    assert cluster.payloads_held == 0, "syndrome RAM not fully freed"
    assert all(not frags for frags in cluster.payload_store.values()), "syndrome RAM leaked"


def _independent_patches(n):
    """n memory operations on distinct qubits -- no dependencies, fully concurrent."""
    return [Operation(i, f"mem{i}", (i,), clifford=True) for i in range(n)]


def test_many_patches_few_units_stays_consistent():
    """Twelve independent patches contend for four units under a slow decoder: the units
    saturate (peak busy hits 4) and every window still commits exactly once."""
    guard_box = {}
    result = build_and_run(
        _independent_patches(12), num_units=4, d=3, rounds_per_op=40, round_us=TAU,
        scheme=SlidingWindowScheme(), decoder=PerRoundDecoder(3.0),
        make_metrics=lambda e, c, ch, fa: [guard_box.setdefault("g", InvariantGuard(c))],
        verbose=False)
    guard = guard_box["g"]
    _assert_clean(result, guard)
    assert guard.peak_busy["default"] == 4, "four patches should saturate four units"


def test_parallel_scheme_under_load_stays_consistent():
    """The parallel A/B scheme exposes concurrent windows: with a slow decoder several units
    run at once, and the commit/RAM bookkeeping survives the out-of-order commits."""
    guard_box = {}
    result = build_and_run(
        _independent_patches(3), num_units=4, d=3, rounds_per_op=90, round_us=TAU,
        scheme=ParallelWindowScheme(), decoder=PerRoundDecoder(12.0),
        make_metrics=lambda e, c, ch, fa: [guard_box.setdefault("g", InvariantGuard(c))],
        verbose=False)
    guard = guard_box["g"]
    _assert_clean(result, guard)
    assert guard.peak_busy["default"] >= 2, "the parallel scheme should run windows concurrently"


def _switching_run(escalation_probability, rounds, patches, pools, seed=3,
                   scheduler=None, metrics_box=None):
    weak = SampledSoftOutputDecoder(PerRoundDecoder(0.2 * TAU), escalation_probability, seed=seed)
    strong = PerRoundDecoder(5.0 * TAU)
    return build_and_run(
        _independent_patches(patches), d=3, rounds_per_op=rounds, round_us=TAU,
        scheme=DoubleWindowScheme(g_th=0.5), decoder=weak,
        router=SwitchingRouter(weak, strong), unit_pools=pools, scheduler=scheduler,
        make_metrics=lambda e, c, ch, fa: [metrics_box.setdefault("g", InvariantGuard(c))]
                     if metrics_box is not None else [],
        verbose=False)


def test_high_escalation_switching_stays_consistent():
    """Half of all windows escalate, across several patches, hammering a small strong pool:
    weak windows all commit exactly once and the strong pool never over-dispatches."""
    box = {}
    result = _switching_run(0.5, rounds=300, patches=3,
                            pools={"default": 2, "strong": 2}, metrics_box=box)
    guard = box["g"]
    _assert_clean(result, guard)
    assert result["cluster"].escalations > 50, "the run did not actually stress switching"
    assert guard.peak_busy.get("strong", 0) >= 1, "strong pool never ran a job"


def test_switching_with_deadline_scheduler_stays_consistent():
    """The same switching stress under the EDF scheduler (a different queue-ordering policy)
    must keep every invariant -- the scheduler swap must not break unit accounting."""
    box = {}
    result = _switching_run(0.3, rounds=200, patches=4,
                            pools={"default": 2, "strong": 1},
                            scheduler=EarliestDeadlineScheduler(), metrics_box=box)
    _assert_clean(result, box["g"])
    assert result["cluster"].escalations > 0


def test_switching_stress_is_deterministic():
    """Same seed, same everything -> identical results. A stress run must be reproducible:
    the unit pools, the scheduler, and the escalation draws must not depend on event order
    in a way that varies between runs."""
    a = _switching_run(0.5, rounds=250, patches=3, pools={"default": 2, "strong": 2})
    b = _switching_run(0.5, rounds=250, patches=3, pools={"default": 2, "strong": 2})
    assert a["cluster"].op_results == b["cluster"].op_results
    assert a["cluster"].escalations == b["cluster"].escalations
    assert len(a["cluster"].committed_windows) == len(b["cluster"].committed_windows)
    assert a["engine"].now == b["engine"].now
