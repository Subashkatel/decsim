"""Queueing acid test (spec §2.1.1, plan Task 20 step 1) — real simulations.

Windowed decoding is a queueing system: windows arrive every `commit` rounds
(λ_source set by the round clock) and one decode unit serves them in E[S]
ticks, so the utilization is ρ = λ·E[S]/c (Skoric arXiv:2209.08552; Terhal
backlog argument). The physics the simulator must reproduce:

  ρ < 1  →  backlog bounded: its peak is a property of the pipeline, NOT of
            how long the run is (doubling the circuit leaves the peak flat).
  ρ > 1  →  backlog grows without bound: it rises monotonically through the
            run and its peak scales with the circuit length.

The companion accuracy anchor (windowed LER == global LER on frozen shots)
lives in test_golden_decoding.py on the frozen decode corpus.
"""
import pytest

from decsim.decoders import PresetLatencyDecoder
from decsim.frontends.circuit import CircuitFrontend
from decsim.run_spec import simulate
from decsim.message import Operation
from decsim.metrics import DecodeBacklog
from decsim.planner import FixedRounds
from decsim.run_spec import RunSpec

ROUND_US = 1.1          # today's TimingConfig round clock
COMMIT = 3              # d=3 surface code: commit_rounds == d
WINDOW_US = COMMIT * ROUND_US   # inter-arrival time of decode windows


def _memory_op(rounds):
    return CircuitFrontend([
        Operation(0, "M:mem(q0)", (0,), clifford=True),
    ]).build()


def _run(rho, rounds):
    """One-patch memory run: single decode unit, service time E[S] = rho·(window
    inter-arrival), backlog sampled every event by DecodeBacklog."""
    backlog = {}

    def make_metrics(engine, cluster, chip, factory):
        backlog["m"] = DecodeBacklog(cluster)
        return [backlog["m"]]

    res = simulate(RunSpec(ops=_memory_op(rounds), d=3,
                           rounds_policy=FixedRounds(rounds),
                           decoder=PresetLatencyDecoder(rho * WINDOW_US),
                           num_units=1, make_metrics=make_metrics))
    return res, backlog["m"]


def _peak_during_emission(res, metric):
    """Peak backlog while the source is still emitting (up to chip_done)."""
    return max((b for t, b in metric.trace if t <= res["chip_done"]), default=0)


@pytest.mark.parametrize("rho", [0.8])
def test_subcritical_backlog_bounded(rho):
    res_short, m_short = _run(rho, rounds=200)
    res_long, m_long = _run(rho, rounds=400)
    peak_short = _peak_during_emission(res_short, m_short)
    peak_long = _peak_during_emission(res_long, m_long)
    # bounded: the peak is pipeline-sized and does NOT scale with run length
    assert peak_long - peak_short <= 2 * COMMIT
    assert peak_long < 10 * COMMIT
    # stable second half: the late-run peak does not exceed the mid-run peak
    # by more than one window (p99-stability proxy on the full event trace)
    half = res_long["chip_done"] / 2
    mid = max(b for t, b in m_long.trace if half <= t <= 1.5 * half)
    late = max(b for t, b in m_long.trace if 1.5 * half <= t <= 2 * half)
    assert late <= mid + COMMIT


@pytest.mark.parametrize("rho", [1.2])
def test_supercritical_backlog_grows(rho):
    res_short, m_short = _run(rho, rounds=200)
    res_long, m_long = _run(rho, rounds=400)
    peak_short = _peak_during_emission(res_short, m_short)
    peak_long = _peak_during_emission(res_long, m_long)
    # unbounded: doubling the run length ~doubles the peak backlog
    assert peak_long >= 1.6 * peak_short
    assert peak_long - peak_short >= 20            # theory: ~(1-1/rho)·rounds/2
    # monotone growth through the emission phase
    end = res_long["chip_done"]
    samples = [max((b for t, b in m_long.trace if t <= frac * end), default=0)
               for frac in (0.25, 0.5, 0.75, 1.0)]
    assert samples == sorted(samples)
    assert samples[0] < samples[-1]


def test_backlog_drains_after_emission_when_subcritical():
    res, metric = _run(0.8, rounds=200)
    # the queue empties: every arrived round is eventually decoded
    assert metric.trace[-1][1] == 0
    assert res["fully_done"] > res["chip_done"]    # tail exists but finite
