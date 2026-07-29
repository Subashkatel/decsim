"""Literature-reproduction benchmarks (plan Task 23).

Each benchmark reproduces a published quantitative result with a config-only
setup: pick RunSpec parts, state the paper's observable and tolerance, run.

  1. Skoric et al., arXiv:2209.08552 — streaming backlog: bounded at ρ<1,
     grows at rate λ−μ (±10%) at ρ>1; windowed == global decode accuracy.
  2. Gidney–Ekerå, arXiv:1905.09749 — reaction-limited serial chain: runtime
     matches the analytic per-layer period within ±15% (QLX's own tolerance).
"""
import pytest

from decsim.decoders import PresetLatencyDecoder
from decsim.frontends.circuit import CircuitFrontend
from decsim.run_spec import simulate
from decsim.metrics import DecodeBacklog
from decsim.message import Operation
from decsim.planner import FixedRounds
from decsim.run_spec import RunSpec
from decsim.config import TICKS_PER_US, TimingConfig, us

D = 3
COMMIT = D                       # d=3 surface code: commit_rounds == d
ROUND_US = 1.0


def _memory_op(rounds):
    return CircuitFrontend([Operation(0, "M:mem(q0)", (0,), clifford=True)]).build()


def _chain(k):
    ops = [Operation(0, "T0(q0)", (0,), clifford=False)]
    ops += [Operation(i, f"T{i}(q0)", (0,), clifford=False, blocked_by=i - 1)
            for i in range(1, k)]
    return CircuitFrontend(ops).build()


#==================================================================
# 1. Skoric arXiv:2209.08552 — streaming backlog regimes
#==================================================================

T_DD_US = 0.5      # boundary-handoff hop between consecutive windows
T_WDO_US = 1.0     # accepted weak result before boundary publication

def _backlog_run(rho, rounds):
    """Sliding windows on one unit decode STRICTLY in sequence — "sliding
    window decoding is inherently sequential" (Skoric §I.B p. 2): window k+1
    needs window k's committed boundary (artificial) defects (+t_dd hop), so
    the service cycle per window is E[S] + t_wdo + t_dd and
    ρ = (E[S] + t_wdo + t_dd) / (commit·t_round)."""
    backlog = {}

    def make_metrics(engine, window_manager, decoder_manager, chip, factory):
        backlog["m"] = DecodeBacklog(window_manager, decoder_manager)
        return [backlog["m"]]

    service_us = rho * COMMIT * ROUND_US - T_WDO_US - T_DD_US
    res = simulate(RunSpec(ops=_memory_op(rounds), d=D,
                           rounds_policy=FixedRounds(rounds),
                           round_us=ROUND_US, num_units=1,
                           decoder=PresetLatencyDecoder(service_us),
                           make_metrics=make_metrics))
    return res, backlog["m"]


def test_skoric_supercritical_backlog_grows_at_lambda_minus_mu():
    """ρ>1: the backlog grows linearly at λ−μ rounds per unit time (±10%).

    Skoric's keep-pace condition is τ_W < n_com·τ_rd (§I.B p. 2); violated
    here by construction. NB the papers' EXPONENTIAL slowdown (Terhal's
    f^k, quoted in Skoric's intro) is exponential in ADAPTIVE T-depth k —
    this streaming memory workload has no feed-forward, so the backlog is
    the linear λ−μ regime, which is what the assertion checks."""
    rho, rounds = 1.25, 600
    res, metric = _backlog_run(rho, rounds)

    lam = 1.0 / ROUND_US                        # rounds arriving per us
    mu = COMMIT / (rho * COMMIT * ROUND_US)     # rounds decoded per us
    expected = (lam - mu) / TICKS_PER_US        # rounds per tick

    t1, t2 = 0.25 * res.result.chip_done_ticks, 0.95 * res.result.chip_done_ticks
    b1 = max(b for t, b in metric.trace if t <= t1)
    b2 = max(b for t, b in metric.trace if t <= t2)
    measured = (b2 - b1) / (t2 - t1)
    assert measured == pytest.approx(expected, rel=0.10)


def test_skoric_subcritical_backlog_bounded():
    """ρ<1: the p99 backlog is pipeline-sized and flat across the run."""
    res, metric = _backlog_run(0.8, rounds=600)
    during = sorted(b for t, b in metric.trace if t <= res.result.chip_done_ticks)
    p99 = during[int(0.99 * (len(during) - 1))]
    assert p99 <= 4 * COMMIT                    # ~a window, not ~the run length
    assert metric.trace[-1][1] == 0             # drains completely


def test_skoric_windowed_decode_matches_global_on_frozen_shots():
    """Accuracy anchor: sliding-window MWPM reproduces the global decode on
    the frozen 2000-shot d=3 corpus (windowed == global within 1%) — the
    regime Skoric validates numerically (App. C, Fig. 6a: windowed matches
    global at n_buf = n_com = d)."""
    stim = pytest.importorskip("stim")
    np = pytest.importorskip("numpy")
    pytest.importorskip("pymatching")
    import json, pathlib

    from decsim.detector_error_model import build_window_error_models, decode_windowed
    from decsim.mwpm_decoder import matching_window_decoder

    data = pathlib.Path(__file__).resolve().parent / "data"
    g = json.loads((data / "golden_decoding.json").read_text())["scenarios"]
    g = g["rsc-d3-r6-p0.005"]
    circ = stim.Circuit.from_file(str(data / "rsc-d3-r6-p0.005.stim"))
    shots = np.load(data / "rsc-d3-r6-p0.005.shots.npz")
    dets, obs = shots["dets"], shots["obs"]

    models = build_window_error_models(circ, [tuple(w) for w in g["plan"]])
    inner = matching_window_decoder()
    windowed = sum(int(decode_windowed(models, dets[i], inner)[0] != obs[i, 0])
                   for i in range(g["n"]))
    assert abs(windowed - g["global_mwpm_fails"]) <= max(2, int(0.01 * g["n"]))


#==================================================================
# 2. Gidney–Ekerå arXiv:1905.09749 — reaction-limited serial chain
#==================================================================

def test_gidney_ekera_reaction_limited_runtime_within_15_percent():
    """A serial non-Clifford chain is reaction-limited: each layer costs its
    body (rounds·t_round) plus one reaction time (syndrome forward path +
    decode + result return). GE define reaction time as "the amount of time
    it takes the classical control system to trigger a logical measurement,
    collect and error-correct the result, and decide on which measurement
    basis to use for the next set of measurements" (App. B p. 24; they
    assume 10 µs, Table II) and run one dependent-measurement layer per
    reaction time (App. A p. 22, §2.J). NB the full GE computation is only
    "nearly reaction limited" (the lookup phase is code-depth limited);
    this chain is the purely reaction-limited construction. Measured
    per-layer period and total runtime must match the analytic formula
    within ±15% — the tolerance QLX validates its own Ekerå–Håstad
    runtimes at."""
    from decsim.schemes import NaiveOnlineScheme
    from decsim.links import LinkModelConfig
    k, rounds, decode_us = 6, 11, 5.0
    links = LinkModelConfig.reference_fixed_latency_profile()
    res = simulate(RunSpec(ops=_chain(k), d=D,
                           rounds_policy=FixedRounds(rounds),
                           round_us=ROUND_US, num_units=1,
                           scheme=NaiveOnlineScheme(),   # one window per layer
                           decoder=PresetLatencyDecoder(decode_us)))
    gate = res.chip

    body = rounds * us(ROUND_US)
    reaction = (
        links.qc.channel.propagation_latency_ticks
        + links.cwd.channel.propagation_latency_ticks
        + us(decode_us)
        + links.wdo.channel.propagation_latency_ticks
        + links.oc.channel.propagation_latency_ticks
        + links.cq.channel.propagation_latency_ticks
    )
    period = body + reaction

    releases = [gate.decode_release_time[i] for i in range(1, k)]
    measured_periods = [b - a for a, b in zip(releases, releases[1:])]
    for measured in measured_periods:
        assert measured == pytest.approx(period, rel=0.15)
    # the LAST layer releases no successor, so its return hops never happen:
    # total = k*period - (t_oc + t_cq)   (Codex re-derivation, exact)
    total = k * period - (
        links.oc.channel.propagation_latency_ticks
        + links.cq.channel.propagation_latency_ticks
    )
    assert res.result.fully_done_ticks == pytest.approx(total, rel=0.15)
