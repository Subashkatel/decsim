"""Reaction-barrier acid test (spec §2.1.2, plan Task 20 step 2) — real runs.

A chain of k sequential non-Clifford layers is reaction-limited (Terhal;
Gidney–Ekerå arXiv:1905.09749): each layer stalls until the previous layer's
decode releases it. With a weak decoder that escalates any given window with
probability p, the number of escalation-free chains over many seeds is
geometric in the number of decodes: P(clean chain) = (1-p)^W with W the
window count of the whole chain — the f^k model with f = (1-p)^(W/k) per
layer. The stall clock runs on λ_eff (feedback idle rounds are emitted and
counted while blocked), not λ_source, and a chain whose decode never returns
must terminate at max_idle_rounds.
"""
import pytest

from decsim.decoders import (PerRoundDecoder, PresetLatencyDecoder,
                             SampledConfidenceDecoder, SwitchingRouter)
from decsim.frontends.circuit import CircuitFrontend
from decsim.run_spec import simulate
from decsim.message import Operation
from decsim.planner import FixedRounds
from decsim.run_spec import RunSpec
from decsim.switching import Switching

K = 4                   # non-Clifford layers in the chain
P_ESCALATE = 0.1        # per-window weak->strong escalation probability
N_CHAINS = 200          # seeds


def _chain():
    ops = [Operation(0, "T0(q0)", (0,), clifford=False)]
    ops += [Operation(i, f"T{i}(q0)", (0,), clifford=False, blocked_by=i - 1)
            for i in range(1, K)]
    return CircuitFrontend(ops).build()


def _run_chain(p_escalate, seed, **overrides):
    weak = SampledConfidenceDecoder(PerRoundDecoder(0.2), p_escalate)
    return simulate(RunSpec(
        ops=_chain(), d=3, rounds_policy=FixedRounds(11),
        strategy=Switching(confidence_threshold=0.5),
        decoder=weak, router=SwitchingRouter(weak, PerRoundDecoder(3.0)),
        unit_pools={"default": 1, "strong": 1}, seed=seed, **overrides))


def _stalls(res):
    """Inter-barrier stalls: layer i's release wait after layer i-1's body."""
    gate = res.chip
    return [gate.decode_release_time[i] - gate.body_done_time[i - 1]
            for i in range(1, K) if i in gate.decode_release_time]


def test_geometric_escalation_statistics_over_seeds():
    """The fraction of escalation-free chains matches (1-p)^W within 3σ."""
    windows = _run_chain(0.0, seed=0).cluster.total_windows   # W, measured
    assert windows >= K                        # at least one window per layer

    clean = 0
    for seed in range(N_CHAINS):
        res = _run_chain(P_ESCALATE, seed=seed)
        assert len(_stalls(res)) == K - 1      # every layer released
        clean += not res.cluster.op_strong_commit_time
    q = (1 - P_ESCALATE) ** windows
    sigma = (N_CHAINS * q * (1 - q)) ** 0.5
    assert abs(clean - N_CHAINS * q) <= 3 * sigma


def test_escalated_chains_stall_longer_on_lambda_eff():
    """Strong redos lengthen the inter-barrier stall, and the stall time is
    filled with emitted feedback-idle rounds (λ_eff), not silence."""
    clean_stalls, hot_stalls, hot_idle = [], [], []
    for seed in range(60):
        res = _run_chain(P_ESCALATE, seed=seed)
        stalls = _stalls(res)
        # total idle rounds EMITTED; the per-patch counter is consumed by the
        # released successor at start (Contract 3.6), so read the run total
        idle = res.chip.idle_rounds_emitted
        if res.cluster.op_strong_commit_time:
            hot_stalls += stalls
            hot_idle.append(idle)
        else:
            clean_stalls += stalls
    assert clean_stalls and hot_stalls         # both populations sampled
    mean = lambda xs: sum(xs) / len(xs)
    assert mean(hot_stalls) > mean(clean_stalls)
    assert min(hot_idle) > 0                   # blocked layers emitted idle rounds


def test_idle_emission_terminates_at_max_idle_rounds():
    """A decode that takes ~1e5 µs would otherwise force ~90k idle rounds per
    blocked layer; max_idle_rounds terminates each layer's emission at the cap
    (Contract 3 rule 5's cap guard) and the run still winds down."""
    cap = 12
    res = simulate(RunSpec(ops=_chain(), d=3, rounds_policy=FixedRounds(11),
                           decoder=PresetLatencyDecoder(1e5), num_units=1,
                           max_idle_rounds=cap))
    gate = res.chip
    assert len(gate.idle_cap_hits) == K - 1            # every blocked layer capped
    assert gate.idle_rounds_emitted == (K - 1) * cap   # emission stopped exactly there
    assert len(_stalls(res)) == K - 1                  # decode returns still release
