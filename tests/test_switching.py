"""Double-window decoder switching (arXiv:2510.25222 Sec III.C). Each test maps to a claim
in the paper, validated against the paper PDF on 2026-06-17.

The scheme runs a fast WEAK decoder on ordinary sliding windows and, whenever a window's
soft output is below the threshold g_th, ALSO hands that window to a slow STRONG decoder
over strong_rounds = commit + 2*buffer rounds. The weak stream never waits for the strong
decoder, so only the strong decoder can accumulate a backlog -- and Theorem 1 (Eq. 6) /
Eq. 8 bound it.

Setup: a single timing-only memory operation that streams many rounds, so the weak decoder
produces a long stream of sliding windows. The weak decoder escalates each window with a
fixed probability (SampledSoftOutputDecoder), which sets the switching rate; the strong
decoder runs on its own unit pool, so StrongDecoderBacklog reads the strong backlog directly.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from decsim.codes import SurfaceCodeModel
from decsim.config import us
from decsim.decoders import (SampledSoftOutputDecoder, SwitchingDecoder, SwitchingRouter)
from decsim.message import DecodeResult, Operation
from decsim.metrics import DecodeBacklog, StrongDecoderBacklog
from decsim.schemes import DoubleWindowScheme, SlidingWindowScheme
from decsim.wiring import build_and_run

TAU_GEN_US = 1.0          # syndrome round time
D = 3
F_WEAK = 0.1              # weak decode time per round / round time (well inside Eq. 7)
F_STRONG = 10             # the paper's tau_strong = 10 * tau_gen

# Theorem 1 (Eq. 6) strong-decoder boundary for the standard r_com = r_buf = d windows:
# escalations cost strong_rounds = 3d rounds each, the strong decoder clears
# d / f_strong rounds per commit region, so the per-window switch rate must satisfy
#   gamma <= d / (strong_rounds * f_strong) = 1 / (3 * f_strong).
# (This is the r_com = d value, 1/30; Eq. 8's 0.047 is the same bound after substituting the
# Eq. 7-MINIMUM r_com instead -- a different window geometry. See decoder-switching memory.)
GAMMA_BOUND = D / (3 * D * F_STRONG)


class PerRoundDecoder:
    """Decode time proportional to the window's rounds; no real logical value (timing only)."""
    def __init__(self, tau_us):
        self.tau_us = tau_us

    def latency(self, job):
        return us(job.n_rounds * self.tau_us)

    def decode(self, job):
        return DecodeResult(job.op_id, job.window_id)


def _memory_op():
    """One Clifford memory operation -- a long, non-gated syndrome stream."""
    return Operation(0, "memory", (0,), clifford=True)


def _strong_backlog_peak(escalation_probability, rounds, seed=1):
    """Peak outstanding strong jobs over a double-window run at a given switching rate."""
    weak = SampledSoftOutputDecoder(PerRoundDecoder(F_WEAK * TAU_GEN_US),
                                    escalation_probability, seed=seed)
    strong = PerRoundDecoder(F_STRONG * TAU_GEN_US)
    res = build_and_run([_memory_op()], num_units=1, d=D, rounds_per_op=rounds,
                        round_us=TAU_GEN_US, scheme=DoubleWindowScheme(g_th=0.5),
                        decoder=weak, router=SwitchingRouter(weak, strong),
                        unit_pools={"default": 1, "strong": 1},
                        make_metrics=lambda e, c, ch, fa: [StrongDecoderBacklog(c)],
                        verbose=False)
    return res["metrics"]["strong_backlog"]["peak_jobs"]


# ---- Eq. 7: the weak decoder must be able to keep pace -------------------------------

@pytest.mark.parametrize("f_weak, raises", [(0.7, True), (0.4, False), (None, False)])
def test_eq7_keep_up_guard(f_weak, raises):
    """plan_windows rejects a commit region too small for the weak decoder to keep pace
    (Eq. 7). At d=3 (commit=buffer=3): f_weak=0.7 needs commit>=7 rounds (raises);
    f_weak=0.4 needs commit>=2 (fine); f_weak=None skips the check."""
    scheme = DoubleWindowScheme(g_th=0.5, f_weak=f_weak)
    if raises:
        with pytest.raises(ValueError):
            scheme.plan_windows(0, 12, SurfaceCodeModel(d=D))
    else:
        scheme.plan_windows(0, 12, SurfaceCodeModel(d=D))


def test_f_weak_must_be_below_one():
    """The weak decoder is faster than syndrome generation: f_weak in (0, 1)."""
    with pytest.raises(ValueError):
        DoubleWindowScheme(g_th=0.5, f_weak=1.0)


def test_scheme_needs_exactly_one_of_threshold_or_policy():
    """The switch decision is configured by g_th OR a custom policy, not both, not neither."""
    with pytest.raises(ValueError):
        DoubleWindowScheme()
    with pytest.raises(ValueError):
        DoubleWindowScheme(g_th=0.5, switch_policy=object())


def test_custom_switch_policy_decides_when_to_switch():
    """The switch decision is a swappable SwitchPolicy, not baked into the scheme: a policy
    that escalates the first few windows and then stops produces exactly that many switches,
    independent of the soft outputs the weak decoder emits."""
    class FirstNWindowsSwitch:
        """Escalate the first `n` windows of each operation, then never again."""
        def __init__(self, n):
            self.n = n

        def should_escalate(self, job, result):
            return job.window_id < self.n

    scheme = DoubleWindowScheme(switch_policy=FirstNWindowsSwitch(n=2))
    weak = SampledSoftOutputDecoder(PerRoundDecoder(F_WEAK * TAU_GEN_US), 0.0, seed=1)
    res = build_and_run([_memory_op()], num_units=1, d=D, rounds_per_op=60,
                        round_us=TAU_GEN_US, scheme=scheme, decoder=weak,
                        router=SwitchingRouter(weak, PerRoundDecoder(F_STRONG * TAU_GEN_US)),
                        unit_pools={"default": 1, "strong": 1}, verbose=False)
    assert res["cluster"].escalations == 2


def test_strong_region_is_commit_plus_two_buffers():
    """r_strong = commit + 2*buffer = 3d for the standard d/d window (Fig 12)."""
    scheme = DoubleWindowScheme(g_th=0.5)
    windows = scheme.plan_windows(0, 4 * D, SurfaceCodeModel(d=D))
    from decsim.message import Window
    commit_lo, commit_hi, buffer_hi = windows[0]
    w = Window(op_id=0, k=0, commit_lo=commit_lo, commit_hi=commit_hi,
               buffer_hi=buffer_hi, n_rounds=buffer_hi - commit_lo + 1)
    assert scheme.strong_rounds(w) == 3 * D


# ---- the weak stream never stalls; with no switching it IS plain sliding -------------

def test_no_switching_matches_plain_sliding():
    """With the escalation probability at zero, no window ever switches, so the run is
    identical to the plain sliding-window scheme: same finish time, same committed windows,
    and zero escalations."""
    rounds = 120
    weak = SampledSoftOutputDecoder(PerRoundDecoder(F_WEAK * TAU_GEN_US), 0.0, seed=1)
    double = build_and_run([_memory_op()], num_units=1, d=D, rounds_per_op=rounds,
                           round_us=TAU_GEN_US, scheme=DoubleWindowScheme(g_th=0.5),
                           decoder=weak, verbose=False)
    sliding = build_and_run([_memory_op()], num_units=1, d=D, rounds_per_op=rounds,
                            round_us=TAU_GEN_US, scheme=SlidingWindowScheme(),
                            decoder=PerRoundDecoder(F_WEAK * TAU_GEN_US), verbose=False)
    assert double["cluster"].escalations == 0
    assert double["engine"].now == sliding["engine"].now
    assert (len(double["cluster"].committed_windows)
            == len(sliding["cluster"].committed_windows))


# ---- Theorem 1 / Eq. 8: strong backlog bounded inside, divergent outside -------------

def test_strong_backlog_bounded_inside_boundary():
    """Well inside the Theorem 1 boundary (a fraction of GAMMA_BOUND) the strong backlog
    stays bounded: its peak does not grow as the run gets longer."""
    inside = 0.15 * GAMMA_BOUND                  # ~0.005, comfortably stable
    peaks = [_strong_backlog_peak(inside, rounds) for rounds in (300, 600, 1200)]
    assert max(peaks) <= 3                        # a couple of jobs at most
    assert peaks[-1] <= peaks[0] + 1              # does not grow with run length


def test_strong_backlog_diverges_outside_boundary():
    """Well outside the boundary (several times GAMMA_BOUND) the strong backlog diverges:
    its peak grows roughly linearly with the run length -- the Fig 14 / Theorem 1 signature."""
    outside = 6 * GAMMA_BOUND                     # ~0.2, comfortably unstable
    short = _strong_backlog_peak(outside, 300)
    long = _strong_backlog_peak(outside, 1200)
    assert short >= 5
    assert long > 2 * short                       # unbounded growth, not a fixed queue


# ---- headline: double window keeps the weak stream on pace where naive does not ------

def test_double_window_keeps_weak_stream_on_pace_unlike_naive():
    """The paper's headline (Sec III.C): naive switching runs the strong decoder inline on
    the sliding chain, so an escalation stalls the whole weak stream and its backlog blows
    up; the double-window scheme runs the strong decoder in parallel, so the weak stream
    stays on pace and only the (isolated) strong pool absorbs the cost."""
    rounds, rate = 400, 0.05
    naive_decoder = SwitchingDecoder(PerRoundDecoder(F_WEAK * TAU_GEN_US),
                                     PerRoundDecoder(F_STRONG * TAU_GEN_US),
                                     gamma_switch=rate, seed=1)
    naive = build_and_run([_memory_op()], num_units=1, d=D, rounds_per_op=rounds,
                          round_us=TAU_GEN_US, scheme=SlidingWindowScheme(),
                          decoder=naive_decoder,
                          make_metrics=lambda e, c, ch, fa: [DecodeBacklog(c)], verbose=False)

    weak = SampledSoftOutputDecoder(PerRoundDecoder(F_WEAK * TAU_GEN_US), rate, seed=1)
    double = build_and_run([_memory_op()], num_units=1, d=D, rounds_per_op=rounds,
                           round_us=TAU_GEN_US, scheme=DoubleWindowScheme(g_th=0.5),
                           decoder=weak,
                           router=SwitchingRouter(weak, PerRoundDecoder(F_STRONG * TAU_GEN_US)),
                           unit_pools={"default": 1, "strong": 1},
                           make_metrics=lambda e, c, ch, fa: [DecodeBacklog(c)], verbose=False)

    naive_weak_backlog = naive["metrics"]["decode_backlog"]["peak_rounds"]
    double_weak_backlog = double["metrics"]["decode_backlog"]["peak_rounds"]
    assert naive_weak_backlog > 100              # the weak stream is hopelessly behind
    assert double_weak_backlog < naive_weak_backlog / 5    # double window keeps it on pace
