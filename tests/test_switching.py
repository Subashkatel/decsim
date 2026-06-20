"""Decoder switching (arXiv:2510.25222). Each test maps to a paper claim, validated against the
paper PDF on 2026-06-17.

Switching is one object, `Switching` (switching.py), passed to the cluster as `switching=`. It
pairs a fast weak decoder with a slow strong one: each window is decoded by the weak decoder, and
the windows it is unsure about (confidence below the threshold) are re-decoded by the strong one.
`run_both_at_once=False` (default, serial) starts the strong decoder only when a window is unsure;
`run_both_at_once=True` (parallel) starts it on every window and cancels it when the weak answer
turns out confident. The windowing stays plain SlidingWindowScheme.

Setup: a single timing-only memory operation streaming many rounds, so the weak decoder produces a
long stream of sliding windows. The weak decoder reports low confidence with a fixed probability
(SampledSoftOutputDecoder), which sets the switching rate; the strong decoder runs on its own unit
pool, so StrongDecoderBacklog reads the strong backlog directly.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from decsim.codes import SurfaceCodeModel
from decsim.decoders import (PerRoundDecoder, SampledSoftOutputDecoder, SwitchingDecoder,
                             SwitchingRouter, switch_probability_per_round)
from decsim.message import DecodeJob, Operation, Window
from decsim.metrics import DecodeBacklog, StrongDecoderBacklog
from decsim.schemes import SlidingWindowScheme
from decsim.switching import Switching
from decsim.wiring import build_and_run

TAU_GEN_US = 1.0          # syndrome round time
D = 3
F_WEAK = 0.1              # weak decode time per round / round time (well inside the keep-up bound)
F_STRONG = 10             # the paper's tau_strong = 10 * tau_gen

# Theorem 1 (Eq. 6) strong-decoder boundary for the standard commit = buffer = d windows: a
# re-decode covers 3d rounds, the strong decoder clears d / f_strong rounds per commit region, so
# the per-window switch rate must satisfy gamma <= d / (3d * f_strong) = 1 / (3 * f_strong).
# (This is the commit = d value, 1/30; Eq. 8's 0.047 substitutes the Eq.7-minimum commit instead.)
GAMMA_BOUND = D / (3 * D * F_STRONG)


def _memory_op():
    """One Clifford memory operation, with a long syndrome stream and no feedback block."""
    return Operation(0, "memory", (0,), clifford=True)


def _switch_run(switching, low_confidence_probability, rounds, seed=1, pools=None, metrics=None):
    weak = SampledSoftOutputDecoder(PerRoundDecoder(F_WEAK * TAU_GEN_US),
                                    low_confidence_probability, seed=seed)
    strong = PerRoundDecoder(F_STRONG * TAU_GEN_US)
    return build_and_run([_memory_op()], num_units=1, d=D, rounds_per_op=rounds,
                         round_us=TAU_GEN_US, scheme=SlidingWindowScheme(), switching=switching,
                         decoder=weak, router=SwitchingRouter(weak, strong),
                         unit_pools=pools or {"default": 1, "strong": 1}, make_metrics=metrics,
                         verbose=False)


def _strong_backlog_peak(low_confidence_probability, rounds, seed=1):
    res = _switch_run(Switching(confidence_threshold=0.5), low_confidence_probability, rounds,
                      seed=seed, metrics=lambda e, c, ch, fa: [StrongDecoderBacklog(c)])
    return res["metrics"]["strong_backlog"]["peak_jobs"]


# ---- window-size check: the weak decoder must keep pace (Eq. 7 of the paper) ---------

@pytest.mark.parametrize("ratio, raises", [(0.7, True), (0.4, False), (None, False)])
def test_window_size_check(ratio, raises):
    """check_window_size rejects a commit region too small for the weak decoder to keep pace. At
    d=3 (commit=buffer=3): a weak decoder at 0.7 of a round needs commit>=7 (raises); 0.4 needs >=2
    (fine); no ratio skips the check."""
    s = Switching(confidence_threshold=0.5, weak_keepup_ratio=ratio)
    if raises:
        with pytest.raises(ValueError):
            s.check_window_size(D, D)
    else:
        s.check_window_size(D, D)


def test_window_size_check_fires_through_the_engine():
    """The window-size check runs when the plan loads, so a bad configuration fails the run."""
    with pytest.raises(ValueError):
        _switch_run(Switching(confidence_threshold=0.5, weak_keepup_ratio=0.7), 0.0, 30)


def test_weak_keepup_ratio_must_be_below_one():
    """The weak decoder must be faster than one syndrome round: weak_keepup_ratio in (0, 1)."""
    with pytest.raises(ValueError):
        Switching(confidence_threshold=0.5, weak_keepup_ratio=1.0)


def test_strong_reprocess_region_is_commit_plus_two_buffers():
    """The strong decoder reprocesses commit + 2*buffer = 3d rounds for the standard d/d window."""
    commit_lo, commit_hi, buffer_hi = SlidingWindowScheme().plan_windows(
        0, 4 * D, SurfaceCodeModel(d=D))[0]
    w = Window(op_id=0, k=0, commit_lo=commit_lo, commit_hi=commit_hi,
               buffer_hi=buffer_hi, n_rounds=buffer_hi - commit_lo + 1)
    assert Switching(confidence_threshold=0.5).calculate_strong_redo_rounds(w) == 3 * D


def test_custom_decision_rule_can_replace_the_threshold():
    """The keep-or-redecode decision is overridable: a subclass that never keeps the weak result
    sends every window to the strong decoder, regardless of the reported confidence."""
    class AlwaysUseStrong(Switching):
        def keep_weak_result(self, result):
            return False

    res = _switch_run(AlwaysUseStrong(confidence_threshold=0.5), 0.0, 60)
    assert res["cluster"].strong_needed == res["cluster"].total_windows


# ---- the weak stream never stalls; with no switching it IS plain sliding -------------

def test_no_switching_matches_plain_sliding():
    """With every window confident, no window switches, so a serial-switch run is identical to the
    plain sliding-window scheme: same finish time, same committed windows, nothing sent to strong."""
    rounds = 120
    switched = _switch_run(Switching(confidence_threshold=0.5), 0.0, rounds)
    plain = build_and_run([_memory_op()], num_units=1, d=D, rounds_per_op=rounds,
                          round_us=TAU_GEN_US, scheme=SlidingWindowScheme(),
                          decoder=PerRoundDecoder(F_WEAK * TAU_GEN_US), verbose=False)
    assert switched["cluster"].strong_needed == 0
    assert switched["cluster"].strong_cancelled == 0
    assert switched["engine"].now == plain["engine"].now
    assert (len(switched["cluster"].committed_windows)
            == len(plain["cluster"].committed_windows))


# ---- Theorem 1 / Eq. 8: strong backlog bounded inside, divergent outside (serial) ----

def test_strong_backlog_bounded_inside_boundary():
    """Well inside the Theorem 1 boundary (a fraction of GAMMA_BOUND) the strong backlog stays
    bounded: its peak does not grow as the run gets longer."""
    inside = 0.15 * GAMMA_BOUND
    peaks = [_strong_backlog_peak(inside, rounds) for rounds in (300, 600, 1200)]
    assert max(peaks) <= 3
    assert peaks[-1] <= peaks[0] + 1


def test_strong_backlog_diverges_outside_boundary():
    """Well outside the boundary the strong backlog diverges: its peak grows with run length."""
    outside = 6 * GAMMA_BOUND
    short = _strong_backlog_peak(outside, 300)
    long = _strong_backlog_peak(outside, 1200)
    assert short >= 5
    assert long > 2 * short


def test_serial_keeps_weak_stream_on_pace_unlike_naive():
    """Headline (Sec III.C): naive switching runs the strong decoder inline on the sliding chain,
    so a switch stalls the whole weak stream and its backlog blows up; serial double-window
    switching runs the strong decoder in parallel, so the weak stream stays on pace."""
    rounds, rate = 400, 0.05
    naive_decoder = SwitchingDecoder(PerRoundDecoder(F_WEAK * TAU_GEN_US),
                                     PerRoundDecoder(F_STRONG * TAU_GEN_US),
                                     gamma_switch=rate, seed=1)
    naive = build_and_run([_memory_op()], num_units=1, d=D, rounds_per_op=rounds,
                          round_us=TAU_GEN_US, scheme=SlidingWindowScheme(), decoder=naive_decoder,
                          make_metrics=lambda e, c, ch, fa: [DecodeBacklog(c)], verbose=False)
    double = _switch_run(Switching(confidence_threshold=0.5), rate, rounds,
                         metrics=lambda e, c, ch, fa: [DecodeBacklog(c)])
    assert naive["metrics"]["decode_backlog"]["peak_rounds"] > 100
    assert (double["metrics"]["decode_backlog"]["peak_rounds"]
            < naive["metrics"]["decode_backlog"]["peak_rounds"] / 5)


# ---- run both at once (the paper's "both decode" mode, Sec III.A) --------------------

def test_run_both_at_once_starts_strong_every_window_and_cancels_confident_ones():
    """In "run both at once" mode the strong decoder starts on EVERY window alongside the weak one;
    the confident windows are cancelled, the unsure ones run. So every window's strong job is either
    cancelled or needed, and serial mode starts strong jobs ONLY on the windows that need them."""
    parallel = _switch_run(Switching(confidence_threshold=0.5, run_both_at_once=True), 0.3, 200,
                           pools={"default": 1, "strong": 2})
    cp = parallel["cluster"]
    assert cp.strong_cancelled > 0
    assert cp.strong_cancelled + cp.strong_needed == cp.total_windows   # one strong job per window

    serial = _switch_run(Switching(confidence_threshold=0.5), 0.3, 200,
                         pools={"default": 1, "strong": 2})
    assert serial["cluster"].strong_cancelled == 0                      # serial never cancels
    assert serial["cluster"].strong_needed == cp.strong_needed          # same windows need strong


def test_run_both_at_once_keeps_the_weak_stream_byte_identical_to_plain_sliding():
    """With the strong decoder on its own pool, cancelling its jobs never perturbs the weak stream:
    the weak windows commit at exactly the same times as plain sliding, and every window commits once."""
    rounds = 90
    parallel = _switch_run(Switching(confidence_threshold=0.5, run_both_at_once=True), 1.0, rounds,
                           pools={"default": 1, "strong": 1})
    plain = build_and_run([_memory_op()], num_units=1, d=D, rounds_per_op=rounds,
                          round_us=TAU_GEN_US, scheme=SlidingWindowScheme(),
                          decoder=PerRoundDecoder(F_WEAK * TAU_GEN_US), verbose=False)
    weak_done = lambda r: [w.t_done for _, w in sorted(r["cluster"].windows.items())]
    assert weak_done(parallel) == weak_done(plain)
    assert len(parallel["cluster"].committed_windows) == parallel["cluster"].total_windows


# ---- backlog-vs-time trace: the metrics retain a series, not just a summary ----------

def test_strong_backlog_trace_is_a_step_series_matching_the_peak():
    """StrongDecoderBacklog keeps a time series: rows() gives one (t, jobs, rounds) record per
    change, with non-decreasing times, rounds = jobs * (commit + 2*buffer) = jobs * 3d, and the
    largest sampled job count equal to the reported peak."""
    captured = {}
    def metrics(e, c, ch, fa):
        m = StrongDecoderBacklog(c)
        captured["m"] = m
        return [m]
    res = _switch_run(Switching(confidence_threshold=0.5), 6 * GAMMA_BOUND, 600, metrics=metrics)
    rows = captured["m"].rows()
    assert rows                                                    # a series was recorded
    assert [r["t"] for r in rows] == sorted(r["t"] for r in rows)  # non-decreasing in time
    assert all(rows[i]["jobs"] != rows[i - 1]["jobs"] for i in range(1, len(rows)))  # step trace
    assert all(r["rounds"] == r["jobs"] * 3 * D for r in rows)     # jobs -> rounds (commit+2*buffer)
    assert max(r["jobs"] for r in rows) == res["metrics"]["strong_backlog"]["peak_jobs"]


def test_decode_backlog_trace_tracks_the_rising_backlog():
    """DecodeBacklog keeps a time series too: under a too-slow naive (inline) decoder the backlog
    rises, so rows() is a non-empty step series whose largest value matches result()['peak_rounds']."""
    captured = {}
    def metrics(e, c, ch, fa):
        m = DecodeBacklog(c)
        captured["m"] = m
        return [m]
    naive_decoder = SwitchingDecoder(PerRoundDecoder(F_WEAK * TAU_GEN_US),
                                     PerRoundDecoder(F_STRONG * TAU_GEN_US),
                                     gamma_switch=0.05, seed=1)
    build_and_run([_memory_op()], num_units=1, d=D, rounds_per_op=400, round_us=TAU_GEN_US,
                  scheme=SlidingWindowScheme(), decoder=naive_decoder, make_metrics=metrics,
                  verbose=False)
    m = captured["m"]
    rows = m.rows()
    assert rows
    assert [r["t"] for r in rows] == sorted(r["t"] for r in rows)
    assert m.result()["peak_rounds"] > 0                          # the backlog actually moved
    assert max(r["backlog_rounds"] for r in rows) == m.result()["peak_rounds"]


# ---- configurable commit/buffer: size the window independently of d (Eq. 7) ----------

def test_commit_buffer_override_sizes_the_window_and_strong_redo():
    """SurfaceCodeModel commit/buffer default to d but are overridable. A 5-commit/2-buffer code
    lays the first window out as commit 1-5 + 2 buffer, and the strong decoder reprocesses
    commit + 2*buffer = 9 rounds."""
    code = SurfaceCodeModel(d=3, commit_rounds_override=5, buffer_rounds_override=2)
    assert (code.commit_rounds(), code.buffer_rounds()) == (5, 2)
    commit_lo, commit_hi, buffer_hi = SlidingWindowScheme().plan_windows(0, 20, code)[0]
    assert (commit_lo, commit_hi, buffer_hi) == (1, 5, 7)
    w = Window(op_id=0, k=0, commit_lo=commit_lo, commit_hi=commit_hi,
               buffer_hi=buffer_hi, n_rounds=buffer_hi - commit_lo + 1)
    assert Switching(confidence_threshold=0.5).calculate_strong_redo_rounds(w) == 5 + 2 * 2


def test_commit_buffer_override_defaults_to_d_and_rejects_nonpositive():
    """No override means commit = buffer = d (unchanged); a non-positive override is rejected."""
    assert (SurfaceCodeModel(d=5).commit_rounds(), SurfaceCodeModel(d=5).buffer_rounds()) == (5, 5)
    with pytest.raises(ValueError):
        SurfaceCodeModel(d=3, commit_rounds_override=0)


# ---- size-dependent switch probability (paper Sec III.C / Fig 10) --------------------

def test_switch_probability_per_round_scales_with_commit_rounds():
    """switch_probability_per_round gives gamma * commit_rounds / d: equal to gamma at commit = d,
    scaled up for a larger commit region, and scaled by n_rounds for a window-less batch job."""
    p = switch_probability_per_round(0.1, D)
    w_d = Window(op_id=0, k=0, commit_lo=1, commit_hi=D, buffer_hi=2 * D, n_rounds=2 * D)
    assert p(DecodeJob(op_id=0, window_id=0, n_rounds=2 * D, window=w_d)) == pytest.approx(0.1)
    w_2d = Window(op_id=0, k=0, commit_lo=1, commit_hi=2 * D, buffer_hi=3 * D, n_rounds=3 * D)
    assert p(DecodeJob(op_id=0, window_id=0, n_rounds=3 * D, window=w_2d)) == pytest.approx(0.2)
    assert p(DecodeJob(op_id=0, window_id=0, n_rounds=4 * D, window=None)) == pytest.approx(0.4)


def test_sampled_soft_output_uses_the_probability_for_callback():
    """SampledSoftOutputDecoder consults probability_for per job: a rule returning 1.0 escalates
    every window, overriding the flat escalation_probability of 0.0."""
    weak = SampledSoftOutputDecoder(PerRoundDecoder(F_WEAK * TAU_GEN_US), 0.0,
                                    probability_for=lambda job: 1.0)
    strong = PerRoundDecoder(F_STRONG * TAU_GEN_US)
    res = build_and_run([_memory_op()], num_units=1, d=D, rounds_per_op=60, round_us=TAU_GEN_US,
                        scheme=SlidingWindowScheme(), switching=Switching(confidence_threshold=0.5),
                        decoder=weak, router=SwitchingRouter(weak, strong),
                        unit_pools={"default": 1, "strong": 1}, verbose=False)
    assert res["cluster"].strong_needed == res["cluster"].total_windows
