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
(SampledConfidenceDecoder), which sets the switching rate; the strong decoder runs on its own unit
pool, so StrongDecoderBacklog reads the strong backlog directly.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from decsim.codes import SurfaceCodeModel
from decsim.config import us
from decsim.decoders import (PerRoundDecoder, SampledConfidenceDecoder, SwitchingDecoder,
                             SwitchingRouter, switch_probability_per_round)
from decsim.message import DecodeJob, DecodeResult, Operation, Window
from decsim.metrics import DecodeBacklog, StrongDecoderBacklog
from decsim.schemes import SlidingWindowScheme
from decsim.switching import Switching
from decsim.run_spec import RunSpec, simulate
from decsim.planner import FixedRounds

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
    weak = SampledConfidenceDecoder(PerRoundDecoder(F_WEAK * TAU_GEN_US),
                                    low_confidence_probability, seed=seed)
    strong = PerRoundDecoder(F_STRONG * TAU_GEN_US)
    return simulate(RunSpec(
               ops=[_memory_op()],
               num_units=1,
               d=D,
               rounds_policy=FixedRounds(rounds),
               round_us=TAU_GEN_US,
               scheme=SlidingWindowScheme(),
               strategy=switching,
               decoder=weak,
               router=SwitchingRouter(weak, strong),
               unit_pools=pools or {"default": 1, "strong": 1},
               make_metrics=metrics,
           ), verbose=False)


def _strong_backlog_peak(low_confidence_probability, rounds, seed=1):
    res = _switch_run(Switching(confidence_threshold=0.5), low_confidence_probability, rounds,
                      seed=seed, metrics=lambda e, c, ch, fa: [StrongDecoderBacklog(c)])
    return res["metrics"]["strong_backlog"]["peak_jobs"]


class FixedLogicalDecoder:
    """Small test decoder with fixed timing and fixed logical output."""

    def __init__(self, logical_value: int, tau_us: float = 0.01):
        self.logical_value = logical_value
        self.tau_us = tau_us

    def latency(self, job: DecodeJob) -> int:
        """Scale latency with the submitted round count."""
        from decsim.config import us

        return us(job.n_rounds * self.tau_us)

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Return the configured logical value."""
        return DecodeResult(job.op_id, job.window_id, logical_value=self.logical_value)


class RecordingLogicalDecoder(FixedLogicalDecoder):
    """Fixed decoder that records every job it decodes."""

    def __init__(self, logical_value: int, tau_us: float = 0.01):
        super().__init__(logical_value, tau_us)
        self.jobs = []

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Record the job before returning the configured logical value."""
        self.jobs.append(job)
        return super().decode(job)


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


def test_window_size_check_accepts_exact_paper_boundary():
    """Eq. 7 accepts equality. The guard must not reject it because of float roundoff."""
    switching = Switching(confidence_threshold=0.5, weak_keepup_ratio=0.9)
    for buffer_rounds in (3, 5, 7):
        switching.check_window_size(9 * buffer_rounds, buffer_rounds)

    with pytest.raises(ValueError):
        switching.check_window_size(26, 3)


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
    assert Switching(confidence_threshold=0.5).strong_redo_rounds(w) == 3 * D


def test_custom_decision_rule_can_replace_the_threshold():
    """The keep-or-redecode decision is overridable: a subclass that never keeps the weak result
    sends every window to the strong decoder, regardless of the reported confidence."""
    class AlwaysUseStrong(Switching):
        def keep_weak_result(self, result):
            return False

    res = _switch_run(AlwaysUseStrong(confidence_threshold=0.5), 0.0, 60)
    assert res["cluster"].strong_needed == res["cluster"].total_windows


def test_escalated_window_uses_strong_logical_result():
    """A low-confidence window advances with the weak boundary but finalizes with strong logic."""
    weak = SampledConfidenceDecoder(FixedLogicalDecoder(logical_value=0),
                                    escalation_probability=1.0, seed=1)
    strong = FixedLogicalDecoder(logical_value=1)
    res = simulate(RunSpec(
              ops=[_memory_op()],
              num_units=1,
              d=D,
              rounds_policy=FixedRounds(27),
              round_us=TAU_GEN_US,
              scheme=SlidingWindowScheme(),
              strategy=Switching(confidence_threshold=0.5),
              decoder=weak,
              router=SwitchingRouter(weak, strong),
              unit_pools={"default": 1, "strong": 1},
          ), verbose=False)
    cluster = res["cluster"]
    assert cluster.total_windows % 2 == 1
    assert cluster.strong_needed == cluster.total_windows
    assert cluster.op_results[0] == 1


def test_parallel_strong_result_can_finish_before_weak_decision():
    """Run-both mode can store an early strong result until the weak decoder decides to switch."""
    weak = SampledConfidenceDecoder(FixedLogicalDecoder(logical_value=0, tau_us=0.05),
                                    escalation_probability=1.0, seed=1)
    strong = FixedLogicalDecoder(logical_value=1, tau_us=0.0)
    res = simulate(RunSpec(
              ops=[_memory_op()],
              num_units=1,
              d=D,
              rounds_policy=FixedRounds(27),
              round_us=TAU_GEN_US,
              scheme=SlidingWindowScheme(),
              strategy=Switching(confidence_threshold=0.5, run_both_at_once=True),
              decoder=weak,
              router=SwitchingRouter(weak, strong),
              unit_pools={"default": 1, "strong": 1},
          ), verbose=False)
    assert res["cluster"].op_results[0] == 1


def test_switching_requires_a_router_to_reach_the_strong_decoder():
    """Switching must not silently route strong jobs back to the weak decoder."""
    weak = SampledConfidenceDecoder(FixedLogicalDecoder(logical_value=0),
                                    escalation_probability=1.0, seed=1)
    with pytest.raises(RuntimeError, match="SwitchingRouter"):
        simulate(RunSpec(
            ops=[_memory_op()],
            num_units=1,
            d=D,
            rounds_policy=FixedRounds(9),
            round_us=TAU_GEN_US,
            scheme=SlidingWindowScheme(),
            strategy=Switching(confidence_threshold=0.5),
            decoder=weak,
            unit_pools={"default": 1, "strong": 1},
        ), verbose=False)


def test_strong_redecode_receives_two_sided_context_payloads():
    """The strong job is charged and fed commit + leading buffer + trailing buffer."""
    weak = SampledConfidenceDecoder(RecordingLogicalDecoder(logical_value=0),
                                    escalation_probability=1.0, seed=1)
    strong = RecordingLogicalDecoder(logical_value=1)
    simulate(RunSpec(
        ops=[_memory_op()],
        num_units=1,
        d=D,
        rounds_policy=FixedRounds(9),
        round_us=TAU_GEN_US,
        scheme=SlidingWindowScheme(),
        strategy=Switching(confidence_threshold=0.5),
        decoder=weak,
        router=SwitchingRouter(weak, strong),
        unit_pools={"default": 1, "strong": 1},
    ), verbose=False)

    middle_job = next(job for job in strong.jobs if job.window_id == 1)
    assert middle_job.n_rounds == 3 * D
    assert middle_job.window.start_round == 1
    assert middle_job.window.commit_lo == 4
    assert middle_job.window.commit_hi == 6
    assert middle_job.window.buffer_hi == 9
    assert [payload.round_index for payload in middle_job.payloads] == list(range(1, 10))


def test_stim_strong_redecode_receives_two_sided_window_model():
    """Stim-backed strong re-decodes get an enlarged detector model, not the weak slice."""
    stim = pytest.importorskip("stim")
    from decsim.adapters.stim_device import StimDevice
    from decsim.planner import FixedRounds

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=D, rounds=9,
        after_clifford_depolarization=0.001,
        after_reset_flip_probability=0.001,
        before_measure_flip_probability=0.001,
        before_round_data_depolarization=0.001)
    op = Operation(0, "memory", (0,), clifford=True, circuit=circuit)
    weak_inner = RecordingLogicalDecoder(logical_value=0)
    weak = SampledConfidenceDecoder(weak_inner, escalation_probability=1.0, seed=1)
    strong = RecordingLogicalDecoder(logical_value=1)

    simulate(RunSpec(
        ops=[op],
        num_units=1,
        d=D,
        rounds_policy=FixedRounds(9),
        round_us=TAU_GEN_US,
        scheme=SlidingWindowScheme(),
        strategy=Switching(confidence_threshold=0.5),
        device=StimDevice(seed=3),
        decoder=weak,
        router=SwitchingRouter(weak, strong),
        unit_pools={"default": 1, "strong": 1},
    ), verbose=False)

    weak_middle = next(job for job in weak_inner.jobs if job.window_id == 1)
    strong_middle = next(job for job in strong.jobs if job.window_id == 1)
    assert strong_middle.dem is not None
    assert weak_middle.dem is not None
    assert strong_middle.window.start_round == 1
    assert strong_middle.window.buffer_hi == 9
    assert len(strong_middle.dem.detector_ids) > len(weak_middle.dem.detector_ids)


# ---- the weak stream never stalls; with no switching it IS plain sliding -------------

def test_no_switching_matches_plain_sliding():
    """With every window confident, no window switches, so a serial-switch run is identical to the
    plain sliding-window scheme: same finish time, same committed windows, nothing sent to strong."""
    rounds = 120
    switched = _switch_run(Switching(confidence_threshold=0.5), 0.0, rounds)
    plain = simulate(RunSpec(
                ops=[_memory_op()],
                num_units=1,
                d=D,
                rounds_policy=FixedRounds(rounds),
                round_us=TAU_GEN_US,
                scheme=SlidingWindowScheme(),
                decoder=PerRoundDecoder(F_WEAK * TAU_GEN_US),
            ), verbose=False)
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
    naive = simulate(RunSpec(
                ops=[_memory_op()],
                num_units=1,
                d=D,
                rounds_policy=FixedRounds(rounds),
                round_us=TAU_GEN_US,
                scheme=SlidingWindowScheme(),
                decoder=naive_decoder,
                make_metrics=lambda e, c, ch, fa: [DecodeBacklog(c)],
            ), verbose=False)
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
    plain = simulate(RunSpec(
                ops=[_memory_op()],
                num_units=1,
                d=D,
                rounds_policy=FixedRounds(rounds),
                round_us=TAU_GEN_US,
                scheme=SlidingWindowScheme(),
                decoder=PerRoundDecoder(F_WEAK * TAU_GEN_US),
            ), verbose=False)
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
    simulate(RunSpec(
        ops=[_memory_op()],
        num_units=1,
        d=D,
        rounds_policy=FixedRounds(400),
        round_us=TAU_GEN_US,
        scheme=SlidingWindowScheme(),
        decoder=naive_decoder,
        make_metrics=metrics,
    ), verbose=False)
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
    assert Switching(confidence_threshold=0.5).strong_redo_rounds(w) == 5 + 2 * 2


def test_commit_buffer_override_defaults_to_d_and_rejects_nonpositive():
    """No override means commit = buffer = d (unchanged); a non-positive override is rejected."""
    assert (SurfaceCodeModel(d=5).commit_rounds(), SurfaceCodeModel(d=5).buffer_rounds()) == (5, 5)
    with pytest.raises(ValueError):
        SurfaceCodeModel(d=3, commit_rounds_override=0)


# ---- configurable switch probability helper ------------------------------------------

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
    """SampledConfidenceDecoder consults probability_for per job: a rule returning 1.0 escalates
    every window, overriding the flat escalation_probability of 0.0."""
    weak = SampledConfidenceDecoder(PerRoundDecoder(F_WEAK * TAU_GEN_US), 0.0,
                                    probability_for=lambda job: 1.0)
    strong = PerRoundDecoder(F_STRONG * TAU_GEN_US)
    res = simulate(RunSpec(
              ops=[_memory_op()],
              num_units=1,
              d=D,
              rounds_policy=FixedRounds(60),
              round_us=TAU_GEN_US,
              scheme=SlidingWindowScheme(),
              strategy=Switching(confidence_threshold=0.5),
              decoder=weak,
              router=SwitchingRouter(weak, strong),
              unit_pools={"default": 1, "strong": 1},
          ), verbose=False)
    assert res["cluster"].strong_needed == res["cluster"].total_windows


# ---- faithful double window (paper Sec. III C, Fig. 12) --------------------
#
# On a switching event the slab of commit + 2*buffer rounds starts AT the
# suspicious commit and extends FORWARD (Fig. 12 panel 4); the weak chain
# SKIPS the windows whose commit regions the slab absorbs and restarts on the
# first window past the slab (panel 5); the strong decoder commits the whole
# slab, and it starts only once the weak decoder has determined the boundary
# conditions at both slab ends: the escalated window's own entry defects on
# the left, the restart window's weak commit on the right (panel 6), or the
# terminal boundary at the stream end. The weak pipeline never waits.

class _DispatchRecorder:
    """Record the tick a decoder unit STARTS each job: the pool calls
    latency() exactly once, at dispatch."""

    def __init__(self, inner, env):
        self.inner = inner
        self.env = env
        self.starts = []

    def latency(self, job):
        self.starts.append((self.env["engine"].now, job))
        return self.inner.latency(job)

    def decode(self, job):
        return self.inner.decode(job)


def _double_window_run(escalate_window, rounds=30, strong_tau=F_STRONG):
    """One memory op on sliding d/d windows; exactly the window with index
    escalate_window reports low confidence (deterministic, no sampling)."""
    env = {}
    weak = _DispatchRecorder(SampledConfidenceDecoder(
        PerRoundDecoder(F_WEAK * TAU_GEN_US), 0.0,
        probability_for=lambda job: 1.0 if job.window_id == escalate_window
        else 0.0), env)
    strong = _DispatchRecorder(PerRoundDecoder(strong_tau * TAU_GEN_US), env)
    res = simulate(RunSpec(
              ops=[_memory_op()],
              num_units=1,
              d=D,
              rounds_policy=FixedRounds(rounds),
              round_us=TAU_GEN_US,
              scheme=SlidingWindowScheme(),
              strategy=Switching(confidence_threshold=0.5,
                                 double_window=True),
              decoder=weak,
              router=SwitchingRouter(weak, strong),
              unit_pools={"default": 1, "strong": 1},
              make_metrics=lambda e, c, ch, fa: env.update(engine=e) or [],
          ), verbose=False)
    return res, weak, strong


def test_double_window_slab_extends_forward_and_absorbs_the_weak_chain():
    """Fig. 12 panels 4-5: W2 (commit 7-9) escalates; the slab is rounds
    7-15 (commit + two buffers FORWARD), the windows committing 10-12 and
    13-15 are never weak-decoded, and the weak chain restarts at W5."""
    res, weak, strong = _double_window_run(escalate_window=2)
    runtime = res["cluster"].window_manager
    (start_tick, job), = strong.starts
    assert job.strong_decode_for == (0, 2)
    assert (job.window.commit_lo, job.window.commit_hi) == (7, 15)
    assert job.n_rounds == 3 * D                    # the paper's r_strong
    assert runtime.absorbed_windows == {(0, 3), (0, 4)}
    weak_windows_decoded = [j.window_id for _, j in weak.starts]
    assert weak_windows_decoded == [0, 1, 2, 5, 6, 7, 8, 9]
    for absorbed_key in runtime.absorbed_windows:
        window = runtime.windows[absorbed_key]
        assert window.committed and window.t_done is None


def test_double_window_strong_waits_for_the_restart_windows_weak_commit():
    """Fig. 12 panel 6: the slab starts only after the restart window's
    (W5's) weak decode commits the far-side boundary, plus the weak->strong
    hop. Without the gate the slab would start right after W2's outcome."""
    res, weak, strong = _double_window_run(escalate_window=2)
    cluster = res["cluster"]
    w5 = cluster.windows[(0, 5)]
    (start_tick, job), = strong.starts
    assert w5.t_done is not None
    assert start_tick == w5.t_done + us(0.5)
    assert cluster.window_manager.pending_escalations == {}


def test_double_window_weak_pipeline_never_stalls_on_strong_work():
    """The weak chain's schedule is independent of the strong decoder: a
    10x slower strong decoder changes no weak window's decode time."""
    fast, _, _ = _double_window_run(escalate_window=2, strong_tau=F_STRONG)
    slow, _, _ = _double_window_run(escalate_window=2,
                                    strong_tau=10 * F_STRONG)
    assert {k: w.t_done for k, w in fast["cluster"].windows.items()} \
        == {k: w.t_done for k, w in slow["cluster"].windows.items()}


def test_double_window_last_window_uses_the_terminal_boundary():
    """The final window's slab clamps at the stream end and has no restart
    window: the terminal time boundary already exists, so the slab is
    submitted at escalation without any extra wait."""
    res, weak, strong = _double_window_run(escalate_window=9)  # last of 10
    cluster = res["cluster"]
    w9 = cluster.windows[(0, 9)]
    (start_tick, job), = strong.starts
    assert (job.window.commit_lo, job.window.commit_hi) == (28, 30)
    assert job.n_rounds == 3                        # clamped r_strong
    assert start_tick == w9.t_done + us(0.5)
    assert cluster.window_manager.absorbed_windows == set()


def test_double_window_end_clamped_slab_absorbs_the_tail():
    """W8 (commit 25-27) escalates: the slab clamps to rounds 25-30,
    absorbs the final window, and needs no restart gate."""
    res, weak, strong = _double_window_run(escalate_window=8)
    runtime = res["cluster"].window_manager
    (start_tick, job), = strong.starts
    assert (job.window.commit_lo, job.window.commit_hi) == (25, 30)
    assert job.n_rounds == 6
    assert runtime.absorbed_windows == {(0, 9)}
    assert [j.window_id for _, j in weak.starts] == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    w8 = res["cluster"].windows[(0, 8)]
    assert start_tick == w8.t_done + us(0.5)


def test_double_window_exactly_one_strong_job_per_escalation():
    """One switching event creates exactly one strong job; a duplicate
    registration for the same window is an illegal transition."""
    res, weak, strong = _double_window_run(escalate_window=2)
    assert len(strong.starts) == 1
    runtime = res["cluster"].window_manager
    runtime._deferred_strong[(0, 2)] = {"state": "waiting_far_boundary"}
    with pytest.raises(RuntimeError, match="duplicate strong escalation"):
        runtime.defer_strong_escalation(
            DecodeJob(op_id=0, window_id=2, n_rounds=9, ready_time=0,
                      deadline=0), 9, "dup")


def test_double_window_strong_result_owns_the_whole_slab():
    """Fig. 12 panels 6-8: the slab's logical value replaces the escalated
    window's weak value, and the absorbed windows contribute nothing. Eight
    weak windows at logical 1 XOR to 0; replacing W2's 1 with the strong 0
    leaves seven ones -> 1."""
    weak = SampledConfidenceDecoder(
        FixedLogicalDecoder(logical_value=1), 0.0,
        probability_for=lambda job: 1.0 if job.window_id == 2 else 0.0)
    strong = FixedLogicalDecoder(logical_value=0)
    res = simulate(RunSpec(
              ops=[_memory_op()],
              num_units=1,
              d=D,
              rounds_policy=FixedRounds(30),
              round_us=TAU_GEN_US,
              scheme=SlidingWindowScheme(),
              strategy=Switching(confidence_threshold=0.5, double_window=True),
              decoder=weak,
              router=SwitchingRouter(weak, strong),
              unit_pools={"default": 1, "strong": 1},
          ), verbose=False)
    assert res["cluster"].op_results[0] == 1
    assert res["cluster"].strong_needed == 1


def test_double_window_slab_payloads_cover_slab_plus_trailing_context():
    """Slab payloads are assembled at submission: the committed slab rounds
    7-15 plus one trailing buffer of read-only context (16-18), the same
    role a buffer plays for every weak window; the timing charge stays the
    paper's r_strong (asserted above via n_rounds)."""
    res, weak, strong = _double_window_run(escalate_window=2)
    (start_tick, job), = strong.starts
    assert job.window.buffer_hi == 18
    assert [payload.round_index for payload in job.payloads] \
        == list(range(7, 19))


def test_double_window_rejects_contradictory_switch_flags():
    with pytest.raises(ValueError, match="double_window"):
        Switching(confidence_threshold=0.5, double_window=True,
                  run_both_at_once=True)
    with pytest.raises(ValueError, match="double_window"):
        Switching(confidence_threshold=0.5, double_window=True,
                  bulk_strong=True)


def test_double_window_rejects_unsupported_runspec_shapes():
    """Held would deadlock the far-boundary wait; parallel two-layer windows,
    runtime streams, frontends, and cross-op window chains need skip
    semantics the runtime does not model yet."""
    from decsim.policies import Held
    from decsim.schemes import ParallelWindowScheme
    strategy = Switching(confidence_threshold=0.5, double_window=True)
    base = dict(num_units=1, d=D, rounds_policy=FixedRounds(30),
                round_us=TAU_GEN_US, strategy=strategy)
    with pytest.raises(ValueError, match="Held"):
        RunSpec(ops=[_memory_op()], scheme=SlidingWindowScheme(),
                boundary_policy=Held(), **base).validate()
    with pytest.raises(ValueError, match="parallel"):
        RunSpec(ops=[_memory_op()], scheme=ParallelWindowScheme(),
                **base).validate()
    with pytest.raises(ValueError, match="dynamic_streams"):
        RunSpec(ops=[_memory_op()],
                dynamic_streams=[Operation(7, "stream", (1,))],
                **base).validate()
    chained = [Operation(0, "a", (0,), has_successor=True),
               Operation(1, "b", (1,), predecessors=(0,))]
    with pytest.raises(ValueError, match="single-patch"):
        RunSpec(ops=chained, **base).validate()
