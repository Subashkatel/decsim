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
import math
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from decsim.codes import SurfaceCodeModel
from decsim.config import us
from decsim.decoders import (
    PerRoundDecoder,
    SAMPLED_CONFIDENCE_SOURCE,
    SampledConfidenceDecoder,
    SwitchingDecoder,
    SwitchingRouter,
    switch_probability_per_round,
)
from decsim.devices import TimingOnlyDevice
from decsim.message import (DecodeJob, DecodeResult, Operation,
                            ResolvedCodeGeometry,
                            SeamFaultOwner, StrongRegionPlan, Window,
                            WindowInfo)
from decsim.metrics import DecodeBacklog, StrongDecoderBacklog
from decsim.schemes import SlidingWindowScheme
from decsim.switching import Switching
from decsim.run_spec import RunSpec, simulate
from decsim.planner import FixedRounds
from decsim.window_interactions import DefaultWindowInteraction
from decsim.window_manager import (
    WindowManager,
    _EscalationPhase,
    _EscalationRegistry,
    _PendingEscalation,
    _ResolvedStrongRegion,
)

TAU_GEN_US = 1.0          # syndrome round time
D = 3


def _code_geometry(commit_rounds, buffer_rounds):
    return ResolvedCodeGeometry(
        code_name="test",
        distance=D,
        commit_round_count=commit_rounds,
        buffer_round_count=buffer_rounds,
        minimum_leading_buffer_round_count=D,
        minimum_trailing_buffer_round_count=D,
        one_patch_spatial_node_count=D * D,
        buffer_floor_override_active=True,
    )
F_WEAK = 0.1              # weak decode time per round / round time (well inside the keep-up bound)
F_STRONG = 10             # the paper's tau_strong = 10 * tau_gen


class _IntSubclass(int):
    pass

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
                                    low_confidence_probability)
    strong = PerRoundDecoder(F_STRONG * TAU_GEN_US)
    return simulate(RunSpec(
               ops=[_memory_op()],
               num_units=1,
               d=D,
               rounds_policy=FixedRounds(rounds),
               round_us=TAU_GEN_US,
               scheme=SlidingWindowScheme(),
               strategy=switching,
               router=SwitchingRouter(weak, strong),
               unit_pools=pools or {"default": 1, "strong": 1},
               make_metrics=metrics,
               seed=seed,
           ), verbose=False)


def _strong_backlog_peak(low_confidence_probability, rounds, seed=1):
    res = _switch_run(Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5), low_confidence_probability, rounds,
                      seed=seed, metrics=lambda e, wm, dm, ch, fa: [StrongDecoderBacklog(wm, dm)])
    return res.result.metric_values()["strong_backlog"]["peak_jobs"]


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
        return DecodeResult(
            job.op_id,
            job.window_id,
            logical_observables=(self.logical_value,),
        )


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
    s = Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5, weak_keepup_ratio=ratio)
    if raises:
        with pytest.raises(ValueError):
            s.validate_code_geometry(_code_geometry(D, D))
    else:
        s.validate_code_geometry(_code_geometry(D, D))


def test_window_size_check_fires_through_the_engine():
    """The window-size check runs when the plan loads, so a bad configuration fails the run."""
    with pytest.raises(ValueError):
        _switch_run(Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5, weak_keepup_ratio=0.7), 0.0, 30)


def test_window_size_check_accepts_exact_paper_boundary():
    """Eq. 7 accepts equality. The guard must not reject it because of float roundoff."""
    switching = Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5, weak_keepup_ratio=0.9)
    for buffer_rounds in (3, 5, 7):
        switching.validate_code_geometry(
            _code_geometry(9 * buffer_rounds, buffer_rounds)
        )

    with pytest.raises(ValueError):
        switching.validate_code_geometry(_code_geometry(26, 3))


def test_window_size_check_distinguishes_adjacent_floats_at_the_boundary():
    boundary = D / (D + D)
    Switching(
        expected_source=SAMPLED_CONFIDENCE_SOURCE,
        confidence_threshold=0.5,
        weak_keepup_ratio=boundary,
    ).validate_code_geometry(_code_geometry(D, D))
    Switching(
        expected_source=SAMPLED_CONFIDENCE_SOURCE,
        confidence_threshold=0.5,
        weak_keepup_ratio=math.nextafter(boundary, 0.0),
    ).validate_code_geometry(_code_geometry(D, D))
    with pytest.raises(ValueError):
        Switching(
            expected_source=SAMPLED_CONFIDENCE_SOURCE,
            confidence_threshold=0.5,
            weak_keepup_ratio=math.nextafter(boundary, 1.0),
        ).validate_code_geometry(_code_geometry(D, D))


def test_weak_keepup_ratio_must_be_below_one():
    """The weak decoder must be faster than one syndrome round: weak_keepup_ratio in (0, 1)."""
    with pytest.raises(ValueError):
        Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5, weak_keepup_ratio=1.0)


def test_strong_reprocess_region_is_commit_plus_two_buffers():
    """The strong decoder reprocesses commit + 2*buffer = 3d rounds for the standard d/d window."""
    geometry = SlidingWindowScheme().plan_operation(
        0,
        4 * D,
        commit_round_count=D,
        buffer_round_count=D,
    ).windows[0]
    commit_lo, commit_hi, buffer_hi = (
        geometry.commit_lo,
        geometry.commit_hi,
        geometry.buffer_hi,
    )
    w = Window(op_id=0, k=0, commit_lo=commit_lo, commit_hi=commit_hi,
               buffer_hi=buffer_hi, n_rounds=buffer_hi - commit_lo + 1)
    assert Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5).strong_redo_rounds(w) == 3 * D


def test_custom_decision_rule_can_replace_the_threshold():
    """The keep-or-redecode decision is overridable: a subclass that never keeps the weak result
    sends every window to the strong decoder, regardless of the reported confidence."""
    class AlwaysUseStrong(Switching):
        def keep_weak_result(self, result, job):
            return False

    res = _switch_run(
        AlwaysUseStrong(
            expected_source=SAMPLED_CONFIDENCE_SOURCE,
            confidence_threshold=0.5,
        ),
        0.0,
        60,
    )
    assert res.decoder_manager.strong_needed == res.window_manager.total_windows


def test_escalated_window_uses_strong_logical_result():
    """A low-confidence window advances with the weak boundary but finalizes with strong logic."""
    weak = SampledConfidenceDecoder(FixedLogicalDecoder(logical_value=0),
                                    escalation_probability=1.0)
    strong = FixedLogicalDecoder(logical_value=1)
    res = simulate(RunSpec(
              ops=[_memory_op()],
              num_units=1,
              d=D,
              rounds_policy=FixedRounds(27),
              round_us=TAU_GEN_US,
              scheme=SlidingWindowScheme(),
              strategy=Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5),
              router=SwitchingRouter(weak, strong),
              unit_pools={"default": 1, "strong": 1},
              seed=1,
          ), verbose=False)
    assert res.window_manager.total_windows % 2 == 1
    assert res.decoder_manager.strong_needed == res.window_manager.total_windows
    assert res.window_manager.op_results[0] == (1,)


def test_parallel_strong_result_can_finish_before_weak_decision():
    """Run-both mode can store an early strong result until the weak decoder decides to switch."""
    weak = SampledConfidenceDecoder(FixedLogicalDecoder(logical_value=0, tau_us=0.05),
                                    escalation_probability=1.0)
    strong = FixedLogicalDecoder(logical_value=1, tau_us=0.0)
    res = simulate(RunSpec(
              ops=[_memory_op()],
              num_units=1,
              d=D,
              rounds_policy=FixedRounds(27),
              round_us=TAU_GEN_US,
              scheme=SlidingWindowScheme(),
              strategy=Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5, run_both_at_once=True),
              router=SwitchingRouter(weak, strong),
              unit_pools={"default": 1, "strong": 1},
              seed=1,
          ), verbose=False)
    assert res.window_manager.op_results[0] == (1,)


def test_switching_requires_a_router_to_reach_the_strong_decoder():
    """Switching must not silently route strong jobs back to the weak decoder."""
    weak = SampledConfidenceDecoder(FixedLogicalDecoder(logical_value=0),
                                    escalation_probability=1.0)
    with pytest.raises(RuntimeError, match="SwitchingRouter"):
        simulate(RunSpec(
            ops=[_memory_op()],
            num_units=1,
            d=D,
            rounds_policy=FixedRounds(9),
            round_us=TAU_GEN_US,
            scheme=SlidingWindowScheme(),
            strategy=Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5),
            decoder=weak,
            unit_pools={"default": 1, "strong": 1},
            seed=1,
        ), verbose=False)


def test_strong_redecode_receives_two_sided_context_payloads():
    """The strong job is charged and fed commit + leading buffer + trailing buffer."""
    weak = SampledConfidenceDecoder(RecordingLogicalDecoder(logical_value=0),
                                    escalation_probability=1.0)
    strong = RecordingLogicalDecoder(logical_value=1)
    simulate(RunSpec(
        ops=[_memory_op()],
        num_units=1,
        d=D,
        rounds_policy=FixedRounds(9),
        round_us=TAU_GEN_US,
        scheme=SlidingWindowScheme(),
        strategy=Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5),
        router=SwitchingRouter(weak, strong),
        unit_pools={"default": 1, "strong": 1},
        seed=1,
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
    weak = SampledConfidenceDecoder(weak_inner, escalation_probability=1.0)
    strong = RecordingLogicalDecoder(logical_value=1)

    simulate(RunSpec(
        ops=[op],
        num_units=1,
        d=D,
        rounds_policy=FixedRounds(9),
        round_us=TAU_GEN_US,
        scheme=SlidingWindowScheme(),
        strategy=Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5),
        device=StimDevice(),
        router=SwitchingRouter(weak, strong),
        unit_pools={"default": 1, "strong": 1},
        seed=3,
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
    switched = _switch_run(Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5), 0.0, rounds)
    plain = simulate(RunSpec(
                ops=[_memory_op()],
                num_units=1,
                d=D,
                rounds_policy=FixedRounds(rounds),
                round_us=TAU_GEN_US,
                scheme=SlidingWindowScheme(),
                decoder=PerRoundDecoder(F_WEAK * TAU_GEN_US),
            ), verbose=False)
    assert switched.decoder_manager.strong_needed == 0
    assert switched.decoder_manager.strong_cancelled == 0
    assert switched.engine.now == plain.engine.now
    assert (len(switched.window_manager.committed_windows)
            == len(plain.window_manager.committed_windows))


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
                                     gamma_switch=rate)
    naive = simulate(RunSpec(
                ops=[_memory_op()],
                num_units=1,
                d=D,
                rounds_policy=FixedRounds(rounds),
                round_us=TAU_GEN_US,
                scheme=SlidingWindowScheme(),
                decoder=naive_decoder,
                make_metrics=lambda e, wm, dm, ch, fa: [DecodeBacklog(wm, dm)],
                seed=1,
            ), verbose=False)
    double = _switch_run(Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5), rate, rounds,
                         metrics=lambda e, wm, dm, ch, fa: [DecodeBacklog(wm, dm)])
    assert naive.result.metric_values()["decode_backlog"]["peak_rounds"] > 100
    assert (double.result.metric_values()["decode_backlog"]["peak_rounds"]
            < naive.result.metric_values()["decode_backlog"]["peak_rounds"] / 5)


# ---- run both at once (the paper's "both decode" mode, Sec III.A) --------------------

def test_run_both_at_once_starts_strong_every_window_and_cancels_confident_ones():
    """In "run both at once" mode the strong decoder starts on EVERY window alongside the weak one;
    the confident windows are cancelled, the unsure ones run. So every window's strong job is either
    cancelled or needed, and serial mode starts strong jobs ONLY on the windows that need them."""
    parallel = _switch_run(Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5, run_both_at_once=True), 0.3, 200,
                           pools={"default": 1, "strong": 2})
    cp, windows = parallel.decoder_manager, parallel.window_manager.total_windows
    assert cp.strong_cancelled > 0
    assert cp.strong_cancelled + cp.strong_needed == windows   # one strong job per window

    serial = _switch_run(Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5), 0.3, 200,
                         pools={"default": 1, "strong": 2})
    assert serial.decoder_manager.strong_cancelled == 0                      # serial never cancels
    assert serial.decoder_manager.strong_needed == cp.strong_needed          # same windows need strong


def test_run_both_at_once_keeps_the_weak_decode_work_identical_to_plain_sliding():
    """A separate strong pool leaves weak compute unchanged; selected transport may
    shift later readiness because unconfident WSD and accepted WDO are different paths."""
    rounds = 90
    parallel = _switch_run(Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5, run_both_at_once=True), 1.0, rounds,
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
    weak_durations = lambda r: [
        w.t_done - w.t_dispatch
        for _, w in sorted(r.window_manager.windows.items())
    ]
    assert weak_durations(parallel) == weak_durations(plain)
    assert len(parallel.window_manager.committed_windows) == parallel.window_manager.total_windows


# ---- backlog-vs-time trace: the metrics retain a series, not just a summary ----------

def test_strong_backlog_trace_is_a_step_series_matching_the_peak():
    """The strong trace is ordered and its exact job/round peaks reconcile."""
    captured = {}
    def metrics(e, wm, dm, ch, fa):
        m = StrongDecoderBacklog(wm, dm)
        captured["m"] = m
        return [m]
    res = _switch_run(Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5), 6 * GAMMA_BOUND, 600, metrics=metrics)
    rows = captured["m"].rows()
    assert rows                                                    # a series was recorded
    assert [r["t_ticks"] for r in rows] == sorted(
        r["t_ticks"] for r in rows
    )
    summary = res.result.metric_values()["strong_backlog"]
    assert max(r["total_jobs"] for r in rows) == summary["peak_jobs"]
    assert max(r["total_full_input_rounds"] for r in rows) == (
        summary["peak_full_input_rounds"]
    )
    phases = (
        "waiting_far_boundary", "waiting_terminal_data",
        "in_transit", "queued", "running",
    )
    for row in rows:
        assert row["total_jobs"] == sum(
            row[f"{phase}_jobs"] for phase in phases
        )
        assert row["total_full_input_rounds"] == sum(
            row[f"{phase}_full_input_rounds"] for phase in phases
        )


def test_decode_backlog_trace_tracks_the_rising_backlog():
    """DecodeBacklog keeps a time series too: under a too-slow naive (inline) decoder the backlog
    rises, so rows() is a non-empty step series whose largest value matches result()['peak_rounds']."""
    captured = {}
    def metrics(e, wm, dm, ch, fa):
        m = DecodeBacklog(wm, dm)
        captured["m"] = m
        return [m]
    naive_decoder = SwitchingDecoder(PerRoundDecoder(F_WEAK * TAU_GEN_US),
                                     PerRoundDecoder(F_STRONG * TAU_GEN_US),
                                     gamma_switch=0.05)
    simulate(RunSpec(
        ops=[_memory_op()],
        num_units=1,
        d=D,
        rounds_policy=FixedRounds(400),
        round_us=TAU_GEN_US,
        scheme=SlidingWindowScheme(),
        decoder=naive_decoder,
        make_metrics=metrics,
        seed=1,
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
    geometry = SlidingWindowScheme().plan_operation(
        0,
        20,
        commit_round_count=code.commit_rounds(),
        buffer_round_count=code.buffer_rounds(),
    ).windows[0]
    commit_lo, commit_hi, buffer_hi = (
        geometry.commit_lo,
        geometry.commit_hi,
        geometry.buffer_hi,
    )
    assert (commit_lo, commit_hi, buffer_hi) == (1, 5, 7)
    w = Window(op_id=0, k=0, commit_lo=commit_lo, commit_hi=commit_hi,
               buffer_hi=buffer_hi, n_rounds=buffer_hi - commit_lo + 1)
    assert Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5).strong_redo_rounds(w) == 5 + 2 * 2


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
              strategy=Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5),
              router=SwitchingRouter(weak, strong),
              unit_pools={"default": 1, "strong": 1},
          ), verbose=False)
    assert res.decoder_manager.strong_needed == res.window_manager.total_windows


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


def _double_window_run(
    escalate_window, rounds=30, strong_tau=F_STRONG,
    window_interaction=None, device=None, metrics=None,
):
    """One memory op on sliding d/d windows; exactly the window with index
    escalate_window reports low confidence (deterministic, no sampling)."""
    env = {}
    weak = _DispatchRecorder(SampledConfidenceDecoder(
        PerRoundDecoder(F_WEAK * TAU_GEN_US), 0.0,
        probability_for=lambda job: 1.0 if job.window_id == escalate_window
        else 0.0), env)
    strong = _DispatchRecorder(PerRoundDecoder(strong_tau * TAU_GEN_US), env)

    def make_metrics(engine, window_manager, decoder_manager, chip, factory):
        env["engine"] = engine
        if metrics is None:
            return []
        return metrics(engine, window_manager, decoder_manager, chip, factory)

    res = simulate(RunSpec(
              ops=[_memory_op()],
              num_units=1,
              d=D,
              rounds_policy=FixedRounds(rounds),
              round_us=TAU_GEN_US,
              scheme=SlidingWindowScheme(),
              strategy=Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5,
                                 double_window=True),
              router=SwitchingRouter(weak, strong),
              window_interaction=window_interaction,
              device=device,
              unit_pools={"default": 1, "strong": 1},
              make_metrics=make_metrics,
          ), verbose=False)
    return res, weak, strong


def test_double_window_backlog_records_pending_assignment_before_admission():
    """The real metric sees one exact 15-round strong slab while it waits for
    its far boundary, before the decoder manager owns the admitted job."""
    captured = {}

    def make_metrics(engine, window_manager, decoder_manager, chip, factory):
        metric = StrongDecoderBacklog(window_manager, decoder_manager)
        captured["metric"] = metric
        return [metric]

    result, _, strong = _double_window_run(
        escalate_window=2,
        metrics=make_metrics,
    )

    waiting_rows = [
        row for row in captured["metric"].rows()
        if row["waiting_far_boundary_jobs"] > 0
    ]
    assert waiting_rows == [{
        "t_ticks": 14_750_000,
        "waiting_far_boundary_jobs": 1,
        "waiting_far_boundary_full_input_rounds": 15,
        "waiting_terminal_data_jobs": 0,
        "waiting_terminal_data_full_input_rounds": 0,
        "in_transit_jobs": 0,
        "in_transit_full_input_rounds": 0,
        "queued_jobs": 0,
        "queued_full_input_rounds": 0,
        "running_jobs": 0,
        "running_full_input_rounds": 0,
        "total_jobs": 1,
        "total_full_input_rounds": 15,
    }]
    assert strong.starts[0][1].n_rounds == 15
    assert result.result.metric_values()["strong_backlog"]["peak_jobs"] == 1


def test_double_window_uses_the_interaction_region_plan():
    class RecordingDevice(TimingOnlyDevice):
        def __init__(self):
            self.exclusions = []

        def strong_window_model_for_operation(
            self, op, window, round_count, *, belief_matching=False,
            exclude_faults_touching=None,
        ):
            if exclude_faults_touching is not None:
                exclude_lo, exclude_hi = exclude_faults_touching
                assert isinstance(exclude_lo, int)
                assert isinstance(exclude_hi, int)
            self.exclusions.append(
                (window.commit_lo, window.commit_hi, exclude_faults_touching))
            return None

    class ShorterRegion(DefaultWindowInteraction):
        def __init__(self):
            self.calls = []

        def plan_strong_region(
            self, weak_window, later_windows, operation_round_count,
        ):
            self.calls.append(weak_window.key)
            return StrongRegionPlan(
                commit_lo=7,
                commit_hi=12,
                context_lo=4,
                context_hi=15,
                restart_buffer_lo=10,
                restart_seam_fault_owner=SeamFaultOwner.STRONG_REGION,
            )

    interaction = ShorterRegion()
    device = RecordingDevice()
    result, weak, strong = _double_window_run(
        escalate_window=2,
        window_interaction=interaction,
        device=device,
    )

    assert interaction.calls == [(0, 2)]
    assert result.window_manager.absorbed_windows == {(0, 3)}
    assert [job.window_id for _, job in weak.starts] == [
        0, 1, 2, 4, 5, 6, 7, 8, 9,
    ]
    (_, strong_job), = strong.starts
    assert (
        strong_job.window.commit_lo,
        strong_job.window.commit_hi,
    ) == (7, 12)
    # priced for the rounds it is handed (context 4-15), not its commit extent
    assert strong_job.n_rounds == 12
    assert len({p.round_index for p in strong_job.payloads}) == 12
    assert device.exclusions == [
        (13, 15, (1, 12)),
        (7, 12, (1, 6)),
    ]


def test_restart_model_failure_leaves_strong_plan_state_unchanged():
    class RestartModelFailure(RuntimeError):
        pass

    class FailingDevice(TimingOnlyDevice):
        def strong_window_model_for_operation(
            self, op, window, round_count, *, belief_matching=False,
            exclude_faults_touching=None,
        ):
            raise RestartModelFailure("injected restart model failure")

    weak = SampledConfidenceDecoder(
        FixedLogicalDecoder(logical_value=0),
        0.0,
        probability_for=lambda job: 1.0 if job.window_id == 2 else 0.0,
    )
    strong = FixedLogicalDecoder(logical_value=0)
    captured = {}

    def configure(engine, runtime, _decoder_manager, _chip, _factory):

        def snapshot():
            restart = runtime.windows[(0, 5)]
            captured["runtime"] = runtime
            captured["before"] = {
                "absorbed": set(runtime.absorbed_windows),
                "escalations": dict(runtime.pending_escalations),
                "restart_buffer_lo": restart.buffer_lo,
                "restart_deps": list(restart.deps),
                "restart_refs": list(runtime.store._leases[(restart.key)]),
            }

        engine.schedule(0, snapshot, label="capture restart plan")

    with pytest.raises(RestartModelFailure, match="restart model"):
        RunSpec(
            ops=[_memory_op()],
            d=D,
            rounds_policy=FixedRounds(21),
            scheme=SlidingWindowScheme(),
            strategy=Switching(
                expected_source=SAMPLED_CONFIDENCE_SOURCE,
                confidence_threshold=0.5,
                double_window=True,
            ),
            router=SwitchingRouter(weak, strong),
            device=FailingDevice(),
            unit_pools={"default": 1, "strong": 1},
            make_metrics=lambda engine, window_manager, decoder_manager, chip, factory: (
                configure(
                    engine, window_manager, decoder_manager, chip, factory
                ) or []
            ),
        ).build(verbose=False)

    runtime = captured["runtime"]
    restart = runtime.windows[(0, 5)]
    assert {
        "absorbed": runtime.absorbed_windows,
        "escalations": runtime.pending_escalations,
        "restart_buffer_lo": restart.buffer_lo,
        "restart_deps": restart.deps,
        "restart_refs": runtime.store._leases[restart.key],
    } == captured["before"]


def test_strong_decoder_result_identity_is_checked_before_finalization():
    class WrongIdentityStrongDecoder(FixedLogicalDecoder):
        def decode(self, job):
            return DecodeResult(
                job.op_id,
                job.window_id + 1,
                logical_observables=(self.logical_value,),
            )

    weak = SampledConfidenceDecoder(
        FixedLogicalDecoder(logical_value=0),
        0.0,
        probability_for=lambda job: 1.0 if job.window_id == 2 else 0.0,
    )
    strong = WrongIdentityStrongDecoder(logical_value=1)

    with pytest.raises(RuntimeError, match="result identity"):
        simulate(RunSpec(
            ops=[_memory_op()],
            d=D,
            rounds_policy=FixedRounds(21),
            scheme=SlidingWindowScheme(),
            strategy=Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5, double_window=True),
            router=SwitchingRouter(weak, strong),
            unit_pools={"default": 1, "strong": 1},
        ), verbose=False)


def test_restart_owned_seam_requires_multi_range_device_capability():
    class SingleRangeOnlyDevice:
        def __init__(self):
            self._timing = TimingOnlyDevice()
            self.operation_circuit_scope = "none"

        def __getattr__(self, name):
            if name == "strong_window_model_for_operation_with_exclusions":
                raise AttributeError(name)
            return getattr(self._timing, name)

        def strong_window_model_for_operation(
            self, op, window, round_count, *, belief_matching=False,
            exclude_faults_touching=None,
        ):
            return None

    class RestartOwned(DefaultWindowInteraction):
        def plan_strong_region(self, *args, **kwargs):
            plan = super().plan_strong_region(*args, **kwargs)
            return StrongRegionPlan(
                commit_lo=plan.commit_lo,
                commit_hi=plan.commit_hi,
                context_lo=plan.context_lo,
                context_hi=plan.context_hi,
                restart_buffer_lo=plan.restart_buffer_lo,
                restart_seam_fault_owner=SeamFaultOwner.RESTART_WINDOW,
            )

    with pytest.raises(
        TypeError,
        match="multiple fault-exclusion ranges",
    ):
        _double_window_run(
            escalate_window=2,
            rounds=21,
            window_interaction=RestartOwned(),
            device=SingleRangeOnlyDevice(),
        )


def test_double_window_retains_every_round_added_to_the_restart_buffer():
    class EarlierRetainedRestart(DefaultWindowInteraction):
        def plan_strong_region(
            self, weak_window, later_windows, operation_round_count,
        ):
            return StrongRegionPlan(
                commit_lo=7,
                commit_hi=12,
                context_lo=7,
                context_hi=15,
                restart_buffer_lo=4,
                restart_seam_fault_owner=SeamFaultOwner.STRONG_REGION,
            )

    result, weak, _ = _double_window_run(
        escalate_window=2,
        window_interaction=EarlierRetainedRestart(),
    )

    restart = result.window_manager.windows[(0, 4)]
    restart_job = next(
        job for _, job in weak.starts if job.window_id == restart.k)
    assert restart.start_round == 4
    assert [payload.round_index for payload in restart_job.payloads] == \
        list(range(restart.start_round, restart.buffer_hi + 1))


def test_double_window_rejects_restart_reads_that_are_already_freed():
    class FreedRestartHistory(DefaultWindowInteraction):
        def plan_strong_region(
            self, weak_window, later_windows, operation_round_count,
        ):
            return StrongRegionPlan(
                commit_lo=7,
                commit_hi=12,
                context_lo=7,
                context_hi=15,
                restart_buffer_lo=1,
                restart_seam_fault_owner=SeamFaultOwner.STRONG_REGION,
            )

    with pytest.raises(RuntimeError, match="retained payload"):
        _double_window_run(
            escalate_window=2,
            window_interaction=FreedRestartHistory(),
        )


def test_double_window_rejects_commit_ownership_before_the_escalated_window():
    class BackwardCommitRegion(DefaultWindowInteraction):
        def plan_strong_region(
            self, weak_window, later_windows, operation_round_count,
        ):
            return StrongRegionPlan(
                commit_lo=4,
                commit_hi=12,
                context_lo=4,
                context_hi=15,
                restart_buffer_lo=10,
                restart_seam_fault_owner=SeamFaultOwner.STRONG_REGION,
            )

    with pytest.raises(RuntimeError, match="cannot precede"):
        _double_window_run(
            escalate_window=2,
            rounds=15,
            window_interaction=BackwardCommitRegion(),
        )


@pytest.mark.parametrize("invalid_owner", [None, "strong", 17])
def test_double_window_rejects_invalid_restart_seam_owner(invalid_owner):
    class InvalidFaultOwnership(DefaultWindowInteraction):
        def plan_strong_region(
            self, weak_window, later_windows, operation_round_count,
        ):
            return StrongRegionPlan(
                commit_lo=7,
                commit_hi=12,
                context_lo=4,
                context_hi=15,
                restart_buffer_lo=10,
                restart_seam_fault_owner=invalid_owner,
            )

    with pytest.raises(RuntimeError, match="valid restart seam fault owner"):
        _double_window_run(
            escalate_window=2,
            window_interaction=InvalidFaultOwnership(),
        )


def test_double_window_rejects_an_interaction_without_a_region_plan():
    class NoRegion(DefaultWindowInteraction):
        def plan_strong_region(
            self, weak_window, later_windows, operation_round_count,
        ):
            return None

    with pytest.raises(TypeError, match="must return StrongRegionPlan"):
        _double_window_run(
            escalate_window=2,
            window_interaction=NoRegion(),
        )


def test_double_window_slab_extends_forward_and_absorbs_the_weak_chain():
    """Fig. 12 panels 4-5: W2 (commit 7-9) escalates; the slab is rounds
    7-15 (commit + two buffers FORWARD), the windows committing 10-12 and
    13-15 are never weak-decoded, and the weak chain restarts at W5."""
    res, weak, strong = _double_window_run(escalate_window=2)
    runtime = res.window_manager
    (start_tick, job), = strong.starts
    assert job.strong_decode_for == (0, 2)
    assert (job.window.commit_lo, job.window.commit_hi) == (7, 15)
    assert job.window.commit_hi - job.window.commit_lo + 1 == 3 * D  # r_strong
    # Thm 1 bounds tau_strong against rounds of decoder INPUT, so the job is
    # priced for the context it reads (4-18), not for the slab it commits.
    assert job.n_rounds == 3 * D + 2 * D
    assert len({p.round_index for p in job.payloads}) == job.n_rounds
    assert runtime.absorbed_windows == {(0, 3), (0, 4)}
    weak_windows_decoded = [j.window_id for _, j in weak.starts]
    assert weak_windows_decoded == [0, 1, 2, 5, 6, 7, 8, 9]
    for absorbed_key in runtime.absorbed_windows:
        window = runtime.windows[absorbed_key]
        assert window.committed and window.t_done is None


def test_double_window_restart_window_is_priced_for_its_widened_read():
    """Fig. 12 panel 5: the restart window carries a LEADING buffer that
    reaches back into the slab, so it reads r_buf + r_com + r_buf rounds
    while committing only r_com. It must be priced for what it reads: it is
    the one window the protocol deliberately enlarges, and charging it as an
    ordinary window understates the weak decoder exactly there."""
    res, weak, strong = _double_window_run(escalate_window=2)
    runtime = res.window_manager
    (_, slab), = strong.starts
    restart = runtime.windows[(0, 5)]

    assert restart.buffer_lo <= slab.window.commit_hi   # reads into the slab
    assert (restart.commit_lo, restart.commit_hi) == (16, 18)   # commits r_com
    assert restart.buffer_lo == 13                      # slab_hi - r_buf + 1
    assert restart.n_rounds == restart.buffer_hi - restart.buffer_lo + 1

    restart_job = next(job for _, job in weak.starts if job.window_id == 5)
    assert restart_job.n_rounds == 3 * D                # r_buf + r_com + r_buf
    assert len({p.round_index for p in restart_job.payloads}) == 3 * D

    ordinary_job = next(job for _, job in weak.starts if job.window_id == 6)
    assert ordinary_job.n_rounds == 2 * D               # r_com + r_buf
    assert restart_job.n_rounds > ordinary_job.n_rounds


def test_double_window_restart_pricing_separates_commit_and_buffer():
    """The r_com == r_buf == d fixtures cannot tell the restart formula
    r_buf + r_com + r_buf apart from a wrong 3 * r_com. Asymmetric geometry
    (r_com=2, r_buf=3) separates them: 8 rounds versus 6."""
    code = SurfaceCodeModel(d=D, commit_rounds_override=2,
                            buffer_rounds_override=3)
    env = {}
    weak = _DispatchRecorder(SampledConfidenceDecoder(
        PerRoundDecoder(F_WEAK * TAU_GEN_US), 0.0,
        probability_for=lambda job: 1.0 if job.window_id == 2 else 0.0), env)
    strong = _DispatchRecorder(PerRoundDecoder(F_STRONG * TAU_GEN_US), env)
    res = simulate(RunSpec(
        ops=[_memory_op()], num_units=1, code=code,
        rounds_policy=FixedRounds(30), round_us=TAU_GEN_US,
        scheme=SlidingWindowScheme(),
        strategy=Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5, double_window=True),
        router=SwitchingRouter(weak, strong),
        unit_pools={"default": 1, "strong": 1},
        make_metrics=lambda e, wm, dm, ch, fa: env.update(engine=e) or [],
    ), verbose=False)
    runtime = res.window_manager
    commit = runtime._code_geometry.commit_round_count
    buffer = runtime._code_geometry.buffer_round_count
    assert (commit, buffer) == (2, 3)   # r_com != r_buf

    (_, slab), = strong.starts
    plan = runtime.window_interaction.plan_strong_region(
        WindowInfo.from_window(runtime.windows[(0, 2)]),
        [WindowInfo.from_window(runtime.windows[(0, k)])
         for k in runtime.op_windows[0] if k > 2],
        30,
    )
    restart_key = next(
        window.key
        for window in (
            runtime.windows[(0, k)]
            for k in runtime.op_windows[0]
            if k > 2
        )
        if window.commit_lo > plan.commit_hi
    )
    restart_job = next(job for _, job in weak.starts
                       if (0, job.window_id) == restart_key)

    # r_buf + r_com + r_buf = 3 + 2 + 3, NOT 3 * r_com = 6
    assert restart_job.n_rounds == 8
    assert restart_job.n_rounds != 3 * commit
    assert len({p.round_index for p in restart_job.payloads}) == 8

    # and the slab is still priced for its whole context, r_strong + 2*r_buf
    commit_extent = slab.window.commit_hi - slab.window.commit_lo + 1
    assert commit_extent == commit + 2 * buffer   # 8
    assert slab.n_rounds == commit_extent + 2 * buffer    # 14
    assert slab.n_rounds != commit_extent


def test_double_window_restart_pricing_is_not_the_geometry_constant():
    """A restart window that lands on a short commit tail is narrower than
    the geometry constant commit + 2*buffer, so pricing it from that constant
    is wrong even though every full-width fixture agrees with it. The
    invariant is the window's own extent."""
    res, weak, strong = _double_window_run(escalate_window=5, rounds=26)
    runtime = res.window_manager
    (_, slab), = strong.starts
    restart = next(runtime.windows[k] for k in sorted(runtime.windows)
                   if runtime.windows[k].commit_lo > slab.window.commit_hi
                   and runtime.windows[k].buffer_lo <= slab.window.commit_hi)
    restart_job = next(job for _, job in weak.starts
                       if job.window_id == restart.k)

    geometry_constant = (
        runtime._code_geometry.commit_round_count
        + 2 * runtime._code_geometry.buffer_round_count
    )
    assert restart.buffer_hi - restart.buffer_lo + 1 != geometry_constant
    assert restart_job.n_rounds == restart.buffer_hi - restart.buffer_lo + 1
    assert restart_job.n_rounds != geometry_constant


def test_double_window_strong_waits_for_the_restart_windows_weak_commit():
    """Fig. 12 panel 6: the slab starts only after the restart window's
    (W5's) weak decode commits the far-side boundary, plus the weak->strong
    hop. Without the gate the slab would start right after W2's outcome."""
    res, weak, strong = _double_window_run(escalate_window=2)
    cluster = res.window_manager
    w5 = cluster.windows[(0, 5)]
    (start_tick, job), = strong.starts
    assert w5.t_done is not None
    assert start_tick == w5.t_done + us(3.0)
    assert cluster.pending_escalations == {}


def test_double_window_weak_pipeline_never_stalls_on_strong_work():
    """The weak chain's schedule is independent of the strong decoder: a
    10x slower strong decoder changes no weak window's decode time."""
    fast, _, _ = _double_window_run(escalate_window=2, strong_tau=F_STRONG)
    slow, _, _ = _double_window_run(escalate_window=2,
                                    strong_tau=10 * F_STRONG)
    assert {k: w.t_done for k, w in fast.window_manager.windows.items()} \
        == {k: w.t_done for k, w in slow.window_manager.windows.items()}


def test_double_window_last_window_uses_the_terminal_boundary():
    """The final window's slab clamps at the stream end and has no restart
    window: the terminal time boundary already exists, so the slab is
    submitted at escalation without any extra wait."""
    res, weak, strong = _double_window_run(escalate_window=9)  # last of 10
    cluster = res.window_manager
    w9 = cluster.windows[(0, 9)]
    (start_tick, job), = strong.starts
    assert (job.window.commit_lo, job.window.commit_hi) == (28, 30)
    # clamped r_strong is 3 committed rounds; the decoder reads 25-30
    assert job.n_rounds == 6
    assert len({p.round_index for p in job.payloads}) == 6
    assert start_tick == w9.t_done + us(2.0)
    assert cluster.absorbed_windows == set()


def test_double_window_end_clamped_slab_absorbs_the_tail():
    """W8 (commit 25-27) escalates: the slab clamps to rounds 25-30,
    absorbs the final window, and needs no restart gate."""
    res, weak, strong = _double_window_run(escalate_window=8)
    runtime = res.window_manager
    (start_tick, job), = strong.starts
    assert (job.window.commit_lo, job.window.commit_hi) == (25, 30)
    # 6 committed rounds, read with one buffer of leading context (22-30)
    assert job.n_rounds == 9
    assert len({p.round_index for p in job.payloads}) == 9
    assert runtime.absorbed_windows == {(0, 9)}
    assert [j.window_id for _, j in weak.starts] == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    w8 = res.window_manager.windows[(0, 8)]
    assert start_tick == w8.t_done + us(2.0)


def test_double_window_exactly_one_strong_job_per_escalation():
    """One switching event creates exactly one strong job; a duplicate
    registration for the same window is an illegal transition."""
    res, weak, strong = _double_window_run(escalate_window=2)
    assert len(strong.starts) == 1
    runtime = res.window_manager
    pending = runtime._escalations.peek_far((0, 4))
    assert pending is None
    frozen = runtime.pending_escalations
    assert frozen == {}
    with pytest.raises(RuntimeError, match="duplicate strong escalation"):
        runtime.defer_strong_escalation(DecodeJob(
            op_id=0,
            window_id=2,
            n_rounds=9,
            ready_time=0,
            deadline=0,
            strong_label="strong(op0 W2)",
        ))


def _pending_escalation(key, phase):
    plan = StrongRegionPlan(
        commit_lo=7,
        commit_hi=12,
        context_lo=4,
        context_hi=15,
        restart_buffer_lo=(10 if phase is _EscalationPhase.WAITING_FAR_BOUNDARY
                           else None),
        restart_seam_fault_owner=(
            SeamFaultOwner.STRONG_REGION
            if phase is _EscalationPhase.WAITING_FAR_BOUNDARY
            else None
        ),
    )

    resolved = _ResolvedStrongRegion(
        plan=plan,
        absorbed_window_keys=((key[0], key[1] + 1),),
        restart_window_key=(
            (key[0], key[1] + 2)
            if phase is _EscalationPhase.WAITING_FAR_BOUNDARY
            else None
        ),
        restart_read_keys=(),
        strong_fault_exclusion_ranges=(),
        restart_fault_exclusion_ranges=None,
    )
    return _PendingEscalation(
        key=key,
        weak_job=DecodeJob(
            op_id=key[0],
            window_id=key[1],
            n_rounds=9,
            strong_label=f"strong({key})",
        ),
        label=f"strong({key})",
        resolved_region=resolved,
        strong_window=Window(
            op_id=key[0],
            k=key[1],
            commit_lo=7,
            commit_hi=12,
            buffer_lo=4,
            buffer_hi=15,
            n_rounds=12,
        ),
        strong_model=None,
        wsd_arrival_ticks=0,
        phase=phase,
    )


@pytest.mark.parametrize("invalid_bound", [True, 7.0, _IntSubclass(7)])
def test_strong_region_plan_rejects_non_exact_integer_bounds(invalid_bound):
    with pytest.raises(TypeError, match="exact built-in ints"):
        StrongRegionPlan(
            commit_lo=invalid_bound,
            commit_hi=12,
            context_lo=4,
            context_hi=15,
            restart_buffer_lo=10,
            restart_seam_fault_owner=SeamFaultOwner.STRONG_REGION,
        )


def test_escalation_registry_far_transfer_is_exactly_once():
    registry = _EscalationRegistry()
    pending = _pending_escalation(
        (0, 2), _EscalationPhase.WAITING_FAR_BOUNDARY)
    far_boundary_key = (0, 4)

    registry.register_far(pending, far_boundary_key)

    assert registry.peek_key(pending.key) is pending
    assert registry.peek_far(far_boundary_key) is pending
    assert dict(registry.snapshot_phases()) == {
        pending.key: _EscalationPhase.WAITING_FAR_BOUNDARY,
    }
    assert registry.snapshot_work() == (
        ((0, 2), "waiting_far_boundary", 12),
    )
    with pytest.raises(RuntimeError, match="wrong-phase take"):
        registry.take_terminal(0, pending)
    assert registry.peek_far(far_boundary_key) is pending

    assert registry.take_far(far_boundary_key, pending) is pending
    assert registry.peek_key(pending.key) is None
    assert registry.peek_far(far_boundary_key) is None
    assert dict(registry.snapshot_phases()) == {}


def test_escalation_registry_rejects_collisions_before_mutation():
    registry = _EscalationRegistry()
    first = _pending_escalation(
        (0, 2), _EscalationPhase.WAITING_FAR_BOUNDARY)
    collision = _pending_escalation(
        (0, 3), _EscalationPhase.WAITING_FAR_BOUNDARY)
    far_boundary_key = (0, 4)
    registry.register_far(first, far_boundary_key)
    before = dict(registry.snapshot_phases())

    with pytest.raises(RuntimeError, match="readiness index collision"):
        registry.register_far(collision, far_boundary_key)

    assert dict(registry.snapshot_phases()) == before
    assert registry.peek_key(collision.key) is None
    assert registry.peek_far(far_boundary_key) is first


def test_escalation_registry_terminal_transfer_uses_only_terminal_index():
    registry = _EscalationRegistry()
    pending = _pending_escalation(
        (7, 8), _EscalationPhase.WAITING_TERMINAL_DATA)

    with pytest.raises(RuntimeError, match="expected WAITING_FAR_BOUNDARY"):
        registry.register_far(pending, (7, 9))
    assert dict(registry.snapshot_phases()) == {}

    registry.register_terminal(pending, 7)
    assert registry.snapshot_work() == (
        ((7, 8), "waiting_terminal_data", 12),
    )
    assert registry.peek_terminal(7) is pending
    assert registry.peek_far((7, 9)) is None
    assert registry.take_terminal(7, pending) is pending
    assert registry.peek_terminal(7) is None
    assert registry.peek_key(pending.key) is None


def test_escalation_work_snapshot_orders_mixed_stable_identities():
    registry = _EscalationRegistry()
    integer_key = _pending_escalation(
        (0, 2), _EscalationPhase.WAITING_FAR_BOUNDARY
    )
    string_key = _pending_escalation(
        ("0", 2), _EscalationPhase.WAITING_FAR_BOUNDARY
    )
    registry.register_far(string_key, ("far", 1))
    registry.register_far(integer_key, ("far", 0))

    assert registry.snapshot_work() == (
        ((0, 2), "waiting_far_boundary", 12),
        (("0", 2), "waiting_far_boundary", 12),
    )


@pytest.mark.parametrize("invalid_key", [True, (0, True), object()])
def test_escalation_registry_rejects_unstable_keys_before_mutation(invalid_key):
    registry = _EscalationRegistry()
    pending = _pending_escalation(
        (0, 2), _EscalationPhase.WAITING_FAR_BOUNDARY)

    with pytest.raises(TypeError, match="readiness key"):
        registry.register_far(pending, invalid_key)

    assert dict(registry.snapshot_phases()) == {}


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
              strategy=Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5, double_window=True),
              router=SwitchingRouter(weak, strong),
              unit_pools={"default": 1, "strong": 1},
          ), verbose=False)
    assert res.window_manager.op_results[0] == (1,)
    assert res.decoder_manager.strong_needed == 1
    runtime = res.window_manager
    contribution = runtime.logical_contributions[(0, 2)]
    assert (
        contribution.ownership_kind,
        contribution.commit_lo,
        contribution.commit_hi,
        contribution.logical_observables,
    ) == ("strong_slab", 7, 15, (0,))
    assert (0, 3) not in runtime.logical_contributions
    assert (0, 4) not in runtime.logical_contributions


def test_double_window_installs_slab_owner_before_every_submission_path(
        monkeypatch):
    snapshots = []
    original_far_submit = WindowManager._submit_far_strong
    original_terminal_submit = WindowManager._submit_terminal_strong

    def record(runtime, pending):
        key = pending.key
        contribution = runtime.logical_contributions[key]
        snapshots.append({
            "key": key,
            "commit_lo": contribution.commit_lo,
            "commit_hi": contribution.commit_hi,
            "kind": contribution.ownership_kind,
            "prediction": contribution.logical_observables,
            "weak_committed": runtime.windows[key].committed,
            "absorbed_have_contributions": (
                runtime.absorbed_windows
                & set(runtime.logical_contributions)
            ),
        })
    def record_then_submit_far(runtime, far_boundary_key, pending):
        record(runtime, pending)
        return original_far_submit(runtime, far_boundary_key, pending)

    def record_then_submit_terminal(runtime, operation_id, pending):
        record(runtime, pending)
        return original_terminal_submit(runtime, operation_id, pending)

    monkeypatch.setattr(
        WindowManager,
        "_submit_far_strong",
        record_then_submit_far,
    )
    monkeypatch.setattr(
        WindowManager,
        "_submit_terminal_strong",
        record_then_submit_terminal,
    )

    _double_window_run(escalate_window=2)
    _double_window_run(escalate_window=2, rounds=14)
    _double_window_run(escalate_window=9)

    assert snapshots == [
        {
            "key": (0, 2),
            "commit_lo": 7,
            "commit_hi": 15,
            "kind": "strong_slab",
            "prediction": None,
            "weak_committed": True,
            "absorbed_have_contributions": set(),
        },
        {
            "key": (0, 2),
            "commit_lo": 7,
            "commit_hi": 14,
            "kind": "strong_slab",
            "prediction": None,
            "weak_committed": True,
            "absorbed_have_contributions": set(),
        },
        {
            "key": (0, 9),
            "commit_lo": 28,
            "commit_hi": 30,
            "kind": "strong_slab",
            "prediction": None,
            "weak_committed": False,
            "absorbed_have_contributions": set(),
        },
    ]


def test_double_window_slab_payloads_cover_slab_plus_two_sided_context():
    """Slab payloads are assembled at submission: the committed slab rounds
    7-15 plus one buffer of read-only raw context on EACH face (4-6 and
    16-18), the same role a buffer plays for every weak window (B-side
    formalism; no decoded defects are folded in). The timing charge stays
    the paper's r_strong (asserted above via n_rounds)."""
    res, weak, strong = _double_window_run(escalate_window=2)
    (start_tick, job), = strong.starts
    assert (job.window.start_round, job.window.buffer_hi) == (4, 18)
    assert job.window.boundary_in == {}
    assert [payload.round_index for payload in job.payloads] \
        == list(range(4, 19))


def test_double_window_rejects_contradictory_switch_flags():
    with pytest.raises(ValueError, match="double_window"):
        Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5, double_window=True,
                  run_both_at_once=True)
    with pytest.raises(ValueError, match="double_window"):
        Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5, double_window=True,
                  bulk_strong=True)


def test_double_window_rejects_unsupported_runspec_shapes():
    """Held would deadlock the far-boundary wait; parallel two-layer windows,
    runtime streams, frontends, and cross-op window chains need skip
    semantics the runtime does not model yet."""
    from decsim.policies import Held
    from decsim.schemes import ParallelWindowScheme
    strategy = Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5, double_window=True)
    base = dict(num_units=1, d=D, rounds_policy=FixedRounds(30),
                round_us=TAU_GEN_US, strategy=strategy)
    with pytest.raises(ValueError, match="Held"):
        RunSpec(ops=[_memory_op()], scheme=SlidingWindowScheme(),
                boundary_policy=Held(), **base).build()
    with pytest.raises(ValueError, match="SlidingWindowScheme"):
        RunSpec(ops=[_memory_op()], scheme=ParallelWindowScheme(),
                **base).build()
    with pytest.raises(ValueError, match="dynamic_streams"):
        RunSpec(ops=[_memory_op()],
                dynamic_streams=[Operation(7, "stream", (1,))],
                **base).build()
    chained = [Operation(0, "a", (0,), has_successor=True),
               Operation(1, "b", (1,), predecessors=(0,))]
    with pytest.raises(ValueError, match="single-patch"):
        RunSpec(ops=chained, **base).build()


def test_double_window_terminal_slab_waits_for_its_final_rounds():
    """A clamped terminal slab has no restart window, but Fig. 12's blocks
    are STORED syndrome data: with 14 rounds, W2's slab is rounds 7-14 while
    W2's own weak decode finishes before rounds 13-14 are even generated.
    The slab must wait for them and then carry every slab round."""
    res, weak, strong = _double_window_run(escalate_window=2, rounds=14)
    runtime = res.window_manager
    (start_tick, job), = strong.starts
    assert (job.window.commit_lo, job.window.commit_hi) == (7, 14)
    assert [payload.round_index for payload in job.payloads] \
        == list(range(4, 15))
    w2 = res.window_manager.windows[(0, 2)]
    assert start_tick > w2.t_done + us(0.5)   # NOT submitted at escalation
    # round 14 reaches the cluster at 14.0 (generation) + 2.15 (t_qc + t_cd);
    # the slab then crosses the controller-to-strong data path
    assert start_tick == us(14.0 + 2.15 + 2.0)
    assert runtime.absorbed_windows == {(0, 3), (0, 4)}
    assert runtime.pending_escalations == {}
