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
from fractions import Fraction

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from conftest import fixed_latency_link_config
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
from decsim.detector_error_model import NO_FAULT_MODEL_REQUIRED
from decsim.devices import TimingOnlyDevice
from decsim.message import (CsdInput, DecodeJob, DecodeResult, DecoderRequestKey,
                            DecoderTier, EndpointRole, Operation, PendingStrong,
                            PotentialStrong, Replay, RephaseGuard,
                            ResolvedCodeGeometry,
                            SeamFaultOwner, StrongRegionPlan, Window,
                            WindowInfo)
from decsim.metrics import DecodeBacklog, StrongDecoderBacklog
from decsim.schemes import SlidingWindowScheme
from decsim.switching import Switching
from decsim.run_spec import RunSpec, simulate
from decsim.planner import FixedRounds
from decsim.payload_store import PayloadStore
from decsim.protocols import Directive, OutcomeDirective
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

    fault_model_requirement = NO_FAULT_MODEL_REQUIRED

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


def test_switch_probability_per_round_rejects_invalid_scaled_probability():
    with pytest.raises(ValueError, match="gamma_switch"):
        switch_probability_per_round(-0.1, D)
    assert switch_probability_per_round(Fraction(1, 2), D)(
        DecodeJob(0, 0, D)
    ) == 0.5
    for invalid_distance in (True, 1.0, Fraction(1, 1), _IntSubclass(1)):
        with pytest.raises(TypeError, match="d"):
            switch_probability_per_round(0.1, invalid_distance)
    with pytest.raises(ValueError, match="d"):
        switch_probability_per_round(0.1, 0)

    probability_for = switch_probability_per_round(1.0, D)
    with pytest.raises(ValueError, match="switch probability"):
        probability_for(DecodeJob(0, 0, D + 1))


def test_switch_probability_per_round_rejects_invalid_effective_counts():
    probability_for = switch_probability_per_round(0.1, 3)
    for commit_rounds in (0, -1, True, 1.0, Fraction(1, 1), "1"):
        with pytest.raises((TypeError, ValueError), match="commit_rounds"):
            probability_for(DecodeJob(0, 0, commit_rounds))

    for commit_hi in (0, 1.5):
        window = Window(
            op_id=0,
            k=0,
            commit_lo=1,
            commit_hi=commit_hi,
            buffer_hi=3,
            n_rounds=3,
        )
        with pytest.raises((TypeError, ValueError), match="commit_rounds"):
            probability_for(DecodeJob(0, 0, 3, window=window))


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
        self.fault_model_requirement = inner.fault_model_requirement

    def latency(self, job):
        self.starts.append((self.env["engine"].now, job))
        return self.inner.latency(job)

    def decode(self, job):
        return self.inner.decode(job)


def _double_window_run(
    escalate_window, rounds=30, strong_tau=F_STRONG,
    window_interaction=None, device=None, metrics=None, code=None,
    record_switching_windows=False, links=None,
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
              d=None if code is not None else D,
              code=code,
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
              record_switching_windows=record_switching_windows,
              links=links,
          ), verbose=False)
    return res, weak, strong


def test_real_deferred_escalation_transfers_potential_pending_csd_in_order(
    monkeypatch,
):
    events = []
    register_owner = PayloadStore.register_owner
    release_owner = PayloadStore.release_owner

    def record_register(store, role, owner, identities):
        if role is EndpointRole.SB1:
            events.append(("acquire", owner))
        return register_owner(store, role, owner, identities)

    def record_release(store, role, owner):
        if role is EndpointRole.SB1:
            events.append(("release", owner))
        return release_owner(store, role, owner)

    monkeypatch.setattr(PayloadStore, "register_owner", record_register)
    monkeypatch.setattr(PayloadStore, "release_owner", record_release)
    _double_window_run(escalate_window=2, rounds=21)

    pending = next(owner for action, owner in events
                   if action == "acquire" and type(owner) is PendingStrong)
    request_key = pending.request_key
    relevant = [(action, owner) for action, owner in events if owner in (
        PotentialStrong((0, 2)), pending, CsdInput(request_key))]
    assert relevant == [
        ("acquire", PotentialStrong((0, 2))),
        ("acquire", pending),
        ("release", PotentialStrong((0, 2))),
        ("acquire", CsdInput(request_key)),
        ("release", pending),
        ("release", CsdInput(request_key)),
    ]


def test_confident_and_absorbed_windows_release_potential_after_replacement(
    monkeypatch,
):
    events = []
    registrations = {}
    register_owner = PayloadStore.register_owner
    release_owner = PayloadStore.release_owner

    def record_register(store, role, owner, identities):
        if role is EndpointRole.SB1:
            events.append(("acquire", owner))
            registrations[owner] = tuple(identities)
        return register_owner(store, role, owner, identities)

    def record_release(store, role, owner):
        if role is EndpointRole.SB1:
            events.append(("release", owner))
        return release_owner(store, role, owner)

    monkeypatch.setattr(PayloadStore, "register_owner", record_register)
    monkeypatch.setattr(PayloadStore, "release_owner", record_release)
    _double_window_run(escalate_window=2, rounds=21)

    pending_index = next(index for index, event in enumerate(events)
                         if event[0] == "acquire"
                         and type(event[1]) is PendingStrong)
    for window_key in ((0, 0), (0, 1), (0, 3), (0, 4), (0, 5), (0, 6)):
        release_index = events.index(("release", PotentialStrong(window_key)))
        if window_key in ((0, 3), (0, 4)):
            assert pending_index < release_index
    pending = next(owner for owner in registrations
                   if type(owner) is PendingStrong)
    needed = set(registrations[PotentialStrong((0, 3))])
    needed.update(registrations[PotentialStrong((0, 4))])
    replacement = set(registrations[pending])
    replacement.update(registrations[PotentialStrong((0, 5))])
    assert needed <= replacement


def test_absorption_rejects_partial_acquired_replacement(monkeypatch):
    owner_packet_identities = PayloadStore.owner_packet_identities

    def omit_restart_tail(store, role, owner):
        keys = owner_packet_identities(store, role, owner)
        if owner == PotentialStrong((0, 5)):
            return keys[:-3]
        return keys

    monkeypatch.setattr(
        PayloadStore, "owner_packet_identities", omit_restart_tail)
    with pytest.raises(RuntimeError, match="replacement does not cover"):
        _double_window_run(escalate_window=2, rounds=21)


def _nonaligned_code():
    return SurfaceCodeModel(
        d=D,
        commit_rounds_override=7,
        buffer_rounds_override=3,
    )


def test_strong_selection_is_not_published_before_do_delivery():
    selections_before_delivery = []

    def inspect_commit(engine, window_manager, decoder_manager, chip, factory):
        original = window_manager._commit_strong_decode_done
        def commit(completion):
            key = (completion.request_key.operation_id,
                   completion.request_key.window_id)
            selections_before_delivery.append(
                window_manager._selected_request_keys.get(key))
            original(completion)
        window_manager._commit_strong_decode_done = commit
        return []

    result, _, _ = _double_window_run(
        1, metrics=inspect_commit, record_switching_windows=True)
    assert selections_before_delivery == [None]
    assert result.window_manager._selected_request_keys[(0, 1)].tier is DecoderTier.STRONG


def test_deferred_directive_key_must_match_registered_pending_key():
    exposed = {}

    def expose_state(engine, window_manager, decoder_manager, chip, factory):
        exposed.update(window_manager=window_manager,
                       decoder_manager=decoder_manager)
        return []

    class WrongDeferredKey(Switching):
        def on_decode_outcome(self, outcome, services):
            directive = super().on_decode_outcome(outcome, services)
            if directive.directive is not Directive.AWAIT_STRONG:
                return directive
            key = directive.strong_request_key
            return OutcomeDirective(
                Directive.AWAIT_STRONG,
                strong_request_key=DecoderRequestKey(
                    key.operation_id, key.window_id, key.tier,
                    key.run_sequence + 1))

    with pytest.raises(RuntimeError, match="deferred.*key"):
        _switch_run(WrongDeferredKey(
            expected_source=SAMPLED_CONFIDENCE_SOURCE,
            confidence_threshold=0.5, double_window=True), 1.0, 30,
            metrics=expose_state)

    runtime = exposed["window_manager"]
    manager = exposed["decoder_manager"]
    selected_paths = {row["path"] for row in runtime.links.traffic_json_value()[
        "transfers"] if row["path"] in ("wsd", "csd")}
    assert selected_paths == set()
    assert runtime.pending_escalations
    assert runtime._escalations.peek_key((0, 0)).wsd_arrival_ticks is None
    assert manager.strong_needed == 0
    assert not manager._running_strong_decodes
    assert not manager._windows_waiting_for_strong_selection
    assert not manager._windows_waiting_for_strong_result
    assert manager._terminal_request_records is None
    assert manager._terminal_service_records is None

    class AbandonDeferred(Switching):
        def on_decode_outcome(self, outcome, services):
            directive = super().on_decode_outcome(outcome, services)
            return (OutcomeDirective(Directive.FINALIZE)
                    if directive.directive is Directive.AWAIT_STRONG
                    else directive)

    with pytest.raises(RuntimeError, match="WSD reservation|pending strong escalations"):
        _switch_run(AbandonDeferred(
            expected_source=SAMPLED_CONFIDENCE_SOURCE,
            confidence_threshold=0.5, double_window=True), 1.0, 30)


def test_double_window_rephases_residual_inside_crossing_final_commit():
    result, weak, strong = _double_window_run(
        escalate_window=1,
        rounds=21,
        code=_nonaligned_code(),
    )

    restart = result.window_manager.windows[(0, 2)]
    assert (
        restart.buffer_lo,
        restart.commit_lo,
        restart.commit_hi,
        restart.buffer_hi,
        restart.n_rounds,
    ) == (18, 21, 21, 21, 4)
    assert [job.window_id for _, job in weak.starts] == [0, 1, 2]
    (strong_start_tick, strong_job), = strong.starts
    restart_job = next(job for _, job in weak.starts if job.window_id == 2)
    assert strong_job.strong_decode_for == (0, 1)
    assert restart.t_done is not None
    assert strong_start_tick > restart.t_done
    assert restart_job.n_rounds == 4


def test_double_window_rephases_and_clamps_complete_nonaligned_suffix():
    recorded_model_windows = []

    class RecordingDevice(TimingOnlyDevice):
        def window_models_for_operation(
            self, op, windows, round_count, *,
            fault_model_requirement, fault_exclusion_ranges,
        ):
            recorded_model_windows.append((
                tuple(
                    (
                        window.buffer_lo,
                        window.commit_lo,
                        window.commit_hi,
                        window.buffer_hi,
                        window.n_rounds,
                    )
                    for window in windows
                ),
                fault_exclusion_ranges,
            ))
            return []

    result, weak, _ = _double_window_run(
        escalate_window=1,
        rounds=50,
        code=_nonaligned_code(),
        device=RecordingDevice(),
        record_switching_windows=True,
    )

    runtime = result.window_manager
    suffix_keys = [(0, index) for index in range(2, 7)]
    assert [
        (
            runtime.windows[key].buffer_lo,
            runtime.windows[key].commit_lo,
            runtime.windows[key].commit_hi,
            runtime.windows[key].buffer_hi,
            runtime.windows[key].n_rounds,
        )
        for key in suffix_keys
    ] == [
        (18, 21, 27, 30, 13),
        (28, 28, 34, 37, 10),
        (35, 35, 41, 44, 10),
        (42, 42, 48, 50, 9),
        (49, 49, 50, 50, 2),
    ]
    assert (0, 7) not in runtime.windows
    suffix_model_call = next(
        windows
        for windows, exclusions in recorded_model_windows
        if exclusions == ((1, 20),)
    )
    assert suffix_model_call[-1] == (49, 49, 50, 50, 2)
    suffix_jobs = [
        job for _, job in weak.starts
        if job.window_id in {2, 3, 4, 5, 6}
    ]
    assert max(
        payload.round_index
        for job in suffix_jobs
        for payload in job.payloads
    ) <= 50
    rows = result.result.metric_values()["window_switching_records"]["windows"]
    assert len(rows) == 7
    assert [row["destination_key"] for row in rows] == [
        [0, index] for index in range(7)]
    assert all(row["window_disposition"] != "absorbed" for row in rows)


def test_double_window_rephase_preserves_conflicting_registry_owner():
    class SeedConflict(DefaultWindowInteraction):
        runtime = None
        conflict = None

        def plan_strong_region(
            self, weak_window, later_windows, operation_round_count,
        ):
            plan = super().plan_strong_region(
                weak_window, later_windows, operation_round_count)
            self.conflict = _PendingEscalation(
                key=weak_window.key,
                weak_job=None,
                label="existing",
                resolved_region=None,
                strong_window=Window(
                    op_id=0,
                    k=1,
                    commit_lo=8,
                    commit_hi=20,
                    buffer_lo=5,
                    buffer_hi=23,
                    n_rounds=19,
                ),
                strong_model=None,
                wsd_arrival_ticks=0,
                phase=_EscalationPhase.WAITING_FAR_BOUNDARY,
                strong_request_key=DecoderRequestKey(
                    weak_window.op_id, weak_window.k, DecoderTier.STRONG, 0),
                strong_request_created_ticks=0,
            )
            self.runtime._escalations.register_far(
                self.conflict, (77, 0))
            return plan

    interaction = SeedConflict()
    captured = {}

    def connect_interaction(engine, window_manager, decoder_manager, chip, factory):
        interaction.runtime = window_manager
        captured["runtime"] = window_manager
        return []

    with pytest.raises(RuntimeError, match="duplicate strong escalation"):
        _double_window_run(
            escalate_window=1,
            rounds=50,
            code=_nonaligned_code(),
            window_interaction=interaction,
            metrics=connect_interaction,
        )

    runtime = captured["runtime"]
    assert runtime._escalations.peek_key((0, 1)) is interaction.conflict
    assert runtime._escalations.peek_far((77, 0)) is interaction.conflict
    assert runtime._escalations._by_key == {(0, 1): interaction.conflict}
    assert runtime._escalations._by_far_boundary == {(77, 0): (0, 1)}
    assert runtime._escalations._by_terminal_operation == {}
    assert runtime.op_windows[0] == list(range(8))
    assert runtime.window_count[0] == 8
    assert runtime.total_windows == 8
    assert runtime._committed_per_op == {0: 1}
    assert runtime.absorbed_windows == set()
    assert (0, 7) in runtime.windows



def test_double_window_rephase_rejects_affected_far_readiness_owner():
    def manager_state(runtime):
        return {
            "windows": {
                key: (
                    window.buffer_lo,
                    window.commit_lo,
                    window.commit_hi,
                    window.buffer_hi,
                    window.n_rounds,
                    tuple(window.deps),
                    window.deps_remaining,
                    tuple(window.dependents),
                    window.queued,
                    window.committed,
                )
                for key, window in runtime.windows.items()
            },
            "op_windows": {
                op_id: tuple(indices)
                for op_id, indices in runtime.op_windows.items()
            },
            "window_count": dict(runtime.window_count),
            "total_windows": runtime.total_windows,
            "committed_per_op": dict(runtime._committed_per_op),
            "committed_windows": set(runtime.committed_windows),
            "absorbed_windows": set(runtime.absorbed_windows),
            "logical_contributions": dict(runtime.logical_contributions),
            "registry_by_key": dict(runtime._escalations._by_key),
            "registry_by_far": dict(runtime._escalations._by_far_boundary),
            "registry_by_terminal": dict(
                runtime._escalations._by_terminal_operation
            ),
            "owners": tuple(sorted(
                ((role, owner, record.packet_identities)
                 for (role, owner), record
                 in runtime.store._future_owners.items()),
                key=repr,
            )),
        }

    class SeedAffectedFarOwner(DefaultWindowInteraction):
        runtime = None
        conflict = None
        before = None

        def plan_strong_region(
            self, weak_window, later_windows, operation_round_count,
        ):
            plan = super().plan_strong_region(
                weak_window, later_windows, operation_round_count)
            self.conflict = _PendingEscalation(
                key=("other-escalation", 0),
                weak_job=None,
                label="existing",
                resolved_region=None,
                strong_window=Window(
                    op_id=0,
                    k=99,
                    commit_lo=1,
                    commit_hi=1,
                    buffer_lo=1,
                    buffer_hi=1,
                    n_rounds=1,
                ),
                strong_model=None,
                wsd_arrival_ticks=0,
                phase=_EscalationPhase.WAITING_FAR_BOUNDARY,
                strong_request_key=DecoderRequestKey(
                    "other-escalation", 0, DecoderTier.STRONG, 0),
                strong_request_created_ticks=0,
            )
            self.runtime._escalations.register_far(
                self.conflict, (0, 4))
            self.before = manager_state(self.runtime)
            return plan

    interaction = SeedAffectedFarOwner()

    def connect_interaction(
        engine, window_manager, decoder_manager, chip, factory,
    ):
        interaction.runtime = window_manager
        return []

    with pytest.raises(
        RuntimeError,
        match=r"cannot rephase readiness key [(]0, 4[)]",
    ):
        _double_window_run(
            escalate_window=1,
            rounds=50,
            code=_nonaligned_code(),
            window_interaction=interaction,
            metrics=connect_interaction,
        )

    assert manager_state(interaction.runtime) == interaction.before
    assert interaction.runtime._escalations.peek_key(
        interaction.conflict.key
    ) is interaction.conflict
    assert interaction.runtime._escalations.peek_far(
        (0, 4)
    ) is interaction.conflict
    affected = interaction.runtime.windows[(0, 4)]
    assert (
        affected.buffer_lo,
        affected.commit_lo,
        affected.commit_hi,
        affected.buffer_hi,
    ) == (29, 29, 35, 38)


@pytest.mark.parametrize("published_state", ["boundary", "external_edge"])
def test_double_window_rephase_rejects_historical_or_external_suffix_state(
    published_state,
):
    captured = {}

    def inject_state(engine, window_manager, decoder_manager, chip, factory):
        captured["runtime"] = window_manager
        original_defer = window_manager.defer_strong_escalation

        def defer_with_published_state(weak_job):
            if published_state == "boundary":
                window_manager._held_boundary[(0, 2)] = (0, {})
            else:
                window_manager.windows[(0, 7)].dependents.append((99, 0))
            original_defer(weak_job)

        window_manager.defer_strong_escalation = defer_with_published_state
        return []

    expected = "published state|external or non-serial edge|external edge"
    with pytest.raises(RuntimeError, match=expected):
        _double_window_run(
            escalate_window=1,
            rounds=50,
            code=_nonaligned_code(),
            metrics=inject_state,
        )

    runtime = captured["runtime"]
    assert runtime.op_windows[0] == list(range(8))
    assert runtime.pending_escalations == {}
    assert (0, 7) in runtime.windows


def test_double_window_rephase_rejects_a_queued_suffix_window():
    captured = {}

    def inject_queued_window(
        engine, window_manager, decoder_manager, chip, factory,
    ):
        captured["runtime"] = window_manager
        original_defer = window_manager.defer_strong_escalation

        def defer_after_queue_started(weak_job):
            window_manager.windows[(0, 2)].queued = True
            original_defer(weak_job)

        window_manager.defer_strong_escalation = defer_after_queue_started
        return []

    with pytest.raises(RuntimeError, match="decode lifecycle already started"):
        _double_window_run(
            escalate_window=1,
            rounds=50,
            code=_nonaligned_code(),
            metrics=inject_queued_window,
        )

    runtime = captured["runtime"]
    assert runtime.op_windows[0] == list(range(8))
    assert runtime.pending_escalations == {}


def test_double_window_rephase_rolls_back_manager_and_owner_state():
    captured = {}

    def inject_failure(engine, window_manager, decoder_manager, chip, factory):
        captured["runtime"] = window_manager
        register_guard = window_manager.store.register_rephase_guard

        def capture_guard(guard, sb0_ids, sb1_ids):
            captured["guard"] = guard
            return register_guard(guard, sb0_ids, sb1_ids)

        window_manager.store.register_rephase_guard = capture_guard
        original_replace = window_manager.store.replace_owner_membership
        target_tokens = [
            token
            for window_index in range(2, 8)
            for token in (
                (EndpointRole.SB0, (0, window_index)),
                (EndpointRole.SB1, PotentialStrong((0, window_index))),
            )
        ]
        captured["old_owners"] = None
        replacement_count = 0

        def fail_second_replacement(role, owner, packet_identities):
            nonlocal replacement_count
            token = (role, owner)
            if token in target_tokens:
                if captured["old_owners"] is None:
                    captured["old_owners"] = {
                        target: window_manager.store.owner_packet_identities(*target)
                        for target in target_tokens
                        if window_manager.store.has_owner(*target)
                    }
                replacement_count += 1
                if replacement_count == 2:
                    raise RuntimeError("injected suffix owner failure")
            original_replace(role, owner, packet_identities)

        window_manager.store.replace_owner_membership = fail_second_replacement
        return []

    with pytest.raises(RuntimeError, match="injected suffix owner failure"):
        _double_window_run(
            escalate_window=1,
            rounds=50,
            code=_nonaligned_code(),
            metrics=inject_failure,
        )

    runtime = captured["runtime"]
    assert runtime.op_windows[0] == list(range(8))
    assert runtime.window_count[0] == 8
    assert runtime.total_windows == 8
    assert [
        (
            runtime.windows[(0, index)].buffer_lo,
            runtime.windows[(0, index)].commit_lo,
            runtime.windows[(0, index)].commit_hi,
            runtime.windows[(0, index)].buffer_hi,
        )
        for index in range(2, 8)
    ] == [
        (15, 15, 21, 24),
        (22, 22, 28, 31),
        (29, 29, 35, 38),
        (36, 36, 42, 45),
        (43, 43, 49, 52),
        (50, 50, 50, 53),
    ]
    assert runtime.absorbed_windows == set()
    assert runtime.pending_escalations == {}
    assert set(runtime.logical_contributions) == {(0, 0)}
    assert runtime.windows[(0, 1)].dependents == [(0, 2)]
    for token, old_packet_identities in captured["old_owners"].items():
        assert runtime.store.owner_packet_identities(*token) == \
            old_packet_identities
    assert type(captured["guard"]) is RephaseGuard
    assert not any(runtime.store.has_owner(role, captured["guard"])
                   for role in EndpointRole)
    pending = PendingStrong(captured["guard"].request_key)
    assert not runtime.store.has_owner(EndpointRole.SB1, pending)


def test_double_window_rephase_rollback_restores_absent_commit_count():
    captured = {}

    def inject_failure(engine, window_manager, decoder_manager, chip, factory):
        captured["runtime"] = window_manager
        original_replace = window_manager.store.replace_owner_membership
        targets = {
            (EndpointRole.SB0, (0, 1)),
            (EndpointRole.SB1, PotentialStrong((0, 1))),
        }
        replacement_count = 0

        def fail_second_replacement(role, owner, packet_identities):
            nonlocal replacement_count
            if (role, owner) in targets:
                replacement_count += 1
                if replacement_count == 2:
                    raise RuntimeError("injected first-window suffix failure")
            original_replace(role, owner, packet_identities)

        window_manager.store.replace_owner_membership = fail_second_replacement
        return []

    with pytest.raises(RuntimeError, match="first-window suffix failure"):
        _double_window_run(
            escalate_window=0,
            rounds=50,
            code=_nonaligned_code(),
            metrics=inject_failure,
        )

    runtime = captured["runtime"]
    assert runtime._committed_per_op == {}
    assert runtime.pending_escalations == {}
    assert runtime.op_windows[0] == list(range(8))


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
            self, op, window, round_count, *, fault_model_requirement,
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
            self, op, window, round_count, *, fault_model_requirement,
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
                "restart_refs": runtime.store.owner_packet_identities(
                    EndpointRole.SB0, restart.key),
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
        "restart_refs": runtime.store.owner_packet_identities(
            EndpointRole.SB0, restart.key),
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
            self, op, window, round_count, *, fault_model_requirement,
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



def test_aligned_double_window_rejects_float_commit_bounds():
    class FloatCommitBounds(DefaultWindowInteraction):
        def plan_strong_region(
            self, weak_window, later_windows, operation_round_count,
        ):
            plan = super().plan_strong_region(
                weak_window, later_windows, operation_round_count)
            object.__setattr__(plan, "commit_lo", float(plan.commit_lo))
            object.__setattr__(plan, "commit_hi", float(plan.commit_hi))
            return plan

    with pytest.raises(
        TypeError,
        match="logical contribution bounds must be exact ints",
    ):
        _double_window_run(
            escalate_window=2,
            rounds=30,
            window_interaction=FloatCommitBounds(),
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


def test_double_window_last_window_uses_the_terminal_boundary(monkeypatch):
    """The final window's slab clamps at the stream end and has no restart
    window: the terminal time boundary already exists, so the slab is
    submitted at escalation without any extra wait."""
    events = []
    original_prepare = WindowManager.prepare_strong_selection
    original_submit = WindowManager._submit_terminal_strong

    def record_prepare(runtime, *args, **kwargs):
        events.append(("prepare", runtime.engine.now))
        return original_prepare(runtime, *args, **kwargs)

    def record_submit(runtime, *args):
        events.append(("submit", runtime.engine.now))
        return original_submit(runtime, *args)

    monkeypatch.setattr(WindowManager, "prepare_strong_selection", record_prepare)
    monkeypatch.setattr(WindowManager, "_submit_terminal_strong", record_submit)
    res, weak, strong = _double_window_run(
        escalate_window=9, strong_tau=0, record_switching_windows=True,
        links=fixed_latency_link_config())  # last of 10
    cluster = res.window_manager
    w9 = cluster.windows[(0, 9)]
    (start_tick, job), = strong.starts
    assert (job.window.commit_lo, job.window.commit_hi) == (28, 30)
    # clamped r_strong is 3 committed rounds; the decoder reads 25-30
    assert job.n_rounds == 6
    assert len({p.round_index for p in job.payloads}) == 6
    assert start_tick == w9.t_done
    assert events == [("prepare", w9.t_done), ("submit", w9.t_done)]
    requests = res.result.metric_values()["window_switching_records"]["requests"]
    assert any(row["terminal_processing_outcome"] == "strong_forwarded_for_delivery" for row in requests)
    transfers = [row for row in res.result.link_traffic["transfers"]
                 if row["path"] in ("wsd", "csd")]
    assert [row["path"] for row in transfers] == ["wsd", "csd"]
    assert (transfers[0]["attribution"]["relation"]["request_key"] ==
            transfers[1]["attribution"]["relation"]["request_key"])
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
        strong_request_key=DecoderRequestKey(
            key[0], key[1], DecoderTier.STRONG, 0),
        strong_request_created_ticks=0,
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
    chained = [
        Operation(0, "a", (0,), has_successor=True),
        Operation(
            1,
            "b",
            (1,),
            predecessors=(0,),
            decoder_boundary_predecessors=(0,),
        ),
    ]
    with pytest.raises(ValueError, match="single-patch"):
        RunSpec(ops=chained, **base).build()


def test_double_window_allows_workload_only_operation_ordering():
    strategy = Switching(
        expected_source=SAMPLED_CONFIDENCE_SOURCE,
        confidence_threshold=0.5,
        double_window=True,
    )
    strategy.validate_operations([
        Operation(0, "prepare", (0,)),
        Operation(1, "measure", (1,), predecessors=(0,)),
    ])


def test_double_window_rejects_decoder_boundary_operation_chains():
    strategy = Switching(
        expected_source=SAMPLED_CONFIDENCE_SOURCE,
        confidence_threshold=0.5,
        double_window=True,
    )
    operations = [
        Operation(0, "first stream", (0,), has_successor=True),
        Operation(
            1,
            "second stream",
            (0,),
            decoder_boundary_predecessors=(0,),
        ),
    ]
    with pytest.raises(ValueError, match="single-patch"):
        strategy.validate_operations(operations)


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
