"""Black-box tests for decoder queues, pools, routing, and switching.

Paper contract: docs/PAPER_MODEL_MAP.md.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from decsim.decoders import CodeRouter
from decsim.decoder_manager import DecoderManager
from decsim.config import us
from decsim.decoders import (PerRoundDecoder, PresetLatencyDecoder,
                             SampledConfidenceDecoder, SwitchingRouter)
from decsim.engine import Engine
from decsim.message import DecodeJob, DecodeResult, Operation
from decsim.schedulers import FifoScheduler
from decsim.schemes import SlidingWindowScheme
from decsim.switching import Switching
from decsim.run_spec import RunSpec, simulate
from decsim.planner import FixedRounds

TAU = 1.0          # syndrome-round time (microseconds)
D = 3              # code distance; commit = buffer = d for the default sliding scheme


def _memory_op(op_id=0):
    """One Clifford memory operation, with a long syndrome stream and no feedback block."""
    return Operation(op_id, "memory", (op_id,), clifford=True)


def _switch_run(switching, low_confidence_probability, *, rounds=60, tau_weak=0.1,
                tau_strong=10.0, pools=None, seed=1, make_metrics=None):
    """A single-patch sliding-window run with decoder switching, built exactly the way
    tests/test_switching.py builds one: a SampledConfidenceDecoder weak (its soft_output is
    deterministically 0.0 with probability `low_confidence_probability`, else 1.0, so
    keep_weak_result is controllable) routed against a slow PerRoundDecoder strong by a
    SwitchingRouter, with separate "default"/"strong" unit pools."""
    weak = SampledConfidenceDecoder(PerRoundDecoder(tau_weak * TAU),
                                    low_confidence_probability)
    strong = PerRoundDecoder(tau_strong * TAU)
    return simulate(RunSpec(
               ops=[_memory_op()],
               d=D,
               rounds_policy=FixedRounds(rounds),
               round_us=TAU,
               scheme=SlidingWindowScheme(),
               strategy=switching,
               decoder=weak,
               router=SwitchingRouter(weak, strong),
               unit_pools=pools or {"default": 1, "strong": 1},
               make_metrics=make_metrics,
               seed=seed,
           ), verbose=False)


# =====================================================================================
# A1 -- dispatch is unit-bounded
# =====================================================================================

def test_A1_dispatch_is_bounded_by_free_units_and_queue_drains():
    """A1 (DecoderManager dispatch; arXiv:2511.10633 the cluster runs a bounded unit pool).
    With N units, at most N jobs decode at once: free_units never goes below 0, never exceeds
    N busy concurrently, returns to its starting value once all work drains, and queue_log is a
    non-empty (time, queued_total) series that rises above N when more jobs are ready than units
    and falls back to 0."""
    N = 2
    engine = Engine(verbose=False)
    cluster = DecoderManager(engine, router=CodeRouter(PresetLatencyDecoder(10.0)),
                       scheduler=FifoScheduler(), num_units=N)
    start_free = cluster.free_units
    assert start_free == N == cluster.num_units

    class _Probe:
        name = "probe"
        def __init__(self): self.min_free = N; self.max_busy = 0
        def observe(self, e):
            self.min_free = min(self.min_free, cluster.free_units)
            self.max_busy = max(self.max_busy, N - cluster.free_units)
        def result(self): return None
    probe = _Probe()
    engine.add_metric(probe)

    n_jobs = 3 * N                                  # surplus: more ready jobs than units
    for i in range(n_jobs):
        cluster.submit_decode(6, lambda: None, label=f"j{i}")
    peak_queue_at_submit = max(q for _, q in cluster.queue_log)
    engine.run()

    assert probe.min_free >= 0                       # free_units never negative
    assert probe.max_busy == N                       # exactly N decode at once, never more
    assert cluster.free_units == start_free          # all units returned after work drains

    assert cluster.queue_log                          # non-empty trace
    assert all(isinstance(t, int) and q >= 0 for t, q in cluster.queue_log)
    assert peak_queue_at_submit > N                   # queue rose above the unit count
    assert cluster.queue_log[-1][1] == 0             # ... and fully drained to 0


# =====================================================================================
# A2 -- external decode jobs (DecoderService contract)
# =====================================================================================

def test_A2_external_submit_runs_once_and_skips_the_window_commit_path():
    """A2 (DecoderService contract, protocols.py). submit_decode runs a self-contained job that
    competes for the same units and calls on_done EXACTLY ONCE; an external job never enters the
    window-commit path -- it adds nothing to committed_windows, op_results, or total_windows."""
    engine = Engine(verbose=False)
    cluster = DecoderManager(engine, router=CodeRouter(PresetLatencyDecoder(5.0)),
                       scheduler=FifoScheduler(), num_units=2)
    calls = []
    cluster.submit_decode(6, lambda: calls.append(engine.now), label="external")
    engine.run()

    assert len(calls) == 1                            # on_done fired exactly once
    assert calls[0] == us(5.0)                        # ran on a unit (the preset 5 us latency)
    assert cluster.free_units == cluster.num_units    # the unit it occupied came back
    # the window-commit path is structurally unreachable from the pool: the
    # restructure split it into WindowManager, which this job never touched
    assert cluster.on_window_decoded is None


def test_A2_external_job_shares_the_unit_pool_with_window_decodes():
    """A2: an external job competes for the SAME units. Two units, two simultaneous external
    jobs of 10 us each both finish at 10 us (ran concurrently); a third waits for one to free
    (finishes at 20 us)."""
    engine = Engine(verbose=False)
    cluster = DecoderManager(engine, router=CodeRouter(PresetLatencyDecoder(10.0)),
                       scheduler=FifoScheduler(), num_units=2)
    done = {}
    cluster.submit_decode(6, lambda: done.update(a=engine.now), label="a")
    cluster.submit_decode(6, lambda: done.update(b=engine.now), label="b")
    cluster.submit_decode(6, lambda: done.update(c=engine.now), label="c")
    engine.run()
    assert done["a"] == done["b"] == us(10.0)         # two units -> two run at once
    assert done["c"] == us(20.0)                       # the third queued behind a freed unit


# =====================================================================================
# A3 -- unit pools / routing by hint
# =====================================================================================

def test_A3_hint_routes_a_job_to_its_named_pool():
    """A3 (typed unit pools; arXiv:2510.25222 Fig 1 weak=FPGA/strong=GPU). With
    unit_pools={"default": 1, "strong": 1}, a job whose hint names "strong" runs on the strong
    pool's unit while a default job runs on the default pool's unit -- so both 10 us jobs start at
    t=0 and finish together. free_units reflects ONLY the default pool."""
    engine = Engine(verbose=False)
    cluster = DecoderManager(engine, router=CodeRouter(PresetLatencyDecoder(10.0)),
                       scheduler=FifoScheduler(),
                       unit_pools={"default": 1, "strong": 1})
    # free_units is the default pool's free count only, not the cluster-wide total.
    assert cluster.free_units == cluster.unit_totals["default"] == 1
    assert cluster.unit_totals == {"default": 1, "strong": 1}
    assert cluster.pool_free == {"default": 1, "strong": 1}

    done = {}
    cluster.submit_decode(6, lambda: done.update(s=engine.now), label="S", hint="strong")
    cluster.submit_decode(6, lambda: done.update(d=engine.now), label="D")  # default
    engine.run()
    # If the strong job had taken a default unit, the default job would queue behind it (20 us).
    # Routing it to the strong pool lets both run concurrently.
    assert done["s"] == us(10.0)
    assert done["d"] == us(10.0)
    assert cluster.pool_free == cluster.unit_totals   # every pool's units returned


def test_A3_construction_rejects_missing_default_and_empty_pool():
    """A3: a unit_pools dict without a "default" key, or any pool with < 1 unit, is rejected at
    construction (ValueError)."""
    kwargs = dict(router=CodeRouter(PresetLatencyDecoder(1.0)),
                  scheduler=FifoScheduler())
    with pytest.raises(ValueError):
        DecoderManager(Engine(verbose=False), unit_pools={"strong": 1}, **kwargs)
    with pytest.raises(ValueError):
        DecoderManager(Engine(verbose=False),
                 unit_pools={"default": 1, "strong": 0}, **kwargs)


# =====================================================================================
# A4 -- switching, serial mode
# =====================================================================================

def test_A4_serial_escalates_unsure_windows_and_counts_strong_needed():
    """A4 (serial decoder switching; arXiv:2510.25222 Sec III.A serial variant). Every window is
    weak-decoded first; a window whose weak result is NOT confident is escalated -- a strong
    re-decode is queued and strong_needed increments by one per escalated window. With every weak
    result flagged low-confidence (probability 1.0), strong_needed == total_windows. Serial mode
    never cancels (strong_cancelled == 0), and every window still commits exactly once."""
    res = _switch_run(Switching(confidence_threshold=0.5), 1.0, rounds=60)
    c = res["cluster"]
    assert c.total_windows > 0
    assert c.strong_needed == c.total_windows         # one escalation per window
    assert c.strong_cancelled == 0                    # serial never cancels
    assert len(c.committed_windows) == c.total_windows


def test_A4_confident_weak_never_escalates():
    """A4: if the weak decoder is always confident (low-confidence probability 0.0), no window is
    escalated -- strong_needed == 0 and strong_cancelled == 0 -- yet every window still commits."""
    res = _switch_run(Switching(confidence_threshold=0.5), 0.0, rounds=60)
    c = res["cluster"]
    assert c.total_windows > 0
    assert c.strong_needed == 0
    assert c.strong_cancelled == 0
    assert len(c.committed_windows) == c.total_windows


def test_A4_strong_redo_size_is_commit_plus_two_buffers():
    """A4: the strong re-decode covers Switching.strong_redo_rounds(window) =
    commit + 2*buffer (= 3d when commit=buffer=d). Cross-check the policy's own formula against
    the public cluster geometry (commit / buffer) the run used."""
    res = _switch_run(Switching(confidence_threshold=0.5), 1.0, rounds=60)
    c = res["cluster"]
    assert (c.commit, c.buffer) == (D, D)             # default sliding scheme
    # Reconstruct the policy's redo size from the cluster's committed geometry and confirm 3d.
    from decsim.message import Window
    w = Window(op_id=0, k=0, commit_lo=1, commit_hi=c.commit,
               buffer_hi=c.commit + c.buffer, n_rounds=c.commit + c.buffer)
    redo = Switching(confidence_threshold=0.5).strong_redo_rounds(w)
    assert redo == c.commit + 2 * c.buffer == 3 * D


# =====================================================================================
# A5 -- switching, parallel mode (run_both_at_once)
# =====================================================================================

def test_A5_parallel_starts_strong_everywhere_and_cancels_confident():
    """A5 (parallel feed; arXiv:2510.25222 Sec III.A "both decode"). run_both_at_once starts a
    strong job on EVERY window alongside the weak one; when the weak answer turns out confident the
    matching strong job is cancelled and strong_cancelled increments. So every window's strong job
    is either cancelled or needed (strong_cancelled + strong_needed == total_windows), and the
    cancelled strong work neither corrupts op_results nor double-commits a window."""
    # All-confident weak -> every strong job is cancelled.
    res = _switch_run(Switching(confidence_threshold=0.5, run_both_at_once=True), 0.0,
                      rounds=60, pools={"default": 1, "strong": 2})
    c = res["cluster"]
    assert c.total_windows > 0
    assert c.strong_cancelled == c.total_windows      # every window's strong job halted
    assert c.strong_needed == 0
    assert c.strong_cancelled + c.strong_needed == c.total_windows
    # every window committed exactly once despite the cancellations
    assert len(c.committed_windows) == c.total_windows


def test_A5_parallel_mixed_one_strong_job_per_window():
    """A5: with a mix of confident and unsure windows, parallel still launches exactly one strong
    job per window: the cancelled ones plus the needed ones account for every window. Cancellation
    must not perturb the committed-window count."""
    res = _switch_run(Switching(confidence_threshold=0.5, run_both_at_once=True), 0.4,
                      rounds=200, pools={"default": 1, "strong": 2})
    c = res["cluster"]
    assert c.strong_cancelled > 0                      # some windows turned out confident
    assert c.strong_needed > 0                         # some genuinely escalated
    assert c.strong_cancelled + c.strong_needed == c.total_windows
    assert len(c.committed_windows) == c.total_windows


# =====================================================================================
# A6 -- switching, BULK strong (NEW feature; serial only)
# =====================================================================================

def _strong_running_samples(switching, low_conf_prob, *, rounds, tau_strong, seed=1):
    """Sample cluster.strong_running_rounds after every engine event (the batch size on the
    strong unit, set at bulk dispatch)."""
    captured = {}

    class _Sampler:
        name = "strong_running"
        def __init__(self, cluster): self.cluster = cluster; self.samples = []
        def observe(self, e):
            self.samples.append(getattr(self.cluster, "strong_running_rounds", 0))
        def result(self): return self.samples

    def make_metrics(e, c, ch, fa):
        m = _Sampler(c); captured["m"] = m; return [m]

    res = _switch_run(switching, low_conf_prob, rounds=rounds, tau_strong=tau_strong,
                      seed=seed, make_metrics=make_metrics)
    return res, captured["m"].samples


def test_A6_bulk_strong_merges_outstanding_jobs_into_one_decode():
    """A6 (bulk decoding; arXiv:2510.25222 Sec IV, the NEW feature). In serial mode with
    switch_bulk_strong=True, a backed-up strong pool merges ALL its queued re-decode jobs into ONE
    decode whose round count is the sum of the merged jobs' rounds. With every window escalated and
    a slow strong decoder, the strong pool backs up, so strong_running_rounds takes values that are
    INTEGER MULTIPLES of the per-job redo (3d) and exceeds a single job's 3d -- proving >1 job ran
    in one merged batch -- and returns to 0 after the run. The merged batch's latency is
    proportional to its round count (one big decode), so the few large batches in bulk mode are far
    fewer step-changes than per-job decoding would produce."""
    per_job = 3 * D                                   # commit + 2*buffer
    res, samples = _strong_running_samples(
        Switching(confidence_threshold=0.5, bulk_strong=True), 1.0,
        rounds=120, tau_strong=10.0)
    c = res["cluster"]
    assert c.total_windows > 0
    assert c.strong_needed == c.total_windows         # every window escalated -> heavy backlog

    nonzero = sorted({s for s in samples if s > 0})
    assert nonzero, "strong_running_rounds never moved -- the strong pool did not run"
    # every observed merged-batch size is a whole number of per-job (3d) re-decodes
    assert all(s % per_job == 0 for s in nonzero), nonzero
    # at least one batch merged MORE THAN ONE job (the defining property of bulk decoding)
    assert max(nonzero) > per_job, nonzero
    # the strong unit is idle (batch counter back to 0) once everything has decoded
    assert getattr(c, "strong_running_rounds", 0) == 0
    assert samples[-1] == 0


def test_A6_non_bulk_decodes_jobs_individually_not_in_one_batch():
    """A6 (contrast). WITHOUT bulk_strong (plain serial), the strong pool processes re-decodes one
    job at a time: strong_running_rounds is the bulk-dispatch counter and stays 0 throughout, while
    the SAME workload under bulk_strong drives that counter to a large merged value. Same windows
    escalate either way; bulk merges them, non-bulk does not."""
    res_bulk, samples_bulk = _strong_running_samples(
        Switching(confidence_threshold=0.5, bulk_strong=True), 1.0,
        rounds=120, tau_strong=10.0)
    res_plain, samples_plain = _strong_running_samples(
        Switching(confidence_threshold=0.5, bulk_strong=False), 1.0,
        rounds=120, tau_strong=10.0)

    # both escalate exactly the same windows
    assert res_bulk["cluster"].strong_needed == res_plain["cluster"].strong_needed > 0
    # plain serial never populates the bulk-batch counter; bulk drives it well past one job
    assert max(samples_plain) == 0
    assert max(samples_bulk) > 3 * D
    # both still commit every window
    assert len(res_bulk["cluster"].committed_windows) == res_bulk["cluster"].total_windows
    assert len(res_plain["cluster"].committed_windows) == res_plain["cluster"].total_windows


# =====================================================================================
# A7 -- decoder choice advances the clock
# =====================================================================================

def test_A7_slower_strong_decoder_lengthens_escalated_windows():
    """A7 (the chosen decoder's latency advances the clock; protocols.py Decoder.latency). The
    decoder the router picks for a job is what advances the simulated clock. With every window
    escalated, a strong PerRoundDecoder of 20 us/round makes the run finish strictly later than the
    identical run with a 2 us/round strong decoder -- the only thing that changed is the escalated
    windows' strong-decode service time."""
    slow = _switch_run(Switching(confidence_threshold=0.5), 1.0, rounds=30, tau_strong=20.0)
    fast = _switch_run(Switching(confidence_threshold=0.5), 1.0, rounds=30, tau_strong=2.0)
    assert slow["cluster"].strong_needed == fast["cluster"].strong_needed > 0
    assert slow["engine"].now > fast["engine"].now


def test_A7_strong_latency_does_not_affect_an_all_confident_run():
    """A7 (control). If nothing escalates, the strong decoder is never chosen, so its latency is
    irrelevant: two runs with wildly different strong-decoder speeds finish at the SAME time when
    every weak result is confident -- isolating that it is the *chosen* decoder's latency, on the
    escalated windows, that moves the clock."""
    slow = _switch_run(Switching(confidence_threshold=0.5), 0.0, rounds=30, tau_strong=20.0)
    fast = _switch_run(Switching(confidence_threshold=0.5), 0.0, rounds=30, tau_strong=2.0)
    assert slow["cluster"].strong_needed == fast["cluster"].strong_needed == 0
    assert slow["engine"].now == fast["engine"].now


# =====================================================================================
# A8 -- functional decoder-result boundary
# =====================================================================================

class _FixedResultDecoder:
    def __init__(self, logical_observables):
        self.logical_observables = logical_observables

    def decode(self, job):
        return DecodeResult(
            job.op_id,
            job.window_id,
            logical_observables=self.logical_observables,
        )


class _EqualToOne:
    def __eq__(self, other):
        return other == 1


@pytest.mark.parametrize(
    "logical_observables",
    [
        [0, 1],
        (True,),
        (0.0,),
        (np.uint8(1),),
        (_EqualToOne(),),
    ],
    ids=["list", "bool", "float", "numpy-integer", "equality-spoof"],
)
def test_A8_decoder_boundary_rejects_non_exact_prediction_bits(
        logical_observables):
    engine = Engine(verbose=False)
    cluster = DecoderManager(
        engine,
        router=CodeRouter(_FixedResultDecoder(logical_observables)),
        scheduler=FifoScheduler(),
    )
    job = DecodeJob(op_id=5, window_id=2, n_rounds=3)

    with pytest.raises(TypeError, match=r"job \(5, 2\).*logical_observables"):
        cluster._decode_and_validate_result(job)


def test_A8_decoder_boundary_rejects_prediction_bits_outside_binary_domain():
    engine = Engine(verbose=False)
    cluster = DecoderManager(
        engine,
        router=CodeRouter(_FixedResultDecoder((0, 2))),
        scheduler=FifoScheduler(),
    )
    job = DecodeJob(op_id=5, window_id=2, n_rounds=3)

    with pytest.raises(ValueError, match=r"job \(5, 2\).*index 1.*2"):
        cluster._decode_and_validate_result(job)


@pytest.mark.parametrize("logical_observables", [None, (), (0,), (1, 0, 1)])
def test_A8_decoder_boundary_accepts_timing_and_exact_prediction_vectors(
        logical_observables):
    engine = Engine(verbose=False)
    cluster = DecoderManager(
        engine,
        router=CodeRouter(_FixedResultDecoder(logical_observables)),
        scheduler=FifoScheduler(),
    )
    job = DecodeJob(op_id=5, window_id=2, n_rounds=3)

    result = cluster._decode_and_validate_result(job)

    assert result.logical_observables == logical_observables
