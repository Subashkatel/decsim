"""Strategies + pool completion pipeline (Contract 2b/2c orderings)."""
import pytest

from decsim.engine import Engine
from decsim.message import (
    DecodeJob,
    DecodeOutcome,
    DecodeResult,
    ResolvedCodeGeometry,
    Window,
)
from decsim.decoder_manager import StrategyServicesImpl, DecoderManager
from decsim.protocols import Directive, Submission
from decsim.switching import Baseline, Switching

WS = 500_000


class _FifoScheduler:
    def insert(self, queue, job): queue.append(job)
    def pop(self, queue, now_ticks): return queue.pop(0)


class _Decoder:
    def __init__(self, latency, soft=None, logical=0, name="w"):
        self._latency, self.soft, self.logical, self.name = latency, soft, logical, name
    def latency(self, job): return self._latency
    def decode(self, job):
        return DecodeResult(job.op_id, job.window_id,
                            logical_observables=(self.logical,),
                            soft_output=self.soft)


class _Router:
    def __init__(self, weak, strong): self.weak, self.strong = weak, strong
    def route(self, job): return self.strong if job.hint == "strong" else self.weak


class _RuntimeStub:
    """Provides make_strong_decode_job + records commits (window-manager side)."""
    def __init__(self):
        self.commits, self.strong_commits = [], []
    def make_strong_decode_job(self, weak_job, n_rounds, label):
        w = weak_job.window
        return DecodeJob(op_id=weak_job.op_id, window_id=weak_job.window_id,
                         n_rounds=n_rounds, label=label, hint="strong",
                         attempt=1, window=w,
                         strong_decode_for=(weak_job.op_id, weak_job.window_id))
    def on_decode_done(self, job, result):
        self.commits.append((job.op_id, job.window_id, job.awaiting_strong_result))
    def on_strong_decode_done(self, key, result):
        self.strong_commits.append(key)


def _window():
    return Window(op_id=0, k=0, commit_lo=1, commit_hi=3, buffer_hi=6, n_rounds=6)


def _weak_job(window):
    j = DecodeJob(op_id=0, window_id=0, n_rounds=6, window=window, label="op0 W0")
    j.strong_label = "strong(op0 W0)"
    return j


def _pool(strategy, weak, strong, pools=None):
    eng = Engine(verbose=False)
    rt = _RuntimeStub()
    pool = DecoderManager(eng, router=_Router(weak, strong), scheduler=_FifoScheduler(),
                    unit_pools=pools or {"default": 1, "strong": 1},
                    ws_delay_ticks=WS)
    pool.strategy = strategy
    pool.services = StrategyServicesImpl(eng, rt, pool)
    pool.on_window_decoded = rt.on_decode_done
    pool.on_strong_window_decoded = rt.on_strong_decode_done
    return eng, rt, pool


def test_baseline_finalizes_and_never_escalates():
    weak = _Decoder(10, logical=1)
    eng, rt, pool = _pool(Baseline(), weak, _Decoder(100))
    w = _window()
    for sub in Baseline().on_window_ready(w, _weak_job(w), None):
        pool.enqueue(sub.job, sub.delay_ticks)
    eng.run()
    assert rt.commits == [(0, 0, False)]
    assert pool.strong_needed == 0


def test_serial_escalation_after_ws_and_awaiting_set_before_commit():
    weak = _Decoder(10, soft=0.1)                 # low confidence -> escalate
    strong = _Decoder(1_000, logical=1, name="s")
    strat = Switching(confidence_threshold=0.5)
    eng, rt, pool = _pool(strat, weak, strong)
    w = _window()
    job = _weak_job(w)
    for sub in strat.on_window_ready(w, job, pool.services):
        pool.enqueue(sub.job, sub.delay_ticks)
    eng.run()
    # weak committed with awaiting=True (BEFORE commit callback, 2b.5)
    assert rt.commits == [(0, 0, True)]
    assert pool.strong_needed == 1
    # strong result applied after weak commit (redo enqueued at weak_done + ws)
    assert rt.strong_commits == [(0, 0)]
    # redo covered commit + 2*buffer = 3 + 2*3 = 9 rounds
    assert strat.strong_redo_rounds(w) == 9


def test_confident_weak_result_cancels_parallel_strong():
    weak = _Decoder(10, soft=0.9)                 # confident -> keep weak
    strong = _Decoder(1_000_000, name="s")        # would take forever
    strat = Switching(confidence_threshold=0.5, run_both_at_once=True)
    eng, rt, pool = _pool(strat, weak, strong)
    w = _window()
    job = _weak_job(w)
    subs = strat.on_window_ready(w, job, pool.services)
    assert len(subs) == 2 and subs[1].job.hint == "strong"
    assert subs[1].delay_ticks == 0               # parallel: no ws delay
    for sub in subs:
        pool.enqueue(sub.job, sub.delay_ticks)
    eng.run()
    assert rt.commits == [(0, 0, False)]
    assert pool.strong_cancelled == 1             # in-flight strong cancelled
    assert rt.strong_commits == []                # never applied
    # PARITY QUIRK (dm:189-190): cancel marks the job and frees the unit but
    # leaves the dead completion event queued — engine.now still advances to
    # it. The goldens (fully_done) depend on this; do not engine.cancel here.
    assert eng.now == 1_000_000
    assert pool.pool_free["strong"] == 1          # unit freed at cancel time


def test_early_strong_held_then_applied_when_weak_commits():
    weak = _Decoder(10_000, soft=0.1)             # slow weak, low confidence
    strong = _Decoder(10, logical=1, name="s")    # strong finishes first
    strat = Switching(confidence_threshold=0.5, run_both_at_once=True)
    eng, rt, pool = _pool(strat, weak, strong)
    w = _window()
    job = _weak_job(w)
    for sub in strat.on_window_ready(w, job, pool.services):
        pool.enqueue(sub.job, sub.delay_ticks)
    eng.run()
    # strong completed early -> held; applied right after the weak commit
    assert rt.commits == [(0, 0, True)]
    assert rt.strong_commits == [(0, 0)]


def test_same_decoder_route_raises_at_build_time():
    shared = _Decoder(10)
    strat = Switching(confidence_threshold=0.5, run_both_at_once=True)
    eng, rt, pool = _pool(strat, shared, shared)
    w = _window()
    with pytest.raises(RuntimeError, match="distinct decoder"):
        strat.on_window_ready(w, _weak_job(w), pool.services)


def test_switching_config_validation():
    with pytest.raises(ValueError):
        Switching(0.5, weak_keepup_ratio=1.5)
    with pytest.raises(ValueError):
        Switching(0.5, run_both_at_once=True, bulk_strong=True)
    with pytest.raises(ValueError):
        Switching(0.5, weak_keepup_ratio=0.9).validate_code_geometry(
            ResolvedCodeGeometry(
                code_name="test",
                distance=3,
                commit_round_count=3,
                buffer_round_count=3,
                minimum_leading_buffer_round_count=3,
                minimum_trailing_buffer_round_count=3,
                one_patch_spatial_node_count=9,
                buffer_floor_override_active=True,
            )
        )


def test_pool_validation():
    eng = Engine(verbose=False)
    with pytest.raises(ValueError, match="default"):
        DecoderManager(eng, router=None, scheduler=_FifoScheduler(),
                 unit_pools={"strong": 1})
    with pytest.raises(ValueError, match="at least 1"):
        DecoderManager(eng, router=None, scheduler=_FifoScheduler(),
                 unit_pools={"default": 0})
