"""Gate 6 B1: decoder-cluster queue-conservation invariants.

Event-level invariants of DecoderManager under bursty load (Gate-6 plan
item B1, docs/validation/2026-07-03-gate6-plan.md):

  I1 conservation: submitted == completed + queued + running at every
     completion event (running = units_total - units_free);
  I2 work conservation: after every dispatch pass, a pool with a
     nonempty queue has zero free units;
  I3 clean drain: at engine exhaustion all queues are empty, all units
     free, every job completed exactly once;

Seeded synthetic workload; no latency claims (PerRoundDecoder synthetic
service times only).
"""
import random
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from decsim.config import us
from decsim.engine import Engine
from decsim.decoders import CodeRouter, PerRoundDecoder
from decsim.decoder_manager import DecoderManager
from decsim.schedulers import FifoScheduler


def build(units: int, scheduler):
    eng = Engine(verbose=False)
    manager = DecoderManager(
        eng, router=CodeRouter(default=PerRoundDecoder(tau_us=1.0)),
        scheduler=scheduler, unit_pools={"default": units})
    return eng, manager


def test_conservation_and_drain_under_burst():
    rng = random.Random(20260703)
    eng, manager = build(units=3, scheduler=FifoScheduler())
    total_units = manager.unit_totals["default"]
    n_jobs = 300
    state = {"submitted": 0, "completed": 0}
    violations = []

    def check():
        running = total_units - manager.pool_free["default"]
        lhs = state["submitted"]
        rhs = state["completed"] + manager.queued_total() + running
        if lhs != rhs:
            violations.append((eng.now, lhs, rhs))
        if manager.pool_free["default"] < 0 or manager.queued_total() < 0:
            violations.append((eng.now, "negative counter"))

    def submit_one(i):
        def on_done():
            state["completed"] += 1
            check()
        state["submitted"] += 1
        manager.submit_decode(rng.randint(1, 12), on_done,
                              label=f"job{i}")
        check()

    t = 0
    for i in range(n_jobs):
        t += rng.randint(0, us(2))          # bursty arrivals (0 = same tick)
        eng.schedule(t, lambda i=i: submit_one(i))
    eng.run()

    assert not violations, violations[:5]
    assert state["completed"] == n_jobs
    assert manager.queued_total() == 0
    assert manager.pool_free["default"] == total_units


def test_work_conservation_no_idle_unit_with_nonempty_queue():
    eng, manager = build(units=2, scheduler=FifoScheduler())
    rng = random.Random(7)
    violations = []
    original = manager.try_dispatch

    def wrapped():
        original()
        if manager.queued_total() > 0 and manager.pool_free["default"] > 0:
            violations.append((eng.now, manager.queued_total(),
                               manager.pool_free["default"]))
    manager.try_dispatch = wrapped

    done = []
    for i in range(120):
        eng.schedule(rng.randint(0, us(30)),
                     lambda i=i: manager.submit_decode(
                         rng.randint(1, 8), lambda: done.append(i),
                         label=f"j{i}"))
    eng.run()
    assert not violations, violations[:5]
    assert len(done) == 120


def test_one_dispatch_pass_fills_every_free_unit():
    eng, manager = build(units=3, scheduler=FifoScheduler())
    manager.pool_free["default"] = 0
    completed = []
    for index in range(3):
        manager.submit_decode(
            1, lambda index=index: completed.append(index), label=f"held{index}"
        )
    assert len(manager.ready) == 3

    manager.pool_free["default"] = 3
    manager.try_dispatch()
    assert manager.ready == []
    assert manager.pool_free["default"] == 0
    eng.run()
    assert completed == [0, 1, 2]
    assert manager.pool_free["default"] == 3


def test_fifo_completes_in_arrival_order():
    eng, manager = build(units=1, scheduler=FifoScheduler())
    order = []
    manager.submit_decode(50, lambda: order.append("head"), label="head")
    for k in range(10):
        manager.submit_decode(1, lambda k=k: order.append(k),
                              label=f"fifo{k}")
    eng.run()
    assert order == ["head"] + list(range(10))


# ---------------------------------------------------------------------------
# V9-caveat closure (Codex Gate-6 audit): multi-pool, delayed-enqueue, and
# strong-cancel paths were outside the original invariant net.
# ---------------------------------------------------------------------------

from decsim.message import DecodeJob, DecoderRequestKey, DecoderTier


def test_multi_pool_conservation_and_work_conservation():
    eng = Engine(verbose=False)
    manager = DecoderManager(
        eng, router=CodeRouter(default=PerRoundDecoder(tau_us=1.0)),
        scheduler=FifoScheduler(), unit_pools={"default": 2, "strong": 1})
    rng = random.Random(11)
    totals = dict(manager.unit_totals)
    state = {"submitted": 0, "completed": 0}
    violations = []

    def check():
        running = sum(totals[p] - manager.pool_free[p] for p in totals)
        if state["submitted"] != state["completed"] + manager.queued_total() + running:
            violations.append(("conservation", eng.now))

    # work conservation is checked AFTER each dispatch pass settles
    # (inside on_done the freed unit legitimately precedes the manager's
    # own try_dispatch -- a legal transient, not a violation)
    original_dispatch = manager.try_dispatch

    def wrapped_dispatch():
        original_dispatch()
        for pool in totals:
            if len(manager.queue_for(pool)) > 0 and manager.pool_free[pool] > 0:
                violations.append(("idle-with-queue", pool, eng.now))
    manager.try_dispatch = wrapped_dispatch

    def submit(i):
        state["submitted"] += 1
        hint = "strong" if i % 3 == 0 else None
        manager.submit_decode(rng.randint(1, 8),
                              lambda: (state.__setitem__("completed",
                                                         state["completed"] + 1),
                                       check()),
                              label=f"mp{i}", hint=hint)
        check()

    for i in range(150):
        eng.schedule(rng.randint(0, us(40)), lambda i=i: submit(i))
    eng.run()
    assert not violations, violations[:5]
    assert state["completed"] == 150
    assert manager.pool_free == totals
    assert manager.queued_total() == 0


def test_delayed_enqueue_conservation():
    """enqueue(job, delay) holds the job in the fixed-delay decoder input link:
    neither queued nor running until the delay elapses."""
    eng = Engine(verbose=False)
    manager = DecoderManager(
        eng, router=CodeRouter(default=PerRoundDecoder(tau_us=1.0)),
        scheduler=FifoScheduler(), unit_pools={"default": 1})
    done = []
    job = DecodeJob(op_id=-1, window_id=0, n_rounds=3,
                    ready_time=0,
                    on_done=lambda: done.append(eng.now), label="handoff")
    manager.enqueue(job, delay_ticks=us(5))
    assert manager.queued_total() == 0            # in the link, not queued
    assert manager.pool_free["default"] == 1      # not running either
    eng.run()
    assert done and done[0] == us(5) + us(3)      # link delay + decode
    assert manager.queued_total() == 0
    assert manager.pool_free["default"] == 1


def test_cancel_queued_strong_job():
    eng = Engine(verbose=False)
    manager = DecoderManager(
        eng, router=CodeRouter(default=PerRoundDecoder(tau_us=1.0)),
        scheduler=FifoScheduler(), unit_pools={"default": 1})
    blocker_done = []
    manager.submit_decode(50, lambda: blocker_done.append(eng.now),
                          label="blocker")
    key = (7, 3)
    strong = DecodeJob(op_id=7, window_id=3, n_rounds=4,
                       strong_decode_for=key, label="strong-redo",
                       request_key=DecoderRequestKey(*key, DecoderTier.STRONG, 0))
    manager.enqueue(strong)
    assert manager.queued_total() == 1
    manager.cancel_strong(key)
    assert manager.queued_total() == 0            # removed from the queue
    assert manager.strong_cancelled == 1
    eng.run()
    assert len(blocker_done) == 1
    assert manager.pool_free["default"] == 1


def test_cancel_running_strong_job_frees_unit_exactly_once():
    eng = Engine(verbose=False)
    manager = DecoderManager(
        eng, router=CodeRouter(default=PerRoundDecoder(tau_us=1.0)),
        scheduler=FifoScheduler(), unit_pools={"default": 1})
    key = (9, 1)
    strong = DecodeJob(op_id=9, window_id=1, n_rounds=30,
                       strong_decode_for=key, label="strong-running",
                       request_key=DecoderRequestKey(*key, DecoderTier.STRONG, 0))
    manager.enqueue(strong)                        # starts immediately
    assert manager.pool_free["default"] == 0
    eng.schedule(us(5), lambda: manager.cancel_strong(key))

    freed_at = []
    def watch():
        if manager.pool_free["default"] == 1 and not freed_at:
            freed_at.append(eng.now)
        if eng.now < us(60):
            eng.schedule(us(1), watch)
    eng.schedule(us(5), watch)
    eng.run()
    assert freed_at and freed_at[0] == us(5)       # freed AT the cancel
    # the stale completion event at t=30us must NOT double-free
    assert manager.pool_free["default"] == 1
    assert manager.strong_cancelled == 1
