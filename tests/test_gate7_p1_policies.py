"""Gate 7 P1: BufferExpiryDeadline + DistanceLanes focused tests.

Predeclaration: docs/validation/2026-07-03-gate7-p1-predeclaration.md.
Covers: deadline-stamp semantics, EDF-under-expiry ordering, lane
routing precedence (hint > lane > default), deterministic lane
isolation, and queue-conservation with lanes (extends the V9 net).
Synthetic service times throughout (no latency claims).
"""
import random
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from decsim.config import us
from decsim.codes import SurfaceCodeModel
from decsim.engine import Engine
from decsim.decoders import CodeRouter, PerRoundDecoder
from decsim.decoder_manager import DecoderManager
from decsim.message import (Operation, RunSeedPathSegment, RunSeedReservation,
                            Window)
from decsim.planner import FixedRounds
from decsim.run_spec import RunSpec, simulate
from decsim.schedulers import (BufferExpiryDeadline, DistanceLanes,
                               EarliestDeadlineScheduler, FifoScheduler)
from decsim.seeding import derive_component_seed


def make_window(t_first_round) -> Window:
    w = Window(op_id=0, k=0, commit_lo=1, commit_hi=3, buffer_hi=5,
               n_rounds=5)
    w.t_first_round = t_first_round
    return w


# ------------------------------------------------------ deadline semantics

def test_buffer_expiry_deadline_is_first_round_plus_capacity():
    policy = BufferExpiryDeadline(capacity_rounds=40, round_ticks=us(1))
    w = make_window(t_first_round=us(100))
    expected = us(100) + 40 * us(1)
    # independent of now and of on_reaction_path (pure buffer semantics)
    assert policy.deadline(None, w, now=us(500),
                           on_reaction_path=False) == expected
    assert policy.deadline(None, w, now=us(7),
                           on_reaction_path=True) == expected


def test_buffer_expiry_deadline_rejects_missing_arrival_provenance():
    policy = BufferExpiryDeadline(capacity_rounds=10, round_ticks=us(2))
    w = make_window(t_first_round=None)
    with pytest.raises(
        RuntimeError,
        match=r"window \(0, 0\).*arrival provenance is missing",
    ):
        policy.deadline(None, w, now=us(30), on_reaction_path=False)


def test_older_windows_get_tighter_deadlines():
    """The buffer-expiry consequence: older first-round -> earlier expiry."""
    policy = BufferExpiryDeadline(capacity_rounds=40, round_ticks=us(1))
    old = policy.deadline(None, make_window(us(10)), now=us(200),
                          on_reaction_path=False)
    fresh = policy.deadline(None, make_window(us(150)), now=us(200),
                            on_reaction_path=False)
    assert old < fresh


def test_edf_completes_in_buffer_expiry_order():
    """EDF + expiry-stamped deadlines: completion order == data age order,
    NOT submission order."""
    eng = Engine(verbose=False)
    manager = DecoderManager(
        eng, router=CodeRouter(default=PerRoundDecoder(tau_us=1.0)),
        scheduler=EarliestDeadlineScheduler(), unit_pools={"default": 1})
    policy = BufferExpiryDeadline(capacity_rounds=40, round_ticks=us(1))
    order = []
    manager.submit_decode(50, lambda: order.append("head"), label="head",
                          deadline=us(1))     # busy the unit so the rest queue
    # submission order deliberately != data-age order
    first_rounds = [us(x) for x in (30, 5, 40, 15, 25, 10, 35, 20)]
    for k, fr in enumerate(first_rounds):
        dl = policy.deadline(None, make_window(fr), now=eng.now,
                             on_reaction_path=False)
        manager.submit_decode(1, lambda k=k: order.append(k),
                              label=f"exp{k}", deadline=dl)
    eng.run()
    assert order[0] == "head"
    by_age = [k for k, _ in sorted(enumerate(first_rounds),
                                   key=lambda kv: kv[1])]
    assert order[1:] == by_age, (order[1:], by_age)


# ---------------------------------------------------------- lane routing

def dist_of(job):
    """Test-side distance extraction: code names like 'surface_d7'."""
    if job.code and "_d" in job.code:
        return int(job.code.rsplit("_d", 1)[1])
    return None


def build_laned(units_by_pool: dict):
    eng = Engine(verbose=False)
    manager = DecoderManager(
        eng, router=CodeRouter(default=PerRoundDecoder(tau_us=1.0)),
        scheduler=FifoScheduler(), unit_pools=units_by_pool,
        lane_policy=DistanceLanes({7: "mid", 11: "heavy"}, dist_of))
    return eng, manager


def test_lane_routing_precedence_hint_then_lane_then_default():
    eng, manager = build_laned({"default": 1, "mid": 1, "heavy": 1})
    from decsim.message import DecodeJob
    j_d3 = DecodeJob(op_id=-1, window_id=0, n_rounds=1, code="surface_d3")
    j_d7 = DecodeJob(op_id=-1, window_id=0, n_rounds=1, code="surface_d7")
    j_d11 = DecodeJob(op_id=-1, window_id=0, n_rounds=1, code="surface_d11")
    j_hint = DecodeJob(op_id=-1, window_id=0, n_rounds=1,
                       code="surface_d11", hint="mid")
    j_none = DecodeJob(op_id=-1, window_id=0, n_rounds=1, code=None)
    assert manager.pool_for(j_d3) == "default"     # no lane for d=3
    assert manager.pool_for(j_d7) == "mid"
    assert manager.pool_for(j_d11) == "heavy"
    assert manager.pool_for(j_hint) == "mid"       # explicit hint wins
    assert manager.pool_for(j_none) == "default"   # unknown distance


def test_lane_naming_a_missing_pool_falls_back_to_default():
    eng = Engine(verbose=False)
    manager = DecoderManager(
        eng, router=CodeRouter(default=PerRoundDecoder(tau_us=1.0)),
        scheduler=FifoScheduler(), unit_pools={"default": 1},
        lane_policy=DistanceLanes({5: "nonexistent"}, dist_of))
    from decsim.message import DecodeJob
    j = DecodeJob(op_id=-1, window_id=0, n_rounds=1, code="surface_d5")
    assert manager.pool_for(j) == "default"


def test_lane_conservation_and_drain():
    """V9 invariant net extended to lane-routed jobs."""
    eng, manager = build_laned({"default": 2, "mid": 1, "heavy": 1})
    rng = random.Random(20260703)
    totals = dict(manager.unit_totals)
    state = {"submitted": 0, "completed": 0}
    violations = []

    def check():
        running = sum(totals[p] - manager.pool_free[p] for p in totals)
        if state["submitted"] != (state["completed"]
                                  + manager.queued_total() + running):
            violations.append(("conservation", eng.now))

    def submit(i):
        state["submitted"] += 1
        code = f"surface_d{rng.choice([3, 7, 11])}"
        manager.submit_decode(
            rng.randint(1, 8),
            lambda: (state.__setitem__("completed", state["completed"] + 1),
                     check()),
            label=f"lane{i}", code=code)
        check()

    for i in range(200):
        eng.schedule(rng.randint(0, us(50)), lambda i=i: submit(i))
    eng.run()
    assert not violations, violations[:5]
    assert state["completed"] == 200
    assert manager.queued_total() == 0
    assert manager.pool_free == totals


def test_expiry_stamp_for_contiguous_window_is_c_minus_r_plus_one():
    """Pins the derivation (Codex G7P1 review finding 2): a contiguous
    r-round window whose LAST round arrives at `arrival` has its FIRST
    round at arrival - (r-1)*tick, so the policy deadline is
    arrival + (C - r + 1)*tick — one tick LOOSER than the g7p1 Part-A
    experiment stamp arrival + (C - r)*tick. The experiment formula is
    therefore a 1-tick-tighter variant (documented in the artifacts);
    relative conclusions are unaffected (Codex-recomputed)."""
    policy = BufferExpiryDeadline(capacity_rounds=40, round_ticks=us(1))
    r, arrival = 7, us(100)
    w = make_window(t_first_round=arrival - (r - 1) * us(1))
    assert policy.deadline(None, w, now=arrival, on_reaction_path=False) \
        == arrival + (40 - r + 1) * us(1)


def test_strong_hint_routes_to_strong_pool_despite_lanes():
    """Lane assignment must never capture strong re-decodes."""
    eng = Engine(verbose=False)
    manager = DecoderManager(
        eng, router=CodeRouter(default=PerRoundDecoder(tau_us=1.0)),
        scheduler=FifoScheduler(),
        unit_pools={"default": 1, "heavy": 1, "strong": 1},
        lane_policy=DistanceLanes({11: "heavy"}, dist_of))
    from decsim.message import DecodeJob
    j = DecodeJob(op_id=9, window_id=1, n_rounds=4, code="surface_d11",
                  hint="strong", strong_decode_for=(9, 1))
    assert manager.pool_for(j) == "strong"


def test_cancel_queued_strong_survives_unstable_lane_policy():
    """cancel_strong must find a queued job even when the lane policy
    is not stable between enqueue and cancel (queue scan, not
    pool_for recomputation)."""
    from decsim.message import DecodeJob

    class FlippingLanes:
        def __init__(self):
            self.calls = 0

        def pool_for(self, job):
            self.calls += 1
            return "heavy" if self.calls == 1 else None   # then default

    eng = Engine(verbose=False)
    manager = DecoderManager(
        eng, router=CodeRouter(default=PerRoundDecoder(tau_us=1.0)),
        scheduler=FifoScheduler(), unit_pools={"default": 1, "heavy": 1},
        lane_policy=FlippingLanes())
    # occupy the heavy pool so the strong job queues there
    manager.submit_decode(50, lambda: None, label="blocker", hint="heavy")
    key = (3, 2)
    strong = DecodeJob(op_id=3, window_id=2, n_rounds=4,
                       strong_decode_for=key, label="flip-cancel")
    manager.enqueue(strong)              # FlippingLanes -> heavy (call 1)
    assert len(manager.queue_for("heavy")) == 1
    manager.cancel_strong(key)           # policy now says default
    assert manager.queued_total() == 0, "queued strong job not removed"
    assert manager.strong_cancelled == 1
    eng.run()
    assert manager.pool_free == {"default": 1, "heavy": 1}


def test_lanes_protect_short_jobs_from_heavy_flood_deterministic():
    """Same total units (2): shared pool lets two long d=11 jobs occupy
    both units and starve the d=3 job; lanes keep a unit for it."""
    def run(units_by_pool, lane_policy):
        eng = Engine(verbose=False)
        manager = DecoderManager(
            eng, router=CodeRouter(default=PerRoundDecoder(tau_us=1.0)),
            scheduler=FifoScheduler(), unit_pools=units_by_pool,
            lane_policy=lane_policy)
        done = {}
        for n, (rounds, code) in enumerate(
                [(100, "surface_d11"), (100, "surface_d11")]):
            manager.submit_decode(rounds,
                                  lambda n=n: done.__setitem__(n, eng.now),
                                  label=f"long{n}", code=code)
        eng.schedule(us(1), lambda: manager.submit_decode(
            3, lambda: done.__setitem__("short", eng.now),
            label="short", code="surface_d3"))
        eng.run()
        return done["short"]

    shared = run({"default": 2}, None)
    laned = run({"default": 1, "heavy": 1},
                DistanceLanes({11: "heavy"}, dist_of))
    assert laned == us(4)              # 1us arrival + 3 rounds, no wait
    assert shared >= us(100)           # starved behind a long job


class _SeededRecordingLanePolicy:
    def __init__(self):
        self.jobs = []
        self.committed_seed = None

    def pool_for(self, job):
        self.jobs.append(job)
        return "heavy"

    def reserve_run_seed(self, seed):
        return RunSeedReservation("derived", seed, seed)

    def cancel_run_seed(self, reservation):
        pass

    def commit_run_seed(self, reservation):
        self.committed_seed = reservation.proposed_seed


def test_runspec_routes_through_and_seeds_the_supplied_lane_policy():
    lane_policy = _SeededRecordingLanePolicy()
    completed = simulate(RunSpec(
        ops=[Operation(0, "memory", (0,), patches=(0,))],
        code=SurfaceCodeModel(d=3),
        rounds_policy=FixedRounds(3),
        decoder=PerRoundDecoder(tau_us=1.0),
        unit_pools={"default": 1, "heavy": 1},
        lane_policy=lane_policy,
        seed=17,
    ))

    assert completed.decoder_manager.lane_policy is lane_policy
    assert lane_policy.jobs
    assert {job.pool for job in lane_policy.jobs} == {"heavy"}
    assert lane_policy.committed_seed == derive_component_seed(17, (
        RunSeedPathSegment("field", "lane_policy"),
    ))
