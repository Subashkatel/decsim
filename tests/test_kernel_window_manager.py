"""Window runtime kernel tests — each maps to a numbered contract rule.

These drive a real Engine with fakes at the ports; end-to-end validation
against real decoders/streams happens via the frozen timing goldens (Task 19).
"""
import pytest

from decsim.engine import Engine
from decsim.message import (DecodeJob, DecodeResult, Operation, SyndromePayload,
                          Window, WindowPlan)
from decsim.protocols import Directive, OutcomeDirective, Submission
from decsim.window_manager import WindowManager

T_DD = 500_000
T_DO = 1_000_000


class _Link:
    def __init__(self, ticks): self._t = ticks
    def cost(self): return self._t


class _Links:
    dd = _Link(T_DD)
    do = _Link(T_DO)


class _Code:
    distance = 3
    name = "fake"
    def commit_rounds(self): return 3
    def buffer_rounds(self): return 3


class _Layout:
    def code_for_op(self, op): return _Code()
    def spatial_nodes_for(self, op): return 9


class _Rounds:
    def rounds_for(self, op, code): return 6


class _Scheme:
    batches_idle_rounds_into_next_op = False
    def data_complete(self, w, *, rounds_arrived, successor_rounds,
                      memory_rounds, round_count, has_successor, op, layout):
        # sliding-window rule: all rounds up to buffer_hi present (or op ended)
        return rounds_arrived + memory_rounds >= min(w.buffer_hi, round_count)


class _Deadline:
    def deadline(self, op, window, now, on_reaction_path):
        return now + 1_000


class _Feedback:
    def __init__(self): self.integrated = []
    def integrate(self, op, result): self.integrated.append((op.id, result))


class _Eager:
    def on_commit(self, window, final): return True


class _Held:
    def on_commit(self, window, final): return final


class _RecordingStrategy:
    """Baseline-like: submit the weak job; FINALIZE unless told otherwise."""
    def __init__(self, directive=None):
        self.ready, self.outcomes = [], []
        self._directive = directive or OutcomeDirective(Directive.FINALIZE)
    def on_window_ready(self, window, weak_job, services):
        self.ready.append(weak_job)
        return [Submission(weak_job)]
    def on_decode_outcome(self, outcome, services):
        self.outcomes.append(outcome)
        return self._directive
    def metrics(self): return {}


def _runtime(boundary=None, strategy=None, ops=(0,), deps=(), blocking=()):
    eng = Engine(verbose=False)
    fb = _Feedback()
    rt = WindowManager(eng, scheme=_Scheme(), layout=_Layout(),
                       rounds_policy=_Rounds(), code=_Code(),
                       deadline_policy=_Deadline(), links=_Links(),
                       orchestrator=fb, boundary_policy=boundary or _Eager())
    rt.strategy = strategy or _RecordingStrategy()
    rt.services = object()
    submitted = []
    rt.submit_fn = lambda job, delay: submitted.append((job, delay))
    windows, op_windows, count = {}, {}, {}
    for op_id in ops:
        op = Operation(op_id, f"op{op_id}", (op_id,),
                       blocked_by=(op_id - 1 if op_id in blocking else None))
        rt.register_op(op)
        w = Window(op_id=op_id, k=0, commit_lo=1, commit_hi=3, buffer_hi=6,
                   n_rounds=6)
        windows[(op_id, 0)] = w
        op_windows[op_id] = [0]
        count[op_id] = 1
    for src, dst in deps:
        windows[dst].deps.append(src)
        windows[dst].deps_remaining += 1
        windows[src].dependents.append(dst)
    rt.load_execution_plan(WindowPlan(
        windows=windows, window_count=count, op_windows=op_windows,
        successors={op_id: [] for op_id in ops},
        spatial_nodes={}, total_windows=len(windows)))
    return eng, rt, fb, submitted


def _feed_rounds(rt, op_id, n):
    for r in range(1, n + 1):
        rt.on_syndrome_arrival(SyndromePayload(op_id, 0, r))


def test_not_ready_until_data_and_deps():
    eng, rt, fb, submitted = _runtime(deps=[((0, 0), (1, 0))], ops=(0, 1))
    _feed_rounds(rt, 1, 6)                       # dependent has data...
    assert not submitted                          # ...but dep outstanding
    _feed_rounds(rt, 0, 6)                        # predecessor ready & submits
    assert [j.op_id for j, _ in submitted] == [0]


def test_job_fields_match_contract_2a4():
    eng, rt, fb, submitted = _runtime()
    _feed_rounds(rt, 0, 6)
    job, delay = submitted[0]
    assert delay == 0
    assert (job.op_id, job.window_id, job.n_rounds) == (0, 0, 6)
    assert job.deadline == eng.now + 1_000 and job.spatial_nodes == 9
    assert len(job.payloads) == 6 and job.strong_label == "strong(op0 W0)"


def test_eager_ships_weak_boundary_unconditionally_contract_1_2():
    strat = _RecordingStrategy()
    eng, rt, fb, submitted = _runtime(deps=[((0, 0), (1, 0))], ops=(0, 1),
                                      strategy=strat)
    _feed_rounds(rt, 0, 6)
    job, _ = submitted[0]
    job.awaiting_strong_result = True             # escalated (set by decode layer)
    rt.on_decode_done(job, DecodeResult(0, 0, logical_value=1,
                                        boundary_defects={7: [1, 0, 1]}))
    dep = rt.windows[(1, 0)]
    assert dep.deps_remaining == 1                # not yet: travels t_dd
    eng.run(until=T_DD)
    assert dep.deps_remaining == 0                # Eager shipped despite pending strong
    assert dep.boundary_in == {1: [1, 0, 1]}      # src round 7 - rounds_for(6) -> dep round 1


def test_boundary_shift_rule_contract_1_3():
    eng, rt, fb, submitted = _runtime(deps=[((0, 0), (1, 0))], ops=(0, 1))
    _feed_rounds(rt, 0, 6)
    job, _ = submitted[0]
    # src op0 has rounds_for=6; defects at src rounds 7,8 -> dep rounds 1,2;
    # src round 3 -> dep round -3 dropped
    rt.on_decode_done(job, DecodeResult(0, 0, boundary_defects={
        7: [1], 8: [1, 1], 3: [1, 1, 1]}))
    eng.run(until=T_DD)
    dep = rt.windows[(1, 0)]
    assert set(dep.boundary_in) == {1, 2}


def test_strong_revises_logical_only_contract_1_4():
    eng, rt, fb, submitted = _runtime(deps=[((0, 0), (1, 0))], ops=(0, 1))
    _feed_rounds(rt, 0, 6)
    job, _ = submitted[0]
    job.awaiting_strong_result = True
    rt.on_decode_done(job, DecodeResult(0, 0, logical_value=1,
                                        boundary_defects={7: [1]}))
    eng.run(until=T_DD)
    dep_boundary_before = dict(rt.windows[(1, 0)].boundary_in)
    assert rt.op_results[0] == 1
    rt.on_strong_decode_done((0, 0), DecodeResult(0, 0, logical_value=0,
                                                  boundary_defects={7: [1, 1]}))
    assert rt.op_results[0] == 0                              # XOR-swapped
    assert rt.windows[(1, 0)].boundary_in == dep_boundary_before  # untouched
    assert rt.op_strong_commit_time[0] == eng.now


def test_op_delivery_gated_on_pending_strong_contract_1_5():
    eng, rt, fb, submitted = _runtime()
    _feed_rounds(rt, 0, 6)
    job, _ = submitted[0]
    job.awaiting_strong_result = True
    rt.on_decode_done(job, DecodeResult(0, 0, logical_value=1))
    eng.run()
    assert fb.integrated == []                    # gated: pending strong
    rt.on_strong_decode_done((0, 0), DecodeResult(0, 0, logical_value=1))
    eng.run()
    assert [op_id for op_id, _ in fb.integrated] == [0]   # released after final


def test_op_delivery_immediate_when_not_awaiting():
    eng, rt, fb, submitted = _runtime()
    _feed_rounds(rt, 0, 6)
    job, _ = submitted[0]
    rt.on_decode_done(job, DecodeResult(0, 0, logical_value=1))
    assert fb.integrated == []                    # travels t_do
    eng.run(until=T_DO)
    assert [op_id for op_id, _ in fb.integrated] == [0]
    assert fb.integrated[0][1].logical_value == 1


def test_held_ships_only_when_final():
    eng, rt, fb, submitted = _runtime(boundary=_Held(),
                                      deps=[((0, 0), (1, 0))], ops=(0, 1))
    _feed_rounds(rt, 0, 6)
    job, _ = submitted[0]
    job.awaiting_strong_result = True
    rt.on_decode_done(job, DecodeResult(0, 0, logical_value=1,
                                        boundary_defects={7: [1]}))
    eng.run()
    assert rt.windows[(1, 0)].deps_remaining == 1   # held: nothing shipped
    rt.on_strong_decode_done((0, 0), DecodeResult(0, 0, logical_value=1,
                                                  boundary_defects={7: [1]}))
    eng.run()
    assert rt.windows[(1, 0)].deps_remaining == 0   # shipped at final


def test_late_round_after_op_freed_raises():
    eng, rt, fb, submitted = _runtime()
    _feed_rounds(rt, 0, 6)
    job, _ = submitted[0]
    rt.on_decode_done(job, DecodeResult(0, 0, logical_value=0))
    eng.run()
    with pytest.raises(RuntimeError, match="syndrome RAM was freed"):
        rt.on_syndrome_arrival(SyndromePayload(0, 0, 7))


def test_strong_job_two_sided_context_contract_2b6():
    eng, rt, fb, submitted = _runtime()
    _feed_rounds(rt, 0, 6)
    weak, _ = submitted[0]
    strong = rt.make_strong_decode_job(weak, round_count=9, label="strong")
    w = strong.window
    assert (w.buffer_lo, w.commit_lo, w.commit_hi, w.buffer_hi) == (1, 1, 3, 6)
    assert strong.hint == "strong" and strong.attempt == 1
    assert strong.strong_decode_for == (0, 0) and strong.deadline == eng.now
