"""Typed metric views + metrics-as-observers.

Simulation-driven checks: a real run with the full metric set publishes the
expected numbers, and the frozen view schemas are populated by a real run.
"""
import dataclasses

import pytest

from decsim.config import us
from decsim.decoders import (PerRoundDecoder, SampledConfidenceDecoder,
                             SwitchingRouter)
from decsim.frontends.circuit import CircuitFrontend, cnot_plus_two_t_circuit
from decsim.message import Operation
from decsim.metrics import (BacklogTrajectory, ConditionalReactionTime,
                            DecodeBacklog, DecoderUtilization, ReadyQueueStats,
                            StrongDecoderBacklog, WindowLatencyBreakdown)
from decsim.planner import FixedRounds, GateRounds
from decsim.run_spec import RunSpec, simulate
from decsim.switching import Switching
from decsim.views import (BacklogView, OpReactionInfo, ReactionView,
                          StrongPoolView, TruthView, UtilizationView,
                          WindowLatencyView, WindowStageRow, backlog_view,
                          reaction_view, strong_pool_view, utilization_view,
                          window_latency_view)

SEED = 7


def t_then_blocked_t():
    return CircuitFrontend([
        Operation(0, "A:T(q0)", (0,), clifford=False),
        Operation(1, "B:T(q0)", (0,), clifford=False, blocked_by=0),
    ]).build()


def _full_metrics(engine, cluster, chip, factory):
    return [DecoderUtilization(cluster), ReadyQueueStats(cluster),
            WindowLatencyBreakdown(cluster), DecodeBacklog(cluster),
            BacklogTrajectory(chip), ConditionalReactionTime(chip)]


def _switch_metrics(engine, cluster, chip, factory):
    return _full_metrics(engine, cluster, chip, factory) + [
        StrongDecoderBacklog(cluster)]


#==================================================================
# PUBLIC METRIC NUMBERS from a real simulation
#==================================================================

def test_metric_numbers_feedback_circuit():
    res = simulate(RunSpec(ops=t_then_blocked_t(), d=3,
                           rounds_policy=FixedRounds(11),
                           decoder=PerRoundDecoder(3.0), num_units=1,
                           make_metrics=_full_metrics))
    # the run exercised the reaction path (release recorded, waits non-zero)
    reaction = res["metrics"]["conditional_reaction_time"]
    assert reaction["released_conditionals"] == 1
    assert reaction["max_wait_rounds"] > 0
    assert res["metrics"]["backlog_trajectory"]["n"] == 1


def test_metric_numbers_switching_pools():
    weak = SampledConfidenceDecoder(PerRoundDecoder(0.2), 0.6)
    strong = PerRoundDecoder(3.0)
    res = simulate(RunSpec(ops=cnot_plus_two_t_circuit(),
                           rounds_policy=FixedRounds(11), d=3, decoder=weak,
                           router=SwitchingRouter(weak, strong),
                           unit_pools={"default": 1, "strong": 1},
                           strategy=Switching(confidence_threshold=0.5),
                           make_metrics=_switch_metrics,
                           seed=SEED))
    assert res["metrics"]["strong_backlog"]["peak_jobs"] >= 1


#==================================================================
# VIEW SCHEMAS (spec §8 test 9): frozen, populated by a real run
#==================================================================

def test_views_are_frozen():
    views = [UtilizationView(0, 1, ()),
             BacklogView(0, (), (), (), 0),
             WindowLatencyView(rows=()),
             WindowStageRow(0, 0, 0, 0, 0, 0, 0),
             OpReactionInfo(0, "A", None, 1, 1),
             ReactionView(None, 0, (), (), (), ()),
             TruthView((), ()),
             StrongPoolView("strong", 0, 0, 0, 0, 0)]
    for view in views:
        field = dataclasses.fields(view)[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(view, field, None)


def test_reaction_view_populated_by_real_run():
    res = simulate(RunSpec(ops=t_then_blocked_t(), d=3,
                           rounds_policy=FixedRounds(11),
                           decoder=PerRoundDecoder(3.0), num_units=1))
    view = reaction_view(res["chip"])
    assert view.chip_done == res["chip_done"]
    assert view.fully_done == res["fully_done"]
    body_done = dict(view.body_done_time)
    released = dict(view.decode_release_time)
    assert set(body_done) == {0, 1}
    assert released[1] > body_done[0]          # B waited on A's decode
    assert view.idle_cap_hits == ()            # no cap in this scenario
    infos = {op.op: op for op in view.ops}
    assert infos[1].blocked_by == 0 and infos[0].blocked_by is None
    assert infos[0].rounds == 11 and infos[0].round_ticks > 0


def test_backlog_window_truth_strong_views_populated_by_real_run():
    def with_strong(engine, cluster, chip, factory):
        return [StrongDecoderBacklog(cluster)]

    weak = SampledConfidenceDecoder(PerRoundDecoder(0.2), 0.6)
    res = simulate(RunSpec(ops=cnot_plus_two_t_circuit(), d=3,
                           rounds_policy=FixedRounds(11),
                           strategy=Switching(confidence_threshold=0.5),
                           decoder=weak,
                           router=SwitchingRouter(weak, PerRoundDecoder(3.0)),
                           unit_pools={"default": 1, "strong": 1},
                           make_metrics=with_strong,
                           seed=SEED))
    cluster = res["cluster"]

    util = utilization_view(cluster)
    assert util.total_units == 2 and util.busy_units == 0     # end of run: idle
    assert dict((p, (b, t)) for p, b, t in util.per_pool) == {
        "default": (0, 1), "strong": (0, 1)}

    backlog = backlog_view(cluster)
    assert backlog.total_rounds == sum(w for _, w in backlog.per_op_rounds)
    assert backlog.total_rounds == sum(w for _, w in backlog.per_patch_rounds)
    assert dict(backlog.per_lane)[""] == 0                    # queues drained

    latency = window_latency_view(cluster)
    assert len(latency.rows) == cluster.total_windows
    for row in latency.rows:
        assert row.total == (row.buffer_fill + row.dep_block
                             + row.queue_wait + row.service)

    strong = strong_pool_view(cluster)
    assert strong.total_units == 1 and strong.busy_units == 0
    assert strong.redo_rounds == cluster.commit + 2 * cluster.buffer


#==================================================================
# HAND-COMPUTED UTILIZATION THROUGH THE VIEW
#==================================================================

def test_utilization_metric_reproduces_hand_computed_number():
    class FakeCluster:
        num_units = 2
        free_units = 1          # 1 busy of 2

    class FakeEngine:
        now = 0

    cluster, engine = FakeCluster(), FakeEngine()
    metric = DecoderUtilization(cluster)
    metric.observe(engine)                  # t=0: sample busy=1
    engine.now = 10
    cluster.free_units = 0                  # 2 busy of 2
    metric.observe(engine)                  # [0,10) held busy=1
    engine.now = 20
    metric.observe(engine)                  # [10,20) held busy=2
    assert metric.result() == (1 * 10 + 2 * 10) / (2 * 20)   # 0.75


#==================================================================
# RUN CONFIGURATION DEFAULTS
#==================================================================

def test_runspec_default_resolves_gate_rounds():
    spec = RunSpec(ops=t_then_blocked_t(), decoder=PerRoundDecoder(0.5))
    assert spec.rounds_policy is None          # RunSpec resolves in build()
    world = spec.build()
    assert isinstance(world.window_manager.rounds_policy, GateRounds)
