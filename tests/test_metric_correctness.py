"""Metric correctness (validation-matrix row 20).

The observer metrics sample AFTER each engine event, which gives them a
documented blind spot: activity that happens before the first event (jobs
dispatched straight from submit_decode calls, not from inside an event) is
invisible to them, while the cluster's own submit-time records see it. Any
experiment that reads these metrics under load (e.g. a backlog replication)
must know exactly how big that distortion is and where the ground truth
lives. These tests pin the arithmetic by hand.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from decsim.codes import SurfaceCodeModel
from decsim.config import us
from decsim.controllers import ModularController, LinkModel
from decsim.decoder_manager import DecoderManager
from decsim.decoders import CodeRouter, PresetLatencyDecoder
from decsim.engine import Engine
from decsim.message import DecodeResult, Operation
from decsim.metrics import DecodeBacklog, DecoderUtilization, ReadyQueueStats
from decsim.schedulers import FifoScheduler
from decsim.schemes import SlidingWindowScheme
from decsim.run_spec import RunSpec, simulate
from decsim.planner import FixedRounds


def test_mixed_pool_utilization_and_queue_arithmetic_by_hand():
    """Two pools of one unit each, two 10us jobs per pool, all submitted at
    t=0. Hand-derived timeline:

      t=0     d0 and s0 dispatch immediately (one per pool); d1, s1 queue.
              TRUE state: 2/2 units busy, queue depth peaked at 2.
      t=10us  both dones fire (2 events); each pool dispatches its second job.
      t=20us  second dones fire; run ends. engine.now = 20us.

    TRUE utilization is 1.0 (both units busy the whole run). The observer's
    first sample happens at t=10us, so the first 10us of BOTH units is in the
    blind spot and it must report exactly

        area = 2 units * 10us (the 10..20us stretch) / (2 units * 20us) = 0.5.

    The queue observer's first sample also lands after the t=10us dispatches
    (depth 1 -- only one of d1/s1 left waiting at that instant it samples),
    so its peak is 1, while the submit-time queue_log holds the true peak 2.
    Both numbers are correct FOR WHAT THEY MEASURE; this test pins the
    difference so no experiment mistakes one for the other."""
    engine = Engine(verbose=False)
    cluster = DecoderManager(engine,
                             router=CodeRouter(PresetLatencyDecoder(10.0)),
                             scheduler=FifoScheduler(),
                             unit_pools={"default": 1, "strong": 1})
    util, queue = DecoderUtilization(cluster), ReadyQueueStats(cluster)
    engine.add_metric(util)
    engine.add_metric(queue)
    for i in range(2):
        cluster.submit_decode(6, lambda: None, label=f"d{i}")
        cluster.submit_decode(6, lambda: None, label=f"s{i}", hint="strong")
    engine.run()

    assert engine.now == us(20)
    assert util.result() == 0.5                    # true 1.0 minus blind spot
    assert queue.result()["peak"] == 1             # observer view
    assert max(depth for _, depth in cluster.queue_log) == 2   # submit truth
    # and the log records the exact submit-time sequence at t=0:
    assert [q for t, q in cluster.queue_log if t == 0] == [1, 0, 1, 0, 1, 2]


class _FixedLatencyDecoder:
    def __init__(self, latency_us):
        self._t = us(latency_us)

    def latency(self, job):
        return self._t

    def decode(self, job):
        return DecodeResult(job.op_id, job.window_id, logical_value=0)


def test_decode_backlog_summary_is_consistent_with_its_own_trace():
    """The backlog metric's peak and time-average must be recomputable from
    the trace it publishes (rows()); otherwise no experiment can trust the
    summary numbers or regenerate a backlog-over-time plot from saved rows.
    Overloaded regime (f = 2) so the backlog is genuinely nonzero."""
    captured = []

    def make_metrics(engine, cluster, chip, factory):
        metric = DecodeBacklog(cluster)
        captured.append(metric)
        return [metric]

    op = Operation(0, "mem", (0,), clifford=True, patches=(0,))
    simulate(RunSpec(
        ops=[op],
        num_units=1,
        rounds_policy=FixedRounds(18),
        round_us=1.0,
        decoder=_FixedLatencyDecoder(6.0),
        scheme=SlidingWindowScheme(),
        code=SurfaceCodeModel(d=3),
        make_controller=lambda e: ModularController(
            e, links=LinkModel(qc=0, cd=0, dd=0, do=0, oc=0, cq=0),
            log_syndromes=False),
        make_metrics=make_metrics,
    ), verbose=False)

    metric = captured[0]
    summary = metric.result()
    trace = metric.trace
    assert trace, "overloaded run must produce a nonempty backlog trace"
    assert summary["peak_rounds"] == max(v for _, v in trace)
    assert trace[-1][1] == 0, "backlog must drain to zero by end of run"
    # recompute the time integral of the step function the trace describes,
    # over [0, last-observe-time], exactly as the metric accumulated it
    end = metric._t
    area = 0.0
    for (t0, v0), (t1, _) in zip(trace, trace[1:]):
        area += v0 * (t1 - t0)
    area += trace[-1][1] * (end - trace[-1][0])
    assert summary["time_avg_rounds"] == area / end
