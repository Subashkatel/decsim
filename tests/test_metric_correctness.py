"""Metric correctness against hand-derived event ledgers."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from decsim.codes import SurfaceCodeModel
from conftest import fixed_latency_link_config
from decsim.config import us
from decsim.controllers import ModularController
from decsim.decoder_manager import DecoderManager
from decsim.decoders import CodeRouter, PresetLatencyDecoder
from decsim.engine import Engine
from decsim.detector_error_model import NO_FAULT_MODEL_REQUIRED
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

    Both units are busy for the full 20us, so aggregate and per-pool
    utilization are 1.0. Queue depth is 2 for [0, 10us) and 0 afterward, so
    its peak is 2 and time average is 1.0 job."""
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
    assert util.result() == {
        "observation_span_ticks": us(20),
        "aggregate_busy_fraction": 1.0,
        "aggregate_total_units": 2,
        "per_pool_busy_fraction": {"default": 1.0, "strong": 1.0},
        "per_pool_total_units": {"default": 1, "strong": 1},
    }
    assert queue.result() == {
        "observation_span_ticks": us(20),
        "peak_jobs": 2,
        "time_avg_jobs": 1.0,
    }
    assert max(depth for _, depth in cluster.queue_log) == 2
    # and the log records the exact submit-time sequence at t=0:
    assert [q for t, q in cluster.queue_log if t == 0] == [1, 0, 1, 0, 1, 2]


def test_late_registered_utilization_uses_its_own_observation_epoch():
    engine = Engine(verbose=False)
    engine.now = us(10)
    cluster = DecoderManager(
        engine,
        router=CodeRouter(PresetLatencyDecoder(10.0)),
        scheduler=FifoScheduler(),
        unit_pools={"default": 1},
    )
    cluster.pool_free["default"] = 0
    metric = DecoderUtilization(cluster)
    engine.add_metric(metric)
    engine.schedule(
        us(10),
        lambda: cluster.pool_free.__setitem__("default", 1),
    )

    engine.run()

    assert metric.result()["observation_span_ticks"] == us(10)
    assert metric.result()["aggregate_busy_fraction"] == 1.0


class _FixedLatencyDecoder:
    fault_model_requirement = NO_FAULT_MODEL_REQUIRED

    def __init__(self, latency_us):
        self._t = us(latency_us)

    def latency(self, job):
        return self._t

    def decode(self, job):
        return DecodeResult(job.op_id, job.window_id,
                            logical_observables=(0,))


def test_decode_backlog_summary_is_consistent_with_its_own_trace():
    """The backlog metric's peak and time-average must be recomputable from
    the trace it publishes (rows()); otherwise no experiment can trust the
    summary numbers or regenerate a backlog-over-time plot from saved rows.
    Overloaded regime (f = 2) so the backlog is genuinely nonzero."""
    captured = []

    def make_metrics(engine, window_manager, decoder_manager, chip, factory):
        metric = DecodeBacklog(window_manager, decoder_manager)
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
        links=fixed_latency_link_config(),
        make_controller=lambda e, links, buffering, window_manager: ModularController(
            e, links=links, log_syndromes=False,
            controller_capacity=buffering.controller_ingress_packet_slots,
            window_input_receiver=window_manager,
            feedback_memory_receiver=window_manager),
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
    end = metric._integral.last_tick
    area = 0.0
    for (t0, v0), (t1, _) in zip(trace, trace[1:]):
        area += v0 * (t1 - t0)
    area += trace[-1][1] * (end - trace[-1][0])
    assert summary["time_avg_rounds"] == area / end
