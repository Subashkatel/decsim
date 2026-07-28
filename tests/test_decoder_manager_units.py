"""Typed decoder-unit pools (cluster.unit_pools).

Regression for a resource-accounting inaccuracy: every decode used to draw from one
anonymous unit pool, so a slow strong-decoder job could occupy -- and make ready weak
windows queue behind -- a unit that models weak hardware. Each pool now owns its units
AND its own ready queue, picked by job.hint at enqueue time (arXiv:2510.25222 Fig 1:
weak = FPGA/ASIC, strong = CPU/GPU). No pools configured = one "default" pool,
byte-identical to before."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from decsim.decoders import CodeRouter
from decsim.decoder_manager import DecoderManager
from decsim.config import us
from decsim.decoders import PresetLatencyDecoder
from decsim.engine import Engine
from decsim.frontends.circuit import cnot_plus_two_t_circuit
from decsim.schedulers import FifoScheduler
from decsim.run_spec import RunSpec, simulate
from decsim.planner import FixedRounds


def _run(**kw):
    r = simulate(RunSpec(ops=cnot_plus_two_t_circuit(), d=3,
                         rounds_policy=FixedRounds(11),
                         decoder=PresetLatencyDecoder(1.0), **kw), verbose=False)
    return r.engine.log_lines


def test_default_pool_matches_plain_num_units():
    """unit_pools={"default": n} is exactly num_units=n -- the byte-identical guarantee."""
    assert _run(num_units=2) == _run(unit_pools={"default": 2})


def test_idle_extra_pool_changes_nothing():
    """A strong pool no job targets must not alter the trace in any way."""
    assert _run(num_units=2) == _run(unit_pools={"default": 2, "strong": 1})


def test_strong_jobs_queue_on_their_own_unit():
    """Two strong jobs and one default job, one unit each: the default job runs at t=0
    even though the strong unit is busy, and the second strong job waits for the FIRST
    STRONG job -- not for the default unit."""
    engine = Engine(verbose=False)
    cluster = DecoderManager(engine, router=CodeRouter(PresetLatencyDecoder(10.0)),
                       scheduler=FifoScheduler(),
                       unit_pools={"default": 1, "strong": 1})
    done = {}
    cluster.submit_decode(6, lambda: done.update(A=engine.now), label="A", hint="strong")
    cluster.submit_decode(6, lambda: done.update(B=engine.now), label="B", hint="strong")
    cluster.submit_decode(6, lambda: done.update(C=engine.now), label="C")
    engine.run()
    assert done["A"] == us(10.0)      # started at t=0 on the strong unit
    assert done["C"] == us(10.0)      # started at t=0 on the default unit, unblocked
    assert done["B"] == us(20.0)      # queued behind A on the strong unit only
    assert any("strong units free now 0" in l for l in engine.log_lines)


def test_unknown_hint_runs_on_the_default_pool():
    """A hint that names no pool is only a router hint -- the job uses default units."""
    engine = Engine(verbose=False)
    cluster = DecoderManager(engine, router=CodeRouter(PresetLatencyDecoder(10.0)),
                       scheduler=FifoScheduler(), num_units=1)
    done = {}
    cluster.submit_decode(6, lambda: done.update(A=engine.now), label="A", hint="gpu")
    engine.run()
    assert done["A"] == us(10.0)


def test_pool_validation_fails_loudly():
    kwargs = dict(router=CodeRouter(PresetLatencyDecoder(1.0)),
                  scheduler=FifoScheduler())
    with pytest.raises(ValueError, match='"default" pool'):
        DecoderManager(Engine(verbose=False), unit_pools={"strong": 1}, **kwargs)
    with pytest.raises(ValueError, match="at least 1 unit"):
        DecoderManager(Engine(verbose=False),
                 unit_pools={"default": 1, "strong": 0}, **kwargs)


def test_units_conserved_for_inline_switching_timing_paths():
    """Either sampled timing path returns every occupied decoder unit."""
    from decsim.decoders import SwitchingDecoder
    for seed in range(5):
        sw = SwitchingDecoder(PresetLatencyDecoder(1.0), PresetLatencyDecoder(10.0),
                              gamma_switch=0.5)
        r = simulate(RunSpec(ops=cnot_plus_two_t_circuit(), d=3,
                             rounds_policy=FixedRounds(11), decoder=sw,
                             unit_pools={"default": 2, "strong": 1},
                             seed=seed), verbose=False)
        cluster = r.cluster
        assert cluster.pool_free == cluster.unit_totals, \
            f"seed {seed}: a unit leaked into the wrong pool"


def test_pools_with_deadline_scheduler_complete_and_conserve():
    """EDF sorts each pool's queue independently; every job finishes and every pool's
    units all come back."""
    from decsim.schedulers import EarliestDeadlineScheduler
    engine = Engine(verbose=False)
    cluster = DecoderManager(engine, router=CodeRouter(PresetLatencyDecoder(7.0)),
                       scheduler=EarliestDeadlineScheduler(),
                       unit_pools={"default": 2, "strong": 2})
    done = []
    for i in range(6):
        cluster.submit_decode(6, lambda i=i: done.append(i), label=f"s{i}",
                              hint="strong", deadline=us(100 - i))
        cluster.submit_decode(6, lambda i=i: done.append(i + 10), label=f"d{i}",
                              deadline=us(100 - i))
    engine.run()
    assert len(done) == 12
    assert cluster.pool_free == cluster.unit_totals


def test_metrics_see_every_pool():
    """Utilization and queue depth must count ALL pools: with traffic ONLY on the
    strong pool, a default-pool-only metric would read 0.0 / 0 (the blind spot this
    test pins down)."""
    from decsim.metrics import DecoderUtilization, ReadyQueueStats
    engine = Engine(verbose=False)
    cluster = DecoderManager(engine, router=CodeRouter(PresetLatencyDecoder(10.0)),
                       scheduler=FifoScheduler(),
                       unit_pools={"default": 1, "strong": 1})
    util, queue = DecoderUtilization(cluster), ReadyQueueStats(cluster)
    engine.add_metric(util)
    engine.add_metric(queue)
    for i in range(4):                  # 4 back-to-back 10us jobs on the one strong unit
        cluster.submit_decode(6, lambda: None, label=f"s{i}", hint="strong")
    engine.run()
    # 1 of 2 units busy across the run's decode-done events. The first 10us fall
    # BEFORE the first engine event (the job dispatched from the direct submit call,
    # not inside an event), and this observe-after-event metric pattern cannot see
    # busy time before the first event: 30us busy / (2 units * 40us) = 0.375.
    assert util.result() == 0.375
    # same pattern for the queue: depth 3 existed only before the first event, so
    # the metric's first sample (after decode-done #1) sees 2 ...
    assert queue.result()["peak"] == 2
    # ... while the cluster's own queue_log records AT SUBMIT TIME and saw all 3.
    assert max(q for _, q in cluster.queue_log) == 3
