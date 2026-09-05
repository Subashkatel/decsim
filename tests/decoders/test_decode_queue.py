"""The waiting jobs against the queue closed forms and the scheduler law.

Referents: the trace-exact G/D/k FIFO form (the i-th job in arrival
order starts at max(arrival, earliest unit free) and every unit frees at
start + service) and the M/D/1 mean wait, Pollaczek-Khinchine
E[W_q] = rho S / (2 (1 - rho)) (rowD1, compare_queue_laws.py). Both laws
are computed inside the tests; the random-trace tests say so in their
names.
"""

import random
import statistics

import decsim.config as config
import decsim.decoders.decoders as decoders
import decsim.decoders.schedulers as schedulers
import decsim.engine as engine_module
from decsim.decoders.decoder_manager import DecoderManager

SERVICE_MICROSECONDS = 1.0
SERVICE_TICKS = config.microseconds_to_ticks(SERVICE_MICROSECONDS)


class ShortestJobFirst:
    """The job with the fewest rounds is served first, whatever its age."""

    def pop(self, queue: list):
        shortest = min(queue, key=_round_count)
        queue.remove(shortest)
        return shortest


def _round_count(job):
    return job.n_rounds


def _manager(engine, units, scheduler=None):
    if scheduler is None:
        scheduler = schedulers.FifoScheduler()
    decoder = decoders.PresetLatencyDecoder(SERVICE_MICROSECONDS)
    router = decoders.CodeRouter(decoder)
    return DecoderManager(
        engine,
        router=router,
        scheduler=scheduler,
        num_units=units,
        escalation_policy=None,
    )


def _start_ticks(arrivals, units, scheduler=None):
    """The tick each input-less job's decode began, in arrival order."""
    engine = engine_module.Engine(verbose=False)
    manager = _manager(engine, units, scheduler)
    starts = {}
    original_begin = manager.service.begin

    def recording_begin(job, gated=True):
        starts[job.label] = engine.now
        original_begin(job, gated)

    manager.service.begin = recording_begin
    for index, arrival in enumerate(arrivals):
        label = f"job{index}"
        engine.schedule(arrival, lambda label=label: _submit(manager, label))
    engine.run()
    ordered = []
    for index in range(len(arrivals)):
        ordered.append(starts[f"job{index}"])
    return ordered


def _submit(manager, label, round_count=1):
    manager.enqueue_without_input(round_count, _nothing, label=label)


def _nothing():
    pass


def _g_d_k_closed_form(arrivals, units):
    free_at = [0] * units
    starts = []
    for arrival in arrivals:
        unit = min(range(units), key=lambda index: free_at[index])
        start = max(arrival, free_at[unit])
        free_at[unit] = start + SERVICE_TICKS
        starts.append(start)
    return starts


def test_random_traces_start_as_the_g_d_k_fifo_closed_form_says_property():
    generator = random.Random(3)
    for _ in range(40):
        units = generator.choice([1, 2, 3])
        count = generator.randrange(2, 30)
        span = 20 * config.TICKS_PER_MICROSECOND
        arrivals = sorted(generator.randrange(0, span) for _ in range(count))
        assert _start_ticks(arrivals, units) == _g_d_k_closed_form(
            arrivals, units
        )


def test_poisson_arrivals_on_one_unit_wait_pollaczek_khinchine_property():
    generator = random.Random(5)
    utilization = 0.6
    job_count = 6000
    rate_per_tick = utilization / SERVICE_TICKS
    arrivals = []
    now = 0.0
    for _ in range(job_count):
        now += generator.expovariate(rate_per_tick)
        arrival = round(now)
        arrivals.append(arrival)
    starts = _start_ticks(arrivals, 1)
    waits = []
    for start, arrival in zip(starts, arrivals):
        wait = start - arrival
        waits.append(wait)
    settled = waits[job_count // 10 :]
    mean = statistics.fmean(settled)
    batch_size = len(settled) // 20
    batch_means = []
    for batch in range(20):
        first = batch * batch_size
        last = first + batch_size
        batch_mean = statistics.fmean(settled[first:last])
        batch_means.append(batch_mean)
    spread = statistics.stdev(batch_means)
    # t(19) at 99 percent over 20 batch means
    half_width = 2.86 * spread / 20**0.5
    predicted = utilization * SERVICE_TICKS / (2 * (1 - utilization))
    gap = mean - predicted
    difference = abs(gap)
    assert difference <= half_width


def test_a_job_waits_in_scheduler_order():
    # one unit: a takes it at 0; b (3 rounds) and c (2 rounds) wait, and
    # the shorter c is served first though b was admitted earlier
    engine = engine_module.Engine(verbose=False)
    scheduler = ShortestJobFirst()
    manager = _manager(engine, 1, scheduler)
    starts = {}
    original_begin = manager.service.begin

    def recording_begin(job, gated=True):
        starts[job.label] = engine.now
        original_begin(job, gated)

    manager.service.begin = recording_begin
    _submit(manager, "a", 1)
    _submit(manager, "b", 3)
    _submit(manager, "c", 2)
    engine.run()
    assert starts == {"a": 0, "c": SERVICE_TICKS, "b": 2 * SERVICE_TICKS}


def test_the_depth_is_sampled_at_every_change():
    engine = engine_module.Engine(verbose=False)
    manager = _manager(engine, 1)
    _submit(manager, "a")
    _submit(manager, "b")
    engine.run()
    # a queues and leaves at once; b queues behind a's compute and
    # leaves when a's decode ends
    depths = []
    for _tick, depth in manager.queue.depth_samples:
        depths.append(depth)
    assert depths == [1, 0, 1, 0]
