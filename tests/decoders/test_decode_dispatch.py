"""The dispatcher's laws: a job takes a unit only when it may start.

gem5 O3 picks a functional unit at issue, for ready instructions only
(src/cpu/o3/inst_queue.cc:920), and CUDA-Q's host dispatcher picks an
idle worker when data is present and copies the input to it then
(libs/qec/lib/realtime/host_side_dispatcher_design.md lines 36-42); a
job bound early waits for its one worker while another is idle (lines 22
and 118). So a blocked job holds no slot and no input copy, and a window
released by its neighbours decodes at once on any idle unit (Skoric et
al. 2209.08552 lines 416-421: a layer B window starts as soon as both
adjacent windows have completed). Two released jobs that read the same
rounds take two units while two are free, so the unit count alone
decides whether they overlap; on one unit the second joins the first
one's transfer (the staging's landing rule, gem5's MSHR answering every
request for one block on a single fill, src/mem/cache/mshr.hh, the
TargetList targets).
"""

import functools

import decsim.config as config
import decsim.decoders.decoder_manager as decoder_manager
import decsim.decoders.decoder_pool as decoder_pool
import decsim.decoders.decoders as decoders
import decsim.decoders.schedulers as schedulers
import decsim.decoders.strong_requests as strong_requests_module
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records

DECODE_MICROSECONDS = 4.0
TRANSFER_MICROSECONDS = 10.0


class _Gate:
    """A window gate whose boundary mask changes nothing."""

    def mask_input(self, job):
        del job


def _window(index, deps_remaining):
    return window_records.Window(
        operation_id=1,
        window_index=index,
        commit_lo=1,
        commit_hi=1,
        buffer_hi=1,
        round_count=1,
        deps_remaining=deps_remaining,
    )


def _job(index, label, deps_remaining, gate=None, input_key=None, window=None):
    payload = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=("p",),
        round_index=index,
        bits=(0, 1),
        size_bits=2,
        fragment_index=0,
    )
    if window is None:
        window = _window(index, deps_remaining)
    window_id = window.window_index
    request_key = window_records.DecoderRequestKey(
        1, window_id, window_records.DecoderTier.WEAK, index
    )
    return decoding_records.DecodeJob(
        operation_id=1,
        window_id=window_id,
        round_count=1,
        payloads=[payload],
        label=label,
        request_key=request_key,
        input_key=input_key,
        window=window,
        gate=gate,
    )


def _manager(engine, unit_count):
    """The manager these tests dispatch on, at that many units."""
    decoder = decoders.PresetLatencyDecoder(DECODE_MICROSECONDS)
    scheduler = schedulers.FifoScheduler()
    pool_settings = decoder_pool.PoolSettings(
        name="default", unit_count=unit_count
    )
    manager = decoder_manager.DecoderManager(
        engine,
        scheduler=scheduler,
        pool_settings=pool_settings,
    )
    manager.strong_requests = strong_requests_module.StrongRequests()
    manager.decoder = decoder
    return manager


def _ignore(_job, _result):
    pass


def _recording_starts(engine, manager):
    """The tick and unit each job's decode starts at, by label."""
    starts = {}
    original_begin = manager.service.begin

    def recording_begin(job):
        starts[job.label] = (engine.now, job.decoding_unit_name)
        original_begin(job)

    manager.service.begin = recording_begin
    return starts


def _release(manager, window):
    """The window's last boundary arrives."""
    window.deps_remaining = 0
    manager.release_window(window.key)


def test_a_blocked_job_waits_in_the_queue_while_startable_ones_pass():
    engine = engine_module.Engine()
    manager = _manager(engine, unit_count=1)
    dispatched = []
    original_dispatch_to = manager.service.dispatch_to

    def recording_dispatch_to(job, unit, claim_compute):
        dispatched.append(job.label)
        original_dispatch_to(job, unit, claim_compute)

    manager.service.dispatch_to = recording_dispatch_to
    # a computes and b and d each wait for a slot; the blocked c, admitted
    # before d, never takes one
    job_a = _job(0, "a", deps_remaining=0)
    manager.enqueue(job_a, None, _ignore)
    job_b = _job(1, "b", deps_remaining=0)
    manager.enqueue(job_b, None, _ignore)
    gate = _Gate()
    job_c = _job(2, "c", deps_remaining=1, gate=gate)
    manager.enqueue(job_c, None, _ignore)
    job_d = _job(3, "d", deps_remaining=0)
    manager.enqueue(job_d, None, _ignore)
    engine.run()
    assert dispatched == ["a", "b", "d"]
    assert manager.queue.waiting == [job_c]
    assert job_c.unit is None


def test_a_released_job_starts_at_once_on_the_idle_unit():
    """Its boundary arrives while one unit computes and the other idles.

    a decodes on one unit from 0 to 4 us and b on the other from 2 to
    6 us. The blocked job holds neither, so at its boundary, 5 us, it
    takes the unit a left and starts there at once; bound to the unit
    b took, it would wait until 6 us.
    """
    engine = engine_module.Engine()
    manager = _manager(engine, unit_count=2)
    starts = _recording_starts(engine, manager)
    gate = _Gate()
    blocked = _job(0, "blocked", deps_remaining=1, gate=gate)
    manager.enqueue(blocked, None, _ignore)
    job_a = _job(1, "a", deps_remaining=0)
    manager.enqueue(job_a, None, _ignore)
    job_b = _job(2, "b", deps_remaining=0)
    enqueue_b = functools.partial(manager.enqueue, job_b, None, _ignore)
    b_ticks = config.microseconds_to_ticks(2.0)
    engine.schedule(b_ticks, enqueue_b, label="enqueue b")
    release = functools.partial(_release, manager, blocked.window)
    boundary_ticks = config.microseconds_to_ticks(5.0)
    engine.schedule(boundary_ticks, release, label="boundary")
    engine.run()
    blocked_start, blocked_unit = starts["blocked"]
    _a_start, a_unit = starts["a"]
    _b_start, b_unit = starts["b"]
    assert blocked_start == boundary_ticks
    assert blocked_unit == a_unit
    assert blocked_unit != b_unit


def _sending_into(engine, transfers):
    """A send that records its start tick and lands after one transfer."""
    transfer_ticks = config.microseconds_to_ticks(TRANSFER_MICROSECONDS)

    def send_input(on_landed):
        transfers.append(engine.now)
        engine.schedule(transfer_ticks, on_landed, label="input")
        return transfer_ticks

    return send_input


def test_a_blocked_job_holds_no_slot_and_no_input_copy():
    """Before its boundary the job is in the queue, not on a unit."""
    engine = engine_module.Engine()
    manager = _manager(engine, unit_count=1)
    transfers = []
    send_input = _sending_into(engine, transfers)
    gate = _Gate()
    blocked = _job(0, "blocked", deps_remaining=1, gate=gate)
    manager.enqueue(blocked, send_input, _ignore)
    engine.run()
    unit = manager.pool.units[0]
    assert manager.queue.waiting == [blocked]
    assert unit.residents == []
    assert transfers == []


def test_a_blocked_jobs_input_copy_starts_at_its_boundary():
    """The copy does not overlap the wait: the decode starts one hop later.

    Its wait for a unit starts at the boundary too, so a queue wait
    holds no boundary wait.
    """
    engine = engine_module.Engine()
    manager = _manager(engine, unit_count=1)
    starts = _recording_starts(engine, manager)
    transfers = []
    send_input = _sending_into(engine, transfers)
    gate = _Gate()
    blocked = _job(0, "blocked", deps_remaining=1, gate=gate)
    manager.enqueue(blocked, send_input, _ignore)
    release = functools.partial(_release, manager, blocked.window)
    boundary_ticks = config.microseconds_to_ticks(30.0)
    engine.schedule(boundary_ticks, release, label="boundary")
    engine.run()
    transfer_ticks = config.microseconds_to_ticks(TRANSFER_MICROSECONDS)
    blocked_start, _unit = starts["blocked"]
    assert blocked.ready_time == boundary_ticks
    assert transfers == [boundary_ticks]
    assert blocked_start == boundary_ticks + transfer_ticks


def _released_companions(unit_count):
    """(start ticks by label, transfers) of one window's two blocked solves.

    The forced-class solves of one window: their own requests, one input
    identity and one boundary.
    """
    engine = engine_module.Engine()
    manager = _manager(engine, unit_count)
    starts = _recording_starts(engine, manager)
    transfers = []
    send_input = _sending_into(engine, transfers)
    gate = _Gate()
    first = _job(0, "first", deps_remaining=1, gate=gate)
    companion = _job(
        1,
        "companion",
        deps_remaining=1,
        gate=gate,
        input_key=first.request_key,
        window=first.window,
    )
    manager.enqueue(first, send_input, _ignore)
    manager.enqueue(companion, send_input, _ignore)
    release = functools.partial(_release, manager, first.window)
    boundary_ticks = config.microseconds_to_ticks(30.0)
    engine.schedule(boundary_ticks, release, label="boundary")
    engine.run()
    return starts, transfers


def test_two_released_companions_on_one_unit_share_a_transfer_and_queue():
    starts, transfers = _released_companions(unit_count=1)
    decode_ticks = config.microseconds_to_ticks(DECODE_MICROSECONDS)
    first_start, _first_unit = starts["first"]
    companion_start, _companion_unit = starts["companion"]
    assert companion_start - first_start == decode_ticks
    assert len(transfers) == 1


def test_two_released_companions_overlap_when_two_units_are_free():
    """The unit count alone decides whether the two solves overlap."""
    starts, transfers = _released_companions(unit_count=2)
    first_start, _first_unit = starts["first"]
    companion_start, _companion_unit = starts["companion"]
    assert companion_start == first_start
    assert len(transfers) == 2
