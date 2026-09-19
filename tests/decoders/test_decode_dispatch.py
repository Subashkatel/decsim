"""The dispatcher's laws: what a blocked job may take, and from whom.

gem5 O3 issues from its ready set oldest-first and non-ready work never
displaces ready work (src/cpu/o3/inst_queue.hh, scheduleReadyInsts). A
startable job also leaves the compute a parked job waits for alone while
another unit is idle, so a window released by its neighbours decodes at
once (Skoric et al. 2209.08552 lines 416-421: a layer B window starts as
soon as both adjacent windows have completed), and a job whose rounds a
unit is already receiving goes to that unit and joins the one transfer
(the staging's landing rule, gem5's MSHR answering every request for one
block on a single fill, src/mem/cache/mshr.hh, the TargetList targets).
"""

import functools

import decsim.config as config
import decsim.decoders.decoders as decoders
import decsim.decoders.schedulers as schedulers
import decsim.decoders.strong_requests as strong_requests_module
import decsim.engine as engine_module
import decsim.escalation.policies as escalation_policies
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
from decsim.decoders.decoder_manager import DecoderManager


class _OpenGate:
    """A gate that lets a blocked job take a slot and start when it can."""

    def may_stage(self, job):
        del job
        return True

    def may_start(self, job):
        del job
        return True

    def mask_input(self, job):
        del job


class _BoundaryPendingGate:
    """A gate that lets a blocked job take a slot but not start."""

    def may_stage(self, job):
        del job
        return True

    def may_start(self, job):
        del job
        return False

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
        patch_id="p",
        round_index=index,
        bits=(0, 1),
        size_bits=2,
        fragment_index=0,
    )
    request_key = window_records.DecoderRequestKey(
        1, index, window_records.DecoderTier.WEAK, index
    )
    if window is None:
        window = _window(index, deps_remaining)
    return decoding_records.DecodeJob(
        operation_id=1,
        window_id=index,
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
    decoder = decoders.PresetLatencyDecoder(4.0)
    router = decoders.CodeRouter(decoder)
    scheduler = schedulers.FifoScheduler()
    policy = escalation_policies.Baseline(escalation_policies.NO_CONFIDENCE)
    strong_requests = strong_requests_module.StrongRequests()
    return DecoderManager(
        engine,
        router=router,
        scheduler=scheduler,
        strong_requests=strong_requests,
        num_units=unit_count,
        escalation_policy=policy,
    )


def _ignore(_job, _result):
    pass


def test_a_blocked_job_never_displaces_a_startable_one():
    engine = engine_module.Engine()
    manager = _manager(engine, unit_count=1)
    dispatched = []
    original_dispatch_to = manager.service.dispatch_to

    def recording_dispatch_to(pool, job, unit, claim_compute):
        dispatched.append(job.label)
        original_dispatch_to(pool, job, unit, claim_compute)

    manager.service.dispatch_to = recording_dispatch_to
    # a computes and b lands in the second slot: the unit is full, so
    # the blocked c and the startable d wait, c admitted first
    job_a = _job(0, "a", deps_remaining=0)
    manager.enqueue(job_a, None, _ignore)
    job_b = _job(1, "b", deps_remaining=0)
    manager.enqueue(job_b, None, _ignore)
    gate = _OpenGate()
    job_c = _job(2, "c", deps_remaining=1, gate=gate)
    manager.enqueue(job_c, None, _ignore)
    job_d = _job(3, "d", deps_remaining=0)
    manager.enqueue(job_d, None, _ignore)
    engine.run()
    assert dispatched == ["a", "b", "d", "c"]


def test_a_startable_job_takes_an_empty_unit_over_one_holding_a_parked_job():
    engine = engine_module.Engine()
    manager = _manager(engine, unit_count=2)
    # the running decode holds unit 0, so the blocked job parks on the
    # other unit and leaves unit 0 empty for the later job
    running = _job(0, "running", deps_remaining=0)
    manager.enqueue(running, None, _ignore)
    gate = _BoundaryPendingGate()
    parked = _job(1, "parked", deps_remaining=1, gate=gate)
    manager.enqueue(parked, None, _ignore)
    later = _job(2, "later", deps_remaining=0)
    enqueue_later = functools.partial(manager.enqueue, later, None, _ignore)
    delay = config.microseconds_to_ticks(10.0)
    engine.schedule(delay, enqueue_later, label="enqueue later")
    engine.run()
    parked_unit_name = parked.decoding_unit_name
    later_unit_name = later.decoding_unit_name
    assert parked_unit_name is not None
    assert later_unit_name is not None
    assert later_unit_name != parked_unit_name


def test_two_blocked_companions_of_one_window_share_one_transfer():
    engine = engine_module.Engine()
    manager = _manager(engine, unit_count=2)
    transfer_ticks = config.microseconds_to_ticks(10.0)
    landings = []

    def send_input(on_landed):
        landings.append(on_landed)
        engine.schedule(transfer_ticks, on_landed, label="input")
        return transfer_ticks

    # the forced-class solves of one window: their own requests, one
    # input identity, so the second reads the rounds the first brings
    gate = _BoundaryPendingGate()
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
    engine.run()
    first_unit_name = first.decoding_unit_name
    companion_unit_name = companion.decoding_unit_name
    assert first_unit_name is not None
    assert companion_unit_name == first_unit_name
    assert len(landings) == 1
