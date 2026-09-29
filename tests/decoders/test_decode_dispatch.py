"""The dispatcher's laws: what a blocked job may take, and from whom.

gem5 O3 issues from its ready set oldest-first and non-ready work never
displaces ready work (src/cpu/o3/inst_queue.hh, scheduleReadyInsts). A
startable job also leaves the compute a parked job waits for alone while
another unit is idle, so a window released by its neighbours decodes at
once (Skoric et al. 2209.08552 lines 416-421: a layer B window starts as
soon as both adjacent windows have completed). Two blocked jobs that read
the same rounds are staged on two units while two have room, so the unit
count alone decides whether they overlap; on one unit the second joins
the first one's transfer (the staging's landing rule, gem5's MSHR
answering every request for one block on a single fill,
src/mem/cache/mshr.hh, the TargetList targets).
"""

import functools

import decsim.config as config
import decsim.decoders.decoder_manager as decoder_manager
import decsim.decoders.decoders as decoders
import decsim.decoders.schedulers as schedulers
import decsim.decoders.strong_requests as strong_requests_module
import decsim.engine as engine_module
import decsim.escalation.policies as escalation_policies
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records

DECODE_MICROSECONDS = 4.0


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


class _ClosedUntilOpenedGate:
    """A gate the test opens by hand, as a boundary arriving does."""

    def __init__(self):
        self.is_open = False

    def may_stage(self, job):
        del job
        return True

    def may_start(self, job):
        del job
        return self.is_open

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
    policy = escalation_policies.Baseline(escalation_policies.NO_CONFIDENCE)
    manager = decoder_manager.DecoderManager(
        engine,
        scheduler=scheduler,
        unit_pools={"default": unit_count},
    )
    manager.strong_requests = strong_requests_module.StrongRequests()
    manager.decoder = decoder
    manager.escalation_policy = policy
    return manager


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


def test_the_placement_line_names_the_unit_the_job_was_given():
    """Two jobs at once on two units: the log says which unit took which."""
    engine = engine_module.Engine()
    manager = _manager(engine, unit_count=2)
    lines = []
    engine.line.connect(lines.append)
    first = _job(0, "first", deps_remaining=0)
    second = _job(1, "second", deps_remaining=0)

    manager.enqueue(first, None, _ignore)
    manager.enqueue(second, None, _ignore)
    engine.run()

    placements = _lines_containing(lines, "ASSIGN UNIT")
    assert "ASSIGN UNIT default#0 to first" in placements[0]
    assert "ASSIGN UNIT default#1 to second" in placements[1]


def _lines_containing(lines, needle):
    """The log lines that hold the text, to pick out one kind of line."""
    found = []
    for line in lines:
        if needle in line:
            found.append(line)
    return found


def _released_companions(unit_count):
    """(start ticks by label, transfers) of one window's two parked solves.

    The forced-class solves of one window: their own requests, one input
    identity and one boundary, which arrives after both inputs landed.
    """
    engine = engine_module.Engine()
    manager = _manager(engine, unit_count)
    transfer_ticks = config.microseconds_to_ticks(10.0)
    transfers = []
    starts = {}
    original_begin = manager.service.begin

    def recording_begin(job, gated=True):
        starts[job.label] = engine.now
        original_begin(job, gated)

    manager.service.begin = recording_begin

    def send_input(on_landed):
        transfers.append(on_landed)
        engine.schedule(transfer_ticks, on_landed, label="input")
        return transfer_ticks

    gate = _ClosedUntilOpenedGate()
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

    def boundary_arrives():
        gate.is_open = True
        first.window.deps_remaining = 0
        manager.release_parked(first.window.key)

    boundary_ticks = 3 * transfer_ticks
    engine.schedule(boundary_ticks, boundary_arrives, label="boundary")
    engine.run()
    return starts, transfers


def test_two_parked_companions_on_one_unit_share_a_transfer_and_queue():
    starts, transfers = _released_companions(unit_count=1)
    decode_ticks = config.microseconds_to_ticks(DECODE_MICROSECONDS)
    assert starts["companion"] - starts["first"] == decode_ticks
    assert len(transfers) == 1


def test_two_parked_companions_overlap_when_two_units_have_room():
    """The unit count alone decides whether the two solves overlap."""
    starts, transfers = _released_companions(unit_count=2)
    assert starts["companion"] == starts["first"]
    assert len(transfers) == 2
