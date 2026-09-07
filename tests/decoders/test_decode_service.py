"""The service's laws: the start, the park, the overlap, the pipeline.

Smith 1982 decoupled access-execute (rowD2, compare_overlap_laws.py):
with two slots the next window's transfer overlaps the compute, one
window per max(T, C). Tomasulo's rule at the boundary hazard: a landed
job whose window owes a boundary keeps its slot and never the compute.
Hennessy and Patterson App. C (rowD4, compare_pipeline_law.py): a
pipelined unit starts one decode per initiation interval, at most depth
in flight, each result a fixed latency after its start. The laws are
computed inside the tests.
"""

import decsim.config as config
import decsim.decoders.decoders as decoders
import decsim.decoders.schedulers as schedulers
import decsim.decoders.staged_decoder as staged_decoder
import decsim.engine as engine_module
import decsim.escalation.policies as escalation_policies
import decsim.message as message
import decsim.records.rounds as round_records
from decsim.decoders.decoder_manager import DecoderManager


class _Gate:
    """A gate the test opens by hand."""

    def __init__(self):
        self.is_open = True
        self.masked = []

    def may_stage(self, job):
        del job
        return True

    def may_start(self, job):
        del job
        return self.is_open

    def mask_input(self, job):
        self.masked.append(job.label)


def _job(index, gate=None, deps_remaining=0):
    payload = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_id="p",
        round_index=index,
        bits=(0, 1),
        size_bits=2,
        fragment_index=0,
    )
    request_key = message.DecoderRequestKey(
        1, index, message.DecoderTier.WEAK, index
    )
    window = message.Window(
        op_id=1,
        k=index,
        commit_lo=1,
        commit_hi=1,
        buffer_hi=1,
        n_rounds=1,
        deps_remaining=deps_remaining,
    )
    return message.DecodeJob(
        op_id=1,
        window_id=index,
        n_rounds=1,
        payloads=[payload],
        label=f"w{index}",
        request_key=request_key,
        window=window,
        gate=gate,
    )


def _send_after(engine, ticks):
    def send(on_landed):
        engine.schedule(ticks, on_landed)
        return ticks

    return send


def _manager(engine, decoder):
    router = decoders.CodeRouter(decoder)
    scheduler = schedulers.FifoScheduler()
    policy = escalation_policies.Baseline()
    return DecoderManager(
        engine,
        router=router,
        scheduler=scheduler,
        num_units=1,
        escalation_policy=policy,
    )


def _ignore(_job, _result):
    pass


def _ending_in(ends, engine):
    """An on_decoded that keeps the tick each job's result arrived."""

    def on_decoded(job, _result):
        ends[job.label] = engine.now

    return on_decoded


def _recording(manager, engine):
    """Compute start ticks by label, through service.begin."""
    starts = {}
    original_begin = manager.service.begin

    def recording_begin(job, gated=True):
        starts[job.label] = engine.now
        original_begin(job, gated)

    manager.service.begin = recording_begin
    return starts


def test_a_job_starts_when_its_input_landed_and_the_gate_allows():
    engine = engine_module.Engine()
    decoder = decoders.PresetLatencyDecoder(1.0)
    manager = _manager(engine, decoder)
    starts = _recording(manager, engine)
    done = {}
    gate = _Gate()
    transfer_ticks = config.microseconds_to_ticks(2.0)
    job = _job(0, gate)
    send_input = _send_after(engine, transfer_ticks)
    on_decoded = _ending_in(done, engine)
    manager.enqueue(job, send_input, on_decoded)
    engine.run()
    assert starts["w0"] == transfer_ticks
    assert gate.masked == ["w0"]
    assert done["w0"] == transfer_ticks + config.microseconds_to_ticks(1.0)


def test_a_parked_job_keeps_its_slot_and_releases_the_compute():
    engine = engine_module.Engine()
    decoder = decoders.PresetLatencyDecoder(1.0)
    manager = _manager(engine, decoder)
    starts = _recording(manager, engine)
    gate = _Gate()
    gate.is_open = False
    job = _job(0, gate)
    manager.enqueue(job, None, _ignore)
    (unit,) = manager.pool.units_by_pool["default"]
    assert job.is_parked is True
    assert unit.residents == [job]
    assert unit.holder is None
    assert manager.pool.free_count("default") == 1
    gate.is_open = True
    release_ticks = config.microseconds_to_ticks(3.0)
    window_key = (1, 0)
    engine.schedule(release_ticks, lambda: manager.release_parked(window_key))
    engine.run()
    assert starts["w0"] == release_ticks
    manager.check_decode_work_settled()


def test_the_second_slots_transfer_overlaps_the_compute():
    # T = 1 us, C = 2 us, three windows ready at once (rowD2's two-slot
    # law): w0 lands at 1 and computes 1..3; w1 landed at 1 and waits for
    # the compute until 3; w2 takes w0's slot at 3, lands at 4, starts 5
    engine = engine_module.Engine()
    decoder = decoders.PresetLatencyDecoder(2.0)
    manager = _manager(engine, decoder)
    starts = _recording(manager, engine)
    transfer_ticks = config.microseconds_to_ticks(1.0)
    for index in range(3):
        job = _job(index)
        send_input = _send_after(engine, transfer_ticks)
        manager.enqueue(job, send_input, _ignore)
    engine.run()
    assert starts == {
        "w0": config.microseconds_to_ticks(1.0),
        "w1": config.microseconds_to_ticks(3.0),
        "w2": config.microseconds_to_ticks(5.0),
    }


def test_a_pipelined_unit_issues_at_its_initiation_interval():
    # II = 0.5 us, C = 4 us, full depth: three windows landing at once
    # start 0.5 us apart and each returns 4 us after its start
    engine = engine_module.Engine()
    timing = staged_decoder.UnitTiming((), (), 1.0, initiation_interval_us=0.5)
    algorithm = decoders.PresetLatencyDecoder(4.0)
    decoder = staged_decoder.StagedDecoder(algorithm, timing)
    manager = _manager(engine, decoder)
    starts = _recording(manager, engine)
    ends = {}
    on_decoded = _ending_in(ends, engine)
    for index in range(3):
        job = _job(index)
        manager.enqueue(job, None, on_decoded)
    engine.run()
    interval = config.microseconds_to_ticks(0.5)
    latency = config.microseconds_to_ticks(4.0)
    assert starts == {"w0": 0, "w1": interval, "w2": 2 * interval}
    assert ends == {
        "w0": latency,
        "w1": interval + latency,
        "w2": 2 * interval + latency,
    }
    manager.check_decode_work_settled()
