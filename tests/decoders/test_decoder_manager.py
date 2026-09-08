"""The facade's laws: a row plugs in through the pool, a job is admitted once.

sinter's BUILT_IN_DECODERS
(sinter/_decoding/_decoding_all_built_in_decoders.py): a new decoder is
one class on the port and one row; here the row is routed by the pool
and its result reaches on_decoded once, with the manager's log naming
the job it started.
"""

import pytest

import decsim.config as config
import decsim.decoders.decoder as decoder_module
import decsim.decoders.decoders as decoders
import decsim.decoders.schedulers as schedulers
import decsim.decoders.staged_decoder as staged_decoder
import decsim.engine as engine_module
import decsim.escalation.policies as escalation_policies
import decsim.observe.log_writers as log_writers
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
from decsim.decoders.decoder_manager import DecoderManager


class FixedRow(decoder_module.DecoderBase):
    """A row for the plug-in law: two microseconds, one fixed observable."""

    def latency(self, job):
        del job
        return config.microseconds_to_ticks(2.0)

    def decode(self, job):
        return decoding_records.DecodeResult(
            job.operation_id, job.window_id, logical_observables=(1,)
        )


def _manager(engine, row):
    router = decoders.CodeRouter(row)
    scheduler = schedulers.FifoScheduler()
    policy = escalation_policies.Baseline()
    return DecoderManager(
        engine,
        router=router,
        scheduler=scheduler,
        num_units=1,
        escalation_policy=policy,
    )


def _window_job():
    payload = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_id="p",
        round_index=1,
        bits=(0, 1),
        size_bits=2,
        fragment_index=0,
    )
    request_key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.WEAK, 0
    )
    return decoding_records.DecodeJob(
        operation_id=1,
        window_id=0,
        round_count=1,
        payloads=[payload],
        label="mem W0",
        request_key=request_key,
    )


def test_a_fake_row_through_the_pool_decodes_the_window_once():
    engine = engine_module.Engine()
    log = log_writers.LogWriter()
    engine.line.connect(log.write)
    row = FixedRow()
    manager = _manager(engine, row)
    delivered = []

    def on_decoded(job, result):
        delivered.append((engine.now, result))
        manager.resolve_weak_request(job, result, decoding_records.Verdict.KEEP)

    job = _window_job()
    manager.enqueue(job, None, on_decoded)
    engine.run()
    (delivery,) = delivered
    tick, result = delivery
    assert tick == config.microseconds_to_ticks(2.0)
    assert result.logical_observables == (1,)
    assert "Decoder manager: START DECODE mem W0" in log.lines[-1]
    manager.check_decode_work_settled()


def _resolving(manager):
    """An on_decoded that closes the request, as the window side does."""
    keep = decoding_records.Verdict.KEEP

    def on_decoded(job, result):
        manager.resolve_weak_request(job, result, keep)

    return on_decoded


def test_a_spent_job_is_refused():
    engine = engine_module.Engine()
    row = FixedRow()
    manager = _manager(engine, row)
    job = _window_job()
    manager.enqueue(job, None, lambda _job, _result: None)
    with pytest.raises(RuntimeError, match="submitted once"):
        manager.enqueue(job, None, lambda _job, _result: None)


def test_a_withdrawn_window_leaves_the_queue_and_the_ledger():
    engine = engine_module.Engine()
    row = FixedRow()
    manager = _manager(engine, row)
    busy = _window_job()
    busy.label = "busy"
    on_decoded = _resolving(manager)
    manager.enqueue(busy, None, on_decoded)
    waiting = _window_job()
    waiting.window_id = 1
    waiting.label = "waiting"
    waiting.request_key = window_records.DecoderRequestKey(
        1, 1, window_records.DecoderTier.WEAK, 1
    )
    manager.enqueue(waiting, None, on_decoded)
    assert manager.queue.total() == 0  # both took a slot of the one unit
    manager.withdraw_window((1, 1))
    assert waiting.cancelled is True
    assert waiting.unit is None
    engine.run()
    manager.check_decode_work_settled()


def test_an_escalation_routed_to_a_pipelined_unit_is_refused():
    """The strong escalation tier is not pipelined yet, so it refuses.

    A pipelined route serves plain window and external decodes only; a
    strong re-decode, a gap sibling and a merged batch keep occupancy
    equal to latency until they get their own design pass, and routing
    one to a pipelined unit would silently serialize it instead of
    honoring the declared card (decode_service.py, _pipeline_of). The
    escalation request is the one a switching run submits: it names the
    destination window it re-decodes and asks for the strong pool.
    """
    engine = engine_module.Engine()
    timing = staged_decoder.UnitTiming((), (), 1.0, initiation_interval_us=1.0)
    algorithm = FixedRow()
    strong = staged_decoder.StagedDecoder(algorithm, timing)
    weak = FixedRow()
    router = decoders.SwitchingRouter(weak=weak, strong=strong)
    scheduler = schedulers.FifoScheduler()
    policy = escalation_policies.Baseline()
    manager = DecoderManager(
        engine,
        router=router,
        scheduler=scheduler,
        unit_pools={"default": 1, "strong": 1},
        escalation_policy=policy,
    )
    job = _window_job()
    job.hint = "strong"
    job.strong_decode_for = (1, 0)
    job.request_key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.STRONG, 0
    )
    with pytest.raises(RuntimeError, match="not pipelined yet"):
        manager.enqueue(job, None, lambda _job, _result: None)
