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
import decsim.decoders.weak_strong_switching as weak_strong_switching
import decsim.engine as engine_module
import decsim.message as message
from decsim.decoders.decoder_manager import DecoderManager


class FixedRow(decoder_module.DecoderBase):
    """A row for the plug-in law: two microseconds, one fixed observable."""

    def latency(self, job):
        del job
        return config.microseconds_to_ticks(2.0)

    def decode(self, job):
        return message.DecodeResult(
            job.op_id, job.window_id, logical_observables=(1,)
        )


def _manager(engine, row):
    router = decoders.CodeRouter(row)
    scheduler = schedulers.FifoScheduler()
    policy = weak_strong_switching.Baseline()
    return DecoderManager(
        engine,
        router=router,
        scheduler=scheduler,
        num_units=1,
        escalation_policy=policy,
        services=None,
    )


def _window_job():
    payload = message.RetainedSyndromeFragment(
        operation_id=1,
        patch_id="p",
        round_index=1,
        bits=(0, 1),
        size_bits=2,
        fragment_index=0,
    )
    request_key = message.DecoderRequestKey(1, 0, message.DecoderTier.WEAK, 0)
    return message.DecodeJob(
        op_id=1,
        window_id=0,
        n_rounds=1,
        payloads=[payload],
        label="mem W0",
        request_key=request_key,
    )


def test_a_fake_row_through_the_pool_decodes_the_window_once():
    engine = engine_module.Engine(verbose=False)
    row = FixedRow()
    manager = _manager(engine, row)
    delivered = []

    def on_decoded(_job, result):
        delivered.append((engine.now, result))

    job = _window_job()
    manager.enqueue(job, None, on_decoded)
    engine.run()
    (delivery,) = delivered
    tick, result = delivery
    assert tick == config.microseconds_to_ticks(2.0)
    assert result.logical_observables == (1,)
    assert "DecoderCluster: START DECODE mem W0" in engine.log_lines[-1]
    manager.check_decode_work_settled()


def test_a_spent_job_is_refused():
    engine = engine_module.Engine(verbose=False)
    row = FixedRow()
    manager = _manager(engine, row)
    job = _window_job()
    manager.enqueue(job, None, lambda _job, _result: None)
    with pytest.raises(RuntimeError, match="submitted once"):
        manager.enqueue(job, None, lambda _job, _result: None)


def test_a_withdrawn_window_leaves_the_queue_and_the_ledger():
    engine = engine_module.Engine(verbose=False)
    row = FixedRow()
    manager = _manager(engine, row)
    busy = _window_job()
    busy.label = "busy"
    manager.enqueue(busy, None, lambda _job, _result: None)
    waiting = _window_job()
    waiting.window_id = 1
    waiting.label = "waiting"
    waiting.request_key = message.DecoderRequestKey(
        1, 1, message.DecoderTier.WEAK, 1
    )
    manager.enqueue(waiting, None, lambda _job, _result: None)
    assert manager.queue.total() == 0  # both took a slot of the one unit
    manager.withdraw_window((1, 1))
    assert waiting.cancelled is True
    assert waiting.unit is None
    engine.run()
    manager.check_decode_work_settled()
