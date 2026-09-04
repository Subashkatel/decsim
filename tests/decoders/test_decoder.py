"""The Decoder port's laws on a fake row, through decoder.py's defaults.

sinter's abstract class with defaults is the shape
(.pydeps/sinter/_decoding/_decoding_decoder_class.py).
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
    """A priced row: three ticks, one fixed observable."""

    def latency(self, job):
        del job
        return 3

    def decode(self, job):
        return message.DecodeResult(
            job.op_id, job.window_id, logical_observables=(1,)
        )


class MeasuredRow(decoder_module.DecoderBase):
    """A measured row: 2.5 microseconds on the host clock."""

    def latency(self, job):
        del job
        raise NotImplementedError

    def occupancy(self, job):
        del job
        return None

    def decode(self, job):
        return message.DecodeResult(job.op_id, job.window_id)

    def decode_timed(self, job):
        return self.decode(job), 2500


class EmptyWindowRow(decoder_module.WindowDecoderBase):
    """A window row that never gets a model in these tests."""

    def compile(self, faults, model):
        del faults
        del model
        raise AssertionError("no model, nothing to compile")

    def decode_window(self, backend, model, faults, syndrome):
        del backend
        del model
        del faults
        del syndrome
        raise AssertionError("no model, nothing to decode")


def _job(**fields) -> message.DecodeJob:
    return message.DecodeJob(
        op_id=1, window_id=0, n_rounds=2, label="W0", **fields
    )


def _started(row, job):
    engine = engine_module.Engine(verbose=False)
    delivered = []
    row.start(
        job, engine, lambda result: delivered.append((engine.now, result))
    )
    engine.run()
    return delivered


def test_start_delivers_the_result_after_latency_ticks():
    delivered = _started(FixedRow(), _job())
    assert len(delivered) == 1
    tick, result = delivered[0]
    assert tick == 3
    assert result.logical_observables == (1,)


def test_a_cancelled_job_delivers_none_after_its_time():
    job = _job()
    job.cancelled = True
    assert _started(FixedRow(), job) == [(3, None)]


def test_a_job_without_a_window_delivers_none():
    job = _job(on_done=lambda: None)
    assert _started(FixedRow(), job) == [(3, None)]


def test_a_measured_row_delivers_after_the_measured_ticks():
    delivered = _started(MeasuredRow(), _job())
    assert len(delivered) == 1
    tick, result = delivered[0]
    assert tick == config.microseconds_to_ticks(2.5)
    assert result.window_id == 0


def test_occupancy_is_the_latency_of_a_priced_row():
    assert FixedRow().occupancy(_job()) == 3


def test_occupancy_is_none_for_a_measured_row():
    assert MeasuredRow().occupancy(_job()) is None
    assert EmptyWindowRow(latency_model=None).occupancy(_job()) is None


def test_a_window_row_without_a_latency_model_has_no_latency():
    with pytest.raises(NotImplementedError, match="measured on the host"):
        EmptyWindowRow(latency_model=None).latency(_job())


def test_pipeline_depth_is_one():
    assert FixedRow().pipeline_depth(_job()) == 1


def test_cancel_on_a_plain_row_changes_nothing():
    row = FixedRow()
    job = _job()
    row.cancel(job)
    assert _started(row, job)[0][1].logical_observables == (1,)


def test_a_window_row_without_a_model_gives_the_empty_result():
    row = EmptyWindowRow(latency_model=decoders.PresetLatencyDecoder(1.0))
    result = row.decode(_job())
    assert result.correction is None
    assert result.logical_observables is None


def test_a_spent_job_is_refused_by_the_manager():
    engine = engine_module.Engine(verbose=False)
    manager = DecoderManager(
        engine,
        router=decoders.CodeRouter(FixedRow()),
        scheduler=schedulers.FifoScheduler(),
        num_units=1,
        escalation_policy=weak_strong_switching.Baseline(),
        services=None,
        on_window_decoded=lambda _job, _result: None,
        on_strong_window_decoded=None,
    )
    job = _job(
        request_key=message.DecoderRequestKey(1, 0, message.DecoderTier.WEAK, 0)
    )
    manager.enqueue(job)
    with pytest.raises(RuntimeError, match="submitted once"):
        manager.enqueue(job)
