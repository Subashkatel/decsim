"""The Decoder port's laws on a fake row, through decoder.py's defaults.

sinter's abstract class with defaults is the shape
(.pydeps/sinter/_decoding/_decoding_decoder_class.py).
"""

import pytest

import decsim.config as config
import decsim.decoders.decoder as decoder_module
import decsim.decoders.decoders as decoders
import decsim.engine as engine_module
import decsim.message as message

MEASURED_NS = 2500


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
        result = self.decode(job)
        return result, MEASURED_NS


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
    engine = engine_module.Engine()
    delivered = []

    def on_result(result):
        delivered.append((engine.now, result))

    row.start(job, engine, on_result)
    engine.run()
    return delivered


def test_start_delivers_the_result_after_latency_ticks():
    job = _job()
    row = FixedRow()
    delivered = _started(row, job)
    assert len(delivered) == 1
    tick, result = delivered[0]
    assert tick == 3
    assert result.logical_observables == (1,)


def test_a_cancelled_job_delivers_none_after_its_time():
    job = _job()
    job.cancelled = True
    row = FixedRow()
    delivered = _started(row, job)
    assert delivered == [(3, None)]


def test_a_job_without_a_window_delivers_none():
    job = _job(on_done=lambda: None)
    row = FixedRow()
    delivered = _started(row, job)
    assert delivered == [(3, None)]


def test_a_measured_row_delivers_after_the_measured_ticks():
    job = _job()
    row = MeasuredRow()
    delivered = _started(row, job)
    assert len(delivered) == 1
    tick, result = delivered[0]
    assert tick == config.microseconds_to_ticks(2.5)
    assert result.window_id == 0


def test_occupancy_is_the_latency_of_a_priced_row():
    job = _job()
    row = FixedRow()
    assert row.occupancy(job) == 3


def test_occupancy_is_none_for_a_measured_row():
    job = _job()
    measured = MeasuredRow()
    assert measured.occupancy(job) is None
    window_row = EmptyWindowRow(latency_model=None)
    assert window_row.occupancy(job) is None


def test_a_window_row_without_a_latency_model_has_no_latency():
    job = _job()
    row = EmptyWindowRow(latency_model=None)
    with pytest.raises(NotImplementedError, match="measured on the host"):
        row.latency(job)


def test_pipeline_depth_is_one():
    job = _job()
    row = FixedRow()
    assert row.pipeline_depth(job) == 1


def test_cancel_on_a_plain_row_changes_nothing():
    row = FixedRow()
    job = _job()
    row.cancel(job)
    delivered = _started(row, job)
    _tick, result = delivered[0]
    assert result.logical_observables == (1,)


def test_a_window_row_without_a_model_gives_the_empty_result():
    latency_model = decoders.PresetLatencyDecoder(1.0)
    row = EmptyWindowRow(latency_model=latency_model)
    job = _job()
    result = row.decode(job)
    assert result.correction is None
    assert result.logical_observables is None
