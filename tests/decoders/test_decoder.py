"""The Decoder port's laws on a fake row, through decoder.py's defaults.

sinter's abstract class with defaults is the shape
(.pydeps/sinter/_decoding/_decoding_decoder_class.py).
"""

import numpy
import pytest
import scipy.sparse

import decsim.config as config
import decsim.decoders.decoder as decoder_module
import decsim.decoders.decoders as decoders
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records

MEASURED_NS = 2500


class FixedRow(decoder_module.DecoderBase):
    """A priced row: three ticks, one fixed observable."""

    def latency(self, job):
        del job
        return 3

    def decode(self, job):
        return decoding_records.DecodeResult(
            job.operation_id, job.window_id, logical_observables=(1,)
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
        return decoding_records.DecodeResult(job.operation_id, job.window_id)

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


def _job(**fields) -> decoding_records.DecodeJob:
    return decoding_records.DecodeJob(
        operation_id=1, window_id=0, round_count=2, label="W0", **fields
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
    with pytest.raises(NotImplementedError, match="a decoder measured"):
        row.latency(job)


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


def test_a_committed_fault_reaching_behind_the_window_is_a_crossing_commit():
    """The seam commit a region ending there and its own redo are pinned on.

    A window that owns the faults crossing behind its commit region
    carries that part of its correction on its own, because the residual
    XORs every committed fault's detectors together and the crossing
    part cannot be taken out of it again (Toshio et al. 2510.25222 lines
    1248-1250).
    """
    check = scipy.sparse.csc_matrix([[1, 0], [1, 1]], dtype=numpy.uint8)
    observables = scipy.sparse.csc_matrix([[1, 1]], dtype=numpy.uint8)
    placed = fault_models.PlacedFaultModel(
        representation=fault_models.FaultRepresentation.GRAPHLIKE,
        check=check,
        priors=[0.1, 0.2],
        observables=observables,
        owned=[True, True],
        source_fault_ids=[4, 9],
        boundary_flips={0: [0, 2], 1: [1]},
    )
    model = fault_models.WindowErrorModel(
        detector_ids=(0, 1),
        detector_coordinates=None,
        # detector 2 sits on round 5, the round before the commit region
        defect_positions={0: (6, 0), 1: (6, 1), 2: (5, 0)},
        first_commit_round=6,
        graphlike_faults=placed,
        physical_faults=None,
    )
    job = _job()
    result = decoder_module.result_from_selected_faults(
        job, model, placed, [1, 1]
    )
    assert result.logical_observables == (0,)
    assert result.boundary_data.detector_ids == (0, 1, 2)
    assert result.crossing_commit.residual.detector_ids == (0, 2)
    assert result.crossing_commit.logical_observables == (1,)
