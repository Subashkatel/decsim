"""The box_correlated_pymatching row on the chip's packed box.

A box wide enough to hold every cluster must give correlated matching's
own answer on the whole window. Any box's joined correction must explain
the window's syndrome, since each cluster's correction explains its own
events. The row reads the box, never the rounds, and the unit is held
for the fitted law at the box's detection events, then the join. A box
with no event keeps union-find's correction. A job with no box, a row
that reads rounds given a box, and a box of another window's rows are
refused.
"""

import dataclasses

import numpy
import pytest

import decsim.config as config
import decsim.decoders.minimum_weight_perfect_matching.box_decoder as box_decoder  # noqa: E501
import decsim.decoders.minimum_weight_perfect_matching.decoder as matching
import decsim.decoders.minimum_weight_perfect_matching.detection_event_time as event_time  # noqa: E501
import decsim.escalation.doubtful_box as doubtful_box
import decsim.records.fault_model_contracts as fault_models
import tests.escalation.test_doubtful_box as box_tests

GRAPHLIKE = fault_models.FaultRepresentation.GRAPHLIKE
INTERCEPT = 5.0
COEFFICIENT = 0.5
EXPONENT = 1.2
JOIN_MICROSECONDS = 0.75
LAW = event_time.DetectionEventTime(
    intercept_microseconds=INTERCEPT,
    coefficient_microseconds=COEFFICIENT,
    exponent=EXPONENT,
    scale=1.0,
    source="test",
)
BOX_ROW = box_decoder.BoxCorrelatedPyMatchingDecoder


def _row(timing=LAW, join_microseconds=JOIN_MICROSECONDS):
    settings = BOX_ROW.Settings(
        timing=timing, join_microseconds=join_microseconds
    )
    return settings.build()


def _box_job(job, box):
    """The strong job on the box: no round, the box as its request."""
    return dataclasses.replace(job, payloads=[], request=box)


def _law_ticks(events: int) -> int:
    microseconds = INTERCEPT + COEFFICIENT * events**EXPONENT
    return config.microseconds_to_ticks(microseconds)


def test_a_box_that_holds_every_cluster_gives_correlated_matchings_answer():
    whole_settings = matching.CorrelatedPyMatchingDecoder.Settings()
    whole = whole_settings.build()
    row = _row()
    compared = 0
    for job, _result, box in box_tests.packed_windows(box_tests.WIDE_MARGIN):
        box_job = _box_job(job, box)
        box_result = row.decode(box_job)
        whole_result = whole.decode(job)
        assert (
            box_result.logical_observables == whole_result.logical_observables
        )
        compared += 1
    assert compared > box_tests.SHOTS // 4


def test_every_boxs_joined_correction_explains_the_syndrome():
    row = _row()
    for margin in (0, 1, 2):
        for job, result, box in box_tests.packed_windows(margin):
            model = job.detector_error_model
            faults = model.require_faults(GRAPHLIKE)
            dense_check = faults.check.toarray()
            check = dense_check.astype(int)
            backend = row.compiled_for(faults, model)
            box_job = _box_job(job, box)
            syndrome = row.job_syndrome(box_job, faults)
            answer = row.window_answer(
                box_job, backend, model, faults, syndrome
            )
            selected = numpy.asarray(answer.selected_faults, dtype=int)
            explained = check @ selected % 2
            window_syndrome = numpy.asarray(result.cluster_evidence.syndrome)
            numpy.testing.assert_array_equal(explained, window_syndrome)


def test_the_row_reads_the_box_and_not_the_rounds():
    row = _row()
    for job, _result, box in box_tests.packed_windows(1):
        faults = job.detector_error_model.require_faults(GRAPHLIKE)
        box_job = _box_job(job, box)
        syndrome = row.job_syndrome(box_job, faults)
        event_rows = numpy.flatnonzero(syndrome)
        assert event_rows.tolist() == box.event_rows.tolist()


def test_the_unit_is_held_for_the_law_at_the_boxs_events_then_the_join():
    row = _row()
    join_ticks = config.microseconds_to_ticks(JOIN_MICROSECONDS)
    smaller = 0
    for job, _result, box in box_tests.packed_windows(0):
        box_job = _box_job(job, box)
        result = row.decode(box_job)
        box_events = int(result.evidence["box_detection_events"])
        window_events = int(result.evidence["window_detection_events"])
        assert box_events == len(box.event_rows) <= window_events
        smaller += box_events < window_events
        expected = _law_ticks(box_events) + join_ticks
        assert row.ticks_after_decode(result, 0, 0) == expected
    assert smaller > 0


def test_a_box_without_events_keeps_union_finds_correction():
    row = _row()
    packed = box_tests.packed_windows(1)
    job, result, box = packed[0]
    weak = numpy.asarray(result.cluster_evidence.selected_faults, numpy.uint8)
    empty = dataclasses.replace(
        box, event_rows=box.event_rows[:0], outside=weak, cluster_count=0
    )
    model = job.detector_error_model
    faults = model.require_faults(GRAPHLIKE)
    backend = row.compiled_for(faults, model)
    empty_job = _box_job(job, empty)
    syndrome = row.job_syndrome(empty_job, faults)
    answer = row.window_answer(empty_job, backend, model, faults, syndrome)
    numpy.testing.assert_array_equal(answer.selected_faults, weak)
    assert answer.evidence["box_detection_events"] == 0
    decoded = row.decode(empty_job)
    assert row.ticks_after_decode(decoded, 0, 0) == _law_ticks(
        0
    ) + config.microseconds_to_ticks(JOIN_MICROSECONDS)


def test_a_job_without_a_box_is_refused():
    row = _row()
    packed = box_tests.packed_windows(1)
    job, _result, _box = packed[0]
    with pytest.raises(ValueError, match="carries none"):
        row.decode(job)


def test_a_row_that_reads_rounds_refuses_a_box_job():
    whole_settings = matching.CorrelatedPyMatchingDecoder.Settings()
    whole = whole_settings.build()
    packed = box_tests.packed_windows(1)
    job, _result, box = packed[0]
    box_job = _box_job(job, box)
    with pytest.raises(ValueError, match="reads a window's rounds"):
        whole.decode(box_job)


def test_a_box_of_another_windows_rows_is_refused():
    row = _row()
    packed = box_tests.packed_windows(1)
    job, _result, box = packed[0]
    other_count = box.detector_count + 1
    other = dataclasses.replace(box, detector_count=other_count)
    other_job = _box_job(job, other)
    with pytest.raises(ValueError, match="strong window has"):
        row.decode(other_job)


@pytest.mark.parametrize("join", [-1.0, float("nan"), "1"])
def test_a_join_time_is_a_finite_number_of_zero_or_more(join):
    with pytest.raises(ValueError, match="join_microseconds"):
        BOX_ROW.Settings(join_microseconds=join)


def test_the_packer_reads_union_find_growth_only():
    packed = box_tests.packed_windows(1)
    job, result, _box = packed[0]
    other = dataclasses.replace(result, cluster_evidence=None)
    shape = doubtful_box.BoxShape(1, 1)
    packer = doubtful_box.BoxPacker(shape)
    with pytest.raises(ValueError, match="union-find growth"):
        packer.pack(job, other)
