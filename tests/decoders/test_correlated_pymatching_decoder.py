"""The correlated_pymatching row against PyMatching's own correlated decode.

PyMatching reads the whole circuit's decomposed detector error model
with enable_correlations=True and decodes each shot with the same flag
(PyMatching 2.4.0, Matching.decode). decsim's row writes its own model
from the window's linked fault models; on a whole-circuit window the
two must predict the same observable for every shot. With a fitted
timing, each decode holds its unit for DetectionEventTime's closed form
at the shot's own detection events, counted here from Stim's sample.
"""

import numpy
import pymatching
import pytest

import decsim.config as config
import decsim.decoders.decoder as decoder_module
import decsim.decoders.minimum_weight_perfect_matching.decoder as adapter
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models
from decsim.decoders.minimum_weight_perfect_matching import (
    detection_event_time,
)
from tests.decoders import windows

ROUNDS = 3
SHOTS = 300


def test_the_row_predicts_what_pymatchings_correlated_decode_does():
    circuit = windows.memory_circuit(3, ROUNDS, 0.01)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.LINKED_FAULT_MODELS_REQUIRED
    )
    detection_events, _ = windows.sampled_shots(circuit, SHOTS, 11)
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    referee = pymatching.Matching.from_detector_error_model(
        detector_error_model, enable_correlations=True
    )
    settings = adapter.CorrelatedPyMatchingDecoder.Settings()
    row = settings.build()
    expected_observables = []
    decoded_observables = []
    for shot in detection_events:
        syndrome = numpy.asarray(shot, dtype=numpy.uint8)
        expected = referee.decode(syndrome, enable_correlations=True)
        prediction = int(expected[0])
        expected_observables.append((prediction,))
        job = windows.job_for(model, shot)
        result = row.decode(job)
        decoded_observables.append(result.logical_observables)
    assert decoded_observables == expected_observables


def test_the_row_differs_from_plain_matching_on_some_shots():
    circuit = windows.memory_circuit(3, ROUNDS, 0.01)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.LINKED_FAULT_MODELS_REQUIRED
    )
    detection_events, _ = windows.sampled_shots(circuit, SHOTS, 11)
    plain_settings = adapter.PyMatchingDecoder.Settings()
    plain = plain_settings.build()
    correlated_settings = adapter.CorrelatedPyMatchingDecoder.Settings()
    correlated = correlated_settings.build()
    differing_shots = 0
    for shot in detection_events:
        job = windows.job_for(model, shot)
        plain_result = plain.decode(job)
        correlated_result = correlated.decode(job)
        plain_observables = plain_result.logical_observables
        if plain_observables != correlated_result.logical_observables:
            differing_shots += 1
    assert differing_shots > 0


def test_a_fitted_timing_prices_each_decode_at_its_own_events():
    circuit = windows.memory_circuit(3, ROUNDS, 0.01)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.LINKED_FAULT_MODELS_REQUIRED
    )
    detection_events, _ = windows.sampled_shots(circuit, SHOTS, 11)
    timing = _fitted_timing()
    settings = adapter.CorrelatedPyMatchingDecoder.Settings(timing=timing)
    row = settings.build()
    expected_ticks = []
    charged_ticks = []
    event_counts = set()
    for shot in detection_events:
        events = numpy.count_nonzero(shot)
        event_counts.add(events)
        microseconds = 1.5 * (3.0 + 0.5 * events**1.2)
        expected = config.microseconds_to_ticks(microseconds)
        expected_ticks.append(expected)
        job = windows.job_for(model, shot)
        _result, ticks = decoder_module.timed_decode(row, job, 0)
        charged_ticks.append(ticks)
    assert charged_ticks == expected_ticks
    assert len(event_counts) > 3


def test_a_fitted_timing_leaves_the_rows_answer_alone():
    circuit = windows.memory_circuit(3, ROUNDS, 0.01)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.LINKED_FAULT_MODELS_REQUIRED
    )
    detection_events, _ = windows.sampled_shots(circuit, SHOTS, 11)
    measured_settings = adapter.CorrelatedPyMatchingDecoder.Settings()
    measured = measured_settings.build()
    timing = _fitted_timing()
    timed_settings = adapter.CorrelatedPyMatchingDecoder.Settings(timing=timing)
    timed = timed_settings.build()
    measured_observables = []
    timed_observables = []
    for shot in detection_events:
        job = windows.job_for(model, shot)
        measured_result = measured.decode(job)
        measured_observables.append(measured_result.logical_observables)
        timed_result = timed.decode(job)
        timed_observables.append(timed_result.logical_observables)
    assert timed_observables == measured_observables


def test_a_job_with_no_window_holds_a_fitted_unit_for_nothing():
    timing = _fitted_timing()
    settings = adapter.CorrelatedPyMatchingDecoder.Settings(timing=timing)
    row = settings.build()
    job = decoding_records.DecodeJob(
        stream_id=1,
        window_id=0,
        round_count=1,
        detector_error_model=None,
        payloads=[],
        label="W0",
    )
    _result, ticks = decoder_module.timed_decode(row, job, 0)
    assert ticks == 0


def test_a_preset_latency_and_a_fitted_timing_together_are_refused():
    timing = _fitted_timing()
    with pytest.raises(ValueError, match="a preset latency or a fitted"):
        adapter.CorrelatedPyMatchingDecoder.Settings(
            preset_latency_microseconds=60.3, timing=timing
        )


def _fitted_timing() -> detection_event_time.DetectionEventTime:
    return detection_event_time.DetectionEventTime(
        intercept_microseconds=3.0,
        coefficient_microseconds=0.5,
        exponent=1.2,
        scale=1.5,
        source="a test's own numbers",
    )
