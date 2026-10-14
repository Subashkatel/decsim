"""The correlated_pymatching row against PyMatching's own correlated decode.

PyMatching reads the whole circuit's decomposed detector error model
with enable_correlations=True and decodes each shot with the same flag
(PyMatching 2.4.0, Matching.decode). decsim's row writes its own model
from the window's linked fault models; on a whole-circuit window the
two must predict the same observable for every shot.
"""

import numpy
import pymatching

import decsim.decoders.minimum_weight_perfect_matching.decoder as adapter
import decsim.records.fault_model_contracts as fault_models
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
    row = adapter.CorrelatedPyMatchingDecoder.Settings().build()
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
    plain = adapter.PyMatchingDecoder.Settings().build()
    correlated = adapter.CorrelatedPyMatchingDecoder.Settings().build()
    differing_shots = 0
    for shot in detection_events:
        job = windows.job_for(model, shot)
        plain_result = plain.decode(job)
        correlated_result = correlated.decode(job)
        plain_observables = plain_result.logical_observables
        if plain_observables != correlated_result.logical_observables:
            differing_shots += 1
    assert differing_shots > 0
