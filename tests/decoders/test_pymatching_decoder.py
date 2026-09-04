"""The pymatching row against PyMatching itself and sinter's pymatching row.

One d=3 memory circuit as a single window: decsim's row, a
pymatching.Matching built directly from the window's graph, and sinter's
pymatching row on the whole-circuit detector error model
(.pydeps/sinter/_decoding/_decoding_pymatching.py) predict the same
observable per shot, or tie at the same weight.
"""

import numpy
import pymatching
import sinter

import decsim.decoders.mwpm.decoder as mwpm
import decsim.decoders.mwpm.weights as weights
import decsim.detector_error_model.fault_model_contracts as fault_models
from tests.decoders import windows

ROUNDS = 3
SHOTS = 100


def _window_and_shots():
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    )
    detection_events, observables = windows.sampled_shots(circuit, SHOTS, 5)
    return circuit, model, detection_events, observables


def test_the_row_matches_pymatching_on_the_same_graph():
    _, model, detection_events, _ = _window_and_shots()
    faults = model.require_faults(fault_models.FaultRepresentation.GRAPHLIKE)
    row = mwpm.PyMatchingDecoder()
    direct = pymatching.Matching.from_check_matrix(
        faults.check.copy(),
        weights=weights.matching_weights(faults.priors),
        error_probabilities=weights.finite_priors(faults.priors),
        merge_strategy="independent",
    )
    for shot in detection_events:
        job = windows.job_for(model, shot)
        result = row.decode(job)
        syndrome = numpy.asarray(shot)[list(model.detector_ids)]
        expected = direct.decode(syndrome.astype(numpy.uint8))
        assert result.correction.tolist() == expected.tolist()


def test_the_row_predicts_what_sinters_pymatching_row_predicts():
    circuit, model, detection_events, _ = _window_and_shots()
    dem = circuit.detector_error_model(decompose_errors=True)
    compiled = sinter.BUILT_IN_DECODERS["pymatching"].compile_decoder_for_dem(
        dem=dem
    )
    packed = numpy.packbits(detection_events, axis=1, bitorder="little")
    predictions = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed
    )
    predictions = numpy.unpackbits(predictions, axis=1, bitorder="little")
    whole = pymatching.Matching.from_detector_error_model(dem)
    row = mwpm.PyMatchingDecoder()
    faults = model.require_faults(fault_models.FaultRepresentation.GRAPHLIKE)
    matching = row.compiled_for(faults, model)
    ties = 0
    for shot, predicted in zip(detection_events, predictions):
        result = row.decode(windows.job_for(model, shot))
        if result.logical_observables == (int(predicted[0]),):
            continue
        # the two graphs may pick different minimum-weight matchings of
        # the same weight; that is a tie, not a disagreement
        syndrome = numpy.asarray(shot)[list(model.detector_ids)]
        _, row_weight = matching.decode(syndrome, return_weight=True)
        _, whole_weight = whole.decode(shot, return_weight=True)
        assert abs(row_weight - whole_weight) < 1e-6
        ties += 1
    assert ties < SHOTS / 10
