"""The tesseract row against tesseract_decoder on the same model and seed.

decsim rebuilds a Stim detector error model out of the window's physical
faults (tesseract/window_decoder.py, detector_error_model_of) and
compiles the official backend with a detector-order seed; the same
backend built here from that model with the same configuration and
seed returns the same error indices.
"""

import numpy
import pytest

import decsim.decoders.tesseract.decoder as tesseract
import decsim.decoders.tesseract.window_decoder as tesseract_window
import decsim.detector_error_model.fault_model_contracts as fault_models
from tests.decoders import windows

ROUNDS = 3
SHOTS = 25
SEED = 11


def test_the_row_returns_the_backends_error_indices():
    backend = pytest.importorskip("tesseract_decoder")
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    )
    physical = model.require_faults(fault_models.FaultRepresentation.PHYSICAL)
    detection_events, _ = windows.sampled_shots(circuit, SHOTS, 3)
    configuration = tesseract_window.TesseractDecoderConfig(
        detector_order_seed=SEED, detector_order_count=4
    )
    row = tesseract.TesseractDecoder(configuration=configuration)
    dem, _coordinates = tesseract_window.detector_error_model_of(
        model, physical
    )
    orders = backend.utils.build_det_orders(
        dem, 4, backend.utils.DetOrder.DetIndex, SEED
    )
    direct = backend.tesseract.TesseractConfig(
        dem=dem,
        det_beam=15,
        beam_climbing=True,
        no_revisit_dets=True,
        verbose=False,
        merge_errors=False,
        pqlimit=200_000,
        det_orders=orders,
        det_penalty=0.0,
        create_visualization=False,
        sparsify_errors=False,
        sparsify_base_degree=-1,
        sparsify_max_degree=-1,
        sparsify_reactivate_limit=-1,
    ).compile_decoder()
    for shot in detection_events:
        syndrome = numpy.asarray(shot)[list(model.detector_ids)]
        expected = sorted(direct.decode_to_errors(syndrome.astype(bool)))
        result = row.decode(windows.job_for(model, shot))
        selected = sorted(
            int(index) for index in numpy.flatnonzero(result.correction)
        )
        assert selected == expected
