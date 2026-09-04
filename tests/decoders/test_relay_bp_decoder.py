"""The relay_bp row against relay-bp's RelayDecoderF32 on the same model.

The wheel is not installed here (only its Rust source, at
tmp/reference-decoders/relay-bp); the identity test skips until it is
built. The profile refusals live in tests/15_decoders/test_relay_bp_profile.py.
"""

import numpy
import pytest

import decsim.decoders.relay_bp.decoder as relay_bp
import decsim.detector_error_model.fault_model_contracts as fault_models
from tests.decoders import windows

ROUNDS = 3


def test_the_row_returns_the_backends_correction():
    backend = pytest.importorskip("relay_bp")
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    )
    physical = model.require_faults(fault_models.FaultRepresentation.PHYSICAL)
    detection_events, _ = windows.sampled_shots(circuit, 10, 3)
    row = relay_bp.RelayBpDecoder(gamma_table_seed=5, relay_set_count=20)
    faults_backend = row.window_decoder._compiled_model(physical).backend
    del backend
    for shot in detection_events:
        syndrome = numpy.asarray(shot)[list(model.detector_ids)]
        expected = faults_backend.decode_detailed(syndrome.astype(numpy.uint8))
        result = row.decode(windows.job_for(model, shot))
        assert result.correction.tolist() == list(expected.decoding)
