"""The relay_bp row against relay-bp's RelayDecoderF32 on the same model.

The wheel is not installed here (only its Rust source, at
tmp/reference-decoders/relay-bp); the identity test skips until it is
built. The profile refusals live in
tests/15_decoders/test_relay_belief_propagation_profile.py.
"""

import pytest

import decsim.decoders.relay_belief_propagation.decoder as relay
import decsim.detector_error_model.fault_model_contracts as fault_models
from tests.decoders import windows

ROUNDS = 3
PHYSICAL = fault_models.FaultRepresentation.PHYSICAL


def test_the_row_returns_the_backends_correction():
    pytest.importorskip("relay_bp")
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    )
    physical = model.require_faults(PHYSICAL)
    detection_events, _ = windows.sampled_shots(circuit, 10, 3)
    row = relay.RelayBeliefPropagationDecoder(
        gamma_table_seed=5, relay_set_count=20
    )
    compiled = row.window_decoder._compiled_model(physical)
    backend = compiled.backend
    for shot in detection_events:
        syndrome = windows.row_syndrome(model, shot)
        detailed = backend.decode_detailed(syndrome)
        expected = list(detailed.decoding)
        job = windows.job_for(model, shot)
        result = row.decode(job)
        assert result.correction.tolist() == expected
