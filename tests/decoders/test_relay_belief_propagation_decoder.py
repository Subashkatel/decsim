"""The relay_bp row against relay-bp's RelayDecoderF32 on the same model.

The wheel is the bb-decoders extra; the identity test skips until it
is installed. The profile refusal below needs no wheel: it is decided while
the profile is built, from what relay.rs does with a first leg that
never runs.
"""

import pytest

import decsim.decoders.relay_belief_propagation.decoder as relay
import decsim.decoders.relay_belief_propagation.window_decoder as window_decoder
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


def test_the_first_relay_leg_must_run_at_least_once():
    """A zero first leg would hand back the previous window's answer.

    relay-bp's decode_inner runs its first leg for pre_iter iterations
    and leaves the previous call's decoding in place when that loop
    never runs (relay.rs in the relay-bp source), so a
    profile with no pre-iterations would report a stale correction for
    every window. The profile refuses it while it is built, and one
    iteration is enough, so the boundary is exclusive at zero.
    """
    with pytest.raises(ValueError, match="pre_iterations must be positive"):
        window_decoder.RelayBeliefPropagationWindowDecoder(pre_iterations=0)
    accepted = window_decoder.RelayBeliefPropagationWindowDecoder(
        pre_iterations=1
    )
    assert accepted.profile.pre_iterations == 1
