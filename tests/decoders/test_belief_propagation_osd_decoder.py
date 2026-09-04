"""The bposd row against ldpc's BpOsdDecoder as qLDPC builds it.

qldpc.decoders.get_decoder_BP_OSD (retrieval.py:132) is
ldpc.BpOsdDecoder(pcm, error_channel=..., **args) on the same matrix;
the same syndromes give the same correction. The OSD order clamp is
stimbposd's (bp_osd.py:62-68): max(0, min(order, n - m)).
"""

import numpy
import qldpc.decoders

import decsim.decoders.belief_propagation_osd.decoder as belief_propagation_osd
import decsim.detector_error_model.fault_model_contracts as fault_models
from tests.decoders import windows

ROUNDS = 3
SHOTS = 25


def _window_and_shots():
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    )
    detection_events, _ = windows.sampled_shots(circuit, SHOTS, 9)
    return model, detection_events


def test_the_row_matches_qldpcs_bp_osd_on_the_same_matrix():
    model, detection_events = _window_and_shots()
    physical = model.require_faults(fault_models.FaultRepresentation.PHYSICAL)
    referee = qldpc.decoders.get_decoder_BP_OSD(
        physical.check.toarray(),
        error_channel=list(physical.priors),
        max_iter=5,
        bp_method="product_sum",
        schedule="serial",
        osd_method="osd_cs",
        osd_order=2,
    )
    row = belief_propagation_osd.BeliefPropagationOsdDecoder(
        max_iterations=5, osd_order=2
    )
    for shot in detection_events:
        syndrome = numpy.asarray(shot)[list(model.detector_ids)]
        expected = numpy.asarray(referee.decode(syndrome.astype(numpy.uint8)))
        result = row.decode(windows.job_for(model, shot))
        assert (
            result.correction.tolist() == expected.astype(numpy.uint8).tolist()
        )


def test_the_osd_order_clamp_is_stimbposds():
    model, _ = _window_and_shots()
    physical = model.require_faults(fault_models.FaultRepresentation.PHYSICAL)
    row_count, column_count = physical.check.shape
    row = belief_propagation_osd.BeliefPropagationOsdDecoder(
        max_iterations=5, osd_order=10_000
    )
    backend = row.compile(physical, model)
    assert backend.osd_order == max(0, min(10_000, column_count - row_count))
