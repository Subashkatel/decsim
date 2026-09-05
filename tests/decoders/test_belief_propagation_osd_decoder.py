"""The bposd row against ldpc's BpOsdDecoder as qLDPC builds it.

qldpc.decoders.get_decoder_BP_OSD (retrieval.py:132) is
ldpc.BpOsdDecoder(pcm, error_channel=..., **args) on the same matrix;
the same syndromes give the same correction. The OSD order clamp is
stimbposd's (bp_osd.py:62-68): max(0, min(order, n - m)).
"""

import numpy
import qldpc.decoders

import decsim.decoders.belief_propagation_osd.decoder as adapter
import decsim.detector_error_model.fault_model_contracts as fault_models
from tests.decoders import windows

ROUNDS = 3
SHOTS = 25
OSD_ORDER = 2
HUGE_ORDER = 10_000
PHYSICAL = fault_models.FaultRepresentation.PHYSICAL


def _window_and_shots():
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    )
    detection_events, _ = windows.sampled_shots(circuit, SHOTS, 9)
    return model, detection_events


def _qldpc_referee(physical):
    check = physical.check.toarray()
    error_channel = list(physical.priors)
    return qldpc.decoders.get_decoder_BP_OSD(
        check,
        error_channel=error_channel,
        max_iter=5,
        bp_method="product_sum",
        schedule="serial",
        osd_method="osd_cs",
        osd_order=OSD_ORDER,
    )


def test_the_row_matches_qldpcs_bp_osd_on_the_same_matrix():
    model, detection_events = _window_and_shots()
    physical = model.require_faults(PHYSICAL)
    referee = _qldpc_referee(physical)
    row = adapter.BeliefPropagationOsdDecoder(
        max_iterations=5, osd_order=OSD_ORDER
    )
    for shot in detection_events:
        syndrome = windows.row_syndrome(model, shot)
        expected = referee.decode(syndrome)
        expected = numpy.asarray(expected, dtype=numpy.uint8)
        job = windows.job_for(model, shot)
        result = row.decode(job)
        assert result.correction.tolist() == expected.tolist()


def test_the_osd_order_clamp_is_stimbposds():
    model, _ = _window_and_shots()
    physical = model.require_faults(PHYSICAL)
    row_count, column_count = physical.check.shape
    window_rank = column_count - row_count
    stimbposd_order = max(0, min(HUGE_ORDER, window_rank))
    row = adapter.BeliefPropagationOsdDecoder(
        max_iterations=5, osd_order=HUGE_ORDER
    )
    backend = row.compile(physical, model)
    assert backend.osd_order == stimbposd_order
