"""BP-OSD inner decoder: the OSD order is clamped to the window's n - m."""

import numpy as np
import stim

from decsim.decoders.belief_propagation_osd.decoder import (
    BeliefPropagationOsdDecoder,
)
from decsim.detector_error_model.fault_model_contracts import (
    PHYSICAL_FAULT_MODEL_REQUIRED,
    FaultRepresentation,
)
from decsim.detector_error_model.window_model_builders import (
    build_single_window_error_model,
)


def test_osd_order_above_window_rank_does_not_overrun():
    """An osd_order above the window's n - m is clamped, never overrun.

    ldpc's osd_cs indexes candidates by osd_order with no bound
    (osd.hpp:90-99); an order above n - m overruns and segfaults. A
    swept osd_order must be safe on every window, the small ones too.
    """
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
    )
    model = build_single_window_error_model(
        circuit,
        (1, 3, 3),
        round_count=3,
        fault_model_requirement=PHYSICAL_FAULT_MODEL_REQUIRED,
    )
    faults = model.require_faults(FaultRepresentation.PHYSICAL)
    decoder = BeliefPropagationOsdDecoder(max_iterations=5, osd_order=60)
    backend = decoder.compile(faults, model)
    syndrome = np.zeros(faults.check.shape[0], dtype=np.uint8)
    syndrome[0] = 1
    selected, _status = decoder.decode_window(backend, model, faults, syndrome)
    correction = np.asarray(selected, dtype=np.uint8)
    assert correction.shape == (faults.check.shape[1],)
    assert np.array_equal((faults.check @ correction) % 2, syndrome)
