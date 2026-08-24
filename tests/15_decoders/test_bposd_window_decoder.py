"""BP-OSD inner decoder: the OSD order is clamped to the window's n - m."""

import numpy as np
import stim

from decsim.decoders.bposd.window_decoder import bposd_window_decoder
from decsim.detector_error_model.fault_model_contracts import (
    FaultRepresentation,
    PHYSICAL_FAULT_MODEL_REQUIRED,
)
from decsim.detector_error_model.window_model_builders import (
    build_single_window_error_model,
)


def test_osd_order_above_window_rank_does_not_overrun():
    """ldpc osd_cs indexes candidates by osd_order with no bound (osd.hpp:90-99);
    an order above n - m overruns and segfaults. A swept osd_order must be safe
    on every window, including the small ones."""
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=3, rounds=3,
        after_clifford_depolarization=0.01, before_measure_flip_probability=0.01)
    model = build_single_window_error_model(
        circuit, (1, 3, 3), round_count=3,
        fault_model_requirement=PHYSICAL_FAULT_MODEL_REQUIRED)
    faults = model.require_faults(FaultRepresentation.PHYSICAL)
    decode = bposd_window_decoder(max_iter=5, osd_order=60)
    syndrome = np.zeros(faults.check.shape[0], dtype=np.uint8)
    syndrome[0] = 1
    correction = np.asarray(decode(model, syndrome), dtype=np.uint8)
    assert correction.shape == (faults.check.shape[1],)
    assert np.array_equal((faults.check @ correction) % 2, syndrome)
