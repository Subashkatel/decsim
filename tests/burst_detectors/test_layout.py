"""Where an operation's checks sit, and the scale on their priors.

The scale solves the odd-number marginal of Tan et al. (2406.18897
lines 956-960) for the measured rate, on a Stim circuit whose Z checks
see no fault.
"""

import numpy
import pytest
import stim

import decsim.detector_error_model.fault_model_contracts as fault_contracts
import decsim.detector_error_model.window_slicer as window_slicer
import decsim.engine as engine_module
import decsim.frontends.settings as workload_settings
import decsim.records.windows as window_records
import tests.burst_detectors.burst_rounds as burst_rounds
from tests.burst_detectors.burst_rounds import (
    CODE_TASK,
    CUSUM,
    DISTANCE,
    GRAPHLIKE,
    PHYSICAL_ERROR_PROBABILITY,
)


def _dephasing_circuit(probability):
    """A d = 3, 30-round memory whose only noise is Z on the data.

    Z flips fire only the X checks, so every Z check is noiseless: a
    position with no fault behind it.
    """
    generated = stim.Circuit.generated(
        CODE_TASK, distance=3, rounds=30, before_round_data_depolarization=0.1
    )
    circuit = stim.Circuit()
    for instruction in generated.flattened():
        if instruction.name == "DEPOLARIZE1":
            targets = instruction.targets_copy()
            circuit.append("Z_ERROR", targets, probability)
            continue
        circuit.append(instruction)
    return circuit


def _bulk_window(circuit):
    """Rounds 2 to 30 of the 30-round circuit as one window, and its model."""
    window = window_records.Window(
        operation_id=1,
        window_index=0,
        commit_lo=2,
        commit_hi=30,
        buffer_hi=30,
        round_count=29,
    )
    slicer = window_slicer.WindowSlicer(
        circuit,
        round_count=30,
        fault_model_requirement=fault_contracts.GRAPHLIKE_FAULT_MODEL_REQUIRED,
    )
    model = slicer.slice_window(2, 2, 30, 30, is_last=True)
    return window, model


def test_burst_priors_over_noiseless_checks_scale_the_noisy_ones():
    """The whole patch under a 0.1 dephasing burst, half its checks silent.

    The silent checks count in the region's rate as never firing and set
    no saturation bound. The largest usual prior is 0.001998, two Z
    flips merged into one fault, and the scale that explains the
    measured rate is 94.26, so the largest raised prior is 0.18833.
    """
    usual = _dephasing_circuit(0.001)
    settings = CUSUM.Settings(
        calibration_shots=2000,
        region_radii=(),
        mask_count=None,
        raise_strong_priors=True,
    )
    engine = engine_module.Engine()
    detector = CUSUM(settings, engine, {1: (usual, 30)}, 1.0)
    burst = _dephasing_circuit(0.1)
    sampler = burst.compile_detector_sampler(seed=3)
    shots = sampler.sample(1)
    rounds = burst_rounds.rounds_of_events(usual, 30, shots[0])
    burst_rounds.feed(detector, rounds)
    window, model = _bulk_window(usual)

    raised = detector.with_burst_priors(window, model)
    faults = model.require_faults(GRAPHLIKE)
    raised_faults = raised.require_faults(GRAPHLIKE)
    changed_checks = faults.check != raised_faults.check

    assert changed_checks.nnz == 0
    assert numpy.max(raised_faults.priors) == pytest.approx(0.1883318804)


def test_an_operation_with_no_bulk_round_is_refused():
    circuit = workload_settings.memory_circuit(
        CODE_TASK, 1, DISTANCE, PHYSICAL_ERROR_PROBABILITY
    )
    engine = engine_module.Engine()
    settings = CUSUM.Settings()

    with pytest.raises(ValueError, match="has no bulk detector"):
        CUSUM(settings, engine, {1: (circuit, 1)}, 1.0)
