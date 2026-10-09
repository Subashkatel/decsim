"""A shot drawn from a circuit's detector error model, with its errors.

Source: Stim's CompiledDemSampler.sample(return_errors=True), whose
error columns follow the flattened model's error instructions; the
fired errors' detectors and observables XOR to the drawn events and
flips, as every mechanism flips its own targets.
"""

import numpy
import stim

import decsim.detector_error_model.error_model_sampler as error_model_sampler


def memory_circuit(rounds, noise):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=rounds,
        after_clifford_depolarization=noise,
        before_round_data_depolarization=noise,
        before_measure_flip_probability=noise,
        after_reset_flip_probability=noise,
    )


def test_the_fired_errors_flip_exactly_the_drawn_events_and_flips():
    circuit = memory_circuit(4, 0.03)
    sampler = error_model_sampler.ErrorModelSampler(circuit)
    events, flips, fired = sampler.draw(11)
    detectors = numpy.zeros(circuit.num_detectors, dtype=bool)
    observables = numpy.zeros(circuit.num_observables, dtype=bool)
    for error_index in fired:
        for detector in sampler.detectors_by_error[error_index]:
            detectors[detector] ^= True
        for observable in sampler.observables_by_error[error_index]:
            observables[observable] ^= True
    assert numpy.array_equal(detectors, events)
    assert numpy.array_equal(observables, flips)
    assert len(fired) > 0


def test_a_fired_error_belongs_to_the_round_of_its_earliest_detector():
    circuit = stim.Circuit(
        "R 0\nX_ERROR(0.5) 0\nM 0\nDETECTOR(0, 0, 0) rec[-1]\n"
        "M 0\nDETECTOR(0, 0, 1) rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    sampler = error_model_sampler.ErrorModelSampler(circuit)
    first_error = numpy.array([0])
    fired = sampler.fired_errors(first_error, {0: 3, 1: 4})
    assert fired[0].first_round == 3
    assert fired[0].logical_observables == (1,)


def test_a_circuit_is_read_once_while_it_lives():
    circuit = memory_circuit(2, 0.01)
    first = error_model_sampler.sampler_of(circuit)
    second = error_model_sampler.sampler_of(circuit)
    assert first is second
