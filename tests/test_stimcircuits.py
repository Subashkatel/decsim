"""Smoke tests for qecsim.stimcircuits (vendored Oscar Higgott / Apache-2.0 generator).

These pin the first-class circuit-generation capability: it imports, returns stim.Circuit
objects for the code tasks we rely on (incl. toric, which stim's built-in generator lacks),
and behaves sanely with/without noise.
"""
import pytest

stim = pytest.importorskip("stim")
from qecsim.stimcircuits import generate_circuit


def test_toric_memory_x_generates():
    c = generate_circuit(
        "toric_code:unrotated_memory_x", distance=3, rounds=3,
        after_clifford_depolarization=0.001, before_round_data_depolarization=0.001,
        after_reset_flip_probability=0, before_measure_flip_probability=0.001)
    assert isinstance(c, stim.Circuit)
    assert c.num_observables == 1
    assert c.num_detectors > 0
    assert c.compile_detector_sampler().sample(5).shape[0] == 5


def test_rotated_surface_memory_generates():
    c = generate_circuit("surface_code:rotated_memory_z", distance=3, rounds=3,
                         after_clifford_depolarization=0.001)
    assert isinstance(c, stim.Circuit)
    assert c.num_observables == 1


def test_noiseless_circuit_has_no_detection_events():
    c = generate_circuit("surface_code:rotated_memory_x", distance=3, rounds=3)
    assert c.compile_detector_sampler().sample(16).sum() == 0
