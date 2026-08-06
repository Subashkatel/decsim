"""Smoke tests for decsim.stimcircuits (vendored Oscar Higgott / Apache-2.0 generator).

These pin the first-class circuit-generation capability: it imports, returns stim.Circuit
objects for the code tasks we rely on (incl. toric, which stim's built-in generator lacks),
and behaves sanely with/without noise.
"""
from fractions import Fraction
import math

import pytest

stim = pytest.importorskip("stim")
from decsim.stimcircuits import generate_circuit
from decsim.stimcircuits.noise import NoiseModel


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


@pytest.mark.parametrize("model_field,generator_field", [
    ("p_clifford", "after_clifford_depolarization"),
    ("p_data", "before_round_data_depolarization"),
    ("p_meas", "before_measure_flip_probability"),
    ("p_reset", "after_reset_flip_probability"),
])
def test_noise_boundaries_cover_models_and_public_generator(model_field, generator_field):
    for value in (0, 1, -0.0, 0.25, Fraction(1, 2)):
        model = NoiseModel(**{model_field: value})
        assert type(getattr(model, model_field)) is float
        generate_circuit(
            "surface_code:rotated_memory_z", distance=2, rounds=2,
            **{generator_field: value},
        )

    for value in (-0.1, float("nan"), float("inf"), -float("inf"), 1.1):
        with pytest.raises(ValueError, match=model_field):
            NoiseModel(**{model_field: value})
        with pytest.raises(ValueError, match=generator_field):
            generate_circuit(
                "surface_code:rotated_memory_z", distance=2, rounds=2,
                **{generator_field: value},
            )

def test_phenomenological_noise_applies_the_physical_transform():
    boundary = 2 / 3
    model = NoiseModel.phenomenological(boundary)
    assert model.p_data == 1.5 * boundary
    assert model.p_meas == boundary

    with pytest.raises(ValueError, match="p_data"):
        NoiseModel.phenomenological(0.8)


def test_surface_generator_keeps_distance_minimum_local():
    with pytest.raises(ValueError, match="distance >= 2"):
        generate_circuit(
            "surface_code:rotated_memory_z",
            distance=1,
            rounds=2,
        )
    assert isinstance(
        generate_circuit(
            "surface_code:rotated_memory_z",
            distance=2,
            rounds=2,
        ),
        stim.Circuit,
    )


def test_repetition_memory_matches_official_stim():
    noise = {
        "after_clifford_depolarization": 0.001,
        "before_round_data_depolarization": 0.002,
        "before_measure_flip_probability": 0.003,
        "after_reset_flip_probability": 0.004,
    }
    actual = generate_circuit("repetition_code:memory", distance=3, rounds=3, **noise)
    expected = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=3, **noise)
    assert actual == expected
    assert (actual.num_detectors, actual.num_observables) == (8, 1)
    assert actual.get_detector_coordinates() == expected.get_detector_coordinates()
    assert actual.detector_error_model(decompose_errors=True).flattened() == (
        expected.detector_error_model(decompose_errors=True).flattened()
    )


@pytest.mark.parametrize("geometry", [
    {"x_distance": 3}, {"z_distance": 3},
    {"x_distance": 3, "z_distance": 3},
    {"distance": 3, "x_distance": 3, "z_distance": 3},
])
def test_repetition_memory_rejects_rectangular_geometry(geometry):
    with pytest.raises(ValueError):
        generate_circuit("repetition_code:memory", rounds=3, **geometry)


def test_repetition_memory_rejects_surface_detector_filter():
    with pytest.raises(ValueError):
        generate_circuit(
            "repetition_code:memory", distance=3, rounds=3,
            exclude_other_basis_detectors=True,
        )


@pytest.mark.parametrize("distance,rounds", [(1, 3), (3, 0)])
def test_repetition_memory_uses_stim_distance_and_round_bounds(distance, rounds):
    with pytest.raises(ValueError):
        generate_circuit(
            "repetition_code:memory", distance=distance, rounds=rounds
        )


def test_repetition_memory_samples_matching_detector_and_observable_shots():
    circuit = generate_circuit(
        "repetition_code:memory", distance=3, rounds=3,
        after_reset_flip_probability=1,
    )
    detectors, observables = circuit.compile_detector_sampler().sample(
        4, separate_observables=True
    )
    assert detectors.tolist() == [[1, 1, 0, 0, 0, 0, 1, 1]] * 4
    assert observables.tolist() == [[1]] * 4
