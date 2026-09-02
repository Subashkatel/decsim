"""Detector rounds follow Stim's detector coordinates.

Source: Stim, doc/file_format_stim_circuit.md (DETECTOR carries
coordinates, SHIFT_COORDS shifts them) and stim.Circuit.generated, whose
surface-code circuits put the time layer in the last coordinate: layer 0
for the detectors compared against the prepared state, layer t for the
ones formed after round t, and one more layer for the final data readout.
decsim's rule: layer t is round t + 1, the readout layer folds into the
last round, and detectors keep Stim's index order inside a round.
"""

import collections

import pytest
import stim

from decsim.detector_error_model import detector_chronology


def surface_code_circuit(rounds):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=rounds,
        after_clifford_depolarization=0.001,
    )


def rounds_from_layers(coordinates, round_count):
    """Stim's layers as decsim rounds: t + 1, readout folded into the last."""
    expected = {}
    for detector_id, detector_coordinates in coordinates.items():
        layer = int(detector_coordinates[-1])
        next_round = layer + 1
        expected[detector_id] = min(next_round, round_count)
    return expected


def test_layer_t_is_round_t_plus_one_and_the_readout_folds_into_the_last():
    circuit = surface_code_circuit(4)
    coordinates = circuit.get_detector_coordinates()
    round_by_detector = detector_chronology.resolve_detector_rounds(
        circuit, None, 4
    )
    assert round_by_detector == rounds_from_layers(coordinates, 4)
    rounds = round_by_detector.values()
    assert collections.Counter(rounds) == {1: 4, 2: 8, 3: 8, 4: 12}


def test_a_repetition_code_uses_its_two_coordinates():
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=3, rounds=3
    )
    round_by_detector = detector_chronology.resolve_detector_rounds(
        circuit, None, 3
    )
    rounds = round_by_detector.values()
    assert collections.Counter(rounds) == {1: 2, 2: 2, 3: 4}


def test_a_declared_map_replaces_the_coordinates():
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=3, rounds=2
    )
    from_coordinates = detector_chronology.resolve_detector_rounds(
        circuit, None, 2
    )
    assert from_coordinates[2] == 2
    assert from_coordinates[3] == 2
    declared = {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2}
    round_by_detector = detector_chronology.resolve_detector_rounds(
        circuit, declared, 2
    )
    assert round_by_detector == declared
    by_round = detector_chronology.detectors_by_round(round_by_detector)
    assert by_round == {1: [0, 1, 2, 3], 2: [4, 5]}


def test_a_declared_map_that_misses_a_detector_is_refused():
    circuit = surface_code_circuit(2)
    with pytest.raises(ValueError, match="cover every detector"):
        detector_chronology.resolve_detector_rounds(circuit, {0: 1}, 2)


def test_a_declared_round_past_the_last_round_is_refused():
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=3, rounds=2
    )
    declared = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3}
    with pytest.raises(ValueError, match="inside the emitted rounds"):
        detector_chronology.resolve_detector_rounds(circuit, declared, 2)


def test_a_zero_round_count_is_refused():
    circuit = surface_code_circuit(2)
    with pytest.raises(ValueError, match="round_count must be positive"):
        detector_chronology.resolve_detector_rounds(circuit, None, 0)


def test_a_circuit_without_a_detector_is_refused():
    circuit = stim.Circuit("R 0\nM 0\n")
    with pytest.raises(ValueError, match="at least one detector"):
        detector_chronology.resolve_detector_rounds(circuit, None, 1)


def test_detectors_with_different_coordinate_arities_are_refused():
    circuit = stim.Circuit(
        "R 0 1\nM 0 1\nDETECTOR(0,0,0) rec[-1]\nDETECTOR(0,0) rec[-2]\n"
    )
    with pytest.raises(ValueError, match="need one arity"):
        detector_chronology.resolve_detector_rounds(circuit, None, 1)


def test_a_single_coordinate_carries_no_layer_and_is_refused():
    circuit = stim.Circuit(
        "R 0 1\nM 0 1\nDETECTOR(0) rec[-1]\nDETECTOR(0) rec[-2]\n"
    )
    with pytest.raises(ValueError, match="explicit detector_rounds"):
        detector_chronology.resolve_detector_rounds(circuit, None, 1)


def test_a_layer_that_is_not_an_integer_is_refused():
    circuit = stim.Circuit("R 0\nM 0\nDETECTOR(0,0.5) rec[-1]\n")
    with pytest.raises(ValueError, match="must be finite integers"):
        detector_chronology.resolve_detector_rounds(circuit, None, 1)


def test_a_layer_past_the_declared_duration_is_refused():
    circuit = stim.Circuit("R 0\nM 0\nDETECTOR(0,5) rec[-1]\n")
    with pytest.raises(ValueError, match="inside the declared source duration"):
        detector_chronology.resolve_detector_rounds(circuit, None, 1)


def test_the_detectors_of_a_round_keep_stims_index_order():
    circuit = surface_code_circuit(4)
    round_by_detector = detector_chronology.resolve_detector_rounds(
        circuit, None, 4
    )
    by_round = detector_chronology.detectors_by_round(round_by_detector)
    assert by_round[1] == [0, 1, 2, 3]
    assert by_round[2] == [4, 5, 6, 7, 8, 9, 10, 11]
    assert by_round[4] == list(range(20, 32))


def test_a_detectors_position_counts_from_zero_inside_its_round():
    circuit = surface_code_circuit(4)
    round_by_detector = detector_chronology.resolve_detector_rounds(
        circuit, None, 4
    )
    position_by_detector = detector_chronology.detector_position_in_round(
        round_by_detector
    )
    assert position_by_detector[0] == 0
    assert position_by_detector[4] == 0
    assert position_by_detector[11] == 7
    assert position_by_detector[20] == 0
    assert position_by_detector[31] == 11


def test_coordinates_are_returned_for_rows_only_when_every_row_has_some():
    coordinates = {0: [1.0, 2.0, 0.0], 1: [3, 4, 1], 2: []}
    with_all = detector_chronology.coordinates_for_rows(coordinates, [1, 0])
    assert with_all == ((3.0, 4.0, 1.0), (1.0, 2.0, 0.0))
    with_one_missing = detector_chronology.coordinates_for_rows(
        coordinates, [0, 2]
    )
    assert with_one_missing is None


def test_a_checked_map_is_the_object_it_was_given():
    declared = {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2}
    checked = detector_chronology.checked_detector_round_map(declared, 6, 2)
    assert checked is declared
