"""The formation table reads the caps off the circuit, and streaming formation
equals Stim's converter bit for bit, reference parity included."""

import numpy as np
import pytest
import stim

from decsim.detector_error_model.detector_chronology import resolve_detector_rounds
from decsim.detector_error_model.detector_formation import (
    LayerKind,
    StreamingDetectorFormer,
    build_formation_table,
    form_shot,
    split_measurements_into_packets,
)


def memory_circuit(task, distance, rounds, p=0.005):
    return stim.Circuit.generated(
        task, distance=distance, rounds=rounds,
        after_clifford_depolarization=p, before_round_data_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p)


def with_x_after_prep(circuit, qubits):
    """A legal circuit whose noiseless detectors are not all zero."""
    out = stim.Circuit()
    inserted = False
    for instruction in circuit:
        out.append(instruction)
        if not inserted and instruction.name == "R":
            out.append("X", qubits)
            inserted = True
    return out


def test_table_reads_the_caps_off_a_d3_memory_circuit():
    table = build_formation_table(memory_circuit("surface_code:rotated_memory_z", 3, 12), 12)

    assert [table.packet_width[r] for r in range(1, 13)] == [8] * 11 + [17]
    assert table.max_record_span == 1
    assert {recipe.kind for recipe in table.detectors_of_round(1)} == {LayerKind.PREP}
    assert {recipe.kind for recipe in table.detectors_of_round(6)} == {LayerKind.BULK}
    assert {recipe.kind for recipe in table.detectors_of_round(12)} == {LayerKind.BULK, LayerKind.READOUT}
    assert [len(table.detectors_of_round(r)) for r in (1, 2, 11, 12)] == [4, 8, 8, 12]
    assert all(recipe.reference_parity == 0 for recipe in table.detectors)
    assert len(table.observables) == 1


def test_table_rounds_agree_with_resolve_detector_rounds():
    circuit = memory_circuit("surface_code:rotated_memory_z", 3, 12)
    table = build_formation_table(circuit, 12)
    assert table.detector_rounds() == resolve_detector_rounds(circuit, None, 12)


def test_round_count_must_match_the_circuit():
    with pytest.raises(ValueError, match="measurement rounds"):
        build_formation_table(memory_circuit("surface_code:rotated_memory_z", 3, 12), 11)


@pytest.mark.parametrize("name, circuit, rounds", [
    ("memory_z d3", memory_circuit("surface_code:rotated_memory_z", 3, 12), 12),
    ("memory_x d3", memory_circuit("surface_code:rotated_memory_x", 3, 7), 7),
    ("repetition d5", memory_circuit("repetition_code:memory", 5, 9, 0.01), 9),
    ("memory_z d5", memory_circuit("surface_code:rotated_memory_z", 5, 10, 0.003), 10),
    ("x after prep", with_x_after_prep(memory_circuit("surface_code:rotated_memory_z", 3, 6), [1]), 6),
])
def test_streaming_formation_equals_stim_converter(name, circuit, rounds):
    table = build_formation_table(circuit, rounds)
    shots = 200
    measurements = circuit.compile_sampler(seed=11).sample(shots)
    expected_events, expected_observables = circuit.compile_m2d_converter().convert(
        measurements=measurements, separate_observables=True)

    formed_events = np.zeros_like(expected_events, dtype=np.uint8)
    formed_observables = np.zeros_like(expected_observables, dtype=np.uint8)
    for shot in range(shots):
        packets = split_measurements_into_packets(table, measurements[shot])
        formed_events[shot], formed_observables[shot] = form_shot(table, packets)

    assert np.array_equal(formed_events, expected_events.astype(np.uint8)), name
    assert np.array_equal(formed_observables, expected_observables.astype(np.uint8)), name


def test_reference_parity_is_live_on_a_nonzero_reference_circuit():
    circuit = with_x_after_prep(memory_circuit("surface_code:rotated_memory_z", 3, 6), [1])
    table = build_formation_table(circuit, 6)
    assert sum(recipe.reference_parity for recipe in table.detectors) == 1

    measurements = circuit.compile_sampler(seed=11).sample(100)
    formed = np.array([form_shot(table, split_measurements_into_packets(table, row))[0]
                       for row in measurements], dtype=np.uint8)
    without_reference = circuit.compile_m2d_converter(skip_reference_sample=True).convert(
        measurements=measurements, separate_observables=True)[0]
    assert not np.array_equal(formed, without_reference.astype(np.uint8))


def test_former_rejects_a_packet_of_the_wrong_width_and_keeps_only_two_rounds():
    table = build_formation_table(memory_circuit("surface_code:rotated_memory_z", 3, 4), 4)
    former = StreamingDetectorFormer(table)
    with pytest.raises(ValueError, match="packet has 3 bits"):
        former.feed_packet(1, [0, 1, 0])
    former.feed_packet(1, np.zeros(8, dtype=np.uint8))
    former.feed_packet(2, np.zeros(8, dtype=np.uint8))
    former.feed_packet(3, np.zeros(8, dtype=np.uint8))
    assert set(former.packets) == {2, 3}
