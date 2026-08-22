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
    assert table.readout_slot_start == 8
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
    with pytest.raises(ValueError, match="asked for 11"):
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


def test_declared_measurement_rounds_describe_a_two_block_round_layout():
    """A QLX-style circuit measures Z then X ancillas as two instructions per
    round, has no round-1 detectors, and ends with a readout that feeds only
    the observable; the frontend declares the packet schedule."""
    circuit = stim.Circuit.from_file(
        __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "qlx" / "mem_surface.stim")
    rounds = 8
    checks_per_round = 8
    measurement_count = sum(len(i.targets_copy()) for i in circuit.flattened()
                            if i.name in ("M", "MR"))
    measurement_rounds = {index: min(index // checks_per_round + 1, rounds)
                          for index in range(measurement_count)}
    table = build_formation_table(circuit, rounds, measurement_rounds=measurement_rounds)

    assert [table.packet_width[r] for r in range(1, 9)] == [8] * 7 + [17]
    assert table.readout_slot_start is None
    assert table.detectors_of_round(1) == []
    assert [len(table.detectors_of_round(r)) for r in range(2, 9)] == [8] * 7
    assert len(table.observables) == 1

    shots = 100
    measurements = circuit.compile_sampler(seed=3).sample(shots)
    expected = circuit.compile_m2d_converter().convert(measurements=measurements, separate_observables=True)
    formed = [form_shot(table, split_measurements_into_packets(table, row)) for row in measurements]
    assert np.array_equal(np.array([f[0] for f in formed], dtype=np.uint8), expected[0].astype(np.uint8))
    assert np.array_equal(np.array([f[1] for f in formed], dtype=np.uint8), expected[1].astype(np.uint8))


def test_declared_detector_round_may_not_precede_its_bits():
    circuit = memory_circuit("surface_code:rotated_memory_z", 3, 4)
    too_early = {index: 1 for index in range(circuit.num_detectors)}
    with pytest.raises(ValueError, match="arrives in round"):
        build_formation_table(circuit, 4, detector_rounds=too_early)


def test_lattice_surgery_cnot_circuit_forms_like_stim():
    """A tqec lattice-surgery CNOT (k=1): twelve rounds, a mid-circuit data
    readout at the merge, a final readout, two observables, detector counts
    that change from round to round. The circuit rule assigns every
    measurement block to the round its DETECTOR group announces."""
    circuit = stim.Circuit.from_file(
        __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "tqec_cnot_k1.stim")
    rounds = 12
    table = build_formation_table(circuit, rounds)
    assert table.readout_slot_start is None
    assert [table.packet_width[r] for r in range(1, 13)] == [16, 16, 16, 28, 28, 31, 28, 28, 40, 16, 16, 34]
    assert [len(table.detectors_of_round(r)) for r in range(1, 13)] == [8, 16, 16, 20, 28, 28, 24, 28, 32, 16, 16, 24]
    assert table.detector_rounds() == resolve_detector_rounds(circuit, None, rounds)
    assert len(table.observables) == 2
    assert {recipe.kind for recipe in table.detectors_of_round(9)} >= {LayerKind.BULK, LayerKind.READOUT}

    shots = 150
    measurements = circuit.compile_sampler(seed=5).sample(shots)
    expected = circuit.compile_m2d_converter().convert(measurements=measurements, separate_observables=True)
    formed = [form_shot(table, split_measurements_into_packets(table, row)) for row in measurements]
    assert np.array_equal(np.array([f[0] for f in formed], dtype=np.uint8), expected[0].astype(np.uint8))
    assert np.array_equal(np.array([f[1] for f in formed], dtype=np.uint8), expected[1].astype(np.uint8))


def _forms_like_stim(circuit, rounds, shots=200, seed=3):
    table = build_formation_table(circuit, rounds)
    measurements = circuit.compile_sampler(seed=seed).sample(shots)
    expected = circuit.compile_m2d_converter().convert(measurements=measurements, separate_observables=True)
    formed = [form_shot(table, split_measurements_into_packets(table, row)) for row in measurements]
    assert np.array_equal(np.array([f[0] for f in formed], dtype=np.uint8), expected[0].astype(np.uint8))
    assert np.array_equal(np.array([f[1] for f in formed], dtype=np.uint8), expected[1].astype(np.uint8))


@pytest.mark.parametrize("two_qubit_measurement", ["MZZ 0 1", "MXX 0 1", "MPP Z0*Z1"])
def test_pair_and_product_measurements_count_one_record(two_qubit_measurement):
    """MXX/MYY/MZZ and MPP take several targets but append one record each;
    the record count comes from Stim, not from the target count."""
    circuit = stim.Circuit(f"""
        R 0 1 2
        X_ERROR(0.1) 0 1 2
        {two_qubit_measurement}
        DETECTOR(0,0,0) rec[-1]
        X_ERROR(0.1) 0 1 2
        {two_qubit_measurement}
        DETECTOR(0,0,1) rec[-1] rec[-2]
        M 0 1 2
        DETECTOR(0,0,2) rec[-3] rec[-2] rec[-4]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    _forms_like_stim(circuit, 2)


def test_pauli_targets_in_observable_include_are_skipped():
    circuit = stim.Circuit("""
        R 0 1
        X_ERROR(0.1) 0 1
        M 0
        DETECTOR(0,0,0) rec[-1]
        X_ERROR(0.1) 0 1
        M 0
        DETECTOR(0,0,1) rec[-1] rec[-2]
        M 1
        OBSERVABLE_INCLUDE(0) Z1 rec[-1]
    """)
    _forms_like_stim(circuit, 2)


def test_padding_and_heralded_records_shift_later_lookbacks():
    """MPAD and HERALDED_ERASE append records that no detector reads; the
    lookbacks of later detectors must still count them."""
    circuit = stim.Circuit("""
        R 0 1
        MPAD 0
        HERALDED_ERASE(0.1) 0
        X_ERROR(0.1) 0 1
        M 0
        DETECTOR(0,0,0) rec[-1]
        X_ERROR(0.1) 0 1
        M 0
        DETECTOR(0,0,1) rec[-1] rec[-2]
        M 1
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    _forms_like_stim(circuit, 2)
