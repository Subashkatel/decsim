"""Streaming detector formation equals Stim's measurement-to-detector rule.

Source: stim.Circuit.compile_m2d_converter (measurements_to_detection_events):
a detector is the XOR of the measurement records it names, compared
against the noiseless reference sample; an observable is the XOR of its
records the same way. The packet layout follows Stim's generated
circuits: one measurement block per round, and the final data readout
folded into the last round's packet after that round's own bits.
"""

import pathlib

import numpy
import pytest
import stim

from decsim.detector_error_model import detector_chronology, detector_formation


def surface_code_circuit(rounds):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=rounds,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
    )


def formed_by_decsim(table, measurements):
    """Every shot formed packet by packet, as uint8 rows."""
    events = []
    observables = []
    for row in measurements:
        packets = detector_formation.split_measurements_into_packets(table, row)
        shot_events, shot_observables = detector_formation.form_shot(
            table, packets
        )
        events.append(shot_events)
        observables.append(shot_observables)
    return numpy.array(events, dtype=numpy.uint8), numpy.array(
        observables, dtype=numpy.uint8
    )


def formed_by_stim(circuit, measurements):
    converter = circuit.compile_m2d_converter()
    events, observables = converter.convert(
        measurements=measurements, separate_observables=True
    )
    return events.astype(numpy.uint8), observables.astype(numpy.uint8)


def formation_table(rounds):
    circuit = surface_code_circuit(rounds)
    return detector_formation.build_formation_table(circuit, rounds)


def kinds_of_round(table, round_index):
    return {recipe.kind for recipe in table.detectors_of_round(round_index)}


def empty_packets(table):
    """One all-zero packet per round, keyed by round."""
    packets = {}
    for round_index, width in table.packet_width_by_round.items():
        packets[round_index] = [0] * width
    return packets


def test_the_table_reads_the_packet_layout_off_a_generated_circuit():
    table = formation_table(4)
    assert table.packet_width_by_round == {1: 8, 2: 8, 3: 8, 4: 17}
    assert table.readout_slot_start == 8
    assert table.max_record_span == 1
    assert len(table.detectors) == 32
    assert len(table.observables) == 1


def test_detector_kinds_follow_how_many_records_they_read():
    table = formation_table(4)
    assert kinds_of_round(table, 1) == {
        detector_formation.LayerKind.PREPARATION
    }
    assert kinds_of_round(table, 2) == {detector_formation.LayerKind.BULK}
    assert kinds_of_round(table, 4) == {
        detector_formation.LayerKind.BULK,
        detector_formation.LayerKind.READOUT,
    }


def test_the_tables_rounds_are_the_chronologys_rounds():
    circuit = surface_code_circuit(4)
    table = detector_formation.build_formation_table(circuit, 4)
    resolved = detector_chronology.resolve_detector_rounds(circuit, None, 4)
    assert table.detector_rounds() == resolved


def test_streaming_formation_equals_stims_converter_on_sampled_shots():
    circuit = surface_code_circuit(4)
    table = detector_formation.build_formation_table(circuit, 4)
    sampler = circuit.compile_sampler(seed=7)
    measurements = sampler.sample(64)
    decsim_events, decsim_observables = formed_by_decsim(table, measurements)
    stim_events, stim_observables = formed_by_stim(circuit, measurements)
    assert numpy.array_equal(decsim_events, stim_events)
    assert numpy.array_equal(decsim_observables, stim_observables)
    assert decsim_events.any()


def test_the_reference_parity_is_used_when_the_expected_reading_is_one():
    circuit = stim.Circuit("R 0\nX 0\nM 0\nDETECTOR rec[-1]\n")
    table = detector_formation.build_formation_table(circuit, 1)
    assert table.detectors[0].reference_parity == 1
    events, observables = detector_formation.form_shot(table, {1: (1,)})
    assert events == (0,)
    assert observables == ()


def test_observables_come_out_with_the_last_round_and_are_none_before():
    table = formation_table(4)
    former = detector_formation.StreamingDetectorFormer(table)
    packets = empty_packets(table)
    _, after_first = former.feed_packet(1, packets[1])
    _, after_second = former.feed_packet(2, packets[2])
    _, after_third = former.feed_packet(3, packets[3])
    _, after_last = former.feed_packet(4, packets[4])
    assert after_first is None
    assert after_second is None
    assert after_third is None
    assert after_last == [(0, 0)]


def test_the_record_span_reaches_back_as_far_as_an_observable_reads():
    circuit = stim.Circuit(
        "R 0 1\n"
        "M 0\nDETECTOR(0,0) rec[-1]\n"
        "M 1\nDETECTOR(0,1) rec[-1]\n"
        "M 0\nDETECTOR(0,2) rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-3] rec[-1]\n"
    )
    table = detector_formation.build_formation_table(circuit, 3)
    former = detector_formation.StreamingDetectorFormer(table)
    assert table.detectors[2].records == ((3, 0),)
    assert table.observables[0].records == ((1, 0), (3, 0))
    assert table.max_record_span == 2
    assert former.kept_packet_count == 3


def test_a_packet_of_the_wrong_width_is_refused():
    table = formation_table(4)
    former = detector_formation.StreamingDetectorFormer(table)
    with pytest.raises(ValueError, match="packet has 3 bits"):
        former.feed_packet(1, [0, 1, 0])


def test_the_former_keeps_only_the_packets_a_recipe_can_reach():
    table = formation_table(4)
    former = detector_formation.StreamingDetectorFormer(table)
    empty_packet = [0] * 8
    former.feed_packet(1, empty_packet)
    former.feed_packet(2, empty_packet)
    former.feed_packet(3, empty_packet)
    assert set(former.packets) == {2, 3}


def test_a_round_count_the_circuit_does_not_announce_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(ValueError, match="asked for 3"):
        detector_formation.build_formation_table(circuit, 3)


def test_a_second_group_past_the_last_round_is_refused():
    circuit = stim.Circuit(
        "R 0 1 2\n"
        "M 0\nDETECTOR(0,0) rec[-1]\n"
        "M 1\nDETECTOR(0,1) rec[-1]\n"
        "M 2\nDETECTOR(0,1) rec[-1]\n"
    )
    with pytest.raises(ValueError, match="announces round 2, .* asked for 1"):
        detector_formation.build_formation_table(circuit, 1)


def test_a_readout_announced_two_rounds_past_the_end_is_refused():
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=4,
        after_clifford_depolarization=0.001,
    )
    # Stim's readout detectors announce round 5, the one layer past the
    # last round that folds into it. One more SHIFT_COORDS after the
    # REPEAT block makes them announce round 6, which does not fold.
    text = str(circuit)
    shifted_text = text.replace("}", "}\nSHIFT_COORDS(0, 0, 1)")
    shifted = stim.Circuit(shifted_text)
    with pytest.raises(
        ValueError,
        match="circuit announces round 6, formation table was asked for 4",
    ):
        detector_formation.build_formation_table(shifted, 4)


def test_measurement_blocks_out_of_round_order_are_refused():
    circuit = stim.Circuit(
        "R 0 1\nM 0\nDETECTOR(0,1) rec[-1]\nM 1\nDETECTOR(0,0) rec[-1]\n"
    )
    with pytest.raises(ValueError, match="not in round order"):
        detector_formation.build_formation_table(circuit, 2)


def test_a_declared_measurement_round_outside_the_operation_is_refused():
    circuit = stim.Circuit("R 0\nM 0\nDETECTOR(0,0) rec[-1]\n")
    with pytest.raises(ValueError, match="must lie in 1..round_count"):
        detector_formation.build_formation_table(
            circuit, 1, measurement_rounds={0: 2}
        )


def test_declared_measurement_rounds_that_decrease_are_refused():
    circuit = stim.Circuit(
        "R 0 1\nM 0\nDETECTOR(0,0) rec[-1]\nM 1\nDETECTOR(0,1) rec[-1]\n"
    )
    with pytest.raises(ValueError, match="must be non-decreasing"):
        detector_formation.build_formation_table(
            circuit, 2, measurement_rounds={0: 2, 1: 1}
        )


def test_a_declared_detector_round_may_not_precede_its_bits():
    circuit = surface_code_circuit(4)
    too_early = dict.fromkeys(range(32), 1)
    with pytest.raises(ValueError, match="arrives in round"):
        detector_formation.build_formation_table(
            circuit, 4, detector_rounds=too_early
        )


def test_a_declared_detector_round_just_past_the_last_round_is_refused():
    circuit = surface_code_circuit(4)
    in_time = detector_chronology.resolve_detector_rounds(circuit, None, 4)
    # Stim's last coordinate puts detectors 20..31 in round 4; declared
    # in round 5 they would never be formed.
    assert in_time[20] == 4
    assert in_time[31] == 4
    readout_past_the_end = dict.fromkeys(range(20, 32), 5)
    past_the_end = dict(in_time)
    past_the_end.update(readout_past_the_end)
    with pytest.raises(ValueError, match="inside the emitted rounds"):
        detector_formation.build_formation_table(
            circuit, 4, detector_rounds=past_the_end
        )


def test_a_declared_detector_round_of_zero_is_refused():
    circuit = surface_code_circuit(4)
    in_time = detector_chronology.resolve_detector_rounds(circuit, None, 4)
    readout_before_the_start = dict.fromkeys(range(20, 32), 0)
    before_the_start = dict(in_time)
    before_the_start.update(readout_before_the_start)
    with pytest.raises(ValueError, match="inside the emitted rounds"):
        detector_formation.build_formation_table(
            circuit, 4, detector_rounds=before_the_start
        )


def test_a_declared_detector_round_on_the_last_round_is_accepted():
    circuit = surface_code_circuit(4)
    in_time = detector_chronology.resolve_detector_rounds(circuit, None, 4)
    # Detectors 12..19 arrive in round 3; declared in round 4, the last
    # round, they are formed with the readout.
    assert in_time[12] == 3
    assert in_time[19] == 3
    on_the_last_round = dict.fromkeys(range(12, 20), 4)
    delayed = dict(in_time)
    delayed.update(on_the_last_round)
    table = detector_formation.build_formation_table(
        circuit, 4, detector_rounds=delayed
    )
    third_round = table.detectors_of_round(3)
    last_round = table.detectors_of_round(4)
    assert table.detectors[12].round_index == 4
    assert third_round == []
    assert len(last_round) == 20


def test_an_observables_reference_parity_is_the_noiseless_reading_of_its_bits():
    circuit = stim.Circuit(
        "R 0\nX 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
    )
    table = detector_formation.build_formation_table(circuit, 1)
    assert table.observables[0].reference_parity == 1
    events, observables = detector_formation.form_shot(table, {1: (1,)})
    assert events == (0,)
    assert observables == (0,)


def test_a_declared_detector_round_equal_to_its_arrival_round_is_accepted():
    circuit = surface_code_circuit(4)
    from_the_circuit = detector_formation.build_formation_table(circuit, 4)
    arrival_rounds = from_the_circuit.detector_rounds()
    declared = detector_formation.build_formation_table(
        circuit, 4, detector_rounds=arrival_rounds
    )
    assert arrival_rounds[4] == 2
    assert declared.detectors[4].round_index == 2
    assert declared.detector_rounds() == arrival_rounds


def test_a_declared_detector_round_after_its_bits_is_what_the_recipe_uses():
    circuit = surface_code_circuit(4)
    from_the_circuit = detector_formation.build_formation_table(circuit, 4)
    later = from_the_circuit.detector_rounds()
    later[4] = 3
    later[12] = 4
    table = detector_formation.build_formation_table(
        circuit, 4, detector_rounds=later
    )
    third_round = table.detectors_of_round(3)
    assert table.detectors[4].round_index == 3
    assert table.detectors[4].records == ((2, 0), (1, 0))
    assert table.detectors[12].round_index == 4
    assert len(third_round) == 8
    assert third_round[0].detector_index == 4
    assert third_round[1].detector_index == 13
    assert table.max_record_span == 2


def test_a_round_count_past_the_rounds_the_circuit_announces_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(
        ValueError,
        match=r"announces rounds \[1, 2, 3, 4, 5\], formation table was "
        "asked for 6",
    ):
        detector_formation.build_formation_table(circuit, 6)


def test_a_zero_round_count_is_refused():
    circuit = surface_code_circuit(4)
    with pytest.raises(ValueError, match="round_count must be positive"):
        detector_formation.build_formation_table(circuit, 0)


def test_a_measurement_round_map_that_misses_a_measurement_is_refused():
    circuit = stim.Circuit("R 0 1\nM 0 1\nDETECTOR(0,0) rec[-1] rec[-2]\n")
    with pytest.raises(ValueError, match="cover every measurement exactly"):
        detector_formation.build_formation_table(
            circuit, 1, measurement_rounds={0: 1}
        )


def test_a_detector_round_map_that_misses_a_detector_is_refused():
    circuit = surface_code_circuit(2)
    with pytest.raises(ValueError, match="cover every detector exactly"):
        detector_formation.build_formation_table(
            circuit, 2, detector_rounds={0: 1}
        )


THIS_FILE = pathlib.Path(__file__)
DATA_DIRECTORY = THIS_FILE.parents[1] / "data"


def forms_like_stim(circuit, rounds, seed=3, shots=200):
    """The streaming formation beside Stim's converter, shot for shot."""
    table = detector_formation.build_formation_table(circuit, rounds)
    sampler = circuit.compile_sampler(seed=seed)
    measurements = sampler.sample(shots)
    events, observables = formed_by_decsim(table, measurements)
    stim_events, stim_observables = formed_by_stim(circuit, measurements)
    assert numpy.array_equal(events, stim_events)
    assert numpy.array_equal(observables, stim_observables)
    return table


def test_a_declared_map_folds_two_measurement_blocks_into_one_round():
    """A QLX circuit measures Z then X ancillas as two blocks per round.

    The frontend declares that packet schedule, so the round a
    measurement belongs to comes from the map and not from the circuit's
    detector groups: round 1 carries its bits but announces no detector,
    and the last round's packet carries the data readout after its own
    ancilla bits.
    """
    circuit_path = DATA_DIRECTORY / "qlx" / "mem_surface.stim"
    circuit = stim.Circuit.from_file(circuit_path)
    rounds = 8
    checks_per_round = 8
    measurement_count = 0
    for instruction in circuit.flattened():
        if instruction.name in ("M", "MR"):
            targets = instruction.targets_copy()
            measurement_count += len(targets)
    measurement_rounds = {}
    for index in range(measurement_count):
        announced_round = index // checks_per_round + 1
        measurement_rounds[index] = min(announced_round, rounds)

    table = detector_formation.build_formation_table(
        circuit, rounds, measurement_rounds=measurement_rounds
    )

    widths = [table.packet_width_by_round[index] for index in range(1, 9)]
    detector_counts = []
    for index in range(1, 9):
        detectors = table.detectors_of_round(index)
        detector_counts.append(len(detectors))
    assert widths == [8] * 7 + [17]
    assert table.readout_slot_start is None
    assert detector_counts == [0] + [8] * 7
    assert len(table.observables) == 1

    sampler = circuit.compile_sampler(seed=3)
    measurements = sampler.sample(100)
    events, observables = formed_by_decsim(table, measurements)
    stim_events, stim_observables = formed_by_stim(circuit, measurements)
    assert numpy.array_equal(events, stim_events)
    assert numpy.array_equal(observables, stim_observables)


def test_a_lattice_surgery_cnot_forms_like_stim_round_for_round():
    """Twelve rounds, a mid-circuit readout at the merge, two observables.

    The circuit is a tqec lattice-surgery CNOT at k=1 (tests/data/
    tqec_cnot_k1.stim). With no declared map every measurement block
    belongs to the round its DETECTOR group announces, so the packet
    widths and detector counts change from round to round.
    """
    circuit_path = DATA_DIRECTORY / "tqec_cnot_k1.stim"
    circuit = stim.Circuit.from_file(circuit_path)

    table = forms_like_stim(circuit, 12, seed=5, shots=150)

    widths = [table.packet_width_by_round[index] for index in range(1, 13)]
    detector_counts = []
    for index in range(1, 13):
        detectors = table.detectors_of_round(index)
        detector_counts.append(len(detectors))
    assert table.readout_slot_start is None
    assert widths == [16, 16, 16, 28, 28, 31, 28, 28, 40, 16, 16, 34]
    assert detector_counts == [8, 16, 16, 20, 28, 28, 24, 28, 32, 16, 16, 24]
    assert len(table.observables) == 2
    rounds_of_the_table = table.detector_rounds()
    chronology_rounds = detector_chronology.resolve_detector_rounds(
        circuit, None, 12
    )
    assert rounds_of_the_table == chronology_rounds


@pytest.mark.parametrize(
    "two_qubit_measurement", ["MZZ 0 1", "MXX 0 1", "MPP Z0*Z1"]
)
def test_a_pair_or_product_measurement_appends_one_record(
    two_qubit_measurement,
):
    """The record count is Stim's num_measurements, not the target count.

    A lattice-surgery circuit measures a pair or a product with one
    instruction over two targets, and it appends one measurement record;
    counting targets instead would shift every later lookback.
    """
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
    forms_like_stim(circuit, 2)


def test_a_pauli_target_in_an_observable_is_not_a_record():
    """OBSERVABLE_INCLUDE may name a Pauli beside its records."""
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
    forms_like_stim(circuit, 2)


def test_padded_and_heralded_records_shift_the_lookbacks_after_them():
    """MPAD and HERALDED_ERASE append records no detector reads."""
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
    forms_like_stim(circuit, 2)
