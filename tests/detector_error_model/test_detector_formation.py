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


def test_a_former_takes_a_table_whose_new_round_reads_a_round_never_given():
    """A seat that formed round 2 alone is given round 1 when round 3 needs it.

    The seat never held round 1, so the longer table discards nothing,
    and the read that forms round 3 gives it round 1 raw.
    """
    circuit = stim.Circuit("R 0\nM 0\nDETECTOR rec[-1]\nM 0\nDETECTOR rec[-1]")
    table = detector_formation.build_formation_table(
        circuit, 2, measurement_rounds={0: 1, 1: 2}, live_reach=2
    )
    former = detector_formation.StreamingDetectorFormer(table)
    former.feed_packet(2, (1,))
    circuit.append_from_stim_program_text("M 0\nDETECTOR rec[-1] rec[-3]")
    longer = detector_formation.build_formation_table(
        circuit, 3, measurement_rounds={0: 1, 1: 2, 2: 3}
    )

    former.extend_table(longer)
    former.hold_packet(1, (1,))
    events = former.feed_packet(3, (0,))

    assert events == [(2, 1)]


def test_a_table_extension_cannot_change_an_existing_packet() -> None:
    circuit = stim.Circuit("R 0\nM 0\nDETECTOR rec[-1]")
    table = detector_formation.build_formation_table(
        circuit, 1, measurement_rounds={0: 1}
    )
    former = detector_formation.StreamingDetectorFormer(table)
    circuit.append_from_stim_program_text("M 0\nDETECTOR rec[-1] rec[-2]")
    changed = detector_formation.build_formation_table(
        circuit, 1, measurement_rounds={0: 1, 1: 1}
    )
    with pytest.raises(RuntimeError, match="changes an existing packet"):
        former.extend_table(changed)


def test_a_table_extension_cannot_reinterpret_an_existing_detector() -> None:
    circuit = stim.Circuit("R 0\nM 0\nDETECTOR rec[-1]")
    table = detector_formation.build_formation_table(
        circuit, 1, measurement_rounds={0: 1}
    )
    former = detector_formation.StreamingDetectorFormer(table)
    changed_circuit = stim.Circuit("R 0\nX 0\nM 0\nDETECTOR rec[-1]\nM 0")
    changed = detector_formation.build_formation_table(
        changed_circuit, 2, measurement_rounds={0: 1, 1: 2}
    )
    with pytest.raises(RuntimeError, match="changes an existing detector"):
        former.extend_table(changed)


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


def test_sparse_observable_ids_equal_stims_converter_on_sampled_shots():
    """A circuit naming observables 0 and 3 declares four; 1 and 2 are 0.

    Stim's converter reports circuit.num_observables columns, an ID the
    circuit never names always zero.
    """
    circuit = stim.Circuit(
        "R 0 1\nX_ERROR(0.3) 0 1\nM 0 1\nDETECTOR rec[-2]\n"
        "OBSERVABLE_INCLUDE(0) rec[-2]\nOBSERVABLE_INCLUDE(3) rec[-1]\n"
    )
    table = detector_formation.build_formation_table(circuit, 1)
    sampler = circuit.compile_sampler(seed=7)
    measurements = sampler.sample(64)
    decsim_events, decsim_observables = formed_by_decsim(table, measurements)
    stim_events, stim_observables = formed_by_stim(circuit, measurements)
    assert decsim_observables.shape == (64, 4)
    assert numpy.array_equal(decsim_events, stim_events)
    assert numpy.array_equal(decsim_observables, stim_observables)
    assert decsim_observables[:, 3].any()


def test_the_reference_parity_is_used_when_the_expected_reading_is_one():
    circuit = stim.Circuit("R 0\nX 0\nM 0\nDETECTOR rec[-1]\n")
    table = detector_formation.build_formation_table(circuit, 1)
    assert table.detectors[0].reference_parity == 1
    events, observables = detector_formation.form_shot(table, {1: (1,)})
    assert events == (0,)
    assert observables == ()


def test_a_former_keeps_no_packet_for_an_observable():
    """Round 1's observable record folds in form_shot, not in the former."""
    circuit = _first_round_observed_circuit()
    table = detector_formation.build_formation_table(
        circuit, 3, measurement_rounds={0: 1, 1: 2, 2: 3}
    )
    former = detector_formation.StreamingDetectorFormer(table)

    former.feed_packet(1, (0,))
    former.feed_packet(2, (0,))

    assert former.packets == {}


def test_an_observable_is_the_parity_of_its_rounds_records():
    """Rounds 1 and 3 hold its records; round 2's bit is in no observable."""
    circuit = _first_round_observed_circuit()
    table = detector_formation.build_formation_table(
        circuit, 3, measurement_rounds={0: 1, 1: 2, 2: 3}
    )
    packets = {1: (1,), 2: (1,), 3: (0,)}

    _, observables = detector_formation.form_shot(table, packets)

    assert observables == (1,)


def test_an_observable_reads_records_from_an_earlier_round():
    circuit = stim.Circuit(
        "R 0 1\n"
        "M 0\nDETECTOR(0,0) rec[-1]\n"
        "M 1\nDETECTOR(0,1) rec[-1]\n"
        "M 0\nDETECTOR(0,2) rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-3] rec[-1]\n"
    )
    table = detector_formation.build_formation_table(circuit, 3)
    assert table.detectors[2].records == ((3, 0),)
    assert table.observables[0].records == ((1, 0), (3, 0))


def test_a_round_reads_back_to_the_earliest_record_of_its_detectors():
    """The reset layer reads only itself; a bulk layer the round before."""
    table = formation_table(4)

    assert table.rounds_read_before(1) == 0
    assert table.rounds_read_before(3) == 1


def test_a_read_reaches_back_as_far_as_its_furthest_reaching_round():
    """Round 3 reads only itself; round 4 reads round 1 (rec[-4])."""
    circuit = stim.Circuit(
        "R 0\nREPEAT 3 {\nM 0\nDETECTOR rec[-1]\n}\n"
        "M 0\nDETECTOR rec[-1] rec[-4]\n"
    )
    measurement_rounds = {index: index + 1 for index in range(4)}
    table = detector_formation.build_formation_table(
        circuit, 4, measurement_rounds=measurement_rounds
    )
    read_rounds = range(3, 5)

    assert table.rounds_read_before_first(3, read_rounds) == 2


def test_the_last_round_reads_no_round_back_for_an_observable():
    """Round 3's detector reads round 3; the observable's round 1 folded."""
    circuit = stim.Circuit(
        "R 0 1\n"
        "M 0\nDETECTOR(0,0) rec[-1]\n"
        "M 1\nDETECTOR(0,1) rec[-1]\n"
        "M 0\nDETECTOR(0,2) rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-3] rec[-1]\n"
    )
    table = detector_formation.build_formation_table(circuit, 3)

    assert table.rounds_read_before(3) == 0


def test_a_packet_of_the_wrong_width_is_refused():
    table = formation_table(4)
    former = detector_formation.StreamingDetectorFormer(table)
    with pytest.raises(ValueError, match="packet has 3 bits"):
        former.feed_packet(1, [0, 1, 0])


def test_a_former_holds_nothing_once_the_last_round_has_formed():
    """No round reads the last one, and it read the others already."""
    table = formation_table(2)
    former = detector_formation.StreamingDetectorFormer(table)
    check_bits = [0] * 8
    check_and_data_bits = [0] * 17
    former.feed_packet(1, check_bits)
    former.feed_packet(2, check_and_data_bits)
    assert former.packets == {}


def lookback_circuit(rounds):
    """One noisy qubit whose detector reads two rounds back from round 3."""
    noisy_round = "M(0.15) 0\n"
    later_rounds = rounds - 2
    return stim.Circuit(
        f"R 0\n{noisy_round}DETECTOR rec[-1]\n{noisy_round}DETECTOR rec[-1]\n"
        f"REPEAT {later_rounds} {{\n{noisy_round}DETECTOR rec[-1] rec[-3]\n}}\n"
    )


def lookback_table(circuit, rounds):
    measurement_rounds = {index: index + 1 for index in range(rounds)}
    return detector_formation.build_formation_table(
        circuit, rounds, measurement_rounds=measurement_rounds
    )


def test_a_round_is_read_by_the_later_rounds_whose_recipes_name_it():
    """Round 2 is read by round 4 (rec[-3]); round 10 by no later round."""
    circuit = lookback_circuit(10)
    table = lookback_table(circuit, 10)

    assert table.rounds_reading(2) == (4,)
    assert table.rounds_reading(10) == ()


def test_a_packet_stays_held_while_a_round_that_reads_it_is_unformed():
    """Skoric's parallel order: rounds 1-3, then 5-7, then the seam 4.

    Round 4 reads round 2, formed long before, so round 2's packet is
    kept until round 4 forms; Stim's converter is the referent.
    """
    circuit = lookback_circuit(10)
    table = lookback_table(circuit, 10)
    sampler = circuit.compile_sampler(seed=3)
    measurements = sampler.sample(1)
    packets = detector_formation.split_measurements_into_packets(
        table, measurements[0]
    )
    stim_events, _ = formed_by_stim(circuit, measurements)
    former = detector_formation.StreamingDetectorFormer(table)
    former.feed_packet(1, packets[1])
    former.feed_packet(2, packets[2])
    former.feed_packet(3, packets[3])
    former.hold_packet(4, packets[4])
    former.feed_packet(5, packets[5])
    former.feed_packet(6, packets[6])
    former.feed_packet(7, packets[7])

    events = former.feed_packet(4, packets[4])

    formed = [bit for _, bit in events]
    stim_row = stim_events[0]
    expected = [int(stim_row[index]) for index, _ in events]
    assert formed == expected


def test_a_window_keeps_the_rounds_it_or_a_later_round_reads():
    """Every round from 3 on reads two rounds back, so from 5 on, 3 and 4."""
    circuit = lookback_circuit(10)
    table = lookback_table(circuit, 10)

    assert table.earlier_rounds_read(5) == (3, 4)
    assert table.earlier_rounds_read(1) == ()


def test_a_window_keeps_what_the_last_round_alone_reads_back():
    """Round 4 reads round 1 (rec[-4]); rounds 2 and 3 read themselves."""
    circuit = stim.Circuit(
        "R 0\nREPEAT 3 {\nM 0\nDETECTOR rec[-1]\n}\n"
        "M 0\nDETECTOR rec[-1] rec[-4]\n"
    )
    measurement_rounds = {index: index + 1 for index in range(4)}
    table = detector_formation.build_formation_table(
        circuit, 4, measurement_rounds=measurement_rounds
    )

    assert table.earlier_rounds_read(2) == (1,)


def test_a_live_window_keeps_its_reach_and_no_observed_round():
    """A live table of five rounds that may read two back; round 1 observed."""
    circuit = stim.Circuit(
        "R 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
        "REPEAT 4 {\nM 0\nDETECTOR rec[-1]\n}"
    )
    measurement_rounds = {index: index + 1 for index in range(5)}
    table = detector_formation.build_formation_table(
        circuit, 5, measurement_rounds=measurement_rounds, live_reach=2
    )

    assert table.earlier_rounds_read(5) == (3, 4)


def test_an_observable_record_reaches_no_round_back():
    """Its rec[-3] lies two rounds back, folded when that round arrived."""
    fragment = stim.Circuit(
        "M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-3]"
    )

    assert detector_formation.rounds_read_back(fragment, 1) == 0


def test_a_fragment_reads_back_as_many_rounds_as_its_furthest_record():
    """rec[-4] after its own readout lies three one-bit rounds back."""
    fragment = stim.Circuit("M 0\nDETECTOR rec[-1] rec[-4]")

    assert detector_formation.rounds_read_back(fragment, 1) == 3
    assert detector_formation.rounds_read_back(fragment, 2) == 2


def test_a_fragment_that_reads_only_itself_reaches_no_round_back():
    fragment = stim.Circuit("M 0 1\nDETECTOR rec[-2] rec[-1]")

    assert detector_formation.rounds_read_back(fragment, 2) == 0


def test_a_live_former_keeps_the_rounds_its_reach_covers():
    """Rounds 1 to 3 read themselves; a round still to run may read two back.

    Rounds 2 and 3 stay for round 4, whose recipes come later, and
    round 1 goes.
    """
    circuit = stim.Circuit("R 0\nREPEAT 3 {\nM 0\nDETECTOR rec[-1]\n}")
    measurement_rounds = {0: 1, 1: 2, 2: 3}
    table = detector_formation.build_formation_table(
        circuit, 3, measurement_rounds=measurement_rounds, live_reach=2
    )
    former = detector_formation.StreamingDetectorFormer(table)
    former.feed_packet(1, (0,))
    former.feed_packet(2, (0,))
    former.feed_packet(3, (0,))
    reaching_round = stim.Circuit("M 0\nDETECTOR rec[-1] rec[-3]")
    longer = circuit + reaching_round
    measurement_rounds[3] = 4
    longer_table = detector_formation.build_formation_table(
        longer, 4, measurement_rounds=measurement_rounds
    )
    kept_rounds = set(former.packets)

    former.extend_table(longer_table)
    events = former.feed_packet(4, (1,))

    assert kept_rounds == {2, 3}
    assert events == [(3, 1)]


def test_a_live_former_forms_its_next_round_from_the_round_its_reach_kept():
    """Round 1 stays for a round not yet run; round 2 then reads it."""
    first = stim.Circuit("R 0\nM 0\nDETECTOR rec[-1]")
    live_table = detector_formation.build_formation_table(
        first, 1, measurement_rounds={0: 1}, live_reach=1
    )
    former = detector_formation.StreamingDetectorFormer(live_table)
    first_events = former.feed_packet(1, (1,))
    complete = first + stim.Circuit("M 0\nDETECTOR rec[-1] rec[-2]")
    complete_table = detector_formation.build_formation_table(
        complete, 2, measurement_rounds={0: 1, 1: 2}
    )

    former.extend_table(complete_table)
    second_events = former.feed_packet(2, (1,))

    converter = complete.compile_m2d_converter()
    measurements = numpy.array([[1, 1]], dtype=numpy.bool_)
    expected = converter.convert(
        measurements=measurements, append_observables=False
    )
    assert expected.tolist() == [[True, False]]
    assert first_events + second_events == [(0, 1), (1, 0)]


def test_a_surface_code_former_holds_one_round_between_rounds():
    """Each bulk round reads the one before, so each round lets the last go."""
    table = formation_table(4)
    former = detector_formation.StreamingDetectorFormer(table)
    empty_packet = [0] * 8
    former.feed_packet(1, empty_packet)
    former.feed_packet(2, empty_packet)
    after_second = set(former.packets)

    former.feed_packet(3, empty_packet)

    assert after_second == {2}
    assert set(former.packets) == {3}


def test_a_retired_round_lets_go_of_the_packets_it_alone_read():
    """Round 3 reads round 1 and is formed at another seat."""
    circuit = lookback_circuit(10)
    table = lookback_table(circuit, 10)
    former = detector_formation.StreamingDetectorFormer(table)
    former.feed_packet(1, (0,))
    former.feed_packet(2, (0,))

    former.retire_round(3)

    assert set(former.packets) == {2}


def test_a_held_packet_forms_nothing_and_serves_the_round_after_it():
    """The round before a window, fetched raw, lets its first round form."""
    circuit = surface_code_circuit(4)
    table = detector_formation.build_formation_table(circuit, 4)
    sampler = circuit.compile_sampler(seed=3)
    measurements = sampler.sample(1)
    packets = detector_formation.split_measurements_into_packets(
        table, measurements[0]
    )
    stim_events, _ = formed_by_stim(circuit, measurements)
    former = detector_formation.StreamingDetectorFormer(table)

    former.hold_packet(2, packets[2])
    events = former.feed_packet(3, packets[3])

    formed = [bit for _, bit in events]
    stim_row = stim_events[0]
    expected = [int(stim_row[index]) for index, _ in events]
    assert formed == expected


def test_a_round_whose_round_before_is_not_held_is_refused():
    table = formation_table(4)
    former = detector_formation.StreamingDetectorFormer(table)
    zero_packet = (0,) * 8
    with pytest.raises(RuntimeError, match="round 3 reads round 2"):
        former.feed_packet(3, zero_packet)


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


def detector_counts(table, round_count: int) -> list:
    """The number of detectors each round 1..round_count forms."""
    counts = []
    past_the_last_round = round_count + 1
    for index in range(1, past_the_last_round):
        detectors = table.detectors_of_round(index)
        counts.append(len(detectors))
    return counts


def declared_rounds(measurement_count, checks_per_round, rounds) -> dict:
    """Measurement index -> the round its block of checks belongs to."""
    measurement_rounds = {}
    for index in range(measurement_count):
        announced_round = index // checks_per_round + 1
        measurement_rounds[index] = min(announced_round, rounds)
    return measurement_rounds


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
    """A Z-first memory measures Z then X ancillas as two blocks per round.

    The circuit is a Deltakit rotated surface-code memory at d=3 with the
    Z-first schedule (tests/data/surface_memory_z_first.stim). The map
    declares one round per stabiliser round, so both blocks share one
    packet, Z bits first, and the last packet carries the data readout
    after its own ancilla bits.
    """
    circuit_path = DATA_DIRECTORY / "surface_memory_z_first.stim"
    circuit = stim.Circuit.from_file(circuit_path)
    rounds = 8
    checks_per_round = 8
    measurement_rounds = declared_rounds(
        circuit.num_measurements, checks_per_round, rounds
    )

    table = detector_formation.build_formation_table(
        circuit, rounds, measurement_rounds=measurement_rounds
    )

    widths = [table.packet_width_by_round[index] for index in range(1, 9)]
    round_detector_counts = detector_counts(table, 8)
    first_round_recipes = table.detectors_of_round(1)
    first_round_records = {
        record for recipe in first_round_recipes for record in recipe.records
    }
    assert widths == [8] * 7 + [17]
    assert table.readout_slot_start is None
    assert round_detector_counts == [4] + [8] * 6 + [12]
    assert len(table.observables) == 1
    assert first_round_records == {(1, 0), (1, 1), (1, 2), (1, 3)}

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
    round_detector_counts = detector_counts(table, 12)
    assert table.readout_slot_start is None
    assert widths == [16, 16, 16, 28, 28, 31, 28, 28, 40, 16, 16, 34]
    assert round_detector_counts == [
        8,
        16,
        16,
        20,
        28,
        28,
        24,
        28,
        32,
        16,
        16,
        24,
    ]
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


def test_a_round_done_ahead_of_an_earlier_one_is_listed_above_the_mark():
    done_rounds = detector_formation.DoneRounds()

    done_rounds.add(1)
    done_rounds.add(3)

    assert done_rounds.through == 1
    assert done_rounds.above == {3}
    assert 3 in done_rounds
    assert 2 not in done_rounds


def test_the_round_that_closes_a_gap_moves_the_mark_over_the_rounds_after():
    done_rounds = detector_formation.DoneRounds()
    done_rounds.add(3)
    done_rounds.add(2)

    done_rounds.add(1)

    assert done_rounds.through == 3
    assert done_rounds.above == set()


def test_a_round_marked_not_done_below_the_mark_reopens_only_itself():
    done_rounds = detector_formation.DoneRounds()
    done_rounds.add(1)
    done_rounds.add(2)
    done_rounds.add(3)

    done_rounds.discard(2)

    assert done_rounds.through == 1
    assert done_rounds.above == {3}


def _first_round_observed_circuit() -> stim.Circuit:
    """Three one-bit rounds; the observable reads rounds 1 and 3."""
    return stim.Circuit(
        "R 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
        "M 0\nDETECTOR rec[-1]\nM 0\nDETECTOR rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]"
    )


def test_a_read_back_past_rounds_that_measure_nothing_is_refused():
    """After empty rounds a record lies further back every round."""
    reads_before = stim.Circuit("DETECTOR rec[-1]")

    with pytest.raises(ValueError) as refusal:
        detector_formation.rounds_read_back(reads_before, 0)

    assert str(refusal.value) == (
        "a detector reads a record from before its round, but the "
        "repeated round measures nothing, so the record lies a round "
        "further back every round and no stream can keep it; give the "
        "repeated round a measurement"
    )
