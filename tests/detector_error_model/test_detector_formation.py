"""Streaming detector formation equals Stim's measurement-to-detector rule.

Source: stim.Circuit.compile_m2d_converter (measurements_to_detection_events):
a detector is the XOR of the measurement records it names, compared
against the noiseless reference sample; an observable is the XOR of its
records the same way. The packet layout follows Stim's generated
circuits: one measurement block per round, and the final data readout
folded into the last round's packet after that round's own bits.
"""

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


def test_a_declared_detector_round_outside_the_operation_is_refused():
    circuit = surface_code_circuit(4)
    in_time = detector_chronology.resolve_detector_rounds(circuit, None, 4)
    # Stim's last coordinate puts detectors 20..31 in round 4; declared
    # in round 7 they would never be formed.
    assert in_time[20] == 4
    assert in_time[31] == 4
    readout_past_the_end = dict.fromkeys(range(20, 32), 7)
    past_the_end = dict(in_time)
    past_the_end.update(readout_past_the_end)
    with pytest.raises(ValueError, match="inside the emitted rounds"):
        detector_formation.build_formation_table(
            circuit, 4, detector_rounds=past_the_end
        )
