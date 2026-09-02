"""The Stim device emits the bits Stim's circuit measures, round by round.

Sources: Stim's generated ``surface_code:rotated_memory_z`` circuit
(src/stim/gen/gen_surface_code.cc: MR on every measure qubit each round,
M on every data qubit at the end, so a distance-d round carries d*d - 1
bits and the final round d*d more); Stim's measurement-to-detector
converter (stim.CompiledMeasurementsToDetectionEventsConverter) as the
oracle for detection events and observable flips; validation matrix row
Q4.
"""

import pytest

from decsim.message import Operation
from decsim.qpu.stim_device import RecordedStimDevice, StimDevice

stim = pytest.importorskip("stim")


def memory_circuit(distance, rounds):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.001,
        before_round_data_depolarization=0.001,
        before_measure_flip_probability=0.001,
        after_reset_flip_probability=0.001,
    )


def memory_operation(circuit, operation_id=1, **changes):
    return Operation(
        id=operation_id,
        name="memory",
        qubits=(0,),
        patches=(0,),
        circuit=circuit,
        **changes,
    )


def round_payload(device, operation, round_index):
    payloads = device.round_payloads(operation, round_index)
    return payloads[0]


def test_a_distance_three_round_has_eight_bits_and_the_last_seventeen():
    circuit = memory_circuit(3, 3)
    device = StimDevice(seed=1)
    operation = memory_operation(circuit)
    device.begin_operation(operation, 3, 3)
    first = round_payload(device, operation, 1)
    second = round_payload(device, operation, 2)
    third = round_payload(device, operation, 3)
    assert (len(first.bits), len(second.bits), len(third.bits)) == (8, 8, 17)
    assert (first.size_bits, third.size_bits) == (8, 17)
    assert circuit.num_measurements == 33


def test_a_distance_five_round_has_24_bits_and_the_last_49():
    circuit = memory_circuit(5, 3)
    device = StimDevice(seed=1)
    operation = memory_operation(circuit)
    device.begin_operation(operation, 3, 3)
    first = round_payload(device, operation, 1)
    third = round_payload(device, operation, 3)
    assert (len(first.bits), len(third.bits)) == (24, 49)
    assert circuit.num_measurements == 97


def test_a_replayed_shot_forms_the_events_and_truth_stim_forms():
    circuit = memory_circuit(3, 4)
    sampler = circuit.compile_sampler(seed=3)
    measurements = sampler.sample(shots=2)
    converter = circuit.compile_m2d_converter()
    events, observables = converter.convert(
        measurements=measurements, separate_observables=True
    )
    device = RecordedStimDevice(measurements, 1)
    operation = memory_operation(circuit)
    device.begin_operation(operation, 4, 4)
    expected_events = tuple(bool(bit) for bit in events[1])
    expected_truth = tuple(int(bit) for bit in observables[1])
    assert device.sampled_detection_events(1) == expected_events
    assert device.logical_observable_truth(1) == expected_truth
    second_round = round_payload(device, operation, 2)
    expected_bits = tuple(int(bit) for bit in measurements[1][8:16])
    assert tuple(second_round.bits) == expected_bits


def test_the_same_seed_samples_the_same_shot_in_another_device():
    circuit = memory_circuit(3, 3)
    operation = memory_operation(circuit)
    first = StimDevice(seed=7)
    second = StimDevice(seed=7)
    first.begin_operation(operation, 3, 3)
    second.begin_operation(operation, 3, 3)
    assert first.sampled_truth() == second.sampled_truth()
    first_round = round_payload(first, operation, 1)
    second_round = round_payload(second, operation, 1)
    assert first_round.bits == second_round.bits


def test_a_seeded_device_refuses_an_identity_it_cannot_hash_stably():
    circuit = memory_circuit(3, 3)
    operation = memory_operation(circuit, stream_id=("tuple",), stream_offset=0)
    device = StimDevice(seed=7)
    with pytest.raises(TypeError, match="int or str"):
        device.begin_operation(operation, 3, 3)


def test_a_later_stream_segment_reuses_the_streams_shot():
    circuit = memory_circuit(3, 6)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    tail = memory_operation(circuit, 2, stream_id="s", stream_offset=3)
    device = StimDevice(seed=5)
    device.begin_operation(head, 3, 6)
    device.begin_operation(tail, 3, 6)
    tail_events = device.sampled_detection_events(2)
    stream_events = device.sampled_detection_events("s")
    assert tail_events == stream_events
    fourth = round_payload(device, tail, 1)
    assert fourth.round_index == 4
    assert fourth.operation_id == "s"


def test_a_declared_terminal_fragment_holds_back_the_data_readout():
    circuit = memory_circuit(3, 3)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    finalizer = memory_operation(circuit, 2, stream_id="s", stream_offset=2)
    device = StimDevice(seed=5, terminal_detector_ids={"s": (16,)})
    device.begin_operation(head, 3, 3)
    last_round = round_payload(device, head, 3)
    readouts = device.finalize_stream_round(finalizer, 3)
    readout = readouts[0]
    assert len(last_round.bits) == 8
    assert len(readout.bits) == 9
    assert (readout.operation_id, readout.round_index) == ("s", 3)


def test_a_segment_past_the_end_of_its_source_is_refused():
    circuit = memory_circuit(3, 3)
    operation = memory_operation(circuit, 1, stream_id="s", stream_offset=2)
    device = StimDevice(seed=5)
    with pytest.raises(ValueError, match="beyond its finite source"):
        device.begin_operation(operation, 2, 3)


def test_an_operation_without_a_circuit_is_refused():
    device = StimDevice(seed=5)
    operation = Operation(id=1, name="memory", qubits=(0,))
    with pytest.raises(ValueError, match="require a circuit"):
        device.begin_operation(operation, 3, 3)
