"""The Stim device emits the bits Stim's circuit measures, round by round.

Sources: Stim's generated ``surface_code:rotated_memory_z`` circuit
(src/stim/gen/gen_surface_code.cc: MR on every measure qubit each round,
M on every data qubit at the end, so a distance-3 round carries 8 bits
and the final round 9 more; the detectors of a 4-round circuit are 0-3 in
round 1, 4-11 in round 2, 12-19 in round 3 and 20-31 in round 4); Stim's
measurement-to-detector converter
(stim.CompiledMeasurementsToDetectionEventsConverter) as the oracle for
detection events and observable flips; Stim's compile_sampler(seed=...)
as the oracle for a seeded shot; validation matrix row Q4.
"""

import hashlib

import numpy
import pytest

import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.message as message
import decsim.qpu.stim_device as stim_device
import decsim.records.rounds as round_records
import decsim.records.windows as window_records

stim = pytest.importorskip("stim")

GRAPHLIKE_REPRESENTATION = fault_models.FaultRepresentation.GRAPHLIKE
GRAPHLIKE_REPRESENTATIONS = frozenset({GRAPHLIKE_REPRESENTATION})
GRAPHLIKE = fault_models.DecoderFaultModelRequirement(GRAPHLIKE_REPRESENTATIONS)

# One 41-bit shot of the 4-round distance-3 memory: 8 measure bits per
# round, then the 9 data qubits.
RECORDED_ROW = (
    0, 0, 1, 0, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 1,
)  # fmt: skip
RECORDED_EVENTS = (
    0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0,
)  # fmt: skip

# One 33-bit shot of the 3-round distance-3 memory.
THREE_ROUND_ROW = (
    0, 1, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 1,
    0, 1, 0, 0, 0, 0, 0, 1, 0,
)  # fmt: skip


def memory_circuit(distance, rounds, noise=0.001):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise,
        before_round_data_depolarization=noise,
        before_measure_flip_probability=noise,
        after_reset_flip_probability=noise,
    )


def memory_operation(circuit, operation_id=1, **changes):
    return message.Operation(
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


def recorded_device(row, **settings):
    measurements = numpy.array([row], dtype=bool)
    return stim_device.RecordedStimDevice(measurements, 0, **settings)


def window(commit_lo, commit_hi, buffer_hi, **changes):
    return window_records.Window(
        op_id=1,
        k=0,
        commit_lo=commit_lo,
        commit_hi=commit_hi,
        buffer_hi=buffer_hi,
        n_rounds=buffer_hi,
        **changes,
    )


def test_a_distance_three_round_has_eight_bits_and_the_last_seventeen():
    circuit = memory_circuit(3, 3)
    device = stim_device.StimDevice(seed=1)
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
    device = stim_device.StimDevice(seed=1)
    operation = memory_operation(circuit)
    device.begin_operation(operation, 3, 3)
    first = round_payload(device, operation, 1)
    third = round_payload(device, operation, 3)
    assert (len(first.bits), len(third.bits)) == (24, 49)
    assert circuit.num_measurements == 97


def test_a_replayed_shot_is_cut_into_the_rounds_it_was_measured_in():
    circuit = memory_circuit(3, 4)
    device = recorded_device(RECORDED_ROW)
    operation = memory_operation(circuit)
    device.begin_operation(operation, 4, 4)
    first = round_payload(device, operation, 1)
    second = round_payload(device, operation, 2)
    fourth = round_payload(device, operation, 4)
    assert first.bits == (0, 0, 1, 0, 0, 0, 0, 0)
    assert second.bits == (0, 1, 1, 0, 0, 0, 0, 0)
    assert fourth.bits == (
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 1,
    )  # fmt: skip
    assert fourth.size_bits == 17


def test_a_replayed_shot_forms_the_events_and_truth_stim_forms():
    circuit = memory_circuit(3, 4)
    measurements = numpy.array([RECORDED_ROW], dtype=bool)
    converter = circuit.compile_m2d_converter()
    events, observables = converter.convert(
        measurements=measurements, separate_observables=True
    )
    device = recorded_device(RECORDED_ROW)
    operation = memory_operation(circuit)
    device.begin_operation(operation, 4, 4)
    oracle_events = events[0].tolist()
    assert oracle_events == list(RECORDED_EVENTS)
    assert observables[0].tolist() == [True]
    assert device.sampled_detection_events(1) == tuple(oracle_events)
    assert device.logical_observable_truth(1) == (1,)
    assert device.sampled_truth() == {1: (1,)}


def test_form_round_yields_each_rounds_slice_of_the_shots_events():
    circuit = memory_circuit(3, 4)
    device = recorded_device(RECORDED_ROW)
    operation = memory_operation(circuit)
    device.begin_operation(operation, 4, 4)
    first = round_payload(device, operation, 1)
    second = round_payload(device, operation, 2)
    third = round_payload(device, operation, 3)
    fourth = round_payload(device, operation, 4)
    assert device.form_round(1, 1, first.bits) == (0, 0, 0, 0)
    assert device.form_round(1, 2, second.bits) == (0, 1, 0, 0, 0, 0, 0, 0)
    assert device.form_round(1, 3, third.bits) == (0, 0, 1, 0, 0, 0, 0, 0)
    assert device.form_round(1, 4, fourth.bits) == (
        0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0,
    )  # fmt: skip
    assert device.sampled_detection_events(1) == (
        False, False, False, False,
        False, True, False, False, False, False, False, False,
        False, False, True, False, False, False, False, False,
        False, True, True, False, False, False, False, False,
        False, True, True, False,
    )  # fmt: skip


def test_a_seeded_shot_is_stims_shot_under_the_hashed_substream_seed():
    """The device's shot is Stim's shot under the substream seed.

    The substream seed's blake2b framing (root seed, "stim_device", the
    key's type tag and the key, NUL-separated, into an 8-byte big-endian
    digest) is the pinned cross-process contract; the law is that the
    device samples exactly what Stim's compiled sampler samples under
    that seed.
    """
    circuit = memory_circuit(3, 3)
    hasher = hashlib.blake2b(b"7\x00stim_device\x00int\x001", digest_size=8)
    digest = hasher.digest()
    substream_seed = int.from_bytes(digest, "big")
    assert substream_seed == 5762610574057409091
    sampler = circuit.compile_sampler(seed=substream_seed)
    shots = sampler.sample(1)
    oracle_row = shots[0].tolist()
    device = stim_device.StimDevice(seed=7)
    operation = memory_operation(circuit, 1)
    device.begin_operation(operation, 3, 3)
    first = round_payload(device, operation, 1)
    second = round_payload(device, operation, 2)
    third = round_payload(device, operation, 3)
    assert oracle_row == [
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1,
    ]  # fmt: skip
    assert first.bits == (0, 0, 0, 0, 0, 0, 0, 0)
    assert second.bits == (0, 0, 0, 0, 0, 0, 0, 0)
    assert third.bits == (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1)


def test_the_same_seed_samples_the_same_shot_in_another_device():
    circuit = memory_circuit(3, 3)
    operation = memory_operation(circuit)
    first = stim_device.StimDevice(seed=7)
    second = stim_device.StimDevice(seed=7)
    first.begin_operation(operation, 3, 3)
    second.begin_operation(operation, 3, 3)
    first_round = round_payload(first, operation, 3)
    second_round = round_payload(second, operation, 3)
    assert first_round.bits == (
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1,
    )  # fmt: skip
    assert second_round.bits == first_round.bits


def test_another_root_seed_samples_another_shot():
    circuit = memory_circuit(3, 3)
    operation = memory_operation(circuit)
    device = stim_device.StimDevice(seed=8)
    device.begin_operation(operation, 3, 3)
    first_round = round_payload(device, operation, 1)
    assert first_round.bits == (0, 0, 0, 0, 0, 1, 0, 1)


def test_another_sample_key_under_the_same_seed_samples_another_shot():
    circuit = memory_circuit(3, 3)
    operation = memory_operation(circuit, 2)
    device = stim_device.StimDevice(seed=7)
    device.begin_operation(operation, 3, 3)
    first_round = round_payload(device, operation, 1)
    assert first_round.bits == (1, 0, 0, 0, 0, 1, 0, 1)


def test_a_seed_outside_stims_64_bit_range_is_refused():
    too_large = 1 << 64
    with pytest.raises(ValueError, match="64-bit unsigned"):
        stim_device.StimDevice(seed=too_large)


def test_a_seed_that_is_not_an_integer_is_refused():
    with pytest.raises(ValueError, match="64-bit unsigned"):
        stim_device.StimDevice(seed="7")


def test_a_run_bound_seed_samples_stims_shot_under_that_root():
    circuit = memory_circuit(3, 3)
    operation = memory_operation(circuit)
    device = stim_device.StimDevice(seed=None)
    reservation = device.reserve_run_seed(3)
    device.commit_run_seed(reservation)
    device.begin_operation(operation, 3, 3)
    first_round = round_payload(device, operation, 1)
    assert first_round.bits == (0, 0, 0, 0, 0, 1, 0, 0)


def test_a_seeded_device_refuses_an_identity_it_cannot_hash_stably():
    circuit = memory_circuit(3, 3)
    operation = memory_operation(circuit, stream_id=("tuple",), stream_offset=0)
    device = stim_device.StimDevice(seed=7)
    with pytest.raises(TypeError, match="int or str"):
        device.begin_operation(operation, 3, 3)


def test_a_later_stream_segment_reuses_the_streams_shot():
    circuit = memory_circuit(3, 6)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    tail = memory_operation(circuit, 2, stream_id="s", stream_offset=3)
    device = stim_device.StimDevice(seed=5)
    device.begin_operation(head, 3, 6)
    head_events = device.sampled_detection_events(1)
    head_fourth = round_payload(device, head, 4)
    device.begin_operation(tail, 3, 6)
    assert device.sampled_detection_events(2) == head_events
    assert device.sampled_detection_events("s") == head_events
    tail_first = round_payload(device, tail, 1)
    assert tail_first.bits == head_fourth.bits
    assert tail_first.round_index == 4
    assert tail_first.operation_id == "s"


def test_a_segments_observable_truth_is_its_streams_truth():
    circuit = memory_circuit(3, 6)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    tail = memory_operation(circuit, 2, stream_id="s", stream_offset=3)
    device = stim_device.StimDevice(seed=5)
    device.begin_operation(head, 3, 6)
    device.begin_operation(tail, 3, 6)
    assert device.logical_observable_truth("s") == (0,)
    assert device.logical_observable_truth(1) == (0,)
    assert device.logical_observable_truth(2) == (0,)
    assert device.logical_observable_truth(3) is None
    assert device.sampled_truth() == {"s": (0,), 1: (0,), 2: (0,)}


def test_a_stream_whose_id_equals_a_segments_operation_id_samples_afresh():
    # Stream "a" replays its shot under operation ids 1 and 2; a stream
    # whose id is 2 is another identity with its own substream, so its
    # first round is what a device that only ever saw it samples.
    six_rounds = memory_circuit(3, 6)
    three_rounds = memory_circuit(3, 3)
    head = memory_operation(six_rounds, 1, stream_id="a", stream_offset=0)
    tail = memory_operation(six_rounds, 2, stream_id="a", stream_offset=3)
    other = memory_operation(three_rounds, 3, stream_id=2, stream_offset=0)
    device = stim_device.StimDevice(seed=0)
    device.begin_operation(head, 3, 6)
    device.begin_operation(tail, 3, 6)
    device.begin_operation(other, 3, 3)
    alone = stim_device.StimDevice(seed=0)
    alone.begin_operation(other, 3, 3)
    other_first = round_payload(device, other, 1)
    alone_first = round_payload(alone, other, 1)
    assert other_first.bits == (1, 0, 0, 0, 0, 1, 0, 0)
    assert alone_first.bits == (1, 0, 0, 0, 0, 1, 0, 0)


def test_a_later_segment_with_another_source_duration_is_refused():
    # Seven rounds is a valid chronology for the six-round circuit, so
    # the binding's duration is what refuses the segment.
    circuit = memory_circuit(3, 6)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    tail = memory_operation(circuit, 2, stream_id="s", stream_offset=1)
    device = stim_device.StimDevice(seed=5)
    device.begin_operation(head, 1, 6)
    with pytest.raises(ValueError, match="source duration differs"):
        device.begin_operation(tail, 1, 7)


def test_a_segment_past_the_end_of_its_source_is_refused():
    circuit = memory_circuit(3, 3)
    operation = memory_operation(circuit, 1, stream_id="s", stream_offset=2)
    device = stim_device.StimDevice(seed=5)
    with pytest.raises(ValueError, match="beyond its finite source"):
        device.begin_operation(operation, 2, 3)


def test_a_standalone_operation_runs_for_its_whole_source():
    circuit = memory_circuit(3, 3)
    operation = memory_operation(circuit)
    device = stim_device.StimDevice(seed=5)
    with pytest.raises(ValueError, match="standalone duration must equal"):
        device.begin_operation(operation, 2, 3)


def test_a_stream_keeps_the_circuit_it_was_bound_to():
    circuit = memory_circuit(3, 3)
    noisier_circuit = memory_circuit(3, 3, noise=0.002)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    tail = memory_operation(noisier_circuit, 2, stream_id="s", stream_offset=1)
    device = stim_device.StimDevice(seed=5)
    device.begin_operation(head, 1, 3)
    with pytest.raises(ValueError, match="circuit differs from the bound"):
        device.begin_operation(tail, 1, 3)


def test_an_operation_without_a_circuit_is_refused():
    device = stim_device.StimDevice(seed=5)
    operation = message.Operation(id=1, name="memory", qubits=(0,))
    with pytest.raises(ValueError, match="require a circuit"):
        device.begin_operation(operation, 3, 3)


def test_a_declared_terminal_fragment_holds_back_the_data_readout():
    circuit = memory_circuit(3, 3)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    finalizer = memory_operation(circuit, 2, stream_id="s", stream_offset=2)
    device = recorded_device(
        THREE_ROUND_ROW, terminal_detector_ids={"s": (20,)}
    )
    device.begin_operation(head, 3, 3)
    last_round = round_payload(device, head, 3)
    readouts = device.finalize_stream_round(finalizer, 3)
    assert last_round.bits == (0, 0, 0, 0, 1, 0, 0, 1)
    assert readouts == [
        round_records.QPUReadout(
            "s", 0, 3, bits=(0, 1, 0, 0, 0, 0, 0, 1, 0), size_bits=9
        )
    ]


def test_an_idle_stream_round_replays_the_shots_packet_of_that_round():
    circuit = memory_circuit(3, 3)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    device = recorded_device(THREE_ROUND_ROW)
    device.begin_operation(head, 3, 3)
    payloads = device.idle_round_payloads(head, "s", 2, 0)
    assert payloads == [
        round_records.QPUReadout(
            "s", 0, 2, bits=(0, 0, 0, 0, 1, 0, 0, 0), size_bits=8
        )
    ]


def test_an_idle_round_outside_the_finite_source_is_refused():
    circuit = memory_circuit(3, 3)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    device = recorded_device(THREE_ROUND_ROW)
    device.begin_operation(head, 3, 3)
    with pytest.raises(ValueError, match="outside the finite source"):
        device.idle_round_payloads(head, "s", 4, 0)


def test_a_finalizer_with_another_circuit_is_refused():
    circuit = memory_circuit(3, 3)
    noisier_circuit = memory_circuit(3, 3, noise=0.002)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    finalizer = memory_operation(
        noisier_circuit, 2, stream_id="s", stream_offset=2
    )
    device = recorded_device(
        THREE_ROUND_ROW, terminal_detector_ids={"s": (20,)}
    )
    device.begin_operation(head, 3, 3)
    with pytest.raises(RuntimeError, match="circuit differs"):
        device.finalize_stream_round(finalizer, 3)


def test_a_finalizer_before_the_final_round_is_refused():
    circuit = memory_circuit(3, 3)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    finalizer = memory_operation(circuit, 2, stream_id="s", stream_offset=1)
    device = recorded_device(
        THREE_ROUND_ROW, terminal_detector_ids={"s": (20,)}
    )
    device.begin_operation(head, 3, 3)
    with pytest.raises(RuntimeError, match="not at the final source round"):
        device.finalize_stream_round(finalizer, 3)


def test_a_finalizer_without_declared_terminal_detectors_is_refused():
    circuit = memory_circuit(3, 3)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    finalizer = memory_operation(circuit, 2, stream_id="s", stream_offset=2)
    device = recorded_device(THREE_ROUND_ROW)
    device.begin_operation(head, 3, 3)
    with pytest.raises(RuntimeError, match="no declared detector ids"):
        device.finalize_stream_round(finalizer, 3)


def test_a_finalizer_without_folded_readout_bits_is_refused():
    # Declared measurement rounds place every bit in rounds 1 to 3, so no
    # readout is folded into the last packet.
    circuit = memory_circuit(3, 3)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    finalizer = memory_operation(circuit, 2, stream_id="s", stream_offset=2)
    measurement_rounds = dict.fromkeys(range(0, 8), 1)
    second_round = dict.fromkeys(range(8, 16), 2)
    third_round = dict.fromkeys(range(16, 33), 3)
    measurement_rounds.update(second_round)
    measurement_rounds.update(third_round)
    device = recorded_device(
        THREE_ROUND_ROW,
        terminal_detector_ids={"s": (20,)},
        measurement_rounds={"s": measurement_rounds},
    )
    device.begin_operation(head, 3, 3)
    with pytest.raises(RuntimeError, match="no folded readout bits"):
        device.finalize_stream_round(finalizer, 3)


def test_a_registered_stream_is_as_long_as_its_circuit():
    circuit = memory_circuit(3, 4)
    stream = memory_operation(circuit, 5)
    device = stim_device.StimDevice(seed=1)
    registered = device.register_dynamic_stream(
        stream, 4, fault_model_requirement=GRAPHLIKE
    )
    assert registered == 4
    device.validate_stream_length(stream, 4)


def test_a_stream_sealed_at_another_length_is_refused():
    circuit = memory_circuit(3, 4)
    stream = memory_operation(circuit, 5)
    device = stim_device.StimDevice(seed=1)
    device.register_dynamic_stream(stream, 4, fault_model_requirement=GRAPHLIKE)
    with pytest.raises(RuntimeError, match="sealed at 5 rounds"):
        device.validate_stream_length(stream, 5)


def test_a_stream_window_reads_the_detectors_of_its_buffer_rounds():
    circuit = memory_circuit(3, 4)
    stream = memory_operation(circuit, 5)
    device = stim_device.StimDevice(seed=1)
    device.register_dynamic_stream(stream, 4, fault_model_requirement=GRAPHLIKE)
    leading = window(1, 2, 3)
    model = device.window_model_for_stream(5, leading)
    assert model.detector_ids == tuple(range(0, 20))


def test_the_terminal_stream_window_reads_to_the_last_round():
    # The window whose commit region reaches round 4 is the terminal one;
    # the window manager clamps its buffer to the last round before the
    # call, so it reads rounds 3 and 4 and owns everything it sees.
    circuit = memory_circuit(3, 4)
    stream = memory_operation(circuit, 5)
    device = stim_device.StimDevice(seed=1)
    device.register_dynamic_stream(stream, 4, fault_model_requirement=GRAPHLIKE)
    terminal = window(3, 4, 4)
    model = device.window_model_for_stream(5, terminal)
    assert model.detector_ids == tuple(range(12, 32))


def test_an_unregistered_stream_has_no_window_model():
    device = stim_device.StimDevice(seed=1)
    leading = window(1, 2, 3)
    assert device.window_model_for_stream(5, leading) is None


def test_dependent_windows_split_the_fault_ownership_between_them():
    circuit = memory_circuit(3, 4)
    operation = memory_operation(circuit)
    leading = window(1, 2, 3)
    trailing = window_records.Window(
        op_id=1,
        k=1,
        commit_lo=3,
        commit_hi=4,
        buffer_hi=4,
        n_rounds=4,
        buffer_lo=1,
        closed_temporal_boundaries=True,
        deps=[(1, 0)],
    )
    device = stim_device.StimDevice(seed=1)
    models = device.window_models_for_operation(
        operation,
        [leading, trailing],
        4,
        fault_model_requirement=GRAPHLIKE,
        fault_exclusion_ranges=(),
        window_protocol=window_records.WindowProtocol.GENERIC,
    )
    leading_faults = models[0].require_faults(
        fault_models.FaultRepresentation.GRAPHLIKE
    )
    trailing_faults = models[1].require_faults(
        fault_models.FaultRepresentation.GRAPHLIKE
    )
    assert models[0].detector_ids == tuple(range(0, 20))
    assert models[1].detector_ids == tuple(range(0, 32))
    assert numpy.count_nonzero(leading_faults.owned) == 48
    assert numpy.count_nonzero(trailing_faults.owned) == 62
    assert len(leading_faults.source_fault_ids) == 80
    assert len(trailing_faults.source_fault_ids) == 62


def test_a_closed_boundary_needs_a_dependency_edge():
    # One window commits every round, so every fault has one owner; its
    # closed boundary is the plan's only defect.
    circuit = memory_circuit(3, 4)
    operation = memory_operation(circuit)
    closed = window_records.Window(
        op_id=1,
        k=0,
        commit_lo=1,
        commit_hi=4,
        buffer_hi=4,
        n_rounds=4,
        closed_temporal_boundaries=True,
    )
    device = stim_device.StimDevice(seed=1)
    with pytest.raises(ValueError, match="must be a dependency destination"):
        device.window_models_for_operation(
            operation,
            [closed],
            4,
            fault_model_requirement=GRAPHLIKE,
            fault_exclusion_ranges=(),
            window_protocol=window_records.WindowProtocol.GENERIC,
        )


def test_a_window_declared_past_the_source_reads_to_its_last_round():
    circuit = memory_circuit(3, 4)
    operation = memory_operation(circuit)
    past_the_end = window_records.Window(
        op_id=1, k=0, commit_lo=3, commit_hi=4, buffer_hi=9, n_rounds=4
    )
    device = stim_device.StimDevice(seed=1)
    model = device.strong_window_model_for_operation(
        operation, past_the_end, 4, fault_model_requirement=GRAPHLIKE
    )
    assert model.detector_ids == tuple(range(12, 32))


def test_a_hardware_detector_belongs_to_its_latest_layer_up_to_the_end():
    circuit = stim.Circuit(
        """
        DETECTOR(1, 1, 0) rec[-1]
        DETECTOR(1, 1, 1, 1, 1, 0) rec[-1]
        DETECTOR(2, 2, 3, 2, 2, 2) rec[-1]
        DETECTOR(2, 2, 3, 3, 3, 3, 4, 4, 2) rec[-1]
        """
    )
    rounds = stim_device.RecordedStimDevice.detector_rounds_from_coordinates(
        circuit, 3
    )
    assert rounds == {0: 1, 1: 2, 2: 3, 3: 3}
