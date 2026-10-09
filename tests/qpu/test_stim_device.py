"""The Stim device emits the bits Stim's circuit measures, round by round.

Sources: Stim's generated ``surface_code:rotated_memory_z`` circuit
(src/stim/gen/gen_surface_code.cc: MR on every measure qubit each round,
M on every data qubit at the end, so a distance-3 round carries 8 bits
and the final round 9 more; the detectors of a 4-round circuit are 0-3 in
round 1, 4-11 in round 2, 12-19 in round 3 and 20-31 in round 4); Stim's
measurement-to-detector converter
(stim.CompiledMeasurementsToDetectionEventsConverter) as the oracle for
detection events and observable flips; Stim's compile_sampler(seed=...)
as the oracle for a seeded shot; validation matrix row Q4. The burst
source against the public qec-burst-scaling code
(github.com/AlanPai777/qec-burst-scaling, qecburst/circuit.py
inject_burst_profile and exponential_decay_profile, written out below)
and against the places Stim's generator puts its four noise parameters
(src/stim/gen/circuit_gen_params.cc).
"""

import collections
import hashlib
import math

import numpy
import pytest

import decsim.detector_error_model.detector_formation as detector_formation
import decsim.qpu.stim_device as stim_device
import decsim.records.fault_model_contracts as fault_models
import decsim.records.program as program_records
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
    return program_records.Operation(
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


class _HeardErrors:
    """The fired errors each operation's shot reported."""

    def __init__(self) -> None:
        self.errors_by_operation = {}

    def errors_sampled(self, operation, fired_errors) -> None:
        self.errors_by_operation[operation.id] = fired_errors


def recorded_device(row, **settings):
    measurements = numpy.array([row], dtype=bool)
    return stim_device.RecordedStimDevice(measurements, 0, **settings)


def window(commit_lo, commit_hi, buffer_hi, **changes):
    return window_records.Window(
        operation_id=1,
        window_index=0,
        commit_lo=commit_lo,
        commit_hi=commit_hi,
        buffer_hi=buffer_hi,
        round_count=buffer_hi,
        **changes,
    )


def test_a_replayed_shot_forms_the_events_and_truth_stim_forms():
    circuit = memory_circuit(3, 4)
    measurements = numpy.array([RECORDED_ROW], dtype=bool)
    converter = circuit.compile_m2d_converter()
    events, observables = converter.convert(
        measurements=measurements, separate_observables=True
    )
    device = recorded_device(RECORDED_ROW)
    operation = memory_operation(circuit)
    device.begin_operation(operation, 4, 4, round_period_ticks=1_100_000)
    oracle_events = events[0].tolist()
    assert oracle_events == list(RECORDED_EVENTS)
    assert observables[0].tolist() == [True]
    assert device.sampled_detection_events(1) == tuple(oracle_events)
    assert device.logical_observable_truth(1) == (1,)
    assert device.sampled_truth() == {1: (1,)}


def test_the_formation_table_forms_each_rounds_slice_of_the_shots_events():
    circuit = memory_circuit(3, 4)
    device = recorded_device(RECORDED_ROW)
    operation = memory_operation(circuit)
    device.begin_operation(operation, 4, 4, round_period_ticks=1_100_000)
    table = device.formation_table(1)
    former = detector_formation.StreamingDetectorFormer(table)
    first = round_payload(device, operation, 1)
    second = round_payload(device, operation, 2)
    third = round_payload(device, operation, 3)
    fourth = round_payload(device, operation, 4)
    assert _formed(former, 1, first.bits) == (0, 0, 0, 0)
    assert _formed(former, 2, second.bits) == (0, 1, 0, 0, 0, 0, 0, 0)
    assert _formed(former, 3, third.bits) == (0, 0, 1, 0, 0, 0, 0, 0)
    assert _formed(former, 4, fourth.bits) == (
        0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0,
    )  # fmt: skip
    assert device.sampled_detection_events(1) == (
        False, False, False, False,
        False, True, False, False, False, False, False, False,
        False, False, True, False, False, False, False, False,
        False, True, True, False, False, False, False, False,
        False, True, True, False,
    )  # fmt: skip


def _formed(former, round_index, bits) -> tuple:
    """One round's event values, in the table's detector order."""
    events = former.feed_packet(round_index, bits)
    return tuple(value for _, value in events)


def test_a_seeded_shot_is_stims_shot_under_the_hashed_substream_seed():
    """The device's shot is Stim's shot under the substream seed.

    The substream seed is the run's seed law written out: blake2b-8 over
    the namespace, the root seed as eight big-endian bytes, and the
    stream identity as an integer path segment (tag, four-byte length,
    decimal text). It is the pinned cross-process contract; the law is
    that the device samples exactly what Stim's compiled sampler samples
    under that seed.
    """
    circuit = memory_circuit(3, 3)
    root_bytes = (7).to_bytes(8, "big")
    identity_segment = b"I" + (1).to_bytes(4, "big") + b"1"
    preimage = b"decsim.run-seed.v1" + root_bytes + identity_segment
    hasher = hashlib.blake2b(preimage, digest_size=8)
    digest = hasher.digest()
    substream_seed = int.from_bytes(digest, "big")
    assert substream_seed == 7382560267478030810
    sampler = circuit.compile_sampler(seed=substream_seed)
    shots = sampler.sample(1)
    oracle_row = shots[0].tolist()
    device = stim_device.StimDevice(seed=7)
    operation = memory_operation(circuit, 1)
    device.begin_operation(operation, 3, 3, round_period_ticks=1_100_000)
    first = round_payload(device, operation, 1)
    second = round_payload(device, operation, 2)
    third = round_payload(device, operation, 3)
    assert oracle_row == [
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
    ]  # fmt: skip
    assert first.bits == (0, 0, 0, 0, 0, 0, 0, 0)
    assert second.bits == (0, 0, 0, 0, 0, 0, 0, 0)
    assert third.bits == (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1)


def test_an_error_model_shot_is_stims_error_model_sample_formed_back():
    """The emitted rows form the events and flips the model drew.

    The draw is Stim's CompiledDemSampler under the same substream seed
    the measurement sampler takes (7382560267478030810, pinned above),
    and the fired errors heard add up to the shot's truth.
    """
    circuit = memory_circuit(3, 3, noise=0.03)
    model = circuit.detector_error_model()
    sampler = model.compile_sampler(seed=7382560267478030810)
    events, flips, _ = sampler.sample(1)
    device = stim_device.ErrorModelStimDevice(seed=7)
    heard = _HeardErrors()
    device.errors_sampled.connect(heard.errors_sampled)
    operation = memory_operation(circuit, 1)
    device.begin_operation(operation, 3, 3, round_period_ticks=1_100_000)
    fired_flips = [0]
    for error in heard.errors_by_operation[1]:
        fired_flips[0] ^= error.logical_observables[0]
    drawn_events = events[0].tolist()
    assert device.sampled_detection_events(1) == tuple(drawn_events)
    assert device.logical_observable_truth(1) == (int(flips[0][0]),)
    assert tuple(fired_flips) == device.logical_observable_truth(1)
    assert any(events[0])


def test_a_seed_outside_stims_64_bit_range_is_refused():
    too_large = 1 << 64
    with pytest.raises(ValueError, match="64-bit unsigned"):
        stim_device.StimDevice(seed=too_large)


def test_a_seeded_device_refuses_an_identity_it_cannot_hash_stably():
    circuit = memory_circuit(3, 3)
    operation = memory_operation(circuit, stream_id=("tuple",), stream_offset=0)
    device = stim_device.StimDevice(seed=7)
    with pytest.raises(ValueError, match="int or str"):
        device.begin_operation(operation, 3, 3, round_period_ticks=1_100_000)


def test_a_later_stream_segment_reuses_the_streams_shot():
    circuit = memory_circuit(3, 6)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    tail = memory_operation(circuit, 2, stream_id="s", stream_offset=3)
    device = stim_device.StimDevice(seed=5)
    device.begin_operation(head, 3, 6, round_period_ticks=1_100_000)
    head_events = device.sampled_detection_events(1)
    head_fourth = round_payload(device, head, 4)
    device.begin_operation(tail, 3, 6, round_period_ticks=1_100_000)
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
    device.begin_operation(head, 3, 6, round_period_ticks=1_100_000)
    device.begin_operation(tail, 3, 6, round_period_ticks=1_100_000)
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
    device.begin_operation(head, 3, 6, round_period_ticks=1_100_000)
    device.begin_operation(tail, 3, 6, round_period_ticks=1_100_000)
    device.begin_operation(other, 3, 3, round_period_ticks=1_100_000)
    alone = stim_device.StimDevice(seed=0)
    alone.begin_operation(other, 3, 3, round_period_ticks=1_100_000)
    other_first = round_payload(device, other, 1)
    alone_first = round_payload(alone, other, 1)
    assert other_first.bits == (0, 0, 0, 0, 0, 0, 0, 0)
    assert alone_first.bits == (0, 0, 0, 0, 0, 0, 0, 0)


def test_a_later_segment_with_another_source_duration_is_refused():
    # Seven rounds is a valid chronology for the six-round circuit, so
    # the binding's duration is what refuses the segment.
    circuit = memory_circuit(3, 6)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    tail = memory_operation(circuit, 2, stream_id="s", stream_offset=1)
    device = stim_device.StimDevice(seed=5)
    device.begin_operation(head, 1, 6, round_period_ticks=1_100_000)
    with pytest.raises(ValueError, match="source duration differs"):
        device.begin_operation(tail, 1, 7, round_period_ticks=1_100_000)


def test_a_segment_past_the_end_of_its_source_is_refused():
    circuit = memory_circuit(3, 3)
    operation = memory_operation(circuit, 1, stream_id="s", stream_offset=2)
    device = stim_device.StimDevice(seed=5)
    with pytest.raises(ValueError, match="beyond its finite source"):
        device.begin_operation(operation, 2, 3, round_period_ticks=1_100_000)


def test_a_standalone_operation_runs_for_its_whole_source():
    circuit = memory_circuit(3, 3)
    operation = memory_operation(circuit)
    device = stim_device.StimDevice(seed=5)
    with pytest.raises(ValueError, match="standalone duration must equal"):
        device.begin_operation(operation, 2, 3, round_period_ticks=1_100_000)


def test_a_stream_keeps_the_circuit_it_was_bound_to():
    circuit = memory_circuit(3, 3)
    noisier_circuit = memory_circuit(3, 3, noise=0.002)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    tail = memory_operation(noisier_circuit, 2, stream_id="s", stream_offset=1)
    device = stim_device.StimDevice(seed=5)
    device.begin_operation(head, 1, 3, round_period_ticks=1_100_000)
    with pytest.raises(ValueError, match="circuit differs from the bound"):
        device.begin_operation(tail, 1, 3, round_period_ticks=1_100_000)


def test_an_operation_without_a_circuit_is_refused():
    device = stim_device.StimDevice(seed=5)
    operation = program_records.Operation(id=1, name="memory", qubits=(0,))
    with pytest.raises(ValueError, match="require a circuit"):
        device.begin_operation(operation, 3, 3, round_period_ticks=1_100_000)


def test_a_declared_terminal_fragment_holds_back_the_data_readout():
    circuit = memory_circuit(3, 3)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    finalizer = memory_operation(circuit, 2, stream_id="s", stream_offset=2)
    device = recorded_device(
        THREE_ROUND_ROW, terminal_detector_ids={"s": (20,)}
    )
    device.begin_operation(head, 3, 3, round_period_ticks=1_100_000)
    last_round = round_payload(device, head, 3)
    readouts = device.finalize_stream_round(finalizer, 3)
    assert last_round.bits == (0, 0, 0, 0, 1, 0, 0, 1)
    assert readouts == [
        round_records.QPUReadout(
            "s", (0,), 3, bits=(0, 1, 0, 0, 0, 0, 0, 1, 0), size_bits=9
        )
    ]


def test_an_idle_stream_round_replays_the_shots_packet_of_that_round():
    circuit = memory_circuit(3, 3)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    device = recorded_device(THREE_ROUND_ROW)
    device.begin_operation(head, 3, 3, round_period_ticks=1_100_000)
    payloads = device.idle_round_payloads(
        head, "s", 2, is_final=False, round_period_ticks=1_100_000
    )
    assert payloads == [
        round_records.QPUReadout(
            "s", (0,), 2, bits=(0, 0, 0, 0, 1, 0, 0, 0), size_bits=8
        )
    ]


def test_an_idle_round_outside_the_finite_source_is_refused():
    circuit = memory_circuit(3, 3)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    device = recorded_device(THREE_ROUND_ROW)
    device.begin_operation(head, 3, 3, round_period_ticks=1_100_000)
    with pytest.raises(ValueError, match="outside the finite source"):
        device.idle_round_payloads(
            head, "s", 4, is_final=False, round_period_ticks=1_100_000
        )


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
    device.begin_operation(head, 3, 3, round_period_ticks=1_100_000)
    with pytest.raises(RuntimeError, match="circuit differs"):
        device.finalize_stream_round(finalizer, 3)


def test_a_finalizer_before_the_final_round_is_refused():
    circuit = memory_circuit(3, 3)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    finalizer = memory_operation(circuit, 2, stream_id="s", stream_offset=1)
    device = recorded_device(
        THREE_ROUND_ROW, terminal_detector_ids={"s": (20,)}
    )
    device.begin_operation(head, 3, 3, round_period_ticks=1_100_000)
    with pytest.raises(RuntimeError, match="not at the final source round"):
        device.finalize_stream_round(finalizer, 3)


def test_a_finalizer_without_declared_terminal_detectors_is_refused():
    circuit = memory_circuit(3, 3)
    head = memory_operation(circuit, 1, stream_id="s", stream_offset=0)
    finalizer = memory_operation(circuit, 2, stream_id="s", stream_offset=2)
    device = recorded_device(THREE_ROUND_ROW)
    device.begin_operation(head, 3, 3, round_period_ticks=1_100_000)
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
    device.begin_operation(head, 3, 3, round_period_ticks=1_100_000)
    with pytest.raises(RuntimeError, match="no folded readout bits"):
        device.finalize_stream_round(finalizer, 3)


def test_a_stream_sealed_at_another_length_is_refused():
    circuit = memory_circuit(3, 4)
    stream = memory_operation(circuit, 5)
    device = stim_device.StimDevice(seed=1)
    device.declare_stream(stream, 4)
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


def test_a_closed_boundary_needs_a_dependency_edge():
    # One window commits every round, so every fault has one owner; its
    # closed boundary is the plan's only defect.
    circuit = memory_circuit(3, 4)
    operation = memory_operation(circuit)
    closed = window_records.Window(
        operation_id=1,
        window_index=0,
        commit_lo=1,
        commit_hi=4,
        buffer_hi=4,
        round_count=4,
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
        operation_id=1,
        window_index=0,
        commit_lo=3,
        commit_hi=4,
        buffer_hi=9,
        round_count=4,
    )
    device = stim_device.StimDevice(seed=1)
    model = device.strong_window_model_for_operation(
        operation, past_the_end, 4, fault_model_requirement=GRAPHLIKE
    )
    assert model.detector_ids == tuple(range(12, 32))


def qec_burst_scaling_circuit(circuit, qubits, start_round, probabilities):
    """qecburst/circuit.py inject_burst_profile, written out.

    One DEPOLARIZE1 per burst round on the chosen qubits, right after the
    round's first TICK, finding rounds as every seventh TICK of the
    flattened generated circuit; start_round counts from zero.
    """
    flattened = circuit.flattened()
    flattened_text = str(flattened)
    lines = flattened_text.splitlines()
    ticks = [index for index, line in enumerate(lines) if line == "TICK"]
    round_starts = ticks[::7]
    targets = " ".join(str(qubit) for qubit in qubits)
    by_offset = list(enumerate(probabilities))
    for offset, probability in reversed(by_offset):
        round_start = round_starts[start_round + offset]
        after_the_tick = round_start + 1
        inserted = f"DEPOLARIZE1({probability:.17g}) {targets}"
        lines.insert(after_the_tick, inserted)
    text = "\n".join(lines)
    return stim.Circuit(text)


def qec_burst_scaling_extras(background, peak, decay_rounds, round_count):
    """exponential_decay_profile, then _extra_depolarizing_probability.

    qecburst/circuit.py targets p0 + (p_peak - p0) exp(-t / tau) in burst
    round t and composes an extra DEPOLARIZE1(q) onto the round's own
    DEPOLARIZE1(p0): q = (target - p0) / (1 - 4 p0 / 3).
    """
    composition = 1 - 4 * background / 3
    extras = []
    for offset in range(round_count):
        exponent = -offset / decay_rounds
        decay = math.exp(exponent)
        target = background + (peak - background) * decay
        extra = (target - background) / composition
        extras.append(extra)
    return extras


def data_depolarizing_circuit(distance, rounds, noise):
    """qecburst/circuit.py build_background_circuit: round-start noise only."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=noise,
    )


def burst_of(circuit, rounds, **settings):
    table = detector_formation.build_formation_table(circuit, rounds)
    burst = stim_device.BurstStimDevice.Settings(**settings)
    return stim_device.burst_circuit(circuit, table, burst)


def inserted_lines(circuit, burst):
    flattened = circuit.flattened()
    plain_text = str(flattened)
    burst_text = str(burst)
    plain_lines = plain_text.splitlines()
    burst_lines = burst_text.splitlines()
    plain_counts = collections.Counter(plain_lines)
    burst_counts = collections.Counter(burst_lines)
    added = burst_counts - plain_counts
    added_lines = added.elements()
    return sorted(added_lines)


def test_an_idle_burst_is_qec_burst_scalings_decaying_burst():
    """The same circuit, region and schedule give the same circuit.

    qec-burst-scaling's extra probability in burst round t is
    (p_peak - p0) exp(-t / tau) / (1 - 4 p0 / 3), so
    burst_error_probability is its amplitude. Its disk of radius 2 about
    the middle data qubit of distance 3 holds data qubits 3, 8, 10, 12
    and 17 (qecburst/geometry.py select_disk).
    """
    circuit = data_depolarizing_circuit(3, 5, 0.001)
    extras = qec_burst_scaling_extras(0.001, 0.101, 1.5, 4)
    reference = qec_burst_scaling_circuit(
        circuit, (3, 8, 10, 12, 17), 1, extras
    )

    burst = burst_of(
        circuit,
        5,
        burst_onset_round=2,
        burst_decay_rounds=1.5,
        burst_radius=2.0,
        burst_error_probability=extras[0],
        burst_channels=("idle",),
    )

    assert burst.approx_equals(reference, atol=1e-12)


def test_a_rising_burst_climbs_linearly_then_decays_from_its_peak():
    """A third, two thirds, the peak, then exp(-k / tau) of it k rounds on.

    Willow's shape, a climb over three rounds and an exponential decay,
    drawn as qec-burst-scaling draws its idle burst, one round at a time.
    """
    circuit = data_depolarizing_circuit(3, 8, 0.001)
    peak = 0.06
    one_round_decay = math.exp(-0.2)
    extras = [
        peak / 3,
        peak * 2 / 3,
        peak,
        peak * one_round_decay,
        peak * one_round_decay**2,
        peak * one_round_decay**3,
    ]
    reference = qec_burst_scaling_circuit(
        circuit, (3, 8, 10, 12, 17), 2, extras
    )

    burst = burst_of(
        circuit,
        8,
        burst_onset_round=3,
        burst_rise_rounds=3,
        burst_decay_rounds=5.0,
        burst_radius=2.0,
        burst_error_probability=peak,
        burst_channels=("idle",),
    )

    assert burst.approx_equals(reference, atol=1e-12)


def test_the_burst_noise_alone_is_the_burst_circuits_copies_alone():
    """What it adds to the noiseless circuit, burst_circuit adds to its own."""
    circuit = memory_circuit(3, 4)
    table = detector_formation.build_formation_table(circuit, 4)
    settings = stim_device.BurstStimDevice.Settings(
        burst_onset_round=2,
        burst_rise_rounds=2,
        burst_radius=3.0,
        burst_center=(2.0, 2.0),
        burst_error_probability=0.2,
    )
    with_background = stim_device.burst_circuit(circuit, table, settings)

    alone = stim_device.burst_noise(circuit, table, settings)

    noiseless = circuit.without_noise()
    alone_lines = inserted_lines(noiseless, alone)
    background_lines = inserted_lines(circuit, with_background)
    assert alone_lines == background_lines
    assert alone.without_noise() == noiseless.flattened()


def test_quiet_events_xor_the_burst_alone_fire_as_the_burst_circuit():
    """Each detector's firing, 20,000 shots a side, within five sigma.

    Detection events are linear in Pauli errors and the copies are
    independent channels, so the two draws share one distribution.
    """
    circuit = memory_circuit(3, 6)
    table = detector_formation.build_formation_table(circuit, 6)
    settings = stim_device.BurstStimDevice.Settings(
        burst_onset_round=3,
        burst_rise_rounds=2,
        burst_radius=3.0,
        burst_center=(3.0, 3.0),
        burst_error_probability=0.1,
    )
    combined = stim_device.burst_circuit(circuit, table, settings)
    alone = stim_device.burst_noise(circuit, table, settings)
    quiet_sampler = circuit.compile_detector_sampler(seed=1)
    alone_sampler = alone.compile_detector_sampler(seed=2)
    combined_sampler = combined.compile_detector_sampler(seed=3)

    quiet_events = quiet_sampler.sample(20_000)
    alone_events = alone_sampler.sample(20_000)
    combined_events = combined_sampler.sample(20_000)

    paired_events = quiet_events ^ alone_events
    paired_firing = paired_events.mean(axis=0)
    combined_firing = combined_events.mean(axis=0)
    pooled = (paired_firing + combined_firing) / 2
    variance = pooled * (1 - pooled) * 2 / 20_000
    standard_error = numpy.sqrt(variance) + 1e-12
    signed_difference = paired_firing - combined_firing
    difference = numpy.abs(signed_difference)
    bound = 5 * standard_error
    is_within = difference < bound
    assert is_within.all()
    assert combined_firing.max() > 0.2


def test_each_channel_raises_the_noise_stim_places_there():
    """Gate after a unitary, idle after TICK, flips beside measure and reset.

    A whole-patch burst from the last round of a two-round distance-3
    memory adds, beside each noise instruction of that round, the same
    instruction at the burst's probability.
    """
    circuit = memory_circuit(3, 2)
    burst = burst_of(
        circuit, 2, burst_onset_round=2, burst_error_probability=0.25
    )

    assert inserted_lines(circuit, burst) == [
        "DEPOLARIZE1(0.25) 1 3 5 8 10 12 15 17 19",
        "DEPOLARIZE1(0.25) 2 11 16 25",
        "DEPOLARIZE1(0.25) 2 11 16 25",
        "DEPOLARIZE2(0.25) 16 10 11 5 25 19 8 9 17 18 12 13",
        "DEPOLARIZE2(0.25) 16 8 11 3 25 17 1 9 10 18 5 13",
        "DEPOLARIZE2(0.25) 2 1 16 15 11 10 8 14 3 9 12 18",
        "DEPOLARIZE2(0.25) 2 3 16 17 11 12 15 14 10 9 19 18",
        "X_ERROR(0.25) 1 3 5 8 10 12 15 17 19",
        "X_ERROR(0.25) 2 9 11 13 14 16 18 25",
        "X_ERROR(0.25) 2 9 11 13 14 16 18 25",
        "X_ERROR(0.25) 2 9 11 13 14 16 18 25",
    ]


def test_a_measurement_burst_flips_only_the_readout_of_its_region():
    """The TLS case: one measure qubit's readout, from the onset on."""
    circuit = memory_circuit(3, 3)
    burst = burst_of(
        circuit,
        3,
        burst_onset_round=2,
        burst_radius=0.0,
        burst_center=(2.0, 0.0),
        burst_error_probability=0.2,
        burst_channels=("measurement",),
    )

    assert inserted_lines(circuit, burst) == [
        "X_ERROR(0.2) 2",
        "X_ERROR(0.2) 2",
    ]


def test_a_two_qubit_channel_keeps_a_pair_when_either_qubit_is_hit():
    circuit = memory_circuit(3, 1)
    burst = burst_of(
        circuit,
        1,
        burst_radius=0.0,
        burst_center=(2.0, 0.0),
        burst_error_probability=0.1,
        burst_channels=("gate",),
    )

    assert inserted_lines(circuit, burst) == [
        "DEPOLARIZE1(0.1) 2",
        "DEPOLARIZE1(0.1) 2",
        "DEPOLARIZE2(0.1) 2 1",
        "DEPOLARIZE2(0.1) 2 3",
    ]


def test_a_burst_source_without_a_burst_samples_the_stim_device_shot():
    """With the probability at 0 the two rows draw the same bits."""
    circuit = memory_circuit(3, 3)
    operation = memory_operation(circuit)
    plain = stim_device.StimDevice(seed=7)
    quiet = stim_device.BurstStimDevice(seed=7)
    plain.begin_operation(operation, 3, 3, round_period_ticks=1_100_000)
    quiet.begin_operation(operation, 3, 3, round_period_ticks=1_100_000)

    assert quiet.sampled_detection_events(1) == plain.sampled_detection_events(
        1
    )
    assert round_payload(quiet, operation, 3) == round_payload(
        plain, operation, 3
    )


def test_a_burst_shot_is_stims_shot_of_the_burst_circuit():
    """Drawn under the stim_device seed law, from the burst circuit."""
    circuit = memory_circuit(3, 3)
    operation = memory_operation(circuit)
    settings = stim_device.BurstStimDevice.Settings(
        burst_onset_round=2, burst_error_probability=0.3
    )
    device = stim_device.BurstStimDevice(settings, seed=7)
    device.begin_operation(operation, 3, 3, round_period_ticks=1_100_000)
    burst = burst_of(
        circuit, 3, burst_onset_round=2, burst_error_probability=0.3
    )
    sampler = burst.compile_sampler(seed=7382560267478030810)
    shots = sampler.sample(1)
    oracle_row = shots[0]
    converter = circuit.compile_m2d_converter()
    oracle_events = converter.convert(
        measurements=shots, separate_observables=False
    )

    third = round_payload(device, operation, 3)
    assert third.bits == tuple(int(bit) for bit in oracle_row[16:])
    assert device.sampled_detection_events(1) == tuple(oracle_events[0])


def test_a_burst_the_decoders_are_not_told_of_leaves_their_models_alone():
    circuit = memory_circuit(3, 4)
    operation = memory_operation(circuit)
    settings = stim_device.BurstStimDevice.Settings(burst_error_probability=0.3)
    plain = stim_device.StimDevice(seed=1)
    burst = stim_device.BurstStimDevice(settings, seed=1)
    whole = window(1, 4, 4)
    plain_model = plain.strong_window_model_for_operation(
        operation, whole, 4, fault_model_requirement=GRAPHLIKE
    )
    burst_model = burst.strong_window_model_for_operation(
        operation, whole, 4, fault_model_requirement=GRAPHLIKE
    )

    plain_faults = plain_model.require_faults(GRAPHLIKE_REPRESENTATION)
    burst_faults = burst_model.require_faults(GRAPHLIKE_REPRESENTATION)
    assert burst.window_model_source() is burst
    assert burst_faults.source_fault_ids == plain_faults.source_fault_ids
    assert list(burst_faults.priors) == list(plain_faults.priors)


def test_a_burst_that_starts_after_the_shot_is_refused():
    circuit = memory_circuit(3, 3)
    sentence = "burst_onset_round 4 is after the shot's last round, 3"
    with pytest.raises(ValueError, match=sentence):
        burst_of(circuit, 3, burst_onset_round=4, burst_error_probability=0.1)


def test_a_patch_the_burst_region_misses_draws_its_own_circuit():
    """Nothing is added, so its shot is the one stim_device draws."""
    circuit = memory_circuit(3, 3)
    burst = burst_of(
        circuit,
        3,
        burst_radius=1.0,
        burst_center=(40.0, 0.0),
        burst_error_probability=0.1,
    )

    assert burst is circuit


def test_a_burst_on_a_channel_with_no_noise_is_refused():
    """Where the circuit has none of a named channel's noise, a refusal.

    The burst would add nothing, and qecburst/geometry.py
    get_data_qubits refuses a circuit without background noise
    ("Ensure p0 > 0").
    """
    circuit = memory_circuit(3, 3, noise=0.0)
    settings = {
        "burst_error_probability": 0.1,
        "burst_channels": stim_device.BURST_CHANNELS,
    }

    with pytest.raises(ValueError, match="the circuit has no gate or idle"):
        burst_of(circuit, 3, **settings)


def test_a_patch_the_burst_region_misses_is_not_refused_without_noise():
    """A noiseless patch the region misses has nothing the burst names."""
    circuit = memory_circuit(3, 3, noise=0.0)
    burst = burst_of(
        circuit,
        3,
        burst_radius=1.0,
        burst_center=(40.0, 0.0),
        burst_error_probability=0.1,
    )

    assert burst is circuit


@pytest.mark.parametrize(
    ("key", "value", "sentence"),
    [
        ("burst_onset_round", True, "burst_onset_round is a one-based round"),
        ("burst_rise_rounds", 0, "burst_rise_rounds is a whole number"),
        ("burst_decay_rounds", 0, "burst_decay_rounds is a number of rounds"),
        ("burst_radius", -1.0, "burst_radius is a distance of 0 or more"),
        ("burst_center", [1.0], "burst_center is \\[x, y\\]"),
        ("burst_error_probability", "1e-3", "a number from 0 to 0.75"),
        ("burst_error_probability", 0.8, "a number from 0 to 0.75"),
        ("burst_channels", ["leakage"], "burst_channels is a non-empty list"),
        ("burst_channels", [], "burst_channels is a non-empty list"),
    ],
)
def test_a_burst_key_outside_its_domain_is_refused_naming_it(
    key, value, sentence
):
    with pytest.raises(ValueError, match=sentence):
        stim_device.BurstStimDevice.Settings(**{key: value})
