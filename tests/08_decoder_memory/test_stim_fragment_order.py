from __future__ import annotations

from collections import defaultdict

import stim

from decsim.qpu.stim_device import StimDevice
from decsim.detector_error_model.fault_model_contracts import (
    GRAPHLIKE_FAULT_MODEL_REQUIRED,
)
from decsim.detector_error_model.window_model_builders import (
    build_window_error_models,
)
from decsim.message import Operation, RunOperationBody
from decsim.engine import Engine
from decsim.qpu.cycle_clock import QPUDevice


ROUND_COUNT = 3
DISTANCE = 3
STREAM_ID = 7


def _deterministic_repetition_memory_circuit() -> stim.Circuit:
    """Build generated repetition memory with probability-one witness faults."""
    flattened = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=ROUND_COUNT,
        distance=DISTANCE,
    ).flattened()
    faults_before_extraction = {0: (0,), 1: (4,), 2: (0,)}
    circuit = stim.Circuit()
    measurement_rounds_seen = 0
    faulted_rounds = set()
    for instruction in flattened:
        if (
            instruction.name == "CX"
            and measurement_rounds_seen in faults_before_extraction
            and measurement_rounds_seen not in faulted_rounds
        ):
            for qubit in faults_before_extraction[measurement_rounds_seen]:
                circuit.append("X_ERROR", [qubit], 1.0)
            faulted_rounds.add(measurement_rounds_seen)
        if instruction.name == "M":
            circuit.append("X_ERROR", [4], 1.0)
        circuit.append(instruction)
        if instruction.name == "MR":
            measurement_rounds_seen += 1
    return circuit


def _model_rows_by_round(
    circuit: stim.Circuit,
) -> tuple[object, dict[int, tuple[int, ...]], dict[int, int]]:
    inferred_model = build_window_error_models(
        circuit,
        [(1, ROUND_COUNT, ROUND_COUNT)],
        round_count=ROUND_COUNT,
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
    )[0]
    detector_rounds = {
        detector_id: inferred_model.defect_positions[detector_id][0]
        for detector_id in reversed(inferred_model.detector_ids)
    }
    model = build_window_error_models(
        circuit,
        [(1, ROUND_COUNT, ROUND_COUNT)],
        round_count=ROUND_COUNT,
        detector_rounds=detector_rounds,
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
    )[0]
    positioned_rows = defaultdict(list)
    for detector_id in model.detector_ids:
        round_index, position_in_round = model.defect_positions[detector_id]
        positioned_rows[round_index].append((position_in_round, detector_id))

    rows_by_round = {}
    for round_index, rows in positioned_rows.items():
        ordered_rows = tuple(sorted(rows))
        assert tuple(position for position, _ in ordered_rows) == tuple(
            range(len(ordered_rows))
        )
        detector_ids = tuple(detector_id for _, detector_id in ordered_rows)
        assert detector_ids == tuple(sorted(detector_ids))
        rows_by_round[round_index] = detector_ids
    return model, rows_by_round, detector_rounds


def _operation(
    operation_id: int,
    circuit: stim.Circuit,
    *,
    stream_offset: int | None = None,
    fragment_index: int | None = None,
    finalizes_stream_round: bool = False,
) -> Operation:
    fragment_count = None if fragment_index is None else 2
    return Operation(
        id=operation_id,
        name=f"segment {operation_id}",
        qubits=(0,),
        patches=(0,),
        circuit=circuit,
        stream_id=None if stream_offset is None else STREAM_ID,
        stream_offset=stream_offset,
        syndrome_fragment_index=fragment_index,
        syndrome_fragment_count=fragment_count,
        finalizes_stream_round=finalizes_stream_round,
    )


def _split_final_round_rows(
    rows_by_round: dict[int, tuple[int, ...]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    final_round_rows = rows_by_round[ROUND_COUNT]
    assert len(final_round_rows) % 2 == 0
    split = len(final_round_rows) // 2
    ordinary_rows = final_round_rows[:split]
    terminal_rows = final_round_rows[split:]
    assert len(ordinary_rows) == len(terminal_rows)
    return ordinary_rows, terminal_rows


def test_stim_round_packets_carry_raw_bits_that_form_the_model_row_block() -> None:
    """Each Stim round packet carries that round's raw measurement bits, and
    forming them at the decoder input yields the round's detector-row block
    in ascending row order."""
    circuit = _deterministic_repetition_memory_circuit()
    _, rows_by_round, detector_rounds = _model_rows_by_round(circuit)
    detector_sample = circuit.compile_detector_sampler().sample(shots=1)[0]
    operation = _operation(1, circuit)
    device = StimDevice(detector_rounds={operation.id: detector_rounds}, seed=11)

    device.begin_operation(operation, ROUND_COUNT, ROUND_COUNT)
    table = device._tables[operation.id]

    for round_index, detector_ids in rows_by_round.items():
        (payload,) = device.round_payloads(operation, round_index)
        raw_bits = tuple(int(bit) for bit in payload.bits)
        assert len(raw_bits) == table.packet_width[round_index]
        assert payload.round_index == round_index
        assert payload.size_bits == len(raw_bits)
        expected_block = tuple(int(detector_sample[row]) for row in detector_ids)
        assert device.form_round(operation.id, round_index, raw_bits) == expected_block


def test_terminal_fragment_carries_the_readout_bits_after_the_ancilla_bits() -> None:
    """The final round's raw packet arrives as two fragments: the ordinary
    ancilla fragment, then the data readout; together, in that order, they
    form the round's full detector-row block."""
    circuit = _deterministic_repetition_memory_circuit()
    _, rows_by_round, detector_rounds = _model_rows_by_round(circuit)
    _, terminal_rows = _split_final_round_rows(rows_by_round)
    detector_sample = circuit.compile_detector_sampler().sample(shots=1)[0]
    device = StimDevice(
        seed=13,
        detector_rounds={STREAM_ID: detector_rounds},
        terminal_detector_ids={STREAM_ID: terminal_rows},
    )

    ordinary_payload = None
    final_operation = None
    for stream_offset in range(ROUND_COUNT):
        operation = _operation(stream_offset + 1, circuit, stream_offset=stream_offset)
        device.begin_operation(operation, 1, ROUND_COUNT)
        (ordinary_payload,) = device.round_payloads(operation, 1)
        if stream_offset + 1 < ROUND_COUNT:
            device.form_round(STREAM_ID, stream_offset + 1, ordinary_payload.bits)
        final_operation = operation
    assert ordinary_payload is not None
    assert final_operation is not None
    (terminal_payload,) = device.finalize_stream_round(final_operation, ROUND_COUNT)

    ordinary_bits = tuple(int(bit) for bit in ordinary_payload.bits)
    terminal_bits = tuple(int(bit) for bit in terminal_payload.bits)
    table = device._tables[STREAM_ID]
    last_packet = tuple(int(bit) for bit in device._packets[STREAM_ID][ROUND_COUNT])
    assert ordinary_bits == last_packet[:table.readout_slot_start]
    assert terminal_bits == last_packet[table.readout_slot_start:]
    assert len(ordinary_bits) == DISTANCE - 1      # ancilla qubits of the repetition code
    assert terminal_payload.size_bits == DISTANCE  # data qubits read out at the end
    assert ordinary_payload.round_index == terminal_payload.round_index == ROUND_COUNT
    expected_block = tuple(
        int(detector_sample[row]) for row in rows_by_round[ROUND_COUNT]
    )
    assert device.form_round(STREAM_ID, ROUND_COUNT, ordinary_bits + terminal_bits) == expected_block


class _ReadoutCapture:
    def __init__(self) -> None:
        self.readouts = []

    def accept_qpu_readout(self, payload, route) -> None:
        self.readouts.append(payload)


def test_qpu_stamps_fragment_slots_in_detector_row_order_end_to_end() -> None:
    """The QPU stamps declared fragment slots in the model detector-row order."""
    circuit = _deterministic_repetition_memory_circuit()
    _, rows_by_round, detector_rounds = _model_rows_by_round(circuit)
    _, terminal_rows = _split_final_round_rows(rows_by_round)
    detector_sample = circuit.compile_detector_sampler().sample(shots=1)[0]
    capture = _ReadoutCapture()
    engine = Engine()
    commands = []
    for stream_offset in range(ROUND_COUNT):
        fragment_index = 0 if stream_offset == ROUND_COUNT - 1 else None
        operation = _operation(
            stream_offset + 1,
            circuit,
            stream_offset=stream_offset,
            fragment_index=fragment_index,
        )
        commands.append(RunOperationBody(operation, 1, 1, ROUND_COUNT))
    finalizer = _operation(
        4,
        circuit,
        stream_offset=ROUND_COUNT - 1,
        fragment_index=1,
        finalizes_stream_round=True,
    )
    commands.append(RunOperationBody(finalizer, 1, 0, ROUND_COUNT, finalizes_stream_round=True))
    commands.reverse()

    def issue_next(_operation=None) -> None:      # one segment per cycle, in stream order
        if commands:
            qpu.issue(commands.pop())
        else:
            qpu.finish()

    qpu = QPUDevice(
        engine,
        StimDevice(
            seed=17,
            detector_rounds={STREAM_ID: detector_rounds},
            terminal_detector_ids={STREAM_ID: terminal_rows},
        ),
        1,
        readout_receiver=capture,
        completion_receiver=issue_next,
        idle_receiver=lambda *args: None,
    )
    issue_next()
    engine.run()

    assert tuple(
        (readout.round_index, readout.fragment_index, readout.n_fragments)
        for readout in capture.readouts
    ) == ((1, 0, 1), (2, 0, 1), (3, 0, 2), (3, 1, 2))
    final_round_readouts = sorted(
        (readout for readout in capture.readouts if readout.round_index == ROUND_COUNT),
        key=lambda readout: readout.fragment_index,
    )
    ordered_bits = tuple(
        int(bit)
        for readout in final_round_readouts
        for bit in readout.bits
    )
    swapped_bits = tuple(
        int(bit)
        for readout in reversed(final_round_readouts)
        for bit in readout.bits
    )
    device = qpu.model
    last_packet = tuple(int(bit) for bit in device._packets[STREAM_ID][ROUND_COUNT])
    assert ordered_bits == last_packet
    assert swapped_bits != last_packet
    for round_index in range(1, ROUND_COUNT):
        (readout,) = [r for r in capture.readouts if r.round_index == round_index]
        device.form_round(STREAM_ID, round_index, readout.bits)
    expected_block = tuple(
        int(detector_sample[row]) for row in rows_by_round[ROUND_COUNT]
    )
    assert device.form_round(STREAM_ID, ROUND_COUNT, ordered_bits) == expected_block
