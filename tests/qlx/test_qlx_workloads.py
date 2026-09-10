"""Frozen QLX workload and physical detector-routing contracts.

These tests consume frozen artifacts only. They do not import or execute the
provenance generator, dump, or probe scripts beside those artifacts.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import stim

import decsim.records.program as program_records
from decsim.decoders.decoders import PerRoundDecoder
from decsim.decoders.settings import DecoderSettings
from decsim.detector_error_model.detector_chronology import (
    resolve_detector_rounds,
)
from decsim.frontends.qlx_frontend import qlx_frontend
from decsim.frontends.settings import WorkloadSettings
from decsim.machine import Machine
from decsim.qpu.round_policies import GateRounds
from decsim.qpu.settings import QpuSettings
from decsim.qpu.stim_device import StimDevice
from decsim.settings import MachineSettings

_THIS_FILE = Path(__file__)
_TEST_FILE = _THIS_FILE.resolve()
_TESTS_ROOT = _TEST_FILE.parents[1]
QLX_DATA = _TESTS_ROOT / "data" / "qlx"


def load_json(name):
    fixture_path = QLX_DATA / name
    with fixture_path.open("r", encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def native_physical_program():
    """Load and price the frozen physical workload.

    The frozen chronology is not changed.
    """
    circuit_path = QLX_DATA / "mem_surface.stim"
    circuit = stim.Circuit.from_file(circuit_path)
    decode_operation_id = 100
    schedule = load_json("schedule_mem_surface.json")
    detector_metadata = load_json("mem_surface_decoder_params.json")
    program = qlx_frontend(
        schedule,
        physical_circuit=circuit,
        detector_metadata=detector_metadata,
        decode_operation_id=decode_operation_id,
    )
    measured_operations = []
    for operation in program.operations:
        is_syndrome_measurement = operation.name.startswith("measure_syndrome[")
        if is_syndrome_measurement:
            measure_operation = replace(
                operation, kind=program_records.OpKind.MEASURE
            )
            measured_operations.append(measure_operation)
        else:
            measured_operations.append(operation)
    program.operations = measured_operations
    first_decoder_operation = program.decoder_operations[0]
    memory_operation = replace(
        first_decoder_operation, kind=program_records.OpKind.MEMORY
    )
    program.decoder_operations = (memory_operation,)
    return circuit, program, decode_operation_id


def physical_device(program):
    """Build a Stim device from a physical program's native routing metadata."""
    # These maps are routing metadata, not accuracy evidence.
    return StimDevice(
        detector_rounds=program.detector_rounds_by_stream,
        terminal_detector_ids=program.terminal_detector_ids_by_stream,
        measurement_rounds=program.measurement_rounds_by_stream,
    )


def run_native_physical_program(program, device):
    """Run the native eight-round physical source with timing-only decoding."""
    rounds_policy = GateRounds(merge_step_count=2)
    workload_settings = WorkloadSettings(
        kind="qlx",
        program=program,
        decode_operations=program.decoder_operations,
        rounds_policy=rounds_policy,
    )
    # Native runtime source length, not a code-distance claim.
    qpu_settings = QpuSettings(distance=8, device=device)
    per_round_decoder = PerRoundDecoder(tau_us=0.0)
    weak_decoder_settings = DecoderSettings(decoder=per_round_decoder)
    settings = MachineSettings(
        workload=workload_settings,
        qpu=qpu_settings,
        weak_decoder=weak_decoder_settings,
    )
    machine = Machine.build(settings, 28)
    result = machine.run()
    return machine, result


def test_frozen_mem_surface_schedule_has_stable_structure_and_gate_rounds():
    """Pin frozen schedule structure and GateRounds-owned runtime pricing."""
    schedule = load_json("schedule_mem_surface.json")
    program = qlx_frontend(schedule)

    operations = program.build()
    assert len(operations) == 11
    assert tuple(
        operation.predecessors for operation in program.operations
    ) == (
        (),
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (6,),
        (7,),
        (8,),
        (9,),
    )
    assert len(program.patch_of_cell) == 1
    assert tuple(
        len(operation.patches) for operation in program.operations
    ) == (
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    start_round_values = program.start_rounds.values()
    start_rounds = tuple(start_round_values)
    assert start_rounds == tuple(sorted(start_rounds))
    assert program.feedback_candidates == [(10, 9)]

    rounds_policy = GateRounds(merge_step_count=2)
    workload_settings = WorkloadSettings(
        kind="qlx", program=program, rounds_policy=rounds_policy
    )
    qpu_settings = QpuSettings(distance=3)
    settings = MachineSettings(workload=workload_settings, qpu=qpu_settings)
    completed = Machine.build(settings)
    result = completed.run()

    resolved_rounds = tuple(
        completed.window_manager.planner.round_count_of(operation.id)
        for operation in program.operations
    )
    assert resolved_rounds == (3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3)
    raw_duration_values = program.raw_durations.values()
    assert resolved_rounds != tuple(raw_duration_values)
    assert result.terminal_status == "complete"


def test_mem_surface_native_round_routing_completes_without_quality_claim():
    """Route the frozen circuit on its own eight rounds.

    The baseline round is one of the eight.
    """
    circuit, program, decode_operation_id = native_physical_program()
    detector_rounds = program.detector_rounds_by_stream[decode_operation_id]

    assert len(detector_rounds) == circuit.num_detectors == 56
    expected_detector_rounds = {}
    for detector_id in range(circuit.num_detectors):
        round_offset = detector_id // 8
        round_index = 2 + round_offset
        expected_detector_rounds[detector_id] = round_index
    assert detector_rounds == expected_detector_rounds

    detector_ids_by_round = {}
    for round_index in range(1, 9):
        detector_ids_by_round[round_index] = []
    for detector_id in range(circuit.num_detectors):
        declared_round = detector_rounds[detector_id]
        detector_ids_by_round[declared_round].append(detector_id)
    routed_detector_ids = []
    for round_index in range(1, 9):
        round_detector_ids = detector_ids_by_round[round_index]
        routed_detector_ids.append(tuple(round_detector_ids))
    expected_detector_ids = [()]
    for offset in range(7):
        first_detector_id = 8 * offset
        next_offset = offset + 1
        next_first_detector_id = 8 * next_offset
        block_detector_ids = range(first_detector_id, next_first_detector_id)
        expected_detector_ids.append(tuple(block_detector_ids))
    assert tuple(routed_detector_ids) == tuple(expected_detector_ids)

    assert (
        resolve_detector_rounds(circuit, detector_rounds, 8) == detector_rounds
    )
    with pytest.raises(ValueError) as coordinate_failure:
        resolve_detector_rounds(circuit, None, 8)
    assert "requires supported coordinates or explicit detector_rounds" in str(
        coordinate_failure.value
    )
    assert program.terminal_detector_ids_by_stream == {
        decode_operation_id: (),
    }
    measurement_rounds = program.measurement_rounds_by_stream[
        decode_operation_id
    ]
    # eight checks per submission, eight submissions, then the nine data
    # readouts fold into the last round's packet
    assert len(measurement_rounds) == 8 * 8 + 9
    assert [measurement_rounds[index] for index in (0, 7, 8, 63, 64, 72)] == [
        1,
        1,
        2,
        8,
        8,
        8,
    ]

    device = physical_device(program)
    completed, result = run_native_physical_program(program, device)

    assert (
        completed.window_manager.planner.round_count_of(
            program.decoder_operations[0].id
        )
        == 8
    )
    assert tuple(
        completed.window_manager.planner.round_count_of(operation.id)
        for operation in program.operations
    ) == (8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8)
    submissions = [
        operation
        for operation in program.operations
        if operation.name.startswith("measure_syndrome[")
    ]
    round_payloads = []
    for operation in submissions:
        payloads = device.round_payloads(operation, 1)
        round_payloads.append(payloads[0])
    assert tuple(payload.round_index for payload in round_payloads) == tuple(
        range(1, 9)
    )
    assert tuple(payload.size_bits for payload in round_payloads) == (
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        17,
    )
    assert tuple(len(payload.bits) for payload in round_payloads) == (
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        17,
    )
    assert result.terminal_status == "complete"
    assert result.event_queue_empty
    assert result.decode_work_settled
    assert result.execution_workload_complete
    assert all(
        operation_result.result_status == "no_logical_output"
        for operation_result in result.operation_results
    )


@pytest.mark.parametrize("invalid_round", [0, 9])
def test_frozen_mem_surface_rejects_native_rounds_outside_eight_round_source(
    invalid_round,
):
    """The real native map rejects detector rounds below one or above eight."""
    _, program, decode_operation_id = native_physical_program()
    detector_rounds = dict(
        program.detector_rounds_by_stream[decode_operation_id]
    )
    detector_rounds[0] = invalid_round
    program.detector_rounds_by_stream[decode_operation_id] = detector_rounds

    device = physical_device(program)
    with pytest.raises(ValueError) as failure:
        run_native_physical_program(program, device)
    assert "detector-round map must lie inside the emitted rounds" in str(
        failure.value
    )


def test_frozen_mem_surface_rejects_a_missing_native_detector_identity():
    """The real native map retains every frozen circuit detector identity."""
    _, program, decode_operation_id = native_physical_program()
    detector_rounds = dict(
        program.detector_rounds_by_stream[decode_operation_id]
    )
    detector_rounds.pop(55)
    program.detector_rounds_by_stream[decode_operation_id] = detector_rounds

    device = physical_device(program)
    with pytest.raises(ValueError) as failure:
        run_native_physical_program(program, device)
    assert "detector-round map must cover every detector exactly" in str(
        failure.value
    )


def test_frozen_mem_surface_rejects_a_native_round_order_swap():
    """A swap of two real detector rounds breaks canonical decoder rows."""
    _, program, decode_operation_id = native_physical_program()
    detector_rounds = dict(
        program.detector_rounds_by_stream[decode_operation_id]
    )
    detector_rounds[0], detector_rounds[8] = (
        detector_rounds[8],
        detector_rounds[0],
    )
    program.detector_rounds_by_stream[decode_operation_id] = detector_rounds

    device = physical_device(program)
    with pytest.raises(ValueError) as failure:
        run_native_physical_program(program, device)
    # the formation table rejects it first: detector 8 would be declared in a
    # round before the measurement bit it reads has arrived
    assert "reads a bit that arrives in round" in str(failure.value)
