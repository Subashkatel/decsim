"""Frozen QLX workload and physical detector-routing contracts.

These tests consume frozen artifacts only. They do not import or execute the
provenance generator, dump, or probe scripts beside those artifacts.
"""

from dataclasses import replace
import json
from pathlib import Path

import pytest
import stim

from decsim.qpu.stim_device import StimDevice
from decsim.decoders.decoders import PerRoundDecoder
from decsim.detector_error_model.detector_chronology import resolve_detector_rounds
from decsim.frontends.qlx_frontend import qlx_frontend
from decsim.message import OpKind
from decsim.qpu.round_policies import GateRounds
from decsim.run_spec import RunSpec


QLX_DATA = Path(__file__).resolve().parents[1] / "data" / "qlx"


def _load_json(name):
    with (QLX_DATA / name).open("r", encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def _native_physical_program():
    """Load and price the frozen physical workload without changing its chronology."""
    circuit = stim.Circuit.from_file(QLX_DATA / "mem_surface.stim")
    decode_operation_id = 100
    program = qlx_frontend(
        _load_json("schedule_mem_surface.json"),
        physical_circuit=circuit,
        detector_metadata=_load_json("mem_surface_decoder_params.json"),
        decode_operation_id=decode_operation_id,
    )
    program.operations = [
        replace(operation, kind=OpKind.MEASURE)
        if operation.name.startswith("measure_syndrome[")
        else operation
        for operation in program.operations
    ]
    program.decoder_operations = (
        replace(program.decoder_operations[0], kind=OpKind.MEMORY),
    )
    return circuit, program, decode_operation_id


def _physical_device(program):
    """Build a Stim device from a physical program's native routing metadata."""
    # Authorized StimDevice wiring: these maps are routing metadata, not accuracy evidence.
    return StimDevice(
        detector_rounds=program.detector_rounds_by_stream,
        terminal_detector_ids=program.terminal_detector_ids_by_stream,
        measurement_rounds=program.measurement_rounds_by_stream,
    )


def _run_native_physical_program(program, device):
    """Run the native eight-round physical source with timing-only decoding."""
    return RunSpec(
        frontend=program,
        decode_ops=program.decoder_operations,
        device=device,
        decoder=PerRoundDecoder(tau_us=0.0),
        rounds_policy=GateRounds(merge_steps=2),
        # Native runtime source length, not a code-distance claim.
        d=8,
        seed=28,
    ).build()


def test_frozen_mem_surface_schedule_has_stable_structure_and_gate_rounds():
    """Pin frozen schedule structure and GateRounds-owned runtime pricing."""
    program = qlx_frontend(_load_json("schedule_mem_surface.json"))

    assert len(program.build()) == 11
    assert tuple(
        operation.predecessors for operation in program.operations
    ) == (
        (), (0,), (1,), (2,), (3,), (4,),
        (5,), (6,), (7,), (8,), (9,),
    )
    assert len(program.patch_of_cell) == 1
    assert tuple(len(operation.patches) for operation in program.operations) == (
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    )
    start_rounds = tuple(program.start_rounds.values())
    assert start_rounds == tuple(sorted(start_rounds))
    assert program.feedback_candidates == [(10, 9)]

    rounds_policy = GateRounds(merge_steps=2)
    completed = RunSpec(
        frontend=program,
        rounds_policy=rounds_policy,
        d=3,
    ).build()

    resolved_rounds = tuple(
        completed.window_manager.rounds_for(operation)
        for operation in program.operations
    )
    assert resolved_rounds == (3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3)
    assert resolved_rounds != tuple(program.raw_durations.values())
    assert completed.result.terminal_status == "complete"


def test_frozen_mem_surface_native_round_routing_completes_without_quality_claim():
    """Route the frozen circuit on its own eight rounds, baseline round included."""
    circuit, program, decode_operation_id = _native_physical_program()
    detector_rounds = program.detector_rounds_by_stream[decode_operation_id]

    assert len(detector_rounds) == circuit.num_detectors == 56
    assert detector_rounds == {
        detector_id: 2 + detector_id // 8
        for detector_id in range(circuit.num_detectors)
    }
    assert tuple(
        tuple(
            detector_id
            for detector_id in range(circuit.num_detectors)
            if detector_rounds[detector_id] == round_index
        )
        for round_index in range(1, 9)
    ) == ((),) + tuple(
        tuple(range(8 * offset, 8 * (offset + 1)))
        for offset in range(7)
    )
    assert resolve_detector_rounds(circuit, detector_rounds, 8) == detector_rounds
    with pytest.raises(ValueError) as coordinate_failure:
        resolve_detector_rounds(circuit, None, 8)
    assert "requires supported coordinates or explicit detector_rounds" in str(
        coordinate_failure.value
    )
    assert program.terminal_detector_ids_by_stream == {
        decode_operation_id: (),
    }
    measurement_rounds = program.measurement_rounds_by_stream[decode_operation_id]
    # eight checks per submission, eight submissions, then the nine data
    # readouts fold into the last round's packet
    assert len(measurement_rounds) == 8 * 8 + 9
    assert [measurement_rounds[index] for index in (0, 7, 8, 63, 64, 72)] == [1, 1, 2, 8, 8, 8]

    device = _physical_device(program)
    completed = _run_native_physical_program(program, device)

    assert completed.window_manager.rounds_for(
        program.decoder_operations[0]
    ) == 8
    assert tuple(
        completed.window_manager.rounds_for(operation)
        for operation in program.operations
    ) == (8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8)
    submissions = [
        operation
        for operation in program.operations
        if operation.name.startswith("measure_syndrome[")
    ]
    round_payloads = tuple(
        device.round_payloads(operation, 1)[0]
        for operation in submissions
    )
    assert tuple(payload.round_index for payload in round_payloads) == tuple(
        range(1, 9)
    )
    assert tuple(payload.size_bits for payload in round_payloads) == (
        8, 8, 8, 8, 8, 8, 8, 17,
    )
    assert tuple(len(payload.bits) for payload in round_payloads) == (
        8, 8, 8, 8, 8, 8, 8, 17,
    )
    assert completed.result.terminal_status == "complete"
    assert completed.result.event_queue_empty
    assert completed.result.decode_work_settled
    assert completed.result.execution_workload_complete
    assert all(
        result.result_status == "no_logical_output"
        for result in completed.result.operation_results
    )


@pytest.mark.parametrize("invalid_round", [0, 9])
def test_frozen_mem_surface_rejects_native_rounds_outside_eight_round_source(
    invalid_round,
):
    """The real native map rejects detector rounds below one or above eight."""
    _, program, decode_operation_id = _native_physical_program()
    detector_rounds = dict(
        program.detector_rounds_by_stream[decode_operation_id]
    )
    detector_rounds[0] = invalid_round
    program.detector_rounds_by_stream[decode_operation_id] = detector_rounds

    with pytest.raises(ValueError) as failure:
        _run_native_physical_program(program, _physical_device(program))
    assert "detector-round map must lie inside the emitted rounds" in str(
        failure.value
    )


def test_frozen_mem_surface_rejects_a_missing_native_detector_identity():
    """The real native map must retain every frozen circuit detector identity."""
    _, program, decode_operation_id = _native_physical_program()
    detector_rounds = dict(
        program.detector_rounds_by_stream[decode_operation_id]
    )
    detector_rounds.pop(55)
    program.detector_rounds_by_stream[decode_operation_id] = detector_rounds

    with pytest.raises(ValueError) as failure:
        _run_native_physical_program(program, _physical_device(program))
    assert "detector-round map must cover every detector exactly" in str(
        failure.value
    )


def test_frozen_mem_surface_rejects_a_native_round_order_swap():
    """Swapping two real detector rounds still breaks canonical decoder row order."""
    _, program, decode_operation_id = _native_physical_program()
    detector_rounds = dict(
        program.detector_rounds_by_stream[decode_operation_id]
    )
    detector_rounds[0], detector_rounds[8] = (
        detector_rounds[8], detector_rounds[0]
    )
    program.detector_rounds_by_stream[decode_operation_id] = detector_rounds

    with pytest.raises(ValueError) as failure:
        _run_native_physical_program(program, _physical_device(program))
    # the formation table rejects it first: detector 8 would be declared in a
    # round before the measurement bit it reads has arrived
    assert "reads a bit that arrives in round" in str(failure.value)
