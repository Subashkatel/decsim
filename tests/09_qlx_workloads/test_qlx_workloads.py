"""Frozen QLX workload and physical detector-routing contracts.

These tests consume frozen artifacts only. They do not import or execute the
provenance generator, dump, or probe scripts beside those artifacts.
"""

from dataclasses import replace
import json
from pathlib import Path

import stim

from decsim.adapters.stim_device import StimDevice
from decsim.decoders import PerRoundDecoder
from decsim.frontends.qlx import qlx_frontend
from decsim.message import OpKind
from decsim.planner import GateRounds
from decsim.run_spec import RunSpec


QLX_DATA = Path(__file__).resolve().parents[1] / "data" / "qlx"


def _load_json(name):
    with (QLX_DATA / name).open("r", encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def _detector_bearing_runtime(program):
    """Represent only detector-bearing submissions in Stim's finite source.

    The frozen memory circuit uses its first syndrome submission solely as a
    detector baseline. Its 56 detectors belong to the following seven
    submissions, numbered 2 through 8 by qlx_frontend. StimDevice finite
    sources require every represented round to contain a detector, so this
    test-side physical view omits the detector-free baseline from the stream
    and renumbers the seven detector-bearing submissions to 1 through 7.

    QLX's raw durations remain unused. GateRounds prices all runtime work: the
    baseline and seven physical submissions are measurements. ``d=7`` prices
    only the seven-round detector-bearing stream owner; it is a runtime source
    length here, not a claim that the frozen circuit has code distance seven.
    """
    runtime_operations = []
    for operation in program.operations:
        if not operation.name.startswith("measure_syndrome["):
            runtime_operations.append(operation)
            continue
        if operation.stream_offset == 0:
            runtime_operations.append(replace(
                operation,
                circuit=None,
                stream_id=None,
                stream_offset=None,
                emits_detector_data=False,
                kind=OpKind.MEASURE,
            ))
            continue
        runtime_operations.append(replace(
            operation,
            kind=OpKind.MEASURE,
            stream_offset=operation.stream_offset - 1,
        ))

    program.operations = runtime_operations
    program.decoder_operations = (
        replace(program.decoder_operations[0], kind=OpKind.MEMORY),
    )
    program.detector_rounds_by_stream = {
        stream_id: {
            detector_id: emitted_round - 1
            for detector_id, emitted_round in detector_rounds.items()
        }
        for stream_id, detector_rounds
        in program.detector_rounds_by_stream.items()
    }
    return program


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


def test_frozen_mem_surface_physical_routing_completes_without_quality_claim():
    """Prove detector routing completes without claiming decode quality."""
    circuit = stim.Circuit.from_file(QLX_DATA / "mem_surface.stim")
    decode_operation_id = 100
    program = qlx_frontend(
        _load_json("schedule_mem_surface.json"),
        physical_circuit=circuit,
        detector_metadata=_load_json("mem_surface_decoder_params.json"),
        decode_operation_id=decode_operation_id,
    )

    original_detector_rounds = program.detector_rounds_by_stream[
        decode_operation_id
    ]
    assert len(original_detector_rounds) == circuit.num_detectors == 56
    assert set(original_detector_rounds.values()) == set(range(2, 9))
    assert program.terminal_detector_ids_by_stream == {
        decode_operation_id: (),
    }
    assert program.terminal_data_bits_by_stream == {
        decode_operation_id: 9,
    }

    program = _detector_bearing_runtime(program)
    assert set(
        program.detector_rounds_by_stream[decode_operation_id].values()
    ) == set(range(1, 8))

    # Authorized StimDevice wiring from 43f4ffa^:tests/test_qlx_physical.py
    # lines 610-614. These maps are routing metadata, not accuracy evidence.
    device = StimDevice(
        detector_rounds=program.detector_rounds_by_stream,
        terminal_detector_ids=program.terminal_detector_ids_by_stream,
        terminal_data_bits=program.terminal_data_bits_by_stream,
    )
    completed = RunSpec(
        frontend=program,
        decode_ops=program.decoder_operations,
        device=device,
        decoder=PerRoundDecoder(tau_us=0.0),
        rounds_policy=GateRounds(merge_steps=2),
        d=7,
        seed=28,
    ).build()

    assert completed.window_manager.rounds_for(
        program.decoder_operations[0]
    ) == 7
    assert tuple(
        completed.window_manager.rounds_for(operation)
        for operation in program.operations[1:9]
    ) == (1, 1, 1, 1, 1, 1, 1, 1)
    assert completed.result.terminal_status == "complete"
    assert completed.result.event_queue_empty
    assert completed.result.decode_work_settled
    assert completed.result.execution_workload_complete
    assert all(
        result.result_status == "no_logical_output"
        for result in completed.result.operation_results
    )
