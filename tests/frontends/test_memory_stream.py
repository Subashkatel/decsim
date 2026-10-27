"""A memory task's circuit as a stream (frontends/memory_stream.py).

The referent is Stim's own generated circuit: the fragments of N rounds,
assembled, are Stim's circuit of N rounds, and the one-round fragment is
Stim's one-round circuit, each compared flattened (REPEAT unrolled,
SHIFT_COORDS applied).
"""

import pytest
import stim

import decsim.frontends.memory_stream as memory_stream
import decsim.frontends.settings as workload_settings
import decsim.producers as producers

TASK = "surface_code:rotated_memory_z"


def _generated(rounds: int) -> stim.Circuit:
    noise = workload_settings.StimNoise(0.001, 0.002, 0.003, 0.004)
    return workload_settings.memory_circuit(TASK, rounds, 3, 0.001, noise)


def _lowered(workload) -> workload_settings.WorkloadSettings:
    return workload_settings.WorkloadSettings.running(workload)


@pytest.mark.parametrize("rounds", [2, 3, 7])
def test_the_fragments_of_n_rounds_are_stims_circuit_of_n_rounds(rounds):
    circuit = _generated(4)
    program = memory_stream.repeated_memory(circuit)
    middle_rounds = rounds - 2
    assembled = program.first_round
    for _ in range(middle_rounds):
        assembled = assembled + program.repeated_round
    assembled = assembled + program.final_round
    expected = _generated(rounds)

    assert assembled.flattened() == expected.flattened()


def test_the_one_round_fragment_is_stims_one_round_circuit():
    circuit = _generated(4)
    program = memory_stream.repeated_memory(circuit)
    expected = _generated(1)

    single = program.single_round
    assert single.flattened() == expected.flattened()


def test_the_stream_is_the_memory_protected_and_read_out_at_its_last_round():
    workload = producers.memory_circuit(TASK, 9, 3, 0.002)
    lowered = _lowered(workload)

    streamed = memory_stream.memory_stream(lowered, 500)

    (owner,) = streamed.dynamic_streams
    begin, finish = streamed.operations
    assert owner.circuit is None
    assert (begin.name, finish.name) == ("protect", "readout")
    assert finish.scheduled_start_round == 500
    (region,) = streamed.protected_regions
    endpoints = (region.start_operation_id, region.end_operation_id)
    assert region.stream_id == owner.id
    assert endpoints == (begin.id, finish.id)
    (circuit_owner, _program) = streamed.physical_circuits[0]
    assert circuit_owner == owner.id


def test_a_one_patch_memorys_protect_and_readout_take_the_ids_after_it():
    workload = producers.memory_circuit(TASK, 9, 3, 0.002)
    lowered = _lowered(workload)

    streamed = memory_stream.memory_stream(lowered, 500)

    (owner,) = streamed.dynamic_streams
    begin, finish = streamed.operations
    assert (owner.id, begin.id, finish.id) == (1, 2, 3)


def test_each_memory_patch_streams_with_its_own_protect_and_readout():
    workload = producers.memory_patches(TASK, 9, 2, 3, 0.002)
    lowered = _lowered(workload)

    streamed = memory_stream.memory_stream(lowered, 100)

    owners = [(owner.id, owner.patches) for owner in streamed.dynamic_streams]
    operations = [
        (operation.id, operation.name, operation.patches)
        for operation in streamed.operations
    ]
    regions = [
        (region.stream_id, region.start_operation_id, region.end_operation_id)
        for region in streamed.protected_regions
    ]
    circuit_owners = [owner for owner, _program in streamed.physical_circuits]
    assert owners == [(1, (0,)), (2, (1,))]
    assert operations == [
        (3, "protect", (0,)),
        (4, "readout", (0,)),
        (5, "protect", (1,)),
        (6, "readout", (1,)),
    ]
    assert regions == [(1, 3, 4), (2, 5, 6)]
    assert circuit_owners == [1, 2]


def test_a_memory_with_no_circuit_is_refused():
    workload = producers.memory_without_circuit(9, 2, 0.002)
    lowered = _lowered(workload)

    with pytest.raises(ValueError) as refused:
        memory_stream.memory_stream(lowered, 100)

    assert str(refused.value) == (
        "not implemented yet: a stream samples each memory's Stim circuit, "
        "and memory0 has none (workload.code_task is none)"
    )


def test_a_workload_that_runs_its_own_stream_is_refused():
    circuit = _generated(4)
    program = memory_stream.repeated_memory(circuit)
    workload = producers.live_memory(program, 3)
    lowered = _lowered(workload)

    with pytest.raises(ValueError) as refused:
        memory_stream.memory_stream(lowered, 100)

    assert str(refused.value) == (
        "not implemented yet: a stream runs memory patches' Stim circuits, "
        "and this workload already runs streams of its own"
    )


def test_a_circuit_stim_wrote_with_no_repeat_block_is_refused():
    workload = producers.memory_circuit(TASK, 2, 3, 0.002)
    lowered = _lowered(workload)

    with pytest.raises(ValueError, match="Stim writes one from 3 rounds"):
        memory_stream.memory_stream(lowered, 100)
