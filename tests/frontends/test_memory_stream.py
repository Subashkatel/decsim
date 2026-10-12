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


def test_a_memory_of_several_patches_is_refused():
    workload = producers.memory_patches(TASK, 9, 2, 3, 0.002)
    lowered = _lowered(workload)

    with pytest.raises(ValueError, match="not implemented yet: a stream runs"):
        memory_stream.memory_stream(lowered, 100)


def test_a_circuit_stim_wrote_with_no_repeat_block_is_refused():
    workload = producers.memory_circuit(TASK, 2, 3, 0.002)
    lowered = _lowered(workload)

    with pytest.raises(ValueError, match="Stim writes one from 3 rounds"):
        memory_stream.memory_stream(lowered, 100)
