"""A memory task's circuit run as an unbounded stream, read out at a round.

A memory task runs Stim's generated circuit for its rounds, one
operation for the whole shot (producers.memory_circuit). Stim writes the
rounds after the first as one REPEAT block whose body is one round
(stim.Circuit.generated), so the circuit splits into the four fragments
a stream runs, as Deltakit's export is split
(frontends/deltakit.py _memory_fragments): the preparation and first
round, the repeated round, that round followed by the destructive
readout, and the preparation followed by the readout for a one-round
run. tests/frontends/test_memory_stream.py holds the fragments of N
rounds to Stim's own circuit of N rounds.

The stream's owner is the memory operation, now the stream
(qpu/flip_stream_device.py samples it a round at a time); a protect
operation opens a protected region at round 1 and a readout scheduled at
the stream's last round closes it, so the stream's one logical outcome is
its end readout's.
"""

import dataclasses

import stim

import decsim.frontends.settings as workload_settings
import decsim.qpu.round_policies as round_policies
import decsim.records.circuits as circuit_records
import decsim.records.program as program_records

# the instruction name Stim gives a REPEAT block (stim.CircuitRepeatBlock)
REPEAT = "REPEAT"


def repeated_memory(
    circuit: stim.Circuit,
) -> circuit_records.RepeatedStimCircuit:
    """The four fragments of a memory circuit with one REPEAT block."""
    repeat_index = _the_one_repeat_block(circuit)
    block = circuit[repeat_index]
    first_round = circuit[:repeat_index]
    repeated_round = block.body_copy()
    after_repeat = repeat_index + 1
    readout = circuit[after_repeat:]
    final_round = repeated_round + readout
    single_round = first_round + readout
    return circuit_records.RepeatedStimCircuit(
        first_round, repeated_round, final_round, single_round
    )


def memory_stream(
    workload: workload_settings.WorkloadSettings, rounds: int
) -> workload_settings.WorkloadSettings:
    """The memory's circuit as a stream of the rounds, read out at the last."""
    memory = _the_one_memory(workload)
    program = repeated_memory(memory.circuit)
    return stream_of(memory, program, rounds)


def stream_of(
    memory: program_records.Operation,
    program: circuit_records.RepeatedStimCircuit,
    rounds: int,
) -> workload_settings.WorkloadSettings:
    """The program's stream on the memory's patch, read out at the round.

    The stream keeps the memory's id, which names its substream seed; the
    protect and readout operations take the ids after it.
    """
    owner = program_records.Operation(
        memory.id, memory.name, memory.qubits, patches=memory.patches
    )
    begin_id = memory.id + 1
    end_id = memory.id + 2
    begin = program_records.Operation(
        begin_id,
        "protect",
        memory.qubits,
        patches=memory.patches,
        emits_detector_data=False,
    )
    finish = dataclasses.replace(
        begin,
        id=end_id,
        name="readout",
        predecessors=(begin_id,),
        scheduled_start_round=rounds,
    )
    region = program_records.ProtectedRegion(memory.id, begin_id, end_id)
    no_rounds = ((memory.id, 0), (begin_id, 0), (end_id, 0))
    policy = round_policies.PerOperationRounds(no_rounds)
    return workload_settings.WorkloadSettings(
        operations=(begin, finish),
        dynamic_streams=(owner,),
        protected_regions=(region,),
        rounds_policy=policy,
        physical_circuits=((memory.id, program),),
    )


def _the_one_memory(
    workload: workload_settings.WorkloadSettings,
) -> program_records.Operation:
    """The workload's one operation, a decoded Stim circuit's."""
    operation_count = len(workload.operations)
    if operation_count != 1 or workload.dynamic_streams:
        raise ValueError(
            "not implemented yet: a stream runs one memory patch's Stim "
            f"circuit, and this workload has {operation_count} operations "
            f"and {len(workload.dynamic_streams)} streams"
        )
    if workload.decode_operations == ():
        raise ValueError(
            "not implemented yet: a stream measures its decoding, and this "
            "memory decodes nothing (workload.decoded is false)"
        )
    (memory,) = workload.operations
    if memory.circuit is None:
        raise ValueError(
            "not implemented yet: a stream samples the memory's Stim "
            "circuit, and this memory has none (workload.code_task is none)"
        )
    return memory


def _the_one_repeat_block(circuit: stim.Circuit) -> int:
    """The index of the circuit's one REPEAT block, its repeated round."""
    repeat_indices = []
    for index, instruction in enumerate(circuit):
        if instruction.name == REPEAT:
            repeat_indices.append(index)
    if len(repeat_indices) == 1:
        return repeat_indices[0]
    raise ValueError(
        "not implemented yet: a stream repeats the one REPEAT block of "
        f"Stim's generated memory, and this circuit has "
        f"{len(repeat_indices)}; Stim writes one from 3 rounds on"
    )
