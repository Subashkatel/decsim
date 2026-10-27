"""A memory task's patches run as unbounded streams, read out at a round.

A memory task runs Stim's generated circuit for its rounds, one
operation a patch for the whole shot (producers.memory_circuit,
memory_patches). Stim writes the
rounds after the first as one REPEAT block whose body is one round
(stim.Circuit.generated), so the circuit splits into the four fragments
a stream runs, as Deltakit's export is split
(frontends/deltakit.py _memory_fragments): the preparation and first
round, the repeated round, that round followed by the destructive
readout, and the preparation followed by the readout for a one-round
run. tests/frontends/test_memory_stream.py holds the fragments of N
rounds to Stim's own circuit of N rounds.

Each patch's memory operation becomes its stream's owner
(qpu/flip_stream_device.py samples it a round at a time); a protect
operation opens the patch's protected region at round 1 and a readout
scheduled at the stream's last round closes it, so each patch's one
logical outcome is its end readout's.
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
    """Each memory patch's circuit streamed, read out at the last round.

    The patches run side by side, as producers.memory_patches lays them,
    each its own stream with its own protected region.
    """
    memories = _the_memories(workload)
    programs = []
    for memory in memories:
        program = repeated_memory(memory.circuit)
        programs.append((memory, program))
    return streams_of(programs, rounds)


def streams_of(
    programs: list, rounds: int
) -> workload_settings.WorkloadSettings:
    """Each (memory, program) pair's stream, read out at the round.

    A stream keeps its memory's id, which names its substream seed. Its
    protect and readout operations take the ids after the largest memory
    id, two a stream in order, so one memory's are its id plus one and
    two, and no id names two operations.
    """
    memory_ids = [memory.id for memory, _program in programs]
    largest_id = max(memory_ids)
    next_id = largest_id + 1
    operations = []
    owners = []
    regions = []
    no_rounds = []
    for memory, _program in programs:
        owner, begin, finish = _stream(memory, rounds, next_id)
        region = program_records.ProtectedRegion(owner.id, begin.id, finish.id)
        operations.extend((begin, finish))
        owners.append(owner)
        regions.append(region)
        no_rounds.extend(((owner.id, 0), (begin.id, 0), (finish.id, 0)))
        next_id += 2
    circuits = [(memory.id, program) for memory, program in programs]
    policy = round_policies.PerOperationRounds(tuple(no_rounds))
    return workload_settings.WorkloadSettings(
        operations=tuple(operations),
        dynamic_streams=tuple(owners),
        protected_regions=tuple(regions),
        rounds_policy=policy,
        physical_circuits=tuple(circuits),
    )


def _stream(
    memory: program_records.Operation, rounds: int, begin_id: int
) -> tuple:
    """The memory's stream owner, its protect at begin_id, its readout after."""
    owner = program_records.Operation(
        memory.id, memory.name, memory.qubits, patches=memory.patches
    )
    begin = program_records.Operation(
        begin_id,
        "protect",
        memory.qubits,
        patches=memory.patches,
        emits_detector_data=False,
    )
    end_id = begin_id + 1
    finish = dataclasses.replace(
        begin,
        id=end_id,
        name="readout",
        predecessors=(begin_id,),
        scheduled_start_round=rounds,
    )
    return owner, begin, finish


def _the_memories(
    workload: workload_settings.WorkloadSettings,
) -> tuple:
    """The workload's operations, each a decoded Stim memory circuit's."""
    if workload.dynamic_streams:
        raise ValueError(
            "not implemented yet: a stream runs memory patches' Stim "
            "circuits, and this workload already runs streams of its own"
        )
    if workload.decode_operations == ():
        raise ValueError(
            "not implemented yet: a stream measures its decoding, and this "
            "memory decodes nothing (workload.decoded is false)"
        )
    for memory in workload.operations:
        _refuse_a_memory_with_no_circuit(memory)
    return workload.operations


def _refuse_a_memory_with_no_circuit(memory: program_records.Operation) -> None:
    if memory.circuit is not None:
        return
    raise ValueError(
        "not implemented yet: a stream samples each memory's Stim circuit, "
        f"and {memory.name} has none (workload.code_task is none)"
    )


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
