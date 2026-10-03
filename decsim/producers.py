"""The workload makers decsim ships: Stim's and Deltakit's memories.

A maker is a plain function a yaml names as module:function under
workload.kind producer. decsim calls it with the yaml's arguments as
each sweep point resolves them, and it returns a
records.workload.Workload. A maker written outside decsim has the same
shape; these are the ones that ship.
"""

from typing import Optional, Union

import stim

import decsim.frontends.deltakit as deltakit
import decsim.frontends.settings as workload_settings
import decsim.records.circuits as circuit_records
import decsim.records.program as program_records
import decsim.records.workload as workload_records

# The stream a live memory's segments extend: an identity local to the
# workload, which also names the stream's substream seed
# (seeding.substream_seed), so it is the one
# examples/live_memory_example.py uses and a run reproduces that example's
# draws.
LIVE_STREAM_ID = 100


def memory_circuit(
    code_task: str,
    rounds_per_shot: Union[int, str],
    distance: int,
    physical_error_probability: float,
) -> workload_records.Workload:
    """Stim's generated memory circuit, one operation for the whole shot.

    One probability on all four of Stim's noise channels, as Stim's
    guide does (frontends/settings.py memory_circuit); rounds_per_shot
    is a round count or "<n>d", n rounds per unit of distance.
    """
    circuit, rounds = _generated_memory(
        code_task, rounds_per_shot, distance, physical_error_probability
    )
    operation = program_records.Operation(
        id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit
    )
    return workload_records.Workload(
        operations=(operation,), round_counts={1: rounds}
    )


def memory_patches(
    code_task: str,
    rounds_per_shot: Union[int, str],
    patch_count: int,
    distance: int,
    physical_error_probability: float,
) -> workload_records.Workload:
    """Independent memory patches run at once.

    patch_count copies of memory_circuit's circuit, each its own
    operation on its own patch and logical qubit, all from round one, so
    their windows share the decoder tiers' units, Elastic's "smaller pool
    of decoders" shared across logical qubits (2406.17995 lines 121-123).
    The copies sit side by side along x in one Stim coordinate frame,
    patch p shifted by p (2 d + 2) with a SHIFT_COORDS ahead of its
    circuit, so a surface-code patch, 2 d units wide, starts one lattice
    step past its neighbour and a burst region in that frame covers the
    patches it reaches (McEwen 2104.05219: a burst starts at one spot and
    spreads over the chip). Each copy draws its own shot.
    """
    if patch_count < 1:
        raise ValueError(
            f"workload.patch_count is at least 1, got {patch_count}; with "
            "no patch the run would finish having run nothing"
        )
    circuit, rounds = _generated_memory(
        code_task, rounds_per_shot, distance, physical_error_probability
    )
    pitch = 2 * distance + 2
    operations = []
    round_counts = {}
    for patch in range(patch_count):
        offset = patch * pitch
        shift = stim.Circuit(f"SHIFT_COORDS({offset}, 0)")
        placed = shift + circuit
        operation_id = patch + 1
        operation = program_records.Operation(
            id=operation_id,
            name=f"memory{patch}",
            qubits=(patch,),
            patches=(patch,),
            circuit=placed,
        )
        operations.append(operation)
        round_counts[operation_id] = rounds
    return workload_records.Workload(
        operations=tuple(operations), round_counts=round_counts
    )


def deltakit_memory(
    rounds: int,
    distance: int,
    physical_error_probability: float,
    family: str = "rotated_surface",
    basis: str = "Z",
    patch: str = "memory-patch",
) -> workload_records.Workload:
    """Deltakit's finite memory, one operation for the whole circuit.

    The Explorer's css_code_memory_circuit with SD6 noise at the sweep's
    probability, exported with its measurement-to-round map
    (frontends/deltakit.py memory_circuit), as examples/deltakit_example.py
    builds it by hand. The optional deltakit extra is imported only when
    this maker runs.
    """
    circuit, measurement_rounds = deltakit.memory_circuit(
        family, distance, rounds, basis, physical_error_probability
    )
    operation = program_records.Operation(
        1, "memory", (patch,), patches=(patch,)
    )
    physical = workload_records.FiniteCircuit(circuit, measurement_rounds)
    return workload_records.Workload((operation,), {1: rounds}, physical)


def deltakit_live_memory(
    distance: int,
    physical_error_probability: float,
    round_period_microseconds: float,
    decode_after_rounds: int,
    basis: str = "Z",
    noise_model: str = "sd6",
    relaxation_time_microseconds: Optional[float] = None,
    dephasing_time_microseconds: Optional[float] = None,
    patch: str = "memory-patch",
) -> workload_records.Workload:
    """A live Deltakit memory decoded after some rounds, then read out.

    The four fragments of frontends/deltakit.py memory_rounds, their
    period the sweep's, run as the live program live_memory builds.
    """
    program = deltakit.memory_rounds(
        "rotated_surface",
        distance,
        basis,
        physical_error_probability,
        round_period_microseconds=round_period_microseconds,
        noise_model=noise_model,
        relaxation_time_microseconds=relaxation_time_microseconds,
        dephasing_time_microseconds=dephasing_time_microseconds,
    )
    return live_memory(program, decode_after_rounds, patch)


def live_memory(
    program: circuit_records.RepeatedStimCircuit,
    decode_after_rounds: int,
    patch: str = "memory-patch",
) -> workload_records.Workload:
    """Live fragments decoded after some rounds, then read out.

    The prefix runs the stream's first rounds and is decoded; the patch
    waits, protected, until the decoded result releases the one round
    that resumes it, and the readout ends the stream.
    """
    patches = (patch,)
    prefix = program_records.Operation(
        1,
        "prefix",
        patches,
        patches=patches,
        stream_id=LIVE_STREAM_ID,
        stream_offset=0,
    )
    protect = program_records.Operation(
        2, "protect", patches, patches=patches, emits_detector_data=False
    )
    resume = program_records.Operation(
        3,
        "resume",
        patches,
        patches=patches,
        blocked_by=1,
        emits_detector_data=False,
    )
    readout = program_records.Operation(
        4, "readout", patches, patches=patches, emits_detector_data=False
    )
    operations = (prefix, protect, resume, readout)
    round_counts = {1: decode_after_rounds, 2: 0, 3: 1, 4: 0}
    return workload_records.Workload(operations, round_counts, program)


def _generated_memory(
    code_task: str,
    rounds_per_shot: Union[int, str],
    distance: int,
    physical_error_probability: float,
) -> tuple:
    """Stim's memory circuit at the sweep's point, and its round count."""
    shot_length = workload_settings.RoundsPerShot.from_yaml(rounds_per_shot)
    rounds = shot_length.rounds_for(distance)
    circuit = workload_settings.memory_circuit(
        code_task, rounds, distance, physical_error_probability
    )
    return circuit, rounds
