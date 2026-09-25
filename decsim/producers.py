"""The workload makers decsim ships: Stim's generated memories.

A maker is a plain function a yaml names as module:function under
workload.kind producer. decsim hands it the sweep point's values its
parameters name (experiments/experiment.py SWEEP_AXES) and the yaml's
arguments, and it returns a records.workload.Workload. A maker written
outside decsim has the same shape; these are the ones that ship.
"""

import stim

import decsim.frontends.settings as workload_settings
import decsim.records.program as program_records
import decsim.records.workload as workload_records


def memory_circuit(
    code_task: str,
    rounds_per_shot,
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
    rounds_per_shot,
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


def _generated_memory(
    code_task: str,
    rounds_per_shot,
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
