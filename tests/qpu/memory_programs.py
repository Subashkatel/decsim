"""Repeated memory fragments from Stim's generated surface-code circuit.

The generator supplies one preparation, a repeated extraction round and a
terminal data readout. Their assembled circuit is checked against Stim.
"""

import stim

import decsim.records.circuits as circuit_records


def memory_program(
    distance: int = 3,
    basis: str = "Z",
    physical_error_probability: float = 0.003,
) -> circuit_records.RepeatedStimCircuit:
    """Extract ordinary Stim protection rounds without changing their gates."""
    circuit = memory_circuit(4, distance, basis, physical_error_probability)
    repeated_indices = [
        index
        for index, instruction in enumerate(circuit)
        if isinstance(instruction, stim.CircuitRepeatBlock)
    ]
    assert len(repeated_indices) == 1
    repeat_index = repeated_indices[0]
    block = circuit[repeat_index]
    first = circuit[:repeat_index]
    repeated = block.body_copy()
    after_repeat = repeat_index + 1
    readout = circuit[after_repeat:]
    final = repeated + readout
    single = first + readout
    return circuit_records.RepeatedStimCircuit(first, repeated, final, single)


def memory_circuit(
    round_count: int,
    distance: int = 3,
    basis: str = "Z",
    physical_error_probability: float = 0.003,
) -> stim.Circuit:
    """Generate the complete-circuit referent with the same noise settings."""
    basis_suffix = basis.lower()
    task = f"surface_code:rotated_memory_{basis_suffix}"
    return stim.Circuit.generated(
        task,
        distance=distance,
        rounds=round_count,
        after_clifford_depolarization=physical_error_probability,
        before_measure_flip_probability=physical_error_probability,
    )


def joint_repetition_program(
    is_entangled: bool,
    physical_error_probability: float = 0.001,
) -> circuit_records.RepeatedStimCircuit:
    """Two length-three repetition blocks in one Stim tableau history.

    Each block measures its two neighbouring ZZ checks, the same circuit
    law as Stim's repetition_code:memory. Transversal CX copies the first
    block's logical X superposition into the second. Final ZZ parity is
    deterministic although either block's individual readout is random.
    This is a bit-flip-code example, not full Pauli fault tolerance.
    The explicit schedule stays together so each detector lookback can be
    checked against the neighbouring extraction and readout instructions.
    """
    prepare = stim.Circuit("R 0 1 2 3 4 5")
    if is_entangled:
        prepare += stim.Circuit("H 0\nCX 0 1 1 2\nCX 0 3 1 4 2 5")
    extraction = stim.Circuit(
        f"X_ERROR({physical_error_probability}) 0 1 2 3 4 5\n"
        "R 6 7 8 9\nCX 0 6 1 7 3 8 4 9\n"
        "CX 1 6 2 7 4 8 5 9\nM 6 7 8 9"
    )
    initial_detectors = stim.Circuit(
        "DETECTOR rec[-4]\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]"
    )
    bulk_detectors = stim.Circuit(
        "DETECTOR rec[-4] rec[-8]\nDETECTOR rec[-3] rec[-7]\n"
        "DETECTOR rec[-2] rec[-6]\nDETECTOR rec[-1] rec[-5]"
    )
    readout = stim.Circuit(
        "M 0 1 2 3 4 5\n"
        "DETECTOR rec[-6] rec[-5] rec[-10]\n"
        "DETECTOR rec[-5] rec[-4] rec[-9]\n"
        "DETECTOR rec[-3] rec[-2] rec[-8]\n"
        "DETECTOR rec[-2] rec[-1] rec[-7]"
    )
    if is_entangled:
        readout += stim.Circuit("OBSERVABLE_INCLUDE(0) rec[-6] rec[-3]")
    else:
        readout += stim.Circuit(
            "OBSERVABLE_INCLUDE(0) rec[-6]\nOBSERVABLE_INCLUDE(1) rec[-3]"
        )
    first = prepare + extraction
    first += initial_detectors
    repeated = extraction + bulk_detectors
    final = repeated + readout
    single = first + readout
    return circuit_records.RepeatedStimCircuit(first, repeated, final, single)
