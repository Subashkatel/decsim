"""What the referent tests share: a whole-circuit window and its jobs."""

import numpy
import stim

import decsim.detector_error_model.window_model_builders as builders
import decsim.message as message
import decsim.records.rounds as round_records


def memory_circuit(distance: int, rounds: int, p: float) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p,
    )


def whole_circuit_window(circuit: stim.Circuit, rounds: int, requirement):
    """One window over every round; it owns every fault."""
    return builders.build_single_window_error_model(
        circuit,
        (1, rounds, rounds),
        round_count=rounds,
        fault_model_requirement=requirement,
    )


def row_syndrome(model, shot):
    """One shot's detection events in the window's row order, as uint8."""
    rows = list(model.detector_ids)
    events = numpy.asarray(shot)
    return events[rows].astype(numpy.uint8)


def job_for(model, shot, window_id: int = 0) -> message.DecodeJob:
    """A decode job carrying one shot's detection events in row order."""
    syndrome = row_syndrome(model, shot)
    bits = bit_tuple(syndrome)
    payload = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_id=0,
        round_index=1,
        bits=bits,
        size_bits=len(bits),
        fragment_index=0,
    )
    return message.DecodeJob(
        op_id=1,
        window_id=window_id,
        n_rounds=1,
        dem=model,
        payloads=[payload],
        label=f"W{window_id}",
    )


def bit_tuple(values) -> tuple:
    bits = []
    for value in values:
        bits.append(int(value))
    return tuple(bits)


def sampled_shots(circuit: stim.Circuit, shots: int, seed: int):
    sampler = circuit.compile_detector_sampler(seed=seed)
    return sampler.sample(shots, separate_observables=True)
