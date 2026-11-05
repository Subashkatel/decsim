"""What the referent tests share: the windows of a memory and their jobs."""

import numpy
import stim

import decsim.detector_error_model.window_model_builders as builders
import decsim.detector_error_model.window_slicer as window_slicer
import decsim.records.decoding as decoding_records
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
    slicer = window_slicer.WindowSlicer(
        circuit, round_count=rounds, fault_model_requirement=requirement
    )
    return builders.build_single_window_error_model(slicer, (1, rounds, rounds))


def sliding_windows(distance: int, p: float, requirement) -> tuple:
    """A memory of 4d rounds decoded in three windows, each buffered d rounds.

    The first commits rounds 1 to d. The middle one commits d+1 to 2d,
    and its buffer ends at round 3d, before the memory does, so some of
    its faults reach its future. The last commits the rest. Each window
    leaves out the faults the earlier ones committed, as a sliding
    decode does.
    """
    rounds = 4 * distance
    circuit = memory_circuit(distance, rounds, p)
    plan = [
        (1, distance, 2 * distance),
        (distance + 1, 2 * distance, 3 * distance),
        (2 * distance + 1, rounds, rounds),
    ]
    models = builders.build_window_error_models(
        circuit,
        plan,
        round_count=rounds,
        fault_model_requirement=requirement,
        fault_exclusion_ranges=(),
    )
    return circuit, models


def row_syndrome(model, shot):
    """One shot's detection events in the window's row order, as uint8."""
    rows = list(model.detector_ids)
    events = numpy.asarray(shot)
    return events[rows].astype(numpy.uint8)


def job_for(model, shot, window_id: int = 0) -> decoding_records.DecodeJob:
    """A decode job carrying one shot's detection events in row order."""
    syndrome = row_syndrome(model, shot)
    bits = bit_tuple(syndrome)
    payload = round_records.RetainedSyndromeFragment(
        stream_id=1,
        patch_ids=(0,),
        round_index=1,
        bits=bits,
        size_bits=len(bits),
        fragment_index=0,
    )
    return decoding_records.DecodeJob(
        stream_id=1,
        window_id=window_id,
        round_count=1,
        detector_error_model=model,
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
