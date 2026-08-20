"""A tour of the Stim circuit: what goes in, what comes out, how to break it.

The simulator's QPU replays a Stim circuit. This script builds the same
d=3, 12-round memory circuit the small example uses and shows, with zero
noise so every outcome is deterministic:

  1. the circuit itself, and how many detectors it produces per round,
  2. that a clean run fires no detectors,
  3. one injected X error: exactly two detectors fire and the decoder
     corrects it,
  4. a chain of d injected X errors crossing the patch: zero detectors
     fire, the logical observable flips, and no decoder can see it.

    PYTHONPATH=. python guide/walkthrough/stim_circuit_tour.py
"""

from collections import Counter

import numpy as np
import pymatching
import stim

DISTANCE = 3
ROUNDS = 12
BREAK_AFTER_ROUND = 6      # errors are injected between round 6 and round 7


def noiseless_circuit() -> stim.Circuit:
    return stim.Circuit.generated("surface_code:rotated_memory_z",
                                  rounds=ROUNDS, distance=DISTANCE)


def decoding_graph() -> pymatching.Matching:
    """PyMatching needs a decoding graph, and the graph comes from a noise
    model: the same circuit with a small error probability on every channel.
    The graph then decodes detection events from anywhere, including the
    deterministic broken circuits below."""
    noisy = stim.Circuit.generated(
        "surface_code:rotated_memory_z", rounds=ROUNDS, distance=DISTANCE,
        after_clifford_depolarization=0.001, before_round_data_depolarization=0.001,
        before_measure_flip_probability=0.001, after_reset_flip_probability=0.001)
    return pymatching.Matching.from_detector_error_model(
        noisy.detector_error_model(decompose_errors=True))


def data_qubits(circuit: stim.Circuit) -> list:
    """The data qubits are the targets of the final measurement."""
    final_measurement = [instruction for instruction in circuit.flattened()
                         if instruction.name == "M"][-1]
    return [target.value for target in final_measurement.targets_copy()]


def logical_x_chain(circuit: stim.Circuit) -> list:
    """A vertical chain of X errors crossing the patch is the logical X
    operator: it commutes with every stabilizer, so no detector ever fires,
    yet it flips the measured logical Z. Any column of data qubits works;
    this takes the leftmost."""
    coordinates = circuit.get_final_qubit_coordinates()
    columns = {}
    for qubit in data_qubits(circuit):
        x_coordinate = coordinates[qubit][0]
        columns.setdefault(x_coordinate, []).append(qubit)
    leftmost_column = min(columns)
    return columns[leftmost_column]


def with_x_errors(circuit: stim.Circuit, qubits: list, after_round: int) -> stim.Circuit:
    """The circuit with X_ERROR(1), an always-fired X error, on `qubits`
    between `after_round` and the next round.

    It must be an error channel, not a plain X gate: a detection event is a
    deviation from the circuit's own noiseless reference, so a deterministic
    gate is absorbed into the reference and fires nothing. Rounds are counted
    by their MR (measure and reset the ancillas) instruction; the error adds
    no measurement records, so every detector definition is untouched."""
    broken = stim.Circuit()
    rounds_measured = 0
    inserted = False
    for instruction in circuit.flattened():
        broken.append(instruction)
        if instruction.name == "MR":
            rounds_measured += 1
            if rounds_measured == after_round and not inserted:
                broken.append("X_ERROR", qubits, 1.0)
                inserted = True
    return broken


def sample(circuit: stim.Circuit) -> tuple:
    """(detection events, observable flip) of one shot. With no random noise
    in the circuit the sample is exact, not statistical."""
    events, observable_flips = circuit.compile_detector_sampler().sample(
        shots=1, separate_observables=True)
    return events[0], bool(observable_flips[0][0])


def detectors_per_round(circuit: stim.Circuit) -> Counter:
    """How many detectors compare each round: the third detector coordinate
    is the round (0-based); the final data measurement adds one last layer."""
    rounds = Counter()
    for coordinates in circuit.get_detector_coordinates().values():
        rounds[int(coordinates[2])] += 1
    return rounds


def describe_fired(circuit: stim.Circuit, events: np.ndarray) -> list:
    coordinates = circuit.get_detector_coordinates()
    return [f"detector {index} at (x={coordinates[index][0]:g}, "
            f"y={coordinates[index][1]:g}, round={int(coordinates[index][2]) + 1})"
            for index in np.flatnonzero(events)]


def main() -> None:
    circuit = noiseless_circuit()
    matching = decoding_graph()

    print("=== 1. What goes in: the circuit ===")
    print(circuit)
    print()
    print(f"qubits: {circuit.num_qubits}, data qubits: {sorted(data_qubits(circuit))}")
    print(f"detectors: {circuit.num_detectors}, logical observables: {circuit.num_observables}")
    tally = detectors_per_round(circuit)
    print("detectors per round (round: count):",
          {round_index + 1: count for round_index, count in sorted(tally.items())})
    print()

    print("=== 2. What comes out: one clean shot ===")
    events, observable_flip = sample(circuit)
    print(f"detection events fired: {int(events.sum())} of {events.size}")
    print(f"logical observable flipped: {observable_flip}")
    print()

    print("=== 3. Break a detector: one X error on the middle data qubit ===")
    middle_qubit = 10                    # coordinates (3, 3), the patch center
    one_error = with_x_errors(circuit, [middle_qubit], BREAK_AFTER_ROUND)
    events, observable_flip = sample(one_error)
    print(f"injected: X_ERROR(1) on qubit {middle_qubit} between rounds "
          f"{BREAK_AFTER_ROUND} and {BREAK_AFTER_ROUND + 1}")
    for line in describe_fired(one_error, events):
        print("fired:", line)
    predicted_flip = bool(matching.decode(events)[0])
    print(f"observable actually flipped: {observable_flip}")
    print(f"decoder predicts a flip:     {predicted_flip}")
    print(f"logical error after correction: {observable_flip != predicted_flip}")
    print()

    print("=== 4. Break the code: a chain of X errors crossing the patch ===")
    chain = logical_x_chain(circuit)
    logical_error = with_x_errors(circuit, chain, BREAK_AFTER_ROUND)
    events, observable_flip = sample(logical_error)
    print(f"injected: X_ERROR(1) on qubits {chain}, a full column, between rounds "
          f"{BREAK_AFTER_ROUND} and {BREAK_AFTER_ROUND + 1}")
    print(f"detection events fired: {int(events.sum())} of {events.size}")
    predicted_flip = bool(matching.decode(events)[0])
    print(f"observable actually flipped: {observable_flip}")
    print(f"decoder predicts a flip:     {predicted_flip}")
    print(f"logical error after correction: {observable_flip != predicted_flip}")


if __name__ == "__main__":
    main()
