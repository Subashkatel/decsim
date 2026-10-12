"""The workload settings: what the machine runs, and for how many rounds.

A Python caller hands a maker's records.workload.Workload to
WorkloadSettings.running, which lowers it into the operations the
machine issues (circuit_frontend.lowered), or hands the lowered fields
in directly. A workload lowered here was made at the distance the run
file gave its maker, so it fixes the qpu's distance; a maker row
(decsim/producers.py) is lowered at the task instead and follows it.
"""

import dataclasses
from typing import ClassVar, Optional

import stim

import decsim.frontends.circuit_frontend as circuit_frontend
import decsim.ports as ports
import decsim.records.workload as workload_records

FEEDBACK_BOUNDARY_MODES = ("trailing_buffer", "measurement_closed")
# A memory's data qubits are reset by R or RX and read out by M or MX,
# and Stim's generated memories put a flip on them after the one and
# before the other.
DATA_RESETS = ("R", "RX")
DATA_MEASUREMENTS = ("M", "MX")
BOUNDARY_FLIPS = ("X_ERROR", "Z_ERROR")


@dataclasses.dataclass(frozen=True)
class WorkloadSettings:
    """The program the machine runs: a maker's workload, lowered.

    fixes_the_distance and fields_set_in_the_run_file name what an
    assignment by path may not change (decsim/settings.py): the circuit
    was made at the qpu's distance and lowered to these fields.
    """

    fixes_the_distance: ClassVar[bool] = True
    fields_set_in_the_run_file: ClassVar[tuple] = (
        "operations",
        "decode_operations",
        "dynamic_streams",
        "protected_regions",
        "physical_circuits",
        "rounds_policy",
        "workload_record",
    )

    operations: tuple = ()
    decode_operations: Optional[tuple] = None
    dynamic_streams: tuple = ()
    protected_regions: tuple = ()
    rounds_policy: Optional[ports.RoundsPolicy] = None
    physical_circuits: tuple = ()
    feedback_boundary_mode: str = "trailing_buffer"
    # the record the fields above were lowered from, which a run folder
    # writes to its inputs (experiments/run_folder.py record_task)
    workload_record: Optional[workload_records.Workload] = None

    def __post_init__(self) -> None:
        if self.feedback_boundary_mode not in FEEDBACK_BOUNDARY_MODES:
            raise ValueError(
                "workload.feedback_boundary_mode must be one of "
                f"{FEEDBACK_BOUNDARY_MODES}, got "
                f"{self.feedback_boundary_mode!r}"
            )

    @classmethod
    def running(cls, workload: workload_records.Workload) -> "WorkloadSettings":
        """The settings that run a maker's workload, lowered for the machine.

        A Python caller states its workload here:
        WorkloadSettings.running(producers.memory_circuit(...)).
        """
        lowered = _lowered_fields(workload)
        return cls(**lowered)

    def lowered_for(self, qpu: object) -> "WorkloadSettings":
        """These settings: they were lowered when they were made."""
        del qpu
        return self


@dataclasses.dataclass(frozen=True)
class StimNoise:
    """Stim's four generated noise channels, each its own probability.

    The arguments of stim.Circuit.generated (Stim 1.16 docstring): the
    depolarizing after each Clifford gate, on each data qubit before a
    round, the flip before each measurement and after each reset. Each
    is 0 by default, and a channel at 0 is left out of the circuit. As a
    noise row (ports.CircuitNoise) the generator puts all of it in, at
    these probabilities whatever the memory's p, and nothing is added.
    """

    after_clifford_depolarization: float = 0.0
    before_round_data_depolarization: float = 0.0
    before_measure_flip_probability: float = 0.0
    after_reset_flip_probability: float = 0.0

    def generator_noise(self, probability: float) -> "StimNoise":
        """These channels; the memory's p is only its label."""
        del probability
        return self

    def noisy(self, circuit: stim.Circuit, probability: float) -> stim.Circuit:
        """The generated circuit as it is: Stim put all the noise in."""
        del probability
        return circuit


# The generator noise of a row that adds all its noise afterwards.
NO_GENERATOR_NOISE = StimNoise()


def memory_circuit(
    code_task: str,
    rounds: int,
    distance: int,
    probability: float,
    noise: Optional[ports.CircuitNoise] = None,
    noiseless_time_boundaries: bool = False,
) -> stim.Circuit:
    """Stim's generated memory circuit, made noisy by the noise row.

    noise None is one probability on all four of Stim's channels, as
    Stim's guide does; any row (ports.CircuitNoise) chooses the
    generator's channels and adds its own noise afterwards.
    noiseless_time_boundaries then takes the noise off the data qubits'
    first reset and last measurement, so the code state is prepared and
    read out without noise: the "perfect terminal time boundaries" of
    Toshio et al. 2510.25222 (lines 674, 1408), which follow Gidney et
    al. 2312.04522's noiseless time boundaries (line 1094).
    """
    if noise is None:
        noise = StimNoise(probability, probability, probability, probability)
    generator_noise = noise.generator_noise(probability)
    channels = dataclasses.asdict(generator_noise)
    generated = stim.Circuit.generated(
        code_task, rounds=rounds, distance=distance, **channels
    )
    circuit = noise.noisy(generated, probability)
    if not noiseless_time_boundaries:
        return circuit
    return _without_time_boundary_noise(circuit)


def _lowered_fields(workload: workload_records.Workload) -> dict:
    """The fields a maker's workload fills, lowered for the machine."""
    program = circuit_frontend.lowered(workload)
    return {
        "operations": program.operations,
        "dynamic_streams": program.dynamic_streams,
        "protected_regions": program.protected_regions,
        "rounds_policy": program.rounds_policy,
        "physical_circuits": program.physical_circuits,
        "workload_record": workload,
    }


def _without_time_boundary_noise(circuit: stim.Circuit) -> stim.Circuit:
    """The circuit without noise on its data's first reset and last read.

    The data qubits are the ones the last measurement reads out. The
    first flip after their first reset, past any TICK, loses them, as
    does a flip right before the last measurement, and that measurement
    loses its own flip argument (Deltakit writes M(p)). A flip's other
    qubits keep their noise; a channel at 0 left nothing to take off.
    """
    instructions = list(circuit)
    replaced = _boundary_replacements(instructions)
    kept = stim.Circuit()
    for index, instruction in enumerate(instructions):
        kept_instruction = replaced.get(index, instruction)
        if kept_instruction is not None:
            kept.append(kept_instruction)
    return kept


def _boundary_replacements(instructions: list) -> dict:
    """Each boundary instruction's index and what stands in its place.

    None stands in for a flip that was only on the data qubits.
    """
    readout_index = _last_data_measurement(instructions)
    readout = instructions[readout_index]
    data_qubits = _qubits_of(readout)
    reset_index = _first_reset_of(instructions, data_qubits)
    prepared_flip_index = _next_flip(instructions, reset_index)
    before_readout = readout_index - 1
    replaced = {readout_index: _without_arguments(readout)}
    for flip_index in (prepared_flip_index, before_readout):
        if flip_index is None:
            continue
        flip = instructions[flip_index]
        if flip.name in BOUNDARY_FLIPS:
            replaced[flip_index] = _without_qubits(flip, data_qubits)
    return replaced


def _last_data_measurement(instructions: list) -> int:
    """The index of the circuit's last top-level M or MX."""
    last_index = None
    for index, instruction in enumerate(instructions):
        if instruction.name in DATA_MEASUREMENTS:
            last_index = index
    if last_index is not None:
        return last_index
    raise ValueError(
        "noiseless time boundaries need a final data measurement (M or MX) "
        "at the circuit's top level, and this circuit has none"
    )


def _first_reset_of(instructions: list, data_qubits: frozenset) -> int:
    """The index of the first top-level reset of every data qubit."""
    for index, instruction in enumerate(instructions):
        if instruction.name not in DATA_RESETS:
            continue
        reset_qubits = _qubits_of(instruction)
        if reset_qubits >= data_qubits:
            return index
    raise ValueError(
        "noiseless time boundaries need the data qubits' first reset (an R "
        "or RX of every qubit the last measurement reads) at the "
        "circuit's top level, and this circuit has none"
    )


def _next_flip(instructions: list, reset_index: int) -> Optional[int]:
    """The index of the first flip after the reset, past TICKs; None if not."""
    first = reset_index + 1
    count = len(instructions)
    for index in range(first, count):
        name = instructions[index].name
        if name == "TICK":
            continue
        if name in BOUNDARY_FLIPS:
            return index
        return None
    return None


def _qubits_of(instruction: stim.CircuitInstruction) -> frozenset:
    """The qubit indices an instruction acts on."""
    targets = instruction.targets_copy()
    return frozenset(target.value for target in targets)


def _without_qubits(
    flip: stim.CircuitInstruction, qubits: frozenset
) -> Optional[stim.CircuitInstruction]:
    """The flip on its other qubits; None when it was only on these."""
    targets = flip.targets_copy()
    kept = [target for target in targets if target.value not in qubits]
    if not kept:
        return None
    arguments = flip.gate_args_copy()
    return stim.CircuitInstruction(flip.name, kept, arguments)


def _without_arguments(
    measurement: stim.CircuitInstruction,
) -> stim.CircuitInstruction:
    """The measurement with no flip probability of its own."""
    targets = measurement.targets_copy()
    return stim.CircuitInstruction(measurement.name, targets)
