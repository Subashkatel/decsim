"""Compile finite rotated-code experiments through Deltakit's CircuitBuilder.

Explorer's public stabiliser schedules define extraction; the exporter reduces
public terminal MeasurementReg outputs using the declared logical support.
"""

import contextlib
import importlib.util
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import stim

import decsim.frontends.deltakit as deltakit

if TYPE_CHECKING:
    import deltakit_circuit as physical_api
    import deltakit_compile.frontend.circuit as circuit_api
    import deltakit_explorer.codes as codes


def compile_experiment(
    experiment: str,
    distance: int,
    round_count: int,
    basis: str,
    physical_error_probability: float,
) -> tuple[stim.Circuit, dict[int, int]]:
    """Export memory or terminal Hadamard with SD6 noise and explicit rounds.

    Hadamard acts transversally and immediately measures the conjugate basis.
    It does not continue extraction in the original patch orientation. The
    exporter declares the logical parity from public code metadata; the
    compiler returns terminal measurements, not an inferred logical output.
    SD6 has no duration calibration. The caller owns QPU and decoder timing.
    """
    _require_compiler()
    _check_parameters(experiment, distance, round_count, basis)
    deltakit.check_probability(physical_error_probability)
    schedule = _code_schedule(distance, basis)
    final_basis = basis
    if experiment == "hadamard":
        final_basis = {"X": "Z", "Z": "X"}[basis]
    extraction = _extraction_circuit(schedule, round_count, basis)
    readout = _readout_circuit(schedule, experiment, final_basis)
    circuit = _compile_program(schedule, extraction, readout)
    circuit = _add_noise(circuit, physical_error_probability)
    measurement_rounds, terminal_records = _measurement_records(
        circuit, schedule, round_count, final_basis
    )
    _declare_logical_output(circuit, schedule, terminal_records)
    return circuit, measurement_rounds


@dataclass(frozen=True)
class _CodeSchedule:
    """Ordered qubits, extraction checks and supports share declared indices."""

    coordinates: tuple[tuple[float, ...], ...]
    data_count: int
    checks: tuple[tuple[Optional[tuple[str, int]], ...], ...]
    known_supports: dict[int, tuple[int, ...]]
    logical_support: tuple[int, ...]


def _require_compiler() -> None:
    for module in ("deltakit_compile", "deltakit_explorer"):
        specification = importlib.util.find_spec(module)
        if specification is None:
            raise ValueError(
                "Deltakit compiler requires the optional "
                "decsim[deltakit-compile] extra"
            )


def _check_parameters(
    experiment: str, distance: int, round_count: int, basis: str
) -> None:
    deltakit.check_positive_integer(distance, "distance")
    deltakit.check_positive_integer(round_count, "round_count")
    if distance < 2:
        raise ValueError("compiler distance must be at least two")
    if basis not in ("X", "Z"):
        raise ValueError("compiler basis must be X or Z")
    if experiment not in ("memory", "hadamard"):
        raise ValueError("experiment must be memory or hadamard")


def _code_schedule(distance: int, basis: str) -> _CodeSchedule:
    import deltakit_explorer.codes as codes

    code = codes.RotatedPlanarCode(width=distance, height=distance)
    data_qubits = sorted(
        code.data_qubits, key=lambda qubit: qubit.unique_identifier
    )
    ancillas = sorted(
        code.ancilla_qubits, key=lambda qubit: qubit.unique_identifier
    )
    qubits = data_qubits + ancillas
    indices = {qubit: index for index, qubit in enumerate(qubits)}
    coordinates = tuple(tuple(qubit.unique_identifier) for qubit in qubits)
    assert len(code.stabilisers) == 1, "Rotated code has one extraction group"
    stabilisers = {check.ancilla_qubit: check for check in code.stabilisers[0]}
    checks = tuple(
        _check_schedule(stabilisers[ancilla], indices) for ancilla in ancillas
    )
    known_supports = _known_supports(checks, basis)
    logical_operators = {
        "X": code.x_logical_operators,
        "Z": code.z_logical_operators,
    }
    logical_support = tuple(
        indices[pauli.qubit] for pauli in logical_operators[basis][0]
    )
    return _CodeSchedule(
        coordinates, len(data_qubits), checks, known_supports, logical_support
    )


def _check_schedule(
    stabiliser: "codes.Stabiliser",
    indices: dict["physical_api.Qubit", int],
) -> tuple[Optional[tuple[str, int]], ...]:
    schedule = []
    for pauli in stabiliser.paulis:
        entry = None
        if pauli is not None:
            entry = (pauli.stim_identifier, indices[pauli.qubit])
        schedule.append(entry)
    return tuple(schedule)


def _known_supports(
    checks: tuple[tuple[Optional[tuple[str, int]], ...], ...], basis: str
) -> dict[int, tuple[int, ...]]:
    known = {}
    for offset, check in enumerate(checks):
        support = tuple(
            entry[1]
            for entry in check
            if entry is not None and entry[0] == basis
        )
        if support:
            known[offset] = support
    return known


def _extraction_circuit(
    schedule: _CodeSchedule, round_count: int, basis: str
) -> "circuit_api.Circuit":
    import deltakit_compile.frontend.circuit as circuit_api

    builder = circuit_api.CircuitBuilder()
    register_type = circuit_api.QubitReg(len(schedule.coordinates))
    register = builder.add_arg(register_type)
    preparation_gate = "R" + basis
    builder.gate(preparation_gate, register[: schedule.data_count])
    previous = None
    for _ in range(round_count):
        builder.gate("RX", register[schedule.data_count :])
        _extract_checks(builder, register, schedule)
        measurements = builder.measure("X", register[schedule.data_count :])
        _round_detectors(builder, measurements, previous, schedule)
        previous = measurements
    builder.add_return(previous)
    return builder.build("prepare_and_extract")


def _extract_checks(
    builder: "circuit_api.CircuitBuilder",
    register: "circuit_api.QubitReg",
    schedule: _CodeSchedule,
) -> None:
    import deltakit_compile.frontend.circuit as circuit_api

    with builder.parallel(circuit_api.ParallelAlignment.LOCKSTEP) as parallel:
        for offset, check in enumerate(schedule.checks):
            ancilla_index = schedule.data_count + offset
            ancilla = register[ancilla_index]
            _schedule_check(builder, register, ancilla, check, parallel)


def _schedule_check(
    builder: "circuit_api.CircuitBuilder",
    register: "circuit_api.QubitReg",
    ancilla: "circuit_api.QubitReg",
    check: tuple[Optional[tuple[str, int]], ...],
    parallel: Callable[[], contextlib.AbstractContextManager],
) -> None:
    with parallel():
        for entry in check:
            _entangle_check(builder, register, ancilla, entry)


def _entangle_check(
    builder: "circuit_api.CircuitBuilder",
    register: "circuit_api.QubitReg",
    ancilla: "circuit_api.QubitReg",
    entry: Optional[tuple[str, int]],
) -> None:
    if entry is None:
        builder.gate("I", ancilla)
        return
    basis, index = entry
    gate = "C" + basis
    builder.gate(gate, [ancilla, register[index]])


def _round_detectors(
    builder: "circuit_api.CircuitBuilder",
    measurements: "circuit_api.MeasurementReg",
    previous: Optional["circuit_api.MeasurementReg"],
    schedule: _CodeSchedule,
) -> None:
    if previous is None:
        for offset in schedule.known_supports:
            builder.detector([measurements[offset]])
        return
    for offset in range(len(schedule.checks)):
        builder.detector([previous[offset], measurements[offset]])


def _readout_circuit(
    schedule: _CodeSchedule, experiment: str, final_basis: str
) -> "circuit_api.Circuit":
    import deltakit_compile.frontend.circuit as circuit_api

    builder = circuit_api.CircuitBuilder()
    register_type = circuit_api.QubitReg(len(schedule.coordinates))
    register = builder.add_arg(register_type)
    measurement_type = circuit_api.MeasurementReg(len(schedule.checks))
    previous = builder.add_arg(measurement_type)
    if experiment == "hadamard":
        builder.gate("H", register[: schedule.data_count])
    terminal = builder.measure(final_basis, register[: schedule.data_count])
    for offset, support in schedule.known_supports.items():
        parity = [terminal[index] for index in support]
        parity.append(previous[offset])
        builder.detector(parity)
    builder.add_return(terminal)
    return builder.build("logical_output")


def _compile_program(
    schedule: _CodeSchedule,
    extraction: "circuit_api.Circuit",
    readout: "circuit_api.Circuit",
) -> stim.Circuit:
    import deltakit_compile.frontend.circuit as circuit_api
    import deltakit_compile.frontend.circuit_builder as program_api
    import deltakit_compile.frontend.logical_assembler as assembler
    import deltakit_compile.passes.stabiliser.pipeline as flows
    import deltakit_compile.passes.stim.stim_export.pipeline as export

    program = program_api.CircuitProgramBuilder()
    register_type = circuit_api.QubitReg(
        len(schedule.coordinates), qubit_locations=list(schedule.coordinates)
    )
    register = program.declare_qubits(register_type)
    extraction_call = extraction(register)
    checks = program.call_circuit(extraction_call)
    readout_call = readout(register, checks)
    output = program.call_circuit(readout_call)
    program.add_return(output)
    export_configuration = export.StimExportPipelineConfig()
    flow_configuration = flows.StabiliserFlowPipelineConfig(
        generate_flows=False
    )
    configuration = assembler.LogicalAssemblerConfig(
        export_config=export_configuration,
        stabiliser_flow_config=flow_configuration,
        verify_between_passes=True,
    )
    compiler = assembler.LogicalAssembler(configuration)
    assembled = program.build_program()
    compiled = compiler.compile(assembled)
    return stim.Circuit(compiled.program)


def _add_noise(circuit: stim.Circuit, probability: float) -> stim.Circuit:
    import deltakit_circuit as circuit_api
    import deltakit_explorer.qpu as qpu
    import deltakit_stim as backend

    circuit_text = str(circuit)
    backend_circuit = backend.Circuit(circuit_text)
    physical = circuit_api.Circuit.from_stim_circuit(backend_circuit)
    native_gates = qpu.ExhaustiveGateSet()
    noise = qpu.SD6Noise(p=probability)
    device = qpu.QPU(
        physical.qubits,
        native_gates_and_times=native_gates,
        noise_model=noise,
        maximise_parallelism=False,
    )
    noisy = device.compile_and_add_noise_to_circuit(
        physical, remove_paulis=False
    )
    exported = noisy.as_stim_circuit()
    circuit_text = str(exported)
    return stim.Circuit(circuit_text)


def _measurement_records(
    circuit: stim.Circuit,
    schedule: _CodeSchedule,
    round_count: int,
    final_basis: str,
) -> tuple[dict[int, int], dict[int, int]]:
    declared_indices = _declared_indices(circuit, schedule)
    counts = dict.fromkeys(declared_indices, 0)
    measurement_rounds = {}
    terminal_records = {}
    for instruction, qubit in _measurements(circuit):
        index = declared_indices[qubit]
        counts[qubit] += 1
        absolute_index = len(measurement_rounds)
        round_index = _measurement_round(
            instruction,
            index,
            counts[qubit],
            schedule.data_count,
            round_count,
            final_basis,
        )
        measurement_rounds[absolute_index] = round_index
        if index < schedule.data_count:
            terminal_records[index] = absolute_index
    _check_record_counts(
        circuit, schedule, counts, declared_indices, round_count
    )
    return measurement_rounds, terminal_records


def _measurements(circuit: stim.Circuit) -> Iterator[tuple[str, int]]:
    flattened = circuit.flattened()
    for instruction in flattened:
        if instruction.name not in ("M", "MX"):
            continue
        targets = instruction.targets_copy()
        for target in targets:
            yield instruction.name, target.value


def _declared_indices(
    circuit: stim.Circuit, schedule: _CodeSchedule
) -> dict[int, int]:
    coordinates = circuit.get_final_qubit_coordinates()
    actual_qubits = {
        tuple(point): qubit for qubit, point in coordinates.items()
    }
    if set(actual_qubits) != set(schedule.coordinates):
        raise ValueError("compiler output changed the declared qubit support")
    return {
        actual_qubits[point]: index
        for index, point in enumerate(schedule.coordinates)
    }


def _measurement_round(
    instruction: str,
    index: int,
    count: int,
    data_count: int,
    round_count: int,
    final_basis: str,
) -> int:
    expected_instruction = "MX"
    if index < data_count:
        expected_instruction = {"X": "MX", "Z": "M"}[final_basis]
        count = round_count
    if instruction != expected_instruction:
        raise ValueError("compiler output changed a declared measurement basis")
    return count


def _check_record_counts(
    circuit: stim.Circuit,
    schedule: _CodeSchedule,
    counts: dict[int, int],
    declared_indices: dict[int, int],
    round_count: int,
) -> None:
    for qubit, index in declared_indices.items():
        expected_count = round_count
        if index < schedule.data_count:
            expected_count = 1
        if counts[qubit] != expected_count:
            raise ValueError(
                "compiler output changed the declared measurement count"
            )
    measurement_counts = counts.values()
    if sum(measurement_counts) != circuit.num_measurements:
        raise ValueError("compiler output contains undeclared measurements")


def _declare_logical_output(
    circuit: stim.Circuit,
    schedule: _CodeSchedule,
    terminal_records: dict[int, int],
) -> None:
    targets = []
    for index in schedule.logical_support:
        relative_index = terminal_records[index] - circuit.num_measurements
        target = stim.target_rec(relative_index)
        targets.append(target)
    circuit.append("OBSERVABLE_INCLUDE", targets, 0)
