"""Deltakit memory circuits exported into the supplied-circuit frontend.

Explorer's css_code_memory_circuit and CSSStage define the physical
history and measurement order. Canonical Stim circuits, measurement
schedules and declared cadence leave this optional provider.
"""

import dataclasses
import importlib.util
import itertools
import math
import sys
from typing import TYPE_CHECKING, Optional

import stim

import decsim.records.circuits as circuit_records

# The smallest memory whose export has a distinct first round, a repeated
# interior round and a terminal round: four rounds, so the interior repeat
# block holds two.
TEMPLATE_ROUND_COUNT = 4
TEMPLATE_REPEAT_COUNT = TEMPLATE_ROUND_COUNT - 2

if TYPE_CHECKING:
    import deltakit_circuit as circuit_api
    import deltakit_circuit.gates as gates
    import deltakit_explorer.codes as codes
    import deltakit_explorer.qpu as qpu


def memory_circuit(
    code_family: str,
    distance: int,
    round_count: int,
    basis: str,
    physical_error_probability: float,
) -> tuple[stim.Circuit, dict[int, int]]:
    """Export one single-patch memory with standard depolarising noise.

    Families are rotated_surface and repetition. X or Z selects the
    prepared and measured logical basis, and repetition uses checks of
    that basis. SD6 applies the supplied probability to gates, resets,
    measurements and idle locations; it is not a calibrated device model.

    The map assigns every absolute measurement index to a one-based
    round. The final destructive data readout joins the last packet.
    Deltakit is imported only here; simulation uses the existing source.
    Circuit and schedule are exported together to keep their order explicit.

    The native gates and the qubit numbering are memory_rounds', so one
    memory at one round count has one error model on both paths. SD6
    charges every idle location, and the Explorer's default gate set
    decomposes CZ and MX into more layers, so more idle locations, than
    its exhaustive set (Explorer qpu/_native_gate_set.py).
    """
    _require_explorer()
    _check_memory_parameters(distance, round_count, basis)
    check_probability(physical_error_probability)
    import deltakit_circuit.gates as gates
    import deltakit_explorer.qpu as qpu

    logical_basis = gates.PauliBasis[basis]
    code = _memory_code(code_family, distance, logical_basis)
    noise = qpu.SD6Noise(p=physical_error_probability)
    native_gates = qpu.ExhaustiveGateSet()
    device = qpu.QPU(
        code.qubits,
        native_gates_and_times=native_gates,
        noise_model=noise,
        maximise_parallelism=False,
    )
    compiled = _compile_memory(code, logical_basis, round_count, device)
    qubit_mapping = _qubit_mapping(code.qubits)
    circuit = _export_compiled_memory(compiled, device, qubit_mapping)
    measurement_rounds = _measurement_rounds(code, round_count, basis)
    if len(measurement_rounds) != circuit.num_measurements:
        raise ValueError(
            "Deltakit noise changed the memory measurement schedule"
        )
    return circuit, measurement_rounds


def memory_rounds(
    code_family: str,
    distance: int,
    basis: str,
    physical_error_probability: float,
    *,
    round_period_microseconds: float,
    noise_model: str = "sd6",
    relaxation_time_microseconds: Optional[float] = None,
    dephasing_time_microseconds: Optional[float] = None,
) -> circuit_records.RepeatedStimCircuit:
    """Export protection rounds whose declared cadence matches the QPU period.

    Native gates have equal duration, calibrated by the CSS schedule to the
    QPU round period, including preparation and destructive final readout.

    SD6 uses physical_error_probability at gates, resets, readout and idle
    locations. Physical noise uses it at gates, resets and measurement, with
    T1/T2 noise applied only to idle intervals. Physical requires relaxation
    and dephasing times; SD6 rejects them. T1/T2 use a Pauli approximation.
    """
    _require_explorer()
    check_positive_integer(distance, "distance")
    _check_basis(basis)
    import deltakit_circuit.gates as gates

    logical_basis = gates.PauliBasis[basis]
    code = _memory_code(code_family, distance, logical_basis)
    return css_memory_rounds(
        code,
        basis,
        physical_error_probability,
        round_period_microseconds=round_period_microseconds,
        noise_model=noise_model,
        relaxation_time_microseconds=relaxation_time_microseconds,
        dephasing_time_microseconds=dephasing_time_microseconds,
    )


def css_memory_rounds(
    code: "codes.StabiliserCode",
    basis: str,
    physical_error_probability: float,
    *,
    round_period_microseconds: float,
    noise_model: str = "sd6",
    relaxation_time_microseconds: Optional[float] = None,
    dephasing_time_microseconds: Optional[float] = None,
) -> circuit_records.RepeatedStimCircuit:
    """Export a supplied CSS code with every declared logical observable.

    The public Explorer code stays at this producer boundary. Its observable
    indices retain the supplied generator definitions, which need not be
    canonically paired X/Z logicals. All four physical fragments must fit the
    declared cadence. Noise follows memory_rounds, including its T1/T2 units.
    This function keeps setup together so its noise, indexing and physical
    timing are visibly shared by both finite templates.
    """
    _require_explorer()
    _check_basis(basis)
    check_probability(physical_error_probability)
    _positive_duration(round_period_microseconds, "round_period_microseconds")
    import deltakit_circuit.gates as gates

    logical_basis = gates.PauliBasis[basis]
    noise = _memory_noise(
        noise_model,
        physical_error_probability,
        relaxation_time_microseconds,
        dephasing_time_microseconds,
    )
    device = _memory_device(code, round_period_microseconds, noise)
    qubit_mapping = _qubit_mapping(code.qubits)
    template = _compile_memory(
        code, logical_basis, TEMPLATE_ROUND_COUNT, device
    )
    single = _compile_memory(code, logical_basis, 1, device)
    exported = _export_compiled_memory(template, device, qubit_mapping)
    single_round = _export_compiled_memory(single, device, qubit_mapping)
    fragments = _memory_fragments(
        exported, single_round, round_period_microseconds
    )
    _check_memory_cadence(template, single, device, fragments)
    return fragments


def bell_memory_rounds(
    distance: int,
    basis: str,
    physical_error_probability: float,
    *,
    round_period_microseconds: float,
    noise_model: str = "sd6",
    relaxation_time_microseconds: Optional[float] = None,
    dephasing_time_microseconds: Optional[float] = None,
) -> circuit_records.RepeatedStimCircuit:
    """Export joint XX or ZZ memory of two transversally prepared Bell patches.

    Both patches share one aggregate physical history. The first round
    prepares plus/zero data, applies transversal CX, and measures checks.
    Later rounds include one explicit all-qubit wait with SDK idle noise.
    Its duration makes every round match the first round's declared cadence.
    This is a uniform native-gate schedule, not lattice surgery or routing.

    Setup stays together to share noise, timing and qubit indexing between
    the finite templates and their reusable physical fragments.
    """
    _require_explorer()
    check_positive_integer(distance, "distance")
    _check_basis(basis)
    check_probability(physical_error_probability)
    _positive_duration(round_period_microseconds, "round_period_microseconds")
    left, right = _bell_patches(distance)
    template = _bell_experiment(left, right, basis, TEMPLATE_ROUND_COUNT)
    single = _bell_experiment(left, right, basis, 1)
    noise = _memory_noise(
        noise_model,
        physical_error_probability,
        relaxation_time_microseconds,
        dephasing_time_microseconds,
    )
    device = _calibrated_device(
        single.qubits, single, round_period_microseconds, noise
    )
    template = device.compile_circuit(template)
    single = device.compile_circuit(single)
    mapping = _qubit_mapping(single.qubits)
    exported = _export_compiled_memory(template, device, mapping)
    single_round = _export_compiled_memory(single, device, mapping)
    fragments = _memory_fragments(
        exported, single_round, round_period_microseconds
    )
    fragments = _bell_initial_detectors(fragments, left, right, mapping)
    return _bell_wait_fragments(fragments, template, single, device, mapping)


def check_positive_integer(value: int, name: str) -> None:
    """Refuse a distance or round count that is not a positive int."""
    value_type = type(value)
    if value_type is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def check_probability(probability: float) -> None:
    """Refuse a physical error probability outside [0, 1] or not finite."""
    if not math.isfinite(probability):
        raise ValueError("physical_error_probability must be finite")
    if not 0 <= probability <= 1:
        raise ValueError("physical_error_probability must lie in [0, 1]")


def _require_explorer() -> None:
    if sys.version_info < (3, 10):
        raise ValueError("Deltakit memory requires Python 3.10 or newer")
    specification = importlib.util.find_spec("deltakit_explorer")
    if specification is None:
        raise ValueError(
            "Deltakit memory requires the optional decsim[deltakit] extra"
        )


def _check_memory_parameters(
    distance: int, round_count: int, basis: str
) -> None:
    check_positive_integer(distance, "distance")
    check_positive_integer(round_count, "round_count")
    _check_basis(basis)


def _check_basis(basis: str) -> None:
    if basis not in ("X", "Z"):
        raise ValueError("memory basis must be X or Z")


def _memory_code(
    code_family: str, distance: int, basis: "gates.PauliBasis"
) -> "codes.StabiliserCode":
    import deltakit_explorer.codes as codes

    if code_family == "rotated_surface":
        return codes.RotatedPlanarCode(width=distance, height=distance)
    if code_family == "repetition":
        return codes.RepetitionCode(distance=distance, stabiliser_type=basis)
    raise ValueError("code_family must be rotated_surface or repetition")


def _measurement_rounds(code, round_count: int, basis: str) -> dict[int, int]:
    stage = code.measure_stabilisers(num_rounds=1)
    one_round = stage.first_round.as_stim_circuit()
    measure_logicals = code.measure_z_logicals
    if basis == "X":
        measure_logicals = code.measure_x_logicals
    terminal_stage = measure_logicals()
    terminal = terminal_stage.first_round.as_stim_circuit()
    rounds = []
    after_last_round = round_count + 1
    for round_index in range(1, after_last_round):
        measurements = [round_index] * one_round.num_measurements
        rounds.extend(measurements)
    terminal_rounds = [round_count] * terminal.num_measurements
    rounds.extend(terminal_rounds)
    measurement_items = enumerate(rounds)
    return dict(measurement_items)


def _positive_duration(duration: float, name: str) -> None:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _memory_noise(
    name: str,
    probability: float,
    relaxation_microseconds: Optional[float],
    dephasing_microseconds: Optional[float],
) -> "qpu.NoiseParameters":
    import deltakit_explorer.qpu as qpu

    if name == "sd6":
        if (
            relaxation_microseconds is not None
            or dephasing_microseconds is not None
        ):
            raise ValueError("SD6 does not use relaxation or dephasing times")
        return qpu.SD6Noise(p=probability)
    if name != "physical":
        raise ValueError("noise_model must be sd6 or physical")
    return _physical_noise(
        probability, relaxation_microseconds, dephasing_microseconds
    )


def _physical_noise(
    probability: float,
    relaxation_microseconds: Optional[float],
    dephasing_microseconds: Optional[float],
) -> "qpu.PhysicalNoise":
    """The Explorer's T1/T2 idle noise with one probability at every gate.

    Idle noise per interval is the Pauli channel of Ghosh et al.
    1210.5799 equation 10 (Explorer qpu/_noise/_noise_parameters.py,
    which also refuses a T2 of twice T1 or more). The Explorer takes a
    separate probability for one-qubit gates, two-qubit gates, resets,
    measurement and readout flips; this frontend puts the one supplied
    probability at all five, a simplification of its own, so the T1/T2
    times are the only place the two models differ.
    """
    import deltakit_explorer.qpu as qpu

    if relaxation_microseconds is None or dephasing_microseconds is None:
        raise ValueError(
            "Physical noise requires relaxation and dephasing times"
        )
    _positive_duration(relaxation_microseconds, "relaxation_time_microseconds")
    _positive_duration(dephasing_microseconds, "dephasing_time_microseconds")
    relaxation_seconds = relaxation_microseconds * 1e-6
    dephasing_seconds = dephasing_microseconds * 1e-6
    return qpu.PhysicalNoise(
        t1=relaxation_seconds,
        t2=dephasing_seconds,
        p_1_qubit_gate_error=probability,
        p_2_qubit_gate_error=probability,
        p_reset_error=probability,
        p_meas_qubit_error=probability,
        p_readout_flip=probability,
    )


def _memory_device(
    code: "codes.StabiliserCode",
    round_period_microseconds: float,
    noise: "qpu.NoiseParameters",
) -> "qpu.QPU":
    stage = code.measure_stabilisers(1)
    return _calibrated_device(
        code.qubits, stage.first_round, round_period_microseconds, noise
    )


def _calibrated_device(
    qubits: frozenset["circuit_api.Qubit"],
    reference: "circuit_api.Circuit",
    round_period_microseconds: float,
    noise: "qpu.NoiseParameters",
) -> "qpu.QPU":
    """A device whose native schedule of one round fills the round period.

    Every gate family gets the same duration, the period divided by the
    layers of the reference round, so a round of the schedule takes the
    declared period. A real cycle is dominated by measurement and reset,
    not by the gate layers (Google 2408.13687: a 1.1 us cycle), so this
    even split is the frontend's simplification, and it decides how the
    physical model's idle noise is spread over the round.
    """
    import deltakit_explorer.qpu as qpu

    native_gates = qpu.ExhaustiveGateSet()
    timing_device = qpu.QPU(
        qubits,
        native_gates_and_times=native_gates,
        maximise_parallelism=False,
    )
    compiled_round = timing_device.compile_circuit(reference)
    unit_round_seconds = timing_device.get_circuit_execution_time(
        compiled_round
    )
    round_seconds = round_period_microseconds * 1e-6
    gate_seconds = round_seconds / unit_round_seconds
    _set_gate_durations(native_gates, gate_seconds)
    return qpu.QPU(
        qubits,
        native_gates_and_times=native_gates,
        noise_model=noise,
        maximise_parallelism=False,
    )


def _set_gate_durations(
    native_gates: "qpu.NativeGateSetAndTimes", seconds: float
) -> None:
    families = (
        native_gates.one_qubit_gates,
        native_gates.two_qubit_gates,
        native_gates.reset_gates,
        native_gates.measurement_gates,
    )
    for family in families:
        for gate in family:
            family[gate] = seconds


def _qubit_mapping(
    qubits: frozenset["circuit_api.Qubit"],
) -> dict["circuit_api.Qubit", int]:
    ordered = sorted(qubits, key=_qubit_identifier)
    return {qubit: index for index, qubit in enumerate(ordered)}


def _qubit_identifier(qubit: "circuit_api.Qubit") -> str:
    return repr(qubit.unique_identifier)


def _compile_memory(
    code: "codes.StabiliserCode",
    basis: "gates.PauliBasis",
    round_count: int,
    device: "qpu.QPU",
) -> "circuit_api.Circuit":
    import deltakit_explorer.codes as codes

    original = codes.css_code_memory_circuit(code, round_count, basis)
    return device.compile_circuit(original)


def _check_memory_cadence(
    template: "circuit_api.Circuit",
    single: "circuit_api.Circuit",
    device: "qpu.QPU",
    fragments: circuit_records.RepeatedStimCircuit,
) -> None:
    measurement_counts = _template_measurement_counts(fragments)
    native_rounds = _native_memory_rounds(template, measurement_counts)
    native_rounds.append(single)
    period_microseconds = fragments.round_period_microseconds
    _check_fragment_durations(native_rounds, device, period_microseconds)


def _template_measurement_counts(
    fragments: circuit_records.RepeatedStimCircuit,
) -> tuple[int, ...]:
    return (
        fragments.first_round.num_measurements,
        fragments.repeated_round.num_measurements,
        fragments.repeated_round.num_measurements,
        fragments.final_round.num_measurements,
    )


def _native_memory_rounds(
    template: "circuit_api.Circuit", measurement_counts: tuple[int, ...]
) -> list["circuit_api.Circuit"]:
    """Keep native layers at declared measurement boundaries.

    CSSStage ends extraction with ancilla readout; its following annotations
    add no physical time. Whole native layers retain concurrent reset gates.
    """
    import deltakit_circuit as circuit_api

    cumulative_counts = itertools.accumulate(measurement_counts)
    boundaries = set(cumulative_counts)
    flattened = template.flatten()
    gate_layers = flattened.gate_layers()
    fragments = []
    fragment = circuit_api.Circuit()
    measurement_count = 0
    for layer in gate_layers:
        fragment.append_layers(layer)
        measurement_count += len(layer.measurement_gates)
        if not layer.measurement_gates:
            continue
        if measurement_count in boundaries:
            fragments.append(fragment)
            fragment = circuit_api.Circuit()
    if len(fragments) != len(measurement_counts) or fragment.layers:
        raise ValueError("CSS measurement rounds must end on physical layers")
    return fragments


def _check_fragment_durations(
    fragments: list["circuit_api.Circuit"],
    device: "qpu.QPU",
    period_microseconds: float,
) -> None:
    period_seconds = period_microseconds * 1e-6
    for fragment in fragments:
        duration_seconds = device.get_circuit_execution_time(fragment)
        if not math.isclose(duration_seconds, period_seconds):
            raise ValueError(
                "CSS memory fragments must fit the declared cadence"
            )


def _export_compiled_memory(
    compiled: "circuit_api.Circuit",
    device: "qpu.QPU",
    qubit_mapping: dict["circuit_api.Qubit", int],
) -> stim.Circuit:
    noisy = device.compile_and_add_noise_to_circuit(compiled)
    exported = noisy.as_stim_circuit(qubit_mapping=qubit_mapping)
    circuit_text = str(exported)
    return stim.Circuit(circuit_text)


def _memory_fragments(
    template: stim.Circuit,
    single_round: stim.Circuit,
    period_microseconds: float,
) -> circuit_records.RepeatedStimCircuit:
    repeat_indices = []
    for index, instruction in enumerate(template):
        if isinstance(instruction, stim.CircuitRepeatBlock):
            repeat_indices.append(index)
    if len(repeat_indices) != 1:
        raise ValueError(
            "Deltakit memory must export one syndrome repeat block"
        )
    boundary = repeat_indices[0]
    repeat_block = template[boundary]
    if repeat_block.repeat_count != TEMPLATE_REPEAT_COUNT:
        raise ValueError(
            "Deltakit four-round memory must repeat two interior rounds"
        )
    first_round = template[:boundary]
    repeated_round = repeat_block.body_copy()
    final_start = boundary + 1
    final_round = template[final_start:]
    _check_fragment_measurements(
        first_round, repeated_round, final_round, single_round
    )
    return circuit_records.RepeatedStimCircuit(
        first_round=first_round,
        repeated_round=repeated_round,
        final_round=final_round,
        single_round=single_round,
        round_period_microseconds=period_microseconds,
    )


def _check_fragment_measurements(
    first_round: stim.Circuit,
    repeated_round: stim.Circuit,
    final_round: stim.Circuit,
    single_round: stim.Circuit,
) -> None:
    if first_round.num_measurements != repeated_round.num_measurements:
        raise ValueError(
            "Deltakit initial and repeated syndrome widths must match"
        )
    if final_round.num_measurements != single_round.num_measurements:
        raise ValueError(
            "Deltakit final and single-round measurement widths must match"
        )


def _bell_patches(distance: int) -> tuple["codes.CSSCode", "codes.CSSCode"]:
    import deltakit_circuit as circuit_api
    import deltakit_explorer.codes as codes

    # a rotated planar code of distance d spans x = 0 to 2d, so this shift
    # leaves one empty column between the two blocks
    offset = 2 * distance
    offset += 2
    shift = circuit_api.Coord2DDelta(offset, 0)
    left = codes.RotatedPlanarCode(distance, distance)
    right = codes.RotatedPlanarCode(distance, distance, shift=shift)
    return left, right


def _bell_experiment(
    left: "codes.CSSCode",
    right: "codes.CSSCode",
    basis: str,
    round_count: int,
) -> "circuit_api.Circuit":
    import deltakit_explorer.codes as codes

    initial = _bell_preparation(left, right)
    protected = _bell_protection(left, right, round_count)
    final = _bell_readout(left, right, basis)
    stages = [initial, protected, final]
    return codes.experiment_circuit(stages)


def _bell_preparation(
    left: "codes.CSSCode", right: "codes.CSSCode"
) -> "codes.CSSStage":
    import deltakit_circuit.gates as gates
    import deltakit_explorer.codes as codes

    resets = [gates.RX(qubit) for qubit in left.data_qubits]
    target_resets = [gates.RZ(qubit) for qubit in right.data_qubits]
    resets.extend(target_resets)
    return codes.CSSStage(final_round_resets=resets)


def _bell_protection(
    left: "codes.CSSCode", right: "codes.CSSCode", round_count: int
) -> "codes.CSSStage":
    import deltakit_circuit.gates as gates
    import deltakit_explorer.codes as codes

    left_data = sorted(left.data_qubits, key=_qubit_coordinates)
    right_data = sorted(right.data_qubits, key=_qubit_coordinates)
    pairs = zip(left_data, right_data)
    coupling = [gates.CX(control, target) for control, target in pairs]
    checks = left.stabilisers[0] + right.stabilisers[0]
    return codes.CSSStage(
        stabilisers=[checks],
        num_rounds=round_count,
        first_round_gates=coupling,
    )


def _qubit_coordinates(qubit: "circuit_api.Qubit") -> tuple:
    return tuple(qubit.unique_identifier)


def _bell_readout(
    left: "codes.CSSCode", right: "codes.CSSCode", basis: str
) -> "codes.CSSStage":
    import deltakit_circuit.gates as gates
    import deltakit_explorer.codes as codes

    support = left.x_logical_operators[0] | right.x_logical_operators[0]
    measure = gates.MX
    if basis == "Z":
        support = left.z_logical_operators[0] | right.z_logical_operators[0]
        measure = gates.MZ
    qubits = left.data_qubits | right.data_qubits
    readout = [measure(qubit) for qubit in qubits]
    observable = [pauli.qubit for pauli in support]
    return codes.CSSStage(
        first_round_measurements=readout,
        observable_definitions={0: observable},
    )


def _bell_wait_fragments(
    fragments: circuit_records.RepeatedStimCircuit,
    template: "circuit_api.Circuit",
    single: "circuit_api.Circuit",
    device: "qpu.QPU",
    mapping: dict["circuit_api.Qubit", int],
) -> circuit_records.RepeatedStimCircuit:
    import deltakit_circuit as circuit_api
    import deltakit_circuit.gates as gates

    identities = [gates.I(qubit) for qubit in device.qubits]
    layer = circuit_api.GateLayer(identities)
    wait = circuit_api.Circuit(layer)
    counts = _template_measurement_counts(fragments)
    rounds = _native_memory_rounds(template, counts)
    durations = _padded_native_rounds(rounds, single, wait)
    period_microseconds = fragments.round_period_microseconds
    _check_fragment_durations(durations, device, period_microseconds)
    noisy_wait = _noisy_wait(wait, device, mapping)
    repeated = noisy_wait + fragments.repeated_round
    final = noisy_wait + fragments.final_round
    return circuit_records.RepeatedStimCircuit(
        first_round=fragments.first_round,
        repeated_round=repeated,
        final_round=final,
        single_round=fragments.single_round,
        round_period_microseconds=period_microseconds,
    )


def _padded_native_rounds(
    rounds: list["circuit_api.Circuit"],
    single: "circuit_api.Circuit",
    wait: "circuit_api.Circuit",
) -> list["circuit_api.Circuit"]:
    import deltakit_circuit as circuit_api

    padded = [rounds[0], single]
    for native in rounds[1:]:
        fragment = circuit_api.Circuit(wait.layers)
        fragment.append_layers(native)
        padded.append(fragment)
    return padded


def _noisy_wait(
    wait: "circuit_api.Circuit",
    device: "qpu.QPU",
    mapping: dict["circuit_api.Qubit", int],
) -> stim.Circuit:
    """I marks a timed wait, charged once as idle rather than as a gate."""
    import deltakit_circuit as circuit_api
    import deltakit_stim

    seconds = device.get_circuit_execution_time(wait)
    idle_noise = device.noise_model.idle_noise
    channels = [idle_noise(qubit, seconds) for qubit in device.qubits]
    noise = circuit_api.NoiseLayer(channels)
    exported = deltakit_stim.Circuit()
    wait_layer = wait.layers[0]
    # The initial fragment already declares the common qubit coordinates.
    wait_layer.permute_stim_circuit(exported, mapping)
    noise.permute_stim_circuit(exported, mapping)
    text = str(exported)
    return stim.Circuit(text)


def _bell_initial_detectors(
    fragments: circuit_records.RepeatedStimCircuit,
    left: "codes.CSSCode",
    right: "codes.CSSCode",
    mapping: dict["circuit_api.Qubit", int],
) -> circuit_records.RepeatedStimCircuit:
    """Bell preparation fixes matching X and Z check products across patches."""
    left_checks = sorted(left.stabilisers[0], key=_ancilla_coordinates)
    right_checks = sorted(right.stabilisers[0], key=_ancilla_coordinates)
    pairs = zip(left_checks, right_checks)
    ancillas = [
        (first.ancilla_qubit, second.ancilla_qubit) for first, second in pairs
    ]
    first = _append_joint_checks(fragments.first_round, ancillas, mapping)
    single = _append_joint_checks(fragments.single_round, ancillas, mapping)
    return dataclasses.replace(
        fragments, first_round=first, single_round=single
    )


def _ancilla_coordinates(stabiliser) -> tuple:
    return tuple(stabiliser.ancilla_qubit.unique_identifier)


def _append_joint_checks(
    circuit: stim.Circuit,
    ancillas: list[tuple["circuit_api.Qubit", "circuit_api.Qubit"]],
    mapping: dict["circuit_api.Qubit", int],
) -> stim.Circuit:
    records = _measured_qubit_records(circuit)
    result = circuit.copy()
    for left, right in ancillas:
        left_index = mapping[left]
        right_index = mapping[right]
        left_offset = records[left_index] - circuit.num_measurements
        right_offset = records[right_index] - circuit.num_measurements
        left_record = stim.target_rec(left_offset)
        right_record = stim.target_rec(right_offset)
        result.append("DETECTOR", [left_record, right_record])
    return result


def _measured_qubit_records(circuit: stim.Circuit) -> dict[int, int]:
    records = {}
    measurement_count = 0
    flattened = circuit.flattened()
    for instruction in flattened:
        if instruction.name not in ("M", "MX"):
            continue
        targets = instruction.targets_copy()
        for target in targets:
            records[target.value] = measurement_count
            measurement_count += 1
    return records
