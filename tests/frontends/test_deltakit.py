"""Explorer CSS stages agree with Stim's raw-record conversion oracle.

Sources: pinned Explorer css_code_memory_circuit and CSSStage.first_round;
Stim compile_m2d_converter; detector_formation's explicit packet contract.
"""

import importlib.util
import math
import subprocess
import sys
import types

import numpy
import pytest
import stim

import decsim.detector_error_model.detector_formation as formation
import decsim.frontends.deltakit as deltakit

EXPLORER_SPECIFICATION = importlib.util.find_spec("deltakit_explorer")
HAS_EXPLORER = EXPLORER_SPECIFICATION is not None
# Resets and single-qubit noise act on each target alone.
ORDER_FREE_INSTRUCTIONS = frozenset(
    {"R", "RX", "RY", "X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1"}
)


@pytest.fixture
def explorer() -> types.ModuleType:
    return pytest.importorskip("deltakit_explorer")


def test_importing_the_provider_does_not_import_deltakit() -> None:
    script = (
        "import sys\n"
        "import decsim.frontends.deltakit\n"
        "loaded = [name for name in sys.modules "
        "if name.startswith('deltakit_')]\n"
        "assert loaded == []\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.skipif(
    HAS_EXPLORER,
    reason="the dependency-absent job checks the real missing import",
)
def test_selecting_an_absent_provider_still_stops() -> None:
    with pytest.raises(ImportError):
        deltakit.memory_circuit("rotated_surface", 3, 3, "Z", 0.001)


@pytest.mark.usefixtures("explorer")
@pytest.mark.parametrize("family", ["rotated_surface", "repetition"])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("distance,round_count", [(3, 1), (5, 2)])
def test_property_shared_raw_shots_preserve_every_detector_and_observable(
    family: str, basis: str, distance: int, round_count: int
) -> None:
    circuit, mapping = deltakit.memory_circuit(
        family, distance, round_count, basis, 0.01
    )
    table = formation.build_formation_table(
        circuit, round_count, measurement_rounds=mapping
    )
    sampler = circuit.compile_sampler(seed=163)
    measurements = sampler.sample(shots=32)
    converter = circuit.compile_m2d_converter()
    detectors, observables = converter.convert(
        measurements=measurements, separate_observables=True
    )
    assert circuit.num_observables == 1
    measurement_indices = range(circuit.num_measurements)
    assert set(mapping) == set(measurement_indices)
    declared_rounds = mapping.values()
    after_last_round = round_count + 1
    expected_rounds = range(1, after_last_round)
    assert set(declared_rounds) == set(expected_rounds)
    for index, measurement in enumerate(measurements):
        packets = formation.split_measurements_into_packets(table, measurement)
        formed_detectors, formed_observables = formation.form_shot(
            table, packets
        )
        numpy.testing.assert_array_equal(formed_detectors, detectors[index])
        numpy.testing.assert_array_equal(formed_observables, observables[index])


@pytest.mark.usefixtures("explorer")
@pytest.mark.parametrize("family", ["rotated_surface", "repetition"])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("round_count", [1, 5])
def test_the_finite_memory_has_the_error_model_of_its_assembled_rounds(
    family: str, basis: str, round_count: int
) -> None:
    fragments = deltakit.memory_rounds(
        family, 3, basis, 0.003, round_period_microseconds=1.1
    )
    assembled, assembled_rounds = fragments.assemble(round_count)
    finite, finite_rounds = deltakit.memory_circuit(
        family, 3, round_count, basis, 0.003
    )
    assembled_model = assembled.detector_error_model(flatten_loops=True)
    finite_model = finite.detector_error_model(flatten_loops=True)
    assert finite_model == assembled_model
    assert finite_rounds == assembled_rounds


@pytest.mark.usefixtures("explorer")
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_noiseless_memory_has_no_detection_events_or_observable_flips(
    basis: str,
) -> None:
    circuit, mapping = deltakit.memory_circuit(
        "rotated_surface", 3, 4, basis, 0
    )
    sampler = circuit.compile_detector_sampler(seed=7)
    detectors, observables = sampler.sample(32, separate_observables=True)
    assert not numpy.any(detectors)
    assert not numpy.any(observables)
    declared_rounds = mapping.values()
    assert max(declared_rounds) == 4


@pytest.mark.usefixtures("explorer")
def test_one_seed_draws_the_same_samples_in_every_process() -> None:
    script = (
        "import decsim.frontends.deltakit as deltakit\n"
        "circuit, _ = deltakit.memory_circuit("
        "'rotated_surface', 3, 3, 'Z', 0.05)\n"
        "sampler = circuit.compile_detector_sampler(seed=7)\n"
        "print(sampler.sample(200).tobytes().hex())\n"
    )
    samples = set()
    for _ in range(3):
        finished = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        samples.add(finished.stdout)
    assert len(samples) == 1


@pytest.mark.usefixtures("explorer")
@pytest.mark.parametrize(
    "family,distance,sentence",
    [
        ("unknown", 3, "code_family must be rotated_surface or repetition"),
        ("repetition", 0, "Code distance must be at least 2."),
    ],
)
def test_invalid_selected_memory_is_refused(
    family: str, distance: int, sentence: str
) -> None:
    """An unknown family is decsim's refusal; a bad distance the Explorer's."""
    with pytest.raises(ValueError, match=sentence):
        deltakit.memory_circuit(family, distance, 3, "Z", 0.01)


@pytest.mark.usefixtures("explorer")
def test_shifted_measurement_map_cannot_make_detectors_arrive_early() -> None:
    circuit, mapping = deltakit.memory_circuit("rotated_surface", 3, 4, "Z", 0)
    table = formation.build_formation_table(
        circuit, 4, measurement_rounds=mapping
    )
    shifted = {index: round_index + 1 for index, round_index in mapping.items()}
    detector_rounds = table.detector_rounds()
    with pytest.raises(ValueError, match="reads a bit that arrives in round 2"):
        formation.build_formation_table(
            circuit,
            5,
            measurement_rounds=shifted,
            detector_rounds=detector_rounds,
        )


@pytest.mark.usefixtures("explorer")
def test_a_round_with_no_detection_events_is_not_a_missing_round() -> None:
    circuit, mapping = deltakit.memory_circuit("rotated_surface", 3, 4, "Z", 0)
    table = formation.build_formation_table(
        circuit, 4, measurement_rounds=mapping
    )
    sampler = circuit.compile_sampler(seed=12)
    measurements = sampler.sample(1)
    packets = formation.split_measurements_into_packets(table, measurements[0])
    detectors, observables = formation.form_shot(table, packets)
    assert not any(detectors)
    assert observables == (0,)
    del packets[2]
    with pytest.raises(KeyError, match="2"):
        formation.form_shot(table, packets)


@pytest.mark.usefixtures("explorer")
@pytest.mark.parametrize("family", ["rotated_surface", "repetition"])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("round_count", [1, 2, 7])
@pytest.mark.parametrize("noise_model", ["sd6", "physical"])
def test_repeated_memory_matches_the_finite_native_schedule(
    family: str, basis: str, round_count: int, noise_model: str
) -> None:
    periods = {"rotated_surface": 0.6, "repetition": 0.4}
    period = periods[family]
    noise_arguments = _noise_arguments(noise_model)
    fragments = deltakit.memory_rounds(
        family,
        3,
        basis,
        0.003,
        round_period_microseconds=period,
        **noise_arguments,
    )
    assembled, measurement_rounds = fragments.assemble(round_count)
    reference, duration_seconds = _finite_native_reference(
        family, basis, round_count, noise_model
    )
    actual = _physical_operations(assembled)
    expected = _physical_operations(reference)
    assert actual == expected
    assert fragments.round_period_microseconds == period
    expected_microseconds = round_count * period
    expected_seconds = expected_microseconds * 1e-6
    assert duration_seconds == pytest.approx(expected_seconds)
    assert len(measurement_rounds) == assembled.num_measurements
    assembled.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=False
    )


@pytest.mark.usefixtures("explorer")
def test_physical_idle_noise_uses_the_declared_cadence() -> None:
    period_microseconds = 0.6
    fragments = deltakit.memory_rounds(
        "rotated_surface",
        3,
        "Z",
        0.003,
        round_period_microseconds=period_microseconds,
        noise_model="physical",
        relaxation_time_microseconds=20,
        dephasing_time_microseconds=30,
    )
    channel = _first_instruction(fragments.repeated_round, "PAULI_CHANNEL_1")
    actual = channel.gate_args_copy()
    layer_count = _layer_count(fragments.repeated_round)
    gate_microseconds = period_microseconds / layer_count
    expected = _idle_probabilities(gate_microseconds, 20, 30)
    numpy.testing.assert_allclose(actual, expected, rtol=1e-5)
    gate_channel = _first_instruction(fragments.repeated_round, "DEPOLARIZE2")
    gate_probabilities = gate_channel.gate_args_copy()
    assert gate_probabilities == [0.003]


@pytest.mark.usefixtures("explorer")
@pytest.mark.parametrize(
    "arguments",
    [
        {"noise_model": "unknown"},
        {"relaxation_time_microseconds": 20},
        {
            "noise_model": "physical",
            "relaxation_time_microseconds": 20,
            "dephasing_time_microseconds": 40,
        },
        {
            "noise_model": "physical",
            "relaxation_time_microseconds": float("inf"),
            "dephasing_time_microseconds": 30,
        },
    ],
    ids=[
        "unknown model",
        "sd6 given times",
        "explorer's t2 bound",
        "infinite t1",
    ],
)
def test_invalid_repeated_memory_noise_is_refused(arguments: dict) -> None:
    """The Explorer bounds T2 by T1; the other three are decsim's refusals.

    An infinite T1 builds the right limit but would be saved as
    Infinity, which a strict JSON reader refuses.
    """
    selected = {
        "code_family": "rotated_surface",
        "distance": 3,
        "basis": "Z",
        "physical_error_probability": 0.001,
        "round_period_microseconds": 1,
    }
    selected.update(arguments)
    with pytest.raises(ValueError):
        deltakit.memory_rounds(**selected)


@pytest.mark.usefixtures("explorer")
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("round_count", [1, 7])
@pytest.mark.parametrize("noise_model", ["sd6", "physical"])
def test_supplied_css_memory_preserves_the_full_bb_native_circuit(
    basis: str, round_count: int, noise_model: str
) -> None:
    import deltakit_explorer.codes as codes

    code = codes.BivariateBicycleCode(3, 5, [1, 1, 4], [0, 1, 2])
    noise_arguments = _noise_arguments(noise_model)
    fragments = deltakit.css_memory_rounds(
        code,
        basis,
        0.003,
        round_period_microseconds=0.9,
        **noise_arguments,
    )
    assembled, mapping = fragments.assemble(round_count)
    reference, duration_seconds = _finite_native_reference(
        "bivariate_bicycle", basis, round_count, noise_model
    )
    actual = _physical_operations(assembled)
    expected = _physical_operations(reference)
    assert actual == expected
    assert assembled.num_observables == 8
    expected_seconds = round_count * 0.9e-6
    assert duration_seconds == pytest.approx(expected_seconds)
    assert len(mapping) == assembled.num_measurements
    assembled.detector_error_model(approximate_disjoint_errors=False)


@pytest.mark.usefixtures("explorer")
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_supplied_css_preserves_a_fault_beyond_the_first_observable(
    basis: str,
) -> None:
    import deltakit_explorer.codes as codes

    code = codes.BivariateBicycleCode(3, 5, [1, 1, 4], [0, 1, 2])
    fragments = deltakit.css_memory_rounds(
        code, basis, 0, round_period_microseconds=0.9
    )
    circuit, mapping = fragments.assemble(4)
    faulty, expected = _with_bb_logical_fault(circuit, code, basis)
    sampler = faulty.compile_sampler(seed=619)
    measurements = sampler.sample(32)
    converter = circuit.compile_m2d_converter(skip_reference_sample=True)
    detectors, observables = converter.convert(
        measurements=measurements, separate_observables=True
    )
    assert not numpy.any(detectors)
    matches = observables == expected
    assert numpy.all(matches)
    assert expected[1] == 1
    table = formation.build_formation_table(
        circuit, 4, measurement_rounds=mapping
    )
    packets = formation.split_measurements_into_packets(table, measurements[0])
    formed_detectors, formed_observables = formation.form_shot(table, packets)
    numpy.testing.assert_array_equal(formed_detectors, detectors[0])
    numpy.testing.assert_array_equal(formed_observables, expected)


@pytest.mark.usefixtures("explorer")
def test_supplied_css_refuses_extra_ancilla_preparation_time() -> None:
    code = _repetition_with_explicit_ancilla_preparation()
    with pytest.raises(ValueError, match="CSS memory fragments must fit"):
        deltakit.css_memory_rounds(code, "Z", 0, round_period_microseconds=1)


def _noise_arguments(noise_model: str) -> dict:
    arguments = {"noise_model": noise_model}
    if noise_model == "physical":
        arguments["relaxation_time_microseconds"] = 20
        arguments["dephasing_time_microseconds"] = 30
    return arguments


def _finite_native_reference(
    family: str, basis: str, round_count: int, noise_model: str
) -> tuple[stim.Circuit, float]:
    import deltakit_circuit.gates as gates
    import deltakit_explorer.codes as codes
    import deltakit_explorer.qpu as qpu

    logical_basis = gates.PauliBasis[basis]
    code = _reference_code(family, logical_basis)
    native_gates = qpu.ExhaustiveGateSet()
    _native_reference_durations(native_gates)
    noise = _reference_noise(noise_model)
    device = qpu.QPU(
        code.qubits,
        native_gates_and_times=native_gates,
        noise_model=noise,
        maximise_parallelism=False,
    )
    original = codes.css_code_memory_circuit(code, round_count, logical_basis)
    compiled = device.compile_circuit(original)
    seconds = device.get_circuit_execution_time(compiled)
    noisy = device.compile_and_add_noise_to_circuit(original)
    qubits = sorted(code.qubits, key=_reference_qubit_identifier)
    mapping = {qubit: index for index, qubit in enumerate(qubits)}
    exported = noisy.as_stim_circuit(qubit_mapping=mapping)
    text = str(exported)
    return stim.Circuit(text), seconds


def _reference_code(family: str, basis):
    import deltakit_explorer.codes as codes

    if family == "repetition":
        return codes.RepetitionCode(distance=3, stabiliser_type=basis)
    if family == "bivariate_bicycle":
        return codes.BivariateBicycleCode(3, 5, [1, 1, 4], [0, 1, 2])
    return codes.RotatedPlanarCode(width=3, height=3)


def _with_bb_logical_fault(
    circuit: stim.Circuit, code, basis: str
) -> tuple[stim.Circuit, numpy.ndarray]:
    supports = code.x_logical_operators
    conjugates = code.z_logical_operators
    gate = "Z_ERROR"
    if basis == "Z":
        supports = code.z_logical_operators
        conjugates = code.x_logical_operators
        gate = "X_ERROR"
    fault_support = _logical_qubits(conjugates[1])
    expected = _logical_fault_signature(supports, fault_support)
    qubits = sorted(code.qubits, key=_reference_qubit_identifier)
    mapping = {qubit: index for index, qubit in enumerate(qubits)}
    targets = [mapping[qubit] for qubit in fault_support]
    faulty = _insert_before_measurement(circuit, gate, targets)
    return faulty, expected


def _logical_qubits(logical) -> set:
    return {pauli.qubit for pauli in logical}


def _logical_fault_signature(supports, fault_support: set) -> numpy.ndarray:
    expected = []
    for support in supports:
        qubits = _logical_qubits(support)
        overlap = qubits & fault_support
        parity = len(overlap) % 2
        expected.append(parity)
    return numpy.asarray(expected, dtype=bool)


def _insert_before_measurement(
    circuit: stim.Circuit, gate: str, targets: list[int]
) -> stim.Circuit:
    result = stim.Circuit()
    has_inserted_fault = False
    flattened = circuit.flattened()
    for instruction in flattened:
        if not has_inserted_fault and instruction.name in ("M", "MX"):
            result.append(gate, targets, 1)
            has_inserted_fault = True
        result.append(instruction)
    return result


def _native_reference_durations(native_gates) -> None:
    families = (
        native_gates.one_qubit_gates,
        native_gates.two_qubit_gates,
        native_gates.reset_gates,
        native_gates.measurement_gates,
    )
    for family in families:
        for gate in family:
            family[gate] = 100e-9


def _reference_noise(noise_model: str):
    import deltakit_explorer.qpu as qpu

    if noise_model == "sd6":
        return qpu.SD6Noise(p=0.003)
    return qpu.PhysicalNoise(
        t1=20e-6,
        t2=30e-6,
        p_1_qubit_gate_error=0.003,
        p_2_qubit_gate_error=0.003,
        p_reset_error=0.003,
        p_meas_qubit_error=0.003,
        p_readout_flip=0.003,
    )


def _reference_qubit_identifier(qubit) -> str:
    return repr(qubit.unique_identifier)


def _physical_operations(circuit: stim.Circuit) -> stim.Circuit:
    """Ignore timing annotations and orders that carry no meaning.

    XOR targets and order-free targets are sorted; all noise is kept.
    """
    result = stim.Circuit()
    flattened = circuit.flattened()
    for instruction in flattened:
        if instruction.name == "TICK":
            continue
        targets = instruction.targets_copy()
        if instruction.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            targets.sort(key=_target_value)
        if instruction.name in ORDER_FREE_INSTRUCTIONS:
            targets.sort(key=_target_value)
        arguments = instruction.gate_args_copy()
        result.append(instruction.name, targets, arguments)
    return result


def _target_value(target: stim.GateTarget) -> int:
    return target.value


def _first_instruction(
    circuit: stim.Circuit, name: str
) -> stim.CircuitInstruction:
    for instruction in circuit:
        if instruction.name == name:
            return instruction
    raise AssertionError(f"Expected a {name} noise channel")


def _layer_count(fragment) -> int:
    """The gate layers of an exported round: one TICK closes each."""
    ticks = [
        instruction for instruction in fragment if instruction.name == "TICK"
    ]
    return len(ticks)


def _idle_probabilities(
    gate_microseconds: float,
    relaxation_microseconds: float,
    dephasing_microseconds: float,
) -> list[float]:
    """Explorer _noise_parameters.py: Ghosh et al. 1210.5799, equation 10."""
    relaxation_ratio = -gate_microseconds / relaxation_microseconds
    dephasing_ratio = -gate_microseconds / dephasing_microseconds
    relaxation_survival = math.exp(relaxation_ratio)
    dephasing_survival = math.exp(dephasing_ratio)
    relaxation_loss = 1 - relaxation_survival
    dephasing_loss = 1 - dephasing_survival
    bit_probability = 0.25 * relaxation_loss
    dephasing_probability = 0.5 * dephasing_loss
    phase_probability = dephasing_probability - bit_probability
    return [bit_probability, bit_probability, phase_probability]


def _repetition_with_explicit_ancilla_preparation():
    import deltakit_circuit.gates as gates
    import deltakit_explorer.codes as codes

    class _PreparedAncillas(codes.RepetitionCode):
        """Initialize ancillary qubits before the ordinary extraction reset."""

        def encode_logical_zeroes(self) -> codes.CSSStage:
            resets = [gates.RZ(qubit) for qubit in self.qubits]
            return codes.CSSStage(final_round_resets=resets)

    return _PreparedAncillas(3, stabiliser_type=gates.PauliBasis.Z)
