"""Public CircuitBuilder output agrees with Stim and whole-shot PyMatching.

The logical supports come independently from Explorer RotatedPlanarCode;
raw parities use Stim's converter without reference-sample subtraction.
"""

import importlib.util
import subprocess
import sys
import types

import numpy
import pymatching
import pytest
import stim

import decsim.detector_error_model.detector_formation as formation
import decsim.frontends.deltakit_compiler as compiler
import decsim.machine as machine_module
import tools.deltakit_example as example

COMPILER_SPECIFICATION = importlib.util.find_spec("deltakit_compile")
HAS_COMPILER = COMPILER_SPECIFICATION is not None


@pytest.fixture
def compiler_dependencies() -> types.ModuleType:
    pytest.importorskip("deltakit_compile")
    return pytest.importorskip("deltakit_explorer")


def test_importing_the_provider_keeps_compiler_dependencies_optional() -> None:
    script = (
        "import sys\n"
        "import decsim.frontends.deltakit_compiler\n"
        "loaded = [name for name in sys.modules "
        "if name.startswith('deltakit_')]\n"
        "assert loaded == []\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.skipif(HAS_COMPILER, reason="requires a dependency-absent job")
def test_selecting_an_absent_compiler_names_its_extra() -> None:
    with pytest.raises(
        ValueError, match=r"optional decsim\[deltakit-compile\] extra"
    ):
        compiler.compile_experiment("memory", 3, 2, "Z", 0)


@pytest.mark.usefixtures("compiler_dependencies")
@pytest.mark.parametrize("experiment", ["memory", "hadamard"])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("distance,round_count", [(2, 1), (3, 2), (5, 4)])
@pytest.mark.parametrize("has_logical_fault", [False, True])
def test_declared_output_has_absolute_logical_parity(
    experiment: str,
    basis: str,
    distance: int,
    round_count: int,
    has_logical_fault: bool,
) -> None:
    recursion_limit = sys.getrecursionlimit()
    circuit, _ = compiler.compile_experiment(
        experiment, distance, round_count, basis, 0
    )
    faulty = _with_initial_logical_fault(
        circuit, distance, basis, has_logical_fault
    )
    sampler = faulty.compile_sampler(seed=173)
    measurements = sampler.sample(shots=64)
    converter = faulty.compile_m2d_converter(skip_reference_sample=True)
    detectors, observables = converter.convert(
        measurements=measurements, separate_observables=True
    )
    assert not detectors.any()
    expected = numpy.full((64, 1), has_logical_fault)
    numpy.testing.assert_array_equal(observables, expected)
    assert sys.getrecursionlimit() == recursion_limit


@pytest.mark.usefixtures("compiler_dependencies")
@pytest.mark.parametrize("experiment", ["memory", "hadamard"])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("distance,round_count", [(3, 1), (3, 3), (5, 5)])
def test_property_explicit_rounds_preserve_every_noisy_record(
    experiment: str, basis: str, distance: int, round_count: int
) -> None:
    circuit, mapping = compiler.compile_experiment(
        experiment, distance, round_count, basis, 0.003
    )
    table = formation.build_formation_table(
        circuit, round_count, measurement_rounds=mapping
    )
    sampler = circuit.compile_sampler(seed=178)
    measurements = sampler.sample(shots=32)
    converter = circuit.compile_m2d_converter()
    detectors, observables = converter.convert(
        measurements=measurements, separate_observables=True
    )
    after_last_round = round_count + 1
    expected_rounds = range(1, after_last_round)
    actual_rounds = mapping.values()
    assert set(actual_rounds) == set(expected_rounds)
    for index, measurement in enumerate(measurements):
        packets = formation.split_measurements_into_packets(table, measurement)
        formed_detectors, formed_observables = formation.form_shot(
            table, packets
        )
        numpy.testing.assert_array_equal(formed_detectors, detectors[index])
        numpy.testing.assert_array_equal(formed_observables, observables[index])


@pytest.mark.usefixtures("compiler_dependencies")
@pytest.mark.parametrize("experiment", ["memory", "hadamard"])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("distance,round_count", [(3, 2), (5, 4)])
def test_machine_decodes_the_compiled_noisy_experiment(
    experiment: str, basis: str, distance: int, round_count: int
) -> None:
    circuit, mapping = compiler.compile_experiment(
        experiment, distance, round_count, basis, 0.003
    )
    model = circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=False
    )
    workload = example.memory_workload(
        circuit, mapping, round_count, "test-patch"
    )
    settings = example.supplied_settings(
        workload,
        distance=distance,
        round_count=round_count,
        period_microseconds=1.25,
        feedback_microseconds=0.02,
        decoder_microseconds=0.1,
    )
    machine = machine_module.Machine.build(settings, seed=73)
    result = machine.run()
    source = machine.syndrome_source
    detection = source.sampled_detection_events(1)
    matcher = pymatching.Matching.from_detector_error_model(model)
    prediction = matcher.decode(detection)
    output = result.operation_results[0]
    assert result.terminal_status == "complete"
    assert output.logical_observables == tuple(prediction)


@pytest.mark.usefixtures("compiler_dependencies")
@pytest.mark.parametrize(
    "experiment,distance,round_count,basis,probability,message",
    [
        ("rotation", 3, 2, "Z", 0, "experiment must be memory or hadamard"),
        ("memory", 3, 2, "Y", 0, "compiler basis must be X or Z"),
        ("memory", 1, 2, "Z", 0, "compiler distance must be at least two"),
        ("memory", True, 2, "Z", 0, "distance must be a positive integer"),
        ("memory", 3, 0, "Z", 0, "round_count must be a positive integer"),
        (
            "memory",
            3,
            2,
            "Z",
            -0.1,
            r"physical_error_probability must lie in \[0, 1\]",
        ),
        (
            "memory",
            3,
            2,
            "Z",
            float("nan"),
            r"physical_error_probability must lie in \[0, 1\]",
        ),
    ],
)
def test_invalid_experiments_explain_the_boundary(
    experiment: str,
    distance: int,
    round_count: int,
    basis: str,
    probability: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compiler.compile_experiment(
            experiment, distance, round_count, basis, probability
        )


@pytest.mark.usefixtures("compiler_dependencies")
@pytest.mark.parametrize("basis,readout", [("X", "M"), ("Z", "MX")])
@pytest.mark.parametrize("distance", [3, 5])
def test_hadamard_measures_every_data_qubit_in_the_conjugate_basis(
    basis: str, readout: str, distance: int
) -> None:
    import deltakit_explorer.codes as codes

    circuit, _ = compiler.compile_experiment("hadamard", distance, 2, basis, 0)
    code = codes.RotatedPlanarCode(width=distance, height=distance)
    coordinates = circuit.get_final_qubit_coordinates()
    actual_qubits = {
        tuple(point): qubit for qubit, point in coordinates.items()
    }
    expected_data = {
        actual_qubits[tuple(qubit.unique_identifier)]
        for qubit in code.data_qubits
    }
    measured = _measured_bases(circuit)
    terminal_bases = {qubit: measured[qubit] for qubit in expected_data}
    expected_bases = dict.fromkeys(expected_data, readout)
    assert terminal_bases == expected_bases


def _with_initial_logical_fault(
    circuit: stim.Circuit, distance: int, basis: str, has_fault: bool
) -> stim.Circuit:
    if not has_fault:
        return circuit
    support = _logical_fault_support(circuit, distance, basis)
    error_gate = {"X": "Z_ERROR", "Z": "X_ERROR"}[basis]
    faulty = stim.Circuit()
    has_injected = False
    for instruction in circuit:
        is_preparation = instruction.name in (
            "QUBIT_COORDS",
            "RX",
            "R",
            "RZ",
            "TICK",
        )
        if not has_injected and not is_preparation:
            faulty.append(error_gate, support, 1)
            has_injected = True
        faulty.append(instruction)
    return faulty


def _logical_fault_support(
    circuit: stim.Circuit, distance: int, basis: str
) -> list[int]:
    import deltakit_explorer.codes as codes

    code = codes.RotatedPlanarCode(width=distance, height=distance)
    operators = {"X": code.z_logical_operators, "Z": code.x_logical_operators}
    coordinates = circuit.get_final_qubit_coordinates()
    qubits = {tuple(point): index for index, point in coordinates.items()}
    return [
        qubits[tuple(pauli.qubit.unique_identifier)]
        for pauli in operators[basis][0]
    ]


def _measured_bases(circuit: stim.Circuit) -> dict[int, str]:
    measured = {}
    flattened = circuit.flattened()
    for instruction in flattened:
        if instruction.name not in ("M", "MX"):
            continue
        targets = instruction.targets_copy()
        for target in targets:
            measured[target.value] = instruction.name
    return measured
