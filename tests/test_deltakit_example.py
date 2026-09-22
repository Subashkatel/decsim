"""Supplied Explorer circuits obey the existing machine and cycle contracts.

Referents: qpu/cycle_clock.py, qpu/stim_device.py, detector_formation.py,
and tools/deltakit_example.py. Whole-shot PyMatching is an integration oracle,
not an independent decoding algorithm. All tests execute functional decoding.
"""

import dataclasses
import json
import pathlib
import subprocess
import sys
import types

import numpy
import pymatching
import pytest
import stim

import decsim.config as config
import decsim.detector_error_model.detector_formation as formation
import decsim.frontends.deltakit as deltakit
import decsim.machine as machines
import decsim.qpu.stim_device as sources
import decsim.syndrome_buffer.settings as buffer_settings
import tools.deltakit_example as example

pytestmark = pytest.mark.usefixtures("explorer")


@pytest.fixture
def explorer() -> types.ModuleType:
    return pytest.importorskip("deltakit_explorer")


@pytest.mark.parametrize("distance,round_count", [(3, 2), (5, 4)])
def test_exported_text_replaces_the_provider_with_identical_results_and_trace(
    distance: int, round_count: int
) -> None:
    circuit, mapping = deltakit.memory_circuit(
        "rotated_surface", distance, round_count, "Z", 0.001
    )
    first = _memory_machine(circuit, mapping, distance, round_count)
    circuit_text = str(circuit)
    exported = stim.Circuit(circuit_text)
    second = _memory_machine(exported, mapping, distance, round_count)
    first_result = first.run()
    second_result = second.run()
    assert first_result.terminal_status == "complete"
    first_fields = dataclasses.asdict(first_result)
    second_fields = dataclasses.asdict(second_result)
    assert first_fields == second_fields
    first_trace = first.observation.trace_writer.document()
    second_trace = second.observation.trace_writer.document()
    assert first_trace == second_trace
    assert first_result.operation_results[0].observable_truth is not None


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_repetition_uses_its_own_geometry_and_a_whole_shot_oracle(
    basis: str,
) -> None:
    circuit, mapping = deltakit.memory_circuit("repetition", 5, 3, basis, 0.01)
    sampler = circuit.compile_sampler(seed=73)
    measurements = sampler.sample(1)
    converter = circuit.compile_m2d_converter()
    detectors, observables = converter.convert(
        measurements=measurements, separate_observables=True
    )
    model = circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=False
    )
    matcher = pymatching.Matching.from_detector_error_model(model)
    prediction = matcher.decode(detectors[0])
    machine = _repetition_replay(circuit, mapping, measurements, 5, 3)
    result = machine.run()
    output = result.operation_results[0]
    assert result.terminal_status == "complete"
    assert output.logical_observables == tuple(prediction)
    assert output.observable_truth == tuple(observables[0])


def test_an_undetectable_logical_fault_is_still_a_logical_failure() -> None:
    circuit, mapping = deltakit.memory_circuit("repetition", 3, 4, "Z", 0.001)
    noiseless = circuit.without_noise()
    faulty = _logical_fault_after_first_round(noiseless, 3)
    sampler = faulty.compile_sampler(seed=10)
    measurements = sampler.sample(1)
    converter = circuit.compile_m2d_converter()
    detectors, observables = converter.convert(
        measurements=measurements, separate_observables=True
    )
    assert not numpy.any(detectors)
    assert observables.tolist() == [[True]]
    machine = _repetition_replay(circuit, mapping, measurements, 3, 4)
    result = machine.run()
    assert result.operation_results[0].logical_observables == (0,)
    assert result.operation_results[0].observable_truth == (1,)
    assert result.operation_results[0].logical_failure is True


@pytest.mark.parametrize("period_microseconds", [1.1, 1.25])
@pytest.mark.parametrize("instruction_microseconds", [0.0, 0.15])
def test_feedback_restart_obeys_the_declared_clock(
    period_microseconds: float, instruction_microseconds: float
) -> None:
    machine = _protected_machine(
        period_microseconds, 4.0, instruction_microseconds
    )
    readouts = []
    machine.qpu.trace.round_emitted.connect(readouts.append)
    result = machine.run()
    events = machine.observation.command_events.events
    arrival = _command_tick(events, "ARRIVED", 3)
    started = _command_tick(events, "STARTED", 3)
    period_ticks = config.microseconds_to_ticks(period_microseconds)
    clock = config.Clock(period_ticks)
    assert started == clock.edge(0, arrival)
    cycle_offset_ticks = arrival % period_ticks
    instruction_ticks = config.microseconds_to_ticks(instruction_microseconds)
    assert cycle_offset_ticks == instruction_ticks
    assert [packet.round_index for packet in readouts] == list(range(1, 25))
    assert sum(packet.size_bits for packet in readouts) == 201
    assert readouts[-1].size_bits == 17
    horizon_ticks = 24 * period_ticks
    assert result.execution_done_ticks == horizon_ticks
    assert result.terminal_status == "complete"
    assert result.decode_work_settled
    assert result.event_queue_empty


def test_longer_feedback_protects_more_rounds_before_continuation() -> None:
    faster = _protected_machine(1.1, 4.0, 0.15)
    slower = _protected_machine(1.1, 8.0, 0.15)
    first_result = faster.run()
    second_result = slower.run()
    first_start = _command_tick(
        faster.observation.command_events.events, "STARTED", 3
    )
    second_start = _command_tick(
        slower.observation.command_events.events, "STARTED", 3
    )
    period_ticks = config.microseconds_to_ticks(1.1)
    additional_wait_ticks = second_start - first_start
    four_round_ticks = 4 * period_ticks
    assert additional_wait_ticks == four_round_ticks
    assert (
        first_result.execution_done_ticks == second_result.execution_done_ticks
    )
    assert (
        first_result.operation_results[-1].observable_truth
        == second_result.operation_results[-1].observable_truth
    )
    readout_edge = first_result.link_traffic["semantic_edges"][0]
    assert readout_edge["counters"]["transfer_count"] == 24


def test_a_feedback_wait_past_the_declared_horizon_exhausts_the_source() -> (
    None
):
    machine = _protected_machine(1.1, 100.0, 0.15)
    with pytest.raises(ValueError, match="outside the finite source"):
        machine.run()


def test_protected_segments_emit_one_faulted_history_without_resampling() -> (
    None
):
    circuit, mapping = deltakit.memory_circuit(
        "rotated_surface", 3, 24, "Z", 0.001
    )
    noiseless = circuit.without_noise()
    faulty = _logical_fault_after_first_round(noiseless, 9)
    sampler = faulty.compile_sampler(seed=10)
    measurements = sampler.sample(1)
    workload = example.protection_workload(circuit, 24, 3, "patch")
    settings = _settings(circuit, mapping, workload, 3, 24)
    source = sources.RecordedStimDevice(
        measurements, 0, measurement_rounds={100: mapping}
    )
    qpu = dataclasses.replace(settings.qpu, device=source)
    settings = dataclasses.replace(settings, qpu=qpu)
    machine = machines.Machine.build(settings, 0)
    readouts = []
    machine.qpu.trace.round_emitted.connect(readouts.append)
    result = machine.run()
    emitted = tuple(bit for packet in readouts for bit in packet.bits)
    assert emitted == tuple(measurements[0])
    assert [packet.round_index for packet in readouts] == list(range(1, 25))
    assert {packet.operation_id for packet in readouts} == {100}
    assert result.operation_results[-1].observable_truth == (1,)
    assert result.operation_results[-1].logical_failure is True


def test_a_protected_stream_cannot_use_truth_from_a_later_final_readout() -> (
    None
):
    circuit, mapping = deltakit.memory_circuit("rotated_surface", 3, 24, "Z", 0)
    workload = example.protection_workload(circuit, 24, 3, "patch")
    workload.operations[-1].scheduled_start_round = 20
    settings = _settings(circuit, mapping, workload, 3, 24)
    machine = machines.Machine.build(settings, 0)
    with pytest.raises(RuntimeError, match="sealed at 20 rounds"):
        machine.run()


def test_a_circuit_without_logical_outputs_is_refused_before_decoding() -> None:
    circuit, mapping = deltakit.memory_circuit(
        "rotated_surface", 3, 2, "Z", 0.001
    )
    instructions = circuit.flattened()
    kept = [
        instruction
        for instruction in instructions
        if instruction.name != "OBSERVABLE_INCLUDE"
    ]
    text = "\n".join(str(instruction) for instruction in kept)
    without_output = stim.Circuit(text)
    with pytest.raises(ValueError, match="requires one logical observable"):
        _memory_machine(without_output, mapping, 3, 2)


def test_extending_a_finite_input_cannot_add_empty_rounds_after_readout() -> (
    None
):
    circuit, mapping = deltakit.memory_circuit(
        "rotated_surface", 3, 2, "Z", 0.001
    )
    with pytest.raises(
        ValueError, match="horizon must equal the final readout round"
    ):
        _memory_machine(circuit, mapping, 3, 24)


def test_command_replay_preserves_repetition_geometry_and_horizon(
    tmp_path: pathlib.Path,
) -> None:
    original = tmp_path / "original"
    replay = tmp_path / "replay"
    command = [sys.executable, "tools/deltakit_example.py"]
    generate = command + [
        "--family",
        "repetition",
        "--rounds",
        "4",
        "--output",
        str(original),
    ]
    subprocess.run(generate, check=True, capture_output=True)
    repeat = command + ["--input", str(original), "--output", str(replay)]
    subprocess.run(repeat, check=True, capture_output=True)
    original_result_path = original / "result.json"
    replay_result_path = replay / "result.json"
    original_text = original_result_path.read_text()
    replay_text = replay_result_path.read_text()
    original_result = json.loads(original_text)
    replay_result = json.loads(replay_text)
    assert original_result == replay_result
    altered = repeat + ["--rounds", "24"]
    refused = subprocess.run(
        altered, check=False, capture_output=True, text=True
    )
    assert refused.returncode != 0
    assert (
        "rounds differs from the exported circuit parameters" in refused.stderr
    )


def test_the_whole_shot_repetition_example_refuses_a_prefix_feedback_wait(
    tmp_path: pathlib.Path,
) -> None:
    command = [
        sys.executable,
        "tools/deltakit_example.py",
        "--family",
        "repetition",
        "--mode",
        "protection",
        "--output",
        str(tmp_path),
    ]
    refused = subprocess.run(
        command, check=False, capture_output=True, text=True
    )
    assert refused.returncode != 0
    assert (
        "protection example requires rotated_surface memory" in refused.stderr
    )


def test_bounded_buffer_and_unit_memory_use_the_normal_data_path() -> None:
    circuit, mapping = deltakit.memory_circuit(
        "rotated_surface", 3, 24, "Z", 0.001
    )
    workload = example.protection_workload(circuit, 24, 3, "patch")
    settings = _settings(circuit, mapping, workload, 3, 24)
    buffer = buffer_settings.SyndromeBufferSettings(rounds=12)
    decoder = dataclasses.replace(
        settings.weak_decoder, unit_memory_rounds=12, units=2
    )
    settings = dataclasses.replace(
        settings, weak_syndrome_buffer=buffer, weak_decoder=decoder
    )
    machine = machines.Machine.build(settings, 0)
    result = machine.run()
    assert result.terminal_status == "complete"
    assert result.data_movement["rounds"] == 24
    assert result.data_movement["moved_rounds"] > 24


def _settings(
    circuit,
    mapping,
    workload,
    distance,
    round_count,
    period_microseconds=1.1,
    feedback_microseconds=4.0,
):
    return example.supplied_settings(
        circuit,
        mapping,
        workload,
        distance=distance,
        round_count=round_count,
        period_microseconds=period_microseconds,
        feedback_microseconds=feedback_microseconds,
    )


def _memory_machine(circuit, mapping, distance, round_count):
    workload = example.memory_workload(circuit, round_count, "patch")
    settings = _settings(circuit, mapping, workload, distance, round_count)
    return machines.Machine.build(settings, 0)


def _repetition_replay(circuit, mapping, measurements, distance, round_count):
    workload = example.memory_workload(circuit, round_count, "patch")
    settings = _settings(circuit, mapping, workload, distance, round_count)
    table = formation.build_formation_table(
        circuit, round_count, measurement_rounds=mapping
    )
    detector_rounds = table.detector_rounds()
    source = sources.RecordedStimDevice(
        measurements,
        0,
        measurement_rounds={1: mapping},
        detector_rounds={1: detector_rounds},
    )
    code = example.RepetitionMemory(distance, round_count)
    qpu = dataclasses.replace(
        settings.qpu, distance=None, code=code, device=source
    )
    settings = dataclasses.replace(settings, qpu=qpu)
    return machines.Machine.build(settings, 0)


def _protected_machine(
    period_microseconds, feedback_microseconds, instruction_microseconds
):
    circuit, mapping = deltakit.memory_circuit(
        "rotated_surface", 3, 24, "Z", 0.001
    )
    workload = example.protection_workload(circuit, 24, 3, "patch")
    settings = _settings(
        circuit,
        mapping,
        workload,
        3,
        24,
        period_microseconds,
        feedback_microseconds,
    )
    instruction_ticks = config.microseconds_to_ticks(instruction_microseconds)
    channel = dataclasses.replace(
        settings.links.controller_to_qpu.channel,
        propagation_latency_ticks=instruction_ticks,
    )
    path = dataclasses.replace(
        settings.links.controller_to_qpu, channel=channel
    )
    links = dataclasses.replace(settings.links, controller_to_qpu=path)
    settings = dataclasses.replace(settings, links=links)
    return machines.Machine.build(settings, 0)


def _command_tick(events, kind, operation_id):
    selected = [
        event.tick
        for event in events
        if (event.kind, event.command.operation.id) == (kind, operation_id)
    ]
    assert len(selected) == 1
    return selected[0]


def _logical_fault_after_first_round(circuit, data_qubit_count):
    data_qubits = _final_data_qubits(circuit, data_qubit_count)
    flattened = circuit.flattened()
    faulty = stim.Circuit()
    reset_count = 0
    for instruction in flattened:
        if instruction.name == "R":
            reset_count += 1
        if instruction.name == "R" and reset_count == 2:
            faulty.append("X_ERROR", data_qubits, 1.0)
        faulty.append(instruction)
    return faulty


def _final_data_qubits(circuit, data_qubit_count):
    # Explorer's last M includes the final data measurements after checks.
    flattened = circuit.flattened()
    measurements = [
        instruction for instruction in flattened if instruction.name == "M"
    ]
    last_targets = measurements[-1].targets_copy()
    data_targets = last_targets[-data_qubit_count:]
    return [target.value for target in data_targets]
