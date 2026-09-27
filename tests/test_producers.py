"""The makers decsim ships, against Stim's generator and the example tools.

memory_circuit is stim.Circuit.generated with one probability on all
four of its noise channels (Stim src/stim/gen/circuit_gen_params.cc);
memory_patches places copies of it with SHIFT_COORDS (Stim
doc/file_format_stim_circuit.md, SHIFT_COORDS). The Deltakit makers run
from a yaml what tools/deltakit_example.py and tools/live_memory_example.py
build by hand in Python, and the runs are compared field by field.
"""

import dataclasses

import pytest
import stim

import decsim.experiments.experiment as experiment
import decsim.frontends.deltakit as deltakit
import decsim.machine as machine_module
import decsim.producers as producers
import tests.experiments.yaml_configs as yaml_configs
import tools.deltakit_example as deltakit_example
import tools.live_memory_example as live_memory_example


def test_the_memory_circuit_is_stims_generated_circuit():
    workload = producers.memory_circuit(
        "surface_code:rotated_memory_z",
        "2d",
        distance=3,
        physical_error_probability=0.002,
    )
    generated = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=6,
        distance=3,
        after_clifford_depolarization=0.002,
        before_round_data_depolarization=0.002,
        before_measure_flip_probability=0.002,
        after_reset_flip_probability=0.002,
    )
    operation = workload.operations[0]

    assert operation.circuit == generated
    assert workload.round_counts == {1: 6}


def test_memory_patches_sit_side_by_side_two_d_plus_two_apart():
    """Patch p is the memory circuit with every x shifted by p (2 d + 2).

    Stim's SHIFT_COORDS moves the qubit and the detector coordinates that
    follow it; the detectors' last coordinate, their round, stays.
    """
    workload = producers.memory_patches(
        "surface_code:rotated_memory_z",
        4,
        2,
        distance=3,
        physical_error_probability=0.001,
    )
    first_circuit = workload.operations[0].circuit
    second_circuit = workload.operations[1].circuit
    first = first_circuit.get_final_qubit_coordinates()
    second = second_circuit.get_final_qubit_coordinates()
    first_detectors = first_circuit.get_detector_coordinates()
    second_detectors = second_circuit.get_detector_coordinates()

    assert workload.round_counts == {1: 4, 2: 4}
    assert first[10] == [3.0, 3.0]
    assert second[10] == [11.0, 3.0]
    assert first_detectors[30] == [4.0, 4.0, 4.0]
    assert second_detectors[30] == [12.0, 4.0, 4.0]


def test_memory_patches_runs_one_memory_per_patch_at_once(tmp_path):
    """Three patches, three operations, a shot drawn for each."""
    workload = {
        "kind": "producer",
        "function": "decsim.producers:memory_patches",
        "arguments": {
            "code_task": "surface_code:rotated_memory_z",
            "rounds_per_shot": 6,
            "patch_count": 3,
            "distance": "${qpu.distance}",
        },
    }
    config_path = yaml_configs.write_config(tmp_path, {"workload": workload})
    config = experiment.load_experiment(config_path)
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    shots = machine.observation.sampled_shots.shots_by_operation

    assert result.terminal_status == "complete"
    assert [operation.patches for operation in machine.operations] == [
        (0,),
        (1,),
        (2,),
    ]
    assert len(result.operation_results) == 3
    assert sorted(shots) == [1, 2, 3]


def test_no_patches_is_refused():
    """With no patch the run would finish having run nothing."""
    with pytest.raises(ValueError, match="patch_count is at least 1"):
        producers.memory_patches(
            "surface_code:rotated_memory_z",
            4,
            0,
            distance=3,
            physical_error_probability=0.001,
        )


def _yaml_run(config_path, seed):
    config = experiment.load_experiment(config_path)
    values = {"qpu.distance": 3, "qpu.round_period_microseconds": 1.1}
    point = config.point_task(values)
    settings = point.settings
    machine = machine_module.Machine.build(settings, seed)
    result = machine.run()
    return machine, result


def _commands(machine) -> list:
    events = machine.observation.command_events.events
    commands = []
    for event in events:
        operation_id = event.command.operation.id
        commands.append((event.kind, event.tick, operation_id))
    return commands


def _producer(function: str, arguments: dict) -> dict:
    return {"kind": "producer", "function": function, "arguments": arguments}


def test_the_yaml_deltakit_memory_runs_what_the_example_tool_runs(tmp_path):
    pytest.importorskip("deltakit_explorer")
    arguments = {
        "rounds": 24,
        "distance": "${qpu.distance}",
        "physical_error_probability": 0.001,
    }
    workload = _producer("decsim.producers:deltakit_memory", arguments)
    config_path = yaml_configs.example_tool_config(
        tmp_path, "stim_device", workload
    )
    machine, result = _yaml_run(config_path, 0)
    circuit, rounds = deltakit.memory_circuit(
        "rotated_surface", 3, 24, "Z", 0.001
    )
    tool_workload = deltakit_example.memory_workload(
        circuit, rounds, 24, "memory-patch"
    )
    tool_settings = deltakit_example.supplied_settings(
        tool_workload,
        distance=3,
        round_count=24,
        period_microseconds=1.1,
        feedback_microseconds=4.0,
    )
    tool_machine = machine_module.Machine.build(tool_settings, 0)
    tool_result = tool_machine.run()

    assert dataclasses.asdict(result) == dataclasses.asdict(tool_result)
    assert _commands(machine) == _commands(tool_machine)


def test_the_yaml_live_deltakit_memory_runs_what_the_live_tool_runs(tmp_path):
    """The owner, region and rounds the tool writes by hand are derived."""
    pytest.importorskip("deltakit_explorer")
    arguments = {
        "decode_after_rounds": 3,
        "distance": "${qpu.distance}",
        "physical_error_probability": 0.001,
        "round_period_microseconds": "${qpu.round_period_microseconds}",
    }
    workload = _producer("decsim.producers:deltakit_live_memory", arguments)
    config_path = yaml_configs.example_tool_config(
        tmp_path, "streaming_stim", workload
    )
    machine, result = _yaml_run(config_path, 17)
    program = deltakit.memory_rounds(
        "rotated_surface", 3, "Z", 0.001, round_period_microseconds=1.1
    )
    stream_id = producers.LIVE_STREAM_ID
    tool_settings = live_memory_example.live_settings(
        program,
        distance=3,
        round_period_microseconds=1.1,
        prefix_round_count=3,
        patch="memory-patch",
        feedback_microseconds=4.0,
        decoder_microseconds=0.1,
    )
    tool_machine = machine_module.Machine.build(tool_settings, 17)
    tool_result = tool_machine.run()
    measurements = machine.syndrome_source.sampled_measurements(stream_id)
    tool_source = tool_machine.syndrome_source
    tool_measurements = tool_source.sampled_measurements(stream_id)

    assert dataclasses.asdict(result) == dataclasses.asdict(tool_result)
    assert _commands(machine) == _commands(tool_machine)
    assert measurements == tool_measurements
