"""The makers decsim ships, against Stim's own generator.

memory_circuit is stim.Circuit.generated with one probability on all
four of its noise channels (Stim src/stim/gen/circuit_gen_params.cc);
memory_patches places copies of it with SHIFT_COORDS (Stim
doc/file_format_stim_circuit.md, SHIFT_COORDS).
"""

import pytest
import stim

import decsim.experiments.experiment as experiment
import decsim.machine as machine_module
import decsim.producers as producers
import tests.experiments.yaml_configs as yaml_configs


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
        },
    }
    config_path = yaml_configs.write_config(tmp_path, {"workload": workload})
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.001,
        distance=3,
        round_period_microseconds=1.0,
    )
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
