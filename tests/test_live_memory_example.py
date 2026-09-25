"""Live example replay obeys Stim conversion and the QPU feedback clock.

Canonical inputs use tests.qpu.memory_programs and ordinary Stim; optional
producer tests additionally execute Explorer SD6 and duration-aware noise.
"""

import dataclasses
import json
import pathlib
import subprocess
import sys

import numpy
import pytest
import stim

import decsim.config as config
import decsim.experiments.experiment as experiment
import decsim.frontends.circuit_frontend as circuit_frontend
import decsim.machine as machine_module
import decsim.qpu.streaming_stim_device as streaming_stim_device
import decsim.records.circuits as circuit_records
import tests.experiments.yaml_configs as yaml_configs
import tests.qpu.memory_programs as memory_programs
import tools.live_memory_example as example

# The tool's protection workload as a decsim.ops/1 file: decode after
# three rounds, wait for the answer, resume one round, read out. The
# stream's owner, its protected region and its rounds are derived.
LIVE_OPERATIONS = {
    "schema": "decsim.ops/1",
    "operations": [
        {
            "id": 1,
            "name": "prefix",
            "patches": ["memory-patch"],
            "stream_id": 100,
            "stream_offset": 0,
            "rounds": 3,
        },
        {
            "id": 2,
            "name": "protect",
            "patches": ["memory-patch"],
            "emits_detector_data": False,
            "rounds": 0,
        },
        {
            "id": 3,
            "name": "resume",
            "patches": ["memory-patch"],
            "blocked_by": 1,
            "emits_detector_data": False,
            "rounds": 1,
        },
        {
            "id": 4,
            "name": "readout",
            "patches": ["memory-patch"],
            "emits_detector_data": False,
            "rounds": 0,
        },
    ],
}


def test_saved_fragments_reproduce_results_trace_and_actual_history(
    tmp_path: pathlib.Path,
) -> None:
    original = tmp_path / "original"
    replay = tmp_path / "replay"
    inputs = _canonical_inputs(tmp_path)
    _run_example(inputs, original)
    _run_example(original, replay)
    original_result = _read_json(original, "result.json")
    replay_result = _read_json(replay, "result.json")
    assert original_result == replay_result
    original_trace = _read_json(original, "trace.json")
    replay_trace = _read_json(replay, "trace.json")
    assert original_trace == replay_trace
    original_measurements = _read_json(original, "measurements.json")
    replay_measurements = _read_json(replay, "measurements.json")
    assert original_measurements == replay_measurements
    original_circuit = _executed_circuit(original)
    replay_circuit = _executed_circuit(replay)
    assert original_circuit == replay_circuit


def test_later_feedback_extends_the_live_history_and_actual_readout(
    tmp_path: pathlib.Path,
) -> None:
    inputs = _canonical_inputs(tmp_path)
    faster = tmp_path / "faster"
    slower = tmp_path / "slower"
    _run_example(inputs, faster, feedback_microseconds=4.0)
    _run_example(inputs, slower, feedback_microseconds=8.0)
    first_mapping = _read_json(faster, "measurement_rounds.json")
    second_mapping = _read_json(slower, "measurement_rounds.json")
    first_rounds = first_mapping.values()
    second_rounds = second_mapping.values()
    extra_rounds = max(second_rounds) - max(first_rounds)
    assert extra_rounds > 0
    first_result = _read_json(faster, "result.json")
    second_result = _read_json(slower, "result.json")
    readout_delay = (
        second_result["execution_done_ticks"]
        - first_result["execution_done_ticks"]
    )
    period_ticks = config.microseconds_to_ticks(1.1)
    expected_delay = extra_rounds * period_ticks
    assert readout_delay == expected_delay
    first_bits = _read_json(faster, "measurements.json")
    second_bits = _read_json(slower, "measurements.json")
    final_round = max(first_rounds)
    shared_indices = [
        int(index)
        for index, round_index in first_mapping.items()
        if round_index < final_round
    ]
    shared_count = len(shared_indices)
    assert first_bits[:shared_count] == second_bits[:shared_count]


def test_saved_actual_measurements_reproduce_the_final_logical_truth(
    tmp_path: pathlib.Path,
) -> None:
    inputs = _canonical_inputs(tmp_path)
    output = tmp_path / "executed"
    _run_example(inputs, output)
    circuit = _executed_circuit(output)
    recorded = _read_json(output, "measurements.json")
    measurements = numpy.array([recorded], dtype=numpy.bool_)
    converter = circuit.compile_m2d_converter()
    _, observables = converter.convert(
        measurements=measurements, separate_observables=True
    )
    result = _read_json(output, "result.json")
    owner = result["operation_results"][-1]
    expected = list(observables[0])
    assert owner["observable_truth"] == expected
    assert result["terminal_status"] == "complete"
    assert result["decode_work_settled"]
    assert result["event_queue_empty"]


@pytest.mark.parametrize(
    "flag,value", [("--round-period-microseconds", "1.2"), ("--distance", "5")]
)
def test_replay_refuses_changes_to_physical_inputs(
    tmp_path: pathlib.Path, flag: str, value: str
) -> None:
    inputs = _canonical_inputs(tmp_path)
    output = tmp_path / "refused"
    command = [
        sys.executable,
        "tools/live_memory_example.py",
        "--input",
        str(inputs),
        "--output",
        str(output),
        flag,
        value,
    ]
    refused = subprocess.run(
        command, check=False, capture_output=True, text=True
    )
    assert refused.returncode != 0
    assert "differs from the exported physical parameters" in refused.stderr


def test_replay_does_not_select_the_optional_producer(
    tmp_path: pathlib.Path,
) -> None:
    inputs = _canonical_inputs(tmp_path)
    output = tmp_path / "ordinary"
    script = (
        "import sys\n"
        "import tools.live_memory_example as example\n"
        "example.main()\n"
        "loaded = [name for name in sys.modules "
        "if name.startswith('deltakit_')]\n"
        "assert loaded == []\n"
    )
    command = [
        sys.executable,
        "-c",
        script,
        "--input",
        str(inputs),
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True)


@pytest.mark.parametrize("noise_model", ["sd6", "physical"])
def test_optional_producer_saves_reusable_physical_inputs(
    tmp_path: pathlib.Path, noise_model: str
) -> None:
    pytest.importorskip("deltakit_explorer")
    output = tmp_path / noise_model
    command = _producer_command(output, noise_model)
    subprocess.run(command, check=True, capture_output=True)
    parameters = _read_json(output, "physical_parameters.json")
    assert parameters["round_period_microseconds"] == 1.25
    result = _read_json(output, "result.json")
    assert result["terminal_status"] == "complete"
    mapping = _read_json(output, "measurement_rounds.json")
    rounds = mapping.values()
    actual_duration = max(rounds) * 1_250_000
    assert result["execution_done_ticks"] == actual_duration


def test_prefix_requires_a_physical_measurement_round() -> None:
    with pytest.raises(ValueError, match="prefix_round_count must be positive"):
        example.protection_workload(0, "patch")


def test_public_settings_keep_the_user_patch_in_a_complete_live_run() -> None:
    program = memory_programs.memory_program()
    source = streaming_stim_device.StreamingStimDevice(
        programs={example.STREAM_OWNER_ID: program}
    )
    settings = example.live_settings(
        source,
        distance=3,
        round_period_microseconds=1.1,
        prefix_round_count=3,
        patch="user-patch",
        feedback_microseconds=4.0,
        decoder_microseconds=0.1,
    )
    owner = settings.workload.dynamic_streams[0]
    assert owner.patches == ("user-patch",)
    machine = machine_module.Machine.build(settings, seed=81)
    result = machine.run()
    assert result.terminal_status == "complete"
    assert source.logical_observable_truth(example.STREAM_OWNER_ID) is not None


@pytest.mark.parametrize("feedback_microseconds", [4.0, 8.0, 20.0])
def test_the_files_row_runs_what_the_tool_builds_by_hand(
    tmp_path: pathlib.Path, feedback_microseconds: float
) -> None:
    """The canonical fragments from a yaml, against the tool's own run."""
    program = memory_programs.memory_program()
    program = dataclasses.replace(program, round_period_microseconds=1.1)
    folder = _live_files(tmp_path, program)
    workload = {
        "kind": "files",
        "operations": "live/operations.json",
        "fragments": "live/fragments",
    }
    config_path = yaml_configs.example_tool_config(
        folder, "streaming_stim", workload, feedback_microseconds
    )
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.003,
        distance=3,
        round_period_microseconds=1.1,
    )
    machine = machine_module.Machine.build(settings, 17)
    result = machine.run()
    source = streaming_stim_device.StreamingStimDevice(
        programs={example.STREAM_OWNER_ID: program}
    )
    tool_settings = example.live_settings(
        source,
        distance=3,
        round_period_microseconds=1.1,
        prefix_round_count=3,
        patch="memory-patch",
        feedback_microseconds=feedback_microseconds,
        decoder_microseconds=0.1,
    )
    tool_machine = machine_module.Machine.build(tool_settings, 17)
    tool_result = tool_machine.run()
    stream_id = example.STREAM_OWNER_ID
    yaml_source = machine.syndrome_source
    measurements = yaml_source.sampled_measurements(stream_id)
    tool_measurements = source.sampled_measurements(stream_id)
    executed = yaml_source.executed_circuit(stream_id)
    tool_executed = source.executed_circuit(stream_id)

    assert dataclasses.asdict(result) == dataclasses.asdict(tool_result)
    assert measurements == tool_measurements
    assert executed == tool_executed


def _live_files(folder: pathlib.Path, program) -> pathlib.Path:
    """The live operations and the fragments, as the files row reads them."""
    live = folder / "live"
    live.mkdir()
    operations_text = json.dumps(LIVE_OPERATIONS)
    operations_path = live / "operations.json"
    operations_path.write_text(operations_text)
    fragments = live / "fragments"
    fragments.mkdir()
    _write_fragments(fragments, program)
    physical = {"round_period_microseconds": 1.1}
    physical_path = fragments / circuit_frontend.PHYSICAL_FILE_NAME
    physical_text = json.dumps(physical)
    physical_path.write_text(physical_text)
    return folder


def _canonical_inputs(folder: pathlib.Path) -> pathlib.Path:
    inputs = folder / "canonical"
    inputs.mkdir()
    program = memory_programs.memory_program()
    program = dataclasses.replace(program, round_period_microseconds=1.1)
    _write_fragments(inputs, program)
    parameters = {
        "distance": 3,
        "basis": "Z",
        "physical_error_probability": 0.003,
        "round_period_microseconds": 1.1,
        "noise_model": "stim_fixture",
        "relaxation_time_microseconds": None,
        "dephasing_time_microseconds": None,
    }
    parameter_text = json.dumps(parameters)
    path = inputs / "physical_parameters.json"
    path.write_text(parameter_text)
    return inputs


def _write_fragments(
    folder: pathlib.Path, program: circuit_records.RepeatedStimCircuit
) -> None:
    for name in (
        "first_round",
        "repeated_round",
        "final_round",
        "single_round",
    ):
        fragment = getattr(program, name)
        path = folder / f"{name}.stim"
        fragment.to_file(str(path))


def _run_example(
    inputs: pathlib.Path,
    output: pathlib.Path,
    feedback_microseconds: float = 4.0,
) -> None:
    command = [
        sys.executable,
        "tools/live_memory_example.py",
        "--input",
        str(inputs),
        "--output",
        str(output),
        "--feedback-microseconds",
        str(feedback_microseconds),
    ]
    subprocess.run(command, check=True, capture_output=True)


def _read_json(folder: pathlib.Path, filename: str) -> object:
    path = folder / filename
    text = path.read_text()
    return json.loads(text)


def _executed_circuit(folder: pathlib.Path) -> stim.Circuit:
    path = folder / "executed.stim"
    return stim.Circuit.from_file(str(path))


def _producer_command(output: pathlib.Path, noise_model: str) -> list[str]:
    command = [
        sys.executable,
        "tools/live_memory_example.py",
        "--output",
        str(output),
        "--round-period-microseconds",
        "1.25",
        "--noise-model",
        noise_model,
    ]
    if noise_model == "physical":
        command.extend(
            [
                "--relaxation-time-microseconds",
                "50",
                "--dephasing-time-microseconds",
                "40",
            ]
        )
    return command
