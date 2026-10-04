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
import decsim.frontends.settings as workload_settings
import decsim.frontends.workload_files as workload_files
import decsim.machine as machine_module
import decsim.producers as producers
import decsim.records.circuits as circuit_records
import examples.live_memory_example as example
import tests.qpu.memory_programs as memory_programs

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
    saved = _saved_fragments(original)
    _run_example(saved, replay)
    original_result = _read_json(original, "result.json")
    replay_result = _read_json(replay, "result.json")
    assert original_result == replay_result
    original_trace = _read_json(original, "trace/seed17.trace.json")
    replay_trace = _read_json(replay, "trace/seed17.trace.json")
    assert original_trace == replay_trace
    original_measurements = _read_json(original, "measurements.json")
    replay_measurements = _read_json(replay, "measurements.json")
    assert original_measurements == replay_measurements
    original_circuit = _executed_circuit(original)
    replay_circuit = _executed_circuit(replay)
    assert original_circuit == replay_circuit


def test_a_rerun_from_a_run_folders_fragments_runs_its_recorded_point(
    tmp_path: pathlib.Path,
) -> None:
    """The point's distance comes back from its record."""
    program = memory_programs.memory_program(distance=5)
    program = dataclasses.replace(program, round_period_microseconds=1.1)
    inputs = tmp_path / "distance_5" / "fragments"
    _write_fragments(inputs, program)
    original = tmp_path / "original"
    replay = tmp_path / "replay"
    physical = ("--distance", "5")
    _run_example(inputs, original, physical=physical)
    saved = _saved_fragments(original)
    _run_example(saved, replay)
    original_result = _read_json(original, "result.json")
    replay_result = _read_json(replay, "result.json")
    assert replay_result == original_result
    original_points = _resolved_names(original)
    replay_points = _resolved_names(replay)
    assert replay_points == original_points


def _measurements_before(mapping: dict, round_index: int) -> list:
    """The measurement indices the mapping places before round_index."""
    indices = []
    for index, measured_round in mapping.items():
        if measured_round < round_index:
            indices.append(int(index))
    return indices


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
    shared_indices = _measurements_before(first_mapping, final_round)
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


def test_replay_refuses_a_round_period_the_fragments_do_not_declare(
    tmp_path: pathlib.Path,
) -> None:
    inputs = _canonical_inputs(tmp_path)
    output = tmp_path / "refused"
    command = [
        sys.executable,
        "examples/live_memory_example.py",
        "--input",
        str(inputs),
        "--output",
        str(output),
        "--round-period-microseconds",
        "1.2",
    ]
    refused = subprocess.run(
        command, check=False, capture_output=True, text=True
    )
    assert refused.returncode != 0
    assert (
        "physical circuit period differs from the QPU cadence" in refused.stderr
    )


def test_a_replay_refuses_a_flag_that_would_relabel_its_circuit(
    tmp_path: pathlib.Path,
) -> None:
    """The loaded fragments' basis is their circuit's, not the flag's."""
    inputs = _canonical_inputs(tmp_path)
    output = tmp_path / "relabelled"
    command = [
        sys.executable,
        "examples/live_memory_example.py",
        "--input",
        str(inputs),
        "--output",
        str(output),
        "--basis",
        "X",
    ]
    refused = subprocess.run(
        command, check=False, capture_output=True, text=True
    )
    assert refused.returncode != 0
    assert "--basis cannot change the fragments --input loads" in refused.stderr
    assert not output.exists()


def test_a_replay_names_the_circuit_its_saved_fragments_hold(
    tmp_path: pathlib.Path,
) -> None:
    """An X-basis memory with physical noise is replayed under its name.

    The replay's arguments come from the record the producing run left
    beside the fragments, and its executed history is the original's.
    """
    pytest.importorskip("deltakit_explorer")
    original = tmp_path / "original"
    command = _producer_command(original, "physical")
    command.extend(["--basis", "X"])
    subprocess.run(command, check=True, capture_output=True)
    saved = _saved_fragments(original)
    replay = tmp_path / "replay"
    _run_example(saved, replay)
    made = _read_json(original, "arguments.json")
    replayed = _read_json(replay, "arguments.json")
    replayed_circuit = {name: replayed[name] for name in example.CIRCUIT_VALUES}
    made_circuit = {name: made[name] for name in example.CIRCUIT_VALUES}
    assert replayed_circuit == made_circuit
    assert replayed["basis"] == "X"
    assert _executed_circuit(replay) == _executed_circuit(original)


def test_fragments_that_bind_no_period_replay_at_the_chosen_one(
    tmp_path: pathlib.Path,
) -> None:
    """A null period leaves the cadence to the run, as the files row does."""
    program = memory_programs.memory_program()
    inputs = tmp_path / "unbound" / "fragments"
    _write_fragments(inputs, program)
    output = tmp_path / "replay"
    physical = ("--round-period-microseconds", "1.1")
    _run_example(inputs, output, physical=physical)
    result = _read_json(output, "result.json")
    assert result["event_queue_empty"]


def test_replay_does_not_select_the_optional_producer(
    tmp_path: pathlib.Path,
) -> None:
    inputs = _canonical_inputs(tmp_path)
    output = tmp_path / "ordinary"
    script = (
        "import sys\n"
        "import examples.live_memory_example as example\n"
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
    saved = _saved_fragments(output)
    parameters = _read_json(saved, workload_files.PHYSICAL_FILE_NAME)
    assert parameters["round_period_microseconds"] == 1.25
    result = _read_json(output, "result.json")
    assert result["terminal_status"] == "complete"
    mapping = _read_json(output, "measurement_rounds.json")
    rounds = mapping.values()
    actual_duration = max(rounds) * 1_250_000
    assert result["execution_done_ticks"] == actual_duration


def test_prefix_requires_a_physical_measurement_round() -> None:
    """The QPU refuses a decode before the stream's first round."""
    program = memory_programs.memory_program()
    settings = example.live_settings(
        program,
        distance=3,
        round_period_microseconds=1.1,
        prefix_round_count=0,
        patch="patch",
        feedback_microseconds=4.0,
        decoder_microseconds=0.1,
    )

    machine = machine_module.Machine.build(settings, seed=81)

    with pytest.raises(ValueError, match="zero-duration detector emitters"):
        machine.run()


def test_public_settings_keep_the_user_patch_in_a_complete_live_run() -> None:
    program = memory_programs.memory_program()
    settings = example.live_settings(
        program,
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
    source = machine.qpu.syndrome_source
    assert result.terminal_status == "complete"
    assert source.logical_observable_truth(producers.LIVE_STREAM_ID) is not None


def test_the_files_reader_runs_what_the_tool_builds_by_hand(
    tmp_path: pathlib.Path,
) -> None:
    """The canonical fragments read from files, against the tool's own run."""
    program = memory_programs.memory_program()
    program = dataclasses.replace(program, round_period_microseconds=1.1)
    folder = _live_files(tmp_path, program)
    live = folder / "live"
    operations_path = live / "operations.json"
    fragments_path = live / "fragments"
    workload = workload_files.read_workload(
        operations_path, fragments_path=fragments_path
    )
    tool_settings = example.live_settings(
        program,
        distance=3,
        round_period_microseconds=1.1,
        prefix_round_count=3,
        patch="memory-patch",
        feedback_microseconds=4.0,
        decoder_microseconds=0.1,
    )
    read_workload = workload_settings.WorkloadSettings.running(workload)
    settings = dataclasses.replace(tool_settings, workload=read_workload)
    machine = machine_module.Machine.build(settings, 17)
    result = machine.run()
    tool_machine = machine_module.Machine.build(tool_settings, 17)
    tool_result = tool_machine.run()
    stream_id = producers.LIVE_STREAM_ID
    files_source = machine.qpu.syndrome_source
    source = tool_machine.qpu.syndrome_source
    measurements = files_source.sampled_measurements(stream_id)
    tool_measurements = source.sampled_measurements(stream_id)
    executed = files_source.executed_circuit(stream_id)
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
    _write_fragments(fragments, program)
    return folder


def _canonical_inputs(folder: pathlib.Path) -> pathlib.Path:
    """The canonical fragments, as a files row fragments folder."""
    program = memory_programs.memory_program()
    program = dataclasses.replace(program, round_period_microseconds=1.1)
    fragments = folder / "canonical" / "fragments"
    _write_fragments(fragments, program)
    return fragments


def _write_fragments(
    fragments: pathlib.Path, program: circuit_records.RepeatedStimCircuit
) -> None:
    """The four fragments and physical.json, as the files row reads them."""
    fragments.mkdir(parents=True)
    for name in workload_files.FRAGMENT_NAMES:
        fragment = getattr(program, name)
        path = fragments / f"{name}.stim"
        fragment.to_file(str(path))
    physical = {"round_period_microseconds": program.round_period_microseconds}
    physical_path = fragments / workload_files.PHYSICAL_FILE_NAME
    physical_text = json.dumps(physical)
    physical_path.write_text(physical_text)


def _saved_fragments(run_folder: pathlib.Path) -> pathlib.Path:
    """The fragments a tool run saved with its one point's inputs."""
    saved = run_folder.glob("points/*/inputs/fragments")
    (fragments,) = saved
    return fragments


def _run_example(
    inputs: pathlib.Path,
    output: pathlib.Path,
    feedback_microseconds: float = 4.0,
    physical: tuple = (),
) -> None:
    command = [
        sys.executable,
        "examples/live_memory_example.py",
        "--input",
        str(inputs),
        "--output",
        str(output),
        "--feedback-microseconds",
        str(feedback_microseconds),
        *physical,
    ]
    subprocess.run(command, check=True, capture_output=True)


def _resolved_names(folder: pathlib.Path) -> list:
    """The point ids a results folder recorded, which name its every value."""
    paths = folder.glob("points/*/machine.json")
    point_ids = []
    for path in paths:
        record = _read_json(path.parent, path.name)
        point_ids.append(record["id"])
    return sorted(point_ids)


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
        "examples/live_memory_example.py",
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
