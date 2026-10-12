"""The makers decsim ships, against Stim's generator and the examples.

memory_circuit is stim.Circuit.generated with one probability on all
four of its noise channels (Stim src/stim/gen/circuit_gen_params.cc);
memory_patches places copies of it with SHIFT_COORDS (Stim
doc/file_format_stim_circuit.md, SHIFT_COORDS). The Deltakit makers
make the workloads examples/deltakit_example.py and
examples/live_memory_example.py build by hand, and the runs on the
tools' machines are compared field by field. The rows are lowered at a
qpu: memory is memory_circuit or memory_patches at the qpu's distance,
Stim's four channels at p unless noise sets them apart, and live_memory
is live_memory over the fragments a folder holds.
"""

import dataclasses
import pathlib

import pytest
import stim

import decsim.frontends.deltakit as deltakit
import decsim.frontends.settings as workload_settings
import decsim.frontends.workload_files as workload_files
import decsim.machine as machine_module
import decsim.presets.machines as machines
import decsim.producers as producers
import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.settings as qpu_settings
import examples.deltakit_example as deltakit_example
import examples.live_memory_example as live_memory_example
import tests.qpu.memory_programs as memory_programs

Z_MEMORY = "surface_code:rotated_memory_z"
X_MEMORY = "surface_code:rotated_memory_x"


def test_the_memory_circuit_is_stims_generated_circuit():
    workload = producers.memory_circuit(
        "surface_code:rotated_memory_z",
        6,
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
    assert workload.round_counts == ((1, 6),)


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

    assert workload.round_counts == ((1, 4), (2, 4))
    assert first[10] == [3.0, 3.0]
    assert second[10] == [11.0, 3.0]
    assert first_detectors[30] == [4.0, 4.0, 4.0]
    assert second_detectors[30] == [12.0, 4.0, 4.0]


def test_memory_patches_runs_one_memory_per_patch_at_once():
    """Three patches, three operations, a shot drawn for each."""
    patches = producers.memory_patches(
        "surface_code:rotated_memory_z",
        6,
        3,
        distance=3,
        physical_error_probability=0.001,
    )
    weak_base = machines.weak_baseline(3, 0.001, 1.0)
    workload = workload_settings.WorkloadSettings.running(patches)
    graph = weak_base.with_held("workload", workload)
    machine = machine_module.Machine.build(graph, 0)
    result = machine.run()
    shots = machine.observation.sampled_shots.shots_by_operation

    assert result.terminal_status == "complete"
    assert [operation.patches for operation in machine.plan.all_operations] == [
        (0,),
        (1,),
        (2,),
    ]
    assert len(result.operation_results) == 3
    assert sorted(shots) == [1, 2, 3]


def test_no_patches_is_refused():
    """With no patch the run would finish having run nothing."""
    sentence = (
        "workload.patch_count is at least 1, got 0; with no patch the run "
        "would finish having run nothing"
    )
    with pytest.raises(ValueError, match=sentence):
        producers.memory_patches(
            "surface_code:rotated_memory_z",
            4,
            0,
            distance=3,
            physical_error_probability=0.001,
        )


def test_a_memory_is_the_presets_workload_at_the_qpus_distance():
    memory = producers.MemorySettings(0.002, rounds=12)
    qpu = _qpu(5)

    lowered = memory.lowered_for(qpu)

    assert lowered == machines.memory_workload(5, 0.002, 12)


def test_rounds_per_distance_runs_that_many_rounds_a_unit_of_distance():
    memory = producers.MemorySettings(0.002, rounds_per_distance=10)
    qpu = _qpu(7)

    lowered = memory.lowered_for(qpu)

    assert lowered == machines.memory_workload(7, 0.002, 70)


def test_a_qpu_that_names_no_distance_takes_its_cards_own():
    memory = producers.MemorySettings(0.002, rounds_per_distance=2)
    qpu = _qpu(None)

    lowered = memory.lowered_for(qpu)

    assert lowered == machines.memory_workload(3, 0.002, 6)


def test_a_memory_states_its_rounds_once():
    with pytest.raises(ValueError, match="states its rounds once"):
        producers.MemorySettings(0.002, rounds=5, rounds_per_distance=2)
    with pytest.raises(ValueError, match="states its rounds once"):
        producers.MemorySettings(0.002)


def test_noise_at_p_on_every_channel_is_the_memory_at_p():
    noise = workload_settings.StimNoise(0.002, 0.002, 0.002, 0.002)
    noisy = producers.MemorySettings(0.002, rounds=6, noise=noise)
    plain = producers.MemorySettings(0.002, rounds=6)
    qpu = _qpu(3)

    noisy_lowered = noisy.lowered_for(qpu)
    plain_lowered = plain.lowered_for(qpu)

    assert noisy_lowered == plain_lowered


def test_noise_sets_stims_four_channels_apart_and_p_stays_the_label():
    noise = workload_settings.StimNoise(0.001, 0.002, 0.003, 0.004)
    memory = producers.MemorySettings(0.005, rounds=4, noise=noise)
    qpu = _qpu(3)

    lowered = memory.lowered_for(qpu)

    generated = stim.Circuit.generated(
        Z_MEMORY,
        rounds=4,
        distance=3,
        after_clifford_depolarization=0.001,
        before_round_data_depolarization=0.002,
        before_measure_flip_probability=0.003,
        after_reset_flip_probability=0.004,
    )
    record = lowered.workload_record
    (operation,) = record.operations
    assert operation.circuit == generated
    assert record.physical_error_probability == 0.005


@pytest.mark.parametrize(
    ("code_task", "reset", "flip", "readout"),
    [(Z_MEMORY, "R", "X_ERROR", "M"), (X_MEMORY, "RX", "Z_ERROR", "MX")],
)
def test_noiseless_time_boundaries_drop_the_two_data_flips(
    code_task, reset, flip, readout
):
    """Only the data's flips after their reset and before the readout go.

    Stim joins the two resets, or the two readouts, that a dropped flip
    separated, so the circuits are compared, not their text.
    """
    noisy = workload_settings.memory_circuit(code_task, 4, 3, 0.002)
    noiseless = workload_settings.memory_circuit(
        code_task, 4, 3, 0.002, noiseless_time_boundaries=True
    )

    noisy_text = str(noisy)
    noisy_lines = noisy_text.splitlines()
    readouts = [line for line in noisy_lines if line.startswith(f"{readout} ")]
    last_readout = readouts[-1]
    data = last_readout.removeprefix(f"{readout} ")
    data_flip = f"{flip}(0.002) {data}"
    data_reset = f"{reset} {data}"
    reset_index = noisy_lines.index(data_reset)
    last_readout_index = noisy_lines.index(last_readout)
    assert noisy_lines[reset_index + 1] == data_flip
    assert noisy_lines[last_readout_index - 1] == data_flip
    assert noisy_lines.count(data_flip) == 2
    kept = [line for line in noisy_lines if line != data_flip]
    kept_text = "\n".join(kept)
    assert noiseless == stim.Circuit(kept_text)


def test_a_memory_with_noiseless_time_boundaries_is_lowered_with_them():
    memory = producers.MemorySettings(
        0.002, rounds=4, noiseless_time_boundaries=True
    )
    qpu = _qpu(3)

    lowered = memory.lowered_for(qpu)

    expected = workload_settings.memory_circuit(
        Z_MEMORY, 4, 3, 0.002, noiseless_time_boundaries=True
    )
    (operation,) = lowered.workload_record.operations
    assert operation.circuit == expected


@pytest.mark.parametrize(
    ("text", "missing"),
    [
        ("R 0 1\nX_ERROR(0.1) 0 1\nH 0", "final data measurement"),
        ("H 0 1\nX_ERROR(0.1) 0 1\nM 0 1", "first reset"),
    ],
)
def test_noiseless_time_boundaries_refuse_a_circuit_without_them(text, missing):
    replaced = stim.Circuit(text)
    noise = _ReplacedCircuit(replaced)

    with pytest.raises(ValueError, match=missing):
        workload_settings.memory_circuit(
            Z_MEMORY, 3, 3, 0.001, noise, noiseless_time_boundaries=True
        )


def test_several_patches_are_the_memory_patches_maker():
    memory = producers.MemorySettings(0.002, rounds=4, patches=2)
    qpu = _qpu(3)

    lowered = memory.lowered_for(qpu)

    workload = producers.memory_patches(Z_MEMORY, 4, 2, 3, 0.002)
    assert lowered == workload_settings.WorkloadSettings.running(workload)


def test_an_x_memory_is_stims_x_task():
    memory = producers.MemorySettings(0.002, rounds=4, code_task=X_MEMORY)
    qpu = _qpu(3)

    lowered = memory.lowered_for(qpu)

    workload = producers.memory_circuit(X_MEMORY, 4, 3, 0.002)
    assert lowered == workload_settings.WorkloadSettings.running(workload)


def test_a_task_of_another_code_than_the_card_is_refused():
    memory = producers.MemorySettings(0.002, rounds=4)
    card = code_geometry.BivariateBicycleCodeModel.Settings()
    qpu = qpu_settings.QpuSettings(code_card=card)

    with pytest.raises(ValueError, match="another code than the qpu's card"):
        memory.lowered_for(qpu)


def test_a_memory_with_no_task_runs_patches_with_no_circuit():
    memory = producers.MemorySettings(
        0.002, rounds=4, code_task=None, patches=2
    )
    card = code_geometry.BivariateBicycleCodeModel.Settings()
    qpu = qpu_settings.QpuSettings(code_card=card)

    lowered = memory.lowered_for(qpu)

    workload = producers.memory_without_circuit(4, 2, 0.002)
    circuits = [operation.circuit for operation in workload.operations]
    assert lowered == workload_settings.WorkloadSettings.running(workload)
    assert circuits == [None, None]


def test_a_memory_not_decoded_decodes_no_operation():
    memory = producers.MemorySettings(0.002, rounds=4, decoded=False)
    qpu = _qpu(3)

    lowered = memory.lowered_for(qpu)

    assert lowered.decode_operations == ()


def test_a_live_memory_is_the_live_maker_over_its_fragments(
    tmp_path: pathlib.Path,
):
    program = memory_programs.memory_program()
    program = dataclasses.replace(program, round_period_microseconds=1.1)
    workload = producers.live_memory(program, 2)
    workload_files.write_workload(workload, tmp_path)
    fragments = tmp_path / "fragments"
    memory = producers.LiveMemorySettings(fragments, 2)
    qpu = _qpu(3)

    lowered = memory.lowered_for(qpu)

    assert lowered == workload_settings.WorkloadSettings.running(workload)


def test_a_live_memory_fixes_the_distance_and_a_memory_follows_it():
    fragments = pathlib.Path("fragments")
    live = producers.LiveMemorySettings(fragments, 2)
    memory = producers.MemorySettings(0.002, rounds=4)
    lowered = workload_settings.WorkloadSettings()

    assert live.fixes_the_distance
    assert not memory.fixes_the_distance
    assert lowered.fixes_the_distance


def _run(graph, seed):
    machine = machine_module.Machine.build(graph, seed)
    result = machine.run()
    return machine, result


def _with_workload(graph, workload):
    running = workload_settings.WorkloadSettings.running(workload)
    return graph.with_held("workload", running)


def _commands(machine) -> list:
    events = machine.observation.command_events.events
    commands = []
    for event in events:
        operation_id = event.command.operation.id
        commands.append((event.kind, event.tick, operation_id))
    return commands


def test_the_deltakit_memory_maker_runs_what_the_example_tool_runs():
    pytest.importorskip("deltakit_explorer")
    made = producers.deltakit_memory(24, 3, 0.001)
    circuit, rounds = deltakit.memory_circuit(
        "rotated_surface", 3, 24, "Z", 0.001
    )
    tool_workload = deltakit_example.memory_workload(
        circuit, rounds, 24, "memory-patch"
    )
    tool_graph = deltakit_example.supplied_machine(
        tool_workload,
        distance=3,
        round_count=24,
        period_microseconds=1.1,
        feedback_microseconds=4.0,
    )
    made_graph = _with_workload(tool_graph, made)
    machine, result = _run(made_graph, 0)
    tool_machine, tool_result = _run(tool_graph, 0)

    assert dataclasses.asdict(result) == dataclasses.asdict(tool_result)
    assert _commands(machine) == _commands(tool_machine)


def test_the_live_deltakit_memory_maker_runs_what_the_live_tool_runs():
    """The owner, region and rounds the tool writes by hand are derived."""
    pytest.importorskip("deltakit_explorer")
    made = producers.deltakit_live_memory(
        3, 0.001, round_period_microseconds=1.1, decode_after_rounds=3
    )
    program = deltakit.memory_rounds(
        "rotated_surface", 3, "Z", 0.001, round_period_microseconds=1.1
    )
    stream_id = producers.LIVE_STREAM_ID
    tool_graph = live_memory_example.live_machine(
        program,
        distance=3,
        round_period_microseconds=1.1,
        prefix_round_count=3,
        patch="memory-patch",
        feedback_microseconds=4.0,
        decoder_microseconds=0.1,
    )
    made_graph = _with_workload(tool_graph, made)
    machine, result = _run(made_graph, 17)
    tool_machine, tool_result = _run(tool_graph, 17)
    measurements = machine.qpu.syndrome_source.sampled_measurements(stream_id)
    tool_source = tool_machine.qpu.syndrome_source
    tool_measurements = tool_source.sampled_measurements(stream_id)

    assert dataclasses.asdict(result) == dataclasses.asdict(tool_result)
    assert _commands(machine) == _commands(tool_machine)
    assert measurements == tool_measurements


def _qpu(distance) -> qpu_settings.QpuSettings:
    return qpu_settings.QpuSettings(distance=distance)


@dataclasses.dataclass(frozen=True)
class _ReplacedCircuit:
    """A noise row whose circuit is the one it holds, whatever Stim made."""

    circuit: stim.Circuit
    name = "replaced_circuit"

    def generator_noise(
        self, probability: float
    ) -> workload_settings.StimNoise:
        del probability
        return workload_settings.NO_GENERATOR_NOISE

    def noisy(self, circuit: stim.Circuit, probability: float) -> stim.Circuit:
        del circuit, probability
        return self.circuit
