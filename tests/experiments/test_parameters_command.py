"""`decsim params` and `decsim info` (experiments/parameters_command.py).

Each listing is pinned against the tree it prints: params prints
tree.settings_of of the task the run file builds, and info the
parameters of the record it names, so the command line is never the
only record of what can be set.
"""

import pytest

import decsim.experiments.command as command
import decsim.experiments.experiment as experiment
import decsim.experiments.parameters_command as parameters_command
import decsim.parameters.assignments as assignments
import decsim.parameters.tree as tree
import tests.experiments.run_files as run_files
import tests.experiments.test_describe_command as test_describe_command


@pytest.fixture(autouse=True)
def _no_assignments_left_behind(monkeypatch):
    """Each test starts with DECSIM_SET empty and leaves it as it was."""
    monkeypatch.setenv(assignments.ENVIRONMENT_VARIABLE, "")


SWEEP_RUN_FILE = """
import pathlib

import decsim
import decsim.presets.machines as machines
import decsim.topology.constants as constants_module

HERE = pathlib.Path(__file__)
constants = constants_module.load(HERE.with_name("constants.yaml"))


def machine(distance):
    rate = constants.memory.probability
    return machines.weak_baseline(distance, rate, 1.0)


sweep = decsim.Sweep(
    "rates",
    machine,
    {"distance": [3]},
    "d{distance}",
    stop=decsim.CollectionSettings(max_shots=10),
    constants=constants,
)
experiment = sweep.experiment()
"""
CONSTANTS = "memory:\n  probability: 0.002  # a chosen rate\n"


def _printed_lines(capsys) -> list:
    printed = capsys.readouterr()
    return printed.out.splitlines()


def test_params_prints_every_path_of_the_first_task(tmp_path, capsys):
    run_file = run_files.write_run_file(tmp_path)
    study = experiment.load(run_file)
    first = study.tasks[0]
    settings = tree.settings_of(first.machine)

    command.main(["params", str(run_file)])

    lines = _printed_lines(capsys)
    assert len(lines) == len(settings) + 1
    assert lines[0].startswith(f"# {first.name}:")
    assert "qpu.distance = 3  [none]  (run file only)" in lines


def test_params_shows_a_set_value_its_record_default_and_its_setter(
    tmp_path, capsys
):
    run_file = run_files.write_run_file(tmp_path)

    command.main(["params", str(run_file), "--set", "weak.unit_count=4"])

    lines = _printed_lines(capsys)
    variable = assignments.ENVIRONMENT_VARIABLE
    expected = f"weak.unit_count = 4  [1]  ({variable})"
    assert expected in lines


def test_params_match_keeps_the_paths_its_pattern_reaches(tmp_path, capsys):
    run_file = run_files.write_run_file(tmp_path)

    command.main(["params", str(run_file), "--match", "**.engine.clock"])

    lines = _printed_lines(capsys)
    assert lines[1:] == ["weak.engine.clock = 250MHz  [none]"]


def test_info_with_no_name_lists_every_slot_and_the_records_it_takes(
    capsys,
):
    slots = parameters_command.every_slot()

    command.main(["info"])

    lines = _printed_lines(capsys)
    assert len(lines) == len(slots) + 1
    algorithm = [
        line for line in lines if line.startswith("DecoderPool.algorithm: ")
    ]
    assert len(algorithm) == 1
    assert "union_find" in algorithm[0]


def test_info_reads_a_name_with_a_dot_as_an_import_path(capsys):
    path = "tests.parameters.test_kinds.PluginDecoder.Settings"

    command.main(["info", path])

    lines = _printed_lines(capsys)
    assert lines[0] == f"nested_plugin: {path}"


def test_info_prints_a_records_parameters_and_its_parts_choices(capsys):
    command.main(["info", "DecoderPool"])

    lines = _printed_lines(capsys)
    assert (
        lines[0] == "DecoderPool: decsim.decoders.settings.DecoderPoolSettings"
    )
    assert "  unit_count = 1  (int)" in lines
    algorithm = [line for line in lines if line.startswith("  algorithm.type")]
    assert "union_find" in algorithm[0]


def test_info_refuses_an_unknown_record_with_the_near_names(capsys):
    with pytest.raises(SystemExit):
        command.main(["info", "DecoderPol"])

    printed = capsys.readouterr()
    assert "decsim defines no record DecoderPol. Did you mean" in printed.err


def test_set_flags_are_appended_to_decsim_set_after_it(monkeypatch):
    variable = assignments.ENVIRONMENT_VARIABLE
    monkeypatch.setenv(variable, "a.b=1")

    parameters_command.put_in_force(["a.b=2", "a.c=3"])

    assert assignments.from_environment() == [
        tree.Assignment("a.b", 1, variable),
        tree.Assignment("a.b", 2, variable),
        tree.Assignment("a.c", 3, variable),
    ]


def test_params_prints_a_graph_tasks_paths_by_part_name(tmp_path, capsys):
    run_file = tmp_path / "graph_run.py"
    run_file.write_text(test_describe_command.GRAPH_RUN_FILE)

    command.main(["params", str(run_file)])

    lines = _printed_lines(capsys)
    assert "weak.unit_count = 1" in lines
    assert "qpu.distance = 3  [none]  (run file only)" in lines


def test_params_lists_each_constant_with_its_source(tmp_path, capsys):
    run_file = _sweep_run_file(tmp_path)

    command.main(["params", str(run_file), "--match", "constants.**"])

    lines = _printed_lines(capsys)
    assert lines[-1] == "constants.memory.probability = 0.002  (a chosen rate)"


def test_params_lists_a_set_constant_with_its_setter(tmp_path, capsys):
    run_file = _sweep_run_file(tmp_path)
    setting = "constants.memory.probability=0.004"

    command.main(["params", str(run_file), "--set", setting])

    lines = _printed_lines(capsys)
    variable = assignments.ENVIRONMENT_VARIABLE
    expected = f"constants.memory.probability = 0.004  ({variable} {setting})"
    assert lines[-1] == expected


def _sweep_run_file(tmp_path):
    constants = tmp_path / "constants.yaml"
    constants.write_text(CONSTANTS)
    run_file = tmp_path / "sweep_run.py"
    run_file.write_text(SWEEP_RUN_FILE)
    return run_file
