"""`decsim describe` (experiments/describe_command.py).

The referent is the task the run file builds: the YAML describe prints
names the task's parts, the json holds the same graph, and each path
line is the task's own card.
"""

import json
import pathlib

import pytest
import yaml

import decsim.experiments.command as command
import decsim.parameters.assignments as assignments
import decsim.topology.graph_layout as graph_layout
import tests.experiments.run_files as run_files

# a run file whose one task is a graph
GRAPH_RUN_FILE = '''"""A graph task, for describe's tests."""

import decsim
import tests.experiments.run_files as run_files

machine = run_files.minimal_machine()
collection = decsim.CollectionSettings(max_shots=1)
task = decsim.Task("graph_task", machine, collection=collection)
experiment = decsim.Experiment("graph", [task])
'''


@pytest.fixture(autouse=True)
def _no_assignments_left_behind(monkeypatch):
    """Each test starts with DECSIM_SET empty and leaves it as it was."""
    monkeypatch.setenv(assignments.ENVIRONMENT_VARIABLE, "")


def _graph_run_file(directory: pathlib.Path) -> pathlib.Path:
    path = directory / "graph_run.py"
    path.write_text(GRAPH_RUN_FILE)
    return path


def _card_paths(machine) -> list:
    """The paths the machine's own card prices, at its task."""
    at_task = machine.at_task()
    layout = graph_layout.layout_of(at_task)
    return layout.card.path_names()


def _described(run_file: pathlib.Path, capsys, *flags) -> str:
    command.main(["describe", str(run_file), *flags])
    printed = capsys.readouterr()
    return printed.out


def test_describe_prints_the_graph_and_every_path(tmp_path, capsys):
    run_file = _graph_run_file(tmp_path)
    machine = run_files.minimal_machine()
    card_paths = _card_paths(machine)

    text = _described(run_file, capsys)

    graph_text, _, path_text = text.partition("# its paths")
    graph_document = yaml.safe_load(graph_text)
    lines = path_text.splitlines()
    path_lines = lines[1:]
    part_names = [part.name for part in machine.parts]
    assert list(graph_document["parts"]) == part_names
    assert len(path_lines) == len(card_paths)


def test_describe_shows_a_graph_path_set_on_the_command_line(tmp_path, capsys):
    run_file = _graph_run_file(tmp_path)

    text = _described(run_file, capsys, "--set", "weak.unit_count=5")

    assert "unit_count: 5" in text


def test_describe_writes_a_swept_task_as_a_graph(tmp_path, capsys):
    run_file = run_files.write_run_file(tmp_path)

    text = _described(run_file, capsys)

    assert "weak_buffer:" in text
    assert "qpu_to_controller" in text


def test_describe_json_holds_the_graph_and_paths_the_text_prints(
    tmp_path, capsys
):
    run_file = _graph_run_file(tmp_path)
    text = _described(run_file, capsys)
    graph_text, _, _ = text.partition("# its paths")
    machine = run_files.minimal_machine()
    card_paths = _card_paths(machine)
    from_text = yaml.safe_load(graph_text)

    json_text = _described(run_file, capsys, "--json")
    described = json.loads(json_text)

    assert described["graph"] == from_text
    assert len(described["paths"]) == len(card_paths)
    assert set(described["paths"]["qpu_to_controller"]) == {"latency", "rate"}
