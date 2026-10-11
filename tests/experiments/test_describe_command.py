"""`decsim describe` (experiments/describe_command.py).

The referent is the task the run file builds: the YAML describe prints
reads back as the task's graph, and each path line is the task's own
card, so the printout is never the only record of what ran.
"""

import json
import pathlib

import pytest

import decsim.experiments.command as command
import decsim.parameters.assignments as assignments
import decsim.records.transfers as transfer_records
import decsim.topology.machine_file as machine_file
import tests.experiments.run_files as run_files

# a run file whose one task is a graph
GRAPH_RUN_FILE = '''"""A graph task, for describe's tests."""

import decsim
import decsim.topology.legacy as legacy
import tests.experiments.run_files as run_files

settings = run_files.minimal_machine()
machine = legacy.graph_of(settings)
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


def _described(run_file: pathlib.Path, capsys, *flags) -> str:
    command.main(["describe", str(run_file), *flags])
    printed = capsys.readouterr()
    return printed.out


def test_describe_prints_the_graph_and_every_path(tmp_path, capsys):
    run_file = _graph_run_file(tmp_path)
    settings = run_files.minimal_machine()

    text = _described(run_file, capsys)

    graph_text, _, path_text = text.partition("# its paths")
    workload = {"workload": settings.workload}
    read = machine_file.loads(graph_text, python_parts=workload)
    lines = path_text.splitlines()
    path_lines = lines[1:]
    pool = read.part("weak")
    assert pool.settings == settings.weak_decoder
    assert len(path_lines) == len(transfer_records.LEGACY_PATHS)


def test_describe_shows_a_graph_path_set_on_the_command_line(tmp_path, capsys):
    run_file = _graph_run_file(tmp_path)

    text = _described(run_file, capsys, "--set", "weak.unit_count=5")

    assert "unit_count: 5" in text


def test_describe_writes_a_task_built_from_settings_as_a_graph(
    tmp_path, capsys
):
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
    settings = run_files.minimal_machine()
    workload = {"workload": settings.workload}
    from_text = machine_file.loads(graph_text, python_parts=workload)

    json_text = _described(run_file, capsys, "--json")
    described = json.loads(json_text)

    graph_json = json.dumps(described["graph"])
    from_json = machine_file.loads(graph_json, python_parts=workload)
    assert from_json == from_text
    assert len(described["paths"]) == len(transfer_records.LEGACY_PATHS)
    assert set(described["paths"]["qpu_to_controller"]) == {"latency", "rate"}
