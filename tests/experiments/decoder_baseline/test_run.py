"""The decoder baseline script against decsim's own workload.

The referent for the circuit is decsim.producers.memory_circuit, the
producer the machine's baseline run file calls, so the offline and the
machine baselines sample one circuit. The rest pins the stop rule: 100
errors a task, and a shot limit no time limit reaches, so Slurm's time
limit is the budget; and the command line, gem5 MultiSim's, over
sinter's own resume CSV and combine (sinter/_collection/_collection.py,
sinter/_command/_main_combine.py), on a tiny grid in place of the
baseline's.
"""

import importlib.util
import json
import pathlib

import pytest
import sinter
import stim

import decsim.producers as producers
import decsim.sinter_adapters.relay_bp as relay_bp_adapter

pytest.importorskip("relay_bp")
pytest.importorskip("tesseract_decoder")

_THIS_FILE = pathlib.Path(__file__)
_TEST_FILE = _THIS_FILE.resolve()
REPOSITORY_ROOT = _TEST_FILE.parents[3]
SCRIPT = REPOSITORY_ROOT / "experiments" / "decoder_baseline" / "run.py"


def test_the_circuit_is_decsims_memory_circuit():
    run = script_module()

    circuit = run.circuit("z", 5, 0.001)

    workload = producers.memory_circuit(
        "surface_code:rotated_memory_z", 100, 5, 0.001
    )
    (operation,) = workload.operations
    assert str(circuit) == str(operation.circuit)


def test_the_baseline_has_a_task_per_basis_distance_rate_and_decoder():
    run = script_module()

    tasks = run.baseline()

    assert len(tasks) == 2 * 6 * 6 * 4


def test_every_task_stops_at_100_errors_or_a_billion_shots():
    run = script_module()

    tasks = run.baseline()

    options = [task.sinter_task.collection_options for task in tasks]
    limits = {(option.max_errors, option.max_shots) for option in options}
    assert limits == {(100, 1_000_000_000)}


def test_a_relay_bp_task_decodes_with_decsims_row_built_on_its_circuit():
    run = script_module()

    tasks = run.baseline()

    relay_tasks = _tasks_decoded_by(tasks, run.RELAY_BP)
    row = relay_bp_adapter.RelayBeliefPropagationDecoder
    is_the_row = {isinstance(task.decoder, row) for task in relay_tasks}
    is_on_its_circuit = {
        task.decoder.circuit is task.sinter_task.circuit for task in relay_tasks
    }
    assert is_the_row == {True}
    assert is_on_its_circuit == {True}


def test_list_prints_each_tasks_id_decoder_and_labels(capsys):
    run = script_module()

    run.main(["--list"])

    printed = capsys.readouterr()
    lines = printed.out.splitlines()
    assert len(lines) == 288
    assert lines[0] == "0 decoder=union-find basis=x d=5 p=0.0005"


def test_a_task_run_by_id_writes_only_its_own_csv(tmp_path, monkeypatch):
    run = script_module()
    tiny_baseline = _tiny_baseline(run)
    monkeypatch.setattr(run, "baseline", tiny_baseline)
    folder = tmp_path / "results"

    run.main(["1", "--out", str(folder)])

    saved = sorted(path.name for path in (folder / "tasks").iterdir())
    record_path = folder / "run.json"
    record_text = record_path.read_text()
    record = json.loads(record_text)
    assert saved == ["1.csv"]
    assert record["tasks"] == [0, 1]
    assert (folder / "run.py").exists()
    assert not (folder / "stats.csv").exists()


def test_a_finished_task_run_again_takes_no_new_shots(tmp_path, monkeypatch):
    run = script_module()
    tiny_baseline = _tiny_baseline(run)
    monkeypatch.setattr(run, "baseline", tiny_baseline)
    folder = tmp_path / "results"
    run.main(["0", "--out", str(folder)])
    first = _saved_shots(run, folder, 0)

    run.main(["0", "--out", str(folder)])

    assert _saved_shots(run, folder, 0) == first


def test_a_run_of_every_task_ends_with_stats_csv(tmp_path, monkeypatch):
    run = script_module()
    tiny_baseline = _tiny_baseline(run)
    monkeypatch.setattr(run, "baseline", tiny_baseline)
    folder = tmp_path / "results"

    run.main(["--out", str(folder)])

    stats_path = folder / "stats.csv"
    stats = sinter.read_stats_from_csv_files(stats_path)
    distances = sorted(row.json_metadata["d"] for row in stats)
    assert distances == [3, 5]


@pytest.mark.parametrize(
    "arguments, sentence",
    [
        (["2"], "is no task id"),
        (["x"], "is no task id"),
        (["0"], "name it with --out"),
        (["--workers", "0"], "sinter needs at least one worker"),
    ],
)
def test_a_command_line_that_names_no_run_is_refused(
    tmp_path, monkeypatch, capsys, arguments, sentence
):
    run = script_module()
    tiny_baseline = _tiny_baseline(run)
    monkeypatch.setattr(run, "baseline", tiny_baseline)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        run.main(arguments)

    printed = capsys.readouterr()
    assert sentence in printed.err
    assert not (tmp_path / "results").exists()


def _tiny_baseline(run):
    """Two PyMatching tasks that stop at a few errors, as run.baseline."""

    def tiny() -> list:
        options = sinter.CollectionOptions(max_errors=10, max_shots=5000)
        tasks = []
        for distance in (3, 5):
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                distance=distance,
                rounds=3,
                after_clifford_depolarization=0.02,
            )
            sinter_task = sinter.Task(
                circuit=circuit,
                decoder="pymatching",
                json_metadata={"d": distance},
                collection_options=options,
            )
            task = run.BaselineTask(sinter_task, None)
            tasks.append(task)
        return tasks

    return tiny


def _tasks_decoded_by(tasks, decoder: str) -> list:
    """The tasks whose sinter task names the decoder."""
    found = []
    for task in tasks:
        if task.sinter_task.decoder == decoder:
            found.append(task)
    return found


def _saved_shots(run, folder: pathlib.Path, task_id: int) -> int:
    path = run.task_path(folder, task_id)
    (saved,) = sinter.read_stats_from_csv_files(path)
    return saved.shots


def script_module():
    """experiments/decoder_baseline/run.py, imported, its main not run."""
    spec = importlib.util.spec_from_file_location("decoder_baseline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
