"""The experiment helper: gem5 MultiSim's command line over sinter.

The referents are gem5 v24.0's MultiSim (RELEASE-NOTES.md, "gem5
MultiSim": `script.py --list` names every run by id and `script.py <id>`
runs one) and sinter 1.16.0 itself: a point's CSV is sinter's own
save_resume_filepath, which a rerun reads back and extends
(sinter/_collection/_collection.py), and stats.csv is sinter's combine,
read_stats_from_csv_files with its CSV_HEADER
(sinter/_command/_main_combine.py).
"""

import datetime
import os
import pathlib
import subprocess
import sys

import pytest
import sinter
import stim

import decsim.experiment_runner as experiment_runner
import decsim.experiments.run_folder as run_folder
import decsim.sinter_adapters.union_find as union_find_adapter

_THIS_FILE = pathlib.Path(__file__)
_TEST_FILE = _THIS_FILE.resolve()
REPOSITORY_ROOT = _TEST_FILE.parents[1]
# A small experiment: two PyMatching points that stop at a few errors.
SCRIPT_TEXT = """\
import stim

import decsim.experiment_runner as experiment_runner

experiment = experiment_runner.Experiment("tiny")
for distance in (3, 5):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=3,
        after_clifford_depolarization=0.02,
    )
    experiment.add_offline(
        circuit,
        decoder="pymatching",
        labels={"d": distance},
        max_errors=10,
        max_shots=5000,
    )

if __name__ == "__main__":
    experiment.main()
"""


def test_list_prints_each_points_id_decoder_and_labels(tmp_path, capsys):
    experiment = tiny_experiment()

    experiment.main(arguments=["--list"])

    printed = capsys.readouterr()
    written = os.listdir(tmp_path)
    assert printed.out == "0 decoder=pymatching d=3\n1 decoder=pymatching d=5\n"
    assert written == []


def test_a_point_run_by_id_writes_only_its_own_csv(tmp_path, monkeypatch):
    script = write_script(tmp_path, monkeypatch)
    results_folder = tmp_path / "out"
    experiment = tiny_experiment()

    experiment.main(arguments=["1", "--out", str(results_folder)])

    points = results_folder / "points"
    point_path = points / "1.csv"
    (point_stats,) = sinter.read_stats_from_csv_files(point_path)
    written = os.listdir(points)
    script_copy = results_folder / "run.py"
    copy_text = script_copy.read_text()
    script_text = script.read_text()
    assert written == ["1.csv"]
    assert point_stats.json_metadata == {"d": 5}
    assert point_stats.errors >= 10
    assert copy_text == script_text


def test_a_finished_point_run_again_takes_no_new_shots(tmp_path, monkeypatch):
    write_script(tmp_path, monkeypatch)
    results_folder = tmp_path / "out"
    experiment = tiny_experiment()
    experiment.main(arguments=["0", "--out", str(results_folder)])
    first_text = (results_folder / "points" / "0.csv").read_text()

    experiment.main(arguments=["0", "--out", str(results_folder)])

    assert (results_folder / "points" / "0.csv").read_text() == first_text


def test_a_point_its_shot_cap_stopped_takes_no_new_shots(tmp_path, monkeypatch):
    write_script(tmp_path, monkeypatch)
    results_folder = tmp_path / "out"
    experiment = shot_capped_experiment(300)
    experiment.main(arguments=["0", "--out", str(results_folder)])
    point_path = results_folder / "points" / "0.csv"
    first_text = point_path.read_text()

    experiment.main(arguments=["0", "--out", str(results_folder)])

    (point_stats,) = sinter.read_stats_from_csv_files(point_path)
    assert point_path.read_text() == first_text
    assert point_stats.shots == 300


def test_an_unfinished_point_goes_on_from_its_saved_shots(
    tmp_path, monkeypatch
):
    """A raised shot cap leaves the saved rows and adds only the rest."""
    write_script(tmp_path, monkeypatch)
    results_folder = tmp_path / "out"
    first = shot_capped_experiment(300)
    first.main(arguments=["0", "--out", str(results_folder)])
    point_path = results_folder / "points" / "0.csv"
    first_text = point_path.read_text()
    raised = shot_capped_experiment(700)

    raised.main(arguments=["0", "--out", str(results_folder)])

    (point_stats,) = sinter.read_stats_from_csv_files(point_path)
    raised_text = point_path.read_text()
    assert raised_text.startswith(first_text)
    assert point_stats.shots == 700


def test_each_point_decodes_with_the_decoder_its_name_maps_to(
    tmp_path, monkeypatch
):
    """A named decoder is handed to sinter; None names a sinter built-in."""
    write_script(tmp_path, monkeypatch)
    results_folder = tmp_path / "out"
    experiment = experiment_runner.Experiment("tiny")
    circuit = tiny_circuit(3)
    experiment.add_offline(
        circuit, "union-find", {"d": 3}, max_errors=5, max_shots=2000
    )
    experiment.add_offline(
        circuit, "pymatching", {"d": 3}, max_errors=5, max_shots=2000
    )
    decoders = {
        "union-find": union_find_adapter.UnionFindDecoder(),
        "pymatching": None,
    }

    experiment.main(decoders, arguments=["--out", str(results_folder)])

    stats_path = results_folder / "stats.csv"
    combined = sinter.read_stats_from_csv_files(stats_path)
    decoder_names = {point_stats.decoder for point_stats in combined}
    assert decoder_names == {"union-find", "pymatching"}


def test_two_points_run_at_once_write_different_files(tmp_path):
    script = tmp_path / "run.py"
    script.write_text(SCRIPT_TEXT)
    results_folder = tmp_path / "out"
    environment = dict(os.environ, PYTHONPATH=str(REPOSITORY_ROOT))

    first = run_in_background(script, "0", results_folder, environment)
    second = run_in_background(script, "1", results_folder, environment)

    assert first.wait() == 0
    assert second.wait() == 0
    first_path = results_folder / "points" / "0.csv"
    second_path = results_folder / "points" / "1.csv"
    (first_stats,) = sinter.read_stats_from_csv_files(first_path)
    (second_stats,) = sinter.read_stats_from_csv_files(second_path)
    written = os.listdir(results_folder)
    assert first_stats.json_metadata == {"d": 3}
    assert second_stats.json_metadata == {"d": 5}
    assert sorted(written) == ["commit.txt", "points", "run.py"]


def test_combine_writes_every_points_stats_into_one_csv(tmp_path, monkeypatch):
    write_script(tmp_path, monkeypatch)
    results_folder = tmp_path / "out"
    experiment = tiny_experiment()
    experiment.main(arguments=["0", "--out", str(results_folder)])
    experiment.main(arguments=["1", "--out", str(results_folder)])

    experiment.main(arguments=["combine", "--out", str(results_folder)])

    stats_path = results_folder / "stats.csv"
    first_path = results_folder / "points" / "0.csv"
    second_path = results_folder / "points" / "1.csv"
    combined = sinter.read_stats_from_csv_files(stats_path)
    points = sinter.read_stats_from_csv_files(first_path, second_path)
    stats_text = stats_path.read_text()
    assert combined == points
    assert stats_text.startswith(sinter.CSV_HEADER)


def test_a_run_of_every_point_ends_with_stats_csv(tmp_path, monkeypatch):
    write_script(tmp_path, monkeypatch)
    results_folder = tmp_path / "out"
    experiment = tiny_experiment()

    experiment.main(arguments=["--out", str(results_folder), "--workers", "2"])

    stats_path = results_folder / "stats.csv"
    combined = sinter.read_stats_from_csv_files(stats_path)
    assert len(combined) == 2


def test_the_folder_records_the_commit_that_ran(tmp_path, monkeypatch):
    write_script(tmp_path, monkeypatch)
    set_the_tree(monkeypatch, "abc123", False)
    results_folder = tmp_path / "out"
    experiment = tiny_experiment()

    experiment.main(arguments=["0", "--out", str(results_folder)])

    commit_path = results_folder / "commit.txt"
    commit_text = commit_path.read_text()
    assert commit_text == "commit abc123\ndirty False\n"


def test_a_tree_whose_commit_cannot_be_read_is_refused(tmp_path, monkeypatch):
    write_script(tmp_path, monkeypatch)
    set_the_tree(monkeypatch, None, None)
    results_folder = tmp_path / "out"
    experiment = tiny_experiment()

    with pytest.raises(ValueError) as refused:
        experiment.main(arguments=["0", "--out", str(results_folder)])

    assert str(refused.value) == (
        "the commit of the decsim tree running this script cannot be read, "
        "so commit.txt could not say what ran; run from a git checkout, "
        "whose .git folder holds its HEAD"
    )
    assert not results_folder.exists()


def test_a_node_with_no_git_program_reads_the_commit_from_git_files(
    tmp_path, monkeypatch
):
    """run_folder reads .git itself, as on a container with no git."""
    write_script(tmp_path, monkeypatch)
    no_programs = tmp_path / "no_programs"
    no_programs.mkdir()
    monkeypatch.setenv("PATH", str(no_programs))
    run_folder._tree_reading.cache_clear()
    results_folder = tmp_path / "out"
    experiment = tiny_experiment()

    experiment.main(arguments=["0", "--out", str(results_folder)])

    run_folder._tree_reading.cache_clear()
    commit_path = results_folder / "commit.txt"
    commit_text = commit_path.read_text()
    commit_line, _dirty_line = commit_text.splitlines()
    commit = commit_line.removeprefix("commit ")
    assert len(commit) == 40
    assert int(commit, 16) >= 0


@pytest.mark.parametrize(
    ("held_dirty", "is_dirty"), [("None", False), ("False", None)]
)
def test_a_side_that_could_not_read_git_is_not_refused(
    tmp_path, monkeypatch, held_dirty, is_dirty
):
    """A gitless node records dirty None; the commit still decides."""
    script = write_script(tmp_path, monkeypatch)
    results_folder = folder_of_the_run(tmp_path, script, "abc123", held_dirty)
    set_the_tree(monkeypatch, "abc123", is_dirty)
    experiment = tiny_experiment()

    experiment.main(arguments=["0", "--out", str(results_folder)])

    point_path = results_folder / "points" / "0.csv"
    assert point_path.exists()


@pytest.mark.parametrize(
    ("held_commit", "held_dirty", "is_dirty"),
    [("def456", "False", False), ("abc123", "True", False)],
)
def test_another_commit_or_a_known_dirty_difference_is_refused(
    tmp_path, monkeypatch, held_commit, held_dirty, is_dirty
):
    script = write_script(tmp_path, monkeypatch)
    results_folder = folder_of_the_run(
        tmp_path, script, held_commit, held_dirty
    )
    set_the_tree(monkeypatch, "abc123", is_dirty)
    experiment = tiny_experiment()

    with pytest.raises(ValueError) as refused:
        experiment.main(arguments=["0", "--out", str(results_folder)])

    assert str(refused.value) == (
        f"{results_folder / 'commit.txt'} records commit {held_commit} dirty "
        f"{held_dirty}, and this is commit abc123 dirty {is_dirty}; a "
        "results folder belongs to one script and one commit, so give "
        "--out a new folder"
    )


def test_a_folder_holding_another_scripts_copy_is_refused(
    tmp_path, monkeypatch
):
    write_script(tmp_path, monkeypatch)
    results_folder = tmp_path / "out"
    results_folder.mkdir()
    (results_folder / "run.py").write_text("another script\n")
    experiment = tiny_experiment()

    with pytest.raises(ValueError) as refused:
        experiment.main(arguments=["0", "--out", str(results_folder)])

    assert str(refused.value) == (
        f"{results_folder / 'run.py'} already holds another run.py; a "
        "results folder belongs to one script and one commit, so give "
        "--out a new folder"
    )


def test_a_task_on_another_node_with_the_same_process_id_keeps_its_file(
    tmp_path, monkeypatch
):
    """Process ids repeat across nodes; a staged copy is named at random."""
    write_script(tmp_path, monkeypatch)
    results_folder = tmp_path / "out"
    results_folder.mkdir()
    other_staging = results_folder / ".run.py.4242"
    other_staging.write_text("the other node's staged copy\n")
    monkeypatch.setattr(os, "getpid", lambda: 4242)
    experiment = tiny_experiment()

    experiment.main(arguments=["0", "--out", str(results_folder)])

    other_text = other_staging.read_text()
    assert other_text == "the other node's staged copy\n"


def test_the_script_copy_and_commit_record_keep_the_umasks_permissions(
    tmp_path, monkeypatch
):
    """Another user sharing the folder reads them, as it reads the CSVs."""
    write_script(tmp_path, monkeypatch)
    results_folder = tmp_path / "out"
    experiment = tiny_experiment()
    umask = os.umask(0)
    os.umask(umask)
    ordinary_mode = 0o666 & ~umask

    experiment.main(arguments=["0", "--out", str(results_folder)])

    script_copy = results_folder / "run.py"
    commit_path = results_folder / "commit.txt"
    script_status = script_copy.stat()
    commit_status = commit_path.stat()
    assert script_status.st_mode & 0o777 == ordinary_mode
    assert commit_status.st_mode & 0o777 == ordinary_mode


def test_a_publish_that_fails_leaves_no_staging_file(tmp_path, monkeypatch):
    write_script(tmp_path, monkeypatch)
    results_folder = tmp_path / "out"
    results_folder.mkdir()
    monkeypatch.setattr(os, "link", refuse_the_link)
    experiment = tiny_experiment()

    with pytest.raises(PermissionError):
        experiment.main(arguments=["0", "--out", str(results_folder)])

    left_behind = os.listdir(results_folder)
    assert left_behind == []


@pytest.mark.parametrize("worker_count", ["0", "-2"])
def test_fewer_than_one_worker_is_refused(tmp_path, monkeypatch, worker_count):
    write_script(tmp_path, monkeypatch)
    results_folder = tmp_path / "out"
    experiment = tiny_experiment()
    arguments = ["0", "--workers", worker_count, "--out", str(results_folder)]

    with pytest.raises(ValueError) as refused:
        experiment.main(arguments=arguments)

    assert str(refused.value) == (
        f"--workers is {worker_count}; sinter needs at least one worker process"
    )
    assert not results_folder.exists()


def test_an_id_past_the_last_point_is_refused(tmp_path, monkeypatch):
    write_script(tmp_path, monkeypatch)
    experiment = tiny_experiment()

    results_folder = tmp_path / "out"

    with pytest.raises(ValueError) as refused:
        experiment.main(arguments=["2", "--out", str(results_folder)])

    assert str(refused.value) == (
        "point 2 is not in this experiment; --list shows its 2 points, 0 to 1"
    )


def test_a_target_that_is_no_id_is_refused(tmp_path, monkeypatch):
    write_script(tmp_path, monkeypatch)
    experiment = tiny_experiment()

    results_folder = tmp_path / "out"

    with pytest.raises(ValueError) as refused:
        experiment.main(arguments=["seven", "--out", str(results_folder)])

    assert str(refused.value) == (
        "'seven' is no point id; give a number --list shows, or combine"
    )


def test_without_out_the_folder_is_dated_under_the_results_root(
    tmp_path, monkeypatch
):
    write_script(tmp_path, monkeypatch)
    results_root = tmp_path / "results"
    monkeypatch.setenv("DECSIM_RESULTS", str(results_root))
    experiment = tiny_experiment()

    experiment.main(arguments=["0"])

    today = datetime.date.today()
    date_text = today.isoformat()
    folder = results_root / f"{date_text}_tiny"
    assert (folder / "points" / "0.csv").exists()


def tiny_experiment() -> experiment_runner.Experiment:
    """The script's two points, built in this process."""
    experiment = experiment_runner.Experiment("tiny")
    for distance in (3, 5):
        circuit = tiny_circuit(distance)
        experiment.add_offline(
            circuit,
            decoder="pymatching",
            labels={"d": distance},
            max_errors=10,
            max_shots=5000,
        )
    return experiment


def shot_capped_experiment(max_shots: int) -> experiment_runner.Experiment:
    """One point that its shot cap stops, far short of its error target."""
    experiment = experiment_runner.Experiment("tiny")
    circuit = tiny_circuit(3)
    experiment.add_offline(
        circuit,
        decoder="pymatching",
        labels={"d": 3},
        max_errors=1_000_000,
        max_shots=max_shots,
    )
    return experiment


def tiny_circuit(distance: int) -> stim.Circuit:
    """The script's circuit: memory Z, three rounds, two percent noise."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=3,
        after_clifford_depolarization=0.02,
    )


def write_script(tmp_path, monkeypatch) -> pathlib.Path:
    """The script on disk, named as the running one, as python run.py is."""
    script = tmp_path / "run.py"
    script.write_text(SCRIPT_TEXT)
    monkeypatch.setattr(sys, "argv", [str(script)])
    return script


def refuse_the_link(source, destination) -> None:
    """A file system that will not publish the staged copy."""
    raise PermissionError(f"cannot link {source} to {destination}")


def set_the_tree(monkeypatch, commit: str, is_dirty) -> None:
    """The commit and dirty flag this process reads of its tree."""
    identity = {
        "commit": commit,
        "dirty": is_dirty,
        "host": "node",
        "slurm_job_id": None,
    }
    monkeypatch.setattr(run_folder, "piece_identity", lambda: identity)


def folder_of_the_run(tmp_path, script, commit: str, dirty: str):
    """A results folder an earlier task of the same script recorded."""
    results_folder = tmp_path / "out"
    results_folder.mkdir()
    script_text = script.read_text()
    script_copy = results_folder / "run.py"
    script_copy.write_text(script_text)
    commit_path = results_folder / "commit.txt"
    commit_path.write_text(f"commit {commit}\ndirty {dirty}\n")
    return results_folder


def run_in_background(script, point_id: str, results_folder, environment):
    """The script run on one point in its own process, not waited for."""
    command = [
        sys.executable,
        str(script),
        point_id,
        "--out",
        str(results_folder),
    ]
    return subprocess.Popen(command, env=environment)
