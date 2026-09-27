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
    out = tmp_path / "out"
    experiment = tiny_experiment()

    experiment.main(arguments=["1", "--out", str(out)])

    points = out / "points"
    point_path = points / "1.csv"
    (stats,) = sinter.read_stats_from_csv_files(point_path)
    written = os.listdir(points)
    script_copy = out / "run.py"
    copy_text = script_copy.read_text()
    script_text = script.read_text()
    assert written == ["1.csv"]
    assert stats.json_metadata == {"d": 5}
    assert stats.errors >= 10
    assert copy_text == script_text


def test_a_finished_point_run_again_takes_no_new_shots(tmp_path, monkeypatch):
    write_script(tmp_path, monkeypatch)
    out = tmp_path / "out"
    experiment = tiny_experiment()
    experiment.main(arguments=["0", "--out", str(out)])
    first_text = (out / "points" / "0.csv").read_text()

    experiment.main(arguments=["0", "--out", str(out)])

    assert (out / "points" / "0.csv").read_text() == first_text


def test_two_points_run_at_once_write_different_files(tmp_path):
    script = tmp_path / "run.py"
    script.write_text(SCRIPT_TEXT)
    out = tmp_path / "out"
    environment = dict(os.environ, PYTHONPATH=str(REPOSITORY_ROOT))

    first = run_in_background(script, "0", out, environment)
    second = run_in_background(script, "1", out, environment)

    assert first.wait() == 0
    assert second.wait() == 0
    first_path = out / "points" / "0.csv"
    second_path = out / "points" / "1.csv"
    (first_stats,) = sinter.read_stats_from_csv_files(first_path)
    (second_stats,) = sinter.read_stats_from_csv_files(second_path)
    written = os.listdir(out)
    assert first_stats.json_metadata == {"d": 3}
    assert second_stats.json_metadata == {"d": 5}
    assert sorted(written) == ["commit.txt", "points", "run.py"]


def test_combine_writes_every_points_stats_into_one_csv(tmp_path, monkeypatch):
    write_script(tmp_path, monkeypatch)
    out = tmp_path / "out"
    experiment = tiny_experiment()
    experiment.main(arguments=["0", "--out", str(out)])
    experiment.main(arguments=["1", "--out", str(out)])

    experiment.main(arguments=["combine", "--out", str(out)])

    stats_path = out / "stats.csv"
    first_path = out / "points" / "0.csv"
    second_path = out / "points" / "1.csv"
    combined = sinter.read_stats_from_csv_files(stats_path)
    points = sinter.read_stats_from_csv_files(first_path, second_path)
    stats_text = stats_path.read_text()
    assert combined == points
    assert stats_text.startswith(sinter.CSV_HEADER)


def test_a_run_of_every_point_ends_with_stats_csv(tmp_path, monkeypatch):
    write_script(tmp_path, monkeypatch)
    out = tmp_path / "out"
    experiment = tiny_experiment()

    experiment.main(arguments=["--out", str(out), "--workers", "2"])

    stats_path = out / "stats.csv"
    combined = sinter.read_stats_from_csv_files(stats_path)
    assert len(combined) == 2


def test_the_folder_records_the_commit_that_ran(tmp_path, monkeypatch):
    write_script(tmp_path, monkeypatch)
    out = tmp_path / "out"
    experiment = tiny_experiment()

    experiment.main(arguments=["0", "--out", str(out)])

    commit_path = out / "commit.txt"
    commit_text = commit_path.read_text()
    lines = commit_text.splitlines()
    assert lines[0].startswith("commit ")
    assert lines[1].startswith("dirty ")


def test_a_folder_holding_another_scripts_copy_is_refused(
    tmp_path, monkeypatch
):
    write_script(tmp_path, monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    (out / "run.py").write_text("another script\n")
    experiment = tiny_experiment()

    with pytest.raises(ValueError) as refused:
        experiment.main(arguments=["0", "--out", str(out)])

    assert str(refused.value) == (
        f"{out / 'run.py'} already holds another run.py; a results folder "
        "belongs to one script and one commit, so give --out a new folder"
    )


def test_an_id_past_the_last_point_is_refused(tmp_path, monkeypatch):
    write_script(tmp_path, monkeypatch)
    experiment = tiny_experiment()

    out = tmp_path / "out"

    with pytest.raises(ValueError) as refused:
        experiment.main(arguments=["2", "--out", str(out)])

    assert str(refused.value) == (
        "point 2 is not in this experiment; --list shows its 2 points, 0 to 1"
    )


def test_a_target_that_is_no_id_is_refused(tmp_path, monkeypatch):
    write_script(tmp_path, monkeypatch)
    experiment = tiny_experiment()

    out = tmp_path / "out"

    with pytest.raises(ValueError) as refused:
        experiment.main(arguments=["seven", "--out", str(out)])

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
    return experiment


def write_script(tmp_path, monkeypatch) -> pathlib.Path:
    """The script on disk, named as the running one, as python run.py is."""
    script = tmp_path / "run.py"
    script.write_text(SCRIPT_TEXT)
    monkeypatch.setattr(sys, "argv", [str(script)])
    return script


def run_in_background(script, point_id: str, out, environment):
    """The script run on one point in its own process, not waited for."""
    command = [sys.executable, str(script), point_id, "--out", str(out)]
    return subprocess.Popen(command, env=environment)
