"""`decsim estimates` of decsim/experiments/estimates_command.py.

Its referents are the files a fold and a piece write
(docs/reference/run_folder.md): sweep.csv's sim_wall_seconds_per_shot
and each piece.json's peak_memory_mb.
"""

import json

import pytest

import decsim.experiments.collect_command as collect_command
import decsim.experiments.estimates_command as estimates_command
import decsim.experiments.experiment as experiment
import decsim.experiments.fold as fold
import decsim.experiments.pieces as pieces
import decsim.experiments.refusal as refusal
import tests.experiments.run_files as run_files

# one task, two pieces of one shot each
TWO_PIECES = {"max_shots": 2, "piece_rounds": 15}


def collected_run(tmp_path) -> tuple:
    """One task's two pieces run and folded; its folder and its name."""
    run_file = run_files.write_run_file(tmp_path, collection=TWO_PIECES)
    study = experiment.load(run_file)
    run_dir = tmp_path / "run"
    collect_command.collect_experiment(study, run_dir, run_file)
    (task,) = study.tasks
    return run_dir, task.name


def set_peaks(run_dir, peaks: list) -> None:
    """The pieces' recorded peaks, in folder order, set to these."""
    folders = pieces.every_folder(run_dir)
    for folder, peak in zip(sorted(folders), peaks, strict=True):
        piece_path = folder / pieces.PIECE_FILE
        piece_text = piece_path.read_text()
        piece = json.loads(piece_text)
        piece["peak_memory_mb"] = peak
        written = json.dumps(piece)
        piece_path.write_text(written)


def test_a_tasks_estimate_is_its_seconds_a_shot_and_its_largest_peak(
    tmp_path,
):
    run_dir, task_name = collected_run(tmp_path)
    set_peaks(run_dir, [300.0, 750.5])

    (row,) = estimates_command.estimate_rows(run_dir)

    sweep_path = run_dir / estimates_command.SWEEP_FILE
    (sweep_row,) = fold.row_stream(sweep_path)
    assert row == {
        "task": task_name,
        "core_seconds_per_shot": sweep_row["sim_wall_seconds_per_shot"],
        "peak_memory_mb": 750.5,
    }


def test_a_run_not_yet_folded_is_refused(tmp_path):
    with pytest.raises(refusal.RefusalError) as refused:
        estimates_command.estimate_rows(tmp_path)

    assert str(refused.value) == (
        f"{tmp_path} has no sweep.csv; fold it first with "
        f"decsim run --fold --out {tmp_path}"
    )
