"""The three-file experiments run the shipped experiments' tasks.

examples/three_files holds qecboard's form (notes/qecboard_prototype):
constants.yaml, config.py on decsim's own parts, and run.py's Sweep.
Each folder's tasks are its shipped run file's: the same names in the
same order, each with the same machine, metadata, stop and id, for
Experiment 1's 60 tasks, switching per window's 64, switching's 4 and
the data-movement study's 12.
"""

import pathlib

import pytest

import decsim.experiments.experiment as experiment_module

_HERE = pathlib.Path(__file__)
_REPOSITORY = _HERE.parents[1]
THREE_FILES = _REPOSITORY / "examples" / "three_files"
TASK_COUNTS = {
    "switching_baseline": 60,
    "switching_per_window": 64,
    "switching": 4,
    "data_movement": 12,
}


@pytest.mark.parametrize("name", list(TASK_COUNTS))
def test_each_three_file_experiment_runs_its_shipped_tasks(name):
    shipped_path = experiment_module.EXPERIMENTS_DIR / name / "run.py"
    shipped = experiment_module.load(shipped_path)
    three_files_path = THREE_FILES / name / "run.py"
    three_files = experiment_module.load(three_files_path)

    shipped_tasks = _described(shipped)
    three_file_tasks = _described(three_files)

    assert three_files.name == shipped.name
    assert len(three_file_tasks) == TASK_COUNTS[name]
    assert three_file_tasks == shipped_tasks


def _described(experiment: experiment_module.Experiment) -> list:
    """Each task's name, machine, metadata, stop and id, in order."""
    described = []
    for task in experiment.tasks:
        collection = experiment.collection_of(task)
        row = (
            task.name,
            task.machine,
            task.metadata,
            collection,
            task.strong_id(),
        )
        described.append(row)
    return described
