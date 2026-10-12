"""`decsim check` (experiments/check_command.py).

Each task's machine is built once and no shot runs; a task its build
refuses is listed with the reason, and the run is refused after the
others are built.
"""

import pytest

import decsim.experiments.command as command
import decsim.parameters.assignments as assignments
import tests.experiments.run_files as run_files

UNWRITTEN_LOOK_AHEAD_RUN_FILE = """
import dataclasses

import decsim
import decsim.presets.machines as machines
import decsim.producers as producers
import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.syndrome_devices as syndrome_devices

base = decsim.Task("base", machines.weak_baseline(3, 0.002, 1.0)).machine
card = code_geometry.BivariateBicycleCodeModel.Settings()
source = syndrome_devices.TimingOnlyDevice.Settings()
qpu = dataclasses.replace(
    base.qpu, code_card=card, distance=None, source=source
)
memory = producers.MemorySettings(
    0.002, rounds=24, code_task=None, decoded=False
)
machine = dataclasses.replace(base, qpu=qpu, workload=memory)
stop = decsim.CollectionSettings(max_shots=1)
experiment = decsim.Experiment(
    "look_ahead", [decsim.Task("gross", machine, collection=stop)]
)
"""


@pytest.fixture(autouse=True)
def _no_assignments_left_behind(monkeypatch):
    """Each test starts with DECSIM_SET empty."""
    monkeypatch.setenv(assignments.ENVIRONMENT_VARIABLE, "")


def test_check_builds_every_task_and_prints_each(tmp_path, capsys):
    run_file = run_files.write_run_file(tmp_path)

    command.main(["check", str(run_file)])

    printed = capsys.readouterr()
    lines = printed.out.splitlines()
    assert lines
    assert all(": built (" in line for line in lines)


def test_check_lists_a_task_its_build_refuses_and_exits_one(tmp_path, capsys):
    run_file = tmp_path / "look_ahead.py"
    run_file.write_text(UNWRITTEN_LOOK_AHEAD_RUN_FILE)

    with pytest.raises(SystemExit) as exited:
        command.main(["check", str(run_file)])

    printed = capsys.readouterr()
    assert exited.value.code == 1
    assert "gross: refused: the bivariate-bicycle card" in printed.out
    assert "1 of 1 tasks were refused" in printed.err
