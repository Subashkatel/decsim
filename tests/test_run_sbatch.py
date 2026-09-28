"""slurm/run.sbatch run by bash, as a Slurm array task runs it.

The task imports the checkout its experiment script sits in, two
folders above experiments/<name>/run.py, wherever the job was submitted
from, and keeps whatever PYTHONPATH the job already had after it.
"""

import os
import pathlib
import subprocess
import sys

_THIS_FILE = pathlib.Path(__file__)
_TEST_FILE = _THIS_FILE.resolve()
REPOSITORY_ROOT = _TEST_FILE.parents[1]
SBATCH_SCRIPT = REPOSITORY_ROOT / "slurm" / "run.sbatch"
# An experiment script that prints what the task handed it.
PROBE_TEXT = """\
import os
import sys

print(os.environ["PYTHONPATH"])
print(" ".join(sys.argv[1:]))
"""


def test_a_task_imports_the_checkout_its_script_sits_in(tmp_path):
    checkout = fake_checkout(tmp_path, has_decsim=True)
    script = checkout / "experiments" / "probe" / "run.py"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    environment = task_environment(tmp_path, "/earlier/path")

    completed = run_the_task(script, elsewhere, environment)

    printed_path, printed_arguments = completed.stdout.splitlines()
    assert completed.returncode == 0
    assert printed_path == f"{checkout}:/earlier/path"
    assert printed_arguments == "3 --workers 16 --out results/folder"


def test_a_script_outside_a_decsim_checkout_is_refused(tmp_path):
    checkout = fake_checkout(tmp_path, has_decsim=False)
    script = checkout / "experiments" / "probe" / "run.py"
    environment = task_environment(tmp_path, "")

    completed = run_the_task(script, tmp_path, environment)

    assert completed.returncode == 1
    assert completed.stderr == (
        f"run.sbatch: {checkout} is not a decsim checkout; give the path "
        "of experiments/<name>/run.py inside one\n"
    )


def fake_checkout(tmp_path: pathlib.Path, *, has_decsim: bool) -> pathlib.Path:
    """A folder shaped like a checkout, with a probe experiment in it."""
    checkout = tmp_path / "checkout"
    experiment_folder = checkout / "experiments" / "probe"
    experiment_folder.mkdir(parents=True)
    script = experiment_folder / "run.py"
    script.write_text(PROBE_TEXT)
    if has_decsim:
        package = checkout / "decsim"
        package.mkdir()
        runner = package / "experiment_runner.py"
        runner.write_text("")
    return checkout


def task_environment(tmp_path: pathlib.Path, earlier_path: str) -> dict:
    """What Slurm hands an array task, python first on PATH."""
    python_path = pathlib.Path(sys.executable)
    python_folder = python_path.parent
    search_path = f"{python_folder}:{os.environ['PATH']}"
    return dict(
        os.environ,
        PATH=search_path,
        PYTHONPATH=earlier_path,
        SLURM_ARRAY_TASK_ID="3",
        SLURM_CPUS_PER_TASK="16",
        SLURM_SUBMIT_DIR=str(tmp_path),
    )


def run_the_task(
    script: pathlib.Path, working_folder: pathlib.Path, environment: dict
) -> subprocess.CompletedProcess:
    """The task run by bash on one script, from working_folder."""
    command = ["bash", str(SBATCH_SCRIPT), str(script), "results/folder"]
    return subprocess.run(
        command,
        cwd=working_folder,
        env=environment,
        capture_output=True,
        text=True,
    )
