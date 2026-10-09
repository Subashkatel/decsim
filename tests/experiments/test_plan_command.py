"""`decsim run --slurm`'s packing, against Graham's bound for longest first.

Graham ("Bounds on multiprocessing timing anomalies", SIAM J. Appl.
Math. 17, 1969) bounds a core packed longest first to the mean plus one
piece, so the cores of a packed plan differ by at most the longest
piece, and the walltime law is the busiest core over the 0.8 use
target, in whole minutes, at least 61 and at most the limit.
"""

import json
import shlex
import sys

import pytest

import decsim.experiments.collect_command as collect_command
import decsim.experiments.command as command
import decsim.experiments.plan_command as plan_command
import decsim.experiments.run_folder as run_folder
import tests.experiments.run_files as run_files

# four tasks of five shots, each estimated at 280 core seconds: a
# twentieth of a two-hour job's core budget is 288 s, so a piece is one
# shot, and twenty pieces overfill one single-core job
PACKED_TASKS = {
    "axes": run_files.FOUR_TASK_AXES,
    "collection": {"max_shots": 5, "core_seconds_per_shot": 280.0},
}


@pytest.fixture(autouse=True)
def one_tree_reading_per_test(monkeypatch):
    """A launch runs on the tree as it stands; tests/test_tools.py holds git."""
    monkeypatch.setenv(run_folder.ALLOW_DIRTY_VARIABLE, "1")
    monkeypatch.delenv(run_folder.TREE_DIRTY_VARIABLE, raising=False)
    monkeypatch.delenv(run_folder.TREE_PATCH_VARIABLE, raising=False)
    run_folder._tree_reading.cache_clear()
    yield
    run_folder._tree_reading.cache_clear()


def test_cores_are_packed_longest_first_and_the_walltime_is_the_busiest():
    """Seven pieces on three-core jobs of two hours, by hand.

    Two hours give a core 5760 s at 0.8; less the longest piece, 2760 s
    of room, so 11500 s of work needs two jobs of three cores. Longest
    first deals 3000, 2500, 2000, 1500, 1000 and 1000 to the six cores and
    500 to the first of the two least loaded. Every job asks the busiest
    core's 3000 s over 0.8, 62.5 minutes, rounded up; the quieter job's
    1875 s alone would ask the 61-minute floor.
    """
    job = plan_command.JobShape(cores=3, hours=2, memory_mb=1024)
    seconds = (3000.0, 2500.0, 2000.0, 1500.0, 1000.0, 1000.0, 500.0)
    planned = _pieces_of(seconds)

    jobs = plan_command.pack(planned, job)

    busy_job, quiet_job = jobs
    assert _core_seconds(busy_job) == [3000.0, 2500.0, 2000.0]
    assert _core_seconds(quiet_job) == [1500.0, 1500.0, 1000.0]
    assert plan_command.walltime_minutes(jobs, job.hours) == 63
    assert plan_command.walltime_minutes([quiet_job], job.hours) == 61


def test_pieces_saved_since_the_launch_leave_the_next_plan(tmp_path):
    """A relaunch re-packs only the seeds no piece holds yet."""
    run_file = run_files.write_run_file(tmp_path, **PACKED_TASKS)
    out_dir = tmp_path / "out"
    arguments = [str(run_file), "--out", str(out_dir), "--slurm"]
    slurm_arguments = [*arguments, "--cores", "1", "--hours", "2"]
    command.main(["run", *slurm_arguments, "--dry-run"])
    first_plan = _planned_seeds(out_dir)
    saved_name, saved_first, saved_count = first_plan[0]
    collect_command.run_pieces(
        run_file, out_dir, [[saved_name, saved_first, saved_count]]
    )

    command.main(["run", *slurm_arguments, "--dry-run"])

    second_plan = _planned_seeds(out_dir)
    assert len(first_plan) == 20
    assert sorted(second_plan) == sorted(first_plan[1:])


def test_a_relaunch_with_every_piece_saved_plans_no_job(tmp_path, capsys):
    run_file = run_files.write_run_file(
        tmp_path, collection={"max_shots": 1, "core_seconds_per_shot": 1.0}
    )
    out_dir = tmp_path / "out"
    launch = ["run", str(run_file), "--out", str(out_dir), "--slurm"]
    command.main([*launch, "--dry-run"])
    planned = _planned_seeds(out_dir)
    collect_command.run_pieces(run_file, out_dir, planned)
    capsys.readouterr()

    command.main([*launch, "--dry-run"])

    printed = capsys.readouterr()
    assert f"every piece is saved; fold them with --fold --out {out_dir}" in (
        printed.out
    )


@pytest.mark.parametrize(
    ("arguments", "sentence"),
    [
        (
            {"collection": {"max_failures": 1, "max_shots": 5}},
            "stops on failures or time, so its work is unknown before "
            "launch and --slurm cannot pack it",
        ),
        (
            {"collection": {"max_shots": 5}},
            "has no core_seconds_per_shot, so --slurm cannot size its work",
        ),
        (
            {"collection": {"max_shots": 5, "core_seconds_per_shot": 300.0}},
            "is estimated at 300.0 core seconds, more than a slice of a "
            "core's 5760 at --hours 2",
        ),
        (
            {"machine": "switching", "machine_arguments": {"online": {}}},
            "calibrates its threshold online, so its pieces run one after "
            "another and --slurm cannot pack them",
        ),
    ],
)
def test_a_task_the_launch_cannot_pack_is_refused_before_any_folder(
    tmp_path, capsys, arguments, sentence
):
    run_file = run_files.write_run_file(tmp_path, **arguments)
    out_dir = tmp_path / "out"
    slurm = ["--slurm", "--hours", "2", "--dry-run"]

    with pytest.raises(SystemExit):
        command.main(["run", str(run_file), "--out", str(out_dir), *slurm])

    printed = capsys.readouterr()
    assert sentence in printed.err
    assert not out_dir.exists()


def test_a_dry_run_writes_one_array_and_a_fold_behind_it(tmp_path, capsys):
    """Two jobs, one array, one walltime.

    Twenty pieces of 280 s overfill one single-core job of two hours,
    so two jobs share them, ten each: 2800 s over 0.8 is under the
    61-minute floor, so both ask 1:01:00. Each line reads back as the
    arguments it was written from.
    """
    run_file = run_files.write_run_file(tmp_path, **PACKED_TASKS)
    out_dir = tmp_path / "run folder"
    slurm = ["--slurm", "--cores", "1", "--hours", "2", "--dry-run"]

    command.main(["run", str(run_file), "--out", str(out_dir), *slurm])

    run_script = out_dir / plan_command.RUN_SCRIPT
    run_text = run_script.read_text()
    run_lines = run_text.splitlines()
    fold_script = out_dir / plan_command.FOLD_SCRIPT
    fold_text = fold_script.read_text()
    fold_lines = fold_text.splitlines()
    printed = capsys.readouterr()
    decsim_run = [sys.executable, "-m", "decsim", "run"]
    assert run_lines[1] == (
        "#SBATCH --array=0-1 --cpus-per-task=1 --mem=16384M --time=1:01:00"
    )
    assert shlex.split(run_lines[2]) == [
        "#SBATCH",
        f"--output={out_dir}/logs/%A_%a.log",
    ]
    assert shlex.split(run_lines[3]) == [
        *decsim_run,
        str(run_file),
        "--out",
        str(out_dir),
        "--job",
        "$SLURM_ARRAY_TASK_ID",
        "--processes",
        "1",
    ]
    assert shlex.split(fold_lines[3]) == [
        *decsim_run,
        "--fold",
        "--out",
        str(out_dir),
    ]
    assert "every job asks 1:01:00" in printed.out
    assert "  0      1   77%      10" in printed.out
    assert "  1      1   77%      10" in printed.out


def _planned_seeds(run_dir) -> list:
    """Every piece jobs.json plans, as [task name, first seed, count]."""
    jobs_path = run_dir / plan_command.JOBS_FILE
    plan_text = jobs_path.read_text()
    plan = json.loads(plan_text)
    return [piece for job_pieces in plan["jobs"] for piece in job_pieces]


def _pieces_of(seconds: tuple) -> list:
    """One one-shot piece of its own seed per estimate."""
    return [
        plan_command.PlannedPiece("task", seed, 1, estimate)
        for seed, estimate in enumerate(seconds)
    ]


def _core_seconds(packed: plan_command.PackedJob) -> list:
    """Each core's estimated work, in the job's order."""
    return [
        sum(piece.estimated_seconds for piece in lane) for lane in packed.lanes
    ]
