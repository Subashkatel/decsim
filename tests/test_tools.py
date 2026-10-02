"""The repository's own scripts answer for themselves.

tools/check.sh runs three checkers over the tree, and a checker that
misreads its arguments fails open: it exits 0 having looked at nothing,
and the check silently stops holding. `decsim run --slurm` fails the
same way, by asking for the wrong job or running a task on code nobody
can name, so it is held here beside them, against a stub git, squeue
and sbatch. These tests hold each to what it does with its arguments.
"""

import ast
import importlib.util
import os
import pathlib
import re
import shlex
import subprocess
import sys

import pytest

import decsim.experiments.experiment as experiment
import tests.experiments.yaml_configs as yaml_configs

TESTS_FILE = pathlib.Path(__file__)
TESTS_PATH = TESTS_FILE.resolve()
PACKAGE_ROOT = TESTS_PATH.parent.parent
TOOLS = PACKAGE_ROOT / "tools"
CHECK_SCRIPT = TOOLS / "check.sh"
# What a submitting shell could hand `decsim run --slurm`: an excuse for
# a dirty tree, a pinned python, a limit, a tree reading, or the job the
# suite itself runs in.
UNSET_FOR_SLURM = (
    "ALLOW_DIRTY",
    "DECSIM_PYTHON",
    "DECSIM_TREE_DIRTY",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_ID",
    "STUB_QUEUED_JOBS",
    "SUBMIT_LIMIT",
)


def _tool(name: str):
    """One checker, imported from the tools folder by path."""
    path = TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_uses_graph_check_says_what_it_wants_with_no_argument(capsys):
    """No argument is a usage message and a refusal, not a traceback."""
    tool = _tool("check_uses_graph")
    exit_code = tool.main([])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "usage: check_uses_graph.py" in captured.err


def test_the_uses_graph_check_refuses_a_path_that_is_not_a_directory(capsys):
    """A mistyped root is named, so the reader sees which argument was wrong."""
    tool = _tool("check_uses_graph")
    exit_code = tool.main(["decsim/machine.py"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "is not a directory" in captured.err


def test_the_uses_graph_check_reports_the_levels_of_the_package(capsys):
    """With the package it prints the partial order and passes."""
    tool = _tool("check_uses_graph")
    root = PACKAGE_ROOT / "decsim"
    exit_code = tool.main([str(root)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "0 cycles" in captured.out
    assert "level 0:" in captured.out


def test_the_uses_graph_puts_a_package_importing_none_at_level_zero(
    tmp_path,
):
    tool = _tool("check_uses_graph")
    root_module = tmp_path / "__init__.py"
    root_module.write_text('"""The package."""\n')
    leaf = tmp_path / "leaf.py"
    leaf.write_text('"""Imports nothing of the package."""\n')

    edges = tool.read_edges(tmp_path)
    nodes = tool.nodes_of(edges)

    assert tool.levels_of(edges, nodes) == {"leaf": 0}


def test_the_recognition_check_passes_on_the_tree(capsys):
    """The tree recognises no class off the tool's list."""
    tool = _tool("check_row_recognition")
    root = PACKAGE_ROOT / "decsim"
    exit_code = tool.main([str(root)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "0 unlisted class tests" in captured.out


def test_the_recognition_check_catches_a_row_chosen_by_its_class(
    tmp_path, capsys
):
    """Both shapes of the rule 10 defect fail, by file and line."""
    tool = _tool("check_row_recognition")
    module = tmp_path / "build_something.py"
    module.write_text(
        "def build(row):\n"
        "    if row is PyMatchingDecoder:\n"
        "        return row(1)\n"
        "    if isinstance(row, BeliefMatchingDecoder):\n"
        "        return row(2, 3)\n"
        "    return row()\n"
    )
    exit_code = tool.main([str(tmp_path)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "build_something.py:2: recognises the class PyMatchingDecoder" in (
        captured.out
    )
    assert "build_something.py:4" in captured.out
    assert "BeliefMatchingDecoder" in captured.out


def test_the_recognition_check_reads_an_enum_member_as_a_value(
    tmp_path, capsys
):
    """A member is a value, so comparing one is not a class test."""
    tool = _tool("check_row_recognition")
    module = tmp_path / "reads_a_member.py"
    module.write_text(
        "def is_strong(tier):\n"
        "    return tier is window_records.DecoderTier.STRONG\n"
    )
    exit_code = tool.main([str(tmp_path)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "0 unlisted class tests" in captured.out


def test_the_one_action_check_fails_a_call_or_sum_passed_as_an_argument(
    tmp_path, capsys
):
    """Rule 1's argument check fails the run, by file and line."""
    tool = _tool("check_one_action")
    module = tmp_path / "busy_arguments.py"
    module.write_text(
        "def nested(x):\n"
        "    return g(h(x))\n"
        "\n"
        "\n"
        "def summed(x):\n"
        "    return g(x + 1)\n"
    )
    exit_code = tool.main([str(tmp_path)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "busy_arguments.py:2: nested call" in captured.out
    assert "busy_arguments.py:6: busy argument" in captured.out


def test_a_checkout_under_a_folder_named_tmp_is_still_checked(tmp_path, capsys):
    """Only the part below the target names a skipped folder.

    A clone under /tmp, as on a CI runner, is checked, and a folder
    named tmp inside the target is skipped.
    """
    tool = _tool("check_one_action")
    checkout = tmp_path / "tmp" / "checkout"
    scratch = checkout / "tmp"
    scratch.mkdir(parents=True)
    busy = "def nested(x):\n    return g(h(x))\n"
    (checkout / "checked.py").write_text(busy)
    (scratch / "skipped.py").write_text(busy)
    exit_code = tool.main([str(checkout)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "checked.py:2: nested call" in captured.out
    assert "skipped.py" not in captured.out


def _classes_tested_against(tool, root) -> set:
    tested = set()
    for path in tool.source_paths(root):
        text = path.read_text()
        tree = ast.parse(text)
        for _line, name in tool.tested_classes(tree):
            tested.add(name)
    return tested


def test_every_class_on_the_tools_list_is_still_tested_against_somewhere():
    """The list stays honest: a name nobody tests against is deleted.

    An allow list is only as good as its shortness, and a stale name on
    it silences a real finding, so the list is held against the tree the
    same way STYLE.md's wide-state exemptions are.
    """
    tool = _tool("check_row_recognition")
    root = PACKAGE_ROOT / "decsim"
    tested = _classes_tested_against(tool, root)
    stale = tool.ALLOWED - tested
    assert stale == set()


def _stub_python(tmp_path, checkout=None):
    """An interpreter that writes down the command line it was given.

    It answers the runner's question about where decsim was imported
    from with the checkout it is given, so the test never runs a sweep
    and chooses which tree the runner reads.
    """
    if checkout is None:
        checkout = tmp_path
    recorded = tmp_path / "argv.txt"
    stub = tmp_path / "python"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "-c" ]; then\n'
        f'  echo "{checkout}"\n'
        "  exit 0\n"
        "fi\n"
        f'printf "%s\\n" "$@" > "{recorded}"\n'
    )
    stub.chmod(0o755)
    return stub, recorded


def test_the_check_script_runs_the_active_environments_python(tmp_path):
    """With DECSIM_PYTHON unset, check.sh runs the python on PATH.

    A fresh clone has no .venv of its own, so the default names none.
    """
    _stub, recorded = _stub_python(tmp_path)
    environment = dict(os.environ)
    environment.pop("DECSIM_PYTHON", None)
    environment.pop("DECSIM_PYDEPS", None)
    path = environment.get("PATH", "")
    environment["PATH"] = f"{tmp_path}:{path}"

    completed = subprocess.run(
        ["bash", str(CHECK_SCRIPT), "tools/check.sh"],
        capture_output=True,
        text=True,
        env=environment,
    )

    recorded_text = recorded.read_text()
    assert completed.returncode == 0, completed.stderr
    assert "ruff" in recorded_text


def _page_and_its_own_output(check, page: str):
    """A page, and outputs that print exactly what its blocks show."""
    path = PACKAGE_ROOT / page
    text = path.read_text()
    outputs = []
    for block in check.fenced_blocks(text):
        printed = check.Printed(command="decsim", lines=tuple(block.lines))
        outputs.append(printed)
    return text, outputs


def test_the_tutorial_check_runs_the_pages_own_commands():
    """decsim, cut and ls lines, a continued line joined, nothing else."""
    check = _tool("check_tutorial_runs")
    text = (
        "```bash\n"
        'python -m pip install -e ".[run]"\n'
        "decsim trace follow \\\n"
        "  results/shot/trace/one.trace.json \\\n"
        "  --round 1:1\n"
        "cut -d, -f1 results/shot/sweep.csv\n"
        "```\n"
    )

    commands = check.page_commands(text)

    assert len(commands) == 2
    assert commands[0].split() == [
        "decsim",
        "trace",
        "follow",
        "results/shot/trace/one.trace.json",
        "--round",
        "1:1",
    ]
    assert commands[1] == "cut -d, -f1 results/shot/sweep.csv"


@pytest.mark.parametrize(
    "shown_line, moved_line",
    [
        (r"^(queue wait, mean: )[0-9.]+", r"\g<1>999.000"),
        (r"^27\.224 ", "27.225 "),
    ],
    ids=["summary timing", "trace tick"],
)
def test_a_priced_tutorial_fails_the_check_when_a_value_moves(
    shown_line, moved_line
):
    """On two_tiers every tick is priced, so every line is held."""
    check = _tool("check_tutorial_runs")
    tutorial = check.TUTORIALS[2]
    text, outputs = _page_and_its_own_output(check, tutorial.page)
    moved = re.sub(shown_line, moved_line, text, count=1, flags=re.MULTILINE)

    unmoved_differences = check.page_differences(tutorial, text, outputs)
    moved_differences = check.page_differences(tutorial, moved, outputs)

    assert tutorial.is_priced
    assert moved != text
    assert unmoved_differences == []
    assert len(moved_differences) == 1


def test_a_wall_clock_tutorial_holds_its_counts_and_not_its_timings():
    """first_run's load is the host's; its failure count is the seed's."""
    check = _tool("check_tutorial_runs")
    tutorial = check.TUTORIALS[0]
    text, outputs = _page_and_its_own_output(check, tutorial.page)
    new_load = re.sub(
        r"^(load .*: )[0-9.]+", r"\g<1>99.99", text, count=1, flags=re.MULTILINE
    )
    new_count = text.replace(
        "logical failures: 0 of 2 scored shots",
        "logical failures: 1 of 2 scored shots",
    )

    load_differences = check.page_differences(tutorial, new_load, outputs)
    count_differences = check.page_differences(tutorial, new_count, outputs)

    assert not tutorial.is_priced
    assert new_load != text
    assert load_differences == []
    assert len(count_differences) == 1


def _stub_git(tmp_path, status):
    """A git that answers about the checkout without one existing.

    `status` is what `git status --porcelain` does: print a changed file,
    print nothing, or fail the way git fails outside a repository. Every
    other question gets nothing, so the run records hold no patch.
    """
    answers = {
        "dirty": ('echo "?? edited.py"', "echo 264853ada3"),
        "clean": (":", "echo 264853ada3"),
        "unknown": ("exit 128", "exit 128"),
    }
    porcelain, revision = answers[status]
    stub = tmp_path / "git"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'case "$3" in\n'
        f"  status) {porcelain} ;;\n"
        f"  rev-parse) {revision} ;;\n"
        "esac\n"
    )
    stub.chmod(0o755)


def _stub_squeue(tmp_path):
    """A squeue that lists $STUB_QUEUED_JOBS of the user's jobs, one a line.

    With STUB_SQUEUE_FAILS set it fails as a squeue that cannot reach
    the controller does.
    """
    stub = tmp_path / "squeue"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [ -n "$STUB_SQUEUE_FAILS" ]; then\n'
        '  echo "slurm_load_jobs error: Unable to contact controller" >&2\n'
        "  exit 1\n"
        "fi\n"
        "for ((job = 0; job < ${STUB_QUEUED_JOBS:-0}; job++)); do\n"
        '  echo "$job"\n'
        "done\n"
    )
    stub.chmod(0o755)


def _stub_sbatch(tmp_path):
    """An sbatch that records each submission and answers a job id.

    The ids count up from 1000 in submission order, so no test reaches
    Slurm and a dependency names the arrays it follows.
    """
    stub = tmp_path / "sbatch"
    submissions = tmp_path / "submissions.txt"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> {submissions}\n'
        f"wc -l < {submissions} | awk '{{print 999 + $1}}'\n"
    )
    stub.chmod(0o755)


def _stub_check_python(tmp_path, exit_code: int):
    """The jobs' interpreter, which records each check shot it is asked for.

    A failing one says why on stderr, as a refused decsim run does.
    """
    checks = tmp_path / "checks.txt"
    stub = tmp_path / "job_python"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> {checks}\n'
        f'if [ "{exit_code}" != 0 ]; then\n'
        '  echo "decsim: the build refused the point" >&2\n'
        "fi\n"
        f"exit {exit_code}\n"
    )
    stub.chmod(0o755)
    return stub


def _slurm_environment(tmp_path, status, extra=None, check_exit_code=0):
    """The submitting shell: stub git, squeue, sbatch and jobs' python."""
    _stub_git(tmp_path, status)
    _stub_squeue(tmp_path)
    _stub_sbatch(tmp_path)
    job_python = _stub_check_python(tmp_path, check_exit_code)
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in UNSET_FOR_SLURM
    }
    path = environment.get("PATH", "")
    environment["PATH"] = f"{tmp_path}:{path}"
    environment["PYTHONPATH"] = str(PACKAGE_ROOT)
    environment["DECSIM_PYTHON"] = str(job_python)
    if extra is not None:
        environment.update(extra)
    return environment


def _decsim(tmp_path, arguments: list, environment):
    """One decsim command line, run as the submitting shell runs it."""
    command = [sys.executable, "-m", "decsim", *arguments]
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
    )


def _two_point_config(tmp_path) -> tuple:
    """A yaml sweeping two distances, and its point names in order."""
    card = {
        "sweep": [
            {
                "axes": {
                    "workload.arguments.physical_error_probability": [0.001],
                    "qpu.distance": [3, 5],
                    "qpu.round_period_microseconds": [1.0],
                },
                "collection": {"max_shots": 4},
            }
        ]
    }
    config_path = yaml_configs.write_config(tmp_path, card)
    study = experiment.load(config_path)
    names = [point.name for point in study.points]
    return config_path, names


def _sbatch_lines(printed: str) -> list:
    """The sbatch lines a dry run printed, each split into its words."""
    lines = []
    for line in printed.splitlines():
        if line.startswith("sbatch "):
            words = shlex.split(line)
            lines.append(words)
    return lines


def _value_after(words: list, flag: str) -> str:
    """The word after a flag of one sbatch line."""
    position = words.index(flag)
    return words[position + 1]


def _slurm_arguments(config_path, results_dir, *extra) -> list:
    """`decsim run <yaml> --slurm`, into results_dir."""
    return [
        "run",
        str(config_path),
        "--slurm",
        "--out",
        str(results_dir),
        *extra,
    ]


def test_a_slurm_dry_run_checks_plans_and_prints_its_jobs(tmp_path):
    """The first step checks one shot a shape, plans batch 1, prints.

    Both points are one shape, so the first point's seed 0 is the one
    check, run in the jobs' interpreter. The batch's tasks share a shape
    of job, so one array runs them, each task its share of the plan; the
    next step waits on that array with afterany. A dry run submits
    nothing and makes no task folder.
    """
    config_path, names = _two_point_config(tmp_path)
    results_dir = tmp_path / "results"
    environment = _slurm_environment(tmp_path, "clean")
    arguments = _slurm_arguments(
        config_path, results_dir, "--tasks", "3", "--dry-run"
    )

    completed = _decsim(tmp_path, arguments, environment)

    checks_path = tmp_path / "checks.txt"
    checks_text = checks_path.read_text()
    checks = checks_text.splitlines()
    array_line, next_line = _sbatch_lines(completed.stdout)
    batch_folder = results_dir / "batches" / "1"
    assert completed.returncode == 0, completed.stderr
    assert len(checks) == 1
    assert f"--only {names[0]} --seed 0" in checks[0]
    assert (batch_folder / "plan.csv").exists()
    assert _value_after(array_line, "--array") == "0,1"
    assert _value_after(array_line, "--mem") == "16384M"
    assert _value_after(array_line, "--time") == "24:00:00"
    task_command = _value_after(array_line, "--wrap")
    assert task_command.endswith(
        f"--out {results_dir} --batch 1 --task $SLURM_ARRAY_TASK_ID "
        "--processes $SLURM_CPUS_PER_TASK"
    )
    assert _value_after(next_line, "--dependency") == (
        "afterany:<array job id>"
    )
    step_command = _value_after(next_line, "--wrap")
    assert step_command.endswith(
        f"--slurm --out {results_dir} --tasks 3 --cores 4 --hours 24 "
        "--memory-mb 4096"
    )
    assert not (tmp_path / "submissions.txt").exists()
    assert not (batch_folder / "0").exists()


def test_a_batch_submits_one_array_per_shape_of_job(tmp_path):
    """A point with a measured peak asks for its own memory.

    A Slurm array has one memory request, so the task of the point a
    local run measured and the task of the point nothing measured go in
    two arrays.
    """
    config_path, names = _two_point_config(tmp_path)
    results_dir = tmp_path / "results"
    environment = _slurm_environment(tmp_path, "clean")
    measured = ["run", str(config_path), "--only", names[0], "--shots", "1"]
    _decsim(tmp_path, [*measured, "--out", str(results_dir)], environment)
    arguments = _slurm_arguments(
        config_path, results_dir, "--tasks", "2", "--dry-run"
    )

    completed = _decsim(tmp_path, arguments, environment)

    *array_lines, _next_line = _sbatch_lines(completed.stdout)
    memories = [_value_after(line, "--mem") for line in array_lines]
    tasks = [_value_after(line, "--array") for line in array_lines]
    assert completed.returncode == 0, completed.stderr
    assert len(array_lines) == 2
    assert sorted(tasks) == ["0", "1"]
    assert "16384M" in memories
    assert len(set(memories)) == 2


def test_a_launch_submits_the_arrays_then_the_next_step_behind_them(tmp_path):
    """The next step's dependency names the job ids sbatch answered."""
    config_path, _names = _two_point_config(tmp_path)
    results_dir = tmp_path / "results"
    environment = _slurm_environment(tmp_path, "clean")
    arguments = _slurm_arguments(config_path, results_dir, "--tasks", "2")

    completed = _decsim(tmp_path, arguments, environment)

    submissions_path = tmp_path / "submissions.txt"
    submissions_text = submissions_path.read_text()
    array_text, next_text = submissions_text.splitlines()
    batch_folder = results_dir / "batches" / "1"
    assert completed.returncode == 0, completed.stderr
    assert "--array 0,1" in array_text
    assert "--dependency afterany:1000" in next_text
    assert (batch_folder / "0").is_dir()
    assert (batch_folder / "1").is_dir()


@pytest.mark.parametrize(
    ("status", "sentence"),
    [("dirty", "has uncommitted changes"), ("unknown", "git says nothing")],
)
@pytest.mark.parametrize("where", ["launch", "task"])
def test_a_tree_git_does_not_vouch_for_is_refused_where_it_starts(
    tmp_path, status, sentence, where
):
    """A dirty tree, or one git cannot read, is refused where it starts.

    Every task imports the tree as it stands when that task starts, so a
    batch launched from a tree still being edited runs code no piece of
    it can name. Launching refuses it, so no array is queued, and so
    does every task, since the tree may change after submission.
    """
    config_path, _names = _two_point_config(tmp_path)
    results_dir = tmp_path / "results"
    environment = _slurm_environment(tmp_path, status)
    arguments = {
        "launch": _slurm_arguments(config_path, results_dir, "--tasks", "1"),
        "task": ["run", "--out", str(results_dir), "--batch", "1"]
        + ["--task", "0"],
    }

    completed = _decsim(tmp_path, arguments[where], environment)

    assert completed.returncode == 1
    assert sentence in completed.stderr
    assert "ALLOW_DIRTY=1" in completed.stderr
    assert not (tmp_path / "submissions.txt").exists()
    assert not (tmp_path / "checks.txt").exists()
    assert not results_dir.exists()


@pytest.mark.parametrize("dry_run", [["--dry-run"], []])
def test_a_batch_past_the_submit_limit_is_refused_before_any_array(
    tmp_path, dry_run
):
    """Its two tasks and two queued jobs pass a limit of three.

    The limit counts each array task as a job, so a batch past it would
    be half submitted; it is refused before the first array, dry run or
    not, with a sentence that says to run with fewer tasks.
    """
    config_path, _names = _two_point_config(tmp_path)
    results_dir = tmp_path / "results"
    over_the_limit = {"SUBMIT_LIMIT": "3", "STUB_QUEUED_JOBS": "2"}
    environment = _slurm_environment(tmp_path, "clean", over_the_limit)
    arguments = _slurm_arguments(
        config_path, results_dir, "--tasks", "2", *dry_run
    )

    completed = _decsim(tmp_path, arguments, environment)

    assert completed.returncode == 1
    assert "fewer --tasks" in completed.stderr
    assert _sbatch_lines(completed.stdout) == []
    assert not (tmp_path / "submissions.txt").exists()
    assert not (results_dir / "batches" / "1" / "0").exists()


@pytest.mark.parametrize("limit", ["mistyped", "0", "-3", "1.5"])
def test_a_submit_limit_that_is_no_positive_whole_number_is_refused(
    tmp_path, limit
):
    """A limit the launcher cannot compare would let any batch through."""
    config_path, _names = _two_point_config(tmp_path)
    results_dir = tmp_path / "results"
    malformed = {"SUBMIT_LIMIT": limit}
    environment = _slurm_environment(tmp_path, "clean", malformed)
    arguments = _slurm_arguments(config_path, results_dir, "--tasks", "1")

    completed = _decsim(tmp_path, arguments, environment)

    assert completed.returncode == 1
    assert "SUBMIT_LIMIT must be a whole number" in completed.stderr
    assert not (tmp_path / "submissions.txt").exists()


def test_a_squeue_that_fails_refuses_the_submission(tmp_path):
    """A queue that cannot be read is not an empty queue.

    Counted as empty, a batch could pass the limit and be half
    submitted; only a node with no squeue at all counts none.
    """
    config_path, _names = _two_point_config(tmp_path)
    results_dir = tmp_path / "results"
    failing = {"STUB_SQUEUE_FAILS": "1"}
    environment = _slurm_environment(tmp_path, "clean", failing)
    arguments = _slurm_arguments(config_path, results_dir, "--tasks", "1")

    completed = _decsim(tmp_path, arguments, environment)

    assert completed.returncode == 1
    assert "squeue exited 1" in completed.stderr
    assert "Unable to contact controller" in completed.stderr
    assert not (tmp_path / "submissions.txt").exists()


def test_a_failed_one_shot_check_stops_the_launch_before_any_plan(tmp_path):
    """A shot the jobs' interpreter cannot run queues nothing.

    The check runs before the first batch is planned, so a broken import,
    build or config costs one shot and no array.
    """
    config_path, names = _two_point_config(tmp_path)
    results_dir = tmp_path / "results"
    environment = _slurm_environment(tmp_path, "clean", check_exit_code=1)
    arguments = _slurm_arguments(config_path, results_dir, "--tasks", "1")

    completed = _decsim(tmp_path, arguments, environment)

    assert completed.returncode == 1
    assert f"the one-shot check of point {names[0]} failed" in (
        completed.stderr
    )
    assert "the build refused the point" in completed.stderr
    assert not (results_dir / "batches").exists()
    assert not (tmp_path / "submissions.txt").exists()
