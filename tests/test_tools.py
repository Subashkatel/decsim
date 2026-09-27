"""The repository's own scripts answer for themselves.

tools/check.sh runs three checkers over the tree, and a checker that
misreads its arguments fails open: it exits 0 having looked at nothing,
and the check silently stops holding. slurm/round.sh fails the same way,
by asking for the wrong job or running a task on code nobody can name.
These tests hold each script to what it does with its arguments.
"""

import ast
import importlib.util
import os
import pathlib
import re
import subprocess

import pytest

TESTS_FILE = pathlib.Path(__file__)
TESTS_PATH = TESTS_FILE.resolve()
PACKAGE_ROOT = TESTS_PATH.parent.parent
TOOLS = PACKAGE_ROOT / "tools"
CHECK_SCRIPT = TOOLS / "check.sh"
ROUND_SCRIPT = PACKAGE_ROOT / "slurm" / "round.sh"
# What a submitting shell could hand round.sh: an excuse for a dirty
# tree, a pinned python, a dry run, or the array job the suite itself
# runs in.
UNSET_FOR_THE_ROUND = (
    "ALLOW_DIRTY",
    "DECSIM_PYTHON",
    "DRY_RUN",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_CPUS_PER_TASK",
)
# A round's tasks.csv: two tasks share a shape of job, one has its own.
TASKS_TEXT = (
    "task,cores,memory_mb,hours,estimated_core_hours\n"
    "0,4,16384,24,\n"
    "1,4,6144,24,1.5\n"
    "2,4,16384,24,\n"
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
    print nothing, or fail the way git fails outside a repository.
    """
    answers = {
        "dirty": ('  echo "?? edited.py"', "echo 264853ada3"),
        "clean": ("  :", "echo 264853ada3"),
        "unknown": ("  exit 128", "exit 128"),
    }
    porcelain, revision = answers[status]
    stub = tmp_path / "git"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$3" = "status" ]; then\n'
        f"{porcelain}\n"
        "  exit 0\n"
        "fi\n"
        f"{revision}\n"
    )
    stub.chmod(0o755)


def _round_folder(tmp_path) -> pathlib.Path:
    """An experiment folder holding round 1's tasks.csv."""
    experiment_dir = tmp_path / "experiment"
    round_dir = experiment_dir / "round1"
    round_dir.mkdir(parents=True)
    (round_dir / "tasks.csv").write_text(TASKS_TEXT)
    return experiment_dir


def _run_the_round_script(tmp_path, status, extra_environment):
    """round.sh for round 1, against a stub python and a stub git.

    With DECSIM_PYTHON unset, the script takes the python on PATH, which
    is the stub, as a fresh clone's job does.
    """
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _stub_python(tmp_path, checkout)
    _stub_git(tmp_path, status)
    experiment_dir = _round_folder(tmp_path)
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in UNSET_FOR_THE_ROUND
    }
    environment["SLURM_SUBMIT_DIR"] = str(PACKAGE_ROOT)
    path = environment.get("PATH", "")
    environment["PATH"] = f"{tmp_path}:{path}"
    environment.update(extra_environment)
    completed = subprocess.run(
        ["bash", str(ROUND_SCRIPT), str(experiment_dir), "1"],
        capture_output=True,
        text=True,
        env=environment,
    )
    return completed, experiment_dir


def _sbatch_lines(printed: str) -> list:
    """The sbatch lines a dry run printed, each split into its words."""
    lines = []
    for line in printed.splitlines():
        if line.startswith("sbatch "):
            words = line.split()
            lines.append(words)
    return lines


def _expected_sbatch_line(experiment_dir, tasks: str, memory: str) -> list:
    """The line round.sh submits for one shape of job, word by word."""
    round_dir = experiment_dir / "round1"
    return [
        "sbatch",
        "--job-name",
        "decsim-round1",
        "--array",
        tasks,
        "--nodes",
        "1",
        "--ntasks",
        "1",
        "--cpus-per-task",
        "4",
        "--mem",
        memory,
        "--time",
        "24:00:00",
        "--output",
        f"{round_dir}/%a/log.txt",
        str(ROUND_SCRIPT),
        str(experiment_dir),
        "1",
    ]


def test_a_round_submits_one_array_per_shape_of_job(tmp_path):
    """The referent is tasks.csv: one array per cores, memory and hours.

    A Slurm array has one memory request, so the two tasks of one shape
    share an array and the third has its own. A dry run submits nothing
    and makes no task folder.
    """
    dry_run = {"DRY_RUN": "1"}

    completed, experiment_dir = _run_the_round_script(
        tmp_path, "clean", dry_run
    )

    lines = _sbatch_lines(completed.stdout)
    task_folder = experiment_dir / "round1" / "0"
    assert completed.returncode == 0, completed.stderr
    assert lines == [
        _expected_sbatch_line(experiment_dir, "0,2", "16384M"),
        _expected_sbatch_line(experiment_dir, "1", "6144M"),
    ]
    assert not task_folder.exists()


def test_a_round_task_runs_its_share_of_the_plan(tmp_path):
    """Inside the array the script is the job: one collect --plan task.

    It runs the task the array gives it, one process per core, on the
    python of the job's environment.
    """
    in_the_array = {"SLURM_ARRAY_TASK_ID": "2", "SLURM_CPUS_PER_TASK": "4"}

    completed, experiment_dir = _run_the_round_script(
        tmp_path, "clean", in_the_array
    )

    recorded = tmp_path / "argv.txt"
    recorded_text = recorded.read_text()
    arguments = recorded_text.splitlines()
    plan_path = experiment_dir / "round1" / "plan.csv"
    assert completed.returncode == 0, completed.stderr
    assert "dirty: 0" in completed.stdout
    assert arguments == [
        "-m",
        "decsim",
        "collect",
        "--plan",
        str(plan_path),
        "--task",
        "2",
        "--processes",
        "4",
    ]


@pytest.mark.parametrize(
    ("status", "sentence"),
    [("dirty", "has uncommitted changes"), ("unknown", "git says nothing")],
)
@pytest.mark.parametrize(
    "where", [{"DRY_RUN": "1"}, {"SLURM_ARRAY_TASK_ID": "0"}]
)
def test_a_round_refuses_a_tree_git_does_not_vouch_for(
    tmp_path, status, sentence, where
):
    """A dirty tree, or one git cannot read, is refused where it starts.

    Every task imports the tree as it stands when that task starts, so a
    round launched from a tree still being edited runs code no piece of
    it can name. Submitting refuses it, so no array is queued, and so
    does every task, since the tree may change after submission.
    """
    completed, _experiment_dir = _run_the_round_script(tmp_path, status, where)

    recorded = tmp_path / "argv.txt"
    assert completed.returncode != 0
    assert sentence in completed.stderr
    assert "ALLOW_DIRTY=1" in completed.stderr
    assert _sbatch_lines(completed.stdout) == []
    assert not recorded.exists()
