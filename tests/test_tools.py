"""The repository's own scripts answer for themselves.

tools/check.sh runs three checkers over the tree, and a checker that
misreads its arguments fails open: it exits 0 having looked at nothing,
and the check silently stops holding. slurm/slurm_run.sh fails the same
way, by computing a shard of the wrong count and writing a folder that
holds the wrong share of the sweep. These tests hold each script to what
it does with its arguments.
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
SLURM_RUNNER = PACKAGE_ROOT / "slurm" / "slurm_run.sh"
EXPERIMENT_RUNNER = PACKAGE_ROOT / "slurm" / "experiment_run.sh"
CHECK_SCRIPT = TOOLS / "check.sh"
# What a submitting shell could hand the runner: an excuse for a dirty
# tree, a pinned python, or the array job the suite itself runs in.
UNSET_FOR_THE_RUNNER = (
    "ALLOW_DIRTY",
    "DECSIM_PYTHON",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_ARRAY_TASK_COUNT",
    "RUN",
    "SHARDS",
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


def _run_the_runner(tmp_path, task_id, task_count, shards=None):
    """One array task of the runner, against a stub interpreter."""
    stub, recorded = _stub_python(tmp_path)
    environment = dict(os.environ)
    environment["SLURM_SUBMIT_DIR"] = str(PACKAGE_ROOT)
    environment["SLURM_ARRAY_TASK_ID"] = str(task_id)
    environment["SLURM_ARRAY_TASK_COUNT"] = str(task_count)
    run_dir = tmp_path / "run"
    environment["RUN"] = str(run_dir)
    environment["DECSIM_PYTHON"] = str(stub)
    # the stub reports a tree git says nothing about, which the runner
    # refuses; the refusal has its own tests and is not what this one
    # reads
    environment["ALLOW_DIRTY"] = "1"
    if shards is not None:
        environment["SHARDS"] = str(shards)
    completed = subprocess.run(
        ["bash", str(SLURM_RUNNER), "configs/weak_ler.yaml"],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    recorded_text = recorded.read_text()
    arguments = recorded_text.splitlines()
    return completed.stdout, arguments


def test_the_slurm_runner_shards_by_the_count_it_is_given(tmp_path):
    """A partial re-run of a 500 shard array is one sbatch line.

    Without SHARDS the count would be the array's own, so re-running
    tasks 447 to 499 alone would compute shard 447 of 53 and the folder
    would hold a share of the sweep no other folder holds.
    """
    printed, arguments = _run_the_runner(tmp_path, 447, 53, shards=500)

    assert "shard: 447 of 500" in printed
    assert "--shard" in arguments
    assert arguments[arguments.index("--shard") + 1] == "447/500"


def test_the_slurm_runner_falls_back_to_the_arrays_own_count(tmp_path):
    """The whole sweep in one array needs no count of its own."""
    printed, arguments = _run_the_runner(tmp_path, 3, 200)

    assert "shard: 3 of 200" in printed
    assert arguments[arguments.index("--shard") + 1] == "3/200"


def test_the_experiment_runner_runs_the_python_of_the_jobs_environment(
    tmp_path,
):
    """With DECSIM_PYTHON unset, the task runs the python on PATH.

    The shard is the offset plus the task id, of the experiment's count.
    """
    _stub, recorded = _stub_python(tmp_path)
    environment = dict(os.environ)
    environment.pop("DECSIM_PYTHON", None)
    environment["SLURM_SUBMIT_DIR"] = str(PACKAGE_ROOT)
    environment["SLURM_ARRAY_TASK_ID"] = "3"
    run_dir = tmp_path / "run"
    environment["RUN"] = str(run_dir)
    environment["SHARDS"] = "500"
    environment["OFFSET"] = "150"
    path = environment.get("PATH", "")
    environment["PATH"] = f"{tmp_path}:{path}"

    completed = subprocess.run(
        ["bash", str(EXPERIMENT_RUNNER), "configs/weak_ler.yaml"],
        capture_output=True,
        text=True,
        env=environment,
    )

    recorded_text = recorded.read_text()
    arguments = recorded_text.splitlines()
    assert completed.returncode == 0, completed.stderr
    assert arguments[:3] == ["-m", "decsim", "collect"]
    assert arguments[arguments.index("--shard") + 1] == "153/500"


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
    return stub


def _refused_run(tmp_path, status):
    """The runner started against a tree, with no ALLOW_DIRTY to excuse it.

    No DECSIM_PYTHON is set, so the runner takes the python on PATH,
    which is the stub.
    """
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _stub_python(tmp_path, checkout)
    _stub_git(tmp_path, status)
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in UNSET_FOR_THE_RUNNER
    }
    environment["SLURM_SUBMIT_DIR"] = str(PACKAGE_ROOT)
    path = environment.get("PATH", "")
    environment["PATH"] = f"{tmp_path}:{path}"
    return subprocess.run(
        ["bash", str(SLURM_RUNNER), "configs/weak_ler.yaml"],
        capture_output=True,
        text=True,
        env=environment,
    )


def test_the_slurm_runner_refuses_a_tree_with_uncommitted_changes(tmp_path):
    """A dirty tree is refused, and the message says how to proceed.

    Every task imports the tree as it stands when that task starts, so a
    sweep launched from a tree still being edited runs code no folder of
    it can name.
    """
    completed = _refused_run(tmp_path, "dirty")

    assert completed.returncode != 0
    assert "has uncommitted changes" in completed.stderr
    assert "ALLOW_DIRTY=1" in completed.stderr


def test_the_slurm_runner_refuses_a_tree_git_cannot_read(tmp_path):
    """A tree with no git is refused for the reason the runner states.

    The job must name the code it ran; where git says nothing, nothing
    can, so an unknown tree is as unrunnable as a dirty one.
    """
    completed = _refused_run(tmp_path, "unknown")

    assert completed.returncode != 0
    assert "dirty: unknown" in completed.stdout
    assert "git says nothing about" in completed.stderr
    assert "ALLOW_DIRTY=1" in completed.stderr


def test_the_slurm_runner_starts_from_a_clean_tree(tmp_path):
    """A tree git vouches for runs, on the python of the job's environment.

    The refusals read only the tree, and with DECSIM_PYTHON unset the
    collect runs on the python on PATH, as a fresh clone's job does.
    """
    completed = _refused_run(tmp_path, "clean")

    recorded = tmp_path / "argv.txt"
    recorded_text = recorded.read_text()
    assert completed.returncode == 0, completed.stderr
    assert "dirty: 0" in completed.stdout
    assert "collect" in recorded_text


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
        "logical failures: 0 of 2 shots", "logical failures: 1 of 2 shots"
    )

    load_differences = check.page_differences(tutorial, new_load, outputs)
    count_differences = check.page_differences(tutorial, new_count, outputs)

    assert not tutorial.is_priced
    assert new_load != text
    assert load_differences == []
    assert len(count_differences) == 1
