"""The repository's own scripts answer for themselves.

tools/check.sh runs three checkers over the tree, and a checker that
misreads its arguments fails open: it exits 0 having looked at nothing,
and the check silently stops holding. `decsim run --slurm` fails the
same way, by asking for the wrong job or running a task on code nobody
can name, so it is held here beside them, against a stub git and
sbatch. These tests hold each to what it does with its arguments.
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

TESTS_FILE = pathlib.Path(__file__)
TESTS_PATH = TESTS_FILE.resolve()
PACKAGE_ROOT = TESTS_PATH.parent.parent
TOOLS = PACKAGE_ROOT / "tools"
CHECK_SCRIPT = TOOLS / "check.sh"
# What a submitting shell could hand `decsim run --slurm`: an excuse for
# a dirty tree, a tree reading, or the job the suite itself runs in.
UNSET_FOR_SLURM = (
    "ALLOW_DIRTY",
    "DECSIM_TREE_DIRTY",
    "DECSIM_TREE_PATCH_SHA256",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_ID",
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


def test_rule_six_checks_bind_the_package_and_nothing_else(tmp_path, capsys):
    """An unannotated public function below a private one fails in decsim.

    The same module outside the package passes the check.
    """
    tool = _tool("check_one_action")
    module_text = (
        "def _helper():\n"
        "    return 1\n"
        "\n"
        "\n"
        "def public(value):\n"
        "    return value\n"
    )
    package = tmp_path / "decsim"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "part.py").write_text(module_text)
    (tmp_path / "part.py").write_text(module_text)
    exit_code = tool.main([str(tmp_path)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "decsim/part.py:5: unannotated signature" in captured.out
    assert "decsim/part.py:5: public after private" in captured.out
    assert "2 findings in 1 of 3 files checked" in captured.out


def test_rule_eleven_fails_a_function_that_decides_six_times(tmp_path, capsys):
    """Five decisions pass and six fail, under rule 11's own count.

    An `and`, an `or` and a comprehension's `for` and `if` each count
    one, and the nested `inner` is counted apart from `outer`.
    """
    tool = _tool("check_one_action")
    module = tmp_path / "decisions.py"
    module.write_text(
        "def five(values):\n"
        "    kept = [value for value in values if value]\n"
        "    while kept:\n"
        "        kept.pop()\n"
        "    if values or kept:\n"
        "        return values\n"
        "\n"
        "def six(values):\n"
        "    kept = [value for value in values if value]\n"
        "    while kept and values:\n"
        "        kept.pop()\n"
        "    if values or kept:\n"
        "        return values\n"
        "\n"
        "def outer(values):\n"
        "    kept = [value for value in values if value]\n"
        "    def inner():\n"
        "        while kept and values:\n"
        "            kept.pop()\n"
        "        if values or kept:\n"
        "            return values\n"
        "    return inner\n"
    )
    exit_code = tool.main([str(tmp_path)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "decisions.py:8: over five decisions: six decides 6 times" in (
        captured.out
    )
    assert "1 findings in 1 of 1 files checked" in captured.out


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


def _stub_python(tmp_path):
    """An interpreter that writes down every command line it is given."""
    recorded = tmp_path / "argv.txt"
    stub = tmp_path / "python"
    stub.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "{recorded}"\n'
    )
    stub.chmod(0o755)
    return recorded


def test_the_check_script_runs_the_active_environments_python(tmp_path):
    """With DECSIM_PYTHON unset, check.sh runs the python on PATH.

    A fresh clone has no .venv of its own, so the default names none.
    """
    recorded = _stub_python(tmp_path)
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

    blocks = check.fenced_blocks(text)
    commands = check.block_commands(blocks[0].lines)

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
    "shown, difference_count", [("2", 0), ("3", 1)], ids=["same", "moved"]
)
def test_the_tutorial_check_runs_the_python_blocks_as_one_session(
    tmp_path, shown, difference_count
):
    """A later block reads what an earlier one set, and its print is held."""
    check = _tool("check_tutorial_runs")
    text = (
        "```python\n"
        "count = 1\n"
        "```\n"
        "```python\n"
        "print(count + 1)\n"
        "```\n"
        "```\n"
        f"{shown}\n"
        "```\n"
    )
    tutorial = check.Tutorial(page="page.md", is_priced=True)
    blocks = check.fenced_blocks(text)

    _outputs, differences = check.run_page(tutorial, blocks, tmp_path)

    assert len(differences) == difference_count


@pytest.mark.parametrize(
    "shown_line, moved_line",
    [
        (r"^(3,15,50,)[0-9.]+", r"\g<1>99.9"),
        (r"^88\.424 ", "88.425 "),
    ],
    ids=["sweep column", "trace tick"],
)
def test_a_priced_tutorial_fails_the_check_when_a_value_moves(
    shown_line, moved_line
):
    """On two_tiers every tick is priced, so every line is held."""
    check = _tool("check_tutorial_runs")
    tutorial = check.TUTORIALS[2]
    text, outputs = _page_and_its_own_output(check, tutorial.page)
    moved = re.sub(shown_line, moved_line, text, count=1, flags=re.MULTILINE)
    blocks = check.fenced_blocks(text)
    moved_blocks = check.fenced_blocks(moved)

    unmoved_differences = check.page_differences(tutorial, blocks, outputs)
    moved_differences = check.page_differences(tutorial, moved_blocks, outputs)

    assert tutorial.is_priced
    assert moved != text
    assert unmoved_differences == []
    assert len(moved_differences) == 1


def test_a_wall_clock_tutorial_holds_its_qpu_tick_and_not_its_decode_tick():
    """With a named decoder, fully done is the host's; execution done is not."""
    check = _tool("check_tutorial_runs")
    tutorial = check.Tutorial(page="page.md", is_priced=False)
    printed = (
        "config: page",
        "execution done: 30000000 ticks",
        "fully done: 31084000 ticks",
    )
    outputs = [check.Printed(command="decsim run page.py", lines=printed)]
    text = "```\n" + "\n".join(printed) + "\n```\n"
    new_fully_done = text.replace("fully done: 31084000", "fully done: 99")
    new_execution_done = text.replace(
        "execution done: 30000000", "execution done: 99"
    )
    decode_blocks = check.fenced_blocks(new_fully_done)
    qpu_blocks = check.fenced_blocks(new_execution_done)

    decode_differences = check.page_differences(
        tutorial, decode_blocks, outputs
    )
    qpu_differences = check.page_differences(tutorial, qpu_blocks, outputs)

    assert new_fully_done != text
    assert decode_differences == []
    assert len(qpu_differences) == 1


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


def _stub_sbatch(tmp_path):
    """An sbatch that records each submission and answers a job id.

    The ids count up from 1000 in submission order, as --parsable prints
    them, so no test reaches Slurm and a dependency names the job it
    follows.
    """
    stub = tmp_path / "sbatch"
    submissions = tmp_path / "submissions.txt"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> {submissions}\n'
        f"wc -l < {submissions} | awk '{{print 999 + $1}}'\n"
    )
    stub.chmod(0o755)


def _slurm_environment(tmp_path, status):
    """The submitting shell: a stub git and a stub sbatch."""
    _stub_git(tmp_path, status)
    _stub_sbatch(tmp_path)
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in UNSET_FOR_SLURM
    }
    path = environment.get("PATH", "")
    environment["PATH"] = f"{tmp_path}:{path}"
    environment["PYTHONPATH"] = str(PACKAGE_ROOT)
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


# The weak base at two distances, four shots each.
TWO_POINT_RUN_FILE = """
import decsim
import decsim.settings as machine_settings

points = []
for distance in (3, 5):
    machine = machine_settings.weak_decoder_baseline(distance, 0.001, 1.0)
    point = decsim.Point(f"d{distance}", machine, {"qpu.distance": distance})
    points.append(point)
collection = decsim.CollectionSettings(max_shots=4)
experiment = decsim.Experiment("two_points", points, collection)
"""


def _two_point_run_file(tmp_path) -> pathlib.Path:
    """A run file of two distances."""
    run_file = tmp_path / "two_points.py"
    run_file.write_text(TWO_POINT_RUN_FILE)
    return run_file


def _slurm_arguments(config_path, results_dir, *extra) -> list:
    """`decsim run <run file> --slurm`, into results_dir."""
    return [
        "run",
        str(config_path),
        "--slurm",
        "--out",
        str(results_dir),
        *extra,
    ]


def test_a_slurm_dry_run_writes_one_array_and_one_fold(tmp_path):
    """Task i of the array runs point i of the run file, then a fold.

    A dry run records the points and writes both files, and submits
    nothing. The folder's path holds a space, and each line still reads
    back as the arguments it was written from, the task index left for
    the shell to expand.
    """
    config_path = _two_point_run_file(tmp_path)
    results_dir = tmp_path / "run folder"
    environment = _slurm_environment(tmp_path, "clean")
    arguments = _slurm_arguments(config_path, results_dir, "--dry-run")

    completed = _decsim(tmp_path, arguments, environment)

    run_script = results_dir / "run.sbatch"
    run_text = run_script.read_text()
    run_lines = run_text.splitlines()
    fold_script = results_dir / "fold.sbatch"
    fold_text = fold_script.read_text()
    fold_lines = fold_text.splitlines()
    decsim_run = [sys.executable, "-m", "decsim", "run"]
    run_path = config_path.resolve()
    assert completed.returncode == 0, completed.stderr
    assert run_lines[1] == (
        "#SBATCH --array=0-1 --cpus-per-task=4 --mem=16384M --time=24:00:00"
    )
    assert shlex.split(run_lines[2]) == [
        "#SBATCH",
        f"--output={results_dir}/logs/%a.log",
    ]
    assert shlex.split(run_lines[3]) == [
        *decsim_run,
        str(run_path),
        "--out",
        str(results_dir),
        "--task",
        "$SLURM_ARRAY_TASK_ID",
        "--processes",
        "4",
    ]
    assert shlex.split(fold_lines[2]) == [
        "#SBATCH",
        f"--output={results_dir}/logs/fold.log",
    ]
    assert shlex.split(fold_lines[3]) == [
        *decsim_run,
        "--fold",
        "--out",
        str(results_dir),
    ]
    point_records = results_dir.glob("points/*/machine.json")
    point_record_paths = list(point_records)
    assert len(point_record_paths) == 2
    assert not (tmp_path / "submissions.txt").exists()


def test_a_launch_submits_the_array_then_the_fold_behind_it(tmp_path):
    """The fold's dependency names the job id sbatch answered."""
    config_path = _two_point_run_file(tmp_path)
    results_dir = tmp_path / "results"
    environment = _slurm_environment(tmp_path, "clean")
    arguments = _slurm_arguments(config_path, results_dir)

    completed = _decsim(tmp_path, arguments, environment)

    submissions_path = tmp_path / "submissions.txt"
    submissions_text = submissions_path.read_text()
    array_text, fold_text = submissions_text.splitlines()
    assert completed.returncode == 0, completed.stderr
    assert array_text == f"--parsable {results_dir / 'run.sbatch'}"
    assert fold_text == (
        f"--parsable --dependency=afterany:1000 {results_dir / 'fold.sbatch'}"
    )
    assert "array job 1000, fold job 1001" in completed.stdout


@pytest.mark.parametrize(
    ("status", "sentence"),
    [("dirty", "has uncommitted changes"), ("unknown", "git says nothing")],
)
@pytest.mark.parametrize("where", ["launch", "task"])
def test_a_tree_git_does_not_vouch_for_is_refused_where_it_starts(
    tmp_path, status, sentence, where
):
    """A dirty tree, or one git cannot read, is refused where it starts.

    Every task imports the tree as it stands when that task starts, so
    an array launched from a tree still being edited runs code no piece
    of it can name. Launching refuses it, so nothing is queued, and so
    does every task, since the tree may change after submission.
    """
    config_path = _two_point_run_file(tmp_path)
    results_dir = tmp_path / "results"
    environment = _slurm_environment(tmp_path, status)
    arguments = {
        "launch": _slurm_arguments(config_path, results_dir),
        "task": ["run", str(config_path), "--out", str(results_dir)]
        + ["--task", "0"],
    }

    completed = _decsim(tmp_path, arguments[where], environment)

    assert completed.returncode == 1
    assert sentence in completed.stderr
    assert "ALLOW_DIRTY=1" in completed.stderr
    assert not (tmp_path / "submissions.txt").exists()
    assert not results_dir.exists()
