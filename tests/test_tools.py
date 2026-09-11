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
import subprocess

TESTS_FILE = pathlib.Path(__file__)
TESTS_PATH = TESTS_FILE.resolve()
PACKAGE_ROOT = TESTS_PATH.parent.parent
TOOLS = PACKAGE_ROOT / "tools"
SLURM_RUNNER = PACKAGE_ROOT / "slurm" / "slurm_run.sh"


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


def test_every_class_on_the_tools_list_is_still_tested_against_somewhere():
    """The list stays honest: a name nobody tests against is deleted.

    An allow list is only as good as its shortness, and a stale name on
    it silences a real finding, so the list is held against the tree the
    same way STYLE.md's wide-state exemptions are.
    """
    tool = _tool("check_row_recognition")
    root = PACKAGE_ROOT / "decsim"
    tested = set()
    for path in tool.source_paths(root):
        text = path.read_text()
        tree = ast.parse(text)
        for _line, name in tool.tested_classes(tree):
            tested.add(name)
    stale = tool.ALLOWED - tested
    assert stale == set()


def _stub_python(tmp_path):
    """An interpreter that writes down the command line it was given.

    It answers the runner's question about where decsim was imported
    from with a path of its own, so the test never runs a sweep.
    """
    recorded = tmp_path / "argv.txt"
    stub = tmp_path / "python"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "-c" ]; then\n'
        f'  echo "{tmp_path}"\n'
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
    # the tree under test is the one being edited, which is the case the
    # runner refuses; the refusal has its own reason to exist and is not
    # what this test reads
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
