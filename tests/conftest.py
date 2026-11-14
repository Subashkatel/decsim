"""What the suite needs built before any test runs, and which tests are slow.

The Union-Find row decodes in C, and the library it loads is a build
artifact rather than a tracked file, so a fresh checkout has none. One
session-wide fixture builds it where a compiler is reachable. Where one
is not, or where the build fails, the session stops with the sentence
that names the command, because the alternative is the same build error
retold once per test.

tests/slow_tests.txt lists the slow tests, and a collection hook marks
each of their cases `slow`, so `python -m pytest -m "not slow"` is the
quick set; tools/slow_tests.py writes the list.
"""

import os
import pathlib
import shutil
import subprocess

import pytest

import decsim.decoders.union_find.compiled_decoder as compiled_decoder

# The fixture that runs a pytest session inside a test, for the tests of
# the hook below.
pytest_plugins = ("pytester",)

TESTS_FILE = pathlib.Path(__file__)
TESTS_PATH = TESTS_FILE.resolve()
CHECKOUT = TESTS_PATH.parent.parent
SLOW_TESTS = TESTS_PATH.parent / "slow_tests.txt"
BUILD_SENTENCE = (
    "the Union-Find decoder needs its compiled library: run "
    f"{compiled_decoder.BUILD_COMMAND} on a host with a C compiler, then "
    "run the suite again"
)
STALE_SENTENCE = (
    "tests/slow_tests.txt names tests that their files no longer define; "
    "regenerate it with tools/slow_tests.py or delete these lines:"
)


@pytest.fixture(scope="session", autouse=True)
def compiled_union_find_library():
    """The compiled Union-Find decoder, built once when it is missing."""
    path = compiled_decoder.library_path()
    if path.exists():
        return path
    compiler = os.environ.get("CC", "gcc")
    if shutil.which(compiler) is None:
        pytest.exit(BUILD_SENTENCE, returncode=1)
    command = CHECKOUT / compiled_decoder.BUILD_COMMAND
    build = subprocess.run([str(command)], cwd=CHECKOUT, check=False)
    if build.returncode != 0:
        pytest.exit(BUILD_SENTENCE, returncode=1)
    return path


# First, so the marks are on before pytest's own -m deselects.
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list) -> None:
    """Marks every case of a listed test slow, after refusing a stale list.

    A line is stale when this run collected its file but the file no
    longer defines its test, so a renamed or deleted test cannot stay on
    the list unseen. The test is looked up in the file's module rather
    than among the cases collected, because a run of one test of a file
    collects only that test.
    """
    slow_tests = _listed_slow_tests()
    stale = _stale_lines(slow_tests, items)
    if stale:
        stale_text = "\n".join(stale)
        raise pytest.UsageError(f"{STALE_SENTENCE}\n{stale_text}")
    for item in items:
        test, _, _ = item.nodeid.partition("[")
        if test in slow_tests:
            item.add_marker(pytest.mark.slow)


def _listed_slow_tests() -> set:
    """The tests on the list, each `path::name` without its parameters."""
    text = SLOW_TESTS.read_text()
    slow_tests = set()
    for line in text.splitlines():
        if line and not line.startswith("#"):
            slow_tests.add(line)
    return slow_tests


def _stale_lines(slow_tests: set, items: list) -> list:
    """The listed tests whose file was collected but does not define them."""
    modules_by_file = {}
    for item in items:
        file, _, _ = item.nodeid.partition("::")
        modules_by_file[file] = item.module
    stale = []
    for test in sorted(slow_tests):
        file, _, names = test.partition("::")
        module = modules_by_file.get(file)
        if module is not None and not _defines(module, names):
            stale.append(test)
    return stale


def _defines(module, names: str) -> bool:
    """Whether `name` or `Class::name` leads to an object of the module."""
    holder = module
    for name in names.split("::"):
        holder = getattr(holder, name, None)
    return holder is not None
