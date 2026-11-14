"""The slow list's hook in tests/conftest.py, run inside a pytest session.

pytest tests a plugin by running a session on files the test writes,
with its own pytester (_pytest/pytester.py, pytest 9.1.1). Each test here
writes tests/conftest.py, a test file and a list of its own, and runs
the hook on them in that session.
"""

import pathlib

import pytest

TESTS_FILE = pathlib.Path(__file__)
TESTS_PATH = TESTS_FILE.resolve()
CONFTEST = TESTS_PATH.parent / "conftest.py"
# Two tests, the first with two cases.
TEST_FILE = """
import pytest

@pytest.mark.parametrize("case", [1, 2])
def test_listed(case):
    pass

def test_unlisted():
    pass
"""


def _write_session(pytester, slow_tests: str) -> None:
    """The real conftest, the test file and the list in pytester's folder."""
    conftest_text = CONFTEST.read_text()
    pytester.makeconftest(conftest_text)
    pytester.makepyfile(test_pair=TEST_FILE)
    pytester.makefile(".txt", slow_tests=slow_tests)


def test_the_quick_set_leaves_out_every_case_of_a_listed_test(pytester):
    """A line whose file this run does not collect is no error."""
    _write_session(
        pytester,
        "# a comment\ntest_pair.py::test_listed\ntest_other.py::test_gone\n",
    )

    result = pytester.runpytest("--collect-only", "-q", "-m", "not slow")

    assert result.ret == pytest.ExitCode.OK
    result.stdout.fnmatch_lines(
        ["test_pair.py::test_unlisted", "1/3 tests collected (2 deselected)*"]
    )


def test_a_line_whose_test_its_file_no_longer_defines_stops_the_run(
    pytester,
):
    """Only the stale line is named, though the run collects one test."""
    _write_session(
        pytester, "test_pair.py::test_listed\ntest_pair.py::test_renamed\n"
    )

    result = pytester.runpytest(
        "--collect-only", "-q", "test_pair.py::test_unlisted"
    )

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(
        ["*no longer define*", "test_pair.py::test_renamed"]
    )
    assert "test_pair.py::test_listed" not in result.stderr.str()
