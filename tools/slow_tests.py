"""Writes tests/slow_tests.txt, the tests the quick set leaves out.

    python tools/slow_tests.py whole.txt [quick.txt] [--threshold 0.2]

Each report is what
`python -m pytest tests -q --durations=0 --durations-min=0.02` prints on
one core: whole.txt for the whole suite, and quick.txt for the quick set,
`-m "not slow"`, run on the list whole.txt alone gives. A case's seconds
are its setup, call and teardown summed, and a test function goes on the
list, without its parameters, when any of its cases took the threshold
or more in either report. tests/conftest.py marks every case of a listed
test `slow`.

The quick set's report is there because a module's fixture or a cached
helper is paid by the first test that needs it. When that test is left
out, the cost moves to the next test that needs it, which the whole
suite's report shows as fast.
"""

import argparse
import collections
import decimal
import pathlib
import re
import sys

TOOL_FILE = pathlib.Path(__file__)
TOOL_PATH = TOOL_FILE.resolve()
CHECKOUT = TOOL_PATH.parent.parent
SLOW_TESTS = CHECKOUT / "tests" / "slow_tests.txt"
# Decimal, because the report prints hundredths and a case whose phases
# sum to the threshold exactly is on the list.
THRESHOLD_SECONDS = decimal.Decimal("0.2")
# One line of the report: `24.81s call     tests/x.py::test_name[case]`.
# A case's parameters may hold spaces, so the node id is the rest of it.
DURATION_LINE = re.compile(
    r"^(\d+\.\d+)s +(?:setup|call|teardown) +(.+)$", re.MULTILINE
)
HEADER = """\
# The slow tests: tests/conftest.py marks each one `slow`, and
# `python -m pytest -m "not slow"`, the quick set, leaves them out. A test
# is here when a case of it took {threshold} s or more (setup, call and
# teardown, one core) in the whole suite, or in the quick set run on the
# whole suite's list, where a shared setup moves to the next test that
# needs it. Regenerate it with
#   python -m pytest tests -q --durations=0 --durations-min=0.02 > whole.txt
#   python tools/slow_tests.py whole.txt --threshold {threshold}
#   python -m pytest tests -q --durations=0 --durations-min=0.02 \\
#       -m "not slow" > quick.txt
#   python tools/slow_tests.py whole.txt quick.txt --threshold {threshold}
"""


def main(argv: list) -> None:
    """Rewrite the list from the reports the arguments name."""
    parser = _parser()
    parsed = parser.parse_args(argv)
    functions = set()
    for report_name in parsed.reports:
        report_path = pathlib.Path(report_name)
        report = report_path.read_text()
        slow = slow_functions(report, parsed.threshold)
        functions.update(slow)
    listed = sorted(functions)
    text = list_text(listed, parsed.threshold)
    SLOW_TESTS.write_text(text)
    print(f"{len(listed)} slow tests written to {SLOW_TESTS}")


def slow_functions(report: str, threshold_seconds: decimal.Decimal) -> list:
    """The test functions with a case at or past the threshold, sorted."""
    seconds_by_case = case_seconds(report)
    functions = set()
    for case, seconds in seconds_by_case.items():
        if seconds < threshold_seconds:
            continue
        function, _, _ = case.partition("[")
        functions.add(function)
    return sorted(functions)


def case_seconds(report: str) -> dict:
    """Each case's setup, call and teardown seconds, summed, by node id."""
    seconds_by_case = collections.defaultdict(decimal.Decimal)
    for seconds_text, case in DURATION_LINE.findall(report):
        seconds = decimal.Decimal(seconds_text)
        seconds_by_case[case] += seconds
    return seconds_by_case


def list_text(functions: list, threshold_seconds: decimal.Decimal) -> str:
    """The list file: the header with its threshold, then one test a line."""
    header = HEADER.format(threshold=threshold_seconds)
    lines = "\n".join(functions)
    return f"{header}{lines}\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tools/slow_tests.py")
    parser.add_argument(
        "reports",
        nargs="+",
        help="durations reports: the whole suite's, then the quick set's",
    )
    parser.add_argument(
        "--threshold",
        type=decimal.Decimal,
        default=THRESHOLD_SECONDS,
        help="the seconds a case takes to put its test on the list",
    )
    return parser


if __name__ == "__main__":
    main(sys.argv[1:])
