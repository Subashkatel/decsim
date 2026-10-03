"""STYLE.md rule 1's wide-state report, held to the package.

The guide's exemption list is empty, so no class of the package may set
more than six attributes, and no caller reaches past the windows facade.
"""

import ast
import importlib.util
import pathlib

TESTS_FILE = pathlib.Path(__file__)
TESTS_PATH = TESTS_FILE.resolve()
PACKAGE_ROOT = TESTS_PATH.parent.parent
CHECKER_PATH = PACKAGE_ROOT / "tools" / "check_one_action.py"


def _checker():
    """The style checker, imported from the tools folder by path."""
    spec = importlib.util.spec_from_file_location(
        "check_one_action", CHECKER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wide_state_findings(checker, package):
    """Every wide-state line the checker reports under a folder."""
    wide = []
    for path in checker.python_files([package]):
        findings = checker.check_file(path)
        reported = _wide_state_lines(findings)
        wide.extend(reported)
    return wide


def _wide_state_lines(findings):
    lines = []
    for finding in findings:
        if finding.kind == "wide state":
            lines.append(str(finding))
    return lines


def test_the_checker_reports_no_wide_state_in_the_package():
    checker = _checker()
    package = PACKAGE_ROOT / "decsim"
    wide = _wide_state_findings(checker, package)
    assert wide == []


def _reaches_through(reference: str, package: pathlib.Path) -> dict:
    """Every <reference>.<a>.<b> written outside the facade's own package."""
    reaches = {}
    modules = package.rglob("*.py")
    for path in sorted(modules):
        if path.parent.name == "windows":
            continue
        text = path.read_text()
        tree = ast.parse(text)
        _collect_reaches(path, tree, reference, reaches)
    return reaches


def _collect_reaches(path, tree, reference: str, reaches: dict) -> None:
    """Add one file's two-deep reaches through the reference."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        inner = node.value
        if not isinstance(inner, ast.Attribute):
            continue
        if _base_name(inner.value) != reference:
            continue
        where = f"{path.parent.name}/{path.name}:{node.lineno}"
        reaches[f"{inner.attr}.{node.attr}"] = where


def _base_name(node) -> str:
    """The name an attribute is read from, or the empty string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def test_no_caller_reaches_two_deep_past_the_windows_facade():
    """The exemption sentence's second half, held to the tree.

    A caller asks the facade for what it wants. The one reach left is a
    component's own trace group: observation is fired, never ported, so
    the ports carry no source and an observer names the component that
    fires it.
    """
    package = PACKAGE_ROOT / "decsim"
    reaches = _reaches_through("window_manager", package)
    past_the_facade = _without_trace_groups(reaches)
    assert past_the_facade == {}


def _without_trace_groups(reaches: dict) -> dict:
    kept = {}
    for reach, where in reaches.items():
        if not reach.endswith(".trace"):
            kept[reach] = where
    return kept
