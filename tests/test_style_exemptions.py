"""STYLE.md rule 1's wide-state exemption list is honest and short.

The checker reads the list from the style guide, so the guide is the
one place a reader learns which classes are wide on purpose. A name
that no longer belongs there would silence a real report, so the list
is checked against the code it exempts.
"""

import ast
import importlib.util
import pathlib

TESTS_FILE = pathlib.Path(__file__)
TESTS_PATH = TESTS_FILE.resolve()
PACKAGE_ROOT = TESTS_PATH.parent.parent
CHECKER_PATH = PACKAGE_ROOT / "tools" / "check_one_action.py"
MAX_EXEMPTIONS = 8


def _checker():
    """The style checker, imported from the tools folder by path."""
    spec = importlib.util.spec_from_file_location(
        "check_one_action", CHECKER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _class_definition(path: pathlib.Path, class_name: str):
    """The named class of a file, or None when the file has no such class."""
    text = path.read_text()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name == class_name:
            return node
    return None


def test_every_exempt_class_exists_where_the_guide_says_it_does():
    checker = _checker()
    exemptions = checker.wide_state_exemptions()
    assert exemptions
    for class_name, class_path in exemptions:
        path = PACKAGE_ROOT / class_path
        assert path.exists(), class_path
        node = _class_definition(path, class_name)
        assert node is not None, class_name


def test_every_exempt_class_would_otherwise_be_reported():
    """A class that is no longer wide leaves the list, so it stays honest."""
    checker = _checker()
    exemptions = checker.wide_state_exemptions()
    for class_name, class_path in exemptions:
        path = PACKAGE_ROOT / class_path
        node = _class_definition(path, class_name)
        init = checker.init_method(node)
        assert init is not None, class_name
        names = checker.self_attributes_assigned(init)
        assert len(names) > checker.MAX_ATTRIBUTES, class_name


def test_the_list_stays_short():
    """Past eight names the rule is wrong, not the list (STYLE.md rule 1)."""
    checker = _checker()
    exemptions = checker.wide_state_exemptions()
    assert len(exemptions) <= MAX_EXEMPTIONS


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
