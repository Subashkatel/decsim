"""No construction decides by recognising a class (STYLE.md rule 10).

A plug-in table maps a yaml kind to a row, and the caller asks the table
for the row and calls it. Code that instead recognises the row it got
back, `row is SomeClass` or `isinstance(row, SomeClass)`, moves the
choice out of the table and into a chain of comparisons, so a row added
to the table is reachable from yaml and still not built: gem5's params
object gives every SimObject one constructor signature for exactly this
reason (src/python/m5/SimObject.py:204-205).

The check is name-based and deliberately blunt: every class a module
tests against is held against the list below, and anything else fails.
The list is short because recognising a class is rare in a tree that
dispatches through tables, so a new name on it is a decision a reader
should see in the diff.

An enum member is a value, not a class, so `tier is DecoderTier.STRONG`
is not a class test and is not checked here; the last segment of a
dotted name tells the two apart, since a member is written in capitals.
"""

import ast
import pathlib
import sys

# Types the yaml boundary reads: a section's value is whatever the file
# held, so the settings that parse it say what shape they accept.
YAML_BOUNDARY = frozenset(
    {
        "bool",
        "dict",
        "float",
        "int",
        "list",
        "str",
        "tuple",
        "Mapping",
        "Integral",
        "Real",
        "numbers.Integral",
        "pathlib.Path",
    }
)

# Record types: a record is data, and a reader that meets two shapes of
# it asks which one it has. These are not table rows and never dispatch
# a construction.
RECORD_TYPES = frozenset(
    {
        "decoding_records.DecodeJob",
        "evidence_records.Closed",
        "evidence_records.Open",
        "transfer_records.BoundaryTransferRelation",
        "window_records.DependencyResidual",
    }
)

# Ports: a runtime_checkable Protocol is a question about what an object
# answers, not about which class it is, which is the opposite of
# recognising a row.
PORTS = frozenset(
    {
        "ports.Decoder",
        "ports.DetectionEventFormer",
        "RunSeedComposite",
        "RunSeedConsumer",
        "seeding.RunSeedComposite",
    }
)

# Types decsim does not own, met at the edge of another library.
FOREIGN_TYPES = frozenset(
    {
        "stim.Circuit",
        "stim.CircuitRepeatBlock",
        "TesseractDecoderConfig",
    }
)

ALLOWED = YAML_BOUNDARY | RECORD_TYPES | PORTS | FOREIGN_TYPES

# `x is None` and `x is True` test a value, not a class.
SINGLETONS = frozenset({"None", "True", "False"})

USAGE = (
    "usage: check_row_recognition.py <package root>; "
    "from the checkout, tools/check_row_recognition.py decsim"
)


def source_paths(root: pathlib.Path) -> list:
    """Every module under the root, built sources left out."""
    found = []
    modules = root.rglob("*.py")
    for path in modules:
        if "__pycache__" in path.parts:
            continue
        found.append(path)
    return sorted(found)


def is_class_name(text: str) -> bool:
    """Whether a dotted name reads as a class rather than an enum member."""
    if text in SINGLETONS:
        return False
    parts = text.split(".")
    last = parts[-1]
    if not last[:1].isupper():
        return False
    return last != last.upper()


def isinstance_classes(node: ast.Call) -> list:
    """The classes one isinstance call tests against."""
    if len(node.args) != 2:
        return []
    second = node.args[1]
    if isinstance(second, ast.Tuple):
        return list(second.elts)
    return [second]


def tested_classes(tree: ast.AST) -> list:
    """(line, name) of every class one module tests an object against."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            called = _call_classes(node)
            found.extend(called)
        if isinstance(node, ast.Compare):
            compared = _identity_classes(node)
            found.extend(compared)
    return found


def _call_classes(node: ast.Call) -> list:
    """The classes an isinstance call names, with the call's line."""
    func = node.func
    if not isinstance(func, ast.Name):
        return []
    if func.id != "isinstance":
        return []
    found = []
    for element in isinstance_classes(node):
        text = ast.unparse(element)
        found.append((node.lineno, text))
    return found


def _identity_classes(node: ast.Compare) -> list:
    """The classes an `is` comparison names, with the comparison's line."""
    tests_identity = False
    for operator in node.ops:
        if isinstance(operator, (ast.Is, ast.IsNot)):
            tests_identity = True
    if not tests_identity:
        return []
    found = []
    for comparator in node.comparators:
        text = ast.unparse(comparator)
        if is_class_name(text):
            found.append((node.lineno, text))
    return found


def findings_of(path: pathlib.Path) -> list:
    """Every class this module tests against that is not on the list."""
    text = path.read_text()
    tree = ast.parse(text)
    found = []
    for line, name in tested_classes(tree):
        if name in ALLOWED:
            continue
        found.append(f"{path}:{line}: recognises the class {name}")
    return sorted(found)


def main(arguments) -> int:
    """Print every unlisted class test; a finding fails the check."""
    if len(arguments) != 1:
        print(USAGE, file=sys.stderr)
        return 2
    root = pathlib.Path(arguments[0])
    if not root.is_dir():
        print(f"{root} is not a directory", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    findings = []
    for path in source_paths(root):
        found = findings_of(path)
        findings.extend(found)
    for finding in findings:
        print(finding)
    count = len(findings)
    print(f"class recognition: {count} unlisted class tests")
    if findings:
        return 1
    return 0


if __name__ == "__main__":
    exit_code = main(sys.argv[1:])
    sys.exit(exit_code)
