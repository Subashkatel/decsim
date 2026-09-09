"""Every port declares what a caller outside the owning package calls.

Lampson: an interface is the set of assumptions two programs make about
each other, so a row written from the port alone must answer every call
the shipped callers make (note 17 P4). The law is read out of the tree
with ast: the Protocol bodies of decsim/ports.py against the attribute
names the callers reach for through their collaborator reference.
"""

import ast
import pathlib

TESTS_FILE = pathlib.Path(__file__)
TESTS_PATH = TESTS_FILE.resolve()
PACKAGE_ROOT = TESTS_PATH.parent.parent
PORTS_PATH = PACKAGE_ROOT / "decsim" / "ports.py"


def _port_names(port: str) -> set:
    """The methods and declared facts of one Protocol in decsim/ports.py."""
    text = PORTS_PATH.read_text()
    tree = ast.parse(text)
    body = _protocol_body(tree, port)
    names = set()
    for statement in body:
        if isinstance(statement, ast.FunctionDef):
            names.add(statement.name)
        if isinstance(statement, ast.AnnAssign):
            names.add(statement.target.id)
    return names


def _protocol_body(tree: ast.Module, port: str) -> list:
    """The class body of the named Protocol; the port must exist."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name == port:
            return node.body
    raise AssertionError(f"decsim/ports.py declares no {port}")


def _called_on(reference: str, packages: tuple) -> dict:
    """The attribute names reached for through self.<reference>, by site."""
    sites = {}
    for package in packages:
        directory = PACKAGE_ROOT / "decsim" / package
        modules = directory.glob("*.py")
        for path in sorted(modules):
            _collect_calls(path, reference, sites)
    return sites


def _collect_calls(path: pathlib.Path, reference: str, sites: dict) -> None:
    """Add every self.<reference>.<name> of one file to the sites map."""
    text = path.read_text()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not _is_reference_attribute(node, reference):
            continue
        where = f"{path.parent.name}/{path.name}:{node.lineno}"
        found = sites.setdefault(node.attr, [])
        found.append(where)


def _is_reference_attribute(node, reference: str) -> bool:
    """True for the node self.<reference>.<name>, whatever the name."""
    if not isinstance(node, ast.Attribute):
        return False
    inner = node.value
    if not isinstance(inner, ast.Attribute):
        return False
    if inner.attr != reference:
        return False
    return isinstance(inner.value, ast.Name) and inner.value.id == "self"


def test_the_decode_queue_port_declares_what_the_window_side_calls():
    """The window and confidence sides hold a manager only as this port."""
    callers = ("windows", "escalation", "confidence", "controller")
    called = _called_on("decode_queue", callers)
    declared = _port_names("DecodeQueue")
    undeclared = {}
    for name, sites in called.items():
        if name not in declared:
            undeclared[name] = sites
    assert undeclared == {}
