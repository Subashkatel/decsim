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


def _called_on(references: tuple, packages: tuple) -> dict:
    """The attribute names reached for through one collaborator, by site.

    A reference is matched wherever it stands: a field (self.weak_store),
    a local or a parameter (store), so a package that passes the
    collaborator down to a helper is walked too.
    """
    sites = {}
    for package in packages:
        directory = PACKAGE_ROOT / "decsim" / package
        modules = directory.glob("*.py")
        for path in sorted(modules):
            _collect_calls(path, references, sites)
    return sites


def _collect_calls(path: pathlib.Path, references: tuple, sites: dict) -> None:
    """Add every <reference>.<name> of one file to the sites map."""
    text = path.read_text()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if _base_name(node.value) not in references:
            continue
        where = f"{path.parent.name}/{path.name}:{node.lineno}"
        found = sites.setdefault(node.attr, [])
        found.append(where)


def _base_name(node) -> str:
    """The name the attribute is read from, or the empty string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _undeclared(called: dict, ports: tuple, exempt: tuple = ()) -> dict:
    """The reached names no listed port declares, with their sites."""
    declared = set(exempt)
    for port in ports:
        declared |= _port_names(port)
    undeclared = {}
    for name, sites in called.items():
        if name not in declared:
            undeclared[name] = sites
    return undeclared


def test_the_decode_queue_port_declares_what_the_window_side_calls():
    """The window and confidence sides hold a manager only as this port."""
    callers = ("windows", "escalation", "confidence", "controller")
    called = _called_on(("decode_queue",), callers)
    assert _undeclared(called, ("DecodeQueue",)) == {}


def test_the_store_ports_declare_what_a_caller_outside_the_buffer_calls():
    """Two neighbours cross a store, so two ports carry its promises.

    trace is exempt by the ports' own rule: observation reaches a
    component through the callbacks it fires, never through a port.
    """
    callers = ("controller", "windows", "observe")
    references = (
        "store",
        "weak_store",
        "strong_store",
        "primary_store",
        "round_store",
    )
    called = _called_on(references, callers)
    ports = ("RoundStore", "RetainedRounds")
    assert _undeclared(called, ports, exempt=("trace",)) == {}


def test_the_window_side_ports_declare_what_a_strong_shape_calls():
    """A shape row written outside decsim programs against these five."""
    references = ("planner", "retention", "builder", "requester", "ledger")
    called = _called_on(references, ("escalation",))
    ports = (
        "WindowPlan",
        "WindowRetention",
        "WindowJobBuilder",
        "WindowRequests",
        "LogicalLedger",
    )
    assert _undeclared(called, ports) == {}


def test_the_window_transfers_port_declares_what_its_senders_call():
    """The decoder outputs and a store's output hold the fabric adapter."""
    callers = ("decoders", "syndrome_buffer")
    called = _called_on(("transfers",), callers)
    assert _undeclared(called, ("WindowTransfers",)) == {}


def test_the_window_model_port_declares_what_the_planner_calls():
    """The planner's provider is a port, not an Any-typed field."""
    called = _called_on(("provider",), ("windows",))
    assert _undeclared(called, ("WindowModelSource",)) == {}


def test_the_qpu_port_declares_what_the_controller_calls():
    called = _called_on(("qpu",), ("controller",))
    assert _undeclared(called, ("Qpu",)) == {}


def test_the_window_input_port_declares_what_the_controller_calls():
    called = _called_on(("windows",), ("controller",))
    assert _undeclared(called, ("WindowInput",)) == {}
