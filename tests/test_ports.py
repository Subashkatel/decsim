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
DECSIM_ROOT = PACKAGE_ROOT / "decsim"
PORTS_PATH = DECSIM_ROOT / "ports.py"
# STYLE.md rule 7: a shared module beside ports in the package order, so
# its two Protocols cannot move into ports without a cycle of levels
SHARED_PROTOCOL_MODULE = "seeding"


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


def test_the_placement_port_declares_what_the_assembler_calls():
    """The assembler holds the run's detection event row only as this port."""
    called = _called_on(("detection_events",), ("controller",))
    assert _undeclared(called, ("DetectionEventPlacement",)) == {}


def test_the_former_port_declares_what_every_row_that_forms_calls():
    """Both placements and each tier's logic hold a former as this port."""
    callers = ("detector_error_model", "decoders")
    called = _called_on(("former",), callers)
    assert _undeclared(called, ("DetectionEventFormer",)) == {}


def test_the_window_input_port_declares_what_its_two_callers_call():
    """The controller and Buffer 0's incoming port hold the manager."""
    callers = ("controller", "syndrome_buffer")
    called = _called_on(("windows",), callers)
    assert _undeclared(called, ("WindowInput",)) == {}


def test_the_memory_round_port_declares_what_the_transmitter_calls():
    """The controller holds the decoders' memory end only as this port."""
    called = _called_on(("memory_arrivals",), ("controller",))
    assert _undeclared(called, ("MemoryRoundArrivals",)) == {}


def test_the_store_input_port_declares_what_the_transmitter_calls():
    """The controller holds Buffer 0's incoming port only as this port."""
    called = _called_on(("store_input",), ("controller",))
    assert _undeclared(called, ("RoundStoreInput",)) == {}


def _decsim_modules() -> list:
    """Every module of the package, in path order."""
    modules = DECSIM_ROOT.rglob("*.py")
    return sorted(modules)


def _local_protocols() -> dict:
    """Every Protocol declared outside decsim/ports.py, by declaring file."""
    declared = {}
    for path in _decsim_modules():
        if path == PORTS_PATH:
            continue
        _add_protocols(path, declared)
    return declared


def _add_protocols(path: pathlib.Path, declared: dict) -> None:
    """Add one module's top-level Protocol declarations."""
    text = path.read_text()
    tree = ast.parse(text)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if _is_protocol(node):
            declared[node.name] = path


def _is_protocol(node: ast.ClassDef) -> bool:
    """True when the class inherits Protocol directly."""
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "Protocol":
            return True
    return False


def _package_of(path: pathlib.Path) -> str:
    """The package a module belongs to; a root module is its own."""
    if path.parent == DECSIM_ROOT:
        return path.stem
    return path.parent.name


def _named_outside(name: str, package: str, declaring: pathlib.Path) -> list:
    """Every place outside the package that names the Protocol in code."""
    sites = []
    for path in _decsim_modules():
        if path == declaring:
            continue
        if _package_of(path) == package:
            continue
        _add_mentions(path, name, sites)
    return sites


def _add_mentions(path: pathlib.Path, name: str, sites: list) -> None:
    """Add every code mention of the name in one module."""
    text = path.read_text()
    tree = ast.parse(text)
    where = f"{path.parent.name}/{path.name}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            sites.append(f"{where}:{node.lineno}")
        if isinstance(node, ast.Attribute) and node.attr == name:
            sites.append(f"{where}:{node.lineno}")


def test_every_protocol_outside_the_port_file_is_one_packages_seam():
    """STYLE.md rule 7's sentence, held to the tree.

    A Protocol that is not in decsim/ports.py is a seam inside one
    package: nothing outside that package names it, so no component
    learns it and the package may change it alone. decsim/seeding.py is
    the named exception.
    """
    crossing = {}
    protocols = _local_protocols()
    for name, path in protocols.items():
        package = _package_of(path)
        if package == SHARED_PROTOCOL_MODULE:
            continue
        sites = _named_outside(name, package, path)
        if sites:
            crossing[name] = sites
    assert crossing == {}
