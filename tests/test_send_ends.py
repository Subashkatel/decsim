"""Every send is executed at an end of the hop it rides.

OMNeT++ enforces this at runtime, refusing a module that sends a message
it does not own (tmp/resources/omnetpp/src/sim/csimplemodule.cc:333-334,
omnetpp-6.1.0), and gem5 bills a transfer to the port it left by rather
than to whoever arranged it (packet.hh:424-431).
decsim has no runtime check for it, so the rule is read out of the tree:
a component module that names a LinkPath must be one of that path's two
ends.
"""

import ast
import pathlib

TESTS_FILE = pathlib.Path(__file__)
TESTS_PATH = TESTS_FILE.resolve()
PACKAGE_ROOT = TESTS_PATH.parent.parent
DECSIM_ROOT = PACKAGE_ROOT / "decsim"

# each hop's two ends, as the packages that own them
ENDS_OF_PATH = {
    "QPU_TO_CONTROLLER": ("qpu", "controller"),
    "CONTROLLER_TO_WEAK_BUFFER": ("controller", "syndrome_buffer"),
    "WEAK_BUFFER_TO_WEAK_DECODER": ("syndrome_buffer", "decoders"),
    "WEAK_DECODER_TO_STRONG_DECODER": ("decoders",),
    "STRONG_BUFFER_TO_STRONG_DECODER": ("syndrome_buffer", "decoders"),
    "WEAK_DECODER_TO_FRAME": ("decoders", "pauli_frame"),
    "DECODER_TO_DECODER": ("decoders",),
    "STRONG_DECODER_TO_FRAME": ("decoders", "pauli_frame"),
    "FRAME_TO_CONTROLLER": ("pauli_frame", "controller"),
    "CONTROLLER_TO_QPU": ("controller", "qpu"),
    "CONTROLLER_TO_STRONG_BUFFER": ("controller", "syndrome_buffer"),
}

# the components on the reaction path; the root wires the paths onto the
# ports (build, machine), the fabric carries them (links) and the
# reporting side reads them (observe, front, collect), so none of those
# is a sender and none is walked
COMPONENT_PACKAGES = (
    "qpu",
    "controller",
    "syndrome_buffer",
    "windows",
    "decoders",
    "escalation",
    "confidence",
    "pauli_frame",
)


def _named_paths(path: pathlib.Path) -> list:
    """Every LinkPath member this module names, with its line."""
    text = path.read_text()
    tree = ast.parse(text)
    named = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        inner = node.value
        if not isinstance(inner, ast.Attribute):
            continue
        if inner.attr != "LinkPath":
            continue
        named.append((node.attr, node.lineno))
    return named


def test_every_component_that_names_a_hop_is_an_end_of_it():
    wrong_end = {}
    for package in COMPONENT_PACKAGES:
        directory = DECSIM_ROOT / package
        modules = directory.glob("*.py")
        for module in sorted(modules):
            _add_wrong_ends(module, package, wrong_end)
    assert wrong_end == {}


def _add_wrong_ends(module: pathlib.Path, package: str, wrong: dict) -> None:
    """Add every hop this module names that it is not an end of."""
    for name, lineno in _named_paths(module):
        ends = ENDS_OF_PATH[name]
        if package in ends:
            continue
        wrong[f"{package}/{module.name}:{lineno}"] = name


def test_the_table_of_ends_covers_every_hop():
    """A new hop joins the table, so the law never silently shrinks."""
    text = (DECSIM_ROOT / "records" / "transfers.py").read_text()
    tree = ast.parse(text)
    members = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name != "LinkPath":
            continue
        members = _member_names(node)
    assert sorted(members) == sorted(ENDS_OF_PATH)


def _member_names(node: ast.ClassDef) -> list:
    """The enum members declared in the class body."""
    names = []
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            assigned = _assigned_names(statement)
            names.extend(assigned)
    return names


def _assigned_names(statement: ast.Assign) -> list:
    """The plain names one assignment writes to."""
    names = []
    for target in statement.targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
    return names
