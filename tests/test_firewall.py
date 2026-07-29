"""AST firewall: the frozen core never imports or names parts/experiments.

CORE = {engine, message, links, pauli_frame, window_manager, speculative_recovery,
chip, payload_store, dynamic_windows, protocols, run_spec, config}. Only
run_spec.py may import parts (never experiments); experiments reach run_spec.py
only as pre-built objects.
"""
import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parent.parent / "decsim"
CORE = {"engine", "message", "links", "pauli_frame", "window_manager",
        "speculative_recovery", "chip", "payload_store", "dynamic_windows",
        "protocols", "run_spec", "config"}
PARTS_ALLOWED_IN = {"run_spec"}
# Experiment-flavored identifiers the core must never name (fabric/speculation
# are v1.1/v1.2; strategy/policy impls are parts wired via run_spec only).
BANNED_IDENTIFIERS = {"DecoderFabric", "FabricRouter", "DecoderCluster",
                      "DecoderUnit", "Speculation", "Switching"}  # DecoderCluster: retired god object
def _local_imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level > 0 and node.module:          # from .x import ...
                yield node.module.split(".")[0]
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("decsim."):
                    yield alias.name.split(".")[1]


def _module_names():
    return {p.stem for p in PKG.glob("*.py")} - {"__init__"}


def test_core_modules_exist():
    assert CORE <= _module_names()


def test_core_imports_no_parts():
    part_modules = _module_names() - CORE
    for mod in sorted(CORE - PARTS_ALLOWED_IN):
        tree = ast.parse((PKG / f"{mod}.py").read_text())
        leaked = set(_local_imports(tree)) & part_modules
        assert not leaked, f"core module {mod}.py imports parts: {sorted(leaked)}"


def test_core_names_no_experiments():
    for mod in sorted(CORE - {"run_spec"}):
        tree = ast.parse((PKG / f"{mod}.py").read_text())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        hit = names & BANNED_IDENTIFIERS
        assert not hit, f"core module {mod}.py names experiments: {sorted(hit)}"


def test_spec_never_names_experiments_it_should_receive_prebuilt():
    # run_spec.py may import PART modules, but experiment classes (fabric,
    # speculation) must arrive pre-built — never be named.
    tree = ast.parse((PKG / "run_spec.py").read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    hit = names & {"DecoderFabric", "FabricRouter", "Speculation"}
    assert not hit, f"run_spec.py names experiments: {sorted(hit)}"
