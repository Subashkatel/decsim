"""Behaviour and structure tests for the decsim/detector_error_model/ package.

Real Stim circuits cover the end-to-end contract: a distance-3 repetition
memory circuit is generated, its actual detector error model is derived, and
the sliced windows are checked against the circuit's own structure. Small
local fakes cover the edge cases a real circuit cannot easily produce, and are
built only from the accessor contract that the production package documents.
Every symbol is imported from the submodule that owns it, so the tests also
demonstrate that no consumer needs a package-level facade.
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import json
import math
import os
import pathlib
import subprocess
import sys
import types
from types import MappingProxyType

import pytest

numpy = pytest.importorskip("numpy")

import decsim.detector_error_model as detector_error_model_package
import decsim.detector_error_model.detector_chronology as detector_chronology
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.detector_error_model.fault_identity_validation as fault_identity_validation
import decsim.detector_error_model.fault_model_contracts as fault_model_contracts
import decsim.detector_error_model.stim_dem_catalog as stim_dem_catalog
import decsim.detector_error_model.window_model_builders as window_model_builders
import decsim.detector_error_model.window_ownership_dag as window_ownership_dag
import decsim.detector_error_model.window_placement as window_placement
import decsim.detector_error_model.window_protocol_policy as window_protocol_policy
import decsim.detector_error_model.window_slicer as window_slicer
from decsim.message import WindowProtocol


# --------------------------------------------------------------------------
# Local fakes for the Stim surface this module consumes.
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FakeDemTarget:
    """One DEM target exposing only the documented Stim accessors."""

    kind: str
    value: int = -1

    def is_separator(self) -> bool:
        return self.kind == "separator"

    def is_relative_detector_id(self) -> bool:
        return self.kind == "detector"

    def is_logical_observable_id(self) -> bool:
        return self.kind == "observable"

    @property
    def val(self) -> int:
        return self.value


@dataclasses.dataclass(frozen=True)
class FakeDemInstruction:
    """One DEM instruction exposing type, args_copy and targets_copy only."""

    instruction_type: str
    args: tuple
    targets: tuple

    @property
    def type(self) -> str:
        return self.instruction_type

    def args_copy(self) -> list:
        return list(self.args)

    def targets_copy(self) -> list:
        return list(self.targets)


class FakeDetectorErrorModel:
    """A DEM whose instructions are reachable only through flattened()."""

    def __init__(self, instructions):
        object.__setattr__(self, "instructions", tuple(instructions))
        object.__setattr__(self, "flattened_calls", [])

    def __setattr__(self, name, value):
        raise AssertionError(
            "production code must not mutate the detector error model"
        )

    def flattened(self):
        self.flattened_calls.append(True)
        return tuple(self.instructions)


class FakeCircuit:
    """A circuit exposing only the four documented circuit queries."""

    def __init__(
        self,
        *,
        detector_count,
        observable_count,
        detector_coordinates,
        decomposed_model,
        undecomposed_model=None,
    ):
        object.__setattr__(self, "_detector_count", detector_count)
        object.__setattr__(self, "_observable_count", observable_count)
        object.__setattr__(self, "_detector_coordinates", detector_coordinates)
        object.__setattr__(self, "_decomposed_model", decomposed_model)
        object.__setattr__(
            self,
            "_undecomposed_model",
            decomposed_model if undecomposed_model is None else undecomposed_model,
        )
        object.__setattr__(self, "accessed_names", [])

    def __setattr__(self, name, value):
        raise AssertionError("production code must not mutate the circuit")

    @property
    def num_detectors(self):
        self.accessed_names.append("num_detectors")
        return self._detector_count

    @property
    def num_observables(self):
        self.accessed_names.append("num_observables")
        return self._observable_count

    def get_detector_coordinates(self):
        self.accessed_names.append("get_detector_coordinates")
        return {
            detector_id: list(coordinate)
            for detector_id, coordinate in self._detector_coordinates.items()
        }

    def detector_error_model(self, *, decompose_errors):
        self.accessed_names.append("detector_error_model")
        if decompose_errors:
            return self._decomposed_model
        return self._undecomposed_model


def targets_from_text(text: str) -> tuple:
    """Build DEM targets from a compact "D0 L1 ^ D2" description."""
    targets = []
    for token in text.split():
        if token == "^":
            targets.append(FakeDemTarget("separator"))
        elif token.startswith("D"):
            targets.append(FakeDemTarget("detector", int(token[1:])))
        elif token.startswith("L"):
            targets.append(FakeDemTarget("observable", int(token[1:])))
        else:
            raise AssertionError(f"unsupported fake target token {token!r}")
    return tuple(targets)


def error_instruction(probability: float, text: str) -> FakeDemInstruction:
    return FakeDemInstruction("error", (probability,), targets_from_text(text))


def make_model(rows) -> FakeDetectorErrorModel:
    return FakeDetectorErrorModel(
        [error_instruction(probability, text) for probability, text in rows]
    )


CHAIN_ERROR_ROWS = (
    (0.1, "D0 L0"),
    (0.1, "D0 D1"),
    (0.1, "D1 D2"),
    (0.1, "D2 D3"),
    (0.1, "D3 L0"),
)
CHAIN_DETECTOR_ROUNDS = {0: 1, 1: 2, 2: 3, 3: 4}
CHAIN_ROUND_COUNT = 4


def chain_circuit() -> FakeCircuit:
    """A four-detector chain: one detector per emitted round."""
    model = make_model(CHAIN_ERROR_ROWS)
    return FakeCircuit(
        detector_count=4,
        observable_count=1,
        detector_coordinates={
            detector_id: [0.0, float(detector_id)] for detector_id in range(4)
        },
        decomposed_model=model,
    )


GRAPHLIKE_REQUIREMENT = fault_model_contracts.GRAPHLIKE_FAULT_MODEL_REQUIRED
PHYSICAL_REQUIREMENT = fault_model_contracts.PHYSICAL_FAULT_MODEL_REQUIRED
LINKED_REQUIREMENT = fault_model_contracts.LINKED_FAULT_MODELS_REQUIRED
NO_REQUIREMENT = fault_model_contracts.NO_FAULT_MODEL_REQUIRED
GRAPHLIKE = fault_model_contracts.FaultRepresentation.GRAPHLIKE
PHYSICAL = fault_model_contracts.FaultRepresentation.PHYSICAL


def chain_models(plan, **keyword_arguments):
    """Slice the chain circuit with the shared chronology and defaults."""
    keyword_arguments.setdefault("fault_model_requirement", GRAPHLIKE_REQUIREMENT)
    keyword_arguments.setdefault("fault_exclusion_ranges", ())
    return window_model_builders.build_window_error_models(
        chain_circuit(),
        plan,
        round_count=CHAIN_ROUND_COUNT,
        detector_rounds=dict(CHAIN_DETECTOR_ROUNDS),
        **keyword_arguments,
    )


# --------------------------------------------------------------------------
# Module scope and Stim access
# --------------------------------------------------------------------------


def test_module_docstring_declares_slicing_job_and_consumers():
    """The module docstring states the slicing job, the per-shot/compile-time split, its decoder consumers and its producer."""
    docstring = " ".join(detector_error_model_package.__doc__.split())
    assert "Slice a global Stim detector error model into per-window decoder inputs" in docstring
    assert "measured bits change every shot" in docstring
    assert "compile-time data shared across shots" in docstring
    assert "MWPM / BP+OSD / belief matching" in docstring
    assert "adapters/stim_device.py" in docstring


def test_stim_objects_are_read_only_and_only_documented_accessors_are_used():
    """Slicing reads the circuit only through the four documented queries and never mutates the circuit or the error model."""
    circuit = chain_circuit()
    slicer = window_slicer.WindowSlicer(
        circuit,
        round_count=CHAIN_ROUND_COUNT,
        detector_rounds=dict(CHAIN_DETECTOR_ROUNDS),
        fault_model_requirement=GRAPHLIKE_REQUIREMENT,
    )
    slicer.slice_window(1, 1, 2, 2, is_last=False)
    assert set(circuit.accessed_names) <= {
        "num_detectors",
        "num_observables",
        "get_detector_coordinates",
        "detector_error_model",
    }
    assert "num_detectors" in circuit.accessed_names
    assert "num_observables" in circuit.accessed_names
    # The fakes are non-iterable models and frozen instructions, so reaching
    # the errors at all proves flattened()/type/args_copy()/targets_copy().
    assert circuit._decomposed_model.flattened_calls
    with pytest.raises(AssertionError):
        circuit._decomposed_model.instructions = ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        circuit._decomposed_model.instructions[0].args = (0.5,)


PACKAGE_LAYERS = {
    "fault_model_contracts": 0,
    "fault_identity_validation": 0,
    "detector_chronology": 0,
    "detector_formation": 0,
    "stim_dem_catalog": 1,
    "window_placement": 1,
    "window_slicer": 2,
    "window_ownership_dag": 3,
    "window_protocol_policy": 3,
    "window_model_builders": 4,
}
PACKAGE_MODULES = {
    "fault_model_contracts": fault_model_contracts,
    "fault_identity_validation": fault_identity_validation,
    "detector_chronology": detector_chronology,
    "detector_formation": detector_formation,
    "stim_dem_catalog": stim_dem_catalog,
    "window_placement": window_placement,
    "window_slicer": window_slicer,
    "window_ownership_dag": window_ownership_dag,
    "window_protocol_policy": window_protocol_policy,
    "window_model_builders": window_model_builders,
}
PACKAGE_DIRECTORY = pathlib.Path(detector_error_model_package.__file__).parent
REPOSITORY_ROOT = PACKAGE_DIRECTORY.parent.parent
EXPECTED_LOCAL_NUMPY_IMPORTS = 8


def package_syntax_trees():
    """Parse every source file of the package once."""
    return {
        path.stem: ast.parse(path.read_text())
        for path in sorted(PACKAGE_DIRECTORY.glob("*.py"))
    }


def parents_of(tree):
    """Return a child-to-parent map for one parsed source file."""
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def test_package_init_is_a_docstring_and_exports_nothing():
    """The package __init__ holds a docstring only, with no imports, assignments or __all__."""
    tree = ast.parse((PACKAGE_DIRECTORY / "__init__.py").read_text())
    assert len(tree.body) == 1
    only_statement = tree.body[0]
    assert isinstance(only_statement, ast.Expr)
    assert isinstance(only_statement.value, ast.Constant)
    assert isinstance(only_statement.value.value, str)
    assert not hasattr(detector_error_model_package, "__all__")
    public_names = {
        name for name in vars(detector_error_model_package) if not name.startswith("__")
    }
    assert public_names <= set(PACKAGE_LAYERS)
    for name in public_names:
        assert isinstance(vars(detector_error_model_package)[name], types.ModuleType)


def test_package_modules_form_a_one_way_acyclic_layer_graph():
    """Every intra-package import points to a strictly lower layer, so the import graph cannot cycle."""
    trees = package_syntax_trees()
    assert set(trees) == set(PACKAGE_LAYERS) | {"__init__"}
    edges = set()
    outward_imports = set()
    for module_name, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level == 1 and node.module in PACKAGE_LAYERS:
                edges.add((module_name, node.module))
            elif node.level >= 2:
                outward_imports.add(
                    (module_name, node.module, tuple(alias.name for alias in node.names))
                )
    assert edges
    assert not any(importer == "__init__" for importer, _ in edges)
    for importer, imported in edges:
        assert PACKAGE_LAYERS[imported] < PACKAGE_LAYERS[importer]
    reachable = {module: {target for source, target in edges if source == module}
                 for module in PACKAGE_LAYERS}
    for _ in range(len(PACKAGE_LAYERS)):
        for module in reachable:
            for target in list(reachable[module]):
                reachable[module] |= reachable[target]
    for module, targets in reachable.items():
        assert module not in targets
    assert outward_imports == {
        ("window_protocol_policy", "message", ("WindowProtocol",)),
        ("window_model_builders", "message", ("WindowProtocol",)),
    }


def test_numpy_is_imported_only_inside_functions_and_exactly_eight_times():
    """All eight numpy imports of the package sit inside functions, and none at module scope."""
    local_numpy_imports = []
    for module_name, tree in package_syntax_trees().items():
        parents = parents_of(tree)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            if not any(name.split(".")[0] == "numpy" for name in names):
                continue
            enclosing = parents.get(node)
            while enclosing is not None and not isinstance(
                enclosing, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                enclosing = parents.get(enclosing)
            assert enclosing is not None, (module_name, node.lineno)
            local_numpy_imports.append((module_name, node.lineno))
    assert len(set(local_numpy_imports)) == EXPECTED_LOCAL_NUMPY_IMPORTS
    assert len(local_numpy_imports) == EXPECTED_LOCAL_NUMPY_IMPORTS


def test_no_package_module_binds_numpy_or_the_adapters_package():
    """No imported package module carries a numpy or adapters binding, and only the protocol modules bind the window protocol."""
    for module_name, module in PACKAGE_MODULES.items():
        assert not hasattr(module, "np"), module_name
        assert not hasattr(module, "numpy"), module_name
        assert not hasattr(module, "adapters"), module_name
    modules_binding_the_protocol = {
        module_name
        for module_name, module in PACKAGE_MODULES.items()
        if hasattr(module, "WindowProtocol")
    }
    assert modules_binding_the_protocol == {
        "window_protocol_policy",
        "window_model_builders",
    }


def test_importing_a_leaf_module_loads_no_higher_layer_and_no_numpy():
    """Importing the contracts leaf in a fresh interpreter pulls in no sibling layer and no numpy."""
    probe = (
        "import json, sys\n"
        "import decsim.detector_error_model.fault_model_contracts\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPOSITORY_ROOT), environment.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPOSITORY_ROOT),
        env=environment,
    )
    loaded = set(json.loads(completed.stdout.strip().splitlines()[-1]))
    assert "decsim.detector_error_model.fault_model_contracts" in loaded
    assert not any(name.split(".")[0] == "numpy" for name in loaded)
    package_modules_loaded = {
        name for name in loaded if name.startswith("decsim.detector_error_model")
    }
    assert package_modules_loaded == {
        "decsim.detector_error_model",
        "decsim.detector_error_model.fault_model_contracts",
    }
    for module_name in PACKAGE_LAYERS:
        if module_name == "fault_model_contracts":
            continue
        assert f"decsim.detector_error_model.{module_name}" not in loaded


# --------------------------------------------------------------------------
# Fault-domain requirement values
# --------------------------------------------------------------------------


def test_fault_representation_has_exactly_two_stable_values():
    """The fault representation enum has exactly the two stable string values graphlike and physical."""
    assert [member.value for member in fault_model_contracts.FaultRepresentation] == [
        "graphlike",
        "physical",
    ]


def test_requirement_fields_and_defaults():
    """A decoder fault-model requirement has exactly two fields and defaults to no representations and no link."""
    fields = dataclasses.fields(fault_model_contracts.DecoderFaultModelRequirement)
    assert [field.name for field in fields] == [
        "representations",
        "require_physical_to_graphlike_link",
    ]
    empty = fault_model_contracts.DecoderFaultModelRequirement()
    assert empty.representations == frozenset()
    assert empty.require_physical_to_graphlike_link is False


def test_empty_requirement_builds_no_catalog_and_no_link():
    """An empty requirement builds no fault catalog, no link and a window with neither placed view."""
    slicer = window_slicer.WindowSlicer(
        chain_circuit(),
        round_count=CHAIN_ROUND_COUNT,
        detector_rounds=dict(CHAIN_DETECTOR_ROUNDS),
        fault_model_requirement=NO_REQUIREMENT,
    )
    assert slicer.catalogs == {}
    assert slicer.catalog_link is None
    window = slicer.slice_window(1, 1, 2, 2, is_last=False)
    assert window.graphlike_faults is None
    assert window.physical_faults is None
    assert window.physical_to_graphlike_detector_projection is None


@pytest.mark.parametrize(
    "representations",
    [
        frozenset(),
        frozenset({GRAPHLIKE}),
        frozenset({PHYSICAL}),
    ],
)
def test_link_requires_both_representations(representations):
    """Asking for the physical-to-graphlike link without both representations is rejected at construction."""
    with pytest.raises(ValueError) as failure:
        fault_model_contracts.DecoderFaultModelRequirement(
            representations, require_physical_to_graphlike_link=True
        )
    assert "requires both fault representations" in str(failure.value)


def test_link_with_both_representations_is_accepted():
    """Asking for the link together with both representations is accepted."""
    requirement = fault_model_contracts.DecoderFaultModelRequirement(
        frozenset({GRAPHLIKE, PHYSICAL}), require_physical_to_graphlike_link=True
    )
    assert requirement.require_physical_to_graphlike_link is True


def test_requirement_is_frozen():
    """A requirement value is frozen, so assigning to a field raises."""
    requirement = fault_model_contracts.DecoderFaultModelRequirement()
    with pytest.raises(dataclasses.FrozenInstanceError):
        requirement.require_physical_to_graphlike_link = True
    assert isinstance(dataclasses.FrozenInstanceError(), AttributeError)


def test_joined_is_union_and_logical_or():
    """Joining two requirements unions the representations and ors the link flag."""
    joined = GRAPHLIKE_REQUIREMENT.joined(PHYSICAL_REQUIREMENT)
    assert joined.representations == frozenset({GRAPHLIKE, PHYSICAL})
    assert joined.require_physical_to_graphlike_link is False
    assert LINKED_REQUIREMENT.joined(NO_REQUIREMENT).require_physical_to_graphlike_link


def test_four_requirement_singletons_have_exact_values():
    """The four shared requirement singletons carry exactly the declared representations and link flags."""
    assert NO_REQUIREMENT == fault_model_contracts.DecoderFaultModelRequirement()
    assert GRAPHLIKE_REQUIREMENT.representations == frozenset({GRAPHLIKE})
    assert GRAPHLIKE_REQUIREMENT.require_physical_to_graphlike_link is False
    assert PHYSICAL_REQUIREMENT.representations == frozenset({PHYSICAL})
    assert PHYSICAL_REQUIREMENT.require_physical_to_graphlike_link is False
    assert LINKED_REQUIREMENT.representations == frozenset({GRAPHLIKE, PHYSICAL})
    assert LINKED_REQUIREMENT.require_physical_to_graphlike_link is True


# --------------------------------------------------------------------------
# Parity semantics of one error instruction
# --------------------------------------------------------------------------


def test_canonical_records_are_frozen_with_both_identities():
    """The canonical error records are frozen and carry both the instruction-wide identity and its components."""
    component_fields = [
        field.name for field in dataclasses.fields(stim_dem_catalog.CanonicalErrorComponent)
    ]
    assert component_fields == [
        "component_ordinal",
        "detectors",
        "logical_observables",
    ]
    instruction_fields = [
        field.name
        for field in dataclasses.fields(stim_dem_catalog.CanonicalErrorInstruction)
    ]
    assert instruction_fields == [
        "error_ordinal",
        "probability",
        "aggregate_detectors",
        "aggregate_logical_observables",
        "components",
    ]
    component = stim_dem_catalog.CanonicalErrorComponent(0, (1,), ())
    with pytest.raises(dataclasses.FrozenInstanceError):
        component.detectors = ()


def test_merge_probability_uses_the_independent_fault_rule():
    """Two independent faults with the same identity merge with p(1-q) + q(1-p)."""
    assert stim_dem_catalog._merge_probability(0.1, 0.2) == pytest.approx(
        0.1 * 0.8 + 0.2 * 0.9
    )
    assert stim_dem_catalog._merge_probability(0.0, 0.3) == pytest.approx(0.3)


def test_merged_probabilities_are_not_domain_checked():
    """Probability merging applies the arithmetic rule without complaining about values outside [0, 1]."""
    assert stim_dem_catalog._merge_probability(2.0, 3.0) == pytest.approx(
        2.0 * (1 - 3.0) + 3.0 * (1 - 2.0)
    )
    assert stim_dem_catalog._merge_probability(-1.0, 0.5) == pytest.approx(-1.0 * 0.5 + 0.5 * 2.0)


def test_xor_reduction_drops_even_multiplicities_and_sorts():
    """Identity canonicalisation drops ids occurring an even number of times and returns the rest sorted."""
    assert fault_identity_validation._xor_target_ids([5, 1, 5, 3, 3, 3]) == (1, 3)
    assert fault_identity_validation._xor_target_ids([]) == ()
    assert fault_identity_validation._xor_target_ids([2, 2]) == ()


def test_one_instruction_is_parsed_before_components_are_exposed():
    """One error instruction is reduced modulo two across separators before its components are exposed."""
    model = make_model([(0.25, "D2 L0 ^ D3 L0")])
    (record,) = stim_dem_catalog.canonical_error_instructions(model)
    assert record.error_ordinal == 0
    assert record.probability == 0.25
    assert record.aggregate_detectors == (2, 3)
    # The two L0 targets cancel across the separator.
    assert record.aggregate_logical_observables == ()
    assert [component.detectors for component in record.components] == [(2,), (3,)]
    assert [component.component_ordinal for component in record.components] == [0, 1]
    assert [component.logical_observables for component in record.components] == [
        (0,),
        (0,),
    ]


def test_probability_is_taken_from_the_first_instruction_argument():
    """An error's probability is the first instruction argument, as a float."""
    instruction = FakeDemInstruction("error", (0.5, 9.0), targets_from_text("D0 D1"))
    model = FakeDetectorErrorModel([instruction])
    (record,) = stim_dem_catalog.canonical_error_instructions(model)
    assert record.probability == 0.5
    assert isinstance(record.probability, float)


def test_fully_inert_component_is_dropped_silently():
    """A component that reduces to nothing at all is dropped without complaint."""
    model = make_model([(0.1, "D0 D1 ^ D2 D2")])
    (record,) = stim_dem_catalog.canonical_error_instructions(model)
    assert [component.detectors for component in record.components] == [(0, 1)]


def test_detectorless_aggregate_returns_none_or_raises():
    """An instruction with no surviving detectors is skipped, unless it still flips observables, which raises."""
    inert = make_model([(0.1, "D0 D0")])
    assert stim_dem_catalog.canonical_error_instructions(inert) == ()
    with pytest.raises(ValueError) as failure:
        stim_dem_catalog.canonical_error_instructions(make_model([(0.1, "D0 D0 L1")]))
    message = str(failure.value)
    assert "detectorless logical" in message
    assert "error 0" in message


def test_detectorless_component_raises_with_component_ordinal():
    """A detectorless component that still flips observables raises and names the offending component."""
    with pytest.raises(ValueError) as failure:
        stim_dem_catalog.canonical_error_instructions(
            make_model([(0.1, "D0 D1 ^ L1")])
        )
    message = str(failure.value)
    assert "component 1" in message
    assert "detectorless logical" in message


def test_error_ordinal_counts_every_error_instruction():
    """Non-error instructions are skipped while the error ordinal advances once per error instruction."""
    model = FakeDetectorErrorModel(
        [
            FakeDemInstruction("detector", (1.0,), targets_from_text("D0")),
            error_instruction(0.1, "D0 D0"),
            error_instruction(0.2, "D1 D2"),
            FakeDemInstruction("shift_detectors", (1.0,), ()),
            error_instruction(0.3, "D2 D3"),
        ]
    )
    records = stim_dem_catalog.canonical_error_instructions(model)
    assert [record.error_ordinal for record in records] == [1, 2]
    assert [record.aggregate_detectors for record in records] == [(1, 2), (2, 3)]


def test_unknown_target_kinds_are_ignored():
    """A target that is neither detector, observable nor separator is ignored silently."""
    instruction = FakeDemInstruction(
        "error",
        (0.2,),
        (FakeDemTarget("sweep", 4),) + targets_from_text("D1 D2"),
    )
    (record,) = stim_dem_catalog.canonical_error_instructions(
        FakeDetectorErrorModel([instruction])
    )
    assert record.aggregate_detectors == (1, 2)


# --------------------------------------------------------------------------
# Global fault catalogs
# --------------------------------------------------------------------------


def test_fault_catalog_is_a_frozen_parallel_record():
    """A fault catalog is a frozen record whose detector, observable and prior tuples stay index aligned."""
    fields = [field.name for field in dataclasses.fields(fault_model_contracts._FaultCatalog)]
    assert fields == [
        "representation",
        "detector_sets",
        "observable_sets",
        "priors",
    ]
    catalog = stim_dem_catalog._catalog_from_dem(make_model(CHAIN_ERROR_ROWS), GRAPHLIKE)
    assert len(catalog.detector_sets) == len(catalog.observable_sets)
    assert len(catalog.detector_sets) == len(catalog.priors)
    with pytest.raises(dataclasses.FrozenInstanceError):
        catalog.priors = ()


def test_equal_components_cancel_within_one_instruction():
    """Equal components of one instruction cancel modulo two before any probability merging."""
    model = make_model([(0.1, "D0 D1 ^ D0 D1 ^ D2 D3")])
    (record,) = stim_dem_catalog.canonical_error_instructions(model)
    keys = stim_dem_catalog._odd_component_keys(
        record, fault_identity_validation.validate_fault_identity
    )
    assert keys == (((2, 3), ()),)


def test_odd_component_keys_uses_the_injected_validator():
    """The per-component canonicaliser is injected, so the same reduction serves the plain and the degree-checked domain."""
    (record,) = stim_dem_catalog.canonical_error_instructions(
        make_model([(0.1, "D0 D1 D2")])
    )
    assert stim_dem_catalog._odd_component_keys(
        record, fault_identity_validation.validate_fault_identity
    ) == (((0, 1, 2), ()),)
    with pytest.raises(ValueError) as failure:
        stim_dem_catalog._odd_component_keys(record, fault_identity_validation.validate_graphlike_fault)
    assert "detector hyperedge" in str(failure.value)


def test_detector_error_model_to_faults_merges_component_identities():
    """Component identities repeated across instructions are merged into one column in first-seen order."""
    model = make_model(
        [(0.1, "D0 D1"), (0.2, "D0 D1"), (0.3, "D1 D2 ^ D2 D3")]
    )
    detector_sets, observable_sets, priors = stim_dem_catalog.detector_error_model_to_faults(
        model
    )
    assert detector_sets == [(0, 1), (1, 2), (2, 3)]
    assert observable_sets == [(), (), ()]
    assert priors[0] == pytest.approx(stim_dem_catalog._merge_probability(0.1, 0.2))
    assert priors[1] == pytest.approx(0.3)
    assert priors[2] == pytest.approx(0.3)


def test_detector_error_model_to_faults_applies_no_degree_bound():
    """Building fault columns from components applies no detector-degree bound."""
    detector_sets, _, _ = stim_dem_catalog.detector_error_model_to_faults(
        make_model([(0.1, "D0 D1 D2")])
    )
    assert detector_sets == [(0, 1, 2)]


def test_graphlike_and_physical_catalogs_key_columns_differently():
    """The graphlike catalog keys columns by component identity while the physical catalog keys them by instruction-wide identity."""
    model = make_model([(0.2, "D1 D2 ^ D2 D3")])
    graphlike = stim_dem_catalog._catalog_from_dem(model, GRAPHLIKE)
    physical = stim_dem_catalog._catalog_from_dem(model, PHYSICAL)
    assert graphlike.detector_sets == ((1, 2), (2, 3))
    assert physical.detector_sets == ((1, 3),)
    assert graphlike.representation is GRAPHLIKE
    assert physical.representation is PHYSICAL
    assert all(isinstance(prior, float) for prior in physical.priors)


def test_graphlike_catalog_separates_equal_detectors_with_different_observables():
    """Two faults with the same detectors but different observables stay separate catalog columns."""
    catalog = stim_dem_catalog._catalog_from_dem(
        make_model([(0.1, "D0 D1"), (0.2, "D0 D1 L0")]), GRAPHLIKE
    )
    assert catalog.detector_sets == ((0, 1), (0, 1))
    assert catalog.observable_sets == ((), (0,))
    assert catalog.priors == (0.1, 0.2)


# --------------------------------------------------------------------------
# Fault and matrix validation
# --------------------------------------------------------------------------


def test_validate_fault_identity_canonicalises_and_guards_logical_loss():
    """Fault canonicalisation returns sorted odd ids, drops an inert fault and rejects a detectorless logical fault."""
    assert fault_identity_validation.validate_fault_identity(
        [3, 1, 3], [2, 2, 5], location="site"
    ) == ((1,), (5,))
    assert fault_identity_validation.validate_fault_identity([4, 4], [], location="site") is None
    with pytest.raises(ValueError) as failure:
        fault_identity_validation.validate_fault_identity([4, 4], [7], location="site")
    assert "site is a detectorless logical fault" in str(failure.value)


def test_validate_graphlike_fault_bounds_the_detector_degree():
    """A graphlike fault may touch at most two detectors, and a third detector is named as a hyperedge."""
    assert fault_identity_validation.validate_graphlike_fault(
        [1, 0], [], location="site"
    ) == ((0, 1), ())
    assert fault_identity_validation.validate_graphlike_fault([2, 2], [], location="site") is None
    with pytest.raises(ValueError) as failure:
        fault_identity_validation.validate_graphlike_fault([0, 1, 2], [], location="site")
    message = str(failure.value)
    assert "site is a detector hyperedge" in message
    assert "one or two detectors per fault" in message


def test_binary_matrix_normalisation():
    """A matrix argument must be rank two and binary, and is normalised to unsigned bytes, with the location in the message."""
    matrix = fault_identity_validation._binary_matrix([[0, 1], [1, 0]], location="site", name="check")
    assert matrix.dtype == numpy.uint8
    with pytest.raises(ValueError) as rank_failure:
        fault_identity_validation._binary_matrix([0, 1], location="site", name="check")
    assert "site check must be a rank-2 matrix" in str(rank_failure.value)
    with pytest.raises(ValueError) as value_failure:
        fault_identity_validation._binary_matrix([[0, 2]], location="site", name="check")
    assert "site check must contain only binary values" in str(value_failure.value)


def test_placed_matrix_faults_returns_plain_int_ids():
    """Each matrix column yields its detector and observable ids as plain integers."""
    faults = fault_identity_validation._placed_matrix_faults(
        [[1, 0], [1, 1]], [[0, 1], [0, 0]], location="site"
    )
    assert faults == ((0, (0, 1), ()), (1, (1,), (0,)))
    for _, detector_ids, logical_ids in faults:
        assert all(type(value) is int for value in detector_ids + logical_ids)


def test_placed_matrix_faults_requires_equal_column_counts():
    """Check and observable matrices must have the same number of fault columns."""
    with pytest.raises(ValueError) as failure:
        fault_identity_validation._placed_matrix_faults(
            [[1, 0]], [[1, 0, 1]], location="site"
        )
    message = str(failure.value)
    assert "site check and observable matrices have different fault counts" in message
    assert "2 and 3" in message


def test_validate_placed_fault_matrices_allows_any_degree_but_not_logical_loss():
    """A placed physical model may use any detector degree but not lose a column's logical identity."""
    fault_identity_validation.validate_placed_fault_matrices(
        [[1], [1], [1]], [[0], [0], [0]], location="physical"
    )
    with pytest.raises(ValueError) as failure:
        fault_identity_validation.validate_placed_fault_matrices(
            [[0], [0]], [[1], [0]], location="physical"
        )
    assert "physical column 0 is a detectorless logical fault" in str(failure.value)


def test_validate_graphlike_matrices_adds_the_degree_bound_per_column():
    """A placed graphlike model additionally bounds every column to at most two detectors, naming the column."""
    fault_identity_validation.validate_graphlike_matrices(
        [[1, 0], [1, 1]], [[0, 0], [0, 0]], location="graph"
    )
    with pytest.raises(ValueError) as failure:
        fault_identity_validation.validate_graphlike_matrices(
            [[0, 1], [0, 1], [0, 1]], [[0, 0], [0, 0], [0, 0]], location="graph"
        )
    assert "graph column 1 is a detector hyperedge" in str(failure.value)


# --------------------------------------------------------------------------
# Belief matching and the linked catalogs
# --------------------------------------------------------------------------


LINKED_DECOMPOSED_ROWS = ((0.1, "D0 D1"), (0.2, "D1 D2 ^ D2 D3"))
LINKED_UNDECOMPOSED_ROWS = ((0.1, "D0 D1"), (0.2, "D1 D3"))


def linked_circuit(
    decomposed_rows=LINKED_DECOMPOSED_ROWS,
    undecomposed_rows=LINKED_UNDECOMPOSED_ROWS,
) -> FakeCircuit:
    return FakeCircuit(
        detector_count=4,
        observable_count=1,
        detector_coordinates={
            detector_id: [0.0, float(detector_id)] for detector_id in range(4)
        },
        decomposed_model=make_model(decomposed_rows),
        undecomposed_model=make_model(undecomposed_rows),
    )


def belief_matching_case():
    """A small consistent (component, physical, map, priors) bundle."""
    component_check = numpy.array([[1, 0], [1, 1], [0, 1]], dtype=numpy.uint8)
    component_observables = numpy.array([[0, 0]], dtype=numpy.uint8)
    physical_check = numpy.array([[1], [0], [1]], dtype=numpy.uint8)
    physical_priors = numpy.array([0.2])
    physical_to_component = numpy.array([[1], [1]], dtype=numpy.uint8)
    return (
        component_check,
        component_observables,
        physical_check,
        physical_priors,
        physical_to_component,
    )


def test_belief_matching_accepts_a_consistent_bundle():
    """A consistent component, physical, prior and decomposition-map bundle passes belief-matching validation."""
    fault_identity_validation.validate_belief_matching_matrices(
        *belief_matching_case(), location="site"
    )


def test_belief_matching_requires_equal_detector_rows_and_map_shape():
    """The two check matrices must share detector rows and the decomposition map must have the exact expected shape."""
    check, observables, physical_check, priors, mapping = belief_matching_case()
    with pytest.raises(ValueError) as row_failure:
        fault_identity_validation.validate_belief_matching_matrices(
            check, observables, physical_check[:2], priors, mapping, location="site"
        )
    assert "different detector counts" in str(row_failure.value)
    with pytest.raises(ValueError) as shape_failure:
        fault_identity_validation.validate_belief_matching_matrices(
            check,
            observables,
            physical_check,
            priors,
            numpy.array([[1, 0], [1, 0]], dtype=numpy.uint8),
            location="site",
        )
    message = str(shape_failure.value)
    assert "physical-to-component map has shape" in message
    assert "expected (2, 1)" in message


@pytest.mark.parametrize(
    "bad_priors, expected",
    [
        (numpy.array([[0.2]]), "physical priors must be rank 1"),
        (numpy.array([0.2, 0.3]), "2 physical priors for 1 physical fault columns"),
        (numpy.array([float("nan")]), "physical priors must be finite"),
        (numpy.array([1.5]), "inclusive range [0, 1]"),
        (numpy.array([-0.1]), "inclusive range [0, 1]"),
    ],
)
def test_belief_matching_prior_domain(bad_priors, expected):
    """Physical priors must be a rank-one vector of the right length, finite and inside [0, 1]."""
    check, observables, physical_check, _, mapping = belief_matching_case()
    with pytest.raises(ValueError) as failure:
        fault_identity_validation.validate_belief_matching_matrices(
            check, observables, physical_check, bad_priors, mapping, location="site"
        )
    assert expected in str(failure.value)


def test_belief_matching_validates_components_as_a_graphlike_model():
    """The component matrices are additionally validated as a graphlike model."""
    component_check = numpy.array([[1], [1], [1]], dtype=numpy.uint8)
    component_observables = numpy.array([[0]], dtype=numpy.uint8)
    physical_check = numpy.array([[1], [1], [1]], dtype=numpy.uint8)
    with pytest.raises(ValueError) as failure:
        fault_identity_validation.validate_belief_matching_matrices(
            component_check,
            component_observables,
            physical_check,
            numpy.array([0.1]),
            numpy.array([[1]], dtype=numpy.uint8),
            location="site",
        )
    message = str(failure.value)
    assert "site component graph column 0 is a detector hyperedge" in message


def test_belief_matching_requires_exact_component_xor_identity():
    """Every physical column's detector identity must equal the parity of its components."""
    check, observables, physical_check, priors, mapping = belief_matching_case()
    wrong_physical_check = numpy.array([[1], [1], [1]], dtype=numpy.uint8)
    with pytest.raises(ValueError) as failure:
        fault_identity_validation.validate_belief_matching_matrices(
            check, observables, wrong_physical_check, priors, mapping, location="site"
        )
    message = str(failure.value)
    assert "physical column 0 detector identity does not equal its component XOR" in message


def test_belief_matching_rejects_an_inert_physical_column():
    """A physical column that reduces to nothing must be removed before belief matching."""
    component_check = numpy.array([[1, 1], [1, 1]], dtype=numpy.uint8)
    component_observables = numpy.array([[0, 0]], dtype=numpy.uint8)
    physical_check = numpy.array([[0], [0]], dtype=numpy.uint8)
    with pytest.raises(ValueError) as failure:
        fault_identity_validation.validate_belief_matching_matrices(
            component_check,
            component_observables,
            physical_check,
            numpy.array([0.1]),
            numpy.array([[1], [1]], dtype=numpy.uint8),
            location="site",
        )
    assert "physical column 0 is inert" in str(failure.value)


def test_linked_catalogs_build_a_verified_link():
    """The linked build returns a graphlike catalog, a physical catalog and the verified incidence matrix between them."""
    graphlike, physical, link = stim_dem_catalog._prepare_linked_fault_catalogs(
        make_model(LINKED_DECOMPOSED_ROWS), make_model(LINKED_UNDECOMPOSED_ROWS)
    )
    assert graphlike.detector_sets == ((0, 1), (1, 2), (2, 3))
    assert physical.detector_sets == ((0, 1), (1, 3))
    assert physical.representation is PHYSICAL
    assert link.shape == (3, 2)
    assert link.toarray().tolist() == [[1, 0], [0, 1], [0, 1]]


def test_linked_catalogs_retain_a_maximum_detector_canceled_from_the_aggregate():
    """Linked catalogs retain a maximum component detector even when it cancels out of the physical aggregate."""
    graphlike, physical, link = stim_dem_catalog._prepare_linked_fault_catalogs(
        make_model([(0.25, "D1 D4 ^ D3 D4")]),
        make_model([(0.25, "D1 D3")]),
    )
    assert graphlike == fault_model_contracts._FaultCatalog(
        representation=GRAPHLIKE,
        detector_sets=((1, 4), (3, 4)),
        observable_sets=((), ()),
        priors=(0.25, 0.25),
    )
    assert physical == fault_model_contracts._FaultCatalog(
        representation=PHYSICAL,
        detector_sets=((1, 3),),
        observable_sets=((),),
        priors=(0.25,),
    )
    assert link.dtype == numpy.uint8
    assert link.toarray().tolist() == [[1], [1]]
    assert max(detector for detectors in graphlike.detector_sets for detector in detectors) == 4
    assert all(4 not in detectors for detectors in physical.detector_sets)


def test_linked_catalogs_keep_distinct_decompositions_of_one_aggregate():
    """Two mechanisms with the same overall identity but different decompositions stay separate physical columns."""
    merged = stim_dem_catalog._merge_probability(0.2, 0.3)
    _, physical, link = stim_dem_catalog._prepare_linked_fault_catalogs(
        make_model([(0.2, "D1 D2 ^ D2 D3"), (0.3, "D1 D0 ^ D0 D3")]),
        make_model([(merged, "D1 D3")]),
    )
    assert physical.detector_sets == ((1, 3), (1, 3))
    assert physical.priors == (0.2, 0.3)
    assert link.shape[1] == 2
    assert link.toarray().tolist() == [[1, 0], [1, 0], [0, 1], [0, 1]]


def test_linked_catalogs_reject_disagreeing_physical_probabilities():
    """Decomposed and undecomposed views that disagree on a physical probability are rejected."""
    with pytest.raises(ValueError) as failure:
        stim_dem_catalog._prepare_linked_fault_catalogs(
            make_model(LINKED_DECOMPOSED_ROWS),
            make_model(((0.1, "D0 D1"), (0.2 + 1e-9, "D1 D3"))),
        )
    assert "decomposed and undecomposed Stim models disagree" in str(failure.value)


def test_linked_catalogs_reject_a_missing_physical_key():
    """Decomposed and undecomposed views that disagree on which physical faults exist are rejected."""
    with pytest.raises(ValueError) as failure:
        stim_dem_catalog._prepare_linked_fault_catalogs(
            make_model(LINKED_DECOMPOSED_ROWS),
            make_model(((0.1, "D0 D1"), (0.2, "D1 D2"))),
        )
    assert "disagree on physical faults" in str(failure.value)


def test_linked_catalog_tolerance_is_a_fixed_absolute_bound():
    """Physical probability agreement uses a fixed absolute tolerance that does not scale with the value."""
    perturbed = 0.2 + 5e-16
    assert perturbed != 0.2
    assert abs(perturbed - 0.2) <= 1e-15
    stim_dem_catalog._prepare_linked_fault_catalogs(
        make_model(LINKED_DECOMPOSED_ROWS),
        make_model(((0.1, "D0 D1"), (perturbed, "D1 D3"))),
    )
    assert math.isclose(perturbed, 0.2, rel_tol=0, abs_tol=1e-15)
    assert not math.isclose(0.2 + 1e-9, 0.2, rel_tol=0, abs_tol=1e-15)


def test_linked_path_rejects_a_hyperedge_component():
    """The linked path rejects a component that touches more than two detectors."""
    with pytest.raises(ValueError) as failure:
        stim_dem_catalog._prepare_linked_fault_catalogs(
            make_model([(0.1, "D0 D1 D2")]), make_model([(0.1, "D0 D1 D2")])
        )
    assert "detector hyperedge" in str(failure.value)


def test_prepare_fault_catalogs_builds_only_requested_domains():
    """Only the requested fault domains are built, and the link exists only when it was asked for."""
    circuit = linked_circuit()
    graphlike_only, no_link = stim_dem_catalog._prepare_fault_catalogs(
        circuit, GRAPHLIKE_REQUIREMENT
    )
    assert set(graphlike_only) == {GRAPHLIKE}
    assert no_link is None
    physical_only, still_no_link = stim_dem_catalog._prepare_fault_catalogs(
        circuit, PHYSICAL_REQUIREMENT
    )
    assert set(physical_only) == {PHYSICAL}
    assert still_no_link is None
    both, link = stim_dem_catalog._prepare_fault_catalogs(circuit, LINKED_REQUIREMENT)
    assert set(both) == {GRAPHLIKE, PHYSICAL}
    assert link is not None
    empty, empty_link = stim_dem_catalog._prepare_fault_catalogs(circuit, NO_REQUIREMENT)
    assert empty == {}
    assert empty_link is None


def test_local_projection_is_the_verified_detector_row_slice():
    """The local link is the slice of the global link for this window's columns and reproduces the local physical detector rows."""
    slicer = window_slicer.WindowSlicer(
        linked_circuit(),
        round_count=4,
        detector_rounds={0: 1, 1: 2, 2: 3, 3: 4},
        fault_model_requirement=LINKED_REQUIREMENT,
    )
    window = slicer.slice_window(1, 1, 4, 4, is_last=True)
    graphlike = window.require_faults(GRAPHLIKE)
    physical = window.require_faults(PHYSICAL)
    projection = window.physical_to_graphlike_detector_projection
    expected = slicer.catalog_link.toarray()[
        numpy.ix_(graphlike.source_fault_ids, physical.source_fault_ids)
    ]
    assert projection.toarray().tolist() == expected.tolist()
    assert projection.shape == (
        graphlike.check.shape[1],
        physical.check.shape[1],
    )
    derived = (graphlike.check.toarray().astype(numpy.uint64)
               @ projection.toarray().astype(numpy.uint64)) % 2
    assert numpy.array_equal(derived, physical.check.toarray())


# --------------------------------------------------------------------------
# Detector chronology
# --------------------------------------------------------------------------


def coordinate_circuit(coordinates, detector_count=None):
    return FakeCircuit(
        detector_count=len(coordinates) if detector_count is None else detector_count,
        observable_count=1,
        detector_coordinates=coordinates,
        decomposed_model=make_model([(0.1, "D0 D1")]),
    )


def test_round_count_must_be_positive():
    """Resolving chronology requires a positive round count."""
    with pytest.raises(ValueError) as failure:
        detector_chronology.resolve_detector_rounds(chain_circuit(), CHAIN_DETECTOR_ROUNDS, 0)
    assert "round_count must be positive" in str(failure.value)


def test_a_detector_free_source_is_still_refused():
    """Empty detector sources remain invalid even though individual rounds may be empty."""
    circuit = coordinate_circuit({}, detector_count=0)
    for detector_rounds in ({}, None):
        with pytest.raises(ValueError) as failure:
            detector_chronology.resolve_detector_rounds(circuit, detector_rounds, 1)
        assert "requires at least one detector" in str(failure.value)


def test_explicit_map_is_shallow_copied_and_not_transformed():
    """An explicit detector-round map is copied and used verbatim, with no coordinate transformation."""
    supplied = dict(CHAIN_DETECTOR_ROUNDS)
    resolved = detector_chronology.resolve_detector_rounds(
        chain_circuit(), supplied, CHAIN_ROUND_COUNT
    )
    assert resolved == CHAIN_DETECTOR_ROUNDS
    assert resolved is not supplied
    supplied[0] = 4
    assert resolved[0] == 1


def test_explicit_map_must_cover_detectors_and_stay_inside_the_rounds():
    """An explicit map may skip a round but must cover every detector within bounds."""
    with pytest.raises(ValueError) as coverage_failure:
        detector_chronology.resolve_detector_rounds(
            chain_circuit(), {0: 1, 1: 2, 2: 3}, CHAIN_ROUND_COUNT
        )
    assert "must cover every detector exactly" in str(coverage_failure.value)

    map_with_middle_gap = {0: 1, 1: 1, 2: 3, 3: 4}
    assert detector_chronology.resolve_detector_rounds(
        chain_circuit(), map_with_middle_gap, CHAIN_ROUND_COUNT
    ) == map_with_middle_gap

    for out_of_bounds_round in (0, CHAIN_ROUND_COUNT + 1):
        invalid_map = dict(CHAIN_DETECTOR_ROUNDS)
        invalid_map[0] = out_of_bounds_round
        with pytest.raises(ValueError) as bounds_failure:
            detector_chronology.resolve_detector_rounds(
                chain_circuit(), invalid_map, CHAIN_ROUND_COUNT
            )
        assert "must lie inside the emitted rounds" in str(bounds_failure.value)


def test_coordinate_layers_are_folded_into_one_based_rounds():
    """Without a map, the last coordinate component is folded into one-based emitted rounds."""
    circuit = coordinate_circuit(
        {0: [0.0, 0.0], 1: [0.0, 1.0], 2: [0.0, 2.0], 3: [0.0, 3.0], 4: [1.0, 3.0]}
    )
    assert detector_chronology.resolve_detector_rounds(circuit, None, 3) == {
        0: 1,
        1: 2,
        2: 3,
        3: 3,
        4: 3,
    }


def test_surface_style_three_dimensional_coordinates_are_accepted():
    """Three-dimensional detector coordinates are accepted for chronology recovery."""
    circuit = coordinate_circuit(
        {0: [1.0, 1.0, 0.0], 1: [1.0, 1.0, 1.0], 2: [2.0, 1.0, 1.0]}
    )
    assert detector_chronology.resolve_detector_rounds(circuit, None, 1) == {0: 1, 1: 1, 2: 1}


def test_mixed_or_short_coordinate_arities_are_refused():
    """Chronology recovery refuses mixed coordinate arities and coordinates too short to carry a time layer."""
    mixed = coordinate_circuit({0: [0.0, 0.0], 1: [0.0, 0.0, 1.0]})
    with pytest.raises(ValueError) as arity_failure:
        detector_chronology.resolve_detector_rounds(mixed, None, 1)
    assert "need one arity" in str(arity_failure.value)
    short = coordinate_circuit({0: [0.0], 1: [1.0]})
    with pytest.raises(ValueError) as short_failure:
        detector_chronology.resolve_detector_rounds(short, None, 1)
    assert "requires supported coordinates or explicit detector_rounds" in str(
        short_failure.value
    )


@pytest.mark.parametrize("layer", [1.5, float("inf"), float("nan")])
def test_raw_layers_must_be_finite_integers(layer):
    """A raw coordinate layer that is not a finite integer is refused."""
    circuit = coordinate_circuit({0: [0.0, 0.0], 1: [0.0, layer]})
    with pytest.raises(ValueError) as failure:
        detector_chronology.resolve_detector_rounds(circuit, None, 1)
    assert "must be finite integers" in str(failure.value)
    assert math.isfinite(0.0)


def test_raw_layers_may_skip_a_round_but_must_stay_inside_the_duration():
    """Coordinate layers may leave a middle round empty but must remain bounded."""
    circuit_with_middle_gap = coordinate_circuit(
        {0: [0.0, 0.0], 1: [0.0, 2.0]}
    )
    assert detector_chronology.resolve_detector_rounds(
        circuit_with_middle_gap, None, 3
    ) == {0: 1, 1: 3}

    for out_of_bounds_layer in (-1.0, 4.0):
        invalid_circuit = coordinate_circuit(
            {0: [0.0, 0.0], 1: [0.0, out_of_bounds_layer]}
        )
        with pytest.raises(ValueError) as failure:
            detector_chronology.resolve_detector_rounds(invalid_circuit, None, 3)
        assert "must lie inside the declared source duration" in str(failure.value)


def test_detector_position_in_round_uses_ascending_detector_id():
    """A detector's position inside its round follows ascending detector id."""
    positions = detector_chronology._detector_position_in_round({2: 1, 0: 1, 1: 2, 3: 2})
    assert positions == {0: 0, 2: 1, 1: 0, 3: 1}


def test_a_detector_free_round_keeps_addresses_and_committed_coverage():
    """A gapped chronology emits an empty model without losing or inventing detectors."""
    detector_rounds = {0: 1, 1: 1, 2: 3, 3: 4}
    slicer = window_slicer.WindowSlicer(
        chain_circuit(),
        round_count=CHAIN_ROUND_COUNT,
        detector_rounds=detector_rounds,
        fault_model_requirement=GRAPHLIKE_REQUIREMENT,
    )
    windows = (
        slicer.slice_window(1, 1, 1, 1, is_last=False),
        slicer.slice_window(2, 2, 2, 2, is_last=False),
        slicer.slice_window(3, 3, 4, 4, is_last=True),
    )

    expected_addresses = {
        0: (1, 0),
        1: (1, 1),
        2: (3, 0),
        3: (4, 0),
    }
    assert slicer.round_of == detector_rounds
    assert {
        detector_id: (slicer.round_of[detector_id], slicer.pos_of[detector_id])
        for detector_id in detector_rounds
    } == expected_addresses
    assert windows[1].detector_ids == ()
    assert windows[1].detector_coordinates == ()
    empty_faults = windows[1].require_faults(GRAPHLIKE)
    assert empty_faults.check.shape == (0, 0)
    assert empty_faults.observables.shape == (1, 0)
    assert empty_faults.priors.shape == (0,)
    assert empty_faults.owned.shape == (0,)

    committed_detector_ids = tuple(
        detector_id
        for window in windows
        for detector_id in window.detector_ids
    )
    assert committed_detector_ids == (0, 1, 2, 3)
    assert len(committed_detector_ids) == len(set(committed_detector_ids))
    for window in windows:
        for detector_id in window.detector_ids:
            assert window.defect_positions[detector_id] == expected_addresses[detector_id]


def test_coordinates_for_rows_are_all_or_nothing():
    """Window coordinates are returned only when every row has coordinates, otherwise nothing."""
    complete = {0: [0.0, 0.0], 1: [1.0, 2.0]}
    assert detector_chronology._coordinates_for_rows(complete, [0, 1]) == (
        (0.0, 0.0),
        (1.0, 2.0),
    )
    partial = {0: [0.0, 0.0], 1: []}
    assert detector_chronology._coordinates_for_rows(partial, [0, 1]) is None
    assert detector_chronology._coordinates_for_rows(partial, [0]) == ((0.0, 0.0),)
    assert detector_chronology._coordinates_for_rows(partial, [0, 7]) is None


def test_coordinates_for_rows_are_normalized_to_float_tuples():
    """Row coordinates are copied out of the mapping as nested tuples of built-in floats."""
    integer_valued = {0: [1, 2, 3], 1: (4, 5, 6)}
    normalized = detector_chronology._coordinates_for_rows(integer_valued, [1, 0])
    assert normalized == ((4.0, 5.0, 6.0), (1.0, 2.0, 3.0))
    assert type(normalized) is tuple
    assert all(type(row) is tuple for row in normalized)
    assert all(type(value) is float for row in normalized for value in row)


# --------------------------------------------------------------------------
# Placed window geometry
# --------------------------------------------------------------------------


def make_placed_model():
    return fault_model_contracts.PlacedFaultModel(
        representation=GRAPHLIKE,
        check=numpy.array([[1, 0], [1, 1]], dtype=numpy.uint8),
        priors=numpy.array([0.1, 0.2]),
        observables=numpy.array([[1, 0]], dtype=numpy.uint8),
        owned=numpy.array([True, False]),
        source_fault_ids=[7, 9],
        boundary_flips={0: [0, 1]},
    )


def test_placed_fault_model_fields_and_docstring():
    """A placed fault model carries the representation, matrices, ownership mask, handoff map and source column ids."""
    fields = [field.name for field in dataclasses.fields(fault_model_contracts.PlacedFaultModel)]
    assert fields == [
        "representation",
        "check",
        "priors",
        "observables",
        "owned",
        "source_fault_ids",
        "boundary_flips",
    ]


def test_placed_fault_model_freezes_every_field():
    """A placed fault model freezes its arrays, its sparse matrices, its column ids and its flip mapping."""
    placed = make_placed_model()
    for field_name in ("priors", "owned"):
        array = getattr(placed, field_name)
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array[0] = array[0]
    for field_name in ("check", "observables"):
        matrix = getattr(placed, field_name)
        assert matrix.format == "csc" and matrix.dtype == numpy.uint8
        assert matrix.data.flags.writeable is False
        assert matrix.indices.flags.writeable is False
    assert placed.check.toarray().tolist() == [[1, 0], [1, 1]]
    assert placed.source_fault_ids == (7, 9)
    assert isinstance(placed.boundary_flips, MappingProxyType)
    assert placed.boundary_flips == {0: (0, 1)}
    with pytest.raises(TypeError):
        placed.boundary_flips[2] = (1,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        placed.representation = PHYSICAL


def test_window_error_model_fields_and_deliberate_non_freezing():
    """A window model exposes its fields and deliberately leaves defect positions a plain mutable dict."""
    fields = [field.name for field in dataclasses.fields(fault_model_contracts.WindowErrorModel)]
    assert fields == [
        "detector_ids",
        "detector_coordinates",
        "defect_positions",
        "graphlike_faults",
        "physical_faults",
        "physical_to_graphlike_detector_projection",
    ]
    window = chain_models([(1, 2, 2), (3, 4, 4)])[0]
    assert type(window.defect_positions) is dict
    window.defect_positions[99] = (9, 9)
    assert window.defect_positions[99] == (9, 9)


def test_window_error_model_freezes_its_projection_as_sparse():
    """A window model holds its projection as a read-only csc matrix of the same entries."""
    projection = numpy.array([[True, False], [False, True]])

    window = fault_model_contracts.WindowErrorModel(
        detector_ids=(),
        detector_coordinates=None,
        defect_positions={},
        graphlike_faults=None,
        physical_faults=None,
        physical_to_graphlike_detector_projection=projection,
    )

    frozen = window.physical_to_graphlike_detector_projection
    assert frozen.format == "csc"
    assert frozen.toarray().tolist() == [[1, 0], [0, 1]]
    assert frozen.data.flags.writeable is False
    with pytest.raises(ValueError):
        frozen.data[0] = 0


def test_window_error_model_accepts_no_projection():
    """A window model continues to accept no physical-to-graphlike projection."""
    window = fault_model_contracts.WindowErrorModel(
        detector_ids=(),
        detector_coordinates=None,
        defect_positions={},
        graphlike_faults=None,
        physical_faults=None,
        physical_to_graphlike_detector_projection=None,
    )

    assert window.physical_to_graphlike_detector_projection is None


def test_require_faults_returns_or_fails_at_the_consuming_boundary():
    """Requesting a fault view returns it, or fails clearly when the view is missing or the argument is not a representation."""
    window = chain_models([(1, 2, 2), (3, 4, 4)])[0]
    assert window.require_faults(GRAPHLIKE) is window.graphlike_faults
    with pytest.raises(ValueError) as missing:
        window.require_faults(PHYSICAL)
    assert "window model does not contain physical faults" in str(missing.value)
    with pytest.raises(TypeError) as wrong_type:
        window.require_faults("graphlike")
    assert "representation must be a FaultRepresentation value" in str(wrong_type.value)


@pytest.mark.parametrize(
    "entry, expected",
    [
        ((3, 5, 7), (3, 3, 5, 7)),
        ((2, 3, 5, 7), (2, 3, 5, 7)),
        ((1, 1, 1, 1), (1, 1, 1, 1)),
    ],
)
def test_parse_window_entry_normalises_three_and_four_value_forms(entry, expected):
    """Three-value and four-value plan entries normalise to the same four-bound form."""
    assert window_placement._parse_window_entry(entry) == expected


@pytest.mark.parametrize(
    "entry, expected_error",
    [
        ((1, 2, True), TypeError),
        ((1.0, 2.0, 3.0), TypeError),
        ((0, 2, 3), ValueError),
        ((3, 2, 4), ValueError),
        ((2, 1, 3, 4), ValueError),
        ((1, 2, 4, 3), ValueError),
    ],
)
def test_parse_window_entry_rejects_bad_bounds(entry, expected_error):
    """Plan bounds must be built-in positive integers in non-decreasing order."""
    with pytest.raises(expected_error):
        window_placement._parse_window_entry(entry)


@pytest.mark.parametrize("entry", [(1, 2), (1, 2, 3, 4, 5)])
def test_parse_window_entry_lets_a_malformed_length_fail_naturally(entry):
    """A plan entry of the wrong length fails when it is unpacked."""
    with pytest.raises(ValueError):
        window_placement._parse_window_entry(entry)


def test_detectors_in_window_selects_rows_by_round():
    """A window takes the detectors of its buffer rounds, and the terminal window takes everything from its start onward."""
    detectors_by_round = {1: [0], 2: [1], 3: [2], 4: [3]}
    assert window_placement._detectors_in_window(detectors_by_round, 2, 3, is_last=False) == [1, 2]
    assert window_placement._detectors_in_window(detectors_by_round, 2, 3, is_last=True) == [1, 2, 3]
    assert window_placement._detectors_in_window(detectors_by_round, 5, 6, is_last=False) == []


# --------------------------------------------------------------------------
# Column selection and ownership
# --------------------------------------------------------------------------


def test_fault_columns_are_touching_columns_not_committed_elsewhere():
    """Candidate columns are those touching the window minus the columns a prior window committed; a candidate list restricts the search."""
    detector_sets = ((0,), (0, 1), (1, 2), (3,))
    row_index = {0: 0, 1: 1, 2: 2}
    assert window_placement._fault_columns_for_window(detector_sets, row_index, set()) == [0, 1, 2]
    assert window_placement._fault_columns_for_window(detector_sets, row_index, {1, 2}) == [0]
    assert window_placement._fault_columns_for_window(detector_sets, row_index, set(), [1, 3]) == [1]


def test_committed_columns_stay_excluded_on_both_ownership_paths():
    """A committed column never comes back, whether ownership is incremental or given explicitly (qLDPC: addressed errors are removed from every later window)."""
    plan = [(1, 1, 2, 2), (2, 3, 4, 4)]
    incremental = chain_models(plan)[1].require_faults(GRAPHLIKE)
    assert incremental.source_fault_ids == (3, 4)
    assert incremental.owned.tolist() == [True, True]
    dependency = chain_models(plan, dependency_edges=((0, 1),))[1].require_faults(
        GRAPHLIKE
    )
    assert dependency.source_fault_ids == (3, 4)
    assert dependency.owned.tolist() == [True, True]


@pytest.mark.parametrize(
    "fault_index, committed, unowned, explicit, is_last, expected",
    [
        (0, set(), {0}, {0}, True, False),
        (0, {0}, set(), {0}, False, True),
        (0, set(), set(), set(), True, False),
        (0, {0}, set(), None, True, False),
        (0, set(), set(), None, True, True),
        (0, set(), set(), None, False, True),
        (1, set(), set(), None, False, False),
    ],
)
def test_fault_owned_by_window_precedence(
    fault_index, committed, unowned, explicit, is_last, expected
):
    """Ownership follows a strict precedence: excluded, then explicit set, then committed elsewhere, then terminal, then commit rounds."""
    fault_rounds = ((2,), (7,))
    assert (
        window_placement._fault_owned_by_window(
            fault_index,
            fault_rounds,
            committed,
            unowned,
            explicit,
            placement_context(commit_lo=1, commit_hi=3, is_last=is_last),
        )
        is expected
    )


def test_detector_bits_are_local_but_observable_bits_are_unconditional():
    """A column sets detector bits only for rows in this window but sets every observable bit."""
    check, observables, _, _ = window_placement._build_window_arrays(
        context=placement_context(rows=[9, 0], row_index={0: 1}, n_obs=3),
        columns=[0],
        det_sets=((0, 5),),
        obs_sets=((2,),),
        fault_rounds=((1, 3),),
        committed_elsewhere=set(),
        unowned_faults=set(),
        explicitly_owned_faults=None,
    )
    assert check.toarray().tolist() == [[0], [1]]
    assert observables.toarray().tolist() == [[0], [0], [1]]


def placement_context(**overrides):
    """Build one window placement context for direct calls into the placement helpers."""
    fields = dict(
        rows=[0, 1],
        row_index={0: 0, 1: 1},
        round_of={0: 1, 1: 2, 2: 3},
        n_obs=1,
        commit_lo=1,
        commit_hi=2,
        is_last=False,
    )
    fields.update(overrides)
    return window_placement.WindowPlacementContext(**fields)


def build_window_arrays_case(context_overrides=None, **overrides):
    arguments = dict(
        context=placement_context(**(context_overrides or {})),
        columns=[0, 1],
        det_sets=((0,), (1, 2)),
        obs_sets=((0,), ()),
        fault_rounds=((1,), (2, 3)),
        committed_elsewhere=set(),
        unowned_faults=set(),
        explicitly_owned_faults=None,
    )
    arguments.update(overrides)
    return arguments


def test_build_window_arrays_produces_four_aligned_outputs():
    """One pass builds the check, observable and ownership arrays plus the boundary flip map."""
    arguments = build_window_arrays_case()
    check, observables, owned, boundary_flips = (
        window_placement._build_window_arrays(**arguments)
    )
    assert check.shape == (2, 2)
    assert check.dtype == numpy.uint8
    assert check.toarray().tolist() == [[1, 0], [0, 1]]
    assert observables.shape == (1, 2)
    assert observables.toarray().tolist() == [[1, 0]]
    assert owned.dtype == bool
    assert owned.tolist() == [True, True]
    assert boundary_flips == {0: (0,), 1: (1, 2)}
    assert arguments["committed_elsewhere"] == {0, 1}


def test_owned_columns_are_not_recorded_on_the_explicit_path():
    """With an explicit owner set, ownership is membership and no incremental commitment state is recorded."""
    arguments = build_window_arrays_case(explicitly_owned_faults={1})
    _, _, owned, _ = window_placement._build_window_arrays(**arguments)
    assert owned.tolist() == [False, True]
    assert arguments["committed_elsewhere"] == set()


def test_detectorless_owned_column_records_no_boundary_flip():
    """An owned column with no detectors contributes no boundary flip entry."""
    arguments = build_window_arrays_case(
        context_overrides={"is_last": True},
        columns=[0],
        det_sets=((),),
        obs_sets=((),),
        fault_rounds=((),),
    )
    _, _, owned, boundary_flips = window_placement._build_window_arrays(
        **arguments
    )
    assert owned.tolist() == [True]
    assert boundary_flips == {}


@pytest.mark.parametrize(
    "ranges, expected_error, expected_message",
    [
        (((True, 3),), TypeError, "built-in integer"),
        (((1.0, 3),), TypeError, "built-in integer"),
        (((4, 3),), ValueError, "is inverted"),
    ],
)
def test_fault_exclusion_ranges_are_validated(ranges, expected_error, expected_message):
    """Exclusion ranges must be built-in integer pairs that are not inverted."""
    with pytest.raises(expected_error) as failure:
        window_placement._validate_fault_exclusion_ranges(ranges)
    assert expected_message in str(failure.value)


def test_valid_exclusion_ranges_pass():
    """Well-formed and empty exclusion range tuples are accepted."""
    window_placement._validate_fault_exclusion_ranges(((1, 3), (5, 5)))
    window_placement._validate_fault_exclusion_ranges(())


def test_unowned_faults_are_those_touching_an_exclusion_range():
    """A fault is excluded from commitment when any of its detectors falls in an exclusion range."""
    fault_rounds = ((1,), (2, 3), (4,))
    assert window_placement._unowned_faults(fault_rounds, ((3, 3),)) == {1}
    assert window_placement._unowned_faults(fault_rounds, ((1, 1), (4, 9))) == {0, 2}
    assert window_placement._unowned_faults(fault_rounds, ()) == set()


def test_exclusion_keeps_a_fault_available_but_uncommittable():
    """An excluded fault still appears as a column to explain the syndrome but cannot be committed."""
    window = window_model_builders.build_single_window_error_model(
        chain_circuit(),
        (1, 2, 2),
        round_count=CHAIN_ROUND_COUNT,
        detector_rounds=dict(CHAIN_DETECTOR_ROUNDS),
        fault_model_requirement=GRAPHLIKE_REQUIREMENT,
        exclude_faults_touching=(2, 2),
    )
    faults = window.require_faults(GRAPHLIKE)
    assert faults.source_fault_ids == (0, 1, 2)
    assert faults.owned.tolist() == [True, False, False]


def test_multi_range_and_single_pair_builders_agree():
    """The single-pair and multi-range single-window builders agree for one exclusion range."""
    single = window_model_builders.build_single_window_error_model(
        chain_circuit(),
        (1, 2, 2),
        round_count=CHAIN_ROUND_COUNT,
        detector_rounds=dict(CHAIN_DETECTOR_ROUNDS),
        fault_model_requirement=GRAPHLIKE_REQUIREMENT,
        exclude_faults_touching=(2, 2),
    )
    multiple = window_model_builders.build_single_window_error_model_with_exclusions(
        chain_circuit(),
        (1, 2, 2),
        round_count=CHAIN_ROUND_COUNT,
        detector_rounds=dict(CHAIN_DETECTOR_ROUNDS),
        fault_model_requirement=GRAPHLIKE_REQUIREMENT,
        fault_exclusion_ranges=((2, 2),),
    )
    single_faults = single.require_faults(GRAPHLIKE)
    multiple_faults = multiple.require_faults(GRAPHLIKE)
    assert single_faults.source_fault_ids == multiple_faults.source_fault_ids
    assert single_faults.owned.tolist() == multiple_faults.owned.tolist()
    assert single.detector_ids == multiple.detector_ids


def test_single_window_builders_are_never_terminal():
    """A single-window build is never treated as the terminal window."""
    window = window_model_builders.build_single_window_error_model(
        chain_circuit(),
        (1, CHAIN_ROUND_COUNT, CHAIN_ROUND_COUNT),
        round_count=CHAIN_ROUND_COUNT,
        detector_rounds=dict(CHAIN_DETECTOR_ROUNDS),
        fault_model_requirement=GRAPHLIKE_REQUIREMENT,
    )
    faults = window.require_faults(GRAPHLIKE)
    # is_last=False keeps the buffer_hi row bound and the future-flip logic on.
    assert window.detector_ids == (0, 1, 2, 3)
    assert faults.owned.tolist() == [True, True, True, True, True]


def test_placed_model_dispatches_domain_appropriate_validation():
    """A placed graphlike view is degree checked while a placed physical view is not."""
    hyperedge_circuit = FakeCircuit(
        detector_count=3,
        observable_count=1,
        detector_coordinates={0: [0.0, 0.0], 1: [0.0, 1.0], 2: [0.0, 2.0]},
        decomposed_model=make_model([(0.1, "D0 D1 D2")]),
    )
    rounds = {0: 1, 1: 2, 2: 3}
    with pytest.raises(ValueError) as failure:
        window_slicer.WindowSlicer(
            hyperedge_circuit,
            round_count=3,
            detector_rounds=dict(rounds),
            fault_model_requirement=GRAPHLIKE_REQUIREMENT,
        ).slice_window(1, 1, 3, 3, is_last=True)
    message = str(failure.value)
    assert "placed graphlike fault model column 0 is a detector hyperedge" in message
    physical_window = window_slicer.WindowSlicer(
        hyperedge_circuit,
        round_count=3,
        detector_rounds=dict(rounds),
        fault_model_requirement=PHYSICAL_REQUIREMENT,
    ).slice_window(1, 1, 3, 3, is_last=True)
    assert physical_window.require_faults(PHYSICAL).check.shape == (3, 1)


def test_empty_column_selection_is_not_rejected():
    """A window that selects no fault columns is allowed and yields zero-width arrays."""
    circuit = FakeCircuit(
        detector_count=4,
        observable_count=1,
        detector_coordinates={
            detector_id: [0.0, float(detector_id)] for detector_id in range(4)
        },
        decomposed_model=make_model([(0.1, "D0 D1")]),
    )
    window = window_slicer.WindowSlicer(
        circuit,
        round_count=4,
        detector_rounds=dict(CHAIN_DETECTOR_ROUNDS),
        fault_model_requirement=GRAPHLIKE_REQUIREMENT,
    ).slice_window(4, 4, 4, 4, is_last=False)
    faults = window.require_faults(GRAPHLIKE)
    assert faults.check.shape == (1, 0)
    assert faults.observables.shape == (1, 0)
    assert faults.owned.tolist() == []
    assert faults.source_fault_ids == ()


# --------------------------------------------------------------------------
# The window slicer
# --------------------------------------------------------------------------


def test_detector_coordinates_are_queried_once_at_construction_and_never_requeried():
    """The slicer takes the coordinate map once when it is built, keeps that object, and slices without touching the circuit again."""
    returned_maps = []

    class RecordingCircuit(FakeCircuit):
        def get_detector_coordinates(self):
            coordinates = super().get_detector_coordinates()
            returned_maps.append(coordinates)
            return coordinates

    circuit = RecordingCircuit(
        detector_count=4,
        observable_count=1,
        detector_coordinates={
            detector_id: [0.0, float(detector_id)] for detector_id in range(4)
        },
        decomposed_model=make_model(CHAIN_ERROR_ROWS),
    )
    slicer = window_slicer.WindowSlicer(
        circuit,
        round_count=CHAIN_ROUND_COUNT,
        detector_rounds=dict(CHAIN_DETECTOR_ROUNDS),
        fault_model_requirement=GRAPHLIKE_REQUIREMENT,
    )
    assert circuit.accessed_names.count("get_detector_coordinates") == 1
    assert len(returned_maps) == 1
    assert slicer.detector_coordinates is returned_maps[0]

    first = slicer.slice_window(1, 1, 2, 2, is_last=False)
    second = slicer.slice_window(3, 3, 4, 4, is_last=True)
    assert circuit.accessed_names.count("get_detector_coordinates") == 1
    assert slicer.detector_coordinates is returned_maps[0]
    assert first.detector_coordinates == ((0.0, 0.0), (0.0, 1.0))
    assert second.detector_coordinates == ((0.0, 2.0), (0.0, 3.0))


def test_slicer_holds_one_operation_state():
    """The slicer holds one operation's catalogs, chronology, positions and per-domain commitment state."""
    circuit = chain_circuit()
    slicer = window_slicer.WindowSlicer(
        circuit,
        round_count=CHAIN_ROUND_COUNT,
        detector_rounds=dict(CHAIN_DETECTOR_ROUNDS),
        fault_model_requirement=GRAPHLIKE_REQUIREMENT,
    )
    assert not hasattr(slicer, "circuit")
    assert slicer.detector_coordinates == {
        detector_id: [0.0, float(detector_id)] for detector_id in range(4)
    }
    assert set(slicer.catalogs) == {GRAPHLIKE}
    assert slicer.catalog_link is None
    assert slicer.n_obs == 1
    assert slicer.round_of == CHAIN_DETECTOR_ROUNDS
    assert slicer.pos_of == {0: 0, 1: 0, 2: 0, 3: 0}
    assert slicer.committed_elsewhere == {GRAPHLIKE: set()}


def test_ownership_state_is_kept_per_representation():
    """Commitment state is tracked separately for each fault representation."""
    requirement = fault_model_contracts.DecoderFaultModelRequirement(
        frozenset({GRAPHLIKE, PHYSICAL})
    )
    slicer = window_slicer.WindowSlicer(
        linked_circuit(),
        round_count=4,
        detector_rounds={0: 1, 1: 2, 2: 3, 3: 4},
        fault_model_requirement=requirement,
    )
    assert set(slicer.committed_elsewhere) == {GRAPHLIKE, PHYSICAL}
    slicer.slice_window(1, 1, 2, 2, is_last=False)
    graphlike_committed = slicer.committed_elsewhere[GRAPHLIKE]
    physical_committed = slicer.committed_elsewhere[PHYSICAL]
    assert graphlike_committed is not physical_committed
    assert graphlike_committed == {0, 1}
    assert physical_committed == {0, 1}
    assert len(slicer.catalogs[GRAPHLIKE].detector_sets) == 3
    assert len(slicer.catalogs[PHYSICAL].detector_sets) == 2


def test_placement_context_is_a_frozen_seven_field_bundle_without_copies():
    """The window placement context is a frozen seven-field bundle that normalises, copies and freezes nothing."""
    rows = [1, 2]
    row_index = {1: 0, 2: 1}
    round_of = {1: 1, 2: 2}
    context = window_placement.WindowPlacementContext(
        rows=rows,
        row_index=row_index,
        round_of=round_of,
        n_obs=1,
        commit_lo=1,
        commit_hi=2,
        is_last=False,
    )
    fields = dataclasses.fields(context)
    assert [field.name for field in fields] == [
        "rows",
        "row_index",
        "round_of",
        "n_obs",
        "commit_lo",
        "commit_hi",
        "is_last",
    ]
    assert all(field.default is dataclasses.MISSING for field in fields)
    assert all(field.default_factory is dataclasses.MISSING for field in fields)
    assert context.rows is rows
    assert context.row_index is row_index
    assert context.round_of is round_of
    rows.append(3)
    assert context.rows == [1, 2, 3]
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.is_last = True
    assert "__post_init__" not in vars(window_placement.WindowPlacementContext)
    own_methods = [
        name
        for name, value in vars(window_placement.WindowPlacementContext).items()
        if callable(value) and not name.startswith("__")
    ]
    assert own_methods == []
    with pytest.raises(TypeError):
        window_placement.WindowPlacementContext(rows=rows, row_index=row_index)


def test_slice_window_builds_one_shared_placement_context(monkeypatch):
    """One placement context is built per slice, before any domain is placed, and the same object is reused for every domain."""
    events = []
    real_context_type = window_placement.WindowPlacementContext
    real_placement = window_placement._placed_faults_for_window

    def recording_context(**fields):
        context = real_context_type(**fields)
        events.append(("built", context))
        return context

    def recording_placement(*, context, **rest):
        events.append(("placed", context))
        return real_placement(context=context, **rest)

    monkeypatch.setattr(window_slicer, "WindowPlacementContext", recording_context)
    monkeypatch.setattr(window_slicer, "_placed_faults_for_window", recording_placement)
    both_domains = fault_model_contracts.DecoderFaultModelRequirement(
        frozenset({GRAPHLIKE, PHYSICAL})
    )
    slicer = window_slicer.WindowSlicer(
        linked_circuit(),
        round_count=4,
        detector_rounds={0: 1, 1: 2, 2: 3, 3: 4},
        fault_model_requirement=both_domains,
    )
    model = slicer.slice_window(1, 1, 2, 3, is_last=False)
    kinds = [kind for kind, _ in events]
    assert kinds == ["built", "placed", "placed"]
    contexts = [context for _, context in events]
    assert contexts[1] is contexts[0]
    assert contexts[2] is contexts[0]
    shared = contexts[0]
    assert shared.rows == list(model.detector_ids)
    assert shared.row_index == {
        detector_id: row for row, detector_id in enumerate(model.detector_ids)
    }
    assert shared.round_of is slicer.round_of
    assert shared.n_obs == slicer.n_obs
    assert (shared.commit_lo, shared.commit_hi, shared.is_last) == (1, 2, False)


def test_explicit_owner_and_predecessor_maps_come_together():
    """Explicit owner and predecessor maps must be supplied together."""
    slicer = window_slicer.WindowSlicer(
        chain_circuit(),
        round_count=CHAIN_ROUND_COUNT,
        detector_rounds=dict(CHAIN_DETECTOR_ROUNDS),
        fault_model_requirement=GRAPHLIKE_REQUIREMENT,
    )
    with pytest.raises(ValueError) as failure:
        slicer.slice_window(
            1, 1, 2, 2, is_last=False, explicitly_owned_faults={GRAPHLIKE: set()}
        )
    assert "must be supplied together" in str(failure.value)
    with pytest.raises(ValueError):
        slicer.slice_window(
            1, 1, 2, 2, is_last=False, explicitly_prior_faults={GRAPHLIKE: set()}
        )


def test_slice_window_products_are_complete():
    """One sliced window carries its rows, coordinates, matrices, ownership, flip maps and residual defect positions."""
    first, last = chain_models([(1, 2, 2), (3, 4, 4)])
    first_faults = first.require_faults(GRAPHLIKE)
    assert first.detector_ids == (0, 1)
    assert first.detector_coordinates == ((0.0, 0.0), (0.0, 1.0))
    assert first_faults.source_fault_ids == (0, 1, 2)
    assert first_faults.check.toarray().tolist() == [[1, 1, 0], [0, 1, 1]]
    assert first_faults.observables.toarray().tolist() == [[1, 0, 0]]
    assert first_faults.owned.tolist() == [True, True, True]
    assert first_faults.boundary_flips == {0: (0,), 1: (0, 1), 2: (1, 2)}
    assert first.defect_positions == {0: (1, 0), 1: (2, 0), 2: (3, 0)}
    last_faults = last.require_faults(GRAPHLIKE)
    assert last.detector_ids == (2, 3)
    assert last_faults.source_fault_ids == (3, 4)
    assert last_faults.owned.tolist() == [True, True]
    assert last.defect_positions == {2: (3, 0), 3: (4, 0)}


# --------------------------------------------------------------------------
# Dependency graph ownership
# --------------------------------------------------------------------------


def test_dependency_depths_and_cycle_rejection():
    """Window depths follow the dependency edges, and negative indices or cycles are rejected."""
    assert window_ownership_dag._dependency_depths(3, ((0, 1), (1, 2))) == (0, 1, 2)
    assert window_ownership_dag._dependency_depths(2, ()) == (0, 0)
    assert window_ownership_dag._dependency_depths(3, ((0, 2), (1, 2))) == (0, 0, 1)
    with pytest.raises(ValueError) as negative:
        window_ownership_dag._dependency_depths(2, ((-1, 1),))
    assert "must be nonnegative" in str(negative.value)
    with pytest.raises(ValueError) as cycle:
        window_ownership_dag._dependency_depths(2, ((0, 1), (1, 0)))
    assert "must form an acyclic graph" in str(cycle.value)


def test_dependency_ancestors_are_transitive():
    """Each window's ancestor set includes indirect predecessors."""
    ancestors = window_ownership_dag._dependency_ancestors(3, ((0, 1), (1, 2)), (0, 1, 2))
    assert ancestors == (frozenset(), frozenset({0}), frozenset({0, 1}))


def test_explicit_prior_faults_union_ancestor_ownership():
    """A window's predecessor fault set is the union of what its ancestors own."""
    ownership = (
        {GRAPHLIKE: {0}},
        {GRAPHLIKE: {1}},
        {GRAPHLIKE: {2}},
    )
    ancestors = (frozenset(), frozenset({0}), frozenset({0, 1}))
    priors = window_ownership_dag._explicit_prior_faults(ownership, ancestors)
    # answered by membership: a fault is prior when its owner is an ancestor
    assert [[fault in prior[GRAPHLIKE] for fault in (0, 1, 2)] for prior in priors] == [
        [False, False, False],
        [True, False, False],
        [True, True, False],
    ]


def isolated_round_circuit():
    """A circuit with one fault whose detectors live only in round 2."""
    return FakeCircuit(
        detector_count=4,
        observable_count=1,
        detector_coordinates={
            detector_id: [0.0, float(detector_id)] for detector_id in range(4)
        },
        decomposed_model=make_model([(0.1, "D1 L0"), (0.1, "D2 D3")]),
    )


def isolated_slicer():
    return window_slicer.WindowSlicer(
        isolated_round_circuit(),
        round_count=CHAIN_ROUND_COUNT,
        detector_rounds=dict(CHAIN_DETECTOR_ROUNDS),
        fault_model_requirement=GRAPHLIKE_REQUIREMENT,
    )


def test_uncovered_fault_raises_only_when_the_plan_covers_the_operation():
    """A fault touching no commit region raises for a full plan and is left unowned for a partial one."""
    full_entries = ((1, 1, 1, 1), (3, 4, 4, 4))
    with pytest.raises(ValueError) as failure:
        window_ownership_dag._explicit_fault_ownership(
            isolated_slicer(), full_entries, (0, 1), round_count=CHAIN_ROUND_COUNT
        )
    assert "touches no window commit region" in str(failure.value)
    partial_entries = ((3, 3, 3, 3), (4, 4, 4, 4))
    ownership = window_ownership_dag._explicit_fault_ownership(
        isolated_slicer(), partial_entries, (0, 1), round_count=CHAIN_ROUND_COUNT
    )
    assert ownership == ({GRAPHLIKE: {1}}, {GRAPHLIKE: set()})


def test_equal_depth_candidates_have_no_causal_owner():
    """A fault straddling two independent commit regions of equal depth has no causal owner and is rejected."""
    with pytest.raises(ValueError) as failure:
        chain_models([(1, 1, 2, 2), (2, 3, 4, 4)], dependency_edges=())
    message = str(failure.value)
    assert "straddles independent commit regions without a causal owner" in message
    assert "graphlike fault 2" in message


def test_partial_suffix_plan_leaves_outside_faults_unowned():
    """A partial plan slices normally and leaves faults outside its commit regions unowned."""
    models = chain_models([(3, 3, 3), (4, 4, 4)], dependency_edges=((0, 1),))
    first = models[0].require_faults(GRAPHLIKE)
    last = models[1].require_faults(GRAPHLIKE)
    assert first.source_fault_ids == (2, 3)
    assert first.owned.tolist() == [True, True]
    assert last.source_fault_ids == (4,)
    assert last.owned.tolist() == [True]


def test_dag_ownership_is_compiled_before_any_window_is_sliced():
    """With dependency edges, ownership follows the shallowest window rather than plan order."""
    models = chain_models([(1, 1, 2, 2), (2, 3, 4, 4)], dependency_edges=((1, 0),))
    first = models[0].require_faults(GRAPHLIKE)
    last = models[1].require_faults(GRAPHLIKE)
    assert first.source_fault_ids == (0, 1)
    assert first.owned.tolist() == [True, True]
    # The plan-order-later window owns the straddling fault because it is the
    # shallower DAG node, which is only possible if ownership precedes slicing.
    assert last.source_fault_ids == (1, 2, 3, 4)
    assert last.owned.tolist() == [False, True, True, True]


# --------------------------------------------------------------------------
# Temporal seams and the window protocol
# --------------------------------------------------------------------------


SEAM_PLAN = [(1, 1, 2), (2, 2, 2), (3, 4, 4)]


def test_closed_boundary_window_must_be_a_dependency_destination():
    """A window declared temporally closed must be the destination of a dependency edge."""
    with pytest.raises(ValueError) as failure:
        chain_models(
            SEAM_PLAN,
            dependency_edges=((0, 1),),
            closed_temporal_boundary_windows=(2,),
        )
    assert "closed temporal boundary window must be a dependency destination" in str(
        failure.value
    )


def test_closed_boundary_window_may_not_truncate_a_global_fault():
    """A temporally closed window may not cut a global fault into an artificial boundary edge."""
    with pytest.raises(ValueError) as failure:
        chain_models(
            SEAM_PLAN,
            dependency_edges=((0, 1),),
            closed_temporal_boundary_windows=(1,),
        )
    message = str(failure.value)
    assert "graphlike closed temporal boundary window 1 truncates global fault 2" in message
    assert "smooth B boundary cannot contain an artificial boundary generator" in message


def test_empty_closed_window_tuple_is_inert():
    """Declaring no closed windows leaves slicing unchanged."""
    models = chain_models(SEAM_PLAN, dependency_edges=((0, 1), (2, 1)))
    assert len(models) == 3


def test_generic_protocol_imposes_nothing_and_unknown_protocols_fail_closed():
    """The generic protocol imposes nothing while any unknown protocol value is refused."""
    protocol = WindowProtocol
    assert protocol.GENERIC.name == "GENERIC"
    assert protocol.TAN_ZERO_SEAM_GRAPHLIKE.name == "TAN_ZERO_SEAM_GRAPHLIKE"
    window_protocol_policy._validate_window_protocol(
        ((1, 1, 2, 2),), protocol.GENERIC, None, (), GRAPHLIKE_REQUIREMENT
    )
    with pytest.raises(ValueError) as failure:
        window_protocol_policy._validate_window_protocol(
            ((1, 1, 2, 2),), object(), None, (), GRAPHLIKE_REQUIREMENT
        )
    assert "unsupported window protocol" in str(failure.value)


def test_valid_tan_zero_seam_plan_is_accepted():
    """A correct zero-seam plan with single-layer seams and two parents per seam is accepted."""
    models = chain_models(
        SEAM_PLAN,
        dependency_edges=((0, 1), (2, 1)),
        closed_temporal_boundary_windows=(1,),
        window_protocol=WindowProtocol.TAN_ZERO_SEAM_GRAPHLIKE,
    )
    assert len(models) == 3
    assert models[1].detector_ids == (1,)
    assert models[1].require_faults(GRAPHLIKE).source_fault_ids == ()


@pytest.mark.parametrize(
    "plan, edges, closed, requirement, expected",
    [
        (
            SEAM_PLAN,
            ((0, 1), (2, 1)),
            (),
            GRAPHLIKE_REQUIREMENT,
            "every Tan type-2 seam, and only a seam, must be temporally closed",
        ),
        (
            SEAM_PLAN,
            ((0, 1),),
            (1,),
            GRAPHLIKE_REQUIREMENT,
            "each Tan type-2 seam must depend on its two adjacent type-1 tasks",
        ),
        (
            [(1, 1, 2), (2, 2, 3), (3, 4, 4)],
            ((0, 1), (2, 1)),
            (1,),
            GRAPHLIKE_REQUIREMENT,
            "a zero-offset Tan type-2 seam must be one detector layer",
        ),
        (
            SEAM_PLAN,
            ((0, 1), (2, 1)),
            (1,),
            PHYSICAL_REQUIREMENT,
            "requires exactly the graphlike correction-edge representation",
        ),
    ],
)
def test_tan_zero_seam_contract_fails_closed(plan, edges, closed, requirement, expected):
    """A zero-seam plan missing its closed seams, its seam parents, its single-layer seam or its graphlike domain is refused."""
    with pytest.raises(ValueError) as failure:
        chain_models(
            plan,
            dependency_edges=edges,
            closed_temporal_boundary_windows=closed,
            window_protocol=WindowProtocol.TAN_ZERO_SEAM_GRAPHLIKE,
            fault_model_requirement=requirement,
        )
    assert expected in str(failure.value)


# --------------------------------------------------------------------------
# Public builders
# --------------------------------------------------------------------------


def test_build_window_error_models_returns_one_model_per_entry_in_plan_order():
    """Slicing returns one window model per plan entry, in plan order."""
    models = chain_models([(1, 1, 1), (2, 2, 2), (3, 4, 4)])
    assert [model.detector_ids for model in models] == [(0,), (1,), (2, 3)]
    assert all(isinstance(model, fault_model_contracts.WindowErrorModel) for model in models)


@pytest.mark.parametrize(
    "plan, expected",
    [
        ([(1, 2, 2), (4, 4, 4)], "must be contiguous in plan order"),
        ([(1, 2, 2), (2, 4, 4)], "must be contiguous in plan order"),
        ([(1, 5, 5)], "window commit region exceeds round_count"),
    ],
)
def test_commit_regions_must_tile_the_plan(plan, expected):
    """Commit regions must be contiguous without gaps or overlaps and may not exceed the round count."""
    with pytest.raises(ValueError) as failure:
        chain_models(plan)
    assert expected in str(failure.value)


def test_non_terminal_last_window_does_not_claim_the_residue():
    """A last window that stops before the final round owns only faults touching its commit region."""
    models = chain_models([(1, 1, 1), (2, 2, 3)])
    last = models[1].require_faults(GRAPHLIKE)
    assert last.source_fault_ids == (2, 3)
    assert last.owned.tolist() == [True, False]


def test_terminal_window_claims_every_surviving_candidate():
    """A terminal window claims every remaining candidate column."""
    last = chain_models([(1, 2, 2), (3, 4, 4)])[1].require_faults(GRAPHLIKE)
    assert last.owned.tolist() == [True, True]


def test_overlapping_buffers_are_allowed_under_the_generic_protocol():
    """Buffer regions may overlap freely under the default protocol."""
    models = chain_models([(1, 1, 2, 4), (2, 3, 4, 4)])
    assert models[0].detector_ids == (0, 1, 2, 3)
    assert models[1].detector_ids == (1, 2, 3)


# --------------------------------------------------------------------------
# Deliberate non-checks and recorded findings
# --------------------------------------------------------------------------


def test_optional_capabilities_are_inert_when_unused():
    """Every optional capability stays inert when it is not requested."""
    models = chain_models([(1, 2, 2), (3, 4, 4)])
    assert models[0].physical_to_graphlike_detector_projection is None
    assert window_placement._unowned_faults(((1,), (2,)), ()) == set()
    assert (
        window_protocol_policy._validate_window_protocol(
            ((1, 1, 2, 2),), WindowProtocol.GENERIC, None, (), NO_REQUIREMENT
        )
        is None
    )


def test_source_fault_ids_are_the_only_catalog_index_carrier():
    """Source fault ids map each local column back to its global catalog column."""
    models = chain_models([(1, 2, 2), (3, 4, 4)])
    last = models[1].require_faults(GRAPHLIKE)
    catalog_columns, _, _ = stim_dem_catalog.detector_error_model_to_faults(
        make_model(CHAIN_ERROR_ROWS)
    )
    for local_column, global_column in enumerate(last.source_fault_ids):
        global_detectors = set(catalog_columns[global_column])
        local_detectors = {
            models[1].detector_ids[row]
            for row in last.check[:, local_column].nonzero()[0]
        }
        assert local_detectors == global_detectors & set(models[1].detector_ids)


# --------------------------------------------------------------------------
# Real Stim artifacts: a generated distance-3 memory circuit end to end
# --------------------------------------------------------------------------


REPETITION_DISTANCE = 3
REPETITION_ROUNDS = 5
SURFACE_DISTANCE = 3
SURFACE_ROUNDS = 3


@pytest.fixture(scope="module")
def stim_module():
    """Provide the real Stim package, skipping the real-artifact tests if it is absent."""
    return pytest.importorskip("stim")


@pytest.fixture(scope="module")
def repetition_circuit(stim_module):
    """A real generated distance-3 repetition-code memory circuit with noise."""
    return stim_module.Circuit.generated(
        "repetition_code:memory",
        rounds=REPETITION_ROUNDS,
        distance=REPETITION_DISTANCE,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
        before_round_data_depolarization=0.01,
    )


@pytest.fixture(scope="module")
def surface_circuit(stim_module):
    """A real generated distance-3 rotated surface-code memory circuit with noise."""
    return stim_module.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=SURFACE_ROUNDS,
        distance=SURFACE_DISTANCE,
        after_clifford_depolarization=0.005,
        before_measure_flip_probability=0.005,
        after_reset_flip_probability=0.005,
        before_round_data_depolarization=0.005,
    )


def global_graphlike_catalog(circuit):
    """Return the whole-circuit graphlike fault catalog of a real circuit."""
    detector_sets, observable_sets, priors = stim_dem_catalog.detector_error_model_to_faults(
        circuit.detector_error_model(decompose_errors=True)
    )
    return detector_sets, observable_sets, priors


def test_real_repetition_round_map_matches_the_circuit_structure(repetition_circuit):
    """Rounds recovered from a real repetition circuit match its coordinates and its per-round detector counts."""
    round_of = detector_chronology.resolve_detector_rounds(
        repetition_circuit, None, REPETITION_ROUNDS
    )
    coordinates = repetition_circuit.get_detector_coordinates()
    assert set(round_of) == set(range(repetition_circuit.num_detectors))
    assert set(round_of.values()) == set(range(1, REPETITION_ROUNDS + 1))
    for detector_id, emitted_round in round_of.items():
        raw_layer = int(coordinates[detector_id][-1])
        expected = (
            REPETITION_ROUNDS
            if raw_layer == REPETITION_ROUNDS
            else raw_layer + 1
        )
        assert emitted_round == expected
    detectors_per_round = collections.Counter(round_of.values())
    ancillas = REPETITION_DISTANCE - 1
    for emitted_round in range(1, REPETITION_ROUNDS):
        assert detectors_per_round[emitted_round] == ancillas
    assert detectors_per_round[REPETITION_ROUNDS] == 2 * ancillas
    assert sum(detectors_per_round.values()) == repetition_circuit.num_detectors


def test_real_surface_round_map_matches_the_circuit_structure(surface_circuit):
    """Rounds recovered from a real rotated surface-code circuit match its three-dimensional coordinates."""
    round_of = detector_chronology.resolve_detector_rounds(surface_circuit, None, SURFACE_ROUNDS)
    coordinates = surface_circuit.get_detector_coordinates()
    assert {len(coordinates[detector_id]) for detector_id in round_of} == {3}
    assert set(round_of.values()) == set(range(1, SURFACE_ROUNDS + 1))
    for detector_id, emitted_round in round_of.items():
        raw_layer = int(coordinates[detector_id][-1])
        expected = SURFACE_ROUNDS if raw_layer == SURFACE_ROUNDS else raw_layer + 1
        assert emitted_round == expected


def test_real_decomposed_model_is_graphlike_and_parity_reduced(repetition_circuit):
    """The catalog built from a real decomposed detector error model is parity reduced and graphlike."""
    detector_sets, observable_sets, priors = global_graphlike_catalog(repetition_circuit)
    assert detector_sets
    assert len(detector_sets) == len(observable_sets) == len(priors)
    for detectors, observables, prior in zip(detector_sets, observable_sets, priors):
        assert 1 <= len(detectors) <= 2
        assert list(detectors) == sorted(set(detectors))
        assert list(observables) == sorted(set(observables))
        assert 0.0 < prior < 1.0
    assert len(set(zip(detector_sets, observable_sets))) == len(detector_sets)


@pytest.mark.parametrize(
    "plan",
    [
        [(1, 2, 3), (3, 5, 5)],
        [(1, 2, 2), (2, 3, 4, 4), (4, 5, 5, 5)],
    ],
)
def test_real_repetition_commit_regions_cover_every_detector_once(
    repetition_circuit, plan
):
    """Every detector of a real circuit is committed by exactly one window of a full plan."""
    round_of = detector_chronology.resolve_detector_rounds(
        repetition_circuit, None, REPETITION_ROUNDS
    )
    entries = [window_placement._parse_window_entry(entry) for entry in plan]
    models = window_model_builders.build_window_error_models(
        repetition_circuit,
        plan,
        round_count=REPETITION_ROUNDS,
        fault_model_requirement=GRAPHLIKE_REQUIREMENT,
        fault_exclusion_ranges=(),
    )
    committed_detectors = []
    for model, (_, commit_lo, commit_hi, _) in zip(models, entries):
        committed_detectors.extend(
            detector_id
            for detector_id in model.detector_ids
            if commit_lo <= round_of[detector_id] <= commit_hi
        )
    assert sorted(committed_detectors) == list(
        range(repetition_circuit.num_detectors)
    )
    assert len(committed_detectors) == len(set(committed_detectors))


def test_real_repetition_windows_partition_and_preserve_the_global_faults(
    repetition_circuit,
):
    """Sliced windows of a real circuit own every global fault once and keep each column's logical identity."""
    detector_sets, observable_sets, priors = global_graphlike_catalog(repetition_circuit)
    plan = [(1, 2, 3), (3, 5, 5)]
    models = window_model_builders.build_window_error_models(
        repetition_circuit,
        plan,
        round_count=REPETITION_ROUNDS,
        fault_model_requirement=GRAPHLIKE_REQUIREMENT,
        fault_exclusion_ranges=(),
    )
    owned_columns = []
    for model in models:
        faults = model.require_faults(GRAPHLIKE)
        rows = model.detector_ids
        assert faults.check.shape == (len(rows), len(faults.source_fault_ids))
        assert faults.observables.shape[0] == repetition_circuit.num_observables
        for local_column, global_column in enumerate(faults.source_fault_ids):
            global_detectors = set(detector_sets[global_column])
            local_detectors = {
                rows[row]
                for row in faults.check[:, local_column].nonzero()[0]
            }
            assert local_detectors == global_detectors & set(rows)
            local_observables = set(
                int(observable)
                for observable in faults.observables[:, local_column].nonzero()[0]
            )
            assert local_observables == set(observable_sets[global_column])
            assert faults.priors[local_column] == pytest.approx(priors[global_column])
            if faults.owned[local_column]:
                owned_columns.append(global_column)
                assert set(
                    faults.boundary_flips.get(local_column, ())
                ) == global_detectors
    assert sorted(owned_columns) == list(range(len(detector_sets)))


def test_real_repetition_boundary_flips_carry_the_full_detector_effect(repetition_circuit):
    """Owned columns of a real circuit carry their complete global detector effect, so the next window can take the post-commit part."""
    round_of = detector_chronology.resolve_detector_rounds(
        repetition_circuit, None, REPETITION_ROUNDS
    )
    detector_sets, _, _ = global_graphlike_catalog(repetition_circuit)
    first, last = window_model_builders.build_window_error_models(
        repetition_circuit,
        [(1, 2, 3), (3, 5, 5)],
        round_count=REPETITION_ROUNDS,
        fault_model_requirement=GRAPHLIKE_REQUIREMENT,
        fault_exclusion_ranges=(),
    )
    first_faults = first.require_faults(GRAPHLIKE)
    handed_on = False
    for local_column, global_column in enumerate(first_faults.source_fault_ids):
        if not first_faults.owned[local_column]:
            continue
        assert first_faults.boundary_flips[local_column] == tuple(detector_sets[global_column])
        beyond_commit = [
            detector_id
            for detector_id in detector_sets[global_column]
            if round_of[detector_id] > 2
        ]
        handed_on = handed_on or bool(beyond_commit)
        for detector_id in detector_sets[global_column]:
            assert detector_id in first.defect_positions
    assert handed_on
    last_faults = last.require_faults(GRAPHLIKE)
    assert set(last_faults.boundary_flips) == {
        column for column, owned in enumerate(last_faults.owned) if owned
    }


def test_real_surface_slicing_preserves_the_global_faults(surface_circuit):
    """A real surface-code circuit slices into windows whose columns keep the global logical identity."""
    detector_sets, observable_sets, _ = global_graphlike_catalog(surface_circuit)
    models = window_model_builders.build_window_error_models(
        surface_circuit,
        [(1, 1, 2), (2, 3, 3)],
        round_count=SURFACE_ROUNDS,
        fault_model_requirement=GRAPHLIKE_REQUIREMENT,
        fault_exclusion_ranges=(),
    )
    owned_columns = []
    for model in models:
        faults = model.require_faults(GRAPHLIKE)
        rows = model.detector_ids
        assert model.detector_coordinates is not None
        assert len(model.detector_coordinates) == len(rows)
        for local_column, global_column in enumerate(faults.source_fault_ids):
            local_detectors = {
                rows[row]
                for row in faults.check[:, local_column].nonzero()[0]
            }
            assert local_detectors == set(detector_sets[global_column]) & set(rows)
            local_observables = set(
                int(observable)
                for observable in faults.observables[:, local_column].nonzero()[0]
            )
            assert local_observables == set(observable_sets[global_column])
            if faults.owned[local_column]:
                owned_columns.append(global_column)
    assert sorted(owned_columns) == list(range(len(detector_sets)))


def test_real_repetition_linked_views_agree_with_both_stim_models(repetition_circuit):
    """Both Stim views of a real circuit build a linked pair whose local projection reproduces the physical rows."""
    slicer = window_slicer.WindowSlicer(
        repetition_circuit,
        round_count=REPETITION_ROUNDS,
        fault_model_requirement=LINKED_REQUIREMENT,
    )
    assert set(slicer.catalogs) == {GRAPHLIKE, PHYSICAL}
    window = slicer.slice_window(1, 1, 3, 3, is_last=False)
    graphlike = window.require_faults(GRAPHLIKE)
    physical = window.require_faults(PHYSICAL)
    projection = window.physical_to_graphlike_detector_projection
    assert projection.shape == (
        len(graphlike.source_fault_ids),
        len(physical.source_fault_ids),
    )
    derived = (graphlike.check.toarray().astype(numpy.uint64)
               @ projection.toarray().astype(numpy.uint64)) % 2
    assert numpy.array_equal(derived, physical.check.toarray())
    fault_identity_validation.validate_belief_matching_matrices(
        graphlike.check,
        graphlike.observables,
        physical.check,
        physical.priors,
        projection,
        location="real repetition window",
    )


def test_real_dependency_plan_partitions_the_real_catalog(repetition_circuit):
    """A dependency-scheduled plan over a real circuit assigns every global fault to exactly one window."""
    detector_sets, _, _ = global_graphlike_catalog(repetition_circuit)
    models = window_model_builders.build_window_error_models(
        repetition_circuit,
        [(1, 2, 3), (3, 5, 5)],
        round_count=REPETITION_ROUNDS,
        fault_model_requirement=GRAPHLIKE_REQUIREMENT,
        fault_exclusion_ranges=(),
        dependency_edges=((0, 1),),
    )
    owned_columns = []
    for model in models:
        faults = model.require_faults(GRAPHLIKE)
        owned_columns.extend(
            global_column
            for local_column, global_column in enumerate(faults.source_fault_ids)
            if faults.owned[local_column]
        )
    assert sorted(owned_columns) == list(range(len(detector_sets)))
