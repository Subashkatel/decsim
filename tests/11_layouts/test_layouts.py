"""Behavior tests for uniform layout selection and resource claims."""

import inspect
import math

import pytest

import decsim.qpu.layouts as layouts_module
from decsim.qpu.code_geometry import SurfaceCodeModel
from decsim.decoders.decoders import PresetLatencyDecoder
from decsim.qpu.layouts import UniformLayout
from decsim.message import Operation, OperationPlanningView, ResourceClaim
from decsim.protocols import CodeModel, LayoutModel
from decsim.run_spec import RunSpec


class RecordingLayout:
    """Record values supplied through every layout hook."""

    def __init__(self, code):
        self.code = code
        self.calls = []

    def code_for_op(self, operation):
        self.calls.append(("code_for_op", operation))
        return self.code

    def code_for_patch(self, patch_identity):
        self.calls.append(("code_for_patch", patch_identity))
        return self.code

    def codes(self):
        self.calls.append(("codes", None))
        return [self.code]

    def spatial_nodes_for(self, operation, *, base_spatial_node_count):
        self.calls.append(("spatial_nodes_for", operation))
        return base_spatial_node_count

    def patch_spatial_nodes_for(self, patch_identity, *, base_spatial_node_count):
        self.calls.append(("patch_spatial_nodes_for", patch_identity))
        return base_spatial_node_count

    def resources_for(self, operation):
        self.calls.append(("resources_for", operation))
        return [ResourceClaim("qubits", frozenset(operation.qubits))]


def make_operation(*, operation_id=4, qubits=(3, 5), patches=(11,)):
    return Operation(
        id=operation_id,
        name="timing-only",
        qubits=qubits,
        patches=patches,
    )


def make_planning_view(*, qubits=(3, 5, 3)):
    return OperationPlanningView.from_operation(make_operation(qubits=qubits))


def test_layout_stores_and_exposes_the_exact_mutable_code_object():
    """Construction stores the supplied code unchanged in a public field."""
    first_code = SurfaceCodeModel(d=3)
    second_code = SurfaceCodeModel(d=5)
    layout = UniformLayout(first_code)

    assert isinstance(first_code, CodeModel)
    assert layout.code is first_code

    layout.code = second_code

    assert layout.code is second_code
    assert layout.code_for_op(None) is second_code
    assert layout.code_for_patch(None) is second_code
    assert layout.codes() == [second_code]


def test_layout_implements_all_six_structural_hooks():
    """The uniform adapter satisfies the complete structural layout seam."""
    layout = UniformLayout(SurfaceCodeModel(d=3))
    expected_hooks = {
        "code_for_op",
        "code_for_patch",
        "codes",
        "spatial_nodes_for",
        "patch_spatial_nodes_for",
        "resources_for",
    }
    protocol_hooks = {
        name
        for name, member in LayoutModel.__dict__.items()
        if callable(member) and not name.startswith("_")
    }

    assert protocol_hooks == expected_hooks
    assert isinstance(layout, LayoutModel)
    assert all(callable(getattr(layout, name)) for name in expected_hooks)


def test_code_selectors_ignore_arbitrary_operation_and_patch_identities():
    """Both selectors return the current code without validating their inputs."""
    code = SurfaceCodeModel(d=3)
    layout = UniformLayout(code)

    for arbitrary_input in (None, object(), "not an operation", -7):
        assert layout.code_for_op(arbitrary_input) is code
        assert layout.code_for_patch(arbitrary_input) is code


def test_spatial_hooks_return_even_invalid_base_counts_unchanged():
    """Both spatial transforms silently pass every base count through unchanged."""
    layout = UniformLayout(SurfaceCodeModel(d=3))
    operation = make_planning_view()

    for invalid_count in (None, -1, 0, "invalid", math.nan, object()):
        assert layout.spatial_nodes_for(
            operation,
            base_spatial_node_count=invalid_count,
        ) is invalid_count
        assert layout.patch_spatial_nodes_for(
            None,
            base_spatial_node_count=invalid_count,
        ) is invalid_count


def test_spatial_base_counts_remain_keyword_only():
    """Spatial base counts cannot be supplied positionally."""
    layout = UniformLayout(SurfaceCodeModel(d=3))

    with pytest.raises(TypeError):
        layout.spatial_nodes_for(None, 4)
    with pytest.raises(TypeError):
        layout.patch_spatial_nodes_for(None, 4)


def test_resources_group_all_planning_qubits_in_one_immutable_claim():
    """Planning qubits become one qubit claim with a frozenset payload."""
    layout = UniformLayout(SurfaceCodeModel(d=3))
    operation = make_planning_view(qubits=(8, 2, 8, 5))
    original_qubits = operation.qubits

    claims = layout.resources_for(operation)

    assert claims == [ResourceClaim("qubits", frozenset({2, 5, 8}))]
    assert len(claims) == 1
    assert type(claims[0]) is ResourceClaim
    assert claims[0].kind == "qubits"
    assert type(claims[0].ids) is frozenset
    assert operation.qubits == original_qubits


def test_resources_return_fresh_lists_and_allow_an_empty_declaration():
    """Each resource call returns a new singleton list even without qubits."""
    layout = UniformLayout(SurfaceCodeModel(d=3))
    operation = make_planning_view(qubits=())

    first_claims = layout.resources_for(operation)
    second_claims = layout.resources_for(operation)

    assert first_claims == [ResourceClaim("qubits", frozenset())]
    assert second_claims == first_claims
    assert second_claims is not first_claims


def test_resources_leave_ordinary_bad_inputs_to_python_failures():
    """Missing or unhashable qubits fail naturally rather than by local guards."""
    layout = UniformLayout(SurfaceCodeModel(d=3))
    unhashable_qubits = make_planning_view(qubits=([],))

    with pytest.raises(AttributeError):
        layout.resources_for(object())
    with pytest.raises(TypeError):
        layout.resources_for(unhashable_qubits)


def test_codes_returns_a_fresh_list_without_mutating_the_layout():
    """Changing a returned code list cannot change the stored code."""
    code = SurfaceCodeModel(d=3)
    layout = UniformLayout(code)

    first_codes = layout.codes()
    second_codes = layout.codes()
    first_codes.clear()

    assert first_codes == []
    assert second_codes == [code]
    assert second_codes is not first_codes
    assert second_codes[0] is code
    assert layout.code is code


def test_removed_aliases_and_operation_annotations_stay_absent():
    """Deleted aliases and concrete operation annotations do not return."""
    layout = UniformLayout(SurfaceCodeModel(d=3))

    assert not hasattr(UniformLayout, "name")
    assert not hasattr(UniformLayout, "distance")
    assert not hasattr(layout, "name")
    assert not hasattr(layout, "distance")
    assert not hasattr(layouts_module, "Operation")
    for method_name, parameter_name in (
        ("code_for_op", "op"),
        ("spatial_nodes_for", "operation"),
        ("resources_for", "op"),
    ):
        parameter = inspect.signature(getattr(UniformLayout, method_name)).parameters[
            parameter_name
        ]
        assert parameter.annotation is inspect.Parameter.empty
    assert UniformLayout.resources_for.__doc__ == "Return one qubit exclusivity claim."


def test_timing_build_dispatches_real_planning_views_through_layout_hooks():
    """A timing-only build supplies planning views through operation hooks."""
    code = SurfaceCodeModel(d=3)
    layout = RecordingLayout(code)
    operation = make_operation(qubits=(2, 6), patches=(13,))

    RunSpec(
        ops=[operation],
        layout=layout,
        decoder=PresetLatencyDecoder(latency_us=1.0),
    ).build()

    calls_by_name = {}
    for name, value in layout.calls:
        calls_by_name.setdefault(name, []).append(value)
    assert isinstance(layout, LayoutModel)
    assert calls_by_name["codes"] == [None]
    for hook_name in ("code_for_op", "spatial_nodes_for", "resources_for"):
        assert calls_by_name[hook_name] == [
            OperationPlanningView.from_operation(operation)
        ]
    assert calls_by_name["code_for_patch"] == [13]
    assert calls_by_name["patch_spatial_nodes_for"] == [13]


def test_run_spec_owns_code_source_and_declared_code_count_checks():
    """The composition root rejects conflicting or non-singleton code sources."""
    code = SurfaceCodeModel(d=3)
    operation = make_operation()

    with pytest.raises(ValueError, match="multiple code sources"):
        RunSpec(ops=[operation], code=code, layout=UniformLayout(code)).build()

    for declared_codes in ([], [code, SurfaceCodeModel(d=5)]):
        layout = RecordingLayout(code)
        layout.codes = lambda values=declared_codes: values
        with pytest.raises(ValueError, match="exactly one code"):
            RunSpec(ops=[operation], layout=layout).build()


def test_planner_rejects_operation_selector_identity_changes():
    """Planning rejects a layout that selects a different operation code object."""
    resolved_code = SurfaceCodeModel(d=3)
    layout = RecordingLayout(resolved_code)
    layout.code_for_op = lambda operation: SurfaceCodeModel(d=3)

    with pytest.raises(ValueError, match="selected a code different"):
        RunSpec(ops=[make_operation()], layout=layout).build()


def test_planner_rejects_patch_selector_identity_changes():
    """Planning rejects a layout that selects a different patch code object."""
    resolved_code = SurfaceCodeModel(d=3)
    layout = RecordingLayout(resolved_code)
    layout.code_for_patch = lambda patch_identity: SurfaceCodeModel(d=3)

    with pytest.raises(ValueError, match="selected a code different"):
        RunSpec(ops=[make_operation()], layout=layout).build()
