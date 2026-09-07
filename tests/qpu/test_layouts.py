"""The layout: which code every patch and every operation runs on.

The layout is the seam between a workload and the code cards it runs on
(decsim/qpu/layouts.py). Today one layout ships, the uniform one, which
gives every patch the same card; a mixed-distance or mixed-code machine
arrives as another class on the same port, so the tests below are the
port's laws, not the uniform class's shape.

Its second job is the resource claim: an operation holds its qubits
exclusively while it runs, which is what keeps two operations off one
patch without a dependency edge between them.

The root and the plan check the layout once, at build: a run has exactly
one declared code, and every selector must return that same object, so
a code card cannot change under a run that has already priced its
geometry.
"""

import pytest

import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.layouts as layouts
import decsim.qpu.settings as qpu_settings
import decsim.records.program as program_records


class RecordingLayout:
    """A layout written outside decsim: the port's hooks, and what it heard."""

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

    def patch_spatial_nodes_for(
        self, patch_identity, *, base_spatial_node_count
    ):
        self.calls.append(("patch_spatial_nodes_for", patch_identity))
        return base_spatial_node_count

    def resources_for(self, operation):
        self.calls.append(("resources_for", operation))
        qubits = frozenset(operation.qubits)
        return [program_records.ResourceClaim("qubits", qubits)]


def one_operation(qubits=(3, 5), patches=(11,)):
    return program_records.Operation(
        id=4, name="timing-only", qubits=qubits, patches=patches
    )


def planning_view(qubits=(3, 5)):
    operation = one_operation(qubits=qubits)
    return program_records.OperationPlanningView.from_operation(operation)


def settings_with(layout=None, code=None):
    """One timing-only operation on the layout or the card given."""
    operation = one_operation()
    workload = workload_settings.WorkloadSettings(operations=[operation])
    qpu = qpu_settings.QpuSettings(code=code, layout=layout)
    decoder = decoders.PresetLatencyDecoder(latency_us=1.0)
    weak_decoder = decoder_settings.DecoderSettings(decoder=decoder)
    return machine_module.MachineSettings(
        workload=workload, qpu=qpu, weak_decoder=weak_decoder
    )


def test_the_uniform_layout_gives_every_operation_and_patch_one_card():
    card = code_geometry.SurfaceCodeModel(distance=3)
    layout = layouts.UniformLayout(card)
    operation = planning_view()
    assert layout.code_for_op(operation) is card
    assert layout.code_for_patch(11) is card
    assert layout.codes() == [card]


def test_the_uniform_layout_claims_the_operations_qubits_exclusively():
    """One claim per operation, its qubits deduplicated into a frozenset."""
    card = code_geometry.SurfaceCodeModel(distance=3)
    layout = layouts.UniformLayout(card)
    operation = planning_view(qubits=(8, 2, 8, 5))

    claims = layout.resources_for(operation)

    claimed_qubits = frozenset({2, 5, 8})
    expected = program_records.ResourceClaim("qubits", claimed_qubits)
    assert claims == [expected]


def test_an_operation_on_no_qubits_claims_nothing():
    card = code_geometry.SurfaceCodeModel(distance=3)
    layout = layouts.UniformLayout(card)
    operation = planning_view(qubits=())

    claims = layout.resources_for(operation)

    no_qubits = frozenset()
    expected = program_records.ResourceClaim("qubits", no_qubits)
    assert claims == [expected]


def test_a_layout_written_outside_decsim_hears_every_hook_of_a_run():
    """A run asks the layout for its codes, then per operation and patch."""
    card = code_geometry.SurfaceCodeModel(distance=3)
    layout = RecordingLayout(card)
    settings = settings_with(layout=layout)
    machine = machine_module.Machine.build(settings)
    machine.run()

    calls_by_name = {}
    for name, value in layout.calls:
        heard = calls_by_name.setdefault(name, [])
        heard.append(value)
    operation = one_operation()
    view = program_records.OperationPlanningView.from_operation(operation)
    assert calls_by_name["codes"] == [None]
    assert calls_by_name["code_for_op"] == [view]
    assert calls_by_name["spatial_nodes_for"] == [view]
    assert calls_by_name["resources_for"] == [view]
    assert calls_by_name["code_for_patch"] == [11]
    assert calls_by_name["patch_spatial_nodes_for"] == [11]


def test_a_card_and_a_layout_together_are_refused_as_two_code_sources():
    card = code_geometry.SurfaceCodeModel(distance=3)
    layout = layouts.UniformLayout(card)
    settings = settings_with(layout=layout, code=card)
    with pytest.raises(ValueError, match="multiple code sources"):
        machine_module.Machine.build(settings)


def test_a_layout_that_declares_no_code_is_refused():
    card = code_geometry.SurfaceCodeModel(distance=3)
    layout = RecordingLayout(card)
    layout.codes = list
    settings = settings_with(layout=layout)
    with pytest.raises(ValueError, match="exactly one code"):
        machine_module.Machine.build(settings)


def test_a_layout_that_declares_two_codes_is_refused():
    card = code_geometry.SurfaceCodeModel(distance=3)
    other = code_geometry.SurfaceCodeModel(distance=5)
    layout = RecordingLayout(card)

    def two_codes():
        return [card, other]

    layout.codes = two_codes
    settings = settings_with(layout=layout)
    with pytest.raises(ValueError, match="exactly one code"):
        machine_module.Machine.build(settings)


def test_an_operation_selector_that_returns_another_card_is_refused():
    """Equal is not enough: the plan priced the card it resolved."""
    card = code_geometry.SurfaceCodeModel(distance=3)
    layout = RecordingLayout(card)

    def another_card(operation):
        del operation
        return code_geometry.SurfaceCodeModel(distance=3)

    layout.code_for_op = another_card
    settings = settings_with(layout=layout)
    with pytest.raises(ValueError, match="selected a code different"):
        machine_module.Machine.build(settings)


def test_a_patch_selector_that_returns_another_card_is_refused():
    card = code_geometry.SurfaceCodeModel(distance=3)
    layout = RecordingLayout(card)

    def another_card(patch_identity):
        del patch_identity
        return code_geometry.SurfaceCodeModel(distance=3)

    layout.code_for_patch = another_card
    settings = settings_with(layout=layout)
    with pytest.raises(ValueError, match="selected a code different"):
        machine_module.Machine.build(settings)
