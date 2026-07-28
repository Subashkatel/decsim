from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from decsim.decoders import PerRoundDecoder
from decsim.devices import TimingOnlyDevice
from decsim.codes import SurfaceCodeModel
from decsim.engine import Engine
from decsim.layouts import UniformLayout
from decsim.message import DecodeResult, Operation, RunSeedReservation
from decsim.planner import FixedRounds, WindowPlanner
from decsim.run_spec import RunSpec, simulate
from decsim.schemes import ParallelWindowScheme, SlidingWindowScheme
from decsim.switching import Switching


class WrongBoundarySignature:
    speculative = False

    def on_commit(self):
        return True


class LegacyBoundaryPolicy:
    def on_commit(self, window, final):
        return True


class StaticOnlyDevice:
    operation_circuit_scope = "none"

    def begin_operation(self, operation):
        return None

    def round_payloads(self, operation, round_index):
        return TimingOnlyDevice().round_payloads(operation, round_index)

    def window_models_for_operation(
        self, operation, windows, round_count, *, belief_matching=False,
    ):
        return []


class MissingCircuitScopeDevice:
    def begin_operation(self, operation):
        return None

    def round_payloads(self, operation, round_index):
        return []

    def window_models_for_operation(
        self, operation, windows, round_count, *, belief_matching=False,
    ):
        return []


class StaticDecoder:
    def latency(self, job):
        return 1

    def decode(self, job):
        return DecodeResult(job.op_id, job.window_id,
                            logical_observables=(0,))


class SeedRecordingDecoder(StaticDecoder):
    def __init__(self, *, fail_reservation=False, source_override=None):
        self.fail_reservation = fail_reservation
        self.source_override = source_override
        self.reserved = []
        self.cancelled = []
        self.committed = []

    def reserve_run_seed(self, seed):
        if self.fail_reservation:
            raise ValueError("deliberate reservation failure")
        reservation = RunSeedReservation(
            proposed_seed_source=(
                self.source_override
                or ("entropy" if seed is None else "derived")
            ),
            proposed_seed=seed,
            prepared_state=object(),
        )
        self.reserved.append(reservation)
        return reservation

    def cancel_run_seed(self, reservation):
        self.cancelled.append(reservation)

    def commit_run_seed(self, reservation):
        self.committed.append(reservation)


class RecordingPlanner:
    def __init__(self, scheme, layout, rounds_policy):
        self.scheme = scheme
        self.layout = layout
        self.rounds_policy = rounds_policy
        self.calls = 0

    def plan(self, operations):
        self.calls += 1
        return WindowPlanner(
            self.scheme,
            self.layout,
            self.rounds_policy,
        ).plan(operations)


class PlanningViewRecordingPlanner(RecordingPlanner):
    def __init__(self, scheme, layout, rounds_policy):
        super().__init__(scheme, layout, rounds_policy)
        self.operations = ()
        self.plan_result = None

    def plan(self, operations):
        self.operations = tuple(operations)
        self.plan_result = super().plan(operations)
        return self.plan_result


class CircuitRecordingTimingDevice(TimingOnlyDevice):
    operation_circuit_scope = "none"

    def __init__(self):
        self.circuits_seen = []

    def begin_operation(self, operation):
        self.circuits_seen.append(operation.circuit)


class CircuitRecordingActiveDevice(CircuitRecordingTimingDevice):
    operation_circuit_scope = "per_operation"


class StaticFrontend:
    def __init__(self, operations):
        self.operations = operations
        self.calls = 0

    def build(self):
        self.calls += 1
        return self.operations


class SelectorLayout(UniformLayout):
    def __init__(
        self,
        declared_code,
        *,
        operation_code=None,
        patch_codes=None,
    ):
        super().__init__(declared_code)
        self.operation_code = operation_code or declared_code
        self.patch_codes = dict(patch_codes or {})

    def code_for_op(self, operation):
        return self.operation_code

    def code_for_patch(self, patch_id):
        return self.patch_codes.get(patch_id, self.code)


class CircuitRejectingLayout(UniformLayout):
    def __init__(self, code):
        super().__init__(code)
        self.operation_calls = 0
        self.resource_calls = 0
        self.spatial_calls = 0

    @staticmethod
    def _require_planning_view(operation):
        assert not hasattr(operation, "circuit")

    def code_for_op(self, operation):
        self._require_planning_view(operation)
        self.operation_calls += 1
        return super().code_for_op(operation)

    def resources_for(self, operation):
        self._require_planning_view(operation)
        self.resource_calls += 1
        return super().resources_for(operation)

    def spatial_nodes_for(self, operation):
        self._require_planning_view(operation)
        self.spatial_calls += 1
        return super().spatial_nodes_for(operation)


class CircuitRejectingRounds(FixedRounds):
    def __init__(self, round_count):
        super().__init__(round_count)
        self.calls = 0

    def rounds_for(self, operation, code):
        assert not hasattr(operation, "circuit")
        self.calls += 1
        return super().rounds_for(operation, code)


class CircuitRejectingScheme(SlidingWindowScheme):
    def __init__(self):
        self.data_complete_calls = 0

    def data_complete(self, window, *, op, **kwargs):
        assert not hasattr(op, "circuit")
        self.data_complete_calls += 1
        return super().data_complete(window, op=op, **kwargs)


class EngineBoundFactory:
    def __init__(self, engine):
        self.engine = engine

    def request(self, operation_id, callback):
        callback()

    def shutdown(self):
        return None


class PrebindingSeedConsumer:
    def __init__(self):
        self.entries = 0

    def reserve_run_seed(self, seed):
        raise AssertionError("pre-binding consumer must not reserve")

    def commit_run_seed(self, reservation):
        raise AssertionError("pre-binding consumer must not commit")

    def cancel_run_seed(self, reservation):
        raise AssertionError("pre-binding consumer must not cancel")

    def build(self):
        self.entries += 1
        return []


class SeedConsumingProviderOwner:
    def __init__(self):
        self.entries = 0

    def reserve_run_seed(self, seed):
        raise AssertionError("pre-binding provider must not reserve")

    def commit_run_seed(self, reservation):
        raise AssertionError("pre-binding provider must not commit")

    def cancel_run_seed(self, reservation):
        raise AssertionError("pre-binding provider must not cancel")

    def make_controller(self, engine):
        self.entries += 1
        raise AssertionError("pre-binding provider must not be called")


class PlainControllerProvider:
    def __init__(self, engine):
        self.engine = engine


class DescriptorPlanner:
    scheme = SlidingWindowScheme()
    rounds_policy = FixedRounds(3)

    def __init__(self):
        self.layout_entries = 0

    @property
    def layout(self):
        self.layout_entries += 1
        return UniformLayout(SurfaceCodeModel(3))

    def plan(self, operations):
        raise AssertionError("descriptor-backed planner must not plan")


@pytest.mark.parametrize("seed", [True, 1.0, "1", object()])
def test_run_seed_rejects_non_integral_values_before_build(seed):
    spec = RunSpec(ops=[], seed=seed)

    with pytest.raises(
        TypeError,
        match=r"seed.*64-bit unsigned integer or None",
    ):
        spec.validate()


@pytest.mark.parametrize("seed", [-1, 1 << 64])
def test_run_seed_rejects_values_outside_unsigned_64_bit_domain(seed):
    spec = RunSpec(ops=[], seed=seed)

    with pytest.raises(
        ValueError,
        match=r"seed.*0.*2\*\*64",
    ):
        spec.validate()


@pytest.mark.parametrize(
    "seed",
    [None, 0, (1 << 64) - 1, np.int64(7), np.uint64(7)],
)
def test_run_seed_accepts_none_and_unsigned_integral_values(seed):
    RunSpec(ops=[], seed=seed).validate()


@pytest.mark.parametrize("seed", [None, 7])
@pytest.mark.parametrize("entrypoint", ["validate", "build"])
def test_prebinding_frontend_seed_consumer_rejects_before_entry(
    seed,
    entrypoint,
):
    frontend = PrebindingSeedConsumer()
    spec = RunSpec(frontend=frontend, seed=seed)

    with pytest.raises(
        ValueError,
        match=r"frontend.*RunSeedConsumer.*before run-seed binding",
    ):
        getattr(spec, entrypoint)()

    assert frontend.entries == 0


@pytest.mark.parametrize("entrypoint", ["validate", "build"])
def test_bound_provider_seed_consumer_rejects_before_signature_or_entry(
    entrypoint,
):
    owner = SeedConsumingProviderOwner()
    spec = RunSpec(
        ops=[],
        make_controller=owner.make_controller,
    )

    with pytest.raises(
        ValueError,
        match=r"make_controller.*RunSeedConsumer.*before run-seed binding",
    ):
        getattr(spec, entrypoint)()

    assert owner.entries == 0


def test_plain_class_controller_provider_is_accepted_without_instantiation():
    spec = RunSpec(
        ops=[],
        make_controller=PlainControllerProvider,
    )

    spec.validate()


@pytest.mark.parametrize("entrypoint", ["validate", "build"])
def test_descriptor_backed_planner_child_rejects_without_descriptor_entry(
    entrypoint,
):
    planner = DescriptorPlanner()
    spec = RunSpec(ops=[], planner=planner)

    with pytest.raises(
        TypeError,
        match=r"planner\.layout.*stored non-descriptor",
    ):
        getattr(spec, entrypoint)()

    assert planner.layout_entries == 0


def test_planner_receives_frozen_circuit_free_operation_view():
    layout = UniformLayout(SurfaceCodeModel(3))
    planner = PlanningViewRecordingPlanner(
        SlidingWindowScheme(),
        layout,
        FixedRounds(3),
    )
    RunSpec(
        ops=[Operation(0, "memory", (0,), circuit=object())],
        planner=planner,
        decoder=StaticDecoder(),
    ).build()

    assert len(planner.operations) == 1
    planning_operation = planner.operations[0]
    assert not hasattr(planning_operation, "circuit")
    with pytest.raises(FrozenInstanceError):
        planning_operation.name = "mutated"


def test_runtime_plan_does_not_alias_the_planner_return_value():
    layout = UniformLayout(SurfaceCodeModel(3))
    planner = PlanningViewRecordingPlanner(
        SlidingWindowScheme(),
        layout,
        FixedRounds(3),
    )
    completed = RunSpec(
        ops=[Operation(0, "memory", (0,))],
        planner=planner,
        decoder=StaticDecoder(),
    ).build()

    returned_window = planner.plan_result.windows[(0, 0)]
    runtime_window = completed.window_manager.windows[(0, 0)]
    assert runtime_window is not returned_window
    assert completed.window_manager.window_count is not (
        planner.plan_result.window_count
    )
    assert completed.window_manager.op_windows is not (
        planner.plan_result.op_windows
    )
    assert completed.window_manager.successors is not (
        planner.plan_result.successors
    )

    expected_commit_hi = runtime_window.commit_hi
    returned_window.commit_hi += 100
    planner.plan_result.window_count[0] += 100

    assert runtime_window.commit_hi == expected_commit_hi
    assert completed.window_manager.window_count[0] == 1


def test_planning_view_fields_are_exactly_operation_fields_without_circuit():
    from decsim.message import OperationPlanningView

    operation_fields = {item.name for item in fields(Operation)}
    planning_fields = {item.name for item in fields(OperationPlanningView)}

    assert planning_fields == operation_fields - {"circuit"}


def test_all_operation_consuming_planning_ports_receive_frozen_views():
    code = SurfaceCodeModel(3)
    layout = CircuitRejectingLayout(code)
    rounds = CircuitRejectingRounds(3)
    scheme = CircuitRejectingScheme()

    RunSpec(
        ops=[Operation(0, "memory", (0,), circuit=object())],
        layout=layout,
        rounds_policy=rounds,
        scheme=scheme,
        decoder=StaticDecoder(),
    ).build()

    assert layout.operation_calls == 2
    assert layout.resource_calls == 1
    assert layout.spatial_calls == 1
    assert rounds.calls == 2
    assert scheme.data_complete_calls > 0


def test_late_dynamic_stream_is_frozen_before_planning():
    code = SurfaceCodeModel(3)
    layout = CircuitRejectingLayout(code)
    rounds = CircuitRejectingRounds(7)
    completed = RunSpec(
        ops=[],
        layout=layout,
        rounds_policy=rounds,
        decoder=StaticDecoder(),
        feedback_boundary_mode="measurement_closed",
    ).build()
    caller_operation = Operation(
        7,
        "late stream",
        (0,),
        circuit=object(),
        feedback_boundary_mode=None,
    )
    rounds_calls_before_registration = rounds.calls

    completed.window_manager.register_dynamic_stream(
        caller_operation,
        code,
    )

    runtime_operation = completed.window_manager.ops[7]
    planning_view = (
        completed.window_manager._planning_view_by_operation_id[7]
    )
    assert runtime_operation is not caller_operation
    assert runtime_operation.feedback_boundary_mode == "measurement_closed"
    assert caller_operation.feedback_boundary_mode is None
    assert not hasattr(planning_view, "circuit")
    assert planning_view.feedback_boundary_mode == "measurement_closed"
    assert rounds.calls == rounds_calls_before_registration + 1


def test_run_uses_private_operation_copy_without_mutating_caller():
    caller_operation = Operation(
        0,
        "memory",
        (0,),
        circuit=object(),
        feedback_boundary_mode=None,
    )
    device = CircuitRecordingTimingDevice()

    RunSpec(
        ops=[caller_operation],
        device=device,
        decoder=StaticDecoder(),
        feedback_boundary_mode="measurement_closed",
    ).build()

    assert caller_operation.feedback_boundary_mode is None
    assert caller_operation.circuit is not None
    assert device.circuits_seen == [None]


def test_device_without_circuit_scope_rejects_during_validation():
    spec = RunSpec(
        ops=[],
        device=MissingCircuitScopeDevice(),
        decoder=StaticDecoder(),
    )

    with pytest.raises(
        TypeError,
        match=r"device.*operation_circuit_scope.*stored.*none.*per_operation",
    ):
        spec.validate()


def test_active_stim_circuit_is_reconstructed_for_private_execution():
    stim = pytest.importorskip("stim")
    caller_circuit = stim.Circuit("X 0")
    device = CircuitRecordingActiveDevice()

    completed = RunSpec(
        ops=[Operation(0, "memory", (0,), circuit=caller_circuit)],
        device=device,
        decoder=StaticDecoder(),
    ).build()

    assert len(device.circuits_seen) == 1
    private_circuit = device.circuits_seen[0]
    assert type(private_circuit) is stim.Circuit
    assert private_circuit is not caller_circuit
    assert str(private_circuit) == str(caller_circuit)
    workload_components = [
        component
        for component in completed.manifest.to_json_value()["components"]
        if component["component_path"][0]["value"] == "workload_circuits"
    ]
    assert [component["component_path"] for component in workload_components] == [
        [
            {"kind": "field", "value": "workload_circuits"},
            {"kind": "integer_key", "value": "0"},
        ],
    ]


def test_active_custom_circuit_rejects_before_device_entry():
    device = CircuitRecordingActiveDevice()

    with pytest.raises(TypeError, match="not an exact stim.Circuit"):
        RunSpec(
            ops=[Operation(0, "memory", (0,), circuit=object())],
            device=device,
            decoder=StaticDecoder(),
        ).build()

    assert device.circuits_seen == []


def test_run_root_binds_nested_consumers_by_stable_semantic_path():
    from decsim.decoders import CodeRouter
    from decsim.message import RunSeedPathSegment
    from decsim.run_spec import _derive_run_component_seed

    default = SeedRecordingDecoder()
    surface = SeedRecordingDecoder()
    RunSpec(
        ops=[],
        decoder=default,
        router=CodeRouter(default, {"surface": surface}),
        seed=7,
    ).build()

    default_path = (
        RunSeedPathSegment("field", "decoder_router"),
        RunSeedPathSegment("field", "default"),
    )
    surface_path = (
        RunSeedPathSegment("field", "decoder_router"),
        RunSeedPathSegment("field", "by_code"),
        RunSeedPathSegment("string_key", "surface"),
    )
    assert [item.proposed_seed for item in default.committed] == [
        _derive_run_component_seed(7, default_path),
    ]
    assert [item.proposed_seed for item in surface.committed] == [
        _derive_run_component_seed(7, surface_path),
    ]


def test_component_graph_records_shared_decoder_alias_once():
    from decsim.decoders import CodeRouter

    shared = SeedRecordingDecoder()
    completed = RunSpec(
        ops=[],
        decoder=shared,
        router=CodeRouter(
            default=shared,
            by_code={"surface": shared},
        ),
        seed=7,
    ).build()
    manifest = completed.manifest.to_json_value()

    assert manifest["aliases"] == [
        {
            "alias_path": [
                {"kind": "field", "value": "decoder_router"},
                {"kind": "field", "value": "default"},
            ],
            "canonical_path": [
                {"kind": "field", "value": "decoder_router"},
                {"kind": "field", "value": "by_code"},
                {"kind": "string_key", "value": "surface"},
            ],
        },
    ]
    shared_bindings = [
        binding
        for binding in manifest["seed_bindings"]
        if binding["component_path"][0]["value"] == "decoder_router"
    ]
    assert len(shared_bindings) == 1
    assert shared_bindings[0]["component_path"] == (
        manifest["aliases"][0]["canonical_path"]
    )


@pytest.mark.parametrize(
    ("kind", "value", "error"),
    [
        ("field", "", ValueError),
        ("field", 1, TypeError),
        ("string_key", 1, TypeError),
        ("none_key", "none", ValueError),
        ("integer_key", 0, TypeError),
        ("integer_key", "", ValueError),
        ("integer_key", "01", ValueError),
        ("integer_key", "+1", ValueError),
        ("integer_key", "-0", ValueError),
        ("unknown", "value", ValueError),
    ],
)
def test_public_typed_path_segment_rejects_noncanonical_wire_values(
    kind,
    value,
    error,
):
    from decsim.run_spec import TypedPathSegmentRecord

    with pytest.raises(error):
        TypedPathSegmentRecord(kind, value)


def test_public_typed_path_segment_has_an_exact_two_key_wire_form():
    from decsim.run_spec import TypedPathSegmentRecord

    records = [
        TypedPathSegmentRecord("field", "owner"),
        TypedPathSegmentRecord("string_key", ""),
        TypedPathSegmentRecord("none_key", None),
        TypedPathSegmentRecord("integer_key", "-9"),
        TypedPathSegmentRecord("integer_key", "0"),
    ]

    assert [record.to_json_value() for record in records] == [
        {"kind": "field", "value": "owner"},
        {"kind": "string_key", "value": ""},
        {"kind": "none_key", "value": None},
        {"kind": "integer_key", "value": "-9"},
        {"kind": "integer_key", "value": "0"},
    ]


def test_stable_identity_record_round_trips_structured_patch_keys():
    from decsim.run_spec import StableIdentityRecord

    identities = (
        1,
        "1",
        (),
        (1,),
        ("1",),
        ("gross_0", (5, "north")),
        "\N{GRINNING FACE}",
    )
    records = tuple(
        StableIdentityRecord.from_identity(identity)
        for identity in identities
    )

    assert tuple(record.to_identity() for record in records) == identities
    assert len({record.canonical_bytes() for record in records}) == len(records)
    assert records[-2].to_json_value() == {
        "kind": "tuple",
        "value": None,
        "items": [
            {"kind": "string", "value": "gross_0", "items": None},
            {
                "kind": "tuple",
                "value": None,
                "items": [
                    {"kind": "integer", "value": "5", "items": None},
                    {"kind": "string", "value": "north", "items": None},
                ],
            },
        ],
    }


def test_surrogate_code_units_are_not_stable_identities():
    from decsim.message import is_stable_identity

    assert not is_stable_identity("\ud800")
    assert not is_stable_identity("\udfff")
    assert not is_stable_identity(("gross", ("\ud800", 5)))
    assert is_stable_identity("\\ud800")
    assert is_stable_identity("\N{GRINNING FACE}")


def test_manifest_json_validator_rejects_surrogates_and_hostile_containers():
    from decsim.run_spec import _validated_json_value

    class HostileList(list):
        def __iter__(self):
            raise AssertionError("hostile list was traversed")

    class HostileDict(dict):
        def items(self):
            raise AssertionError("hostile dict was traversed")

    with pytest.raises(TypeError, match="Unicode scalar"):
        _validated_json_value({"value": "\ud800"})
    with pytest.raises(TypeError, match="Unicode scalar"):
        _validated_json_value({"\udfff": "value"})
    with pytest.raises(TypeError, match="closed JSON domain"):
        _validated_json_value(HostileList())
    with pytest.raises(TypeError, match="closed JSON domain"):
        _validated_json_value(HostileDict())

    value = {"\N{GRINNING FACE}": ["\\ud800"]}
    assert _validated_json_value(value) == value


@pytest.mark.parametrize(
    (
        "code_round_us",
        "run_round_us",
        "timing_round_us",
        "expected_round_us",
        "expected_origin",
    ),
    [
        (2.5, 1.5, 4.5, 2.5, "code.round_us"),
        (None, 1.5, 4.5, 1.5, "run_spec.round_us"),
        (None, None, 4.5, 4.5, "timing.round_us"),
    ],
)
def test_manifest_records_every_code_selection_and_effective_cadence(
    code_round_us,
    run_round_us,
    timing_round_us,
    expected_round_us,
    expected_origin,
):
    from decsim.config import TimingConfig, us

    structured_patch = ("gross_0", (5, "north"))
    completed = RunSpec(
        ops=[
            Operation(
                7,
                "memory",
                (structured_patch,),
                patches=(structured_patch,),
            ),
        ],
        code=SurfaceCodeModel(3, round_us=code_round_us),
        round_us=run_round_us,
        timing=TimingConfig(round_us=timing_round_us),
        decoder=StaticDecoder(),
    ).build()
    manifest = completed.manifest.to_json_value()

    assert manifest["code_selections"] == [
        {
            "consumer_kind": "operation",
            "consumer_identity": {
                "kind": "integer",
                "value": "7",
                "items": None,
            },
            "code_path": [{"kind": "field", "value": "code"}],
        },
        {
            "consumer_kind": "patch",
            "consumer_identity": {
                "kind": "tuple",
                "value": None,
                "items": [
                    {
                        "kind": "string",
                        "value": "gross_0",
                        "items": None,
                    },
                    {
                        "kind": "tuple",
                        "value": None,
                        "items": [
                            {
                                "kind": "integer",
                                "value": "5",
                                "items": None,
                            },
                            {
                                "kind": "string",
                                "value": "north",
                                "items": None,
                            },
                        ],
                    },
                ],
            },
            "code_path": [{"kind": "field", "value": "code"}],
        },
    ]
    assert manifest["cadences"] == [
        {
            "consumer_kind": selection["consumer_kind"],
            "consumer_identity": selection["consumer_identity"],
            "code_path": selection["code_path"],
            "round_ticks": us(expected_round_us),
            "origin": expected_origin,
        }
        for selection in manifest["code_selections"]
    ]


def test_resource_claim_id_is_not_promoted_to_a_patch_selector():
    from decsim.message import ResourceClaim

    class RecordingLayout(UniformLayout):
        def __init__(self, code):
            super().__init__(code)
            self.patch_calls = []

        def code_for_patch(self, patch_id):
            self.patch_calls.append(patch_id)
            return self.code

        def resources_for(self, operation):
            return [
                ResourceClaim(
                    "ancilla",
                    frozenset({"reserved-only"}),
                ),
            ]

    layout = RecordingLayout(SurfaceCodeModel(3))
    completed = RunSpec(
        ops=[Operation(0, "memory", (0,), patches=("data",))],
        layout=layout,
        decoder=StaticDecoder(),
    ).build()

    assert ("ancilla", "reserved-only") not in layout.patch_calls
    patch_identities = [
        record["consumer_identity"]
        for record in completed.manifest.to_json_value()["code_selections"]
        if record["consumer_kind"] == "patch"
    ]
    assert patch_identities == [
        {"kind": "string", "value": "data", "items": None},
    ]


def test_component_graph_records_the_resolved_orchestrator_frame():
    completed = RunSpec(ops=[], decoder=StaticDecoder()).build()
    paths = [
        component["component_path"]
        for component in completed.manifest.to_json_value()["components"]
    ]

    assert [
        {"kind": "field", "value": "orchestrator"},
        {"kind": "field", "value": "frame"},
    ] in paths


@pytest.mark.parametrize(
    ("workload_fields", "expected_optional_fields"),
    [
        ({"ops": []}, set()),
        ({"frontend": StaticFrontend([])}, {"frontend"}),
    ],
)
def test_component_graph_contains_every_resolved_singleton_root(
    workload_fields,
    expected_optional_fields,
):
    completed = RunSpec(
        decoder=StaticDecoder(),
        **workload_fields,
    ).build()
    singleton_fields = {
        component["component_path"][0]["value"]
        for component in completed.manifest.to_json_value()["components"]
        if len(component["component_path"]) == 1
    }

    assert {
        "code",
        "layout",
        "scheme",
        "rounds_policy",
        "planner",
        "device",
        "decoder_router",
        "magic_state_factory",
        "strategy",
        "scheduler",
        "deadline_policy",
        "boundary_policy",
        "window_interaction",
        "idle_policy",
        "orchestrator",
        "controller",
    } | expected_optional_fields <= singleton_fields
    assert ("frontend" in singleton_fields) == bool(expected_optional_fields)
    assert "memory_model" not in singleton_fields


def test_manifest_records_the_exact_fixed_composition_anchors():
    from decsim.payload_store import PayloadStore

    completed = RunSpec(ops=[], decoder=StaticDecoder()).build()
    manifest = completed.manifest.to_json_value()

    assert [
        tuple(segment["value"] for segment in record["component_path"])
        for record in manifest["fixed_composition"]
    ] == [
        ("fixed", "chip"),
        ("fixed", "engine"),
        ("fixed", "cluster"),
        ("fixed", "payload_store"),
        ("fixed", "clocked_device"),
        ("fixed", "window_manager"),
        ("fixed", "decoder_manager"),
        ("fixed", "strategy_services"),
    ]
    fixed_by_path = {
        tuple(
            segment["value"]
            for segment in record["component_path"]
        ): record
        for record in manifest["fixed_composition"]
    }
    assert fixed_by_path[("fixed", "payload_store")]["implementation"] == (
        "decsim.payload_store.PayloadStore"
    )
    assert isinstance(completed.window_manager.store, PayloadStore)
    assert completed.chip.source.cluster is completed.cluster
    assert completed.chip.frame is completed.orchestrator.frame
    assert completed.window_manager.links is completed.controller.links


def test_controller_port_requires_the_actual_retained_links():
    class MissingLinksController:
        def relay_syndrome(self, payload, deliver):
            deliver(payload)

        def relay_instruction(self, decision, deliver):
            deliver(decision)

    with pytest.raises(TypeError, match=r"controller.*Controller"):
        RunSpec(
            ops=[],
            decoder=StaticDecoder(),
            make_controller=lambda _engine: MissingLinksController(),
        ).build()


def test_orchestrator_port_requires_the_actual_retained_frame():
    class MissingFrameOrchestrator:
        def connect(self, controller, decision_sink):
            return None

        def register_blocked_operation(self, blocked_op_id, blocking_op_id):
            return None

        def integrate(self, op, outcome):
            return None

    with pytest.raises(TypeError, match=r"orchestrator.*Orchestrator"):
        RunSpec(
            ops=[],
            decoder=StaticDecoder(),
            orchestrator=MissingFrameOrchestrator(),
        ).build()


def test_factory_decode_service_is_the_fixed_cluster_alias():
    from decsim.factories import DistillationFactory

    completed = RunSpec(
        ops=[],
        decoder=StaticDecoder(),
        make_factory=lambda engine, cluster: DistillationFactory(
            engine,
            num_units=1,
            cycle_ticks=1,
            decode_service=cluster,
            corr_rounds=1,
            n_corr=1,
        ),
    ).build()
    manifest = completed.manifest.to_json_value()

    assert {
        "alias_path": [
            {"kind": "field", "value": "magic_state_factory"},
            {"kind": "field", "value": "decode_service"},
        ],
        "canonical_path": [
            {"kind": "field", "value": "fixed"},
            {"kind": "field", "value": "cluster"},
        ],
    } in manifest["aliases"]
    assert not any(
        component["component_path"] == [
            {"kind": "field", "value": "magic_state_factory"},
            {"kind": "field", "value": "decode_service"},
        ]
        for component in manifest["components"]
    )


def test_factory_rejects_a_foreign_correction_decode_service():
    from decsim.factories import DistillationFactory

    foreign_service = object()
    with pytest.raises(
        ValueError,
        match=r"DistillationFactory.*decode_service.*run-owned cluster",
    ):
        RunSpec(
            ops=[],
            decoder=StaticDecoder(),
            make_factory=lambda engine, _cluster: DistillationFactory(
                engine,
                num_units=1,
                cycle_ticks=1,
                decode_service=foreign_service,
                corr_rounds=1,
                n_corr=1,
            ),
        ).build()


def test_shipped_manifest_part_declares_the_default_factory_configuration():
    completed = RunSpec(ops=[], decoder=StaticDecoder()).build()
    factory_component = next(
        component
        for component in completed.manifest.to_json_value()["components"]
        if component["component_path"] == [
            {"kind": "field", "value": "magic_state_factory"},
        ]
    )

    assert factory_component["configuration"] == {"kind": "infinite"}
    assert factory_component["configuration_status"] == "declared"


def test_external_manifest_configuration_is_copied_before_seed_reservation():
    events = []

    class ConfiguredFactory:
        def __init__(self, engine):
            self.engine = engine
            self.config = {"mode": "external"}

        def run_manifest_config(self):
            events.append("config")
            return self.config

        def reserve_run_seed(self, seed):
            events.append("reserve")
            return RunSeedReservation(
                proposed_seed_source="derived",
                proposed_seed=seed,
                prepared_state=None,
            )

        def commit_run_seed(self, reservation):
            events.append("commit")

        def cancel_run_seed(self, reservation):
            events.append("cancel")

        def request(self, op_id, callback):
            callback()

        def shutdown(self):
            return None

    built = []

    def make_factory(engine, _cluster):
        factory = ConfiguredFactory(engine)
        built.append(factory)
        return factory

    completed = RunSpec(
        ops=[],
        decoder=StaticDecoder(),
        make_factory=make_factory,
        seed=9,
    ).build()
    assert events == ["config", "reserve", "commit"]

    first = completed.manifest.to_json_value()
    factory_component = next(
        component
        for component in first["components"]
        if component["component_path"] == [
            {"kind": "field", "value": "magic_state_factory"},
        ]
    )
    assert factory_component["configuration"] == {"mode": "external"}
    assert factory_component["configuration_status"] == "opaque"

    built[0].config["mode"] = "mutated"
    factory_component["configuration"]["mode"] = "returned-tree mutation"
    second = completed.manifest.to_json_value()
    second_factory_component = next(
        component
        for component in second["components"]
        if component["component_path"] == [
            {"kind": "field", "value": "magic_state_factory"},
        ]
    )
    assert second_factory_component["configuration"] == {"mode": "external"}


def test_shipped_manifest_configuration_drift_invalidates_the_atomic_run(
    monkeypatch,
):
    from decsim.factories import InfiniteFactory

    declarations = 0

    def drifting_configuration(self):
        nonlocal declarations
        declarations += 1
        return {"kind": "infinite", "declaration": declarations}

    monkeypatch.setattr(
        InfiniteFactory,
        "run_manifest_config",
        drifting_configuration,
    )

    with pytest.raises(
        RuntimeError,
        match=r"InfiniteFactory.*changed declared configuration",
    ):
        RunSpec(ops=[], decoder=StaticDecoder()).build()
    assert declarations == 2


def test_manifest_configuration_rejects_non_mapping_before_seed_reservation():
    events = []

    class InvalidConfiguredFactory:
        def __init__(self, engine):
            self.engine = engine

        def run_manifest_config(self):
            events.append("config")
            return []

        def reserve_run_seed(self, seed):
            events.append("reserve")
            raise AssertionError("reservation must follow configuration")

        def commit_run_seed(self, reservation):
            raise AssertionError("unreachable")

        def cancel_run_seed(self, reservation):
            raise AssertionError("unreachable")

        def request(self, op_id, callback):
            callback()

        def shutdown(self):
            return None

    with pytest.raises(TypeError, match=r"run_manifest_config.*mapping"):
        RunSpec(
            ops=[],
            decoder=StaticDecoder(),
            make_factory=lambda engine, _cluster: InvalidConfiguredFactory(
                engine
            ),
        ).build()
    assert events == ["config"]


def test_run_seed_reservation_failure_cancels_prior_leaves_without_committing():
    from decsim.decoders import CodeRouter

    acquired = SeedRecordingDecoder()
    failing = SeedRecordingDecoder(fail_reservation=True)
    spec = RunSpec(
        ops=[],
        decoder=acquired,
        router=CodeRouter(
            default=failing,
            by_code={"surface": acquired},
        ),
    )

    with pytest.raises(ValueError, match="deliberate reservation failure"):
        spec.build()

    assert acquired.cancelled == acquired.reserved
    assert acquired.committed == []


def test_run_root_rejects_leaf_metadata_that_denies_the_derived_seed():
    dishonest = SeedRecordingDecoder(source_override="explicit_local")

    with pytest.raises(
        ValueError,
        match=r"metadata.*derived component seed",
    ):
        RunSpec(ops=[], decoder=dishonest, seed=7).build()

    assert dishonest.committed == []


def test_run_spec_build_is_single_use_even_after_success():
    spec = RunSpec(ops=[], decoder=StaticDecoder())
    spec.build()

    with pytest.raises(RuntimeError, match="already complete"):
        spec.build()


def test_factory_cannot_drive_the_engine_before_seed_binding():
    def malicious_factory(engine, cluster):
        engine.run()
        return EngineBoundFactory(engine)

    with pytest.raises(RuntimeError, match="construction is guarded"):
        RunSpec(
            ops=[],
            decoder=StaticDecoder(),
            make_factory=malicious_factory,
        ).build()


def test_metric_cannot_schedule_work_after_primary_drain():
    class SchedulingMetric:
        name = "scheduling_metric"

        def __init__(self, engine):
            self.engine = engine

        def observe(self, engine):
            return None

        def result(self):
            self.engine.schedule(0, lambda: None)
            return {}

    spec = RunSpec(
        ops=[],
        decoder=StaticDecoder(),
        make_metrics=lambda engine, _cluster, _chip, _factory: [
            SchedulingMetric(engine),
        ],
    )

    with pytest.raises(RuntimeError, match="finalizing"):
        spec.build()
    with pytest.raises(RuntimeError, match="already invalid"):
        spec.build()


def test_primary_result_freezes_logical_outputs_separately_from_diagnostics():
    completed_run = RunSpec(
        ops=[Operation(0, "memory", (0,))],
        decoder=StaticDecoder(),
        rounds_policy=FixedRounds(3),
    ).build()

    assert completed_run.result.logical_results() == {0: (0,)}
    completed_run.window_manager.op_results[0] = (1,)
    assert completed_run.result.logical_results() == {0: (0,)}


@pytest.mark.parametrize(
    ("root_seed", "path_parts", "expected"),
    [
        (0, (("field", "device"),), 11874377230878857407),
        (
            0,
            (
                ("field", "decoder_router"),
                ("field", "by_code"),
                ("string_key", "x"),
            ),
            6126402147709124294,
        ),
        (
            0,
            (
                ("field", "decoder_router"),
                ("field", "by_code"),
                ("none_key", None),
            ),
            6969999416613313845,
        ),
        (
            (1 << 64) - 1,
            (("field", "device"),),
            4727110291892774543,
        ),
        (
            (1 << 64) - 1,
            (
                ("field", "decoder_router"),
                ("field", "by_code"),
                ("string_key", "x"),
            ),
            261415811638871216,
        ),
        (
            (1 << 64) - 1,
            (
                ("field", "decoder_router"),
                ("field", "by_code"),
                ("none_key", None),
            ),
            15694113753403175836,
        ),
        (
            0,
            (
                ("field", "workload_circuits"),
                ("integer_key", 9),
            ),
            6422064279578959929,
        ),
        (
            0,
            (
                ("field", "workload_circuits"),
                ("integer_key", -1),
            ),
            15629268580557925073,
        ),
    ],
)
def test_run_component_seed_derivation_matches_frozen_golden_vectors(
    root_seed,
    path_parts,
    expected,
):
    from decsim.message import RunSeedPathSegment
    from decsim.run_spec import _derive_run_component_seed

    path = tuple(
        RunSeedPathSegment(kind, value)
        for kind, value in path_parts
    )

    assert _derive_run_component_seed(root_seed, path) == expected


def test_integer_seed_path_rejects_bool_and_non_integer_values():
    from decsim.message import RunSeedPathSegment

    for value in (True, "9", 9.0):
        with pytest.raises(TypeError, match="built-in int"):
            RunSeedPathSegment("integer_key", value)


def test_run_spec_rejects_an_incompatible_boundary_policy_before_build():
    spec = RunSpec(ops=[], boundary_policy=WrongBoundarySignature())

    with pytest.raises(
        TypeError,
        match=r"boundary_policy.*BoundaryPolicy.*on_commit.*signature",
    ):
        spec.validate()


@pytest.mark.parametrize("first_field,second_field", [
    ("d", "code"),
    ("d", "layout"),
    ("code", "layout"),
])
def test_run_spec_rejects_multiple_explicit_code_sources(
    first_field, second_field,
):
    code = SurfaceCodeModel(d=5)
    values = {
        "d": 5,
        "code": code,
        "layout": UniformLayout(code),
    }
    spec = RunSpec(
        ops=[],
        **{
            first_field: values[first_field],
            second_field: values[second_field],
        },
    )

    with pytest.raises(
        ValueError,
        match=rf"multiple code sources.*{first_field}.*{second_field}",
    ):
        spec.validate()


@pytest.mark.parametrize(
    ("sibling_name", "sibling_value"),
    [
        ("d", 5),
        ("code", SurfaceCodeModel(d=5)),
        ("layout", UniformLayout(SurfaceCodeModel(d=5))),
        ("scheme", SlidingWindowScheme()),
        ("rounds_policy", FixedRounds(7)),
    ],
)
def test_supplied_planner_rejects_every_sibling_planning_owner_before_plan(
    sibling_name,
    sibling_value,
):
    code = SurfaceCodeModel(d=5)
    planner = RecordingPlanner(
        SlidingWindowScheme(),
        UniformLayout(code),
        FixedRounds(5),
    )
    spec = RunSpec(
        ops=[],
        planner=planner,
        **{sibling_name: sibling_value},
    )

    with pytest.raises(
        ValueError,
        match=rf"planner owns.*{sibling_name}",
    ):
        spec.validate()

    assert planner.calls == 0


@pytest.mark.parametrize(
    ("child_name", "invalid_child", "port_name"),
    [
        ("scheme", object(), "DecodingScheme"),
        ("layout", object(), "LayoutModel"),
        ("rounds_policy", object(), "RoundsPolicy"),
    ],
)
def test_planner_owned_children_are_validated_before_plan(
    child_name,
    invalid_child,
    port_name,
):
    code = SurfaceCodeModel(d=5)
    children = {
        "scheme": SlidingWindowScheme(),
        "layout": UniformLayout(code),
        "rounds_policy": FixedRounds(5),
    }
    children[child_name] = invalid_child
    planner = RecordingPlanner(**children)
    spec = RunSpec(ops=[], planner=planner)

    with pytest.raises(
        TypeError,
        match=rf"planner\.{child_name}.*{port_name}",
    ):
        spec.validate()

    assert planner.calls == 0


def test_cross_part_validation_reads_the_planner_owned_scheme():
    code = SurfaceCodeModel(d=5)
    planner = RecordingPlanner(
        ParallelWindowScheme(),
        UniformLayout(code),
        FixedRounds(5),
    )
    spec = RunSpec(
        ops=[Operation(0, "memory", (0,))],
        planner=planner,
        strategy=Switching(
            confidence_threshold=0.5,
            double_window=True,
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"double_window.*parallel.*not supported",
    ):
        spec.validate()

    assert planner.calls == 0


@pytest.mark.parametrize("operation_source", [
    "ops",
    "frontend",
    "decode_ops",
    "dynamic_streams",
])
def test_layout_operation_selector_must_return_the_resolved_code(
    operation_source,
):
    declared_code = SurfaceCodeModel(d=3)
    selected_code = SurfaceCodeModel(d=5)
    operation = Operation(17, "selected elsewhere", (0,), patches=(0,))
    layout = SelectorLayout(
        declared_code,
        operation_code=selected_code,
    )
    arguments = {
        "ops": [],
        "layout": layout,
        "decoder": StaticDecoder(),
    }
    frontend = None
    if operation_source == "ops":
        arguments["ops"] = [operation]
    elif operation_source == "frontend":
        frontend = StaticFrontend([operation])
        arguments.pop("ops")
        arguments["frontend"] = frontend
    else:
        arguments[operation_source] = [operation]

    with pytest.raises(
        ValueError,
        match=r"layout.*operation 17.*selected.*resolved",
    ):
        RunSpec(**arguments).build()

    if frontend is not None:
        assert frontend.calls == 1


@pytest.mark.parametrize(
    ("operation", "mismatched_patch"),
    [
        (Operation(18, "explicit", (0,), patches=(4, 5)), 5),
        (Operation(19, "qubit fallback", (6, 7)), 7),
        (Operation(20, "zero fallback", ()), 0),
    ],
)
def test_layout_patch_selector_must_return_the_resolved_code(
    operation,
    mismatched_patch,
):
    declared_code = SurfaceCodeModel(d=3)
    selected_code = SurfaceCodeModel(d=5)
    layout = SelectorLayout(
        declared_code,
        patch_codes={mismatched_patch: selected_code},
    )

    with pytest.raises(
        ValueError,
        match=rf"layout.*patch {mismatched_patch}.*selected.*resolved",
    ):
        RunSpec(
            ops=[operation],
            layout=layout,
            decoder=StaticDecoder(),
        ).build()


@pytest.mark.parametrize("declared_codes", [[], [
    SurfaceCodeModel(d=3),
    SurfaceCodeModel(d=5),
]])
def test_layout_must_declare_exactly_one_resolved_code(declared_codes):
    class InventoryLayout(UniformLayout):
        def codes(self):
            return declared_codes

    layout = InventoryLayout(SurfaceCodeModel(d=3))

    with pytest.raises(
        ValueError,
        match=rf"layout.*exactly one.*got {len(declared_codes)}",
    ):
        RunSpec(ops=[], layout=layout).validate()


def test_layout_selection_is_validated_before_engine_construction(monkeypatch):
    declared_code = SurfaceCodeModel(d=3)
    selected_code = SurfaceCodeModel(d=5)
    layout = SelectorLayout(
        declared_code,
        operation_code=selected_code,
    )

    def fail_if_engine_is_constructed(*args, **kwargs):
        raise AssertionError("engine constructed before layout validation")

    monkeypatch.setattr(
        "decsim.engine.Engine",
        fail_if_engine_is_constructed,
    )

    with pytest.raises(ValueError, match=r"layout.*operation 21"):
        RunSpec(
            ops=[Operation(21, "invalid selection", (0,))],
            layout=layout,
            decoder=StaticDecoder(),
        ).build()


def test_factory_builder_result_must_use_the_run_engine():
    foreign_engine = Engine(verbose=False)
    run_engines = []

    def build_foreign_factory(engine, cluster):
        run_engines.append(engine)
        return EngineBoundFactory(foreign_engine)

    with pytest.raises(
        ValueError,
        match=r"EngineBoundFactory.*different engine",
    ):
        RunSpec(
            ops=[Operation(0, "memory", (0,))],
            decoder=StaticDecoder(),
            make_factory=build_foreign_factory,
        ).build()

    assert len(run_engines) == 1
    assert run_engines[0] is not foreign_engine
    assert run_engines[0]._event_queue == []


def test_old_single_argument_cross_part_validator_is_rejected():
    class OldValidatorDecoder(StaticDecoder):
        def validate(self, spec):
            return None

    with pytest.raises(
        TypeError,
        match=r"decoder cross-part validator.*validate.*signature",
    ):
        RunSpec(
            ops=[],
            decoder=OldValidatorDecoder(),
        ).validate()


def test_default_build_exposes_one_resolved_planning_identity():
    completed_run = RunSpec(
        ops=[Operation(0, "memory", (0,))],
        decoder=StaticDecoder(),
    ).build()

    assert completed_run.planning.code.distance == 3
    assert completed_run.window_manager.code is completed_run.planning.code
    assert completed_run.window_manager.layout is completed_run.planning.layout
    assert completed_run.window_manager.scheme is completed_run.planning.scheme
    assert completed_run.window_manager.rounds_policy is completed_run.planning.rounds_policy
    assert not hasattr(completed_run.cluster, "planner")
    assert not hasattr(completed_run.cluster, "strategy")
    assert not hasattr(completed_run.cluster, "prepare")
    assert not hasattr(completed_run.cluster, "build_windows")


def test_legacy_boundary_policy_defaults_to_non_speculative_delivery():
    result = simulate(RunSpec(
        ops=[Operation(0, "memory", (0,))],
        rounds_policy=FixedRounds(3),
        decoder=StaticDecoder(),
        boundary_policy=LegacyBoundaryPolicy(),
    ))

    assert result.result.chip_done_ticks is not None


def test_static_device_needs_only_static_run_capabilities():
    result = simulate(RunSpec(
        ops=[Operation(0, "memory", (0,))],
        rounds_policy=FixedRounds(3),
        decoder=StaticDecoder(),
        device=StaticOnlyDevice(),
    ))

    assert result.result.chip_done_ticks is not None


def test_switching_device_must_supply_the_selected_strong_model_capability():
    spec = RunSpec(
        ops=[Operation(0, "memory", (0,))],
        decoder=StaticDecoder(),
        strategy=Switching(confidence_threshold=0.5),
        device=StaticOnlyDevice(),
    )

    with pytest.raises(
        TypeError,
        match=r"device.*SyndromeDevice.*strong_window_model_for_operation",
    ):
        spec.validate()


@pytest.mark.parametrize(
    ("field", "port"),
    [
        ("code", "CodeModel"),
        ("layout", "LayoutModel"),
        ("decoder", "Decoder"),
        ("router", "DecoderRouter"),
        ("strategy", "DecodingStrategy"),
        ("scheduler", "Scheduler"),
        ("deadline_policy", "DeadlinePolicy"),
        ("scheme", "DecodingScheme"),
        ("rounds_policy", "RoundsPolicy"),
        ("planner", "ExecutionPlanner"),
        ("idle_policy", "IdlePolicy"),
        ("orchestrator", "Orchestrator"),
        ("device", "SyndromeDevice"),
        ("memory_model", "MemoryModel"),
        ("window_interaction", "WindowInteraction"),
    ],
)
def test_every_supplied_part_is_checked_against_its_port(field, port):
    spec = RunSpec(ops=[], **{field: object()})

    with pytest.raises(TypeError, match=rf"{field}.*{port}"):
        spec.validate()


def test_frontend_and_named_decoders_are_validated_too():
    with pytest.raises(TypeError, match=r"frontend.*InputFrontend"):
        RunSpec(frontend=object()).validate()

    with pytest.raises(TypeError, match=r"decoders\['surface'\].*Decoder"):
        RunSpec(ops=[], decoders={"surface": object()}).validate()


@pytest.mark.parametrize(
    ("field", "value", "arity"),
    [
        ("make_controller", lambda: None, 1),
        ("make_factory", lambda: None, 2),
        ("make_metrics", lambda: None, 4),
    ],
)
def test_supplied_factories_must_accept_the_documented_arguments(
    field, value, arity,
):
    spec = RunSpec(ops=[], **{field: value})

    with pytest.raises(
        TypeError,
        match=rf"{field}.*must accept {arity} positional argument",
    ):
        spec.validate()


@pytest.mark.parametrize(
    ("field", "factory", "port"),
    [
        ("make_controller", lambda engine: object(), "Controller"),
        ("make_factory", lambda engine, cluster: object(), "MagicStateFactory"),
        (
            "make_metrics",
            lambda engine, cluster, chip, factory: [object()],
            "Metric",
        ),
    ],
)
def test_objects_created_by_supplied_factories_are_validated(
    field, factory, port,
):
    spec = RunSpec(
        ops=[Operation(0, "memory", (0,))],
        decoder=PerRoundDecoder(1.0),
        **{field: factory},
    )

    with pytest.raises(TypeError, match=port):
        spec.build()
