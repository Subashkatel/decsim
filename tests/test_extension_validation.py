import pytest
import numpy as np

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
    def begin_operation(self, operation):
        return None

    def round_payloads(self, operation, round_index):
        return TimingOnlyDevice().round_payloads(operation, round_index)

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


class EngineBoundFactory:
    def __init__(self, engine):
        self.engine = engine

    def request(self, operation_id, callback):
        callback()

    def shutdown(self):
        return None


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

    with pytest.raises(RuntimeError, match="already ready"):
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
    world = RunSpec(
        ops=[Operation(0, "memory", (0,))],
        decoder=StaticDecoder(),
    ).build()

    assert world.planning.code.distance == 3
    assert world.window_manager.code is world.planning.code
    assert world.window_manager.layout is world.planning.layout
    assert world.window_manager.scheme is world.planning.scheme
    assert world.window_manager.rounds_policy is world.planning.rounds_policy
    assert world.cluster.planner is world.planning.planner


def test_legacy_boundary_policy_defaults_to_non_speculative_delivery():
    result = simulate(RunSpec(
        ops=[Operation(0, "memory", (0,))],
        rounds_policy=FixedRounds(3),
        decoder=StaticDecoder(),
        boundary_policy=LegacyBoundaryPolicy(),
    ))

    assert result["chip_done"] is not None


def test_static_device_needs_only_static_run_capabilities():
    result = simulate(RunSpec(
        ops=[Operation(0, "memory", (0,))],
        rounds_policy=FixedRounds(3),
        decoder=StaticDecoder(),
        device=StaticOnlyDevice(),
    ))

    assert result["chip_done"] is not None


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
