from dataclasses import fields

import numpy as np
import pytest

from decsim.decoders import PerRoundDecoder, SAMPLED_CONFIDENCE_SOURCE
from decsim.detector_error_model import NO_FAULT_MODEL_REQUIRED
from decsim.devices import TimingOnlyDevice
from decsim.codes import SurfaceCodeModel
from decsim.engine import Engine
from decsim.frontends.circuit import CircuitFrontend, SurgeryIRFrontend
from decsim.layouts import UniformLayout
from decsim.message import DecodeResult, Operation, RunSeedReservation
from decsim.metrics import (
    BacklogEarlyWarning,
    BacklogTrajectory,
    ConditionalReactionTime,
    DecodeBacklog,
    DecoderUtilization,
    MagicStateLatency,
    ReadyQueueStats,
    StrongDecoderBacklog,
    WindowLatencyBreakdown,
)
from decsim.planner import FixedRounds
from decsim.policies import Eager, Held, ExtendStream, SeparateDecodeJobs
from decsim.run_spec import RunSpec, simulate
from decsim.schedulers import (
    EarliestDeadlineScheduler,
    BufferExpiryDeadline,
    ReactionPathDeadline,
    WeightedUrgencyCostScheduler,
)
from decsim.schemes import (
    NaiveOnlineScheme,
    ParallelWindowScheme,
    SlidingWindowScheme,
)
from decsim.switching import Baseline, Switching, ThresholdRegister


class WrongBoundarySignature:
    speculative = False

    def on_commit(self):
        return True


class LegacyBoundaryPolicy:
    def on_commit(self, window, final):
        return True


class StaticOnlyDevice:
    operation_circuit_scope = "none"

    def begin_operation(
        self, operation, segment_round_count, source_round_count
    ):
        return None

    def round_payloads(self, operation, round_index):
        return TimingOnlyDevice().round_payloads(operation, round_index)

    def window_models_for_operation(
        self, operation, windows, round_count, *, fault_model_requirement,
        fault_exclusion_ranges,
    ):
        return []


class MissingCircuitScopeDevice:
    def begin_operation(
        self, operation, segment_round_count, source_round_count
    ):
        return None

    def round_payloads(self, operation, round_index):
        return []

    def window_models_for_operation(
        self, operation, windows, round_count, *, fault_model_requirement,
        fault_exclusion_ranges,
    ):
        return []


class StaticDecoder:
    fault_model_requirement = NO_FAULT_MODEL_REQUIRED

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


class CircuitRecordingTimingDevice(TimingOnlyDevice):
    operation_circuit_scope = "none"

    def __init__(self):
        self.circuits_seen = []

    def begin_operation(
        self, operation, segment_round_count, source_round_count
    ):
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

    def spatial_nodes_for(self, operation, *, base_spatial_node_count):
        self._require_planning_view(operation)
        self.spatial_calls += 1
        return super().spatial_nodes_for(
            operation,
            base_spatial_node_count=base_spatial_node_count,
        )


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
        self.plan_calls = []

    def plan_operation(
        self,
        operation_id,
        round_count,
        *,
        commit_round_count,
        buffer_round_count,
    ):
        self.plan_calls.append((
            operation_id,
            round_count,
            commit_round_count,
            buffer_round_count,
        ))
        return super().plan_operation(
            operation_id,
            round_count,
            commit_round_count=commit_round_count,
            buffer_round_count=buffer_round_count,
        )

    def data_complete(self, window, *, operation, **kwargs):
        assert not hasattr(operation, "circuit")
        self.data_complete_calls += 1
        return super().data_complete(
            window,
            operation=operation,
            **kwargs,
        )


class UnenteredCustomScheme:
    def __init__(self):
        self.calls = []

    def plan_operation(
        self,
        operation_id,
        round_count,
        *,
        commit_round_count,
        buffer_round_count,
    ):
        self.calls.append("plan_operation")
        raise AssertionError("custom scheme behavior was entered")

    def data_complete(
        self,
        window,
        *,
        rounds_arrived,
        successor_rounds,
        memory_rounds,
        round_count,
        has_successor,
        operation,
    ):
        self.calls.append("data_complete")
        raise AssertionError("custom scheme behavior was entered")

    def validate_buffer(self, geometry):
        self.calls.append("validate_buffer")
        raise AssertionError("custom scheme behavior was entered")


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

    def make_controller(self, engine, links, buffering, window_manager):
        self.entries += 1
        raise AssertionError("pre-binding provider must not be called")


class PlainControllerProvider:
    def __init__(self, engine, links, buffering, window_manager):
        self.engine = engine
        self.links = links
        self.buffering = buffering
        self.window_manager = window_manager


@pytest.mark.parametrize("seed", [True, 1.0, "1", object()])
def test_run_seed_rejects_non_integral_values_before_build(seed):
    spec = RunSpec(ops=[], seed=seed)

    with pytest.raises(
        TypeError,
        match=r"seed.*64-bit unsigned integer or None",
    ):
        spec.build()


@pytest.mark.parametrize("seed", [-1, 1 << 64])
def test_run_seed_rejects_values_outside_unsigned_64_bit_domain(seed):
    spec = RunSpec(ops=[], seed=seed)

    with pytest.raises(
        ValueError,
        match=r"seed.*0.*2\*\*64",
    ):
        spec.build()


@pytest.mark.parametrize(
    "seed",
    [None, 0, (1 << 64) - 1, np.int64(7), np.uint64(7)],
)
def test_run_seed_accepts_none_and_unsigned_integral_values(seed):
    RunSpec(ops=[], seed=seed).build()


def test_plain_class_controller_provider_is_accepted_without_instantiation():
    spec = RunSpec(
        ops=[],
        make_controller=PlainControllerProvider,
    )

    spec.build()


def test_router_resolves_only_the_selected_codes_fault_model_requirement():
    from decsim.decoders import CodeRouter
    from decsim.detector_error_model import (
        DecoderFaultModelRequirement,
        FaultRepresentation,
    )

    graphlike = DecoderFaultModelRequirement(
        frozenset({FaultRepresentation.GRAPHLIKE}),
    )
    physical = DecoderFaultModelRequirement(
        frozenset({FaultRepresentation.PHYSICAL}),
    )
    no_faults = DecoderFaultModelRequirement()

    router = CodeRouter(
        default=type(
            "TimingDecoder",
            (StaticDecoder,),
            {"fault_model_requirement": no_faults},
        )(),
        by_code={
            "surface": type(
                "GraphlikeDecoder",
                (StaticDecoder,),
                {"fault_model_requirement": graphlike},
            )(),
            "bb": type(
                "PhysicalDecoder",
                (StaticDecoder,),
                {"fault_model_requirement": physical},
            )(),
        },
    )

    assert router.fault_model_requirement_for("surface") == graphlike
    assert router.fault_model_requirement_for("bb") == physical
    assert router.fault_model_requirement_for("unmapped").representations == frozenset()


def test_device_receives_the_exact_frozen_operation_round_count():
    class RoundRecordingDevice(TimingOnlyDevice):
        def __init__(self):
            self.received = []

        def begin_operation(
            self, operation, segment_round_count, source_round_count
        ):
            self.received.append(
                (operation.id, segment_round_count, source_round_count)
            )

    device = RoundRecordingDevice()
    RunSpec(
        ops=[Operation(9, "memory", (0,))],
        rounds_policy=FixedRounds(7),
        decoder=StaticDecoder(),
        device=device,
    ).build()

    assert device.received == [(9, 7, 7)]


def test_syndrome_bit_device_must_share_the_exact_resolved_code():
    from decsim.devices import SyndromeBitDevice

    canonical_code = SurfaceCodeModel(d=3)
    equal_but_distinct_code = SurfaceCodeModel(d=3)

    with pytest.raises(
        ValueError,
        match="SyndromeBitDevice.code must be the exact resolved run code",
    ):
        RunSpec(
            ops=[],
            code=canonical_code,
            decoder=StaticDecoder(),
            device=SyndromeBitDevice(equal_but_distinct_code),
        ).build()

    completed = RunSpec(
        ops=[],
        code=canonical_code,
        decoder=StaticDecoder(),
        device=SyndromeBitDevice(canonical_code),
    ).build()
    assert completed.window_manager._code_geometry.distance == canonical_code.distance


@pytest.mark.parametrize("companion", ["decoder", "decoders"])
def test_supplied_router_is_exclusive_with_decoder_topology(companion):
    from decsim.decoders import CodeRouter

    decoder = StaticDecoder()
    kwargs = {companion: decoder if companion == "decoder" else {"x": decoder}}

    with pytest.raises(
        ValueError,
        match="router is exclusive with decoder and decoders",
    ):
        RunSpec(
            ops=[],
            router=CodeRouter(decoder),
            **kwargs,
        ).build()


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

    assert layout.operation_calls == 1
    assert layout.resource_calls == 1
    assert layout.spatial_calls == 1
    assert rounds.calls == 1
    assert scheme.plan_calls == [(0, 3, 3, 3)]
    assert scheme.data_complete_calls > 0


@pytest.mark.parametrize(
    "scheme",
    [
        NaiveOnlineScheme(),
        ParallelWindowScheme(),
        type("SlidingSubclass", (SlidingWindowScheme,), {})(),
        UnenteredCustomScheme(),
    ],
)
def test_none_weak_keepup_ratio_does_not_restrict_scheme_shape(scheme):
    Switching(
        expected_source=SAMPLED_CONFIDENCE_SOURCE,
        confidence_threshold=0.5,
        weak_keepup_ratio=None,
    ).validate_declared_run(
        scheme=scheme,
        boundary_policy=Eager(),
        has_dynamic_streams=False,
        static_decode_plan_selected=False,
        has_frontend=False,
    )


@pytest.mark.parametrize(
    ("decode_operations", "expected_selected"),
    [
        (None, False),
        ([], True),
        ([Operation(7, "decode", (0,))], True),
    ],
)
def test_static_decode_plan_selection_distinguishes_none_from_empty(
    decode_operations,
    expected_selected,
):
    class RecordingBaseline(Baseline):
        def __init__(self):
            self.selected_values = []

        def validate_declared_run(self, **arguments):
            self.selected_values.append(
                arguments["static_decode_plan_selected"]
            )
            return super().validate_declared_run(**arguments)

    strategy = RecordingBaseline()
    RunSpec(
        ops=[],
        decode_ops=decode_operations,
        strategy=strategy,
        decoder=StaticDecoder(),
    ).build()

    assert strategy.selected_values == [expected_selected]


def test_strategy_selection_and_capabilities_are_resolved_once():
    class OnceStrategy(Baseline):
        def __init__(self):
            self.reads = dict.fromkeys((
                "requires_strong_context", "bulk_strong", "double_window",
            ), 0)

        def __bool__(self):
            raise AssertionError("strategy selection invoked truthiness")

        def _read(self, name):
            self.reads[name] += 1
            if self.reads[name] > 1:
                raise AssertionError(f"{name} was read twice")
            return False

        requires_strong_context = property(lambda self: self._read(
            "requires_strong_context"))
        bulk_strong = property(lambda self: self._read("bulk_strong"))
        double_window = property(lambda self: self._read("double_window"))

    strategy = OnceStrategy()
    completed = RunSpec(ops=[], strategy=strategy).build()

    assert completed.window_manager.strategy is strategy
    assert tuple(strategy.reads.values()) == (1, 1, 1)


def test_strategy_hooks_cannot_change_resolved_owner_modes():
    class MutatingStrategy(Baseline):
        requires_strong_context = bulk_strong = double_window = True

        def validate_declared_run(self, **arguments):
            self.requires_strong_context = False
            self.bulk_strong = False
            self.double_window = False

    strategy = MutatingStrategy()
    completed = RunSpec(ops=[], strategy=strategy).build()

    assert (strategy.requires_strong_context, strategy.bulk_strong,
            strategy.double_window) == (False, False, False)
    owner_modes = (completed.window_manager.retain_strong_context,
                   completed.decoder_manager.bulk_strong,
                   completed.window_manager.speculative_recovery._double_window)
    assert owner_modes == (True, True, True)


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
        ValueError,
        match=r"device operation_circuit_scope.*none.*per_operation",
    ):
        spec.build()


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
    from decsim.seeding import derive_component_seed

    default = SeedRecordingDecoder()
    surface = SeedRecordingDecoder()
    RunSpec(
        ops=[],
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
        derive_component_seed(7, default_path),
    ]
    assert [item.proposed_seed for item in surface.committed] == [
        derive_component_seed(7, surface_path),
    ]


def test_shared_seed_consumer_is_bound_once_by_its_first_semantic_path():
    from decsim.decoders import CodeRouter
    from decsim.message import RunSeedPathSegment
    from decsim.seeding import derive_component_seed

    shared = SeedRecordingDecoder()
    RunSpec(
        ops=[],
        router=CodeRouter(shared, {"surface": shared}),
        seed=7,
    ).build()

    assert len(shared.reserved) == 1
    assert shared.committed == shared.reserved
    assert shared.committed[0].proposed_seed == derive_component_seed(7, (
        RunSeedPathSegment("field", "decoder_router"),
        RunSeedPathSegment("field", "by_code"),
        RunSeedPathSegment("string_key", "surface"),
    ))


def test_surrogate_code_units_are_not_stable_identities():
    from decsim.message import is_stable_identity

    assert not is_stable_identity("\ud800")
    assert not is_stable_identity("\udfff")
    assert not is_stable_identity(("gross", ("\ud800", 5)))
    assert is_stable_identity("\\ud800")
    assert is_stable_identity("\N{GRINNING FACE}")


class _EqualMetricName(str):
    """Wrong identity type that compares equal to an exact metric name."""


@pytest.mark.parametrize(
    "field,mutated_value",
    [
        ("name", _EqualMetricName("equal-type-drift")),
        ("result_schema_version", True),
        ("result_schema_version", 1.0),
    ],
)
def test_metric_equal_but_wrong_identity_type_is_rejected(
    field,
    mutated_value,
):
    class DriftingMetric:
        name = "equal-type-drift"
        result_schema_version = 1

        def observe(self, engine):
            setattr(self, field, mutated_value)

        def result(self):
            return {}

    with pytest.raises(RuntimeError, match="identity changed|changed.*identity"):
        RunSpec(
            ops=[],
            decoder=StaticDecoder(),
            make_metrics=lambda *_args: [DriftingMetric()],
        ).build()


def test_code_geometry_and_spatial_profile_are_resolved_once():
    class SingleReadCode:
        def __init__(self):
            self.calls = {}

        def _read(self, key, value):
            self.calls[key] = self.calls.get(key, 0) + 1
            if self.calls[key] > 1:
                raise AssertionError(f"code input {key!r} was re-read")
            return value

        @property
        def name(self):
            return self._read("name", "single-read-code")

        @property
        def distance(self):
            return self._read("distance", 3)

        def rounds_per_logical_cycle(self):
            return self._read("logical_cycle", 3)

        def round_period_us(self):
            return self._read("round_period_us", None)

        def commit_rounds(self):
            return self._read("commit_rounds", 3)

        def buffer_rounds(self):
            return self._read("buffer_rounds", 3)

        def buffering_floor(self):
            return self._read("buffering_floor", (3, 3))

        def buffer_floor_override_active(self):
            return self._read("buffer_floor_override_active", False)

        def spatial_nodes(self, patch_count):
            return self._read(("spatial_nodes", patch_count), 9 * patch_count)

        def syndrome_bits_per_round(self, patch_count):
            return self._read(
                ("syndrome_bits_per_round", patch_count),
                8 * patch_count,
            )

    code = SingleReadCode()
    RunSpec(
        ops=[Operation(7, "memory", (0,), patches=(0, 1))],
        layout=UniformLayout(code),
        rounds_policy=FixedRounds(5),
        decoder=StaticDecoder(),
    ).build()

    assert code.calls == {
        "round_period_us": 1,
        "name": 1,
        "distance": 1,
        "commit_rounds": 1,
        "buffer_rounds": 1,
        "buffering_floor": 1,
        "buffer_floor_override_active": 1,
        ("spatial_nodes", 1): 1,
        ("spatial_nodes", 2): 1,
    }


def test_static_plan_has_one_private_immutable_materialization_path():
    import decsim.planner as planner_module
    import decsim.run_spec as run_spec_module

    assert not hasattr(planner_module, "_plan_static_operations")
    assert not hasattr(planner_module, "_resolve_code_geometry")
    assert not hasattr(run_spec_module, "_freeze_execution_plan")


def test_runtime_owners_retain_frozen_resolved_lookup_maps():
    completed = RunSpec(
        ops=[Operation(7, "memory", ("patch-a",))],
        rounds_policy=FixedRounds(5),
        decoder=StaticDecoder(),
    ).build()

    frozen_maps = (
        completed.window_manager._resolved_operations,
        completed.window_manager._resolved_patches,
        completed.window_manager._planning_view_by_operation_id,
        completed.chip._resolved_operations,
        completed.chip._resolved_patches,
        completed.chip._resource_claims_by_operation_id,
        completed.chip.source._round_count_by_operation_id,
    )
    for frozen_map in frozen_maps:
        with pytest.raises(TypeError):
            frozen_map["mutation"] = object()


def test_completed_diagnostic_handles_do_not_expose_executable_operations():
    completed = RunSpec(
        ops=[Operation(0, "memory", (0,))],
        rounds_policy=FixedRounds(3),
        decoder=StaticDecoder(),
    ).build()

    assert not hasattr(completed.chip, "ops")
    assert not hasattr(completed.window_manager, "ops")
    assert not hasattr(completed.decoder_manager, "ops")


def test_factory_decode_service_is_the_run_decoder_manager():
    from decsim.factories import DistillationFactory

    completed = RunSpec(
        ops=[],
        decoder=StaticDecoder(),
        make_factory=lambda engine, decoder_manager: DistillationFactory(
            engine,
            num_units=1,
            cycle_ticks=1,
            decode_service=decoder_manager,
            corr_rounds=1,
            n_corr=1,
        ),
    ).build()
    assert completed.factory.decode_service is completed.decoder_manager


def test_factory_rejects_a_foreign_correction_decode_service():
    from decsim.factories import DistillationFactory

    foreign_service = object()
    with pytest.raises(
        ValueError,
        match=r"DistillationFactory.*decode_service.*run-owned",
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


def test_provider_failure_invalidates_the_root_engine_before_any_event_runs():
    from decsim.engine import SimulationFailed

    captured_engines = []
    sentinel_events = []
    originating_failure = RuntimeError("controller construction failed")

    def failing_controller(engine, _links, _buffering, _window_manager):
        captured_engines.append(engine)
        engine.schedule(0, lambda: sentinel_events.append("ran"))
        raise originating_failure

    spec = RunSpec(
        ops=[],
        decoder=StaticDecoder(),
        make_controller=failing_controller,
    )

    with pytest.raises(RuntimeError) as raised:
        spec.build()

    assert raised.value is originating_failure
    assert spec._build_state == "invalid"
    assert len(captured_engines) == 1
    engine = captured_engines[0]
    assert engine._phase == "invalid"
    assert engine._failure_cause is originating_failure
    assert sentinel_events == []
    assert len(engine._event_queue) == 1

    with pytest.raises(SimulationFailed) as run_failure:
        engine.run()
    assert run_failure.value.__cause__ is originating_failure
    with pytest.raises(SimulationFailed) as schedule_failure:
        engine.schedule(0, lambda: sentinel_events.append("late"))
    assert schedule_failure.value.__cause__ is originating_failure
    assert sentinel_events == []


def test_run_seed_reservation_failure_cancels_prior_leaves_without_committing():
    from decsim.decoders import CodeRouter

    acquired = SeedRecordingDecoder()
    failing = SeedRecordingDecoder(fail_reservation=True)
    spec = RunSpec(
        ops=[],
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
        match=r"disagrees with the run seed",
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
        result_schema_version = 1

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
        make_metrics=lambda engine, _wm, _dm, _chip, _factory: [
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
    from decsim.seeding import derive_component_seed

    path = tuple(
        RunSeedPathSegment(kind, value)
        for kind, value in path_parts
    )

    assert derive_component_seed(root_seed, path) == expected


def test_integer_seed_path_rejects_bool_and_non_integer_values():
    from decsim.message import RunSeedPathSegment

    for value in (True, "9", 9.0):
        with pytest.raises(TypeError, match="built-in int"):
            RunSeedPathSegment("integer_key", value)


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
        spec.build()


def test_run_spec_code_selection_uses_explicit_none():
    with pytest.raises(ValueError, match="d"):
        RunSpec(ops=[], d=0).build()

    class FalseySurfaceCode(SurfaceCodeModel):
        def __bool__(self):
            return False

    supplied = FalseySurfaceCode(d=5)
    completed = RunSpec(ops=[], code=supplied).build()
    assert completed.window_manager._code_geometry.distance == 5


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
        RunSpec(ops=[], layout=layout).build()


def test_layout_selection_failure_invalidates_the_root_engine(monkeypatch):
    from decsim.engine import Engine as RuntimeEngine

    declared_code = SurfaceCodeModel(d=3)
    layout = SelectorLayout(
        declared_code,
        operation_code=SurfaceCodeModel(d=5),
    )
    constructed_engines = []

    class CapturingEngine(RuntimeEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            constructed_engines.append(self)

    monkeypatch.setattr("decsim.engine.Engine", CapturingEngine)

    with pytest.raises(ValueError, match=r"layout.*operation 21") as raised:
        RunSpec(
            ops=[Operation(21, "invalid selection", (0,))],
            layout=layout,
            decoder=StaticDecoder(),
        ).build()

    assert len(constructed_engines) == 1
    engine = constructed_engines[0]
    assert engine._phase == "invalid"
    assert engine._failure_cause is raised.value


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
