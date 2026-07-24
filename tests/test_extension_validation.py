import pytest

from decsim.decoders import PerRoundDecoder
from decsim.devices import TimingOnlyDevice
from decsim.message import DecodeResult, Operation
from decsim.planner import FixedRounds
from decsim.run_spec import RunSpec, simulate
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
        return DecodeResult(job.op_id, job.window_id, logical_value=0)


def test_run_spec_rejects_an_incompatible_boundary_policy_before_build():
    spec = RunSpec(ops=[], boundary_policy=WrongBoundarySignature())

    with pytest.raises(
        TypeError,
        match=r"boundary_policy.*BoundaryPolicy.*on_commit.*signature",
    ):
        spec.validate()


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
        ("factory", "MagicStateFactory"),
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
