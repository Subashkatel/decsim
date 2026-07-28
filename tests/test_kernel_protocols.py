"""Port protocols are runtime-checkable and the seam types carry the contract."""
import pytest

from decsim.message import DecodeJob, DecodeOutcome, DecodeResult, Window
from decsim.protocols import (BoundaryPolicy, DecodingStrategy, Directive,
                          OutcomeDirective, RoundsPolicy, StrategyServices,
                          Submission)


class _FakeServices:
    now = 0
    def make_strong_job(self, weak_job, n_rounds, label): return weak_job
    def defer_strong_escalation(self, weak_job, n_rounds, label): pass
    def check_strong_route(self, weak_job, strong_job): pass
    def cancel_strong(self, key): pass
    def ws_delay(self): return 500_000


class _FakeStrategy:
    def validate_declared_run(self, **kwargs): pass
    def validate_operations(self, operations): pass
    def validate_code_geometry(self, geometry): pass
    def on_window_ready(self, window, weak_job, services):
        return [Submission(weak_job)]
    def on_decode_outcome(self, outcome, services):
        return OutcomeDirective(Directive.FINALIZE)
    def metrics(self): return {}


class _FakeBoundary:
    speculative = False
    def on_commit(self, window, final): return True


class _FakeRounds:
    def rounds_for(self, op, code): return 11


def test_protocols_are_structural():
    assert isinstance(_FakeStrategy(), DecodingStrategy)
    assert isinstance(_FakeServices(), StrategyServices)
    assert isinstance(_FakeBoundary(), BoundaryPolicy)
    assert isinstance(_FakeRounds(), RoundsPolicy)


def test_directive_has_exactly_three_members():
    assert {d.name for d in Directive} == {"FINALIZE", "AWAIT_STRONG",
                                           "FINALIZE_STRONG"}


def test_seam_types_flow():
    job = DecodeJob(1, 0, 11)
    w = Window(op_id=1, k=0, commit_lo=1, commit_hi=3, buffer_hi=6, n_rounds=6)
    subs = _FakeStrategy().on_window_ready(w, job, _FakeServices())
    assert subs[0].job is job and subs[0].delay_ticks == 0
    directive = _FakeStrategy().on_decode_outcome(
        DecodeOutcome(job, DecodeResult(1, 0)), _FakeServices())
    assert directive.directive is Directive.FINALIZE and directive.extra is None


def test_run_seed_capabilities_are_structural_and_children_are_typed():
    from decsim.message import RunSeedChild, RunSeedPathSegment
    from decsim.protocols import RunSeedComposite, RunSeedConsumer

    class Consumer:
        def reserve_run_seed(self, seed): return object()
        def commit_run_seed(self, reservation): return None
        def cancel_run_seed(self, reservation): return None

    class Composite:
        def run_seed_children(self):
            return (
                RunSeedChild(
                    relative_path=(
                        RunSeedPathSegment("field", "inner"),
                    ),
                    child=Consumer(),
                ),
            )

    assert isinstance(Consumer(), RunSeedConsumer)
    assert isinstance(Composite(), RunSeedComposite)
    child = tuple(Composite().run_seed_children())[0]
    assert child.relative_path[0].canonical_bytes() == b"F\x00\x00\x00\x05inner"


def test_run_seed_consumer_documents_the_failure_free_commit_phase():
    import inspect

    from decsim.protocols import RunSeedConsumer

    reserve_contract = " ".join(
        (inspect.getdoc(RunSeedConsumer.reserve_run_seed) or "").split()
    )
    commit_contract = " ".join(
        (inspect.getdoc(RunSeedConsumer.commit_run_seed) or "").split()
    )
    cancel_contract = " ".join(
        (inspect.getdoc(RunSeedConsumer.cancel_run_seed) or "").split()
    )

    assert "must not change the active random state" in reserve_contract
    assert "all potentially failing preparation" in reserve_contract
    assert "must be total and must not fail" in commit_contract
    assert "allocate" in commit_contract
    assert "draw" in commit_contract
    assert "callback" in commit_contract
    assert "must be total" in cancel_contract
    assert "exact pending reservation" in cancel_contract


def test_readme_planning_surface_matches_the_public_run_spec():
    from dataclasses import fields
    from pathlib import Path

    from decsim.run_spec import ResolvedPlanningParts, RunSpec

    readme = (Path(__file__).parents[1] / "README.md").read_text()
    run_spec_fields = {item.name for item in fields(RunSpec)}
    planning_fields = {
        item.name for item in fields(ResolvedPlanningParts)
    }

    assert "planner" not in run_spec_fields
    assert planning_fields == {"code", "layout", "scheme", "rounds_policy"}
    assert "`planner=`" not in readme
    for field_name in ("layout", "scheme", "rounds_policy"):
        assert f"`{field_name}=`" in readme
    for field_name in planning_fields:
        assert f"`completed_run.planning.{field_name}`" in readme


def _seed_child_paths(component):
    return {
        tuple((segment.kind, segment.value) for segment in child.relative_path):
        child.child
        for child in component.run_seed_children()
    }


def test_decoder_routers_expose_every_semantic_child():
    from decsim.decoders import CodeRouter, SwitchingRouter

    default = object()
    by_code = object()
    by_none = object()
    code_router = CodeRouter(
        default,
        {"surface": by_code, None: by_none},
    )
    assert _seed_child_paths(code_router) == {
        (("field", "default"),): default,
        (("field", "by_code"), ("string_key", "surface")): by_code,
        (("field", "by_code"), ("none_key", None)): by_none,
    }

    weak = object()
    strong = object()
    assert _seed_child_paths(SwitchingRouter(weak, strong)) == {
        (("field", "weak"),): weak,
        (("field", "strong"),): strong,
    }


def test_code_router_rejects_keys_outside_the_typed_seed_path_domain():
    import pytest

    from decsim.decoders import CodeRouter

    for key in (1, True, 1.0, ("surface",)):
        with pytest.raises(TypeError, match="exact built-in str or None"):
            CodeRouter(object(), {key: object()})


def test_decoder_wrappers_expose_children_and_behavior_callbacks():
    from decsim.decoders import (
        FunctionLatencyDecoder,
        SampledConfidenceDecoder,
        SwitchingDecoder,
    )
    from decsim.soft_output import SoftOutputDecoder

    weak = object()
    strong = object()
    switching = SwitchingDecoder(weak, strong, gamma_switch=0.5)
    assert _seed_child_paths(switching) == {
        (("field", "weak"),): weak,
        (("field", "strong"),): strong,
    }

    inner = object()
    probability_for = lambda job: 0.5
    sampled = SampledConfidenceDecoder(
        inner,
        escalation_probability=0.5,
        probability_for=probability_for,
    )
    assert _seed_child_paths(sampled) == {
        (("field", "inner"),): inner,
        (("field", "probability_for"),): probability_for,
    }

    latency_for = lambda job: 1.0
    assert _seed_child_paths(FunctionLatencyDecoder(latency_for)) == {
        (("field", "latency_us_for"),): latency_for,
    }

    metric_cls = type("Metric", (), {})
    soft = SoftOutputDecoder(inner, metric_cls)
    assert _seed_child_paths(soft) == {
        (("field", "base"),): inner,
        (("field", "metric_cls"),): metric_cls,
    }


def test_devices_and_switching_expose_circuit_scope_and_behavior_children():
    from decsim.adapters.stim_device import StimDevice
    from decsim.codes import SurfaceCodeModel
    from decsim.decoders import SAMPLED_CONFIDENCE_SOURCE
    from decsim.devices import SyndromeBitDevice, TimingOnlyDevice
    from decsim.switching import Switching, ThresholdRegister

    stim_device = StimDevice()
    assert stim_device.operation_circuit_scope == "per_operation"
    assert not hasattr(stim_device, "run_seed_children")
    with pytest.raises(TypeError, match="rounds_for"):
        StimDevice(rounds_for=lambda operation: 3)

    code = SurfaceCodeModel(d=3)
    bit_device = SyndromeBitDevice(code)
    assert bit_device.operation_circuit_scope == "none"
    assert _seed_child_paths(bit_device) == {
        (("field", "code"),): code,
    }
    assert TimingOnlyDevice.operation_circuit_scope == "none"

    register = ThresholdRegister(
        default=0.5,
        expected_source=SAMPLED_CONFIDENCE_SOURCE,
    )
    switching = Switching(
        0.5,
        SAMPLED_CONFIDENCE_SOURCE,
        threshold_register=register,
    )
    assert _seed_child_paths(switching) == {
        (("field", "threshold_register"),): register,
    }


def test_real_decoder_adapters_expose_their_latency_models():
    from decsim.belief_matching_decoder import BeliefMatchingDecoder
    from decsim.bposd_decoder import BPOSDDecoder
    from decsim.mwpm_decoder import PyMatchingDecoder
    from decsim.soft_output import UnionFindDecoder

    latency_model = object()
    expected = {(("field", "latency_model"),): latency_model}
    assert _seed_child_paths(PyMatchingDecoder(latency_model)) == expected
    assert _seed_child_paths(BPOSDDecoder(latency_model)) == expected
    assert _seed_child_paths(BeliefMatchingDecoder(latency_model)) == expected
    assert _seed_child_paths(UnionFindDecoder(latency_model)) == expected
