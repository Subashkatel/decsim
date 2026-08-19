"""One resolver from a RunSpec to every choice a run needs.

resolve_run_configuration applies the defaults, copies and validates the
workload, plans the windows, and rejects incompatible user configuration in
one place. The result is a frozen ResolvedRunConfiguration; RunSpec._build_once
only wires runtime objects from it. Nothing here touches the engine.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .message import OperationPlanningView


@dataclass(frozen=True)
class ResolvedRunConfiguration:
    """Every resolved choice of one run, in the vocabulary the wiring uses."""

    root_seed: Optional[int]
    escalation_policy: Any
    ops: tuple
    decode_ops: tuple
    dynamic_streams: tuple
    protected_regions: tuple
    all_operations: tuple
    planned_operations: tuple
    view_by_id: dict
    code: Any
    layout: Any
    scheme: Any
    rounds_policy: Any
    boundary_policy: Any
    window_interaction: Any
    idle_policy: Any
    plan: Any
    resource_claims: dict
    device: Any
    error_model_provider: Any
    router: Any
    scheduler: Any
    link_config: Any
    buffering: Any
    timing: Any
    feedback_boundary_mode: str
    capture_switching_windows: bool
    unit_pools: Optional[dict]
    num_units: int
    lane_policy: Any
    memory_model: Any
    decoder_memory: Any
    pauli_frame: Any
    syndrome_ingress_policy: Any
    make_syndrome_ingress: Optional[Callable]
    make_decoder_memory_transfer: Optional[Callable]
    make_factory: Optional[Callable]
    make_metrics: Optional[Callable]
    make_conditional_release: Optional[Callable]


def resolve_run_configuration(spec, root_seed) -> ResolvedRunConfiguration:
    from .decoders.decoders import CodeRouter
    from .qpu.syndrome_devices import SyndromeBitDevice, TimingOnlyDevice
    from .links.link_profiles import logical_reference_profile
    from .frontends.planner import (_plan_execution, _validate_operation_graph,
                          _validate_workload_identity)
    from .controller.policies import Eager, Ignore
    from .qpu.round_policies import GateRounds
    from .decoders.schedulers import FifoScheduler
    from .windows.windowing_schemes import SlidingWindowScheme
    from .decoders.weak_strong_switching import Baseline
    from .syndrome_buffer.syndrome_buffer import SyndromeBufferingConfig
    from .controller.syndrome_ingress import SyndromeIngressPolicy
    from .windows.window_interactions import DefaultWindowInteraction

    escalation_policy = spec.escalation_policy if spec.escalation_policy is not None else Baseline()

    if (spec.ops is None) == (spec.frontend is None):
        raise ValueError("provide exactly one of ops= or frontend=")
    if spec.feedback_boundary_mode not in ("trailing_buffer", "measurement_closed"):
        raise ValueError("invalid feedback_boundary_mode")
    source_ops = spec.frontend.build() if spec.frontend is not None else spec.ops
    ops, decode_ops, dynamic_streams = _copy_workload(
        source_ops, spec.decode_ops or (), spec.dynamic_streams or (),
        spec.feedback_boundary_mode)
    _validate_workload_identity(ops, decode_ops, dynamic_streams)
    all_operations = _unique_operations(ops + decode_ops + dynamic_streams)
    views = tuple(OperationPlanningView.from_operation(op) for op in all_operations)
    view_by_id = {view.id: view for view in views}
    _validate_operation_graph(
        list(ops), validate_blockers=True,
        external_blocker_ids=(op.id for op in decode_ops + dynamic_streams))

    code, layout = _select_code(spec.d, spec.code, spec.layout)
    scheme = spec.scheme or SlidingWindowScheme()
    rounds_policy = spec.rounds_policy or GateRounds()
    boundary_policy = spec.boundary_policy or Eager()
    window_interaction = spec.window_interaction or DefaultWindowInteraction()
    if dynamic_streams and type(scheme) is not SlidingWindowScheme:
        raise ValueError("dynamic streams require SlidingWindowScheme")
    escalation_policy.validate_declared_run(
        scheme=scheme, boundary_policy=boundary_policy,
        has_dynamic_streams=bool(dynamic_streams),
        static_decode_plan_selected=spec.decode_ops is not None,
        has_frontend=spec.frontend is not None)
    escalation_policy.validate_operations(views)

    planned_operations = _decode_plan_operations(
        ops, decode_ops, dynamic_streams,
        static_decode_selected=spec.decode_ops is not None)
    plan = _plan_execution(
        operations=views,
        planned_operation_ids=tuple(op.id for op in planned_operations),
        code=code, layout=layout, scheme=scheme, rounds_policy=rounds_policy,
        fallback_round_us=(spec.round_us if spec.round_us is not None
                           else spec.timing.round_us),
        retain_strong_context=escalation_policy.requires_strong_context,
        double_window=escalation_policy.double_window,
        has_open_ended_dynamic_streams=bool(dynamic_streams))
    escalation_policy.validate_code_geometry(plan.code_geometry)
    resource_claims = {op.id: tuple(layout.resources_for(view_by_id[op.id]))
                       for op in ops}

    device = spec.device or TimingOnlyDevice()
    _install_device_circuits(device, all_operations)
    error_model_provider = (device if spec.error_model_provider is None
                            else spec.error_model_provider)
    if (error_model_provider is not device
            and hasattr(error_model_provider, "operation_circuit_scope")):
        _install_device_circuits(error_model_provider, all_operations)
    if type(device) is SyndromeBitDevice and device.code is not code:
        raise ValueError("SyndromeBitDevice.code must be the exact resolved run code")

    if spec.router is not None and (spec.decoder is not None or spec.decoders):
        raise ValueError("router is exclusive with decoder and decoders")
    if spec.router is None and spec.decoder is None and planned_operations:
        raise ValueError("decoder is required when router is omitted")
    router = spec.router or CodeRouter(default=spec.decoder, by_code=dict(spec.decoders))

    link_config = spec.links if spec.links is not None else logical_reference_profile()
    if (spec.timing.ticks("t_binary_availability") > 0
            and not link_config.qc_excludes_controller_processing):
        raise ValueError("a separate controller readout cost requires a link "
                         "profile whose QC latency excludes that cost")
    if spec.make_syndrome_ingress is not None and spec.syndrome_ingress_policy is not None:
        raise ValueError("syndrome_ingress_policy cannot be combined with make_syndrome_ingress")

    return ResolvedRunConfiguration(
        root_seed=root_seed, escalation_policy=escalation_policy,
        ops=tuple(ops), decode_ops=tuple(decode_ops),
        dynamic_streams=tuple(dynamic_streams),
        protected_regions=tuple(spec.protected_regions),
        all_operations=all_operations, planned_operations=tuple(planned_operations),
        view_by_id=view_by_id, code=code, layout=layout, scheme=scheme,
        rounds_policy=rounds_policy, boundary_policy=boundary_policy,
        window_interaction=window_interaction,
        idle_policy=spec.idle_policy or Ignore(),
        plan=plan, resource_claims=resource_claims,
        device=device, error_model_provider=error_model_provider,
        router=router,
        scheduler=FifoScheduler() if spec.scheduler is None else spec.scheduler,
        link_config=link_config,
        buffering=spec.syndrome_buffering or SyndromeBufferingConfig(),
        timing=spec.timing, feedback_boundary_mode=spec.feedback_boundary_mode,
        capture_switching_windows=spec.record_switching_windows,
        unit_pools=spec.unit_pools,
        num_units=spec.num_units if spec.num_units is not None else 1,
        lane_policy=spec.lane_policy, memory_model=spec.memory_model,
        decoder_memory=spec.decoder_memory, pauli_frame=spec.pauli_frame,
        syndrome_ingress_policy=(spec.syndrome_ingress_policy
                                 if spec.syndrome_ingress_policy is not None
                                 else SyndromeIngressPolicy()),
        make_syndrome_ingress=spec.make_syndrome_ingress,
        make_decoder_memory_transfer=spec.make_decoder_memory_transfer,
        make_factory=spec.make_factory, make_metrics=spec.make_metrics,
        make_conditional_release=spec.make_conditional_release)


def _select_code(distance, code, layout):
    from .qpu.code_geometry import SurfaceCodeModel
    from .qpu.layouts import UniformLayout
    if sum(value is not None for value in (distance, code, layout)) > 1:
        supplied = [name for name, value in (
            ("d", distance), ("code", code), ("layout", layout))
            if value is not None]
        raise ValueError(f"multiple code sources supplied: {', '.join(supplied)}")
    if layout is not None:
        codes = list(layout.codes())
        if len(codes) != 1:
            raise ValueError(
                f"layout must declare exactly one code (got {len(codes)})")
        return codes[0], layout
    selected = code if code is not None else SurfaceCodeModel(
        d=3 if distance is None else distance)
    return selected, UniformLayout(selected)


def _copy_workload(source_ops, decode_ops, dynamic_streams, feedback_mode):
    """Private copies of the user's operations, so a run never mutates them;
    an operation without its own feedback boundary mode takes the run's."""
    copies = {}

    def clone(operation):
        if id(operation) not in copies:
            private = copy.copy(operation)
            if private.feedback_boundary_mode is None:
                private.feedback_boundary_mode = feedback_mode
            copies[id(operation)] = private
        return copies[id(operation)]
    return tuple(tuple(clone(op) for op in group)
                 for group in (source_ops, decode_ops, dynamic_streams))


def _install_device_circuits(device, operations):
    scope = getattr(device, "operation_circuit_scope", None)
    if scope == "none":
        for operation in operations:
            operation.circuit = None
        return
    if scope != "per_operation":
        raise ValueError("device operation_circuit_scope must be none or per_operation")
    import stim
    for operation in operations:
        if operation.circuit is None:
            continue
        operation.circuit = stim.Circuit(str(operation.circuit))


def _unique_operations(operations):
    unique = {}
    for operation in operations:
        unique.setdefault(operation.id, operation)
    return tuple(unique.values())


def _decode_plan_operations(ops, decode_ops, dynamic_streams, *,
                            static_decode_selected):
    if static_decode_selected:
        return decode_ops
    if not dynamic_streams:
        return tuple(op for op in ops if op.emits_detector_data)
    dynamic_ids = {stream.id for stream in dynamic_streams}
    return tuple(
        op for op in ops
        if op.emits_detector_data and op.stream_id not in dynamic_ids
    )


def check_factory_decode_service(factory, decoder_manager):
    """A distillation factory that decodes corrections uses the run's decoder manager."""
    from .qpu.magic_state_factories import DistillationFactory, MultiLevelDistillationFactory
    if type(factory) not in (DistillationFactory, MultiLevelDistillationFactory):
        return
    expected = decoder_manager if factory.n_corr > 0 else None
    if factory.decode_service is not expected:
        raise ValueError(
            f"{type(factory).__name__} decode_service must be run-owned")
