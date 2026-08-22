"""Run configuration and the simulator's composition root.

RunSpec selects one implementation per seam, wires the runtime graph, executes
it once, and returns the scientific result with useful runtime owners. Derived
geometry/window planning and component seeding have their own focused owners.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from .config import TimingConfig
from .message import ExecutionProgram, RunSeedPathSegment
from .links.link_traffic_report import traffic_json_value
from .seeding import bind_run_seed

if TYPE_CHECKING:
    from .decoders.decoder_memory import DecoderMemoryConfig


@dataclass(frozen=True)
class LogicalOperationResult:
    """One prediction and its optional sampled logical-observable truth.

    ``logical_failure`` is true when any predicted bit differs from truth.
    """

    operation_id: int
    result_status: str
    logical_observables: Optional[tuple[int, ...]]
    stream_offset: Optional[int]
    observable_truth: Optional[tuple[int, ...]] = None
    logical_failure: Optional[bool] = None


@dataclass(frozen=True)
class MetricResultRecord:
    name: str
    value: Any


@dataclass(frozen=True)
class PrimaryRunResult:
    """Scientific outputs from one completed simulation."""

    terminal_status: str
    event_queue_empty: bool
    decode_work_settled: bool
    execution_workload_complete: bool
    execution_done_ticks: int
    fully_done_ticks: int
    operation_results: tuple[LogicalOperationResult, ...]
    link_traffic: dict
    metric_results: tuple[MetricResultRecord, ...]

    def logical_results(self) -> dict[int, tuple[int, ...]]:
        return {row.operation_id: row.logical_observables
                for row in self.operation_results
                if row.result_status == "logical_observables"}

    def stream_offsets(self) -> dict[int, Optional[int]]:
        return {row.operation_id: row.stream_offset
                for row in self.operation_results}

    def metric_values(self) -> dict[str, Any]:
        return {row.name: copy.deepcopy(row.value)
                for row in self.metric_results}


def capture_primary_result(engine, execution_runtime, window_manager, operations,
                           metric_bindings, links, syndrome_source):
    """Project terminal runtime owners into the immutable run result."""
    operation_by_id = {operation.id: operation for operation in operations}
    truth_for = getattr(syndrome_source, "logical_observable_truth", None)
    rows = []
    for operation_id in sorted(operation_by_id):
        logical = window_manager.op_results.get(operation_id)
        if logical is not None:
            bits = tuple(logical)
            status = "logical_observables"
        else:
            bits = None
            status = "no_logical_output"
        actual = None if truth_for is None else truth_for(operation_id)
        if actual is not None:
            actual = tuple(actual)
        failure = None
        if bits is not None and actual is not None:
            if len(bits) != len(actual):
                raise RuntimeError(
                    f"operation {operation_id} predicted {len(bits)} logical "
                    f"observables but the syndrome source sampled {len(actual)}")
            failure = bits != actual
        binding = execution_runtime.controller.stream_binding_for(operation_id)
        stream_offset = (operation_by_id[operation_id].stream_offset
                         if binding is None else binding.stream_offset)
        rows.append(LogicalOperationResult(
            operation_id, status, bits, stream_offset, actual, failure))
    metric_rows = tuple(MetricResultRecord(name, copy.deepcopy(metric.result()))
                        for name, metric in metric_bindings)
    if not engine.idle or not execution_runtime.workload_complete:
        raise RuntimeError("primary run ended before workload completed")
    return PrimaryRunResult(
        "complete", True, True, True, execution_runtime.last_finish_time, engine.now,
        tuple(rows), copy.deepcopy(traffic_json_value(links.snapshot())), metric_rows)


@dataclass(frozen=True)
class CompletedRun:
    """Scientific result plus runtime owners useful to experiments."""

    result: PrimaryRunResult
    engine: Any
    window_manager: Any
    decoder_manager: Any
    execution_runtime: Any
    controller: Any
    qpu: Any
    conditional_release: Any
    factory: Any
    syndrome_buffer: Any
    syndrome_ingress: Any
    pauli_frame: Any = None


@dataclass
class RunSpec:
    """Select, connect, and execute one simulator configuration."""

    ops: Optional[list] = None
    frontend: Optional[Any] = None
    decode_ops: Optional[list] = None
    dynamic_streams: Optional[list] = None
    protected_regions: tuple = ()
    code: Optional[Any] = None
    layout: Optional[Any] = None
    d: Optional[int] = None
    decoder: Optional[Any] = None
    decoders: dict = field(default_factory=dict)
    router: Optional[Any] = None
    escalation_policy: Optional[Any] = None
    scheduler: Optional[Any] = None
    lane_policy: Optional[Any] = None
    unit_pools: Optional[dict] = None
    num_units: Optional[int] = None
    scheme: Optional[Any] = None
    rounds_policy: Optional[Any] = None
    boundary_policy: Optional[Any] = None
    window_interaction: Optional[Any] = None
    idle_policy: Optional[Any] = None
    feedback_boundary_mode: str = "trailing_buffer"
    timing: TimingConfig = field(default_factory=TimingConfig)
    round_us: Optional[float] = None
    links: Optional[Any] = None
    device: Optional[Any] = None
    error_model_provider: Optional[Any] = None
    memory_model: Optional[Any] = None
    syndrome_buffering: Optional[Any] = None
    decoder_memory: Optional["DecoderMemoryConfig"] = None
    pauli_frame: Optional[Any] = None
    syndrome_ingress_policy: Optional[Any] = None
    make_syndrome_ingress: Optional[Callable] = None
    make_decoder_memory_transfer: Optional[Callable] = None
    make_factory: Optional[Callable] = None
    make_metrics: Optional[Callable] = None
    record_switching_windows: bool = False
    make_conditional_release: Optional[Callable] = None
    seed: Optional[int] = 0
    _built: bool = field(default=False, init=False, repr=False)

    def build(self, verbose: bool = False) -> CompletedRun:
        """Wire and run this configuration once; a RunSpec is one run."""
        if self._built:
            raise RuntimeError("RunSpec was already built; make a new one per run")
        self._built = True
        from .engine import Engine
        return self._build_once(Engine(verbose=verbose), _root_seed(self.seed))

    def _build_once(self, engine, root_seed) -> CompletedRun:
        """Wire the run from its resolved configuration, in dependency order,
        run the engine to quiescence, and capture the result."""
        from .controller.controller import Controller
        from .controller.feedback_streams import FeedbackStreams, NoFeedbackStreams
        from .decoders.decoder_manager import DecoderManager
        from .frontends.execution_runtime import ExecutionRuntime
        from .pauli_frame.conditional_release import ConditionalRelease
        from .qpu.cycle_clock import QPUDevice
        from .run_configuration import (check_factory_decode_service,
                                        resolve_run_configuration)
        from .syndrome_buffer.syndrome_buffer import SyndromeBuffer
        from .controller.syndrome_ingress import SyndromeIngress
        from .windows.window_manager import WindowManager

        config = resolve_run_configuration(self, root_seed)
        escalation_policy, plan, timing = config.escalation_policy, config.plan, config.timing
        conditional_release = (config.make_conditional_release(engine)
                               if config.make_conditional_release
                               else ConditionalRelease(engine))
        links = config.link_config.resolve()
        syndrome_buffer = SyndromeBuffer(
            capacity=config.buffering.upstream_packet_slots,
            memory_model=config.memory_model)
        pauli_frame = (None if config.pauli_frame is None
                       else config.pauli_frame.resolve(engine))

        # The window manager, the decoder manager and the factory refer to each
        # other; the closures below bind those names at first call.
        window_manager = WindowManager(
            engine, scheme=config.scheme, code_geometry=plan.code_geometry,
            resolved_operations=plan.resolved_operations,
            resolved_patches=plan.resolved_patches,
            links=links, conditional_release=conditional_release,
            boundary_policy=config.boundary_policy,
            window_interaction=config.window_interaction,
            planning_view_by_operation_id=config.view_by_id,
            fault_model_requirement_for=config.router.fault_model_requirement_for,
            feedback_boundary_mode=config.feedback_boundary_mode,
            error_model_provider=config.error_model_provider,
            syndrome_buffer=syndrome_buffer, pauli_frame=pauli_frame,
            retain_strong_context=escalation_policy.requires_strong_context,
            double_window=escalation_policy.double_window,
            capture_enabled=config.capture_switching_windows,
            escalation_policy=escalation_policy,
            submit_fn=lambda job, reserve_transfer=None:
                decoder_manager.enqueue(job, reserve_transfer),
            check_strong_route=lambda weak_job, strong_job:
                decoder_manager.check_strong_route(weak_job, strong_job),
            on_workload_complete=lambda: factory.shutdown())
        syndrome_ingress = (
            config.make_syndrome_ingress(engine, links, config.buffering,
                                         window_manager, syndrome_buffer)
            if config.make_syndrome_ingress else SyndromeIngress(
                engine, links=links, t_pack=timing.ticks("t_pack"),
                ingress_context_capacity=config.buffering.upstream_packet_slots,
                window_input_receiver=window_manager,
                feedback_memory_receiver=window_manager,
                syndrome_buffer=syndrome_buffer,
                policy=config.syndrome_ingress_policy,
                detector_formation=config.device))
        decoder_memory_transfer = (
            config.make_decoder_memory_transfer(engine, links, config.buffering)
            if config.make_decoder_memory_transfer else None)
        decoder_manager = DecoderManager(
            engine, router=config.router, scheduler=config.scheduler,
            unit_pools=config.unit_pools, num_units=config.num_units,
            bulk_strong=escalation_policy.bulk_strong, lane_policy=config.lane_policy,
            capture_enabled=config.capture_switching_windows,
            decoder_memory_transfer=decoder_memory_transfer,
            decoder_memory=config.decoder_memory,
            escalation_policy=escalation_policy, services=window_manager.escalation,
            on_window_decoded=window_manager.on_decode_done,
            on_strong_window_decoded=window_manager.on_strong_decode_done)
        window_manager.connect_idle_decode_demand_receiver(decoder_manager.submit_decode)
        factory = (config.make_factory(engine, decoder_manager)
                   if config.make_factory else _make_infinite(engine))
        if factory.engine is not engine:
            raise ValueError(f"{type(factory).__name__} uses a different engine")
        check_factory_decode_service(factory, decoder_manager)

        qpu = QPUDevice(engine, config.device, plan.round_ticks)
        uses_streams = config.protected_regions or any(
            op.stream_id is not None for op in config.all_operations)
        feedback_streams = FeedbackStreams(
            engine, qpu=qpu, window_manager=window_manager,
            regions=config.protected_regions,
            resolved_operations=plan.resolved_operations,
            resolved_patches=plan.resolved_patches,
            retry_ready_operations=lambda: execution_runtime.retry_ready_operations(),
        ) if uses_streams else NoFeedbackStreams()
        controller = Controller(
            engine, qpu=qpu, window_manager=window_manager,
            syndrome_ingress=syndrome_ingress,
            binary_availability_ticks=timing.ticks("t_binary_availability"),
            links=links, round_ticks=plan.round_ticks,
            code_geometry=plan.code_geometry,
            resolved_operations=plan.resolved_operations,
            resolved_patches=plan.resolved_patches, idle_policy=config.idle_policy,
            feedback_streams=feedback_streams)
        execution_runtime = ExecutionRuntime(
            engine, controller=controller, factory=factory,
            resource_claims_by_operation_id=config.resource_claims)
        controller.connect_runtime(execution_runtime)
        qpu.connect_readout_receiver(controller)
        qpu.connect_completion_receiver(controller._body_done)
        qpu.connect_idle_receiver(controller.emit_idle_round)
        metrics = (config.make_metrics(engine, window_manager, decoder_manager,
                                       execution_runtime, factory)
                   if config.make_metrics else [])
        if config.capture_switching_windows:
            from .observe.metrics import WindowSwitchingRecords
            metrics.append(WindowSwitchingRecords(window_manager, decoder_manager))
        metric_bindings = _metric_bindings(metrics)
        bind_run_seed(root_seed, _seed_roots(
            code=config.code, scheme=config.scheme,
            device=config.device, error_model_provider=config.error_model_provider,
            decoder_router=config.router,
            factory=factory, escalation_policy=escalation_policy, scheduler=config.scheduler,
            decoder_memory_transfer=decoder_manager.decoder_memory_transfer,
            lane_policy=config.lane_policy,
            boundary_policy=config.boundary_policy,
            window_interaction=config.window_interaction,
            idle_policy=config.idle_policy,
            conditional_release=conditional_release, syndrome_ingress=syndrome_ingress,
            controller=controller, qpu=qpu,
            execution_runtime=execution_runtime,
            pauli_frame=pauli_frame,
            memory_model=config.memory_model, metrics=metric_bindings))

        conditional_release.connect(controller, execution_runtime.on_decision)
        for op in config.ops:
            if op.blocked_by is not None:
                conditional_release.register_blocked_operation(op.id, op.blocked_by)
        for op in config.planned_operations:
            window_manager.register_op(op)
        window_manager.load_execution_plan(plan.execution, plan.buffering)
        resolved_by_id = {op.operation_id: op for op in plan.resolved_operations}
        for stream in config.dynamic_streams:
            window_manager._register_dynamic_stream(stream, resolved_by_id[stream.id])
        for _, metric in metric_bindings:
            engine.add_metric(metric)
        controller.load_program(ExecutionProgram(
            config.ops, config.decode_ops, config.dynamic_streams,
            config.protected_regions))
        engine.run()
        if window_manager.escalation.pending_escalations:
            raise RuntimeError(
                f"the run ended with pending strong escalations: "
                f"{window_manager.escalation.pending_escalations}")
        decoder_manager.check_decode_work_settled()
        check_ingress_settled = getattr(syndrome_ingress, "check_work_settled", None)
        if callable(check_ingress_settled):
            check_ingress_settled()
        result = capture_primary_result(
            engine, execution_runtime, window_manager, config.all_operations,
            metric_bindings, links, config.device)
        return CompletedRun(
            result, engine, window_manager, decoder_manager, execution_runtime,
            controller, qpu, conditional_release, factory,
            syndrome_buffer, syndrome_ingress, pauli_frame=pauli_frame)


def _metric_bindings(metrics):
    names = set()
    bindings = []
    for metric in metrics:
        if metric.name in names:
            raise ValueError(f"duplicate metric name {metric.name!r}")
        names.add(metric.name)
        bindings.append((metric.name, metric))
    return tuple(bindings)


def _root_seed(value):
    if value is None:
        return None
    value = int(value)
    if not 0 <= value < 2**64:
        raise ValueError("seed must be in [0, 2**64)")
    return value


def _seed_roots(**parts):
    field_path = lambda name: (RunSeedPathSegment("field", name),)
    metrics = parts.pop("metrics")
    roots = [(field_path(name), value) for name, value in parts.items()]
    for name, metric in metrics:
        roots.append((field_path("metrics") +
                      (RunSeedPathSegment("string_key", name),), metric))
    return tuple(roots)


def _make_infinite(engine):
    from .qpu.magic_state_factories import InfiniteFactory
    return InfiniteFactory(engine)


def simulate(run: RunSpec, verbose: bool = False) -> CompletedRun:
    return run.build(verbose=verbose)
