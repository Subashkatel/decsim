"""Run configuration and the simulator's composition root.

RunSpec selects one implementation per seam, wires the runtime graph, executes
it once, and returns the scientific result with useful runtime owners. Derived
geometry/window planning and component seeding have their own focused owners.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Callable, Optional

from .config import TimingConfig
from .message import OperationPlanningView, RunSeedPathSegment, is_stable_string
from .seeding import bind_run_seed


@dataclass(frozen=True)
class LogicalOperationResult:
    operation_id: int
    result_status: str
    logical_observables: Optional[tuple[int, ...]]
    stream_offset: Optional[int]


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
    chip_workload_complete: bool
    chip_done_ticks: int
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


@dataclass(frozen=True)
class CompletedRun:
    """Scientific result plus runtime owners useful to experiments."""

    result: PrimaryRunResult
    engine: Any
    window_manager: Any
    decoder_manager: Any
    chip: Any
    orchestrator: Any
    factory: Any
    controller: Any


@dataclass
class RunSpec:
    """Select, connect, and execute one simulator configuration."""

    ops: Optional[list] = None
    frontend: Optional[Any] = None
    decode_ops: Optional[list] = None
    dynamic_streams: Optional[list] = None
    code: Optional[Any] = None
    layout: Optional[Any] = None
    d: Optional[int] = None
    decoder: Optional[Any] = None
    decoders: dict = field(default_factory=dict)
    router: Optional[Any] = None
    strategy: Optional[Any] = None
    scheduler: Optional[Any] = None
    lane_policy: Optional[Any] = None
    deadline_policy: Optional[Any] = None
    unit_pools: Optional[dict] = None
    num_units: Optional[int] = None
    scheme: Optional[Any] = None
    rounds_policy: Optional[Any] = None
    boundary_policy: Optional[Any] = None
    window_interaction: Optional[Any] = None
    idle_policy: Optional[Any] = None
    max_idle_rounds: Optional[int] = None
    gates_start_on_round_boundaries: bool = False
    feedback_boundary_mode: str = "trailing_buffer"
    timing: TimingConfig = field(default_factory=TimingConfig)
    round_us: Optional[float] = None
    links: Optional[Any] = None
    device: Optional[Any] = None
    memory_model: Optional[Any] = None
    make_controller: Optional[Callable] = None
    make_factory: Optional[Callable] = None
    make_metrics: Optional[Callable] = None
    make_orchestrator: Optional[Callable] = None
    seed: Optional[int] = 0
    _build_state: str = field(default="unstarted", init=False, repr=False)

    def build(self, verbose: bool = False) -> CompletedRun:
        if self._build_state != "unstarted":
            raise RuntimeError(f"RunSpec build is already {self._build_state}")
        self._build_state = "committing"
        from .engine import Engine
        engine = Engine(verbose=verbose, construction_guarded=True)
        try:
            completed = self._build_once(engine)
        except BaseException as error:
            engine._invalidate(error)
            self._build_state = "invalid"
            raise
        self._build_state = "complete"
        return completed

    def _build_once(self, engine) -> CompletedRun:
        from .chip import Chip
        from .controllers import ModularController
        from .decoder_manager import DecoderManager, StrategyServicesImpl
        from .decoders import CodeRouter
        from .devices import ClockedDevice, SyndromeBitDevice, TimingOnlyDevice
        from .links import LinkModelConfig
        from .orchestrators import ExecutionOrchestrator
        from .payload_store import PayloadStore
        from .planner import (
            GateRounds,
            _plan_execution,
            _validate_operation_graph,
            _validate_workload_identity,
        )
        from .policies import Eager, Ignore
        from .schedulers import EnqueueTimeDeadline, FifoScheduler
        from .schemes import SlidingWindowScheme
        from .switching import Baseline
        from .window_interactions import DefaultWindowInteraction
        from .window_manager import WindowManager

        if (self.ops is None) == (self.frontend is None):
            raise ValueError("provide exactly one of ops= or frontend=")
        source_ops = self.frontend.build() if self.frontend is not None else self.ops
        ops, decode_ops, dynamic_streams = _copy_workload(
            source_ops, self.decode_ops or (), self.dynamic_streams or (),
            self.feedback_boundary_mode)
        _validate_workload_identity(ops, decode_ops, dynamic_streams)
        if self.feedback_boundary_mode not in (
            "trailing_buffer", "measurement_closed"):
            raise ValueError("invalid feedback_boundary_mode")
        if {op.id for op in decode_ops} & {op.id for op in dynamic_streams}:
            raise ValueError("an operation cannot be in decode_ops and dynamic_streams")
        all_operations = _unique_operations(ops + decode_ops + dynamic_streams)
        views = tuple(OperationPlanningView.from_operation(op)
                      for op in all_operations)
        view_by_id = {view.id: view for view in views}
        _validate_operation_graph(
            list(ops), validate_blockers=True,
            external_blocker_ids=(op.id for op in decode_ops + dynamic_streams))

        code, layout = _select_code(self.d, self.code, self.layout)
        scheme = self.scheme or SlidingWindowScheme()
        rounds_policy = self.rounds_policy or GateRounds()
        strategy = self.strategy or Baseline()
        boundary_policy = self.boundary_policy or Eager()
        if dynamic_streams and type(scheme) is not SlidingWindowScheme:
            raise ValueError("dynamic streams require SlidingWindowScheme")
        strategy.validate_declared_run(
            scheme=scheme, boundary_policy=boundary_policy,
            has_dynamic_streams=bool(dynamic_streams),
            static_decode_plan_selected=self.decode_ops is not None,
            has_frontend=self.frontend is not None)
        strategy.validate_operations(views)

        planned_operations = _decode_plan_operations(
            ops, decode_ops, dynamic_streams,
            static_decode_selected=self.decode_ops is not None)
        plan = _plan_execution(
            operations=views,
            planned_operation_ids=tuple(op.id for op in planned_operations),
            code=code, layout=layout, scheme=scheme,
            rounds_policy=rounds_policy,
            fallback_round_us=(self.round_us if self.round_us is not None
                               else self.timing.round_us))
        strategy.validate_code_geometry(plan.code_geometry)
        resource_claims = {
            op.id: tuple(layout.resources_for(view_by_id[op.id])) for op in ops}

        device = self.device or TimingOnlyDevice()
        _install_device_circuits(device, all_operations)
        if type(device) is SyndromeBitDevice and device.code is not code:
            raise ValueError(
                "SyndromeBitDevice.code must be the exact resolved run code")
        if self.router is not None and (self.decoder is not None or self.decoders):
            raise ValueError("router is exclusive with decoder and decoders")
        if self.router is None and self.decoder is None and planned_operations:
            raise ValueError("decoder is required when router is omitted")
        router = self.router or CodeRouter(
            default=self.decoder, by_code=dict(self.decoders))
        scheduler = self.scheduler or FifoScheduler()
        deadline_policy = self.deadline_policy or EnqueueTimeDeadline()
        idle_policy = self.idle_policy or Ignore()
        window_interaction = self.window_interaction or DefaultWindowInteraction()
        orchestrator = (self.make_orchestrator(engine)
                        if self.make_orchestrator
                        else ExecutionOrchestrator(engine))
        link_config = (self.links if self.links is not None else
                       LinkModelConfig.reference_fixed_latency_profile())
        links = link_config.resolve()
        controller = (self.make_controller(engine, links)
                      if self.make_controller else
                      ModularController(engine, links=links,
                                        t_pack=self.timing.ticks("t_pack")))
        window_manager = WindowManager(
            engine, scheme=scheme, code_geometry=plan.code_geometry,
            resolved_operations=plan.resolved_operations,
            resolved_patches=plan.resolved_patches,
            deadline_policy=deadline_policy, links=links,
            orchestrator=orchestrator, boundary_policy=boundary_policy,
            window_interaction=window_interaction,
            planning_view_by_operation_id=view_by_id,
            fault_model_requirement_for=router.fault_model_requirement_for,
            feedback_boundary_mode=self.feedback_boundary_mode,
            syndrome_source=device,
            store=PayloadStore(memory_model=self.memory_model),
            switching_active=hasattr(strategy, "keep_weak_result"))
        decoder_manager = DecoderManager(
            engine, router=router, scheduler=scheduler,
            unit_pools=self.unit_pools,
            num_units=self.num_units if self.num_units is not None else 1,
            bulk_strong=getattr(strategy, "bulk_strong", False),
            lane_policy=self.lane_policy)
        services = StrategyServicesImpl(engine, window_manager, decoder_manager)
        window_manager.strategy = strategy
        window_manager.services = services
        window_manager.submit_fn = decoder_manager.enqueue
        decoder_manager.strategy = strategy
        decoder_manager.services = services
        decoder_manager.on_window_decoded = window_manager.on_decode_done
        decoder_manager.on_strong_window_decoded = window_manager.on_strong_decode_done

        factory = (self.make_factory(engine, decoder_manager)
                   if self.make_factory else _make_infinite(engine))
        if factory.engine is not engine:
            raise ValueError(
                f"{type(factory).__name__} uses a different engine")
        _check_factory_decode_service(factory, decoder_manager)
        source = ClockedDevice(
            engine, device, controller, window_manager.on_syndrome_arrival,
            {op.operation_id: op.round_count for op in plan.resolved_operations})
        chip = Chip(
            engine, source=source, controller=controller,
            window_manager=window_manager, decode_service=decoder_manager,
            factory=factory, round_ticks=plan.round_ticks,
            code_geometry=plan.code_geometry,
            resolved_operations=plan.resolved_operations,
            resolved_patches=plan.resolved_patches,
            idle_policy=idle_policy,
            resource_claims_by_operation_id=resource_claims,
            max_idle_rounds=self.max_idle_rounds,
            gates_start_on_round_boundaries=self.gates_start_on_round_boundaries,
            frame=orchestrator.frame)
        metrics = (self.make_metrics(
            engine, window_manager, decoder_manager, chip, factory)
            if self.make_metrics else [])
        metric_bindings = _metric_bindings(metrics)
        bind_run_seed(_root_seed(self.seed), _seed_roots(
            self, code=code, layout=layout, scheme=scheme,
            rounds_policy=rounds_policy, device=device, decoder_router=router,
            factory=factory, strategy=strategy, scheduler=scheduler,
            lane_policy=self.lane_policy,
            deadline_policy=deadline_policy, boundary_policy=boundary_policy,
            window_interaction=window_interaction, idle_policy=idle_policy,
            orchestrator=orchestrator, controller=controller,
            metrics=metric_bindings, operations=all_operations))

        orchestrator.connect(controller, chip.on_decision)
        window_manager.on_workload_complete = factory.shutdown
        for op in ops:
            if op.blocked_by is not None:
                orchestrator.register_blocked_operation(op.id, op.blocked_by)
        for op in planned_operations:
            window_manager.register_op(op)
        window_manager.load_execution_plan(plan.execution)
        resolved_by_id = {op.operation_id: op for op in plan.resolved_operations}
        for stream in dynamic_streams:
            window_manager._register_dynamic_stream(stream, resolved_by_id[stream.id])
        for _, metric in metric_bindings:
            engine.add_metric(metric)
        chip._load(list(ops))
        engine._start_running()
        engine.run()
        decoder_manager.check_decode_work_settled()
        engine._begin_finalization()
        result = _capture_result(
            engine, chip, window_manager, all_operations,
            metric_bindings, links)
        engine._complete()
        return CompletedRun(
            result, engine, window_manager, decoder_manager, chip,
            orchestrator, factory, controller)


def _select_code(distance, code, layout):
    from .codes import SurfaceCodeModel
    from .layouts import UniformLayout
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
    selected = code or SurfaceCodeModel(d=distance or 3)
    return selected, UniformLayout(selected)


def _copy_workload(source_ops, decode_ops, dynamic_streams, feedback_mode):
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
        if type(operation.circuit) is not stim.Circuit:
            raise TypeError("active operation circuit is not an exact stim.Circuit")
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


def _metric_bindings(metrics):
    names = set()
    bindings = []
    for metric in metrics:
        if not is_stable_string(metric.name) or not metric.name:
            raise ValueError("metric names must be nonempty Unicode strings")
        if metric.name in names:
            raise ValueError(f"duplicate metric name {metric.name!r}")
        names.add(metric.name)
        bindings.append((metric.name, metric))
    return tuple(bindings)


def _root_seed(value):
    if value is None:
        return None
    if type(value) is bool or not isinstance(value, Integral):
        raise TypeError("seed must be a 64-bit unsigned integer or None")
    value = int(value)
    if not 0 <= value < 2**64:
        raise ValueError("seed must be in [0, 2**64)")
    return value


def _seed_roots(spec, **parts):
    field_path = lambda name: (RunSeedPathSegment("field", name),)
    metrics = parts.pop("metrics")
    operations = parts.pop("operations")
    roots = [(field_path(name), value) for name, value in parts.items()]
    if spec.frontend is not None:
        roots.append((field_path("frontend"), spec.frontend))
    if spec.memory_model is not None:
        roots.append((field_path("memory_model"), spec.memory_model))
    for name, metric in metrics:
        roots.append((field_path("metrics") +
                      (RunSeedPathSegment("string_key", name),), metric))
    if getattr(parts["device"], "operation_circuit_scope", "none") == "per_operation":
        seen = set()
        for operation in operations:
            if operation.id not in seen and operation.circuit is not None:
                seen.add(operation.id)
                roots.append((field_path("workload_circuits") +
                              (RunSeedPathSegment("integer_key", operation.id),),
                              operation.circuit))
    return tuple(roots)


def _make_infinite(engine):
    from .factories import InfiniteFactory
    return InfiniteFactory(engine)


def _check_factory_decode_service(factory, decoder_manager):
    from .factories import DistillationFactory, MultiLevelDistillationFactory
    if type(factory) not in (DistillationFactory, MultiLevelDistillationFactory):
        return
    expected = decoder_manager if factory.n_corr > 0 else None
    if factory.decode_service is not expected:
        raise ValueError(
            f"{type(factory).__name__} decode_service must be run-owned")


def _capture_result(engine, chip, window_manager, operations,
                    metric_bindings, links):
    operation_by_id = {operation.id: operation for operation in operations}
    rows = []
    for operation_id in sorted(operation_by_id):
        logical = window_manager.op_results.get(operation_id)
        rows.append(LogicalOperationResult(
            operation_id,
            "logical_observables" if logical is not None else "no_logical_output",
            (tuple(_logical_bit(bit) for bit in logical)
             if logical is not None else None),
            operation_by_id[operation_id].stream_offset))
    metric_rows = tuple(MetricResultRecord(
        name, copy.deepcopy(engine._invoke_metric_callback(
            metric.result, callback_kind="result")))
        for name, metric in metric_bindings)
    if engine._event_queue or not chip.workload_complete:
        raise RuntimeError("primary run ended before workload completed")
    return PrimaryRunResult(
        "complete", True, True, True, chip.last_finish_time, engine.now,
        tuple(rows), copy.deepcopy(links.traffic_json_value()), metric_rows)


def _logical_bit(value):
    if type(value) is not int or value not in (0, 1):
        raise TypeError(f"logical observables must contain bits; got {value!r}")
    return value


def simulate(run: RunSpec, verbose: bool = False) -> CompletedRun:
    return run.build(verbose=verbose)
