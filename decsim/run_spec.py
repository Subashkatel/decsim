"""Run configuration and the simulator's composition root.

RunSpec selects one implementation per seam, wires the runtime graph, executes
it once, and returns the scientific result with useful runtime owners. Derived
geometry/window planning and component seeding have their own focused owners.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from numbers import Integral
from typing import TYPE_CHECKING, Any, Callable, Optional

from .config import TimingConfig
from .message import ExecutionProgram, OperationPlanningView, RunSeedPathSegment, is_stable_string
from .seeding import bind_run_seed

if TYPE_CHECKING:
    from .decoder_memory import DecoderMemoryConfig


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
    orchestrator: Any
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
    strategy: Optional[Any] = None
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
    make_orchestrator: Optional[Callable] = None
    seed: Optional[int] = 0
    _build_state: str = field(default="unstarted", init=False, repr=False)

    def build(self, verbose: bool = False) -> CompletedRun:
        if self._build_state != "unstarted":
            raise RuntimeError(f"RunSpec build is already {self._build_state}")
        self._build_state = "committing"
        from .engine import Engine
        engine = Engine(verbose=verbose)
        try:
            root_seed = _root_seed(self.seed)
            completed = self._build_once(engine, root_seed)
        except BaseException as error:
            engine._invalidate(error)
            self._build_state = "invalid"
            raise
        self._build_state = "complete"
        return completed

    def _build_once(self, engine, root_seed) -> CompletedRun:
        from .controller import Controller
        from .syndrome_ingress import SyndromeIngress, SyndromeIngressPolicy
        from .decoder_manager import DecoderManager, StrategyServicesImpl
        from .decoders import CodeRouter
        from .devices import SyndromeBitDevice, TimingOnlyDevice
        from .qpu import QPUDevice
        from .link_profiles import logical_reference_profile
        from .orchestrators import ExecutionOrchestrator
        from .syndrome_buffer import SyndromeBuffer, SyndromeBufferingConfig
        from .planner import (
            _plan_execution,
            _validate_operation_graph,
            _validate_workload_identity,
        )
        from .rounds import GateRounds
        from .policies import Eager, Ignore
        from .schedulers import FifoScheduler
        from .schemes import SlidingWindowScheme
        from .switching import Baseline
        from .window_interactions import DefaultWindowInteraction
        from .window_manager import WindowManager

        strategy = self.strategy if self.strategy is not None else Baseline()
        requires_strong_context = strategy.requires_strong_context
        bulk_strong = strategy.bulk_strong
        double_window = strategy.double_window
        for name, value in (
            ("requires_strong_context", requires_strong_context),
            ("bulk_strong", bulk_strong),
            ("double_window", double_window),
        ):
            if type(value) is not bool:
                raise TypeError(f"strategy capability {name} must be an exact bool")

        if (self.ops is None) == (self.frontend is None):
            raise ValueError("provide exactly one of ops= or frontend=")
        source_ops = self.frontend.build() if self.frontend is not None else self.ops
        protected_regions = tuple(self.protected_regions)
        ops, decode_ops, dynamic_streams = _copy_workload(
            source_ops, self.decode_ops or (), self.dynamic_streams or (),
            self.feedback_boundary_mode)
        _validate_workload_identity(ops, decode_ops, dynamic_streams)
        if self.feedback_boundary_mode not in (
            "trailing_buffer", "measurement_closed"):
            raise ValueError("invalid feedback_boundary_mode")
        if type(self.record_switching_windows) is not bool:
            raise TypeError("record_switching_windows must be an exact bool")
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
        boundary_policy = self.boundary_policy or Eager()
        window_interaction = self.window_interaction or DefaultWindowInteraction()
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
                               else self.timing.round_us),
            retain_strong_context=requires_strong_context,
            double_window=double_window,
            has_open_ended_dynamic_streams=bool(dynamic_streams))
        strategy.validate_code_geometry(plan.code_geometry)
        resource_claims = {
            op.id: tuple(layout.resources_for(view_by_id[op.id])) for op in ops}

        device = self.device or TimingOnlyDevice()
        _install_device_circuits(device, all_operations)
        error_model_provider = (
            device if self.error_model_provider is None
            else self.error_model_provider)
        if (error_model_provider is not device and
                hasattr(error_model_provider, "operation_circuit_scope")):
            _install_device_circuits(error_model_provider, all_operations)
        if type(device) is SyndromeBitDevice and device.code is not code:
            raise ValueError(
                "SyndromeBitDevice.code must be the exact resolved run code")
        if self.router is not None and (self.decoder is not None or self.decoders):
            raise ValueError("router is exclusive with decoder and decoders")
        if self.router is None and self.decoder is None and planned_operations:
            raise ValueError("decoder is required when router is omitted")
        router = self.router or CodeRouter(
            default=self.decoder, by_code=dict(self.decoders))
        scheduler = (
            FifoScheduler() if self.scheduler is None else self.scheduler
        )
        idle_policy = self.idle_policy or Ignore()
        orchestrator = (self.make_orchestrator(engine)
                        if self.make_orchestrator
                        else ExecutionOrchestrator(engine))
        link_config = (self.links if self.links is not None else
                       logical_reference_profile())
        if (
            self.timing.ticks("t_binary_availability") > 0
            and not link_config.qc_excludes_controller_processing
        ):
            raise ValueError(
                "a separate controller readout cost requires a link "
                "profile whose QC latency excludes that cost"
            )
        links = link_config.resolve()
        buffering = self.syndrome_buffering or SyndromeBufferingConfig()
        if type(buffering) is not SyndromeBufferingConfig:
            raise TypeError("syndrome_buffering must be SyndromeBufferingConfig")
        syndrome_buffer = SyndromeBuffer(
            capacity=buffering.upstream_packet_slots,
            memory_model=self.memory_model,
        )
        pauli_frame = (
            None if self.pauli_frame is None else self.pauli_frame.resolve(engine)
        )
        if self.pauli_frame is not None and pauli_frame is None:
            raise TypeError("pauli_frame.resolve must return a PauliFrame")
        window_manager = WindowManager(
            engine, scheme=scheme, code_geometry=plan.code_geometry,
            resolved_operations=plan.resolved_operations,
            resolved_patches=plan.resolved_patches,
            links=links,
            orchestrator=orchestrator, boundary_policy=boundary_policy,
            window_interaction=window_interaction,
            planning_view_by_operation_id=view_by_id,
            fault_model_requirement_for=router.fault_model_requirement_for,
            feedback_boundary_mode=self.feedback_boundary_mode,
            error_model_provider=error_model_provider,
            syndrome_buffer=syndrome_buffer,
            pauli_frame=pauli_frame,
            retain_strong_context=requires_strong_context,
            double_window=double_window,
            capture_enabled=self.record_switching_windows)
        if self.make_syndrome_ingress is not None and self.syndrome_ingress_policy is not None:
            raise ValueError("syndrome_ingress_policy cannot be combined with make_syndrome_ingress")
        syndrome_ingress = (
            self.make_syndrome_ingress(
                engine, links, buffering, window_manager
            )
            if self.make_syndrome_ingress else SyndromeIngress(
                engine, links=links, t_pack=self.timing.ticks("t_pack"),
                ingress_context_capacity=buffering.upstream_packet_slots,
                window_input_receiver=window_manager,
                feedback_memory_receiver=window_manager,
                syndrome_buffer=syndrome_buffer,
                policy=(self.syndrome_ingress_policy
                        if self.syndrome_ingress_policy is not None
                        else SyndromeIngressPolicy()),
            )
        )
        if self.make_syndrome_ingress is not None:
            syndrome_ingress.syndrome_buffer = syndrome_buffer
        # A supplied transfer owns transport timing only: the decoder manager's
        # stager is always the receiver, so decoder memory cannot be
        # bypassed. The returned transfer must satisfy deliver plus cancel.
        if self.make_decoder_memory_transfer is None:
            from .decoder_memory_transfer import FixedLatencyDecoderMemoryTransfer
            decoder_memory_transfer = FixedLatencyDecoderMemoryTransfer(engine)
        else:
            decoder_memory_transfer = self.make_decoder_memory_transfer(
                engine, links, buffering
            )
        if (
            self.make_decoder_memory_transfer is not None
            and decoder_memory_transfer is None
        ):
            raise TypeError(
                "make_decoder_memory_transfer must return a "
                "DecoderMemoryTransfer"
            )
        decoder_manager = DecoderManager(
            engine, router=router, scheduler=scheduler,
            unit_pools=self.unit_pools,
            num_units=self.num_units if self.num_units is not None else 1,
            bulk_strong=bulk_strong,
            lane_policy=self.lane_policy,
            capture_enabled=self.record_switching_windows,
            decoder_memory_transfer=decoder_memory_transfer,
            decoder_memory=self.decoder_memory)
        decoder_memory_transfer = decoder_manager.decoder_memory_transfer
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
        qpu = QPUDevice(engine, device, plan.round_ticks)
        window_manager.connect_idle_decode_demand_receiver(
            decoder_manager.submit_decode)
        controller = Controller(
            engine, qpu=qpu, window_manager=window_manager,
            syndrome_ingress=syndrome_ingress,
            binary_availability_ticks=self.timing.ticks("t_binary_availability"),
            links=links, round_ticks=plan.round_ticks,
            code_geometry=plan.code_geometry,
            resolved_operations=plan.resolved_operations,
            resolved_patches=plan.resolved_patches, idle_policy=idle_policy,
            protected_regions=protected_regions)
        from .execution_runtime import ExecutionRuntime
        execution_runtime = ExecutionRuntime(
            engine, controller=controller, factory=factory,
            resource_claims_by_operation_id=resource_claims)
        controller.connect_runtime(execution_runtime)
        qpu.connect_readout_receiver(controller)
        qpu.connect_completion_receiver(controller._body_done)
        qpu.connect_idle_receiver(controller.emit_idle_round)
        metrics = (self.make_metrics(
            engine, window_manager, decoder_manager, execution_runtime, factory)
            if self.make_metrics else [])
        if self.record_switching_windows:
            from .metrics import WindowSwitchingRecords
            metrics.append(WindowSwitchingRecords(window_manager, decoder_manager))
        metric_bindings = _metric_bindings(metrics)
        bind_run_seed(root_seed, _seed_roots(
            code=code, scheme=scheme,
            device=device, error_model_provider=error_model_provider,
            decoder_router=router,
            factory=factory, strategy=strategy, scheduler=scheduler,
            decoder_memory_transfer=decoder_memory_transfer,
            lane_policy=self.lane_policy,
            boundary_policy=boundary_policy,
            window_interaction=window_interaction, idle_policy=idle_policy,
            orchestrator=orchestrator, syndrome_ingress=syndrome_ingress,
            controller=controller, qpu=qpu,
            execution_runtime=execution_runtime,
            pauli_frame=pauli_frame,
            memory_model=self.memory_model, metrics=metric_bindings))

        orchestrator.connect(controller, execution_runtime.on_decision)
        window_manager.on_workload_complete = factory.shutdown
        for op in ops:
            if op.blocked_by is not None:
                orchestrator.register_blocked_operation(op.id, op.blocked_by)
        for op in planned_operations:
            window_manager.register_op(op)
        window_manager.load_execution_plan(plan.execution, plan.buffering)
        resolved_by_id = {op.operation_id: op for op in plan.resolved_operations}
        for stream in dynamic_streams:
            window_manager._register_dynamic_stream(stream, resolved_by_id[stream.id])
        for _, metric in metric_bindings:
            engine.add_metric(metric)
        controller.load_program(ExecutionProgram(tuple(ops), tuple(decode_ops), tuple(dynamic_streams), tuple(protected_regions)))
        engine._start_running()
        engine.run()
        if window_manager.pending_escalations:
            raise RuntimeError(
                f"the run ended with pending strong escalations: "
                f"{window_manager.pending_escalations}")
        decoder_manager.check_decode_work_settled()
        check_ingress_settled = getattr(
            syndrome_ingress, "check_work_settled", None)
        if callable(check_ingress_settled):
            check_ingress_settled()
        engine._begin_finalization()
        from .views import capture_primary_result
        result = capture_primary_result(
            engine, execution_runtime, window_manager, all_operations,
            metric_bindings, links, device)
        engine._complete()
        return CompletedRun(
            result, engine, window_manager, decoder_manager, execution_runtime,
            controller, qpu, orchestrator, factory,
            syndrome_buffer, syndrome_ingress, pauli_frame=pauli_frame)


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
    selected = code if code is not None else SurfaceCodeModel(
        d=3 if distance is None else distance)
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


def _seed_roots(**parts):
    field_path = lambda name: (RunSeedPathSegment("field", name),)
    metrics = parts.pop("metrics")
    roots = [(field_path(name), value) for name, value in parts.items()]
    for name, metric in metrics:
        roots.append((field_path("metrics") +
                      (RunSeedPathSegment("string_key", name),), metric))
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


def simulate(run: RunSpec, verbose: bool = False) -> CompletedRun:
    return run.build(verbose=verbose)
