"""RunSpec: the typed run configuration and composition root.

The one core module allowed to import part implementations — that is its
job: every field picks one implementation per seam, and ``RunSpec.build()``
wires them into a runnable World in a fixed order (the frozen timing
goldens depend on that order). Experiment code still never appears here;
experiments hand a RunSpec pre-built objects. ``simulate(run)`` (below)
then drives the result.

Defaults: sliding window + Baseline strategy + Eager boundaries +
GateRounds + InfiniteFactory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Callable, Optional, TYPE_CHECKING

from .message import Operation
from .pauli_frame import PauliFrame
from .config import TimingConfig, us

if TYPE_CHECKING:
    from .protocols import (
        CodeModel,
        DecodingScheme,
        ExecutionPlanner,
        LayoutModel,
        RoundsPolicy,
    )

FEEDBACK_BOUNDARY_MODES = ("trailing_buffer", "measurement_closed")


@dataclass(frozen=True)
class ResolvedPlanningParts:
    """Exact planning/runtime collaborators selected for one build."""

    code: "CodeModel"
    layout: "LayoutModel"
    scheme: "DecodingScheme"
    rounds_policy: "RoundsPolicy"
    planner: "ExecutionPlanner"


@dataclass
class RunSpec:
    """Typed simulator configuration with one owner per planning choice.

    Supply at most one of ``d``, ``code``, or ``layout``; omitting all three
    selects distance 3. A supplied ``planner`` owns its layout, scheme, and
    rounds policy, so those sibling fields must be omitted. Custom magic-state
    factories are constructed through ``make_factory(engine, cluster)``.
    """

    # workload (exactly one of ops/frontend)
    ops: Optional[list] = None
    frontend: Optional[Any] = None
    decode_ops: Optional[list] = None
    dynamic_streams: Optional[list] = None

    # code / layout
    code: Optional[Any] = None
    layout: Optional[Any] = None
    d: Optional[int] = None

    # decode stage
    decoder: Optional[Any] = None
    decoders: dict = field(default_factory=dict)
    router: Optional[Any] = None
    strategy: Optional[Any] = None            # default Baseline (build)
    scheduler: Optional[Any] = None           # default FifoScheduler
    deadline_policy: Optional[Any] = None     # default EnqueueTimeDeadline
    unit_pools: Optional[dict] = None
    num_units: Optional[int] = None      # default 1; unit_pools= takes precedence

    # windowing / rounds
    scheme: Optional[Any] = None              # default SlidingWindowScheme
    rounds_policy: Optional[Any] = None       # default GateRounds
    planner: Optional[Any] = None

    # control loop
    boundary_policy: Optional[Any] = None     # default Eager (speculative)
    window_interaction: Optional[Any] = None  # default defect-mask interaction
    idle_policy: Optional[Any] = None         # default Ignore
    max_idle_rounds: Optional[int] = None
    gates_start_on_round_boundaries: bool = False
    feedback_boundary_mode: str = "trailing_buffer"
    orchestrator: Optional[Any] = None       # default ExecutionOrchestrator

    # environment
    timing: TimingConfig = field(default_factory=TimingConfig)
    round_us: Optional[float] = None          # overrides timing.round_us
    device: Optional[Any] = None              # default TimingOnlyDevice
    memory_model: Optional[Any] = None        # port 18; default unbounded
    make_controller: Optional[Callable] = None  # (engine) -> Controller (port 14)
    make_factory: Optional[Callable] = None   # (engine, cluster) -> factory
    make_metrics: Optional[Callable] = None   # (engine, cluster, gate, factory)
    seed: int = 0

    # ------------------------------------------------------------- validate

    def validate(self) -> None:
        """Cross-part validation before any build."""
        planning = self._validate_configuration()
        operations = list(self.ops or [])
        operations.extend(self.decode_ops or [])
        operations.extend(self.dynamic_streams or [])
        self._validate_layout_selection(planning, operations)

    def _validate_configuration(self) -> ResolvedPlanningParts:
        """Validate configuration-only state and resolve planning once."""
        if (self.ops is None) == (self.frontend is None):
            raise ValueError("provide exactly one of ops= or frontend=")
        self._validate_supplied_parts()
        auxiliary_ops = list(self.decode_ops or []) + list(self.dynamic_streams or [])
        if self.ops is not None:
            from .planner import _validate_operation_graph
            _validate_operation_graph(
                self.ops, validate_blockers=True,
                external_blocker_ids=(operation.id for operation in auxiliary_ops))
        for label, operations in (("decode_ops", self.decode_ops),
                                  ("dynamic_streams", self.dynamic_streams)):
            seen_ids = set()
            for operation in operations or []:
                if operation.id in seen_ids:
                    raise ValueError(
                        f"duplicate operation id {operation.id} in {label}")
                seen_ids.add(operation.id)
        if self.feedback_boundary_mode not in FEEDBACK_BOUNDARY_MODES:
            raise ValueError(
                f"feedback_boundary_mode must be one of "
                f"{FEEDBACK_BOUNDARY_MODES} (got {self.feedback_boundary_mode!r})")
        if self.decode_ops and self.dynamic_streams:
            static_ids = {op.id for op in self.decode_ops}
            dyn_ids = {op.id for op in self.dynamic_streams}
            overlap = static_ids & dyn_ids
            if overlap:
                raise ValueError(f"ops {sorted(overlap)} appear in both "
                                 f"decode_ops and dynamic_streams (a stream "
                                 f"is planned statically OR dynamically)")
        planning = self._resolve_planning_parts()
        self._validate_resolved_planning(planning)
        self._validate_cross_part_combinations(planning)
        return planning

    def _validate_supplied_parts(self) -> None:
        """Validate every externally supplied part against its public port."""
        from . import protocols

        parts = (
            ("frontend", self.frontend, protocols.InputFrontend),
            ("code", self.code, protocols.CodeModel),
            ("layout", self.layout, protocols.LayoutModel),
            ("decoder", self.decoder, protocols.Decoder),
            ("router", self.router, protocols.DecoderRouter),
            ("strategy", self.strategy, protocols.DecodingStrategy),
            ("scheduler", self.scheduler, protocols.Scheduler),
            ("deadline_policy", self.deadline_policy, protocols.DeadlinePolicy),
            ("scheme", self.scheme, protocols.DecodingScheme),
            ("rounds_policy", self.rounds_policy, protocols.RoundsPolicy),
            ("planner", self.planner, protocols.ExecutionPlanner),
            ("boundary_policy", self.boundary_policy,
             protocols.BoundaryPolicy),
            ("window_interaction", self.window_interaction,
             protocols.WindowInteraction),
            ("idle_policy", self.idle_policy, protocols.IdlePolicy),
            ("orchestrator", self.orchestrator, protocols.Orchestrator),
            ("memory_model", self.memory_model, protocols.MemoryModel),
        )
        for name, part, protocol in parts:
            _validate_protocol_part(name, part, protocol)
        self._validate_device_capabilities(protocols.SyndromeDevice)
        for name, decoder in self.decoders.items():
            _validate_protocol_part(
                f"decoders[{name!r}]", decoder, protocols.Decoder)
        _validate_callable_arity("make_controller", self.make_controller, 1)
        _validate_callable_arity("make_factory", self.make_factory, 2)
        _validate_callable_arity("make_metrics", self.make_metrics, 4)

    def _validate_device_capabilities(self, device_protocol) -> None:
        """Check only device methods reachable in this run configuration."""
        methods = [
            "begin_operation",
            "round_payloads",
            "window_models_for_operation",
        ]
        if self.dynamic_streams:
            methods.extend([
                "register_dynamic_stream",
                "validate_stream_length",
                "window_model_for_stream",
            ])
            if getattr(self.idle_policy, "mode", "ignore") == "extend_stream":
                methods.append("idle_round_payloads")
        if (self.strategy is not None
                and hasattr(self.strategy, "keep_weak_result")):
            methods.append("strong_window_model_for_operation")
        _validate_protocol_methods(
            "device", self.device, device_protocol, methods)

    def _resolve_planning_parts(self) -> ResolvedPlanningParts:
        from .codes import SurfaceCodeModel
        from .layouts import UniformLayout
        from .planner import GateRounds, WindowPlanner
        from .schemes import SlidingWindowScheme

        if self.planner is not None:
            sibling_names = [
                name
                for name in ("d", "code", "layout", "scheme", "rounds_policy")
                if getattr(self, name) is not None
            ]
            if sibling_names:
                supplied = ", ".join(sibling_names)
                raise ValueError(
                    "planner owns layout, scheme, and rounds_policy; "
                    f"remove sibling planning fields: {supplied}")
            planner = self.planner
            scheme = planner.scheme
            layout = planner.layout
            rounds_policy = planner.rounds_policy
            from . import protocols
            _validate_protocol_part(
                "planner.layout",
                layout,
                protocols.LayoutModel,
            )
            code = _single_layout_code(layout, "planner.layout")
            return ResolvedPlanningParts(
                code=code,
                layout=layout,
                scheme=scheme,
                rounds_policy=rounds_policy,
                planner=planner,
            )

        code_sources = [
            name
            for name in ("d", "code", "layout")
            if getattr(self, name) is not None
        ]
        if len(code_sources) > 1:
            supplied = ", ".join(code_sources)
            raise ValueError(
                f"multiple code sources supplied: {supplied}; "
                "provide exactly one of d, code, or layout")

        if self.layout is not None:
            layout = self.layout
            code = _single_layout_code(layout, "layout")
        else:
            code = self.code
            if code is None:
                distance = self.d if self.d is not None else 3
                code = SurfaceCodeModel(d=distance)
            layout = UniformLayout(code)

        scheme = self.scheme if self.scheme is not None else SlidingWindowScheme()
        rounds_policy = (
            self.rounds_policy
            if self.rounds_policy is not None
            else GateRounds()
        )
        planner = WindowPlanner(scheme, layout, rounds_policy)
        return ResolvedPlanningParts(
            code=code,
            layout=layout,
            scheme=scheme,
            rounds_policy=rounds_policy,
            planner=planner,
        )

    def _validate_resolved_planning(
        self,
        planning: ResolvedPlanningParts,
    ) -> None:
        from . import protocols

        supplied_planner = self.planner is not None
        parts = (
            ("resolved code", planning.code, protocols.CodeModel),
            (
                "planner.layout" if supplied_planner else "resolved layout",
                planning.layout,
                protocols.LayoutModel,
            ),
            (
                "planner.scheme" if supplied_planner else "resolved scheme",
                planning.scheme,
                protocols.DecodingScheme,
            ),
            (
                (
                    "planner.rounds_policy"
                    if supplied_planner
                    else "resolved rounds_policy"
                ),
                planning.rounds_policy,
                protocols.RoundsPolicy,
            ),
            ("resolved planner", planning.planner, protocols.ExecutionPlanner),
        )
        for name, part, protocol in parts:
            _validate_protocol_part(name, part, protocol)

        declared_code = _single_layout_code(
            planning.layout,
            "resolved layout",
        )
        if declared_code is not planning.code:
            raise ValueError(
                "resolved layout declared a code different from the resolved "
                "planning/runtime code")
        if planning.planner.scheme is not planning.scheme:
            raise ValueError("resolved planner uses a different scheme")
        if planning.planner.layout is not planning.layout:
            raise ValueError("resolved planner uses a different layout")
        if planning.planner.rounds_policy is not planning.rounds_policy:
            raise ValueError("resolved planner uses a different rounds_policy")

        if getattr(planning.code, "buffer_rounds_override", None) is None:
            # An explicit buffer override is the expert escape hatch, including
            # measurement-closed streams that need no trailing buffer.
            planning.scheme.validate_buffer(planning.code)

        probe = Operation(-1, "probe", (0,))
        round_count = planning.rounds_policy.rounds_for(
            probe,
            planning.code,
        )
        if round_count < 1:
            raise ValueError(
                f"rounds_policy returned {round_count} (< 1)")

    def _validate_cross_part_combinations(
        self,
        planning: ResolvedPlanningParts,
    ) -> None:
        from . import protocols

        for name, part in (
            ("strategy", self.strategy),
            ("decoder", self.decoder),
        ):
            if part is None or not isinstance(
                part,
                protocols.CrossPartValidator,
            ):
                continue
            _validate_protocol_part(
                f"{name} cross-part validator",
                part,
                protocols.CrossPartValidator,
            )
            part.validate(self, planning)

    @staticmethod
    def _validate_layout_selection(
        planning: ResolvedPlanningParts,
        operations,
    ) -> None:
        for operation in operations:
            selected_code = planning.layout.code_for_op(operation)
            if selected_code is not planning.code:
                raise ValueError(
                    f"layout {planning.layout!r} operation {operation.id} "
                    f"selected {selected_code!r}, but resolved "
                    f"planning/runtime code is {planning.code!r}")

            patch_ids = operation.patches
            if not patch_ids:
                patch_ids = operation.qubits
            if not patch_ids:
                patch_ids = (0,)
            for patch_id in patch_ids:
                selected_code = planning.layout.code_for_patch(patch_id)
                if selected_code is not planning.code:
                    raise ValueError(
                        f"layout {planning.layout!r} patch {patch_id!r} "
                        f"selected {selected_code!r} in "
                        f"operation {operation.id}, but resolved "
                        f"planning/runtime code is {planning.code!r}")

    # ---------------------------------------------------------------- build

    def build(self, verbose: bool = False) -> "World":
        """Construct and wire every component in the canonical order."""
        planning = self._validate_configuration()
        from .policies import Eager
        from .decoders import CodeRouter
        from .engine import Engine
        from .orchestrators import ExecutionOrchestrator
        from .policies import Ignore
        from .payload_store import PayloadStore
        from .decoder_manager import StrategyServicesImpl, DecoderManager
        from .chip import Chip
        from .schedulers import EnqueueTimeDeadline, FifoScheduler
        from .devices import ClockedDevice, TimingOnlyDevice
        from .switching import Baseline
        from .controllers import ModularController, LinkModel
        from .window_manager import WindowManager
        from .window_interactions import DefaultWindowInteraction

        ops = self.frontend.build() if self.frontend is not None else self.ops
        if self.frontend is not None:
            from .planner import _validate_operation_graph
            auxiliary_ids = (operation.id for operation in
                             list(self.decode_ops or [])
                             + list(self.dynamic_streams or []))
            _validate_operation_graph(
                ops, validate_blockers=True,
                external_blocker_ids=auxiliary_ids)
        self._apply_feedback_boundary_default(ops)

        all_operations = list(ops)
        all_operations.extend(self.decode_ops or [])
        all_operations.extend(self.dynamic_streams or [])
        self._validate_layout_selection(planning, all_operations)
        _validate_program_order(ops, planning.layout)

        engine = Engine(verbose=verbose)
        strategy = self.strategy if self.strategy is not None else Baseline()
        scheduler = self.scheduler if self.scheduler is not None \
            else FifoScheduler()
        deadline_policy = self.deadline_policy if self.deadline_policy is not None \
            else EnqueueTimeDeadline()
        boundary_policy = self.boundary_policy if self.boundary_policy is not None \
            else Eager()
        window_interaction = (
            self.window_interaction
            if self.window_interaction is not None
            else DefaultWindowInteraction()
        )
        idle_policy = self.idle_policy if self.idle_policy is not None else Ignore()
        device = self.device if self.device is not None else TimingOnlyDevice()
        decoder = self.decoder
        if decoder is None:
            raise ValueError("RunSpec.decoder is required (a Decoder part), "
                             "e.g. PerRoundDecoder(tau_us=1.0)")
        router = self.router if self.router is not None \
            else CodeRouter(default=decoder, by_code=dict(self.decoders))
        orchestrator = self.orchestrator if self.orchestrator is not None \
            else ExecutionOrchestrator(engine)

        controller = self.make_controller(engine) \
            if self.make_controller is not None \
            else ModularController(engine, links=LinkModel.from_timing(self.timing),
                              t_pack=self.timing.ticks("t_pack"))
        from .protocols import Controller, MagicStateFactory, Metric
        _validate_protocol_part("controller", controller, Controller)
        # the whole fabric shares the controller's LinkModel: the window
        # manager's dd/do hops ride the same links a custom controller set
        links = getattr(controller, "links", None) or LinkModel.from_timing(self.timing)

        store = PayloadStore(memory_model=self.memory_model) \
            if self.memory_model is not None else None
        window_manager = WindowManager(
            engine, scheme=planning.scheme, layout=planning.layout,
            rounds_policy=planning.rounds_policy,
            code=planning.code, deadline_policy=deadline_policy, links=links,
            orchestrator=orchestrator, boundary_policy=boundary_policy,
            window_interaction=window_interaction,
            syndrome_source=device, store=store,
            switching_active=hasattr(strategy, "keep_weak_result"))
        pool = DecoderManager(
            engine, router=router, scheduler=scheduler,
            unit_pools=self.unit_pools,
            num_units=self.num_units if self.num_units is not None else 1,
            ws_delay_ticks=links.ws.cost(),
            bulk_strong=getattr(strategy, "bulk_strong", False))
        services = StrategyServicesImpl(engine, window_manager, pool)
        window_manager.strategy = strategy
        window_manager.services = services
        window_manager.submit_fn = pool.enqueue
        window_manager.needs_hyperedges = getattr(decoder, "needs_hyperedges", False) \
            or any(getattr(dec, "needs_hyperedges", False)
                   for dec in self.decoders.values())
        pool.strategy = strategy
        pool.services = services
        pool.on_window_decoded = window_manager.on_decode_done
        pool.on_strong_window_decoded = window_manager.on_strong_decode_done

        cluster = ClusterFacade(
            window_manager,
            pool,
            planning.layout,
            planning.planner,
            strategy,
            self._decode_plan_operations(ops),
        )

        factory = self.make_factory(engine, cluster) \
            if self.make_factory is not None else _make_infinite(engine)
        _validate_protocol_part("factory", factory, MagicStateFactory)
        if factory.engine is not engine:
            raise ValueError(
                f"{type(factory).__name__} uses a different engine from "
                "the RunSpec build")
        source = ClockedDevice(engine, device, controller, window_manager,
                               window_manager.rounds_for)
        round_us = self.round_us if self.round_us is not None \
            else self.timing.round_us
        gate = Chip(
            engine, source=source, controller=controller, cluster=cluster,
            factory=factory, round_ticks=us(round_us),
            code_distance=planning.code.distance, idle_policy=idle_policy,
            max_idle_rounds=self.max_idle_rounds,
            gates_start_on_round_boundaries=self.gates_start_on_round_boundaries,
            frame=getattr(orchestrator, "frame", None) or PauliFrame())

        orchestrator.connect(controller, gate.on_decision)
        window_manager.on_workload_complete = factory.shutdown
        for op in ops:
            if op.blocked_by is not None:
                orchestrator.register_blocked_operation(op.id, op.blocked_by)
        # Parity ordering (wiring.py): plan + load BEFORE dynamic streams
        # (prepare_execution ran before _register_dynamic_streams).
        cluster.prepare(ops)
        for stream in (self.dynamic_streams or []):
            window_manager.register_dynamic_stream(stream, planning.code)
        if self.make_metrics is not None:
            metrics = self.make_metrics(engine, cluster, gate, factory)
            for index, metric in enumerate(metrics):
                _validate_protocol_part(
                    f"make_metrics result {index}", metric, Metric)
                engine.add_metric(metric)

        return World(
            engine=engine,
            ops=ops,
            window_manager=window_manager,
            pool=pool,
            gate=gate,
            orchestrator=orchestrator,
            factory=factory,
            controller=controller,
            cluster=cluster,
            planning=planning,
        )

    def _apply_feedback_boundary_default(self, ops) -> None:
        operations = list(ops) + list(self.decode_ops or []) \
            + list(self.dynamic_streams or [])
        for operation in operations:
            if operation.feedback_boundary_mode is None:
                operation.feedback_boundary_mode = self.feedback_boundary_mode

    def _decode_plan_operations(self, ops):
        """Operations that receive compile-time decode windows (wiring parity)."""
        if self.decode_ops is not None:
            return self.decode_ops
        if not self.dynamic_streams:
            return None
        dynamic_ids = {stream.id for stream in self.dynamic_streams}
        return [op for op in ops if op.stream_id not in dynamic_ids]


def _single_layout_code(layout, owner_name: str):
    codes = list(layout.codes())
    if len(codes) != 1:
        raise ValueError(
            f"{owner_name} must declare exactly one planning/runtime code "
            f"(got {len(codes)})")
    return codes[0]


def _validate_program_order(ops, layout) -> None:
    """Static twin of Chip._claim_resources' conflict guard.

    The chip raises when two operations hold a shared resource concurrently,
    which makes that check schedule-dependent: a missing ordering edge can
    hide for as long as the timing happens to separate the two holders.
    This walks each resource's holders in list order and requires every
    consecutive pair to be ordered by the dependency DAG (a path of
    predecessor edges, not necessarily a direct edge), so a malformed
    operation list fails at build time, deterministically.
    """
    operation_by_id = {operation.id: operation for operation in ops}
    ancestor_cache: dict = {}

    def ancestors_of(op_id):
        """All op ids reachable from op_id via predecessor edges."""
        cached = ancestor_cache.get(op_id)
        if cached is not None:
            return cached
        ancestors: set = set()
        stack = [op_id]
        while stack:
            operation = operation_by_id.get(stack.pop())
            if operation is None:
                continue
            for predecessor_id in operation.predecessors:
                if predecessor_id not in ancestors:
                    ancestors.add(predecessor_id)
                    stack.append(predecessor_id)
        ancestor_cache[op_id] = ancestors
        return ancestors

    last_holder: dict = {}
    for operation in ops:
        for claim in layout.resources_for(operation):
            for resource_id in sorted(claim.ids, key=repr):
                key = (claim.kind, resource_id)
                previous_id = last_holder.get(key)
                if (previous_id is not None and previous_id != operation.id
                        and previous_id not in ancestors_of(operation.id)):
                    previous = operation_by_id[previous_id]
                    raise ValueError(
                        f"{operation.name} and {previous.name} share qubit "
                        f"{resource_id} but no dependency path orders them. "
                        f"The operation list is missing program-order wiring "
                        f"(run it through _wire_circuit / a frontend)")
                last_holder[key] = operation.id


def _validate_protocol_part(name: str, part, protocol) -> None:
    """Reject a malformed supplied part before an event can invoke it.

    Runtime-checkable protocols verify that required attributes exist.
    Binding a representative call to each declared method additionally catches
    duck-typed methods whose signatures cannot accept the runtime contract.
    """
    if part is None:
        return
    if not isinstance(part, protocol):
        raise TypeError(
            f"{name} must implement {protocol.__name__}; required attributes "
            f"are missing or not callable")
    method_names = [
        method_name
        for method_name, required_method in protocol.__dict__.items()
        if not method_name.startswith("_") and callable(required_method)
    ]
    _validate_method_signatures(name, part, protocol, method_names)


def _validate_protocol_methods(
    name: str, part, protocol, method_names: list[str],
) -> None:
    """Validate the subset of a protocol selected by configuration."""
    if part is None:
        return
    for method_name in method_names:
        method = getattr(part, method_name, None)
        if not callable(method):
            raise TypeError(
                f"{name} must implement {protocol.__name__}; required method "
                f"{method_name} is missing or not callable")
    _validate_method_signatures(name, part, protocol, method_names)


def _validate_method_signatures(
    name: str, part, protocol, method_names: list[str],
) -> None:
    for method_name in method_names:
        method = getattr(part, method_name)
        required_method = getattr(protocol, method_name)
        required = inspect.signature(required_method)
        positional = []
        keywords = {}
        for parameter in list(required.parameters.values())[1:]:
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                positional.append(object())
            elif parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                positional.append(object())
            elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                keywords[parameter.name] = object()
        try:
            inspect.signature(method).bind(*positional, **keywords)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"{name} does not satisfy {protocol.__name__}: "
                f"{method_name} has an incompatible signature ({error})"
            ) from error


def _validate_callable_arity(name: str, factory, arity: int) -> None:
    if factory is None:
        return
    if not callable(factory):
        raise TypeError(f"{name} must be callable")
    try:
        inspect.signature(factory).bind(*[object()] * arity)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{name} must accept {arity} positional argument"
            f"{'s' if arity != 1 else ''} ({error})"
        ) from error


def _make_infinite(engine):
    from .factories import InfiniteFactory
    return InfiniteFactory(engine)


class ClusterFacade:
    """The 'cluster' read surface chip/factory/metrics code expects,
    backed by the new window_manager + pool."""

    def __init__(self, window_manager, pool, layout, planner, strategy,
                 decode_plan_ops):
        self.window_manager = window_manager
        self.pool = pool
        self.layout = layout
        self.planner = planner
        self.strategy = strategy
        self._decode_plan_ops = decode_plan_ops
        self._registered_ops: list = []

    # chip-side surface
    def register_op(self, op) -> None:
        self.window_manager.register_op(op)
        self._registered_ops.append(op)

    def prepare(self, ops) -> None:
        """Compile-time plan handoff: register the planned ops and load the
        window plan. Costs zero ticks."""
        planned = self._decode_plan_ops if self._decode_plan_ops is not None \
            else list(ops)
        for op in planned:
            self.register_op(op)
        self.build_windows()

    def build_windows(self) -> None:
        if self.window_manager._windows_built:
            return
        planned = self._decode_plan_ops if self._decode_plan_ops is not None \
            else self._registered_ops
        plan = self.planner.plan(list(planned))
        check = getattr(self.strategy, "check_window_size", None)
        if check is not None:
            check(plan.summary.get("commit", self.window_manager.commit),
                  plan.summary.get("buffer", self.window_manager.buffer))
        self.window_manager.load_execution_plan(plan)

    def rounds_for(self, op) -> int:
        return self.window_manager.rounds_for(op)

    def prepend_idle_rounds(self, op_id, n) -> None:
        self.window_manager.prepend_idle_rounds(op_id, n)

    def on_memory_round(self, op_id) -> None:
        self.window_manager.on_memory_round(op_id)

    def on_syndrome_arrival(self, payload) -> None:
        self.window_manager.on_syndrome_arrival(payload)

    def close_stream_boundary(self, stream_id, n) -> None:
        self.window_manager.close_stream_boundary(stream_id, n)

    def seal_stream(self, stream_id, n) -> None:
        self.window_manager.seal_stream(stream_id, n)

    def has_dynamic_stream(self, stream_id) -> bool:
        return self.window_manager.has_dynamic_stream(stream_id)

    def committed_stream_round_count(self, stream_id) -> int:
        return self.window_manager.committed_stream_round_count(stream_id)

    def submit_decode(self, *args, **kwargs) -> None:
        self.pool.submit_decode(*args, **kwargs)

    # metrics / summary surface (old DecoderCluster pass-throughs)
    @property
    def code(self):
        return self.window_manager.code

    @property
    def links(self):
        return self.window_manager.links

    @property
    def scheme(self):
        return self.window_manager.scheme

    @property
    def store(self):
        return self.window_manager.store

    @property
    def on_workload_complete(self):
        return self.window_manager.on_workload_complete

    @on_workload_complete.setter
    def on_workload_complete(self, sink):
        self.window_manager.on_workload_complete = sink

    @property
    def window_count(self):
        return self.window_manager.window_count

    @property
    def op_windows(self):
        return self.window_manager.op_windows

    @property
    def commit(self):
        return self.window_manager.commit

    @property
    def buffer(self):
        return self.window_manager.buffer

    @property
    def ops(self):
        return self.window_manager.ops

    @property
    def rounds_arrived(self):
        return self.window_manager.rounds_arrived

    @property
    def memory_rounds(self):
        return self.window_manager.memory_rounds

    @property
    def op_results(self):
        return self.window_manager.op_results

    @property
    def op_strong_commit_time(self):
        return self.window_manager.op_strong_commit_time

    @property
    def total_windows(self):
        return self.window_manager.total_windows

    @property
    def committed_windows(self):
        return self.window_manager.committed_windows

    @property
    def peak_payloads(self):
        return self.window_manager.peak_payloads

    @property
    def payloads_held(self):
        return self.window_manager.payloads_held

    @property
    def windows(self):
        return self.window_manager.windows

    @property
    def memory_rounds_total(self):
        return self.window_manager.memory_rounds_total

    @property
    def unit_totals(self):
        return self.pool.unit_totals

    @property
    def pool_free(self):
        return self.pool.pool_free

    @property
    def free_units(self):
        return self.pool.free_units

    @property
    def num_units(self):
        return self.pool.num_units

    @property
    def ready(self):
        return self.pool.ready

    @property
    def pool_ready(self):
        return self.pool.pool_ready

    @property
    def queue_log(self):
        return self.pool.queue_log

    @property
    def strong_needed(self):
        return self.pool.strong_needed

    @property
    def strong_running_rounds(self):
        return self.pool.strong_running_rounds

    @property
    def strong_cancelled(self):
        return self.pool.strong_cancelled


@dataclass
class World:
    """A fully-wired simulator, ready for simulate() to drive."""

    engine: Any
    ops: list
    window_manager: Any
    pool: Any
    gate: Any
    orchestrator: Any
    factory: Any
    controller: Any
    cluster: Any
    planning: ResolvedPlanningParts


def simulate(run: RunSpec, verbose: bool = False) -> dict:
    """Build the world from a RunSpec, run it, and return the results."""
    world = run.build(verbose=verbose)
    world.gate.load(world.ops)
    world.engine.run()
    world.pool.check_decode_work_settled()
    return {
        "engine": world.engine,
        "cluster": world.cluster,
        "factory": world.factory,
        "chip": world.gate,
        "orchestrator": world.orchestrator,
        "controller": world.controller,
        "chip_done": world.gate.last_finish_time,
        "fully_done": world.engine.now,
        "metrics": world.engine.metric_results(),
    }
