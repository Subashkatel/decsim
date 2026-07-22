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
from typing import Any, Callable, Optional

from .message import Operation
from .pauli_frame import PauliFrame
from .config import TimingConfig, us

FEEDBACK_BOUNDARY_MODES = ("trailing_buffer", "measurement_closed")


@dataclass
class RunSpec:
    """Typed simulator configuration; every knob is a part object or a scalar."""

    # workload (exactly one of ops/frontend)
    ops: Optional[list] = None
    frontend: Optional[Any] = None
    decode_ops: Optional[list] = None
    dynamic_streams: Optional[list] = None

    # code / layout
    code: Optional[Any] = None
    layout: Optional[Any] = None
    d: int = 3

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
    boundary_policy: Optional[Any] = None     # default Eager (faithful)
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
    factory: Optional[Any] = None
    make_factory: Optional[Callable] = None   # (engine, cluster) -> factory
    make_metrics: Optional[Callable] = None   # (engine, cluster, gate, factory)
    seed: int = 0

    # ------------------------------------------------------------- validate

    def validate(self) -> None:
        """Cross-part validation before any build."""
        if (self.ops is None) == (self.frontend is None):
            raise ValueError("provide exactly one of ops= or frontend=")
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
        code = self._resolve_code()
        scheme = self.scheme
        if (scheme is not None and hasattr(scheme, "validate_buffer")
                and getattr(code, "buffer_rounds_override", None) is None):
            # an explicit buffer override is the expert escape hatch
            # (e.g. measurement-closed streams need no trailing buffer)
            scheme.validate_buffer(code)
        rounds = self.rounds_policy
        if rounds is not None:
            probe = Operation(-1, "probe", (0,))
            value = rounds.rounds_for(probe, code)
            if value < 1:
                raise ValueError(f"rounds_policy returned {value} (< 1)")
        if self.decode_ops and self.dynamic_streams:
            static_ids = {op.id for op in self.decode_ops}
            dyn_ids = {op.id for op in self.dynamic_streams}
            overlap = static_ids & dyn_ids
            if overlap:
                raise ValueError(f"ops {sorted(overlap)} appear in both "
                                 f"decode_ops and dynamic_streams (a stream "
                                 f"is planned statically OR dynamically)")
        for part in (self.strategy, self.decoder, self.factory):
            if part is not None and hasattr(part, "validate"):
                part.validate(self)

    def _resolve_code(self):
        if self.code is not None:
            return self.code
        if self.layout is not None:
            return self.layout.codes()[0]
        from .codes import SurfaceCodeModel
        return SurfaceCodeModel(d=self.d)

    # ---------------------------------------------------------------- build

    def build(self, verbose: bool = False) -> "World":
        """Construct and wire every component in the canonical order."""
        self.validate()
        from .policies import Eager
        from .codes import SurfaceCodeModel  # noqa: F401 (via _resolve_code)
        from .decoders import CodeRouter
        from .engine import Engine
        from .orchestrators import ExecutionOrchestrator
        from .policies import Ignore
        from .layouts import UniformLayout
        from .payload_store import PayloadStore
        from .decoder_manager import StrategyServicesImpl, DecoderManager
        from .chip import Chip
        from .planner import WindowPlanner, GateRounds
        from .schedulers import EnqueueTimeDeadline, FifoScheduler
        from .schemes import SlidingWindowScheme
        from .devices import ClockedDevice, TimingOnlyDevice
        from .switching import Baseline
        from .controllers import ModularController, LinkModel
        from .window_manager import WindowManager

        engine = Engine(verbose=verbose)
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

        code = self._resolve_code()
        layout = self.layout if self.layout is not None else UniformLayout(code)
        _validate_program_order(ops, layout)
        scheme = self.scheme if self.scheme is not None else SlidingWindowScheme()
        rounds_policy = self.rounds_policy if self.rounds_policy is not None \
            else GateRounds()
        strategy = self.strategy if self.strategy is not None else Baseline()
        scheduler = self.scheduler if self.scheduler is not None \
            else FifoScheduler()
        deadline_policy = self.deadline_policy if self.deadline_policy is not None \
            else EnqueueTimeDeadline()
        boundary_policy = self.boundary_policy if self.boundary_policy is not None \
            else Eager()
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
        # the whole fabric shares the controller's LinkModel: the window
        # manager's dd/do hops ride the same links a custom controller set
        links = getattr(controller, "links", None) or LinkModel.from_timing(self.timing)

        store = PayloadStore(memory_model=self.memory_model) \
            if self.memory_model is not None else None
        window_manager = WindowManager(
            engine, scheme=scheme, layout=layout, rounds_policy=rounds_policy,
            code=code, deadline_policy=deadline_policy, links=links,
            orchestrator=orchestrator, boundary_policy=boundary_policy,
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

        planner = self.planner if self.planner is not None \
            else WindowPlanner(scheme, layout, rounds_policy)
        cluster = ClusterFacade(window_manager, pool, layout, planner, strategy,
                                self._decode_plan_operations(ops))

        factory = self.factory
        if factory is None:
            factory = self.make_factory(engine, cluster) \
                if self.make_factory is not None else _make_infinite(engine)
        source = ClockedDevice(engine, device, controller, window_manager,
                               window_manager.rounds_for)
        round_us = self.round_us if self.round_us is not None \
            else self.timing.round_us
        gate = Chip(
            engine, source=source, controller=controller, cluster=cluster,
            factory=factory, round_ticks=us(round_us),
            code_distance=code.distance, idle_policy=idle_policy,
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
            window_manager.register_dynamic_stream(stream, code)
        if self.make_metrics is not None:
            for metric in self.make_metrics(engine, cluster, gate, factory):
                engine.add_metric(metric)

        return World(engine=engine, ops=ops, window_manager=window_manager, pool=pool,
                     gate=gate, orchestrator=orchestrator, factory=factory,
                     controller=controller, cluster=cluster, code=code)

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
    code: Any


def simulate(run: RunSpec, verbose: bool = False) -> dict:
    """Build the world from a RunSpec, run it, and return the results."""
    world = run.build(verbose=verbose)
    world.gate.load(world.ops)
    world.engine.run()
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
