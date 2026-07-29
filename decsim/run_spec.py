"""RunSpec: the typed run configuration and composition root.

The one core module allowed to import part implementations — that is its
job: every field picks one implementation per seam, and ``RunSpec.build()``
wires and executes one atomic ``CompletedRun`` in a fixed order (the frozen
timing goldens depend on that order). Experiment code still never appears
here; experiments hand a RunSpec pre-built objects. ``simulate(run)`` delegates
to the same completed boundary.

Defaults: sliding window + Baseline strategy + Eager boundaries +
GateRounds + InfiniteFactory.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import math
from numbers import Integral
from typing import Any, Callable, Optional, TYPE_CHECKING

from .message import (
    IntrinsicMeasurement,
    Operation,
    OperationPlanningView,
    ResolvedCodeGeometry,
    ResolvedCodeSpatialProfile,
    ResolvedOperationPlanning,
    ResolvedPatchPlanning,
    RunSeedChild,
    RunSeedPathSegment,
    RunSeedReservation,
    is_stable_identity,
    is_stable_string,
    same_stable_identity,
    stable_identity_bytes,
)
from .config import TimingConfig, us

if TYPE_CHECKING:
    from .planner import _ResolvedExecutionPlanSpec
    from .protocols import (
        CodeModel,
        DecodingScheme,
        LayoutModel,
        RoundsPolicy,
    )

FEEDBACK_BOUNDARY_MODES = ("trailing_buffer", "measurement_closed")
RUN_SEED_NAMESPACE = b"decsim.run-seed.v1"
def _derive_run_component_seed(
    root_seed: int,
    component_path: tuple[RunSeedPathSegment, ...],
) -> int:
    """Derive one stable unsigned-64-bit component seed."""
    encoded_path = b"".join(
        segment.canonical_bytes()
        for segment in component_path
    )
    digest = hashlib.blake2b(
        RUN_SEED_NAMESPACE
        + root_seed.to_bytes(8, "big")
        + encoded_path,
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big")


@dataclass(frozen=True)
class ResolvedPlanningParts:
    """Exact planning/runtime collaborators selected for one build."""

    code: "CodeModel"
    layout: "LayoutModel"
    scheme: "DecodingScheme"
    rounds_policy: "RoundsPolicy"


@dataclass
class _RunOwnedWorkload:
    """Private executable operations and their circuit-free planning views."""

    executable_operations: tuple[Operation, ...]
    static_decode_operations: tuple[Operation, ...]
    dynamic_stream_operations: tuple[Operation, ...]
    planning_view_by_operation_id: dict[int, OperationPlanningView]
    declared_view_by_operation_id: dict[int, OperationPlanningView]
    memberships_by_operation_id: dict[int, tuple[tuple[str, int], ...]]
    source_circuit_by_operation_id: dict[int, Any]

    def planning_views(self, operations) -> tuple[OperationPlanningView, ...]:
        return tuple(
            self.planning_view_by_operation_id[operation.id]
            for operation in operations
        )


@dataclass(frozen=True)
class _RunSeedPlanEntry:
    """One canonical stochastic leaf in a frozen run component graph."""

    component_path: tuple[RunSeedPathSegment, ...]
    component: Any
    derived_seed: Optional[int]


@dataclass(frozen=True)
class LogicalOperationResult:
    """One operation's logical-output disposition."""

    operation_id: int
    result_status: str
    logical_observables: Optional[tuple[int, ...]]
    stream_offset: Optional[int]


@dataclass(frozen=True)
class _ResolvedMetricBinding:
    metric: Any
    name: str


@dataclass(frozen=True)
class MetricResultRecord:
    """One metric snapshot in declared observation order."""

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
        return {
            record.operation_id: record.logical_observables
            for record in self.operation_results
            if record.result_status == "logical_observables"
        }

    def stream_offsets(self) -> dict[int, Optional[int]]:
        return {
            record.operation_id: record.stream_offset
            for record in self.operation_results
        }

    def metric_values(self) -> dict[str, Any]:
        return {
            record.name: copy.deepcopy(record.value)
            for record in self.metric_results
        }


@dataclass(frozen=True)
class _ResolvedCodeCadencePlan:
    code_geometry: ResolvedCodeGeometry
    operations: tuple[ResolvedOperationPlanning, ...]
    patches: tuple[ResolvedPatchPlanning, ...]
    round_ticks: int


@dataclass
class RunSpec:
    """Typed simulator configuration with one owner per planning choice.

    Supply at most one of ``d``, ``code``, or ``layout``; omitting all three
    selects distance 3. Custom magic-state factories are constructed through
    ``make_factory(engine, cluster)``.
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

    # control loop
    boundary_policy: Optional[Any] = None     # default Eager (speculative)
    window_interaction: Optional[Any] = None  # default defect-mask interaction
    idle_policy: Optional[Any] = None         # default Ignore
    max_idle_rounds: Optional[int] = None
    gates_start_on_round_boundaries: bool = False
    feedback_boundary_mode: str = "trailing_buffer"
    # environment
    timing: TimingConfig = field(default_factory=TimingConfig)
    round_us: Optional[float] = None          # overrides timing.round_us
    links: Optional[Any] = None               # reusable LinkModelConfig
    device: Optional[Any] = None              # default TimingOnlyDevice
    memory_model: Optional[Any] = None        # port 18; default unbounded
    make_controller: Optional[Callable] = None  # (engine, links) -> Controller
    make_factory: Optional[Callable] = None   # (engine, cluster) -> factory
    make_metrics: Optional[Callable] = None   # (engine, cluster, gate, factory)
    make_orchestrator: Optional[Callable] = None  # (engine) -> Orchestrator
    seed: Optional[Integral] = 0
    _build_state: str = field(
        default="unstarted",
        init=False,
        repr=False,
    )

    # ------------------------------------------------------------- validate

    def validate(self) -> None:
        """Cross-part validation before any build."""
        planning = self._validate_configuration()
        workload = _snapshot_run_workload(
            list(self.ops or []),
            list(self.decode_ops or []),
            list(self.dynamic_streams or []),
            self.feedback_boundary_mode,
        )
        planning_operations = workload.planning_views(
            workload.executable_operations
            + workload.static_decode_operations
            + workload.dynamic_stream_operations
        )
        from .switching import Baseline
        strategy = self.strategy if self.strategy is not None else Baseline()
        self._validate_layout_selection(
            planning,
            planning_operations,
            strategy,
        )

    def _validate_configuration(self) -> ResolvedPlanningParts:
        """Validate configuration-only state and resolve planning once."""
        self._validated_root_seed()
        if (self.ops is None) == (self.frontend is None):
            raise ValueError("provide exactly one of ops= or frontend=")
        self._validate_supplied_parts()
        from .policies import Eager
        from .schemes import SlidingWindowScheme
        from .switching import Baseline

        selected_scheme = (
            self.scheme if self.scheme is not None else SlidingWindowScheme()
        )
        selected_strategy = (
            self.strategy if self.strategy is not None else Baseline()
        )
        selected_boundary_policy = (
            self.boundary_policy
            if self.boundary_policy is not None
            else Eager()
        )
        if self.dynamic_streams and (
            type(selected_scheme) is not SlidingWindowScheme
        ):
            raise ValueError(
                "dynamic streams require the exact shipped serial "
                "SlidingWindowScheme"
            )
        selected_strategy.validate_declared_run(
            scheme=selected_scheme,
            boundary_policy=selected_boundary_policy,
            has_dynamic_streams=bool(self.dynamic_streams),
            static_decode_plan_selected=self.decode_ops is not None,
            has_frontend=self.frontend is not None,
        )
        auxiliary_ops = list(self.decode_ops or []) + list(self.dynamic_streams or [])
        if self.ops is not None:
            _validate_run_workload_identity(
                list(self.ops),
                list(self.decode_ops or []),
                list(self.dynamic_streams or []),
            )
            from .planner import _validate_operation_graph
            _validate_operation_graph(
                self.ops, validate_blockers=True,
                external_blocker_ids=(operation.id for operation in auxiliary_ops))
            selected_strategy.validate_operations(tuple(
                _planning_view_from_operation(operation)
                for operation in list(self.ops) + auxiliary_ops
            ))
        self._validate_operation_feedback_contracts(
            list(self.ops or []) + auxiliary_ops,
        )
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

    def _validated_root_seed(self) -> Optional[int]:
        """Return the run root under the unsigned 64-bit seed contract."""
        if self.seed is None:
            return None
        if type(self.seed) is bool or not isinstance(self.seed, Integral):
            raise TypeError(
                "seed must be a 64-bit unsigned integer or None; "
                f"got {self.seed!r}"
            )
        root_seed = int(self.seed)
        if not 0 <= root_seed < (1 << 64):
            raise ValueError(
                "seed must be in [0, 2**64); "
                f"got {self.seed!r}"
            )
        return root_seed

    def _validate_supplied_parts(self) -> None:
        """Reject configuration combinations the runtime cannot interpret."""
        from .links import LinkModelConfig

        if self.links is not None and type(self.links) is not LinkModelConfig:
            raise TypeError("links must be an exact LinkModelConfig")

        if self.router is not None and (
            self.decoder is not None or self.decoders
        ):
            raise ValueError(
                "RunSpec.router is exclusive with decoder and decoders"
            )
        if self.router is not None:
            if type(getattr(self.router, "needs_hyperedges", None)) is not bool:
                raise TypeError(
                    "router.needs_hyperedges must be an exact bool"
                )
        if self.device is not None:
            circuit_scope = getattr(self.device, "operation_circuit_scope", None)
            if circuit_scope not in ("none", "per_operation"):
                raise ValueError(
                    "device operation_circuit_scope must be 'none' or "
                    f"'per_operation'; got {circuit_scope!r}"
                )

    def _resolve_planning_parts(self) -> ResolvedPlanningParts:
        from .codes import SurfaceCodeModel
        from .layouts import UniformLayout
        from .planner import GateRounds
        from .schemes import SlidingWindowScheme

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
        return ResolvedPlanningParts(
            code=code,
            layout=layout,
            scheme=scheme,
            rounds_policy=rounds_policy,
        )

    def _validate_resolved_planning(
        self,
        planning: ResolvedPlanningParts,
    ) -> None:
        declared_code = _single_layout_code(
            planning.layout,
            "resolved layout",
        )
        if declared_code is not planning.code:
            raise ValueError(
                "resolved layout declared a code different from the resolved "
                "planning/runtime code")
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
            part.validate(self, planning)

    def _validate_layout_selection(
        self,
        planning: ResolvedPlanningParts,
        operations,
        strategy,
    ) -> _ResolvedCodeCadencePlan:
        code_round_us = planning.code.round_period_us()
        if code_round_us is not None:
            round_us = code_round_us
        elif self.round_us is not None:
            round_us = self.round_us
        else:
            round_us = self.timing.round_us
        if (
            type(round_us) not in (int, float)
            or not math.isfinite(round_us)
            or round_us <= 0
        ):
            raise ValueError(
                "resolved round_us must be a positive finite built-in number"
            )
        round_ticks = us(round_us)
        if type(round_ticks) is not int or round_ticks < 1:
            raise ValueError(
                "resolved round cadence must be at least one tick"
            )

        code_name = planning.code.name
        distance = planning.code.distance
        commit_round_count = planning.code.commit_rounds()
        buffer_round_count = planning.code.buffer_rounds()
        buffering_floor = planning.code.buffering_floor()
        buffer_floor_override_active = (
            planning.code.buffer_floor_override_active()
        )
        if type(buffering_floor) is not tuple or len(buffering_floor) != 2:
            raise TypeError(
                "CodeModel.buffering_floor() must return an exact pair"
            )
        minimum_leading, minimum_trailing = buffering_floor
        if type(buffer_floor_override_active) is not bool:
            raise TypeError(
                "CodeModel.buffer_floor_override_active() must return an "
                "exact bool"
            )

        patch_counts = {1}
        patch_count_by_operation_id = {}
        for operation in operations:
            patch_count = max(
                1,
                len(operation.patches)
                if operation.patches
                else len(operation.qubits),
            )
            patch_counts.add(patch_count)
            patch_count_by_operation_id[operation.id] = patch_count
        spatial_entries = []
        for patch_count in sorted(patch_counts):
            node_count = planning.code.spatial_nodes(patch_count)
            if type(node_count) is not int or node_count < 1:
                raise TypeError(
                    "CodeModel.spatial_nodes() must return an exact positive "
                    "int"
                )
            spatial_entries.append((patch_count, node_count))
        spatial_profile = ResolvedCodeSpatialProfile(tuple(spatial_entries))
        code_geometry = ResolvedCodeGeometry(
            code_name=code_name,
            distance=distance,
            commit_round_count=commit_round_count,
            buffer_round_count=buffer_round_count,
            minimum_leading_buffer_round_count=minimum_leading,
            minimum_trailing_buffer_round_count=minimum_trailing,
            one_patch_spatial_node_count=spatial_profile.for_patch_count(1),
            buffer_floor_override_active=buffer_floor_override_active,
        )
        planning.scheme.validate_buffer(code_geometry)
        strategy.validate_code_geometry(code_geometry)

        resolved_operations = []
        patch_records_by_bytes = {}
        for operation in operations:
            selected_code = planning.layout.code_for_op(operation)
            if selected_code is not planning.code:
                raise ValueError(
                    f"layout {planning.layout!r} operation {operation.id} "
                    f"selected {selected_code!r}, but resolved "
                    f"planning/runtime code is {planning.code!r}")
            round_count = planning.rounds_policy.rounds_for(
                operation,
                planning.code,
            )
            if type(round_count) is not int or round_count < 1:
                raise TypeError(
                    f"resolved rounds for operation {operation.id} must be "
                    "a positive exact int"
                )
            base_spatial_node_count = spatial_profile.for_patch_count(
                patch_count_by_operation_id[operation.id]
            )
            spatial_node_count = planning.layout.spatial_nodes_for(
                operation,
                base_spatial_node_count=base_spatial_node_count,
            )
            if type(spatial_node_count) is not int or spatial_node_count < 1:
                raise TypeError(
                    "LayoutModel.spatial_nodes_for() must return an exact "
                    "positive int"
                )
            resolved_operations.append(
                ResolvedOperationPlanning(
                    operation_id=operation.id,
                    code_geometry=code_geometry,
                    round_count=round_count,
                    round_ticks=round_ticks,
                    spatial_node_count=spatial_node_count,
                )
            )

            patch_ids = operation.patches
            if not patch_ids:
                patch_ids = operation.qubits
            if not patch_ids:
                patch_ids = (0,)
            for patch_id in patch_ids:
                patch_records_by_bytes.setdefault(
                    stable_identity_bytes(patch_id),
                    patch_id,
                )

        resolved_patches = []
        for patch_id in patch_records_by_bytes.values():
            selected_code = planning.layout.code_for_patch(patch_id)
            if selected_code is not planning.code:
                raise ValueError(
                    f"layout {planning.layout!r} patch {patch_id!r} selected "
                    f"{selected_code!r}, but resolved planning/runtime code "
                    f"is {planning.code!r}"
                )
            spatial_node_count = planning.layout.patch_spatial_nodes_for(
                patch_id,
                base_spatial_node_count=(
                    spatial_profile.for_patch_count(1)
                ),
            )
            if type(spatial_node_count) is not int or spatial_node_count < 1:
                raise TypeError(
                    "LayoutModel.patch_spatial_nodes_for() must return an "
                    "exact positive int"
                )
            resolved_patches.append(
                ResolvedPatchPlanning(
                    patch_identity=patch_id,
                    code_geometry=code_geometry,
                    round_ticks=round_ticks,
                    spatial_node_count=spatial_node_count,
                )
            )

        return _ResolvedCodeCadencePlan(
            code_geometry=code_geometry,
            operations=tuple(resolved_operations),
            patches=tuple(resolved_patches),
            round_ticks=round_ticks,
        )

    # ---------------------------------------------------------------- build

    def build(self, verbose: bool = False) -> "CompletedRun":
        """Construct, execute, and freeze one complete primary run."""
        if self._build_state != "unstarted":
            raise RuntimeError(
                f"RunSpec build is already {self._build_state}; "
                "construct a fresh RunSpec and runtime graph"
            )
        self._build_state = "committing"
        from .engine import Engine
        engine = Engine(verbose=verbose, construction_guarded=True)
        try:
            completed_run = self._build_once(engine, verbose=verbose)
        except BaseException as error:
            engine._invalidate(error)
            self._build_state = "invalid"
            raise
        self._build_state = "complete"
        return completed_run

    def _build_once(self, engine, verbose: bool = False) -> "CompletedRun":
        """Construct and wire every component in the canonical order."""
        planning = self._validate_configuration()
        from .policies import Eager
        from .decoders import CodeRouter
        from .orchestrators import ExecutionOrchestrator
        from .policies import Ignore
        from .payload_store import PayloadStore
        from .decoder_manager import StrategyServicesImpl, DecoderManager
        from .chip import Chip
        from .schedulers import EnqueueTimeDeadline, FifoScheduler
        from .devices import ClockedDevice, TimingOnlyDevice
        from .switching import Baseline
        from .controllers import ModularController
        from .links import LinkModelConfig
        from .window_manager import WindowManager
        from .window_interactions import DefaultWindowInteraction

        source_operations = (
            self.frontend.build()
            if self.frontend is not None
            else self.ops
        )
        if self.frontend is not None:
            _validate_run_workload_identity(
                list(source_operations),
                list(self.decode_ops or []),
                list(self.dynamic_streams or []),
            )
            from .planner import _validate_operation_graph
            auxiliary_ids = (operation.id for operation in
                             list(self.decode_ops or [])
                             + list(self.dynamic_streams or []))
            _validate_operation_graph(
                source_operations, validate_blockers=True,
                external_blocker_ids=auxiliary_ids)
        workload = _snapshot_run_workload(
            list(source_operations),
            list(self.decode_ops or []),
            list(self.dynamic_streams or []),
            self.feedback_boundary_mode,
        )
        ops = list(workload.executable_operations)
        decode_operations = list(workload.static_decode_operations)
        dynamic_streams = list(workload.dynamic_stream_operations)

        all_operations = list(ops)
        all_operations.extend(decode_operations)
        all_operations.extend(dynamic_streams)
        planning_operations = workload.planning_views(all_operations)
        self._validate_operation_feedback_contracts(all_operations)
        strategy = self.strategy if self.strategy is not None else Baseline()
        if self.frontend is not None:
            strategy.validate_operations(tuple(planning_operations))
        code_cadence_plan = self._validate_layout_selection(
            planning,
            planning_operations,
            strategy,
        )
        resource_claims_by_operation_id = _validate_program_order(
            workload.planning_views(ops),
            planning.layout,
        )

        decode_plan_operations = self._decode_plan_operations(
            ops,
            decode_operations,
            dynamic_streams,
            static_decode_selected=self.decode_ops is not None,
        )
        planned_operations = (
            ops
            if decode_plan_operations is None
            else decode_plan_operations
        )
        planned_views = workload.planning_views(planned_operations)
        resolved_operation_by_id = {
            operation.operation_id: operation
            for operation in code_cadence_plan.operations
        }
        planned_resolved_operations = tuple(
            resolved_operation_by_id[operation.id]
            for operation in planned_views
        )
        operation_window_plans = tuple(
            planning.scheme.plan_operation(
                operation.operation_id,
                operation.round_count,
                commit_round_count=(
                    operation.code_geometry.commit_round_count
                ),
                buffer_round_count=(
                    operation.code_geometry.buffer_round_count
                ),
            )
            for operation in planned_resolved_operations
        )
        from .planner import _materialize_execution_plan
        execution_plan_spec = _materialize_execution_plan(
            tuple(planned_views),
            planned_resolved_operations,
            operation_window_plans,
        )
        resolved_rounds_by_operation_id = {
            operation.operation_id: operation.round_count
            for operation in code_cadence_plan.operations
        }

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
        from .devices import SyndromeBitDevice
        if (
            type(device) is SyndromeBitDevice
            and device.code is not planning.code
        ):
            raise ValueError(
                "SyndromeBitDevice.code must be the exact resolved run code"
            )
        _install_private_execution_circuits(workload, device)
        if self.router is None and self.decoder is None:
            raise ValueError(
                "RunSpec.decoder is required when router is omitted"
            )
        router = (
            self.router
            if self.router is not None
            else CodeRouter(
                default=self.decoder,
                by_code=dict(self.decoders),
            )
        )
        orchestrator = (
            self.make_orchestrator(engine)
            if self.make_orchestrator is not None
            else ExecutionOrchestrator(engine)
        )

        link_config = (
            self.links
            if self.links is not None
            else LinkModelConfig.reference_fixed_latency_profile()
        )
        links = link_config.resolve()
        controller = self.make_controller(engine, links) \
            if self.make_controller is not None \
            else ModularController(
                engine,
                links=links,
                t_pack=self.timing.ticks("t_pack"),
            )
        if orchestrator.engine is not engine:
            raise ValueError(
                f"{type(orchestrator).__name__} uses a different engine from "
                "the RunSpec build"
            )
        if controller.links is not links:
            raise ValueError(
                "controller must retain the exact RunSpec-resolved link fabric"
            )
        # One resolved fabric is shared by every reaction-path participant.

        store = PayloadStore(memory_model=self.memory_model)
        window_manager = WindowManager(
            engine,
            scheme=planning.scheme,
            code_geometry=code_cadence_plan.code_geometry,
            resolved_operations=code_cadence_plan.operations,
            resolved_patches=code_cadence_plan.patches,
            deadline_policy=deadline_policy, links=links,
            orchestrator=orchestrator, boundary_policy=boundary_policy,
            window_interaction=window_interaction,
            planning_view_by_operation_id=(
                workload.planning_view_by_operation_id
            ),
            feedback_boundary_mode=self.feedback_boundary_mode,
            syndrome_source=device, store=store,
            switching_active=hasattr(strategy, "keep_weak_result"))
        pool = DecoderManager(
            engine, router=router, scheduler=scheduler,
            unit_pools=self.unit_pools,
            num_units=self.num_units if self.num_units is not None else 1,
            bulk_strong=getattr(strategy, "bulk_strong", False))
        services = StrategyServicesImpl(engine, window_manager, pool)
        window_manager.strategy = strategy
        window_manager.services = services
        window_manager.submit_fn = pool.enqueue
        window_manager.needs_hyperedges = router.needs_hyperedges
        pool.strategy = strategy
        pool.services = services
        pool.on_window_decoded = window_manager.on_decode_done
        pool.on_strong_window_decoded = window_manager.on_strong_decode_done

        cluster = ClusterFacade(window_manager, pool)

        factory = self.make_factory(engine, cluster) \
            if self.make_factory is not None else _make_infinite(engine)
        if factory.engine is not engine:
            raise ValueError(
                f"{type(factory).__name__} uses a different engine from "
                "the RunSpec build")
        _validate_shipped_factory_decode_service(factory, cluster)
        source = ClockedDevice(
            engine,
            device,
            controller,
            cluster,
            {
                operation.operation_id: operation.round_count
                for operation in code_cadence_plan.operations
            },
        )
        gate = Chip(
            engine, source=source, controller=controller, cluster=cluster,
            factory=factory, round_ticks=code_cadence_plan.round_ticks,
            code_geometry=code_cadence_plan.code_geometry,
            resolved_operations=code_cadence_plan.operations,
            resolved_patches=code_cadence_plan.patches,
            idle_policy=idle_policy,
            resource_claims_by_operation_id=(
                resource_claims_by_operation_id
            ),
            max_idle_rounds=self.max_idle_rounds,
            gates_start_on_round_boundaries=self.gates_start_on_round_boundaries,
            frame=orchestrator.frame)

        metrics = []
        metric_bindings = ()
        if self.make_metrics is not None:
            metrics = self.make_metrics(engine, cluster, gate, factory)
            if type(metrics) is not list:
                raise TypeError("make_metrics must return a list")
            metric_names = set()
            for index, metric in enumerate(metrics):
                if not is_stable_string(metric.name) or not metric.name:
                    raise TypeError(
                        f"metric {index} name must be a nonempty Unicode "
                        "scalar string"
                    )
                if metric.name in metric_names:
                    raise ValueError(
                        f"duplicate metric name {metric.name!r}"
                    )
                metric_names.add(metric.name)
            metric_bindings = tuple(
                _ResolvedMetricBinding(
                    metric=metric,
                    name=metric.name,
                )
                for metric in metrics
            )

        seed_roots = self._run_seed_roots(
            frontend=self.frontend,
            planning=planning,
            device=device,
            router=router,
            factory=factory,
            strategy=strategy,
            scheduler=scheduler,
            deadline_policy=deadline_policy,
            boundary_policy=boundary_policy,
            window_interaction=window_interaction,
            idle_policy=idle_policy,
            orchestrator=orchestrator,
            controller=controller,
            metrics=metric_bindings,
            workload=workload,
        )
        seed_plan = _collect_seed_plan(
            seed_roots,
            self._validated_root_seed(),
        )
        _bind_run_seed_plan(seed_plan)

        try:
            orchestrator.connect(controller, gate.on_decision)
            window_manager.on_workload_complete = factory.shutdown
            for op in ops:
                if op.blocked_by is not None:
                    orchestrator.register_blocked_operation(
                        op.id,
                        op.blocked_by,
                    )
            for operation in planned_operations:
                cluster.register_op(operation)
            window_manager.load_execution_plan(
                execution_plan_spec.materialize()
            )
            for stream in dynamic_streams:
                window_manager._register_dynamic_stream(
                    stream,
                    resolved_operation_by_id[stream.id],
                )
            for binding in metric_bindings:
                engine.add_metric(binding.metric)
            gate._load(ops)
            engine._start_running()
            engine.run()
            pool.check_decode_work_settled()
            engine._begin_finalization()
            result = _capture_primary_run_result(
                engine=engine,
                gate=gate,
                window_manager=window_manager,
                operations=all_operations,
                metric_bindings=metric_bindings,
                links=links,
            )
            engine._complete()
        except BaseException as error:
            engine._invalidate(error)
            raise

        return CompletedRun(
            result=result,
            engine=engine,
            window_manager=window_manager,
            pool=pool,
            chip=gate,
            orchestrator=orchestrator,
            factory=factory,
            controller=controller,
            cluster=cluster,
            planning=planning,
        )

    def _run_seed_roots(
        self,
        *,
        frontend,
        planning,
        device,
        router,
        factory,
        strategy,
        scheduler,
        deadline_policy,
        boundary_policy,
        window_interaction,
        idle_policy,
        orchestrator,
        controller,
        metrics,
        workload,
    ):
        """Return the complete runtime root set under fixed semantic paths."""
        field_segment = lambda name: RunSeedPathSegment("field", name)
        roots = [
            ((field_segment("code"),), planning.code),
            ((field_segment("layout"),), planning.layout),
            ((field_segment("scheme"),), planning.scheme),
            ((field_segment("rounds_policy"),), planning.rounds_policy),
            ((field_segment("device"),), device),
            ((field_segment("decoder_router"),), router),
            ((field_segment("magic_state_factory"),), factory),
            ((field_segment("strategy"),), strategy),
            ((field_segment("scheduler"),), scheduler),
            ((field_segment("deadline_policy"),), deadline_policy),
            ((field_segment("boundary_policy"),), boundary_policy),
            ((field_segment("window_interaction"),), window_interaction),
            ((field_segment("idle_policy"),), idle_policy),
            ((field_segment("orchestrator"),), orchestrator),
            ((field_segment("controller"),), controller),
        ]
        if frontend is not None:
            roots.append(((field_segment("frontend"),), frontend))
        if self.memory_model is not None:
            roots.append(
                ((field_segment("memory_model"),), self.memory_model)
            )
        for binding in metrics:
            roots.append(
                (
                    (
                        field_segment("metrics"),
                        RunSeedPathSegment("string_key", binding.name),
                    ),
                    binding.metric,
                )
            )
        if device.operation_circuit_scope == "per_operation":
            seen_operation_ids = set()
            for operation in (
                workload.executable_operations
                + workload.static_decode_operations
                + workload.dynamic_stream_operations
            ):
                if (
                    operation.id in seen_operation_ids
                    or operation.circuit is None
                ):
                    continue
                seen_operation_ids.add(operation.id)
                roots.append(
                    (
                        (
                            field_segment("workload_circuits"),
                            RunSeedPathSegment(
                                "integer_key",
                                operation.id,
                            ),
                        ),
                        operation.circuit,
                    )
                )
        return tuple(roots)

    @staticmethod
    def _validate_operation_feedback_contracts(operations) -> None:
        for operation in operations:
            observable_index = operation.logical_observable_index
            if observable_index is not None:
                if type(observable_index) is not int:
                    raise TypeError(
                        f"operation {operation.id} "
                        "logical_observable_index must be an exact int")
                if observable_index < 0:
                    raise ValueError(
                        f"operation {operation.id} "
                        "logical_observable_index must be nonnegative")

            measurement = operation.intrinsic_measurement
            if measurement is None:
                continue
            if type(measurement) is not IntrinsicMeasurement:
                raise TypeError(
                    f"operation {operation.id} intrinsic_measurement must "
                    "be IntrinsicMeasurement")
            trajectory_id = (
                operation.stream_id
                if operation.stream_id is not None
                else operation.id
            )
            if not is_stable_identity(operation.id):
                raise TypeError(
                    f"operation {operation.id!r} with an intrinsic "
                    "measurement needs a stable operation id")
            if not is_stable_identity(trajectory_id):
                raise TypeError(
                    f"operation {operation.id} intrinsic trajectory identity "
                    "must be a stable built-in int, str, or recursive tuple")
            if not same_stable_identity(
                measurement.operation_id,
                operation.id,
            ):
                raise ValueError(
                    f"operation {operation.id} intrinsic_measurement "
                    f"operation_id does not match")
            if not same_stable_identity(
                measurement.trajectory_id,
                trajectory_id,
            ):
                raise ValueError(
                    f"operation {operation.id} intrinsic_measurement "
                    f"trajectory_id does not match")

    @staticmethod
    def _decode_plan_operations(
        ops,
        decode_operations,
        dynamic_streams,
        *,
        static_decode_selected,
    ):
        """Operations that receive compile-time decode windows (wiring parity)."""
        if static_decode_selected:
            return decode_operations
        if not dynamic_streams:
            return None
        dynamic_ids = {stream.id for stream in dynamic_streams}
        return [op for op in ops if op.stream_id not in dynamic_ids]


def _is_runtime_identity(value) -> bool:
    return is_stable_identity(value)


def _validate_run_workload_identity(
    executable_operations,
    static_decode_operations,
    dynamic_stream_operations,
) -> None:
    """Validate identities before Python mappings can collapse distinct keys."""
    collections = (
        ("ops", executable_operations),
        ("decode_ops", static_decode_operations),
        ("dynamic_streams", dynamic_stream_operations),
    )
    first_object_by_operation_id = {}
    memberships_by_object_id = {}

    for role, operations in collections:
        seen_in_role = set()
        for operation in operations:
            if type(operation) is not Operation:
                raise TypeError(
                    f"{role} entries must be exact Operation values"
                )
            operation_id = operation.id
            if type(operation_id) is not int:
                raise TypeError(
                    "operation id must be an exact built-in int, excluding "
                    f"bool; got {operation_id!r}"
                )
            if not is_stable_string(operation.name):
                raise TypeError(
                    f"operation name for id {operation_id} must be a Unicode "
                    "scalar string"
                )
            if operation_id in seen_in_role:
                raise ValueError(
                    f"operation id {operation_id} appears more than once "
                    f"in {role}"
                )
            seen_in_role.add(operation_id)

            prior = first_object_by_operation_id.get(operation_id)
            if prior is not None and prior is not operation:
                raise ValueError(
                    f"operation id {operation_id} belongs to distinct objects "
                    "across workload roles"
                )
            first_object_by_operation_id[operation_id] = operation

            memberships = memberships_by_object_id.setdefault(
                id(operation),
                [],
            )
            memberships.append(role)
            if "ops" in memberships and "dynamic_streams" in memberships:
                raise ValueError(
                    f"operation id {operation_id} cannot appear in both "
                    "ops and dynamic_streams"
                )
            if "decode_ops" in memberships and "dynamic_streams" in memberships:
                raise ValueError(
                    f"operation id {operation_id} cannot appear in both "
                    "decode_ops and dynamic_streams"
                )

            for reference_name, reference_ids in (
                ("predecessors", operation.predecessors),
            ):
                if type(reference_ids) is not tuple:
                    raise TypeError(
                        f"operation {operation_id} {reference_name} must "
                        "be a tuple"
                    )
                if any(type(reference) is not int for reference in reference_ids):
                    raise TypeError(
                        f"operation {operation_id} {reference_name} must "
                        "contain exact built-in int operation ids"
                    )
            if (
                operation.blocked_by is not None
                and type(operation.blocked_by) is not int
            ):
                raise TypeError(
                    f"operation {operation_id} blocked_by must be an exact "
                    "built-in int operation id or None"
                )

            for identity_field in ("qubits", "patches"):
                identities = getattr(operation, identity_field)
                if type(identities) is not tuple or not all(
                    _is_runtime_identity(identity)
                    for identity in identities
                ):
                    raise TypeError(
                        f"operation {operation_id} {identity_field} must "
                        "contain stable built-in int, str, or recursive tuple "
                        "identities with bool excluded"
                    )

            if (
                operation.stream_id is not None
                and type(operation.stream_id) is not int
            ):
                raise TypeError(
                    f"operation {operation_id} stream_id must be an exact "
                    "built-in int or None"
                )

    static_owner_by_id = {
        operation.id: operation
        for operation in static_decode_operations
    }
    dynamic_owner_by_id = {
        operation.id: operation
        for operation in dynamic_stream_operations
    }
    for operation in executable_operations:
        stream_id = operation.stream_id
        if stream_id is None:
            if (
                static_decode_operations
                and static_owner_by_id.get(operation.id) is not operation
            ):
                raise ValueError(
                    f"operation {operation.id} must name a declared static "
                    "stream owner or share static decode membership"
                )
            continue
        if dynamic_stream_operations:
            owner = dynamic_owner_by_id.get(stream_id)
        elif static_decode_operations:
            owner = static_owner_by_id.get(stream_id)
        else:
            owner = (
                operation
                if stream_id == operation.id
                else None
            )
        if owner is None:
            raise ValueError(
                f"operation {operation.id} stream_id {stream_id} does not "
                "name a declared stream owner"
            )


def _snapshot_run_workload(
    executable_operations,
    static_decode_operations,
    dynamic_stream_operations,
    feedback_boundary_mode: str,
) -> _RunOwnedWorkload:
    """Copy caller-owned workload state into one private run-owned snapshot."""
    clone_by_source_identity = {}
    source_circuit_by_operation_id = {}
    declared_view_by_operation_id = {}
    memberships_by_operation_id = {}
    for role, operations in (
        ("executable", executable_operations),
        ("static_decode", static_decode_operations),
        ("dynamic_stream", dynamic_stream_operations),
    ):
        for collection_index, operation in enumerate(operations):
            memberships_by_operation_id.setdefault(
                operation.id,
                [],
            ).append((role, collection_index))
            declared_view_by_operation_id.setdefault(
                operation.id,
                OperationPlanningView.from_operation(
                    operation,
                    default_feedback_boundary_mode=None,
                ),
            )

    def clone(operation: Operation) -> Operation:
        source_identity = id(operation)
        existing = clone_by_source_identity.get(source_identity)
        if existing is not None:
            return existing
        private_operation = Operation(
            id=operation.id,
            name=operation.name,
            qubits=tuple(operation.qubits),
            clifford=operation.clifford,
            circuit=None,
            consumes_magic_state=operation.consumes_magic_state,
            patches=tuple(operation.patches),
            predecessors=tuple(operation.predecessors),
            has_successor=operation.has_successor,
            stream_id=operation.stream_id,
            stream_offset=operation.stream_offset,
            blocked_by=operation.blocked_by,
            feedback_boundary_mode=(
                operation.feedback_boundary_mode
                if operation.feedback_boundary_mode is not None
                else feedback_boundary_mode
            ),
            requires_result_return_to_chip=(
                operation.requires_result_return_to_chip
            ),
            requires_strong_commit=operation.requires_strong_commit,
            byproduct_pauli=operation.byproduct_pauli,
            measurement_basis=operation.measurement_basis,
            logical_observable_index=operation.logical_observable_index,
            intrinsic_measurement=operation.intrinsic_measurement,
            kind=operation.kind,
        )
        clone_by_source_identity[source_identity] = private_operation
        source_circuit_by_operation_id[operation.id] = operation.circuit
        return private_operation

    executable = tuple(clone(operation) for operation in executable_operations)
    static_decode = tuple(
        clone(operation)
        for operation in static_decode_operations
    )
    dynamic_streams = tuple(
        clone(operation)
        for operation in dynamic_stream_operations
    )
    planning_views = {}
    for operation in (
        executable
        + static_decode
        + dynamic_streams
    ):
        planning_views.setdefault(
            operation.id,
            _planning_view_from_operation(operation),
        )
    return _RunOwnedWorkload(
        executable_operations=executable,
        static_decode_operations=static_decode,
        dynamic_stream_operations=dynamic_streams,
        planning_view_by_operation_id=planning_views,
        declared_view_by_operation_id=declared_view_by_operation_id,
        memberships_by_operation_id={
            operation_id: tuple(memberships)
            for operation_id, memberships in memberships_by_operation_id.items()
        },
        source_circuit_by_operation_id=source_circuit_by_operation_id,
    )


def _planning_view_from_operation(
    operation: Operation,
) -> OperationPlanningView:
    """Freeze every logical operation field while excluding its circuit."""
    return OperationPlanningView.from_operation(operation)


def _install_private_execution_circuits(
    workload: _RunOwnedWorkload,
    device,
) -> None:
    """Install only device-reachable, independently reconstructed circuits."""
    circuit_scope = device.operation_circuit_scope
    if circuit_scope not in ("none", "per_operation"):
        raise ValueError(
            "device operation_circuit_scope must be 'none' or "
            f"'per_operation'; got {circuit_scope!r}"
        )
    if circuit_scope == "none":
        return

    nonempty_circuits = [
        circuit
        for circuit in workload.source_circuit_by_operation_id.values()
        if circuit is not None
    ]
    if not nonempty_circuits:
        return
    try:
        import stim
    except ImportError as error:
        raise RuntimeError(
            "active operation circuits require the stim package"
        ) from error

    private_operation_by_id = {
        operation.id: operation
        for operation in (
            workload.executable_operations
            + workload.static_decode_operations
            + workload.dynamic_stream_operations
        )
    }
    for operation_id, source_circuit in (
        workload.source_circuit_by_operation_id.items()
    ):
        if source_circuit is None:
            continue
        if type(source_circuit) is not stim.Circuit:
            raise TypeError(
                f"operation {operation_id} has an active circuit that is "
                "not an exact stim.Circuit"
            )
        private_operation_by_id[operation_id].circuit = stim.Circuit(
            str(source_circuit)
        )


def _collect_seed_plan(roots, root_seed: Optional[int]):
    """Discover stochastic components once in deterministic semantic order."""
    from .protocols import RunSeedComposite, RunSeedConsumer

    canonical_paths = {}
    active_ids = set()
    seen_paths = set()
    plan = []

    def encoded(path):
        return b"".join(segment.canonical_bytes() for segment in path)

    def walk(path, component):
        path_bytes = encoded(path)
        if path_bytes in seen_paths:
            raise ValueError(f"duplicate component path {_render_run_seed_path(path)}")
        seen_paths.add(path_bytes)

        component_id = id(component)
        if component_id in active_ids:
            first_path = canonical_paths[component_id]
            raise ValueError(
                "component cycle from "
                f"{_render_run_seed_path(path)} to "
                f"{_render_run_seed_path(first_path)}"
            )
        if component_id in canonical_paths:
            return
        canonical_paths[component_id] = path

        active_ids.add(component_id)
        try:
            if isinstance(component, RunSeedConsumer):
                plan.append(_RunSeedPlanEntry(
                    component_path=path,
                    component=component,
                    derived_seed=(
                        None
                        if root_seed is None
                        else _derive_run_component_seed(root_seed, path)
                    ),
                ))
            if not isinstance(component, RunSeedComposite):
                return

            children = []
            child_paths = set()
            for child in component.run_seed_children():
                if type(child) is not RunSeedChild:
                    raise TypeError(
                        f"{type(component).__name__}.run_seed_children() "
                        "must yield exact RunSeedChild values"
                    )
                child_path = encoded(child.relative_path)
                if child_path in child_paths:
                    raise ValueError(
                        "duplicate run-seed child path beneath "
                        f"{_render_run_seed_path(path)}"
                    )
                child_paths.add(child_path)
                children.append((child_path, child))
            for _, child in sorted(children, key=lambda item: item[0]):
                walk(path + child.relative_path, child.child)
        finally:
            active_ids.remove(component_id)

    ordered_roots = sorted(
        ((encoded(path), path, component) for path, component in roots),
        key=lambda item: item[0],
    )
    for _, path, component in ordered_roots:
        walk(path, component)
    return tuple(plan)


def _bind_run_seed_plan(plan: tuple[_RunSeedPlanEntry, ...]) -> None:
    """Reserve all seeds, cancel on failure, then commit the whole plan."""
    acquired = []
    try:
        for entry in plan:
            reservation = entry.component.reserve_run_seed(entry.derived_seed)
            if type(reservation) is not RunSeedReservation:
                raise TypeError(
                    f"{type(entry.component).__name__}.reserve_run_seed() "
                    "must return an exact RunSeedReservation"
                )
            if entry.derived_seed is not None and (
                reservation.proposed_seed_source != "derived"
                or reservation.proposed_seed != entry.derived_seed
            ):
                raise ValueError(
                    f"{type(entry.component).__name__}.reserve_run_seed() "
                    "disagrees with the run seed at "
                    f"{_render_run_seed_path(entry.component_path)}"
                )
            if entry.derived_seed is None and (
                reservation.proposed_seed_source
                not in ("explicit_local", "entropy")
            ):
                raise ValueError(
                    f"{type(entry.component).__name__}.reserve_run_seed() "
                    "must choose explicit_local or entropy without a run seed"
                )
            acquired.append((entry.component, reservation))
    except BaseException:
        for component, reservation in reversed(acquired):
            component.cancel_run_seed(reservation)
        raise

    for component, reservation in acquired:
        component.commit_run_seed(reservation)


def _render_run_seed_path(path: tuple[RunSeedPathSegment, ...]) -> str:
    parts = []
    for segment in path:
        if segment.kind == "none_key":
            parts.append("[None]")
        elif segment.kind in ("string_key", "integer_key"):
            parts.append(f"[{segment.value!r}]")
        else:
            parts.append(segment.value)
    return ".".join(parts)


def _single_layout_code(layout, owner_name: str):
    codes = list(layout.codes())
    if len(codes) != 1:
        raise ValueError(
            f"{owner_name} must declare exactly one planning/runtime code "
            f"(got {len(codes)})")
    return codes[0]


def _validate_program_order(ops, layout) -> dict:
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
    claims_by_operation_id = {}
    for operation in ops:
        claims = tuple(layout.resources_for(operation))
        claims_by_operation_id[operation.id] = claims
        for claim in claims:
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
    return claims_by_operation_id


def _make_infinite(engine):
    from .factories import InfiniteFactory
    return InfiniteFactory(engine)


def _validate_shipped_factory_decode_service(factory, cluster) -> None:
    """Pin shipped correction traffic to the run-owned cluster."""
    from .factories import DistillationFactory, MultiLevelDistillationFactory

    if type(factory) not in (
        DistillationFactory,
        MultiLevelDistillationFactory,
    ):
        return
    if factory.n_corr > 0 and factory.decode_service is not cluster:
        raise ValueError(
            f"{type(factory).__name__} decode_service must be the run-owned "
            "cluster when n_corr is positive"
        )
    if factory.n_corr == 0 and factory.decode_service is not None:
        raise ValueError(
            f"{type(factory).__name__} decode_service must be None when "
            "n_corr is zero"
        )


class ClusterFacade:
    """The 'cluster' read surface chip/factory/metrics code expects,
    backed by the new window_manager + pool."""

    def __init__(self, window_manager, pool):
        self.window_manager = window_manager
        self.pool = pool

    # chip-side surface
    def register_op(self, op) -> None:
        self.window_manager.register_op(op)

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

    def pending_strong_work_snapshot(self):
        return self.window_manager.pending_strong_work_snapshot()

    def admitted_strong_work_snapshot(self):
        return self.pool.admitted_strong_work_snapshot()

    @property
    def strong_cancelled(self):
        return self.pool.strong_cancelled


@dataclass(frozen=True)
class CompletedRun:
    """Scientific results and useful runtime handles from one run."""

    result: PrimaryRunResult
    engine: Any
    window_manager: Any
    pool: Any
    chip: Any
    orchestrator: Any
    factory: Any
    controller: Any
    cluster: Any
    planning: ResolvedPlanningParts


def _capture_primary_run_result(
    *,
    engine,
    gate,
    window_manager,
    operations,
    metric_bindings,
    links,
) -> PrimaryRunResult:
    """Validate and freeze the result while scheduling is sealed."""
    operation_by_id = {}
    for operation in operations:
        operation_by_id.setdefault(operation.id, operation)

    operation_results = []
    for operation_id in sorted(operation_by_id):
        if operation_id in window_manager.op_results:
            logical_observables = tuple(
                _validated_logical_bit(bit)
                for bit in window_manager.op_results[operation_id]
            )
            status = "logical_observables"
        else:
            logical_observables = None
            status = "no_logical_output"
        operation_results.append(
            LogicalOperationResult(
                operation_id=operation_id,
                result_status=status,
                logical_observables=logical_observables,
                stream_offset=operation_by_id[operation_id].stream_offset,
            )
        )

    metric_results = []
    for binding in metric_bindings:
        value = copy.deepcopy(engine._invoke_metric_callback(
            binding.metric.result,
            callback_kind="result",
        ))
        metric_results.append(
            MetricResultRecord(
                name=binding.name,
                value=value,
            )
        )

    if engine._event_queue:
        raise RuntimeError("primary run ended with pending engine events")
    if not gate.workload_complete:
        raise RuntimeError("primary run ended before the chip workload completed")
    return PrimaryRunResult(
        terminal_status="complete",
        event_queue_empty=True,
        decode_work_settled=True,
        chip_workload_complete=True,
        chip_done_ticks=gate.last_finish_time,
        fully_done_ticks=engine.now,
        operation_results=tuple(operation_results),
        link_traffic=copy.deepcopy(links.traffic_json_value()),
        metric_results=tuple(metric_results),
    )


def _validated_logical_bit(value) -> int:
    if type(value) is not int or value not in (0, 1):
        raise TypeError(f"logical observables must contain exact bits; got {value!r}")
    return value


def simulate(run: RunSpec, verbose: bool = False) -> CompletedRun:
    """Execute and return the same completed aggregate as RunSpec.build()."""
    return run.build(verbose=verbose)
