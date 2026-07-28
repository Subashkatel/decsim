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

from dataclasses import dataclass, field
import functools
import hashlib
import inspect
import json
import math
from numbers import Integral
import types
from typing import Any, Callable, Optional, TYPE_CHECKING

from .message import (
    IntrinsicMeasurement,
    Operation,
    OperationPlanningView,
    RunSeedChild,
    RunSeedPathSegment,
    RunSeedReservation,
    Window,
    WindowInfo,
    WindowPlan,
    is_stable_identity,
    same_stable_identity,
)
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
RUN_SEED_NAMESPACE = b"decsim.run-seed.v1"
PREBINDING_OBJECT_FIELDS = (
    "frontend",
    "code",
    "layout",
    "scheme",
    "rounds_policy",
    "planner",
    "strategy",
)
PREBINDING_PROVIDER_FIELDS = (
    "make_controller",
    "make_factory",
    "make_metrics",
)
PLANNER_CHILD_FIELDS = ("layout", "scheme", "rounds_policy")
RUN_SEED_CONSUMER_MEMBERS = (
    "reserve_run_seed",
    "commit_run_seed",
    "cancel_run_seed",
)


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
    planner: "ExecutionPlanner"


@dataclass
class _RunOwnedWorkload:
    """Private executable operations and their circuit-free planning views."""

    executable_operations: tuple[Operation, ...]
    static_decode_operations: tuple[Operation, ...]
    dynamic_stream_operations: tuple[Operation, ...]
    planning_view_by_operation_id: dict[int, OperationPlanningView]
    source_circuit_by_operation_id: dict[int, Any]

    def planning_views(self, operations) -> tuple[OperationPlanningView, ...]:
        return tuple(
            self.planning_view_by_operation_id[operation.id]
            for operation in operations
        )


@dataclass(frozen=True)
class _ResolvedExecutionPlanSpec:
    """Validated immutable copy of the one planner result."""

    windows: tuple[WindowInfo, ...]
    window_count: tuple[tuple[int, int], ...]
    op_windows: tuple[tuple[int, tuple[int, ...]], ...]
    successors: tuple[tuple[int, tuple[int, ...]], ...]
    spatial_nodes: tuple[tuple[int, int], ...]
    rounds_by_operation: tuple[tuple[int, int], ...]
    code_names: tuple[tuple[int, str], ...]
    total_windows: int
    summary_json: bytes

    def materialize(self) -> WindowPlan:
        """Create fresh runtime-owned windows and containers."""
        windows = {
            info.key: Window(
                op_id=info.op_id,
                k=info.k,
                commit_lo=info.commit_lo,
                commit_hi=info.commit_hi,
                buffer_hi=info.buffer_hi,
                n_rounds=info.n_rounds,
                buffer_lo=info.buffer_lo,
                deps=list(info.deps),
                dependents=list(info.dependents),
                deps_remaining=len(info.deps),
            )
            for info in self.windows
        }
        return WindowPlan(
            windows=windows,
            window_count=dict(self.window_count),
            op_windows={
                operation_id: list(window_indices)
                for operation_id, window_indices in self.op_windows
            },
            successors={
                operation_id: list(successor_ids)
                for operation_id, successor_ids in self.successors
            },
            spatial_nodes=dict(self.spatial_nodes),
            rounds_by_operation=dict(self.rounds_by_operation),
            code_names=dict(self.code_names),
            total_windows=self.total_windows,
            summary=json.loads(self.summary_json),
        )


@dataclass(frozen=True)
class _RunSeedPlanEntry:
    """One canonical stochastic leaf in a frozen run component graph."""

    component_path: tuple[RunSeedPathSegment, ...]
    component: Any
    derived_seed: Optional[int]


@dataclass(frozen=True)
class _ComponentGraphEntry:
    component_path: tuple[RunSeedPathSegment, ...]
    parent_path: Optional[tuple[RunSeedPathSegment, ...]]
    edge_path: Optional[tuple[RunSeedPathSegment, ...]]
    component: Any


@dataclass(frozen=True)
class _ComponentAliasEntry:
    alias_path: tuple[RunSeedPathSegment, ...]
    canonical_path: tuple[RunSeedPathSegment, ...]


@dataclass(frozen=True)
class _ResolvedComponentGraph:
    components: tuple[_ComponentGraphEntry, ...]
    aliases: tuple[_ComponentAliasEntry, ...]
    seed_plan: tuple[_RunSeedPlanEntry, ...]


@dataclass(frozen=True)
class LogicalOperationResult:
    """One operation's immutable logical-output disposition."""

    operation_id: int
    result_status: str
    logical_observables: Optional[tuple[int, ...]]


@dataclass(frozen=True)
class MetricResultRecord:
    """One validated metric value in declared observation order."""

    name: str
    canonical_value_json: bytes

    def value(self):
        """Return a fresh JSON-compatible metric value."""
        return json.loads(self.canonical_value_json)


@dataclass(frozen=True)
class PrimaryRunResult:
    """The immutable scientific result of one completed primary drain."""

    schema_version: int
    terminal_status: str
    event_queue_empty: bool
    decode_work_settled: bool
    chip_workload_complete: bool
    chip_done_ticks: int
    fully_done_ticks: int
    operation_results: tuple[LogicalOperationResult, ...]
    metric_results: tuple[MetricResultRecord, ...]

    def logical_results(self) -> dict[int, tuple[int, ...]]:
        """Return logical outputs without conflating absence and empty output."""
        return {
            record.operation_id: record.logical_observables
            for record in self.operation_results
            if record.result_status == "logical_observables"
        }

    def metric_values(self) -> dict[str, Any]:
        """Return fresh decoded metric values keyed by their unique names."""
        return {
            record.name: record.value()
            for record in self.metric_results
        }

    def to_json_value(self) -> dict:
        """Return the closed primary-result schema as fresh JSON values."""
        return {
            "schema_version": self.schema_version,
            "terminal_status": self.terminal_status,
            "event_queue_empty": self.event_queue_empty,
            "decode_work_settled": self.decode_work_settled,
            "chip_workload_complete": self.chip_workload_complete,
            "chip_done_ticks": self.chip_done_ticks,
            "fully_done_ticks": self.fully_done_ticks,
            "operation_results": [
                {
                    "operation_id": record.operation_id,
                    "result_status": record.result_status,
                    "logical_observables": (
                        None
                        if record.logical_observables is None
                        else list(record.logical_observables)
                    ),
                }
                for record in self.operation_results
            ],
            "metric_results": [
                {
                    "name": record.name,
                    "value": record.value(),
                }
                for record in self.metric_results
            ],
        }

    def canonical_json_bytes(self) -> bytes:
        """Encode the primary result with the one canonical JSON policy."""
        return json.dumps(
            self.to_json_value(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")


@dataclass(frozen=True)
class TypedPathSegmentRecord:
    """One closed public typed-path segment."""

    kind: str
    value: Optional[str]

    def __post_init__(self) -> None:
        if type(self.kind) is not str:
            raise TypeError("typed path segment kind must be a built-in str")
        if self.kind == "field":
            if type(self.value) is not str:
                raise TypeError(
                    "typed field path values must be built-in str"
                )
            if not self.value:
                raise ValueError("typed field path values cannot be empty")
            return
        if self.kind == "string_key":
            if type(self.value) is not str:
                raise TypeError(
                    "typed string-key path values must be built-in str"
                )
            return
        if self.kind == "none_key":
            if self.value is not None:
                raise ValueError("typed none-key path values must be None")
            return
        if self.kind == "integer_key":
            if type(self.value) is not str:
                raise TypeError(
                    "typed integer-key path values must be decimal strings"
                )
            digits = (
                self.value[1:]
                if self.value.startswith("-")
                else self.value
            )
            if (
                not digits
                or not digits.isascii()
                or not digits.isdigit()
                or (len(digits) > 1 and digits.startswith("0"))
                or self.value == "-0"
                or self.value.startswith("+")
            ):
                raise ValueError(
                    "typed integer-key path values must be canonical decimal "
                    "strings"
                )
            return
        raise ValueError(f"unknown typed path segment kind {self.kind!r}")

    @classmethod
    def from_seed_segment(
        cls,
        segment: RunSeedPathSegment,
    ) -> "TypedPathSegmentRecord":
        if segment.kind == "integer_key":
            value = str(segment.value)
        else:
            value = segment.value
        return cls(kind=segment.kind, value=value)

    def to_json_value(self) -> dict:
        return {"kind": self.kind, "value": self.value}


@dataclass(frozen=True)
class ResolvedComponent:
    """One canonical externally variable behavior component."""

    component_path: tuple[TypedPathSegmentRecord, ...]
    parent_path: Optional[tuple[TypedPathSegmentRecord, ...]]
    edge_path: Optional[tuple[TypedPathSegmentRecord, ...]]
    implementation: str
    configuration: Any
    configuration_status: str


@dataclass(frozen=True)
class FixedCompositionRecord:
    """One behavior-bearing object constructed directly by the run root."""

    component_path: tuple[TypedPathSegmentRecord, ...]
    implementation: str
    configuration: Any


@dataclass(frozen=True)
class ResolvedAlias:
    """A repeated behavior path and its first canonical object path."""

    alias_path: tuple[TypedPathSegmentRecord, ...]
    canonical_path: tuple[TypedPathSegmentRecord, ...]


@dataclass(frozen=True)
class ResolvedSeedBinding:
    """One canonical stochastic owner and its effective seed source."""

    component_path: tuple[TypedPathSegmentRecord, ...]
    seed_source: str
    seed: Optional[int]


@dataclass(frozen=True)
class ResolvedRunManifest:
    """Immutable seed and result provenance for a completed run."""

    schema_version: int
    root_seed: Optional[int]
    components: tuple[ResolvedComponent, ...]
    fixed_composition: tuple[FixedCompositionRecord, ...]
    aliases: tuple[ResolvedAlias, ...]
    seed_bindings: tuple[ResolvedSeedBinding, ...]
    primary_result_sha256: str

    def to_json_value(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "root_seed": self.root_seed,
            "components": [
                {
                    "component_path": _typed_path_json(
                        component.component_path
                    ),
                    "parent_path": (
                        None
                        if component.parent_path is None
                        else _typed_path_json(component.parent_path)
                    ),
                    "edge_path": (
                        None
                        if component.edge_path is None
                        else _typed_path_json(component.edge_path)
                    ),
                    "implementation": component.implementation,
                    "configuration": component.configuration,
                    "configuration_status": (
                        component.configuration_status
                    ),
                }
                for component in self.components
            ],
            "fixed_composition": [
                {
                    "component_path": _typed_path_json(
                        component.component_path
                    ),
                    "implementation": component.implementation,
                    "configuration": component.configuration,
                }
                for component in self.fixed_composition
            ],
            "aliases": [
                {
                    "alias_path": _typed_path_json(alias.alias_path),
                    "canonical_path": _typed_path_json(
                        alias.canonical_path
                    ),
                }
                for alias in self.aliases
            ],
            "seed_bindings": [
                {
                    "component_path": _typed_path_json(
                        binding.component_path
                    ),
                    "seed_source": binding.seed_source,
                    "seed": binding.seed,
                }
                for binding in self.seed_bindings
            ],
            "primary_result_sha256": self.primary_result_sha256,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_json_value(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )


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
        self._validate_layout_selection(planning, planning_operations)

    def _validate_configuration(self) -> ResolvedPlanningParts:
        """Validate configuration-only state and resolve planning once."""
        self._validated_root_seed()
        self._reject_prebinding_seed_consumers()
        if (self.ops is None) == (self.frontend is None):
            raise ValueError("provide exactly one of ops= or frontend=")
        self._validate_supplied_parts()
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

    def _reject_prebinding_seed_consumers(self) -> None:
        """Reject stochastic owners that would execute before root binding."""
        for field_name in PREBINDING_OBJECT_FIELDS:
            component = object.__getattribute__(self, field_name)
            if component is not None:
                _reject_static_seed_consumer(field_name, component)

        planner = object.__getattribute__(self, "planner")
        if planner is not None:
            for child_name in PLANNER_CHILD_FIELDS:
                child = _stored_planner_child(planner, child_name)
                _reject_static_seed_consumer(
                    f"planner.{child_name}",
                    child,
                )

        for field_name in PREBINDING_PROVIDER_FIELDS:
            provider = object.__getattribute__(self, field_name)
            if provider is not None:
                _scan_prebinding_provider(field_name, provider)

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
        if self.device is not None:
            missing = object()
            circuit_scope = inspect.getattr_static(
                self.device,
                "operation_circuit_scope",
                missing,
            )
            if circuit_scope is missing or type(circuit_scope) is not str:
                raise TypeError(
                    "device does not satisfy SyndromeDevice: "
                    "operation_circuit_scope must be a stored exact string: "
                    "'none' or 'per_operation'"
                )
            if circuit_scope not in ("none", "per_operation"):
                raise ValueError(
                    "device operation_circuit_scope must be 'none' or "
                    f"'per_operation'; got {circuit_scope!r}"
                )
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

        probe = _planning_view_from_operation(
            Operation(-1, "probe", (0,))
        )
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

    def build(self, verbose: bool = False) -> "CompletedRun":
        """Construct, execute, and freeze one complete primary run."""
        if self._build_state != "unstarted":
            raise RuntimeError(
                f"RunSpec build is already {self._build_state}; "
                "construct a fresh RunSpec and runtime graph"
            )
        self._build_state = "committing"
        try:
            completed_run = self._build_once(verbose=verbose)
        except BaseException:
            self._build_state = "invalid"
            raise
        self._build_state = "complete"
        return completed_run

    def _build_once(self, verbose: bool = False) -> "CompletedRun":
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
        self._validate_layout_selection(planning, planning_operations)
        resource_claims_by_operation_id = _validate_program_order(
            workload.planning_views(ops),
            planning.layout,
        )

        strategy = self.strategy if self.strategy is not None else Baseline()
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
        execution_plan_spec = _freeze_execution_plan(
            planning.planner.plan(list(planned_views)),
            (operation.id for operation in planned_views),
        )
        execution_plan_summary = json.loads(
            execution_plan_spec.summary_json
        )
        check_window_size = getattr(strategy, "check_window_size", None)
        if check_window_size is not None:
            check_window_size(
                execution_plan_summary.get(
                    "commit",
                    planning.code.commit_rounds(),
                ),
                execution_plan_summary.get(
                    "buffer",
                    planning.code.buffer_rounds(),
                ),
            )

        engine = Engine(verbose=verbose, construction_guarded=True)
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
        _install_private_execution_circuits(workload, device)
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
        links = controller.links

        store = PayloadStore(memory_model=self.memory_model)
        window_manager = WindowManager(
            engine, scheme=planning.scheme, layout=planning.layout,
            rounds_policy=planning.rounds_policy,
            code=planning.code, deadline_policy=deadline_policy, links=links,
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
        )

        factory = self.make_factory(engine, cluster) \
            if self.make_factory is not None else _make_infinite(engine)
        _validate_protocol_part("factory", factory, MagicStateFactory)
        if factory.engine is not engine:
            raise ValueError(
                f"{type(factory).__name__} uses a different engine from "
                "the RunSpec build")
        _validate_shipped_factory_decode_service(factory, cluster)
        source = ClockedDevice(engine, device, controller, cluster,
                               window_manager.rounds_for)
        round_us = self.round_us if self.round_us is not None \
            else self.timing.round_us
        gate = Chip(
            engine, source=source, controller=controller, cluster=cluster,
            factory=factory, round_ticks=us(round_us),
            code_distance=planning.code.distance, idle_policy=idle_policy,
            operation_code=planning.code,
            resource_claims_by_operation_id=(
                resource_claims_by_operation_id
            ),
            max_idle_rounds=self.max_idle_rounds,
            gates_start_on_round_boundaries=self.gates_start_on_round_boundaries,
            frame=orchestrator.frame)

        metrics = []
        if self.make_metrics is not None:
            metrics = self.make_metrics(engine, cluster, gate, factory)
            if type(metrics) is not list:
                raise TypeError("make_metrics must return a list")
            metric_names = set()
            for index, metric in enumerate(metrics):
                _validate_protocol_part(
                    f"make_metrics result {index}", metric, Metric)
                if type(metric.name) is not str or not metric.name:
                    raise TypeError(
                        f"make_metrics result {index} name must be an exact "
                        "nonempty built-in str"
                    )
                if metric.name in metric_names:
                    raise ValueError(
                        f"duplicate metric name {metric.name!r}"
                    )
                metric_names.add(metric.name)

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
            links=links,
            metrics=metrics,
            workload=workload,
        )
        fixed_composition_anchors = _fixed_composition_anchors(
            engine=engine,
            payload_store=store,
            window_manager=window_manager,
            decoder_manager=pool,
            strategy_services=services,
            cluster=cluster,
            clocked_device=source,
            chip=gate,
        )
        component_graph = _materialize_component_graph(
            seed_roots,
            self._validated_root_seed(),
            anchors=fixed_composition_anchors,
        )
        seed_plan = component_graph.seed_plan
        reservations = _bind_run_seed_plan(seed_plan)
        seed_bindings = tuple(
            ResolvedSeedBinding(
                component_path=_typed_path_records(entry.component_path),
                seed_source=reservation.proposed_seed_source,
                seed=reservation.proposed_seed,
            )
            for entry, reservation in zip(seed_plan, reservations)
        )
        resolved_components = _resolved_components(component_graph)
        fixed_composition = _resolved_fixed_composition(
            fixed_composition_anchors
        )
        resolved_aliases = tuple(
            ResolvedAlias(
                alias_path=_typed_path_records(alias.alias_path),
                canonical_path=_typed_path_records(alias.canonical_path),
            )
            for alias in component_graph.aliases
        )

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
                window_manager.register_dynamic_stream(stream, planning.code)
            for metric in metrics:
                engine.add_metric(metric)
            gate.load(ops)
            engine._start_running()
            engine.run()
            pool.check_decode_work_settled()
            engine._begin_finalization()
            result = _capture_primary_run_result(
                engine=engine,
                gate=gate,
                window_manager=window_manager,
                operations=all_operations,
                metrics=metrics,
            )
            manifest = ResolvedRunManifest(
                schema_version=1,
                root_seed=self._validated_root_seed(),
                components=resolved_components,
                fixed_composition=fixed_composition,
                aliases=resolved_aliases,
                seed_bindings=seed_bindings,
                primary_result_sha256=hashlib.sha256(
                    result.canonical_json_bytes(),
                ).hexdigest(),
            )
            engine._complete()
        except BaseException:
            engine._invalidate()
            raise

        return CompletedRun(
            result=result,
            manifest=manifest,
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
        links,
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
            ((field_segment("planner"),), planning.planner),
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
        for link_name in ("qc", "cd", "dd", "do", "oc", "cq", "ws"):
            roots.append(
                (
                    (
                        field_segment("controller_links"),
                        field_segment(link_name),
                    ),
                    getattr(links, link_name),
                )
            )
        for metric in metrics:
            roots.append(
                (
                    (
                        field_segment("metrics"),
                        RunSeedPathSegment("string_key", metric.name),
                    ),
                    metric,
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
    value_type = type(value)
    if value_type is int or value_type is str:
        return True
    if value_type is tuple:
        return all(_is_runtime_identity(item) for item in value)
    return False


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
        source_circuit_by_operation_id=source_circuit_by_operation_id,
    )


def _planning_view_from_operation(
    operation: Operation,
) -> OperationPlanningView:
    """Freeze every logical operation field while excluding its circuit."""
    return OperationPlanningView.from_operation(operation)


def _freeze_execution_plan(
    candidate,
    planned_operation_ids,
) -> _ResolvedExecutionPlanSpec:
    """Validate and detach the sole planner result from caller-owned state."""
    if type(candidate) is not WindowPlan:
        raise TypeError("planner must return an exact WindowPlan")
    operation_ids = tuple(planned_operation_ids)
    expected_operation_ids = set(operation_ids)

    mapping_fields = (
        "windows",
        "window_count",
        "op_windows",
        "successors",
        "spatial_nodes",
        "rounds_by_operation",
        "code_names",
        "summary",
    )
    for field_name in mapping_fields:
        if type(getattr(candidate, field_name)) is not dict:
            raise TypeError(
                f"WindowPlan.{field_name} must be an exact dict"
            )

    per_operation_fields = (
        "window_count",
        "op_windows",
        "successors",
        "spatial_nodes",
        "rounds_by_operation",
        "code_names",
    )
    for field_name in per_operation_fields:
        keys = set(getattr(candidate, field_name))
        if keys != expected_operation_ids:
            raise ValueError(
                f"WindowPlan.{field_name} keys must exactly match planned "
                "operation ids"
            )

    frozen_windows = []
    for key in sorted(candidate.windows):
        if (
            type(key) is not tuple
            or len(key) != 2
            or any(type(item) is not int for item in key)
        ):
            raise TypeError(
                "WindowPlan window keys must be exact (int, int) tuples"
            )
        window = candidate.windows[key]
        if type(window) is not Window:
            raise TypeError("WindowPlan.windows must contain exact Window values")
        if window.key != key:
            raise ValueError(
                f"WindowPlan key {key!r} does not match Window.key "
                f"{window.key!r}"
            )
        if window.op_id not in expected_operation_ids:
            raise ValueError(
                f"WindowPlan window {key!r} has an unplanned operation id"
            )
        if (
            window.k < 0
            or window.commit_lo < 1
            or window.commit_hi < window.commit_lo
            or window.buffer_hi < window.commit_hi
            or (
                window.buffer_lo is not None
                and (
                    type(window.buffer_lo) is not int
                    or not 1 <= window.buffer_lo <= window.commit_lo
                )
            )
        ):
            raise ValueError(f"WindowPlan window {key!r} has invalid geometry")
        if window.n_rounds != window.buffer_hi - window.start_round + 1:
            raise ValueError(
                f"WindowPlan window {key!r} has inconsistent n_rounds"
            )
        if any(
            type(item) is not tuple
            or len(item) != 2
            or any(type(part) is not int for part in item)
            for item in tuple(window.deps) + tuple(window.dependents)
        ):
            raise TypeError(
                f"WindowPlan window {key!r} dependencies must be exact "
                "(int, int) tuples"
            )
        if (
            window.deps_remaining != len(window.deps)
            or window.committed
            or window.queued
            or window.blocked_logged
            or window.boundary_in != {}
            or any(
                timestamp is not None
                for timestamp in (
                    window.t_first_round,
                    window.t_data_complete,
                    window.t_queued,
                    window.t_dispatch,
                    window.t_done,
                )
            )
        ):
            raise ValueError(
                f"WindowPlan window {key!r} contains runtime state"
            )
        frozen_windows.append(WindowInfo.from_window(window))

    frozen_window_keys = {window.key for window in frozen_windows}
    for window in frozen_windows:
        if (
            any(key not in frozen_window_keys for key in window.deps)
            or any(key not in frozen_window_keys for key in window.dependents)
        ):
            raise ValueError(
                f"WindowPlan window {window.key!r} names an unknown dependency"
            )

    window_count = []
    op_windows = []
    successors = []
    spatial_nodes = []
    rounds_by_operation = []
    code_names = []
    for operation_id in sorted(expected_operation_ids):
        count = candidate.window_count[operation_id]
        if type(count) is not int or count < 1:
            raise ValueError(
                f"WindowPlan.window_count[{operation_id}] must be positive"
            )
        indices = candidate.op_windows[operation_id]
        if type(indices) is not list or any(
            type(index) is not int for index in indices
        ):
            raise TypeError(
                f"WindowPlan.op_windows[{operation_id}] must be a list "
                "of exact ints"
            )
        expected_indices = list(range(count))
        if indices != expected_indices or any(
            (operation_id, index) not in frozen_window_keys
            for index in indices
        ):
            raise ValueError(
                f"WindowPlan.op_windows[{operation_id}] is inconsistent"
            )
        successor_ids = candidate.successors[operation_id]
        if type(successor_ids) is not list or any(
            type(successor_id) is not int
            or successor_id not in expected_operation_ids
            for successor_id in successor_ids
        ):
            raise TypeError(
                f"WindowPlan.successors[{operation_id}] must be a list "
                "of planned exact-int ids"
            )
        spatial_node_count = candidate.spatial_nodes[operation_id]
        if type(spatial_node_count) is not int or spatial_node_count < 1:
            raise ValueError(
                f"WindowPlan.spatial_nodes[{operation_id}] must be positive"
            )
        round_count = candidate.rounds_by_operation[operation_id]
        if type(round_count) is not int or round_count < 1:
            raise ValueError(
                f"WindowPlan.rounds_by_operation[{operation_id}] "
                "must be positive"
            )
        code_name = candidate.code_names[operation_id]
        if type(code_name) is not str or not code_name:
            raise ValueError(
                f"WindowPlan.code_names[{operation_id}] must be nonempty"
            )
        window_count.append((operation_id, count))
        op_windows.append((operation_id, tuple(indices)))
        successors.append((operation_id, tuple(successor_ids)))
        spatial_nodes.append((operation_id, spatial_node_count))
        rounds_by_operation.append((operation_id, round_count))
        code_names.append((operation_id, code_name))

    if (
        type(candidate.total_windows) is not int
        or candidate.total_windows != len(frozen_windows)
        or candidate.total_windows != sum(count for _, count in window_count)
    ):
        raise ValueError("WindowPlan.total_windows is inconsistent")
    summary_json = json.dumps(
        _validated_json_value(candidate.summary),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _ResolvedExecutionPlanSpec(
        windows=tuple(frozen_windows),
        window_count=tuple(window_count),
        op_windows=tuple(op_windows),
        successors=tuple(successors),
        spatial_nodes=tuple(spatial_nodes),
        rounds_by_operation=tuple(rounds_by_operation),
        code_names=tuple(code_names),
        total_windows=candidate.total_windows,
        summary_json=summary_json,
    )


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


def _encoded_component_path(
    path: tuple[RunSeedPathSegment, ...],
) -> bytes:
    return b"".join(segment.canonical_bytes() for segment in path)


def _typed_path_records(
    path: tuple[RunSeedPathSegment, ...],
) -> tuple[TypedPathSegmentRecord, ...]:
    return tuple(
        TypedPathSegmentRecord.from_seed_segment(segment)
        for segment in path
    )


def _typed_path_json(
    path: tuple[TypedPathSegmentRecord, ...],
) -> list[dict]:
    return [segment.to_json_value() for segment in path]


def _implementation_name(component) -> str:
    implementation = type(component)
    return f"{implementation.__module__}.{implementation.__qualname__}"


def _fixed_composition_anchors(**components):
    field_segment = lambda value: RunSeedPathSegment("field", value)
    return tuple(
        (
            (
                field_segment("fixed"),
                field_segment(component_name),
            ),
            component,
        )
        for component_name, component in components.items()
    )


def _resolved_fixed_composition(
    anchors,
) -> tuple[FixedCompositionRecord, ...]:
    records = [
        FixedCompositionRecord(
            component_path=_typed_path_records(component_path),
            implementation=_implementation_name(component),
            configuration=None,
        )
        for component_path, component in anchors
    ]
    records.sort(
        key=lambda record: b"".join(
            _public_path_segment_bytes(segment)
            for segment in record.component_path
        )
    )
    return tuple(records)


def _public_path_segment_bytes(segment: TypedPathSegmentRecord) -> bytes:
    value = (
        int(segment.value)
        if segment.kind == "integer_key"
        else segment.value
    )
    return RunSeedPathSegment(segment.kind, value).canonical_bytes()


def _resolved_components(
    graph: _ResolvedComponentGraph,
) -> tuple[ResolvedComponent, ...]:
    return tuple(
        ResolvedComponent(
            component_path=_typed_path_records(entry.component_path),
            parent_path=(
                None
                if entry.parent_path is None
                else _typed_path_records(entry.parent_path)
            ),
            edge_path=(
                None
                if entry.edge_path is None
                else _typed_path_records(entry.edge_path)
            ),
            implementation=_implementation_name(entry.component),
            configuration=None,
            configuration_status="opaque",
        )
        for entry in graph.components
    )


def _materialize_component_graph(
    roots,
    root_seed: Optional[int],
    *,
    anchors=(),
) -> _ResolvedComponentGraph:
    """Freeze variable topology, aliases, and canonical seed consumers once."""
    from .protocols import RunSeedComposite, RunSeedConsumer

    canonical_paths: dict[int, tuple[RunSeedPathSegment, ...]] = {}
    active_ids: set[int] = set()
    seen_full_paths: set[bytes] = set()
    components = []
    aliases = []
    seed_plan = []

    canonical_anchors = []
    for component_path, component in anchors:
        canonical_anchors.append(
            (_encoded_component_path(component_path), component_path, component)
        )
    canonical_anchors.sort(key=lambda item: item[0])
    for encoded_path, component_path, component in canonical_anchors:
        if encoded_path in seen_full_paths:
            raise ValueError(
                "duplicate component path "
                f"{_render_run_seed_path(component_path)}"
            )
        seen_full_paths.add(encoded_path)
        component_id = id(component)
        if component_id in canonical_paths:
            aliases.append(
                _ComponentAliasEntry(
                    alias_path=component_path,
                    canonical_path=canonical_paths[component_id],
                )
            )
        else:
            canonical_paths[component_id] = component_path

    def walk(component_path, component, parent_path, edge_path) -> None:
        encoded_component_path = _encoded_component_path(component_path)
        if encoded_component_path in seen_full_paths:
            raise ValueError(
                "duplicate component path "
                f"{_render_run_seed_path(component_path)}"
            )
        seen_full_paths.add(encoded_component_path)
        component_id = id(component)
        if component_id in active_ids:
            first_path = canonical_paths[component_id]
            raise ValueError(
                "component cycle from "
                f"{_render_run_seed_path(component_path)} to "
                f"{_render_run_seed_path(first_path)}"
            )
        if component_id in canonical_paths:
            aliases.append(
                _ComponentAliasEntry(
                    alias_path=component_path,
                    canonical_path=canonical_paths[component_id],
                )
            )
            return

        canonical_paths[component_id] = component_path
        components.append(
            _ComponentGraphEntry(
                component_path=component_path,
                parent_path=parent_path,
                edge_path=edge_path,
                component=component,
            )
        )
        active_ids.add(component_id)
        try:
            if isinstance(component, RunSeedConsumer):
                derived_seed = (
                    None
                    if root_seed is None
                    else _derive_run_component_seed(root_seed, component_path)
                )
                seed_plan.append(
                    _RunSeedPlanEntry(
                        component_path=component_path,
                        component=component,
                        derived_seed=derived_seed,
                    )
                )

            if not isinstance(component, RunSeedComposite):
                return
            children = tuple(component.run_seed_children())
            canonical_children = []
            seen_relative_paths = set()
            for child in children:
                if type(child) is not RunSeedChild:
                    raise TypeError(
                        f"{type(component).__name__}.run_seed_children() "
                        "must yield exact RunSeedChild values"
                    )
                encoded_path = b"".join(
                    segment.canonical_bytes()
                    for segment in child.relative_path
                )
                if encoded_path in seen_relative_paths:
                    raise ValueError(
                        "duplicate run-seed child path beneath "
                        f"{_render_run_seed_path(component_path)}"
                    )
                seen_relative_paths.add(encoded_path)
                canonical_children.append((encoded_path, child))
            canonical_children.sort(key=lambda item: item[0])
            for _, child in canonical_children:
                walk(
                    component_path + child.relative_path,
                    child.child,
                    component_path,
                    child.relative_path,
                )
        finally:
            active_ids.remove(component_id)

    canonical_roots = []
    for component_path, component in roots:
        encoded_path = _encoded_component_path(component_path)
        canonical_roots.append((encoded_path, component_path, component))
    canonical_roots.sort(key=lambda item: item[0])
    for _, component_path, component in canonical_roots:
        walk(component_path, component, None, None)
    components.sort(
        key=lambda entry: _encoded_component_path(entry.component_path)
    )
    aliases.sort(
        key=lambda entry: _encoded_component_path(entry.alias_path)
    )
    return _ResolvedComponentGraph(
        components=tuple(components),
        aliases=tuple(aliases),
        seed_plan=tuple(seed_plan),
    )


def _bind_run_seed_plan(
    plan: tuple[_RunSeedPlanEntry, ...],
) -> tuple[RunSeedReservation, ...]:
    """Reserve every leaf, cancel on error, then perform total commits."""
    acquired = []
    try:
        for entry in plan:
            reservation = entry.component.reserve_run_seed(
                entry.derived_seed,
            )
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
                    "returned metadata that does not match the derived "
                    f"component seed at "
                    f"{_render_run_seed_path(entry.component_path)}"
                )
            if entry.derived_seed is None and (
                reservation.proposed_seed_source
                not in ("explicit_local", "entropy")
            ):
                raise ValueError(
                    f"{type(entry.component).__name__}.reserve_run_seed() "
                    "must report explicit_local or entropy under a None "
                    "run root"
                )
            acquired.append((entry, reservation))
    except BaseException:
        for entry, reservation in reversed(acquired):
            entry.component.cancel_run_seed(reservation)
        raise

    for entry, reservation in acquired:
        entry.component.commit_run_seed(reservation)
    return tuple(reservation for _, reservation in acquired)


def _render_run_seed_path(
    component_path: tuple[RunSeedPathSegment, ...],
) -> str:
    """Render typed seed paths only for diagnostics, never for hashing."""
    parts = []
    for segment in component_path:
        if segment.kind == "none_key":
            parts.append("[None]")
        elif segment.kind == "string_key":
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


def _declares_static_seed_consumer(component) -> bool:
    """Inspect the seed capability without invoking user-controlled access."""
    missing = object()
    return all(
        inspect.getattr_static(component, member_name, missing) is not missing
        for member_name in RUN_SEED_CONSUMER_MEMBERS
    )


def _reject_static_seed_consumer(component_path: str, component) -> None:
    if _declares_static_seed_consumer(component):
        raise ValueError(
            f"{component_path} object {type(component).__name__} declares "
            "RunSeedConsumer behavior that would execute before run-seed "
            "binding"
        )


def _stored_planner_child(planner, child_name: str):
    """Read an approved planner child shape without binding a descriptor."""
    try:
        instance_fields = object.__getattribute__(planner, "__dict__")
    except AttributeError:
        instance_fields = {}
    if type(instance_fields) is dict and child_name in instance_fields:
        return instance_fields[child_name]

    missing = object()
    child = inspect.getattr_static(planner, child_name, missing)
    if child is missing:
        raise TypeError(
            "planner does not satisfy ExecutionPlanner: "
            f"planner.{child_name} must be a stored non-descriptor child"
        )
    if inspect.isdatadescriptor(child) or (
        getattr(type(child), "__get__", None) is not None
    ):
        raise TypeError(
            "planner does not satisfy ExecutionPlanner: "
            f"planner.{child_name} must be a stored non-descriptor child"
        )
    return child


def _scan_prebinding_provider(component_path: str, provider) -> None:
    """Classify provider ownership without binding or invoking a wrapper."""
    provider_type = type(provider)
    if provider_type is types.FunctionType:
        _reject_static_seed_consumer(component_path, provider)
        return
    if provider_type is types.MethodType:
        _reject_static_seed_consumer(component_path, provider.__func__)
        _reject_static_seed_consumer(component_path, provider.__self__)
        return
    if provider_type is type:
        # Instance seed capabilities belong to the runtime object returned by
        # construction; that object joins the later binding transaction.
        return
    if (
        provider_type is functools.partial
        or provider_type is types.BuiltinFunctionType
        or provider_type is types.BuiltinMethodType
        or isinstance(provider, type)
    ):
        raise TypeError(
            f"{component_path} has unsupported provider shape "
            f"{provider_type.__name__}"
        )
    if callable(provider):
        _reject_static_seed_consumer(component_path, provider)
        return
    raise TypeError(f"{component_path} must be callable")


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

    def __init__(self, window_manager, pool, layout):
        self.window_manager = window_manager
        self.pool = pool
        self.layout = layout

    # chip-side surface
    def register_op(self, op) -> None:
        self.window_manager.register_op(op)

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


@dataclass(frozen=True)
class CompletedRun:
    """One completed run with immutable result and provenance records."""

    result: PrimaryRunResult
    manifest: ResolvedRunManifest
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
    metrics,
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
            )
        )

    metric_results = []
    for metric in metrics:
        value = _validated_json_value(metric.result())
        canonical_value_json = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        metric_results.append(
            MetricResultRecord(
                name=metric.name,
                canonical_value_json=canonical_value_json,
            )
        )

    if engine._event_queue:
        raise RuntimeError("primary run ended with pending engine events")
    if not gate.workload_complete:
        raise RuntimeError("primary run ended before the chip workload completed")
    return PrimaryRunResult(
        schema_version=1,
        terminal_status="complete",
        event_queue_empty=True,
        decode_work_settled=True,
        chip_workload_complete=True,
        chip_done_ticks=gate.last_finish_time,
        fully_done_ticks=engine.now,
        operation_results=tuple(operation_results),
        metric_results=tuple(metric_results),
    )


def _validated_logical_bit(value) -> int:
    if type(value) is not int or value not in (0, 1):
        raise TypeError(f"logical observables must contain exact bits; got {value!r}")
    return value


def _validated_json_value(value):
    """Copy one value from the closed metric JSON domain."""
    value_type = type(value)
    if value is None or value_type in (bool, int, str):
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"metric floats must be finite; got {value!r}")
        return value
    if value_type is list:
        return [_validated_json_value(item) for item in value]
    if value_type is dict:
        copied = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(
                    f"metric object keys must be exact strings; got {key!r}"
                )
            copied[key] = _validated_json_value(item)
        return copied
    raise TypeError(
        "metric values must use the closed JSON domain; "
        f"got {type(value).__name__}"
    )


def simulate(run: RunSpec, verbose: bool = False) -> CompletedRun:
    """Execute and return the same completed aggregate as RunSpec.build()."""
    return run.build(verbose=verbose)
