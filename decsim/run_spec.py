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
import importlib.metadata
import importlib.util
import inspect
import json
import math
from numbers import Integral
from pathlib import Path
import sys
import types
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
from .config import TICKS_PER_US, TimingConfig, us

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
PREBINDING_OBJECT_FIELDS = (
    "frontend",
    "code",
    "layout",
    "scheme",
    "rounds_policy",
    "strategy",
)
PREBINDING_PROVIDER_FIELDS = (
    "make_controller",
    "make_factory",
    "make_metrics",
    "make_orchestrator",
)
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
class _ComponentGraphEntry:
    component_path: tuple[RunSeedPathSegment, ...]
    parent_path: Optional[tuple[RunSeedPathSegment, ...]]
    edge_path: Optional[tuple[RunSeedPathSegment, ...]]
    component: Any
    configuration_json: Optional[bytes]
    configuration_status: str


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
class _ResolvedMetricBinding:
    metric: Any
    name: str
    result_schema_version: int


@dataclass(frozen=True)
class MetricResultRecord:
    """One validated metric value in declared observation order."""

    name: str
    result_schema_version: int
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
                    "result_schema_version": record.result_schema_version,
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
            if not self.value or not is_stable_string(self.value):
                raise ValueError(
                    "typed field path values must be nonempty Unicode scalar "
                    "strings"
                )
            return
        if self.kind == "string_key":
            if not is_stable_string(self.value):
                raise TypeError(
                    "typed string-key path values must be Unicode scalar "
                    "strings"
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
class StableIdentityRecord:
    """One recursive, collision-free public workload identity."""

    kind: str
    value: Optional[str]
    items: Optional[tuple["StableIdentityRecord", ...]]

    def __post_init__(self) -> None:
        if self.kind == "integer":
            if type(self.value) is not str or self.items is not None:
                raise TypeError(
                    "integer identity records require a decimal string and "
                    "items=None"
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
                    "integer identity values must be canonical decimal strings"
                )
            return
        if self.kind == "string":
            if not is_stable_string(self.value) or self.items is not None:
                raise TypeError(
                    "string identity records require a Unicode scalar string "
                    "and items=None"
                )
            return
        if self.kind == "tuple":
            if self.value is not None or type(self.items) is not tuple:
                raise TypeError(
                    "tuple identity records require value=None and tuple items"
                )
            if not all(
                type(item) is StableIdentityRecord
                for item in self.items
            ):
                raise TypeError(
                    "tuple identity items must be exact StableIdentityRecord "
                    "values"
                )
            return
        raise ValueError(f"unknown stable identity kind {self.kind!r}")

    @classmethod
    def from_identity(cls, identity) -> "StableIdentityRecord":
        if type(identity) is int:
            return cls("integer", str(identity), None)
        if is_stable_string(identity):
            return cls("string", identity, None)
        if type(identity) is tuple and all(
            is_stable_identity(item) for item in identity
        ):
            return cls(
                "tuple",
                None,
                tuple(cls.from_identity(item) for item in identity),
            )
        raise TypeError(
            "stable identities are exact int, Unicode scalar str, or "
            "recursive tuples"
        )

    def to_identity(self):
        if self.kind == "integer":
            return int(self.value)
        if self.kind == "string":
            return self.value
        return tuple(item.to_identity() for item in self.items)

    def canonical_bytes(self) -> bytes:
        return stable_identity_bytes(self.to_identity())

    def to_json_value(self) -> dict:
        return {
            "kind": self.kind,
            "value": self.value,
            "items": (
                None
                if self.items is None
                else [item.to_json_value() for item in self.items]
            ),
        }


@dataclass(frozen=True)
class ResolvedCodeSelectionRecord:
    """The one code path selected for an operation or patch consumer."""

    consumer_kind: str
    consumer_identity: StableIdentityRecord
    code_path: tuple[TypedPathSegmentRecord, ...]


@dataclass(frozen=True)
class ResolvedCadenceRecord:
    """One consumer's exact executable round cadence and winning source."""

    consumer_kind: str
    consumer_identity: StableIdentityRecord
    code_path: tuple[TypedPathSegmentRecord, ...]
    round_ticks: int
    origin: str


@dataclass(frozen=True)
class _ResolvedCodeCadencePlan:
    code_geometry: ResolvedCodeGeometry
    spatial_profile: ResolvedCodeSpatialProfile
    operations: tuple[ResolvedOperationPlanning, ...]
    patches: tuple[ResolvedPatchPlanning, ...]
    code_selections: tuple[ResolvedCodeSelectionRecord, ...]
    cadences: tuple[ResolvedCadenceRecord, ...]
    round_ticks_by_operation_id: tuple[tuple[int, int], ...]
    round_ticks_by_patch: tuple[tuple[Any, int], ...]
    round_ticks: int


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
class ContainedImplementationRecord:
    """One fixed implementation privately owned by a containing component."""

    component_path: tuple[TypedPathSegmentRecord, ...]
    owner_path: tuple[TypedPathSegmentRecord, ...]
    field_path: tuple[TypedPathSegmentRecord, ...]
    implementation: str
    configuration: Any


class _CanonicalJsonRecord:
    """Immutable storage shared by the closed compound manifest records."""

    canonical_json: bytes

    def to_json_value(self) -> dict:
        return json.loads(self.canonical_json)

    @classmethod
    def freeze(cls, value: dict):
        if type(value) is not dict:
            raise TypeError(f"{cls.__name__} requires an exact dict")
        detached = _validated_json_value(value)
        return cls(
            json.dumps(
                detached,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )


@dataclass(frozen=True)
class ResolvedOperationRecord(_CanonicalJsonRecord):
    """One complete immutable effective workload operation."""

    canonical_json: bytes = field(repr=False)


@dataclass(frozen=True)
class ResolvedExecutionPlanRecord(_CanonicalJsonRecord):
    """The complete immutable plan consumed by the window runtime."""

    canonical_json: bytes = field(repr=False)


@dataclass(frozen=True)
class ChipLoadPlanRecord(_CanonicalJsonRecord):
    """The immutable operation order and stream-offset load disposition."""

    canonical_json: bytes = field(repr=False)


@dataclass(frozen=True)
class EffectiveTimingRecord(_CanonicalJsonRecord):
    """Resolved run-wide time quantities with their explicit units."""

    canonical_json: bytes = field(repr=False)


@dataclass(frozen=True)
class ResolvedLinkRecord(_CanonicalJsonRecord):
    """One actual controller link and its effective transport behavior."""

    canonical_json: bytes = field(repr=False)


@dataclass(frozen=True)
class EffectiveResourceRecord(_CanonicalJsonRecord):
    """Resolved decoder pools and component paths that own run resources."""

    canonical_json: bytes = field(repr=False)


@dataclass(frozen=True)
class EffectiveRuntimeFlags(_CanonicalJsonRecord):
    """Behavior-bearing scalar flags installed on runtime owners."""

    canonical_json: bytes = field(repr=False)


@dataclass(frozen=True)
class SoftwareContextRecord(_CanonicalJsonRecord):
    """Descriptive source-tree and dependency context for the completed run."""

    canonical_json: bytes = field(repr=False)


@dataclass(frozen=True)
class AssuranceStatusRecord(_CanonicalJsonRecord):
    """Independent, deliberately conservative provenance assurances."""

    canonical_json: bytes = field(repr=False)


@dataclass(frozen=True)
class ProviderRecord:
    """Descriptive provenance and assurance for one direct provider."""

    component_path: tuple[TypedPathSegmentRecord, ...]
    provider_kind: str
    module: str
    qualname: str
    source_origin: Optional[str]
    source_sha256: Optional[str]
    first_line_number: Optional[int]
    closure_status: str
    assurance: str


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
    contained_implementations: tuple[ContainedImplementationRecord, ...]
    providers: tuple[ProviderRecord, ...]
    code_selections: tuple[ResolvedCodeSelectionRecord, ...]
    cadences: tuple[ResolvedCadenceRecord, ...]
    aliases: tuple[ResolvedAlias, ...]
    seed_bindings: tuple[ResolvedSeedBinding, ...]
    operations: tuple[ResolvedOperationRecord, ...]
    execution_plan: ResolvedExecutionPlanRecord
    chip_load_plan: ChipLoadPlanRecord
    timing: EffectiveTimingRecord
    links: tuple[ResolvedLinkRecord, ...]
    resources: EffectiveResourceRecord
    runtime_flags: EffectiveRuntimeFlags
    software_context: SoftwareContextRecord
    assurance: AssuranceStatusRecord
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
                    "configuration": _validated_json_value(
                        component.configuration
                    ),
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
            "contained_implementations": [
                {
                    "component_path": _typed_path_json(
                        component.component_path
                    ),
                    "owner_path": _typed_path_json(component.owner_path),
                    "field_path": _typed_path_json(component.field_path),
                    "implementation": component.implementation,
                    "configuration": component.configuration,
                }
                for component in self.contained_implementations
            ],
            "providers": [
                {
                    "component_path": _typed_path_json(
                        record.component_path
                    ),
                    "provider_kind": record.provider_kind,
                    "module": record.module,
                    "qualname": record.qualname,
                    "source_origin": record.source_origin,
                    "source_sha256": record.source_sha256,
                    "first_line_number": record.first_line_number,
                    "closure_status": record.closure_status,
                    "assurance": record.assurance,
                }
                for record in self.providers
            ],
            "code_selections": [
                {
                    "consumer_kind": record.consumer_kind,
                    "consumer_identity": (
                        record.consumer_identity.to_json_value()
                    ),
                    "code_path": _typed_path_json(record.code_path),
                }
                for record in self.code_selections
            ],
            "cadences": [
                {
                    "consumer_kind": record.consumer_kind,
                    "consumer_identity": (
                        record.consumer_identity.to_json_value()
                    ),
                    "code_path": _typed_path_json(record.code_path),
                    "round_ticks": record.round_ticks,
                    "origin": record.origin,
                }
                for record in self.cadences
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
            "operations": [
                operation.to_json_value()
                for operation in self.operations
            ],
            "execution_plan": self.execution_plan.to_json_value(),
            "chip_load_plan": self.chip_load_plan.to_json_value(),
            "timing": self.timing.to_json_value(),
            "links": [
                link.to_json_value()
                for link in self.links
            ],
            "resources": self.resources.to_json_value(),
            "runtime_flags": self.runtime_flags.to_json_value(),
            "software_context": self.software_context.to_json_value(),
            "assurance": self.assurance.to_json_value(),
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
    device: Optional[Any] = None              # default TimingOnlyDevice
    memory_model: Optional[Any] = None        # port 18; default unbounded
    make_controller: Optional[Callable] = None  # (engine) -> Controller (port 14)
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
        self._reject_prebinding_seed_consumers()
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

    def _reject_prebinding_seed_consumers(self) -> None:
        """Reject stochastic owners that would execute before root binding."""
        for field_name in PREBINDING_OBJECT_FIELDS:
            component = object.__getattribute__(self, field_name)
            if component is not None:
                _reject_static_seed_consumer(field_name, component)

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

        if self.router is not None and (
            self.decoder is not None or self.decoders
        ):
            raise ValueError(
                "RunSpec.router is exclusive with decoder and decoders"
            )
        if self.router is not None:
            stored_requirement = inspect.getattr_static(
                self.router,
                "needs_hyperedges",
                None,
            )
            if type(stored_requirement) is not bool:
                raise TypeError(
                    "router does not satisfy DecoderRouter: "
                    "needs_hyperedges must be a stored exact bool"
                )
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
            ("boundary_policy", self.boundary_policy,
             protocols.BoundaryPolicy),
            ("window_interaction", self.window_interaction,
             protocols.WindowInteraction),
            ("idle_policy", self.idle_policy, protocols.IdlePolicy),
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
        _validate_callable_arity(
            "make_orchestrator",
            self.make_orchestrator,
            1,
        )

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
        from . import protocols

        parts = (
            ("resolved code", planning.code, protocols.CodeModel),
            ("resolved layout", planning.layout, protocols.LayoutModel),
            ("resolved scheme", planning.scheme, protocols.DecodingScheme),
            (
                "resolved rounds_policy",
                planning.rounds_policy,
                protocols.RoundsPolicy,
            ),
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

    def _validate_layout_selection(
        self,
        planning: ResolvedPlanningParts,
        operations,
        strategy,
    ) -> _ResolvedCodeCadencePlan:
        code_round_us = planning.code.round_period_us()
        if code_round_us is not None:
            round_us = code_round_us
            cadence_origin = "code.round_period_us"
        elif self.round_us is not None:
            round_us = self.round_us
            cadence_origin = "run_spec.round_us"
        else:
            round_us = self.timing.round_us
            cadence_origin = "timing.round_us"
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

        operation_identity_records = {}
        resolved_operations = []
        patch_records_by_bytes = {}
        for operation in operations:
            selected_code = planning.layout.code_for_op(operation)
            if selected_code is not planning.code:
                raise ValueError(
                    f"layout {planning.layout!r} operation {operation.id} "
                    f"selected {selected_code!r}, but resolved "
                    f"planning/runtime code is {planning.code!r}")
            operation_identity_records.setdefault(
                operation.id,
                StableIdentityRecord.from_identity(operation.id),
            )
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
                patch_record = StableIdentityRecord.from_identity(patch_id)
                patch_records_by_bytes.setdefault(
                    patch_record.canonical_bytes(),
                    (patch_id, patch_record),
                )

        resolved_patches = []
        for patch_id, _patch_record in patch_records_by_bytes.values():
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

        code_path = (TypedPathSegmentRecord("field", "code"),)
        ordered_consumers = [
            ("operation", operation_id, identity_record)
            for operation_id, identity_record
            in operation_identity_records.items()
        ]
        ordered_consumers.sort(
            key=lambda item: item[2].canonical_bytes()
        )
        ordered_patches = [
            ("patch", patch_id, identity_record)
            for patch_id, identity_record in patch_records_by_bytes.values()
        ]
        ordered_patches.sort(
            key=lambda item: item[2].canonical_bytes()
        )
        ordered_consumers.extend(ordered_patches)
        code_selections = tuple(
            ResolvedCodeSelectionRecord(
                consumer_kind=consumer_kind,
                consumer_identity=identity_record,
                code_path=code_path,
            )
            for consumer_kind, _identity, identity_record in ordered_consumers
        )
        cadences = tuple(
            ResolvedCadenceRecord(
                consumer_kind=record.consumer_kind,
                consumer_identity=record.consumer_identity,
                code_path=record.code_path,
                round_ticks=round_ticks,
                origin=cadence_origin,
            )
            for record in code_selections
        )
        return _ResolvedCodeCadencePlan(
            code_geometry=code_geometry,
            spatial_profile=spatial_profile,
            operations=tuple(resolved_operations),
            patches=tuple(resolved_patches),
            code_selections=code_selections,
            cadences=cadences,
            round_ticks_by_operation_id=tuple(
                (operation_id, round_ticks)
                for operation_id in sorted(operation_identity_records)
            ),
            round_ticks_by_patch=tuple(
                (patch_id, round_ticks)
                for _, patch_id, _record in ordered_patches
            ),
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
        provider_records = _resolved_provider_records(self)
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

        controller = self.make_controller(engine) \
            if self.make_controller is not None \
            else ModularController(engine, links=LinkModel.from_timing(self.timing),
                              t_pack=self.timing.ticks("t_pack"))
        from .protocols import (
            Controller,
            MagicStateFactory,
            Metric,
            Orchestrator,
        )
        _validate_protocol_part(
            "orchestrator",
            orchestrator,
            Orchestrator,
        )
        if orchestrator.engine is not engine:
            raise ValueError(
                f"{type(orchestrator).__name__} uses a different engine from "
                "the RunSpec build"
            )
        _validate_protocol_part("controller", controller, Controller)
        # the whole fabric shares the controller's LinkModel: the window
        # manager's dd/do hops ride the same links a custom controller set
        links = controller.links

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
            ws_delay_ticks=links.ws.cost(),
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
        _validate_protocol_part("factory", factory, MagicStateFactory)
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
                _validate_protocol_part(
                    f"make_metrics result {index}", metric, Metric)
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
                if (
                    type(metric.result_schema_version) is not int
                    or metric.result_schema_version < 1
                ):
                    raise TypeError(
                        f"metric {index} result_schema_version must be a "
                        "positive built-in int"
                    )
            metric_bindings = tuple(
                _ResolvedMetricBinding(
                    metric=metric,
                    name=metric.name,
                    result_schema_version=metric.result_schema_version,
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
            links=links,
            metrics=metric_bindings,
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
        _validate_metric_component_configurations(
            component_graph, metric_bindings
        )
        seed_plan = component_graph.seed_plan
        resolved_components = _resolved_components(component_graph)
        reservations = _bind_run_seed_plan(seed_plan)
        seed_bindings = tuple(
            ResolvedSeedBinding(
                component_path=_typed_path_records(entry.component_path),
                seed_source=reservation.proposed_seed_source,
                seed=reservation.proposed_seed,
            )
            for entry, reservation in zip(seed_plan, reservations)
        )
        fixed_composition = _resolved_fixed_composition(
            fixed_composition_anchors
        )
        contained_implementations = _resolved_contained_implementations(
            window_manager,
            controller,
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
                window_manager._register_dynamic_stream(
                    stream,
                    resolved_operation_by_id[stream.id],
                )
            for binding in metric_bindings:
                _validate_live_metric_binding(binding)
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
            )
            _validate_shipped_component_configuration(component_graph)
            operation_records = _resolved_operation_records(
                workload,
                device=device,
                resolved_rounds_by_operation_id=(
                    resolved_rounds_by_operation_id
                ),
            )
            execution_plan_record = _resolved_execution_plan_record(
                execution_plan_spec,
                dynamic_streams=dynamic_streams,
                resolved_rounds_by_operation_id=(
                    resolved_rounds_by_operation_id
                ),
                code_geometry=code_cadence_plan.code_geometry,
            )
            chip_load_plan_record = _chip_load_plan_record(
                workload,
                resolved_rounds_by_operation_id=(
                    resolved_rounds_by_operation_id
                ),
                resource_claims_by_operation_id=(
                    resource_claims_by_operation_id
                ),
                resolved_patches=code_cadence_plan.patches,
            )
            timing_record = _effective_timing_record(
                self,
                controller=controller,
            )
            link_records = _resolved_link_records(
                links,
                value_origin=(
                    "timing"
                    if self.make_controller is None
                    else "controller"
                ),
            )
            resource_record = _effective_resource_record(
                pool,
                memory_model_present=self.memory_model is not None,
            )
            runtime_flags = _effective_runtime_flags(
                engine=engine,
                window_manager=window_manager,
                pool=pool,
                chip=gate,
            )
            software_context, source_tree_status = (
                _software_context_record(resolved_components)
            )
            assurance = _assurance_status_record(
                root_seed=self._validated_root_seed(),
                resolved_components=resolved_components,
                seed_bindings=seed_bindings,
                source_tree_status=source_tree_status,
                workload=workload,
                device=device,
            )
            manifest = ResolvedRunManifest(
                schema_version=2,
                root_seed=self._validated_root_seed(),
                components=resolved_components,
                fixed_composition=fixed_composition,
                contained_implementations=contained_implementations,
                providers=provider_records,
                code_selections=code_cadence_plan.code_selections,
                cadences=code_cadence_plan.cadences,
                aliases=resolved_aliases,
                seed_bindings=seed_bindings,
                operations=operation_records,
                execution_plan=execution_plan_record,
                chip_load_plan=chip_load_plan_record,
                timing=timing_record,
                links=link_records,
                resources=resource_record,
                runtime_flags=runtime_flags,
                software_context=software_context,
                assurance=assurance,
                primary_result_sha256=hashlib.sha256(
                    result.canonical_json_bytes(),
                ).hexdigest(),
            )
            engine._complete()
        except BaseException as error:
            engine._invalidate(error)
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
        for field_name in PREBINDING_PROVIDER_FIELDS:
            provider = object.__getattribute__(self, field_name)
            if type(provider) is types.MethodType:
                roots.append(
                    ((field_segment(field_name),), provider.__self__)
                )
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


def _is_shipped_component(component) -> bool:
    source_path = inspect.getsourcefile(type(component))
    if source_path is None:
        return False
    try:
        Path(source_path).resolve().relative_to(
            Path(__file__).resolve().parent
        )
    except ValueError:
        return False
    return True


def _capture_component_configuration(component) -> tuple[Optional[bytes], str]:
    from .protocols import RunManifestPart

    if not isinstance(component, RunManifestPart):
        return None, "opaque"
    candidate = component.run_manifest_config()
    if type(candidate) is not dict:
        raise TypeError(
            f"{type(component).__name__}.run_manifest_config() must return "
            "an exact built-in dict"
        )
    copied = _validated_json_value(candidate)
    canonical_json = json.dumps(
        copied,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    status = "declared" if _is_shipped_component(component) else "opaque"
    return canonical_json, status


def _validate_shipped_component_configuration(
    graph: _ResolvedComponentGraph,
) -> None:
    for entry in graph.components:
        if entry.configuration_status != "declared":
            continue
        current_json, current_status = _capture_component_configuration(
            entry.component
        )
        if (
            current_status != "declared"
            or current_json != entry.configuration_json
        ):
            raise RuntimeError(
                f"{type(entry.component).__name__} changed declared "
                "configuration during the run"
            )


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
    configurations = {
        "engine": {
            "kind": "engine",
            "construction_guarded": True,
            "log_sink": "none",
        },
        "payload_store": {"kind": "payload_store"},
        "window_manager": {"kind": "window_manager"},
        "decoder_manager": None,
        "strategy_services": {"kind": "strategy_services"},
        "cluster": {"kind": "cluster"},
        "clocked_device": {"kind": "clocked_device"},
        "chip": {"kind": "chip"},
    }
    records = [
        FixedCompositionRecord(
            component_path=_typed_path_records(component_path),
            implementation=_implementation_name(component),
            configuration=_fixed_composition_configuration(
                component_path[-1].value,
                component,
                configurations,
            ),
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


def _fixed_composition_configuration(name, component, configurations):
    if name == "decoder_manager":
        if component.lane_policy is not None:
            raise ValueError(
                "RunSpec DecoderManager lane_policy must be None"
            )
        return {
            "kind": "decoder_manager",
            "weak_strong_delay_ticks": component.ws_delay_ticks,
            "log_name": component.log_name,
            "lane_policy": "none",
        }
    try:
        configuration = configurations[name]
    except KeyError as error:
        raise ValueError(
            f"unknown fixed composition anchor {name!r}"
        ) from error
    return _validated_json_value(configuration)


def _resolved_contained_implementations(
    window_manager,
    controller,
) -> tuple[ContainedImplementationRecord, ...]:
    field = lambda value: TypedPathSegmentRecord("field", value)
    records = [
        ContainedImplementationRecord(
            component_path=(
                field("fixed"),
                field("window_manager"),
                field("contained"),
                field("lifecycle"),
            ),
            owner_path=(field("fixed"), field("window_manager")),
            field_path=(field("lifecycle"),),
            implementation=_implementation_name(window_manager.lifecycle),
            configuration={"kind": "dynamic_windows"},
        ),
        ContainedImplementationRecord(
            component_path=(
                field("fixed"),
                field("window_manager"),
                field("contained"),
                field("speculative_recovery"),
            ),
            owner_path=(field("fixed"), field("window_manager")),
            field_path=(field("speculative_recovery"),),
            implementation=_implementation_name(
                window_manager.speculative_recovery
            ),
            configuration={"kind": "speculative_recovery"},
        ),
    ]
    from .controllers import ModularController
    if type(controller) is ModularController:
        records.append(
            ContainedImplementationRecord(
                component_path=(
                    field("controller"),
                    field("contained"),
                    field("links"),
                ),
                owner_path=(field("controller"),),
                field_path=(field("links"),),
                implementation=_implementation_name(controller.links),
                configuration={"kind": "link_model"},
            )
        )
    records.sort(
        key=lambda record: b"".join(
            _public_path_segment_bytes(segment)
            for segment in record.component_path
        )
    )
    return tuple(records)


def _identity_json(identity) -> dict:
    return StableIdentityRecord.from_identity(identity).to_json_value()


def _optional_identity_json(identity):
    return None if identity is None else _identity_json(identity)


def _circuit_provenance_json(circuit, circuit_scope: str) -> dict:
    if circuit is None:
        return {
            "kind": "none",
            "format": None,
            "canonical_text_sha256": None,
            "stim_version": None,
        }
    try:
        import stim
    except ImportError:
        stim = None
    if stim is not None and type(circuit) is stim.Circuit:
        canonical_text = str(circuit).encode("utf-8")
        return {
            "kind": "stim_circuit_text",
            "format": "stim_circuit_text",
            "canonical_text_sha256": hashlib.sha256(
                canonical_text
            ).hexdigest(),
            "stim_version": stim.__version__,
        }
    if circuit_scope != "none":
        raise TypeError(
            "an active operation circuit must be an exact stim.Circuit"
        )
    return {
        "kind": "opaque_dormant",
        "format": None,
        "canonical_text_sha256": None,
        "stim_version": None,
    }


def _intrinsic_measurement_json(measurement):
    if measurement is None:
        return None
    return {
        "operation_id": _identity_json(measurement.operation_id),
        "trajectory_id": _identity_json(measurement.trajectory_id),
        "value": measurement.value,
        "source": measurement.source,
    }


def _resolved_operation_records(
    workload: _RunOwnedWorkload,
    *,
    device,
    resolved_rounds_by_operation_id,
) -> tuple[ResolvedOperationRecord, ...]:
    private_by_id = {}
    for operation in (
        workload.executable_operations
        + workload.static_decode_operations
        + workload.dynamic_stream_operations
    ):
        private_by_id.setdefault(operation.id, operation)

    records = []
    for operation_id in sorted(private_by_id):
        operation = private_by_id[operation_id]
        declared = workload.declared_view_by_operation_id[operation_id]
        initial_offset = declared.stream_offset
        final_offset = operation.stream_offset
        if declared.stream_id is None:
            offset_resolution = "not_applicable"
            initial_offset = None
            final_offset = None
        elif initial_offset is not None:
            offset_resolution = "declared"
        elif final_offset is not None:
            offset_resolution = "runtime_reserved"
        else:
            offset_resolution = "resolved_at_load"

        try:
            resolved_rounds = resolved_rounds_by_operation_id[operation_id]
        except KeyError as error:
            raise ValueError(
                f"operation {operation_id} has no resolved round count"
            ) from error
        record = {
            "operation_id": _identity_json(operation_id),
            "name": declared.name,
            "qubits": [
                _identity_json(identity)
                for identity in declared.qubits
            ],
            "clifford": declared.clifford,
            "circuit": _circuit_provenance_json(
                workload.source_circuit_by_operation_id[operation_id],
                device.operation_circuit_scope,
            ),
            "consumes_magic_state": declared.consumes_magic_state,
            "patches": [
                _identity_json(identity)
                for identity in declared.patches
            ],
            "predecessors": [
                _identity_json(identity)
                for identity in declared.predecessors
            ],
            "has_successor": declared.has_successor,
            "stream_id": _optional_identity_json(declared.stream_id),
            "stream_offset": declared.stream_offset,
            "blocked_by": _optional_identity_json(declared.blocked_by),
            "feedback_boundary_mode": declared.feedback_boundary_mode,
            "requires_result_return_to_chip": (
                declared.requires_result_return_to_chip
            ),
            "requires_strong_commit": declared.requires_strong_commit,
            "byproduct_pauli": declared.byproduct_pauli,
            "measurement_basis": declared.measurement_basis,
            "logical_observable_index": (
                declared.logical_observable_index
            ),
            "intrinsic_measurement": _intrinsic_measurement_json(
                declared.intrinsic_measurement
            ),
            "kind": declared.kind.name,
            "memberships": [
                {
                    "role": role,
                    "collection_index": collection_index,
                }
                for role, collection_index in (
                    workload.memberships_by_operation_id[operation_id]
                )
            ],
            "effective_needs_magic_state": operation.needs_magic_state,
            "effective_feedback_boundary_mode": (
                operation.feedback_boundary_mode
            ),
            "resolved_rounds": resolved_rounds,
            "stream_offset_resolution": offset_resolution,
            "initial_resolved_stream_offset": initial_offset,
            "final_resolved_stream_offset": final_offset,
        }
        records.append(ResolvedOperationRecord.freeze(record))
    return tuple(records)


def _window_key_json(key) -> dict:
    return {
        "operation_id": _identity_json(key[0]),
        "window_index": key[1],
    }


def _resource_claim_json(claim) -> dict:
    identities = [
        (StableIdentityRecord.from_identity(identity), identity)
        for identity in claim.ids
    ]
    identities.sort(key=lambda item: item[0].canonical_bytes())
    return {
        "kind": claim.kind,
        "ids": [
            record.to_json_value()
            for record, _identity in identities
        ],
    }


def _resolved_execution_plan_record(
    spec: _ResolvedExecutionPlanSpec,
    *,
    dynamic_streams,
    resolved_rounds_by_operation_id,
    code_geometry,
) -> ResolvedExecutionPlanRecord:
    successors = dict(spec.successors)
    spatial_nodes = dict(spec.spatial_nodes)
    rounds = dict(spec.rounds_by_operation)
    operation_windows = dict(spec.op_windows)
    windows_by_key = {window.key: window for window in spec.windows}
    windowed_by_operation = dict(spec.windowed_by_operation)
    batch_idle_by_operation = dict(
        spec.batch_preceding_idle_rounds_by_operation
    )
    operation_plans = []
    for operation_id in spec.planned_operation_ids:
        indices = operation_windows[operation_id]
        internal_dependencies = []
        internal_sources = set()
        internal_destinations = set()
        for destination_index in indices:
            window = windows_by_key[(operation_id, destination_index)]
            for dependency_operation_id, source_index in window.deps:
                if dependency_operation_id != operation_id:
                    continue
                internal_dependencies.append(
                    [source_index, destination_index]
                )
                internal_sources.add(source_index)
                internal_destinations.add(destination_index)
        operation_plans.append({
            "operation_id": _identity_json(operation_id),
            "round_count": rounds[operation_id],
            "spatial_node_count": spatial_nodes[operation_id],
            "window_indices": list(indices),
            "internal_dependencies": internal_dependencies,
            "entry_window_indices": [
                index for index in indices
                if index not in internal_destinations
            ],
            "exit_window_indices": [
                index for index in indices
                if index not in internal_sources
            ],
            "windowed": windowed_by_operation[operation_id],
            "batch_preceding_idle_rounds": (
                batch_idle_by_operation[operation_id]
            ),
        })
    record = {
        "code_geometry": {
            "code_name": code_geometry.code_name,
            "distance": code_geometry.distance,
            "commit_round_count": code_geometry.commit_round_count,
            "buffer_round_count": code_geometry.buffer_round_count,
            "minimum_leading_buffer_round_count": (
                code_geometry.minimum_leading_buffer_round_count
            ),
            "minimum_trailing_buffer_round_count": (
                code_geometry.minimum_trailing_buffer_round_count
            ),
            "one_patch_spatial_node_count": (
                code_geometry.one_patch_spatial_node_count
            ),
            "buffer_floor_override_active": (
                code_geometry.buffer_floor_override_active
            ),
        },
        "planned_operation_ids": [
            _identity_json(operation_id)
            for operation_id in spec.planned_operation_ids
        ],
        "operation_plans": operation_plans,
        "windows": [
            {
                "key": _window_key_json(window.key),
                "buffer_lo": window.start_round,
                "commit_lo": window.commit_lo,
                "commit_hi": window.commit_hi,
                "buffer_hi": window.buffer_hi,
                "n_rounds": window.n_rounds,
                "dependencies": [
                    _window_key_json(key)
                    for key in sorted(window.deps)
                ],
                "dependents": [
                    _window_key_json(key)
                    for key in sorted(window.dependents)
                ],
            }
            for window in spec.windows
        ],
        "successors": [
            {
                "operation_id": _identity_json(operation_id),
                "successor_ids": [
                    _identity_json(successor_id)
                    for successor_id in sorted(successor_ids)
                ],
            }
            for operation_id, successor_ids in sorted(successors.items())
        ],
        "total_windows": spec.total_windows,
        "dynamic_streams": [
            {
                "operation_id": _identity_json(operation.id),
                "stream_id": _identity_json(operation.id),
                "initial_offset": operation.stream_offset,
                "declared_rounds": (
                    resolved_rounds_by_operation_id[operation.id]
                ),
                "registration_index": index,
                "feedback_boundary_mode": (
                    operation.feedback_boundary_mode
                ),
                "rounds_policy_path": [
                    {"kind": "field", "value": "rounds_policy"},
                ],
            }
            for index, operation in enumerate(dynamic_streams)
        ],
    }
    return ResolvedExecutionPlanRecord.freeze(record)


def _chip_load_plan_record(
    workload: _RunOwnedWorkload,
    *,
    resolved_rounds_by_operation_id,
    resource_claims_by_operation_id,
    resolved_patches,
) -> ChipLoadPlanRecord:
    entries = []
    shot_owner_by_key = {}
    for operation in workload.executable_operations:
        declared = workload.declared_view_by_operation_id[operation.id]
        if declared.stream_id is None:
            offset_resolution = "not_applicable"
            initial_offset = None
            final_offset = None
        elif declared.stream_offset is not None:
            offset_resolution = "declared"
            initial_offset = declared.stream_offset
            final_offset = operation.stream_offset
        else:
            offset_resolution = "runtime_reserved"
            initial_offset = None
            final_offset = operation.stream_offset
        entries.append({
            "operation_id": _identity_json(operation.id),
            "resolved_rounds": (
                resolved_rounds_by_operation_id[operation.id]
            ),
            "resource_claims": [
                _resource_claim_json(claim)
                for claim in resource_claims_by_operation_id[operation.id]
            ],
            "stream_offset_resolution": offset_resolution,
            "initial_resolved_stream_offset": initial_offset,
            "final_resolved_stream_offset": final_offset,
        })
        shot_key = (
            operation.stream_id
            if operation.stream_id is not None
            else operation.id
        )
        shot_owner_by_key.setdefault(shot_key, operation.id)
    shot_owners = [
        (
            StableIdentityRecord.from_identity(shot_key),
            owner_operation_id,
        )
        for shot_key, owner_operation_id in shot_owner_by_key.items()
    ]
    shot_owners.sort(key=lambda item: item[0].canonical_bytes())
    patch_spatial_geometry = [
        (
            StableIdentityRecord.from_identity(patch.patch_identity),
            patch.spatial_node_count,
        )
        for patch in resolved_patches
    ]
    patch_spatial_geometry.sort(
        key=lambda item: item[0].canonical_bytes()
    )
    return ChipLoadPlanRecord.freeze({
        "entries": entries,
        "shot_owners": [
            {
                "shot_key": shot_key.to_json_value(),
                "owner_operation_id": _identity_json(owner_operation_id),
            }
            for shot_key, owner_operation_id in shot_owners
        ],
        "patch_spatial_geometry": [
            {
                "patch_identity": patch_identity.to_json_value(),
                "spatial_node_count": spatial_node_count,
            }
            for patch_identity, spatial_node_count
            in patch_spatial_geometry
        ],
    })


def _effective_timing_record(
    spec: RunSpec,
    *,
    controller,
) -> EffectiveTimingRecord:
    t_pack_ticks = getattr(
        controller,
        "t_pack",
        spec.timing.ticks("t_pack"),
    )
    if type(t_pack_ticks) is not int or t_pack_ticks < 0:
        raise TypeError(
            "resolved controller t_pack must be a nonnegative exact int"
        )
    return EffectiveTimingRecord.freeze({
        "ticks_per_us": TICKS_PER_US,
        "t_pack_ticks": t_pack_ticks,
        "t_pack_us": t_pack_ticks / TICKS_PER_US,
    })


def _resolved_link_records(
    links,
    *,
    value_origin: str,
) -> tuple[ResolvedLinkRecord, ...]:
    records = []
    for name in ("qc", "cd", "dd", "do", "oc", "cq", "ws"):
        link = getattr(links, name)
        records.append(ResolvedLinkRecord.freeze({
            "name": name,
            "latency_ticks": link.latency_ticks,
            "latency_us": link.latency_ticks / TICKS_PER_US,
            "bandwidth_bits_per_us": link.bandwidth_bits_per_us,
            "serialize": link.serialize,
            "value_origin": value_origin,
        }))
    return tuple(records)


def _component_path_json(*parts: str) -> list[dict]:
    return [
        {"kind": "field", "value": part}
        for part in parts
    ]


def _effective_resource_record(
    pool,
    *,
    memory_model_present: bool,
) -> EffectiveResourceRecord:
    return EffectiveResourceRecord.freeze({
        "code_path": _component_path_json("code"),
        "unit_pools": [
            {"name": name, "units": units}
            for name, units in sorted(pool.unit_totals.items())
        ],
        "memory_model_path": (
            _component_path_json("memory_model")
            if memory_model_present
            else None
        ),
        "device_path": _component_path_json("device"),
        "controller_path": _component_path_json("controller"),
        "orchestrator_path": _component_path_json("orchestrator"),
        "factory_path": _component_path_json("magic_state_factory"),
    })


def _effective_runtime_flags(
    *,
    engine,
    window_manager,
    pool,
    chip,
) -> EffectiveRuntimeFlags:
    return EffectiveRuntimeFlags.freeze({
        "feedback_boundary_mode": (
            window_manager.feedback_boundary_mode
        ),
        "gates_start_on_round_boundaries": (
            chip.gates_start_on_round_boundaries
        ),
        "max_idle_rounds": chip.max_idle_rounds,
        "decoder_bulk_strong": pool.bulk_strong,
        "decoder_needs_hyperedges": window_manager.needs_hyperedges,
        "switching_active": window_manager.switching_active,
        "verbose": engine.verbose,
    })


def _origin_record(module_name: str, origin) -> dict:
    if origin is None:
        return {
            "module_name": module_name,
            "origin_kind": "unknown",
            "origin": None,
        }
    if origin == "built-in":
        kind = "built_in"
    elif origin == "frozen":
        kind = "frozen"
    elif type(origin) is str:
        suffix = Path(origin).suffix.lower()
        if suffix in (".so", ".pyd", ".dll", ".dylib"):
            kind = "native_extension"
        elif suffix in (".py", ".pyw"):
            kind = "python_file"
        elif suffix:
            kind = "loader"
        else:
            kind = "unknown"
    else:
        kind = "unknown"
        origin = None
    return {
        "module_name": module_name,
        "origin_kind": kind,
        "origin": origin,
    }


def _source_tree_snapshot():
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256(b"decsim.code-identity.v1")
    included_paths = set()
    complete = True
    try:
        candidates = sorted(
            package_root.rglob("*"),
            key=lambda path: path.relative_to(package_root).as_posix(),
        )
    except OSError:
        return None, set(), False
    for path in candidates:
        relative = path.relative_to(package_root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(package_root)
        except (OSError, ValueError):
            complete = False
            continue
        if not resolved.is_file():
            continue
        try:
            content = resolved.read_bytes()
        except OSError:
            complete = False
            continue
        relative_text = relative.as_posix()
        encoded_path = relative_text.encode("utf-8")
        digest.update(b"P")
        digest.update(len(encoded_path).to_bytes(4, "big"))
        digest.update(encoded_path)
        digest.update(b"B")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        included_paths.add(resolved)
    return digest.hexdigest(), included_paths, complete


def _loaded_decsim_module_records(package_root, included_paths):
    records = []
    complete = True
    for module_name in sorted(
        name
        for name in sys.modules
        if name == "decsim" or name.startswith("decsim.")
    ):
        module = sys.modules[module_name]
        module_spec = getattr(module, "__spec__", None)
        origin = (
            None
            if module_spec is None
            else getattr(module_spec, "origin", None)
        )
        record = _origin_record(module_name, origin)
        records.append(record)
        if type(origin) is not str or origin in ("built-in", "frozen"):
            complete = False
            continue
        try:
            resolved = Path(origin).resolve(strict=True)
            resolved.relative_to(package_root)
        except (OSError, ValueError):
            complete = False
            continue
        if resolved.suffix == ".pyc":
            source_candidate = Path(importlib.util.source_from_cache(str(resolved)))
            if source_candidate in included_paths:
                record["origin_kind"] = "python_file"
                record["origin"] = str(source_candidate)
                continue
        if resolved not in included_paths:
            complete = False
    return records, complete


def _distribution_details(module_name):
    distribution_name = {
        "stim": "stim",
        "numpy": "numpy",
        "pymatching": "pymatching",
        "ldpc": "ldpc",
        "scipy": "scipy",
    }.get(module_name)
    if distribution_name is None:
        return None, None
    try:
        version = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return distribution_name, version


def _external_dependency_record(module_name):
    module_spec = importlib.util.find_spec(module_name)
    if module_spec is None:
        origin = None
        availability = "unavailable"
    else:
        origin = module_spec.origin
        availability = "available"
    distribution_name, distribution_version = _distribution_details(
        module_name
    )
    origin_fields = _origin_record(module_name, origin)
    return {
        "module_name": module_name,
        "distribution_name": distribution_name,
        "distribution_version": distribution_version,
        "availability": availability,
        "origin_kind": origin_fields["origin_kind"],
        "origin": origin_fields["origin"],
    }


def _selected_external_modules(resolved_components) -> tuple[str, ...]:
    implementation_text = "\n".join(
        component.implementation
        for component in resolved_components
    ).lower()
    selected = set()
    if "stim" in implementation_text:
        selected.update(("stim", "numpy"))
    if "pymatching" in implementation_text or "mwpm" in implementation_text:
        selected.update(("pymatching", "numpy"))
    if "bposd" in implementation_text or "belief" in implementation_text:
        selected.update(("ldpc", "numpy", "scipy"))
    return tuple(sorted(selected))


def _software_context_record(
    resolved_components,
) -> tuple[SoftwareContextRecord, str]:
    package_root = Path(__file__).resolve().parent
    tree_digest, included_paths, tree_complete = _source_tree_snapshot()
    loaded_modules, loaded_complete = _loaded_decsim_module_records(
        package_root,
        included_paths,
    )
    try:
        distribution_version = importlib.metadata.version("decsim")
    except importlib.metadata.PackageNotFoundError:
        distribution_version = None
    external_dependencies = [
        _external_dependency_record(module_name)
        for module_name in _selected_external_modules(resolved_components)
    ]
    status = (
        "declared"
        if tree_complete and loaded_complete and tree_digest is not None
        else "partial"
    )
    return SoftwareContextRecord.freeze({
        "decsim_distribution_version": distribution_version,
        "decsim_source_tree_sha256": tree_digest,
        "python_version": sys.version,
        "loaded_decsim_modules": loaded_modules,
        "external_dependencies": external_dependencies,
    }), status


def _assurance_status_record(
    *,
    root_seed,
    resolved_components,
    seed_bindings,
    source_tree_status,
    workload,
    device,
) -> AssuranceStatusRecord:
    configuration_declared = all(
        component.configuration_status == "declared"
        for component in resolved_components
    )
    seed_coverage = (
        "not_applicable"
        if not seed_bindings
        else (
            "declared"
            if all(
                component.configuration_status == "declared"
                for component in resolved_components
            )
            else "partial"
        )
    )
    workload_declared = all(
        _circuit_provenance_json(
            circuit,
            device.operation_circuit_scope,
        )["kind"] != "opaque_dormant"
        for circuit in workload.source_circuit_by_operation_id.values()
    )
    seed_replay_scope = (
        "partial"
        if seed_coverage == "partial"
        else (
            "root_seeded"
            if root_seed is not None
            else "local_or_entropy"
        )
    )
    return AssuranceStatusRecord.freeze({
        "component_configuration_status": (
            "declared" if configuration_declared else "partial"
        ),
        "seed_coverage_status": seed_coverage,
        "workload_provenance_status": (
            "declared" if workload_declared else "partial"
        ),
        "source_tree_snapshot_status": source_tree_status,
        "executed_software_status": "unattested",
        "seed_replay_scope": seed_replay_scope,
        "reproducibility_scope": "partial",
    })


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
            configuration=(
                None
                if entry.configuration_json is None
                else json.loads(entry.configuration_json)
            ),
            configuration_status=entry.configuration_status,
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
        configuration_json, configuration_status = (
            _capture_component_configuration(component)
        )
        components.append(
            _ComponentGraphEntry(
                component_path=component_path,
                parent_path=parent_path,
                edge_path=edge_path,
                component=component,
                configuration_json=configuration_json,
                configuration_status=configuration_status,
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

    Static inspection verifies the declared surface without executing
    descriptors. Binding a representative call to each declared method
    additionally catches duck-typed methods whose signatures cannot accept
    the runtime contract.
    """
    if part is None:
        return
    missing = object()
    data_names = {
        attribute_name
        for protocol_base in protocol.__mro__
        for attribute_name in getattr(
            protocol_base,
            "__annotations__",
            {},
        )
        if not attribute_name.startswith("_")
    }
    method_names = [
        method_name
        for method_name, required_method in protocol.__dict__.items()
        if not method_name.startswith("_") and callable(required_method)
    ]
    required_names = data_names | set(method_names)
    if any(
        inspect.getattr_static(part, required_name, missing) is missing
        for required_name in required_names
    ):
        raise TypeError(
            f"{name} must implement {protocol.__name__}; required attributes "
            f"are missing or not callable")
    for method_name in method_names:
        if not callable(getattr(part, method_name)):
            raise TypeError(
                f"{name} must implement {protocol.__name__}; required "
                f"method {method_name} is not callable"
            )
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
        raise TypeError(
            f"{component_path} has unsupported provider shape "
            f"{provider_type.__name__}"
        )
    raise TypeError(f"{component_path} must be callable")


def _provider_source_details(target):
    source_origin = inspect.getsourcefile(target)
    try:
        source_text, first_line_number = inspect.getsourcelines(target)
    except (OSError, TypeError):
        source_sha256 = None
        first_line_number = None
    else:
        source_sha256 = hashlib.sha256(
            "".join(source_text).encode("utf-8")
        ).hexdigest()
    return source_origin, source_sha256, first_line_number


def _provider_source_is_shipped(target) -> bool:
    source_path = inspect.getsourcefile(target)
    if source_path is None:
        return False
    try:
        Path(source_path).resolve().relative_to(
            Path(__file__).resolve().parent
        )
    except ValueError:
        return False
    return True


def _resolved_provider_records(spec: RunSpec) -> tuple[ProviderRecord, ...]:
    records = []
    for field_name in PREBINDING_PROVIDER_FIELDS:
        provider = object.__getattribute__(spec, field_name)
        if provider is None:
            continue
        provider_type = type(provider)
        if provider_type is types.FunctionType:
            provider_kind = "function"
            target = provider
            closure_status = (
                "present" if provider.__closure__ else "none"
            )
        elif provider_type is types.MethodType:
            provider_kind = "bound_method"
            target = provider.__func__
            closure_status = (
                "present" if target.__closure__ else "none"
            )
        elif provider_type is type:
            provider_kind = "class"
            target = provider
            closure_status = "not_applicable"
        else:
            raise TypeError(
                f"{field_name} has unsupported provider shape "
                f"{provider_type.__name__}"
            )

        module = getattr(target, "__module__", None)
        qualname = getattr(target, "__qualname__", None)
        if not is_stable_string(module) or not is_stable_string(qualname):
            raise TypeError(
                f"{field_name} provider module and qualname must be Unicode "
                "scalar strings"
            )
        source_origin, source_sha256, first_line_number = (
            _provider_source_details(target)
        )
        if source_origin is not None and not is_stable_string(source_origin):
            raise TypeError(
                f"{field_name} provider source origin must be a Unicode "
                "scalar string"
            )
        # A bound receiver can depend on state outside its declared
        # configuration. Keep that provider's assurance conservative even
        # though the receiver is separately recorded in the component graph.
        receiver_is_represented = provider_kind != "bound_method"
        if _provider_source_is_shipped(target) and receiver_is_represented:
            assurance = "covered_repository_source"
        elif (
            provider_kind == "function"
            and closure_status == "none"
            and target.__name__ != "<lambda>"
            and "<locals>" not in qualname
        ):
            assurance = "external_named_no_closure"
        else:
            assurance = "partial_unattested_callable_state"
        records.append(
            ProviderRecord(
                component_path=(
                    TypedPathSegmentRecord("field", field_name),
                ),
                provider_kind=provider_kind,
                module=module,
                qualname=qualname,
                source_origin=source_origin,
                source_sha256=source_sha256,
                first_line_number=first_line_number,
                closure_status=closure_status,
                assurance=assurance,
            )
        )
    return tuple(records)


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
    metric_bindings,
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
    for binding in metric_bindings:
        _validate_live_metric_binding(binding)
        value = _validated_json_value(engine._invoke_metric_callback(
            binding.metric.result, callback_kind="result"
        ))
        _validate_live_metric_binding(binding)
        canonical_value_json = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        metric_results.append(
            MetricResultRecord(
                name=binding.name,
                result_schema_version=binding.result_schema_version,
                canonical_value_json=canonical_value_json,
            )
        )

    if engine._event_queue:
        raise RuntimeError("primary run ended with pending engine events")
    if not gate.workload_complete:
        raise RuntimeError("primary run ended before the chip workload completed")
    return PrimaryRunResult(
        schema_version=2,
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


def _validate_live_metric_binding(binding: _ResolvedMetricBinding) -> None:
    current_name = binding.metric.name
    current_version = binding.metric.result_schema_version
    if (
        not is_stable_string(current_name)
        or current_name != binding.name
        or type(current_version) is not int
        or current_version != binding.result_schema_version
    ):
        raise RuntimeError(
            f"metric {binding.name!r} changed its frozen result identity"
        )


def _validate_metric_component_configurations(
    graph: _ResolvedComponentGraph,
    bindings: tuple[_ResolvedMetricBinding, ...],
) -> None:
    entries = {id(entry.component): entry for entry in graph.components}
    for binding in bindings:
        entry = entries.get(id(binding.metric))
        if entry is None:
            raise RuntimeError(f"metric {binding.name!r} is absent from component graph")
        if entry.configuration_json is None:
            continue
        configuration = json.loads(entry.configuration_json)
        manifest_version = configuration.get("result_schema_version")
        if (
            type(manifest_version) is not int
            or manifest_version != binding.result_schema_version
        ):
            raise RuntimeError(
                f"metric {binding.name!r} manifest result schema disagrees "
                "with its frozen binding"
            )


def _validated_json_value(value):
    """Copy one value from the closed manifest JSON domain."""
    active_container_ids = set()

    def copy_validated(candidate):
        value_type = type(candidate)
        if candidate is None or value_type in (bool, int):
            return candidate
        if value_type is str:
            if not is_stable_string(candidate):
                raise TypeError(
                    "manifest strings must contain only Unicode scalar values"
                )
            return candidate
        if value_type is float:
            if not math.isfinite(candidate):
                raise ValueError(
                    f"manifest floats must be finite; got {candidate!r}"
                )
            return candidate
        if value_type not in (list, dict):
            raise TypeError(
                "manifest values must use the closed JSON domain; "
                f"got {value_type.__name__}"
            )
        candidate_id = id(candidate)
        if candidate_id in active_container_ids:
            raise ValueError("manifest values cannot contain recursive containers")
        active_container_ids.add(candidate_id)
        try:
            if value_type is list:
                return [copy_validated(item) for item in candidate]
            copied = {}
            for key, item in candidate.items():
                if not is_stable_string(key):
                    raise TypeError(
                        "manifest object keys must be Unicode scalar strings; "
                        f"got {key!r}"
                    )
                copied[key] = copy_validated(item)
            return copied
        finally:
            active_container_ids.remove(candidate_id)

    return copy_validated(value)


def simulate(run: RunSpec, verbose: bool = False) -> CompletedRun:
    """Execute and return the same completed aggregate as RunSpec.build()."""
    return run.build(verbose=verbose)
