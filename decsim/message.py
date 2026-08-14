"""The typed messages the simulator's modules pass to each other.

"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import math
from numbers import Real
from typing import Any, Callable, Optional

def is_stable_string(value: Any) -> bool:
    return (
        type(value) is str
        and all(
            not 0xD800 <= ord(character) <= 0xDFFF
            for character in value
        )
    )


def is_stable_identity(value: Any) -> bool:
    value_type = type(value)
    if value_type is int:
        return True
    if value_type is str:
        return is_stable_string(value)
    if value_type is tuple:
        return all(is_stable_identity(item) for item in value)
    return False


def same_stable_identity(left: Any, right: Any) -> bool:
    """Compare stable identities without Python's cross-type equality."""
    if type(left) is not type(right):
        return False
    if type(left) is tuple:
        return (
            len(left) == len(right)
            and all(
                same_stable_identity(left_item, right_item)
                for left_item, right_item in zip(left, right)
            )
        )
    if type(left) is int or type(left) is str:
        return left == right
    return False


def stable_identity_bytes(identity: Any) -> bytes:
    if not is_stable_identity(identity):
        raise TypeError(
            "stable identities are exact int, Unicode scalar str, or "
            "recursive tuples"
        )
    if type(identity) is int:
        encoded = str(identity).encode("ascii")
        return b"I" + len(encoded).to_bytes(8, "big") + encoded
    if type(identity) is str:
        encoded = identity.encode("utf-8")
        return b"S" + len(encoded).to_bytes(8, "big") + encoded
    encoded_items = tuple(stable_identity_bytes(item) for item in identity)
    return (
        b"T"
        + len(encoded_items).to_bytes(8, "big")
        + b"".join(
            len(item).to_bytes(8, "big") + item
            for item in encoded_items
        )
    )


def stable_identity_order_key(identity: Any) -> bytes:
    return stable_identity_bytes(identity)


def stable_identity_json(identity: Any) -> dict:
    if type(identity) is int:
        return {"kind": "integer", "value": str(identity), "items": None}
    if type(identity) is str:
        return {"kind": "string", "value": identity, "items": None}
    items = [stable_identity_json(item) for item in identity]
    return {"kind": "tuple", "value": None, "items": items}


@dataclass(frozen=True)
class RunSeedPathSegment:
    """One framed semantic edge in the run-level seed component graph."""

    kind: str
    value: Any

    def __post_init__(self) -> None:
        if self.kind not in ("field", "string_key", "none_key", "integer_key"):
            raise ValueError(f"unknown run-seed path segment kind {self.kind!r}")

    def canonical_bytes(self) -> bytes:
        """Return the normative typed and length-framed seed-path bytes."""
        if self.kind == "none_key":
            return b"N" + (0).to_bytes(4, "big")
        if self.kind == "integer_key":
            encoded_value = str(self.value).encode("ascii")
            return (
                b"I"
                + len(encoded_value).to_bytes(4, "big")
                + encoded_value
            )
        encoded_value = self.value.encode()
        tag = b"F" if self.kind == "field" else b"S"
        return tag + len(encoded_value).to_bytes(4, "big") + encoded_value


@dataclass(frozen=True)
class RunSeedChild:
    """One semantic child edge exposed by a seed-graph composite."""

    relative_path: tuple[RunSeedPathSegment, ...]
    child: Any


@dataclass(frozen=True, eq=False)
class RunSeedReservation:
    """A leaf-owned prepared RNG replacement plus manifest seed provenance."""

    proposed_seed_source: str
    proposed_seed: Optional[int]
    prepared_state: Any = field(repr=False)


class SyndromePacketRouteKind(Enum):
    WINDOW_INPUT = auto()
    FEEDBACK_MEMORY_ROUND = auto()

@dataclass(frozen=True)
class PotentialStrong:
    window_key: tuple
@dataclass(frozen=True)
class PendingStrong:
    request_key: DecoderRequestKey
@dataclass(frozen=True)
class CsdInput:
    request_key: DecoderRequestKey
@dataclass(frozen=True)
class DecoderInputHold:
    request_key: DecoderRequestKey
@dataclass(frozen=True)
class Replay:
    window_key: tuple
    boundary_generation: int
@dataclass(frozen=True)
class RephaseGuard:
    request_key: DecoderRequestKey
@dataclass(frozen=True)
class SyndromePacketRoute:
    kind: SyndromePacketRouteKind
    source_operation_id: Optional[Any] = None

    def __post_init__(self) -> None:
        if (
            self.kind is not SyndromePacketRouteKind.WINDOW_INPUT
            and not is_stable_identity(self.source_operation_id)
        ):
            raise TypeError("feedback route needs a stable source operation identity")

    @classmethod
    def feedback_memory_round(cls, source_operation_id) -> "SyndromePacketRoute":
        return cls(SyndromePacketRouteKind.FEEDBACK_MEMORY_ROUND,
                   source_operation_id)

WINDOW_INPUT_ROUTE = SyndromePacketRoute(SyndromePacketRouteKind.WINDOW_INPUT)
@dataclass(frozen=True)
class QPUReadout:
    """One QPU-side result awaiting controller availability handling.

    Values are detector events or timing-only markers; decsim does not simulate the preceding analog or measurement-to-detection stages.
    """

    operation_id: Any
    patch_id: Any
    round_index: int
    bits: Optional[Any] = None
    code: Optional[str] = None
    n_fragments: int = 1
    fragment_index: int = 0
    size_bits: Optional[int] = None


@dataclass
class SyndromePayload:
    """One binary detector-data round accepted by the controller."""

    operation_id: int                 # op whose stream this round belongs to
    patch_id: int                     # patch that produced the round
    round_index: int                  # 1-based round number within the op
    bits: Optional[Any] = None        # detector bits (None = timing-only run)
    code: Optional[str] = None        # code name; drives CodeRouter routing
    n_fragments: int = 1              # link-layer fragments the round arrives in
    fragment_index: int = 0           # stable position within the complete round
    size_bits: Optional[int] = None   # wire size, for bandwidth/packing models


def normalize_binary_bits(bits: Any) -> Optional[tuple[int, ...]]:
    if bits is None:
        return None
    if type(bits) is list or type(bits) is tuple:
        if not all(
            type(bit) is bool or (type(bit) is int and bit in (0, 1))
            for bit in bits
        ):
            raise TypeError("syndrome bits must contain only exact binary values")
        return tuple(int(bit) for bit in bits)

    import numpy as np

    if (
        type(bits) is np.ndarray
        and bits.ndim == 1
        and bits.dtype == np.dtype(bool)
    ):
        return tuple(int(bit) for bit in bits)
    raise TypeError(
        "syndrome bits must be None, an exact binary list/tuple, or an "
        "exact one-dimensional NumPy boolean array"
    )


@dataclass(frozen=True)
class RetainedSyndromeFragment:
    """One validated immutable fragment retained after controller ingress."""

    operation_id: Any
    patch_id: Any
    round_index: int
    bits: Optional[tuple[int, ...]]
    code: Optional[str]
    size_bits: Optional[int]
    fragment_index: int

    def __post_init__(self) -> None:
        if self.bits is not None and (
            type(self.bits) is not tuple
            or any(type(bit) is not int or bit not in (0, 1)
                   for bit in self.bits)
        ):
            raise TypeError("retained bits must be an exact tuple of binary ints")

    @classmethod
    def from_payload(cls, payload: SyndromePayload) -> "RetainedSyndromeFragment":
        return cls(
            operation_id=payload.operation_id,
            patch_id=payload.patch_id,
            round_index=payload.round_index,
            bits=normalize_binary_bits(payload.bits),
            code=payload.code,
            size_bits=payload.size_bits,
            fragment_index=payload.fragment_index,
        )


@dataclass(frozen=True)
class SyndromeRoundPacket:
    """One complete immutable syndrome round in transport-arrival order."""

    operation_id: Any
    round_index: int
    fragments: tuple[RetainedSyndromeFragment, ...]

    def __post_init__(self) -> None:
        if (
            type(self.fragments) is not tuple
            or not self.fragments
            or any(type(fragment) is not RetainedSyndromeFragment
                   for fragment in self.fragments)
        ):
            raise TypeError(
                "packet fragments must be a nonempty tuple of retained fragments"
            )
        seen_fragment_indices = set()
        seen_patch_ids = []
        for fragment in self.fragments:
            if not same_stable_identity(fragment.operation_id, self.operation_id):
                raise ValueError("packet fragments must share operation identity")
            if not same_stable_identity(fragment.round_index, self.round_index):
                raise ValueError("packet fragments must share round_index")
            if fragment.fragment_index in seen_fragment_indices:
                raise ValueError("packet fragment indices must be distinct")
            seen_fragment_indices.add(fragment.fragment_index)
            if any(
                same_stable_identity(fragment.patch_id, patch_id)
                for patch_id in seen_patch_ids
            ):
                raise ValueError("packet patch identities must be distinct")
            seen_patch_ids.append(fragment.patch_id)


# ------------------------------------------------------------------ windows

class WindowProtocol(Enum):
    """Scientific model-building contract for one operation's window plan."""

    GENERIC = auto()
    TAN_ZERO_SEAM_GRAPHLIKE = auto()


@dataclass
class Window:
    """One decoder window inside an operation's syndrome stream.

    Rounds are 1-based and ranges inclusive: the window reads rounds
    [start_round, buffer_hi] and commits corrections for
    [commit_lo, commit_hi]."""

    op_id: int                        # operation that owns the stream
    k: int                            # window index within the op; key = (op_id, k)
    commit_lo: int                    # first round this window commits
    commit_hi: int                    # last round this window commits
    buffer_hi: int                    # last round it reads (trailing buffer)
    n_rounds: int                     # rounds the decode spans (sets job size)
    buffer_lo: Optional[int] = None   # leading-buffer start (for two-sided A windows)
    closed_temporal_boundaries: bool = False
    batched_preceding_idle_round_count: int = 0
    deps: list = field(default_factory=list)        # window keys this one waits on
    dependents: list = field(default_factory=list)  # window keys waiting on this one
    deps_remaining: int = 0           # unfinished deps countdown; 0 = unblocked
    committed: bool = False           # result folded into the op's accumulator
    queued: bool = False              # job handed to the decoder cluster
    blocked_logged: bool = False      # log-once flag for the "blocked" trace line
    boundary_in: Any = field(default_factory=dict)  # state owned by the
                                      # configured WindowInteraction
    t_first_round: Optional[int] = None    # tick the first round arrived
    t_data_complete: Optional[int] = None  # tick the last buffered round arrived
    t_queued: Optional[int] = None         # tick the job entered the decode queue
    t_dispatch: Optional[int] = None       # tick a decoder unit started it
    t_done: Optional[int] = None           # tick the decode finished

    @property
    def start_round(self) -> int:
        """First round this window needs (leading buffer if present, else commit start)."""
        return self.commit_lo if self.buffer_lo is None else self.buffer_lo

    @property
    def key(self) -> tuple:
        return (self.op_id, self.k)


@dataclass(frozen=True)
class WindowInfo:
    """Read-only geometry and topology exposed to interaction policies."""

    op_id: int
    k: int
    commit_lo: int
    commit_hi: int
    buffer_hi: int
    n_rounds: int
    buffer_lo: Optional[int]
    deps: tuple
    dependents: tuple
    detector_positions: Optional[dict] = None

    @classmethod
    def from_window(
        cls,
        window: Window,
        *,
        detector_positions: Optional[dict] = None,
    ) -> "WindowInfo":
        return cls(
            op_id=window.op_id,
            k=window.k,
            commit_lo=window.commit_lo,
            commit_hi=window.commit_hi,
            buffer_hi=window.buffer_hi,
            n_rounds=window.n_rounds,
            buffer_lo=window.buffer_lo,
            deps=tuple(window.deps),
            dependents=tuple(window.dependents),
            detector_positions=(
                None if detector_positions is None
                else dict(detector_positions)
            ),
        )

    @property
    def start_round(self) -> int:
        return self.commit_lo if self.buffer_lo is None else self.buffer_lo


def _exact_positive_int(value, label: str) -> None:
    if type(value) is not int or value < 1:
        raise TypeError(f"{label} must be an exact positive int")


def _exact_nonnegative_int(value, label: str) -> None:
    if type(value) is not int or value < 0:
        raise TypeError(f"{label} must be an exact nonnegative int")


@dataclass(frozen=True)
class ResolvedCodeGeometry:
    """Canonical planning/control geometry resolved once for one run."""

    code_name: str
    distance: int
    commit_round_count: int
    buffer_round_count: int
    minimum_leading_buffer_round_count: int
    minimum_trailing_buffer_round_count: int
    one_patch_spatial_node_count: int
    buffer_floor_override_active: bool

    def __post_init__(self) -> None:
        _exact_positive_int(self.distance, "distance")
        _exact_positive_int(self.commit_round_count, "commit_round_count")
        _exact_nonnegative_int(self.buffer_round_count, "buffer_round_count")
        _exact_nonnegative_int(
            self.minimum_leading_buffer_round_count,
            "minimum_leading_buffer_round_count",
        )
        _exact_nonnegative_int(
            self.minimum_trailing_buffer_round_count,
            "minimum_trailing_buffer_round_count",
        )
        _exact_positive_int(
            self.one_patch_spatial_node_count,
            "one_patch_spatial_node_count",
        )


@dataclass(frozen=True)
class ResolvedOperationPlanning:
    """Exact immutable planning/control facts for one operation."""

    operation_id: int
    code_geometry: ResolvedCodeGeometry
    round_count: int
    round_ticks: int
    spatial_node_count: int

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.round_count, "round_count")
        _exact_positive_int(self.round_ticks, "round_ticks")
        _exact_positive_int(self.spatial_node_count, "spatial_node_count")


@dataclass(frozen=True)
class ResolvedPatchPlanning:
    """Exact immutable cadence and idle-work facts for one patch."""

    patch_identity: Any
    code_geometry: ResolvedCodeGeometry
    round_ticks: int
    spatial_node_count: int

    def __post_init__(self) -> None:
        _exact_positive_int(self.round_ticks, "round_ticks")
        _exact_positive_int(self.spatial_node_count, "spatial_node_count")


@dataclass(frozen=True)
class WindowGeometry:
    """One immutable static window interval."""

    buffer_lo: int
    commit_lo: int
    commit_hi: int
    buffer_hi: int
    closed_temporal_boundaries: bool = False

    def __post_init__(self) -> None:
        for label, value in (
            ("buffer_lo", self.buffer_lo),
            ("commit_lo", self.commit_lo),
            ("commit_hi", self.commit_hi),
            ("buffer_hi", self.buffer_hi),
        ):
            _exact_positive_int(value, label)
        if not (
            self.buffer_lo
            <= self.commit_lo
            <= self.commit_hi
            <= self.buffer_hi
        ):
            raise ValueError("window geometry bounds are not ordered")

    @property
    def round_count(self) -> int:
        return self.buffer_hi - self.buffer_lo + 1


@dataclass(frozen=True)
class OperationWindowPlan:
    """One scheme's complete immutable result for one operation."""

    operation_id: int
    windows: tuple[WindowGeometry, ...]
    internal_dependencies: tuple[tuple[int, int], ...]
    entry_window_indices: tuple[int, ...]
    exit_window_indices: tuple[int, ...]
    windowed: bool
    batch_preceding_idle_rounds: bool
    protocol: WindowProtocol = WindowProtocol.GENERIC

    def __post_init__(self) -> None:
        if (
            not self.windows
            or any(type(window) is not WindowGeometry for window in self.windows)
        ):
            raise TypeError("windows must be nonempty WindowGeometry values")
        window_count = len(self.windows)
        edge_set = set()
        predecessors = [set() for _ in self.windows]
        dependents = [set() for _ in self.windows]
        for edge in self.internal_dependencies:
            if any(type(index) is not int for index in edge):
                raise TypeError("dependency edge indices must be exact ints")
            source, destination = edge
            if (
                source < 0
                or destination < 0
                or source >= window_count
                or destination >= window_count
            ):
                raise ValueError("dependency edge is out of range")
            if edge in edge_set:
                raise ValueError("dependency edges must be unique")
            edge_set.add(edge)
            predecessors[destination].add(source)
            dependents[source].add(destination)

        expected_entries = tuple(
            index for index, sources in enumerate(predecessors) if not sources
        )
        expected_exits = tuple(
            index for index, destinations in enumerate(dependents)
            if not destinations
        )
        if not same_stable_identity(self.entry_window_indices, expected_entries):
            raise ValueError("entry_window_indices must equal all graph roots")
        if not same_stable_identity(self.exit_window_indices, expected_exits):
            raise ValueError("exit_window_indices must equal all graph sinks")

        indegree = [len(sources) for sources in predecessors]
        ready = list(self.entry_window_indices)
        visited = 0
        while ready:
            source = ready.pop()
            visited += 1
            for destination in dependents[source]:
                indegree[destination] -= 1
                if indegree[destination] == 0:
                    ready.append(destination)
        if visited != window_count:
            raise ValueError("operation window graph must be acyclic")
        if type(self.batch_preceding_idle_rounds) is not bool:
            raise TypeError("batch_preceding_idle_rounds must be an exact bool")

@dataclass
class WindowPlan:
    """Compile-time window layout handed to the window manager."""

    windows: dict         # (op_id, k) -> Window
    window_count: dict    # op_id -> number of windows
    op_windows: dict      # op_id -> [window keys, in k order]
    successors: dict      # op_id -> [op ids listing it as predecessor]
    spatial_nodes: dict   # op_id -> decoding-graph nodes per round
    rounds_by_operation: dict  # op_id -> resolved positive round count
    code_names: dict       # op_id -> exact resolved code name
    total_windows: int
    windowed_by_operation: dict
    batch_preceding_idle_rounds_by_operation: dict
    protocol_by_operation: dict = field(default_factory=dict)


# ------------------------------------------------------ window interaction

@dataclass(frozen=True)
class DependencyResidual:
    """Complete global detector effect plus its compatibility mask view."""

    detector_ids: tuple[int, ...] = ()
    defects: dict | None = None

    def __post_init__(self) -> None:
        if any(type(detector_id) is not int or detector_id < 0
               for detector_id in self.detector_ids):
            raise TypeError(
                "dependency residual detector_ids must be nonnegative exact ints"
            )
        if len(set(self.detector_ids)) != len(self.detector_ids):
            raise ValueError("dependency residual detector_ids must be unique")


@dataclass(frozen=True)
class BoundaryDelivery:
    """One versioned boundary message offered to an interaction policy."""

    source_key: tuple
    destination_key: tuple
    source_revision: int
    delivery_revision: int
    latest_source_revision: int
    latest_delivery_revision: int
    source_operation_round_count: int
    dependency_released: bool
    payload: Any

    @property
    def is_current(self) -> bool:
        """Whether no newer source or edge-specific delivery supersedes this."""
        return (
            self.source_revision == self.latest_source_revision
            and self.delivery_revision == self.latest_delivery_revision
        )


@dataclass(frozen=True)
class BoundaryUpdate:
    """A policy's decision for one boundary arrival."""

    state: Any
    accepted: bool
    release_dependency: bool


class SeamFaultOwner(Enum):
    """Which side commits faults crossing a strong-region restart seam."""

    STRONG_REGION = auto()
    RESTART_WINDOW = auto()


@dataclass(frozen=True)
class StrongRegionPlan:
    """Geometry and seam ownership for one deferred strong decode."""

    commit_lo: int
    commit_hi: int
    context_lo: int
    context_hi: int
    restart_buffer_lo: Optional[int]
    restart_seam_fault_owner: Optional[SeamFaultOwner]

    def __post_init__(self) -> None:
        if not 1 <= self.context_lo <= self.commit_lo \
                <= self.commit_hi <= self.context_hi:
            raise ValueError(
                "strong-region bounds must satisfy 1 <= context_lo <= "
                "commit_lo <= commit_hi <= context_hi")
        if self.restart_buffer_lo is not None and self.restart_buffer_lo < 1:
            raise ValueError(
                "strong-region restart_buffer_lo must be at least one")


# ------------------------------------------------------------------- decode

@dataclass(frozen=True)
class SoftOutputSource:
    """Exact provenance required to interpret one confidence threshold."""

    method: str
    cluster_origin: str
    growth_schedule: str
    gap_units: str
    correction: str
    weight_step_natural_log: Optional[float]
    references: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.weight_step_natural_log is not None:
            normalized = float(self.weight_step_natural_log)
            if not math.isfinite(normalized) or normalized <= 0.0:
                raise ValueError(
                    "soft-output source weight step must be finite and positive"
                )
            object.__setattr__(self, "weight_step_natural_log", normalized)

@dataclass(frozen=True)
class SoftOutput:
    """One nonnegative confidence gap with immutable interpretation."""

    gap: float
    source: SoftOutputSource
    w_min: Optional[float] = None
    w_comp: Optional[float] = None

    def __post_init__(self) -> None:
        if isinstance(self.gap, bool) or not isinstance(self.gap, Real):
            raise TypeError("soft output gap must be a real number")
        normalized_gap = float(self.gap)
        if math.isnan(normalized_gap) or normalized_gap < 0:
            raise ValueError(
                "soft output gap must be nonnegative or positive infinity"
            )
        object.__setattr__(self, "gap", normalized_gap)
        for field_name in ("w_min", "w_comp"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(
                    f"soft output {field_name} must be a real number or None"
                )
            normalized_value = float(value)
            if math.isnan(normalized_value):
                raise ValueError(f"soft output {field_name} cannot be NaN")
            object.__setattr__(self, field_name, normalized_value)


class DecoderTier(Enum):
    WEAK = "weak"
    STRONG = "strong"


@dataclass(frozen=True)
class DecoderRequestKey:
    operation_id: Any
    window_id: int
    tier: DecoderTier
    run_sequence: int


@dataclass(frozen=True)
class DecoderServiceKey:
    run_sequence: int


@dataclass
class DecodeJob:
    """One unit of decoder work in the ``logical_reference`` profile.

    During the Phase-A migration, ``payloads`` is an upstream source view used
    to construct the decoder-local input. Decoders must not observe it before
    input-transfer completion. Boundary processing may replace an entry with a
    new immutable transformed fragment. The field is removed in Phase B.
    """

    op_id: int                               # operation the window belongs to
    window_id: int                           # window index within that op
    n_rounds: int                            # syndrome rounds in the window
    dem: Optional[Any] = None                # window detector error model (data-path decoders)
    payloads: list = field(default_factory=list)   # transfer-source view; cleared after materialization
    decoder_input: Optional[Any] = None             # decoder-local materialized input
    input_hold: Optional[Any] = None                # upstream hold released at transfer completion
    ready_time: int = 0                      # tick the job was enqueued (queue-wait accounting)
    on_done: Optional[Callable[[], None]] = None   # completion callback
    label: str = ""                          # log label
    strong_label: Optional[str] = None       # manager-owned label for a strong sibling
    spatial_nodes: Optional[int] = None      # decoding-graph nodes per round (latency models)
    code: Optional[str] = None               # code name, drives CodeRouter routing
    attempt: int = 0                         # 0 = first (weak) decode, 1 = strong redo
    hint: Optional[str] = None               # routing override, e.g. "strong"
    pool: Optional[str] = None               # unit pool assigned at enqueue
    window: Optional[Window] = None          # back-reference to the source window
    strong_decode_for: Optional[tuple] = None      # (op_id, window_id) this strong job re-decodes
    awaiting_strong_result: bool = False     # weak result held non-final until the strong sibling lands
    cancelled: bool = False                  # cancelled siblings discard completion
    completed: bool = False                  # terminal flag; admission refuses reuse of a completed job
    submitted: bool = False                  # admitted once to one queue slot and unit
    request_key: Optional[DecoderRequestKey] = None
    request_created_ticks: Optional[int] = None
    request_admitted_ticks: Optional[int] = None
    service_key: Optional[DecoderServiceKey] = None
    service_original_request_keys: tuple[DecoderRequestKey, ...] = ()
    service_cancelled_request_keys: set[DecoderRequestKey] = field(default_factory=set)
    service_dispatch_ticks: Optional[int] = None

@dataclass
class DecodeResult:
    """One window result; timing-only decoders leave optional fields unset."""

    op_id: int
    window_id: int
    correction: Optional[Any] = None         # correction operator (None = timing-only)
    logical_observables: Optional[tuple[int, ...]] = None  # full prediction
    soft_output: Optional["SoftOutput"] = None  # source-compatible confidence
    boundary_defects: Optional[dict] = None  # defects on window seams (cross-window matching)
    boundary_data: Optional[Any] = None      # optional richer interaction payload


@dataclass(frozen=True)
class StrongDecodeCompletion:
    request_key: DecoderRequestKey
    result: DecodeResult

    def __post_init__(self) -> None:
        if (type(self.request_key), type(self.result)) != (
                DecoderRequestKey, DecodeResult):
            raise TypeError("strong completion requires exact key and result types")
        if (self.request_key.tier is not DecoderTier.STRONG
                or not same_stable_identity(
                    self.request_key.operation_id, self.result.op_id)
                or self.request_key.window_id != self.result.window_id):
            raise ValueError("strong completion identity must match its strong key")


@dataclass
class DecodeOutcome:
    """Joint decode outcome delivered to the strategy hook."""

    job: DecodeJob
    result: DecodeResult


# ---------------------------------------------------------------- resources

@dataclass(frozen=True)
class ResourceClaim:
    """Typed exclusivity claim on shared hardware. Only kind="qubits" is
    used today (layouts derive one claim from an op's qubit tuple)."""

    kind: str
    ids: frozenset


@dataclass(frozen=True)
class Decision:
    """Feedback timing route for one target operation."""

    target_operation_id: int
    releases_operation: bool = True


@dataclass(frozen=True)
class ExecutionProgram:
    """Immutable controller program-load artifact."""

    operations: tuple
    decode_operations: tuple = ()
    dynamic_streams: tuple = ()
    protected_regions: tuple = ()


@dataclass(frozen=True)
class StreamBinding:
    """Immutable runtime association between an operation and stream range."""
    stream_id: Any
    stream_offset: int

    def __post_init__(self) -> None:
        if self.stream_offset < 0:
            raise ValueError("stream_offset must be nonnegative")


@dataclass(frozen=True)
class RunOperationBody:
    """Immutable controller-to-QPU command for one operation body."""

    operation: Any
    round_ticks: int
    round_count: int
    source_round_count: int
    emits_detector_data: bool = True
    finalizes_stream_round: bool = False

    def __post_init__(self) -> None:
        if self.round_ticks <= 0:
            raise ValueError("round_ticks must be positive")
        if self.round_count < 0 or self.source_round_count < 0:
            raise ValueError("round counts must be nonnegative")


# ----------------------------------------------------------------- workload

@dataclass(frozen=True)
class ProtectedRegion:
    """One patch allocation generation with inclusive operation endpoints."""

    patch_id: Any
    stream_id: int
    start_operation_id: int
    end_operation_id: int


@dataclass(frozen=True)
class SuccessorReadiness:
    operation_id: int
    rounds_arrived: int
    round_count: int


@dataclass(frozen=True)
class WindowReadiness:
    local_rounds_arrived: int
    local_round_count: int
    successors: tuple[SuccessorReadiness, ...]
    memory_rounds_arrived: int
    tail_closed: bool


class OpKind(Enum):
    """Logical-op kind vocabulary: lets a RoundsPolicy distinguish a
    measurement (1 round) from a merge (m·d rounds) from an injection (O(1))."""

    IDLE = auto()
    MEMORY = auto()
    MERGE = auto()
    MEASURE = auto()
    INJECT = auto()
    GENERIC = auto()


@dataclass
class Operation:
    """One logical operation in the circuit."""

    id: int
    name: str                         # human-readable label used in traces
    qubits: tuple                     # logical qubit ids the op acts on
    clifford: bool = True             # non-Clifford implies a magic state by default
    circuit: Optional[Any] = None     # stim circuit for real-syndrome (data-path) runs
    consumes_magic_state: Optional[bool] = None  # override; None = infer from clifford
    patches: tuple = ()               # patch ids whose syndrome streams feed the op
    predecessors: tuple = ()          # workload op ids that must complete first
    decoder_boundary_predecessors: tuple = ()  # prior decode streams at a boundary
    # Decode stream this segment's rounds fold into. Seeded StimDevice runs
    # require an exact built-in int/str; unseeded and non-Stim devices may use
    # another identity type accepted by that device.
    stream_id: Optional[Any] = None
    stream_offset: Optional[int] = None  # global-round offset of the segment in its stream
    scheduled_start_round: int = 0
    emits_detector_data: bool = True
    finalizes_stream_round: bool = False
    syndrome_fragment_index: Optional[int] = None
    syndrome_fragment_count: Optional[int] = None
    blocked_by: Optional[int] = None  # op id whose Decision must release this op
    feedback_boundary_mode: Optional[str] = None  # per-op override of the RunSpec mode
    requires_result_return_to_qpu: bool = False  # decision must travel back to the QPU
    kind: OpKind = OpKind.GENERIC     # rounds-policy vocabulary (see OpKind)

    def __post_init__(self) -> None:
        """Keep the identities used as runtime keys exact and reproducible."""
        if self.scheduled_start_round < 0:
            raise ValueError("scheduled_start_round must be nonnegative")
        fragment_fields = (
            self.syndrome_fragment_index,
            self.syndrome_fragment_count,
        )
        if (fragment_fields[0] is None) != (fragment_fields[1] is None):
            raise ValueError(
                "syndrome fragment index and count must be set together")
        if fragment_fields[0] is not None:
            if any(type(value) is not int for value in fragment_fields):
                raise TypeError("syndrome fragment fields must be exact ints")
            if not 0 <= fragment_fields[0] < fragment_fields[1]:
                raise ValueError(
                    "syndrome fragment index must be within fragment count")
            if not self.emits_detector_data:
                raise ValueError("syndrome fragment slots require an emitter")
        if self.finalizes_stream_round:
            if not self.emits_detector_data:
                raise ValueError("stream finalizers must emit detector data")
            if self.stream_id is None:
                raise ValueError("stream finalizers require an explicit stream_id")
            if type(self.stream_offset) is not int or self.stream_offset < 0:
                raise ValueError(
                    "stream finalizers require a nonnegative stream_offset")
            if fragment_fields[0] is None:
                raise ValueError(
                    "stream finalizers require an explicit fragment slot")

    @property
    def needs_magic_state(self) -> bool:
        """True when this operation draws a distilled magic state from the factory."""
        if self.consumes_magic_state is not None:
            return self.consumes_magic_state
        return not self.clifford


@dataclass(frozen=True)
class OperationPlanningView:
    """Immutable operation configuration visible to planning collaborators."""

    id: int
    name: str
    qubits: tuple
    clifford: bool
    consumes_magic_state: Optional[bool]
    patches: tuple
    predecessors: tuple
    decoder_boundary_predecessors: tuple
    stream_id: Optional[Any]
    stream_offset: Optional[int]
    scheduled_start_round: int
    emits_detector_data: bool
    finalizes_stream_round: bool
    syndrome_fragment_index: Optional[int]
    syndrome_fragment_count: Optional[int]
    blocked_by: Optional[int]
    feedback_boundary_mode: str
    requires_result_return_to_qpu: bool
    kind: OpKind

    @classmethod
    def from_operation(
        cls,
        operation: Operation,
        *,
        default_feedback_boundary_mode: str = "trailing_buffer",
    ) -> "OperationPlanningView":
        """Freeze an operation while excluding its executable circuit."""
        return cls(
            id=operation.id,
            name=operation.name,
            qubits=tuple(operation.qubits),
            clifford=operation.clifford,
            consumes_magic_state=operation.consumes_magic_state,
            patches=tuple(operation.patches),
            predecessors=tuple(operation.predecessors),
            decoder_boundary_predecessors=tuple(
                operation.decoder_boundary_predecessors
            ),
            stream_id=operation.stream_id,
            stream_offset=operation.stream_offset,
            scheduled_start_round=operation.scheduled_start_round,
            emits_detector_data=operation.emits_detector_data,
            finalizes_stream_round=operation.finalizes_stream_round,
            syndrome_fragment_index=operation.syndrome_fragment_index,
            syndrome_fragment_count=operation.syndrome_fragment_count,
            blocked_by=operation.blocked_by,
            feedback_boundary_mode=(
                operation.feedback_boundary_mode
                if operation.feedback_boundary_mode is not None
                else default_feedback_boundary_mode
            ),
            requires_result_return_to_qpu=(
                operation.requires_result_return_to_qpu
            ),
            kind=operation.kind,
        )
