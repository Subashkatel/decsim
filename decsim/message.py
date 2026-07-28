"""The typed messages the simulator's modules pass to each other.

Ordered by pipeline stage: syndrome rounds measured on the chip
(SyndromePayload), the decode windows planned over them (Window, WindowPlan),
the jobs and results that flow through the decoder cluster (DecodeJob,
DecodeResult), and the feedback decisions sent back toward the chip
(Decision). Operation, at the bottom, describes the workload itself.

This module imports nothing from the rest of decsim, so any module can
depend on it without creating an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional


def is_stable_string(value: Any) -> bool:
    """Whether a value is an exact Unicode-scalar string."""
    return (
        type(value) is str
        and all(
            not 0xD800 <= ord(character) <= 0xDFFF
            for character in value
        )
    )


def is_stable_identity(value: Any) -> bool:
    """Whether a value has deterministic, recursively typed identity."""
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


@dataclass(frozen=True)
class RunSeedPathSegment:
    """One framed semantic edge in the run-level seed component graph."""

    kind: str
    value: Any

    def __post_init__(self) -> None:
        if self.kind == "field":
            if not is_stable_string(self.value) or not self.value:
                raise ValueError(
                    "run-seed field segments require a nonempty Unicode "
                    "scalar string"
                )
            return
        if self.kind == "string_key":
            if not is_stable_string(self.value):
                raise TypeError(
                    "run-seed string-key segments require a Unicode scalar "
                    "string"
                )
            return
        if self.kind == "none_key":
            if self.value is not None:
                raise ValueError(
                    "run-seed none-key segments cannot carry a value"
                )
            return
        if self.kind == "integer_key":
            if type(self.value) is not int:
                raise TypeError(
                    "run-seed integer-key segments require a built-in int"
                )
            return
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

    def __post_init__(self) -> None:
        if type(self.relative_path) is not tuple or not self.relative_path:
            raise ValueError(
                "run-seed child paths must be nonempty tuples"
            )
        if not all(
            isinstance(segment, RunSeedPathSegment)
            for segment in self.relative_path
        ):
            raise TypeError(
                "run-seed child paths contain only RunSeedPathSegment values"
            )


@dataclass(frozen=True, eq=False)
class RunSeedReservation:
    """A leaf-owned prepared RNG replacement plus manifest seed provenance."""

    proposed_seed_source: str
    proposed_seed: Optional[int]
    prepared_state: Any = field(repr=False)

    def __post_init__(self) -> None:
        if self.proposed_seed_source not in (
            "derived",
            "explicit_local",
            "entropy",
        ):
            raise ValueError(
                f"unknown run-seed source {self.proposed_seed_source!r}"
            )
        if self.proposed_seed_source == "entropy":
            if self.proposed_seed is not None:
                raise ValueError("entropy reservations cannot carry a seed")
            return
        if (
            type(self.proposed_seed) is not int
            or not 0 <= self.proposed_seed < (1 << 64)
        ):
            raise ValueError(
                f"{self.proposed_seed_source} reservations require an unsigned "
                f"64-bit built-in integer seed"
            )


# ----------------------------------------------------------------- syndrome

@dataclass
class SyndromePayload:
    """One measured syndrome round for one logical operation."""

    operation_id: int                 # op whose stream this round belongs to
    patch_id: int                     # patch that produced the round
    round_index: int                  # 1-based round number within the op
    bits: Optional[Any] = None        # detector bits (None = timing-only run)
    code: Optional[str] = None        # code name; drives CodeRouter routing
    n_fragments: int = 1              # link-layer fragments the round arrives in
    size_bits: Optional[int] = None   # wire size, for bandwidth/packing models

    def __post_init__(self) -> None:
        if self.n_fragments < 1:
            raise ValueError(f"n_fragments must be >= 1 (got {self.n_fragments})")


# ------------------------------------------------------------------ windows

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
    buffer_lo: Optional[int] = None   # leading-buffer start (parallel-scheme layer B)
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

    @classmethod
    def from_window(cls, window: Window) -> "WindowInfo":
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
        )

    @property
    def start_round(self) -> int:
        return self.commit_lo if self.buffer_lo is None else self.buffer_lo

    @property
    def key(self) -> tuple:
        return (self.op_id, self.k)


class WindowGraph:
    """Windows plus dependency edges; mutated only by WindowManager/DynamicWindows."""

    def __init__(self) -> None:
        self.windows: dict[tuple, Window] = {}

    def add_window(self, window: Window) -> None:
        if window.key in self.windows:
            raise ValueError(f"duplicate window {window.key}")
        self.windows[window.key] = window

    def wire_dep(self, src_key: tuple, dst_key: tuple) -> None:
        """src must hand its boundary to dst before dst may decode."""
        src, dst = self.windows[src_key], self.windows[dst_key]
        src.dependents.append(dst_key)
        dst.deps.append(src_key)
        dst.deps_remaining += 1


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
        if type(self.code_name) is not str or not self.code_name:
            raise TypeError("code_name must be a nonempty exact str")
        try:
            self.code_name.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("code_name must contain only Unicode scalars") from exc
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
        if type(self.buffer_floor_override_active) is not bool:
            raise TypeError("buffer_floor_override_active must be an exact bool")


@dataclass(frozen=True)
class ResolvedCodeSpatialProfile:
    """One base code-size result for each consumed patch cardinality."""

    entries: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple or not self.entries:
            raise TypeError("spatial profile entries must be a nonempty tuple")
        previous = 0
        for entry in self.entries:
            if type(entry) is not tuple or len(entry) != 2:
                raise TypeError("spatial profile entries must be exact pairs")
            patch_count, node_count = entry
            _exact_positive_int(patch_count, "spatial profile patch_count")
            _exact_positive_int(node_count, "spatial profile node_count")
            if patch_count <= previous:
                raise ValueError(
                    "spatial profile patch counts must be unique and ascending"
                )
            previous = patch_count
        if self.entries[0][0] != 1:
            raise ValueError("spatial profile must contain patch count 1")

    def for_patch_count(self, patch_count: int) -> int:
        _exact_positive_int(patch_count, "patch_count")
        for candidate, node_count in self.entries:
            if candidate == patch_count:
                return node_count
        raise KeyError(f"patch count {patch_count} was not resolved")


@dataclass(frozen=True)
class ResolvedOperationPlanning:
    """Exact immutable planning/control facts for one operation."""

    operation_id: int
    code_geometry: ResolvedCodeGeometry
    round_count: int
    round_ticks: int
    spatial_node_count: int

    def __post_init__(self) -> None:
        if type(self.operation_id) is not int:
            raise TypeError("operation_id must be an exact int")
        if type(self.code_geometry) is not ResolvedCodeGeometry:
            raise TypeError("code_geometry must be an exact ResolvedCodeGeometry")
        _exact_positive_int(self.round_count, "round_count")
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
        if type(self.code_geometry) is not ResolvedCodeGeometry:
            raise TypeError("code_geometry must be an exact ResolvedCodeGeometry")
        _exact_positive_int(self.round_ticks, "round_ticks")
        _exact_positive_int(self.spatial_node_count, "spatial_node_count")


@dataclass(frozen=True)
class WindowGeometry:
    """One immutable static window interval."""

    buffer_lo: int
    commit_lo: int
    commit_hi: int
    buffer_hi: int

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

    def __post_init__(self) -> None:
        if type(self.operation_id) is not int:
            raise TypeError("operation_id must be an exact int")
        if (
            type(self.windows) is not tuple
            or not self.windows
            or any(type(window) is not WindowGeometry for window in self.windows)
        ):
            raise TypeError("windows must be a nonempty tuple of WindowGeometry")
        if type(self.internal_dependencies) is not tuple:
            raise TypeError("internal_dependencies must be an exact tuple")
        window_count = len(self.windows)
        edge_set = set()
        predecessors = [set() for _ in self.windows]
        dependents = [set() for _ in self.windows]
        for edge in self.internal_dependencies:
            if (
                type(edge) is not tuple
                or len(edge) != 2
                or any(type(index) is not int for index in edge)
            ):
                raise TypeError("dependency edges must be exact (int, int) pairs")
            source, destination = edge
            if (
                source < 0
                or destination < 0
                or source >= window_count
                or destination >= window_count
                or source == destination
            ):
                raise ValueError("dependency edge is out of range or self-directed")
            if edge in edge_set:
                raise ValueError("dependency edges must be unique")
            edge_set.add(edge)
            predecessors[destination].add(source)
            dependents[source].add(destination)

        self._validate_boundary_indices(
            self.entry_window_indices,
            "entry_window_indices",
            window_count,
        )
        self._validate_boundary_indices(
            self.exit_window_indices,
            "exit_window_indices",
            window_count,
        )
        expected_entries = tuple(
            index for index, sources in enumerate(predecessors) if not sources
        )
        expected_exits = tuple(
            index for index, destinations in enumerate(dependents)
            if not destinations
        )
        if self.entry_window_indices != expected_entries:
            raise ValueError("entry_window_indices must equal all graph roots")
        if self.exit_window_indices != expected_exits:
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
        if type(self.windowed) is not bool:
            raise TypeError("windowed must be an exact bool")
        if type(self.batch_preceding_idle_rounds) is not bool:
            raise TypeError("batch_preceding_idle_rounds must be an exact bool")

    @staticmethod
    def _validate_boundary_indices(indices, label, window_count) -> None:
        if (
            type(indices) is not tuple
            or not indices
            or any(type(index) is not int for index in indices)
        ):
            raise TypeError(f"{label} must be a nonempty tuple of exact ints")
        if len(set(indices)) != len(indices):
            raise ValueError(f"{label} must contain unique indices")
        if tuple(sorted(indices)) != indices:
            raise ValueError(f"{label} must be ascending")
        if any(index < 0 or index >= window_count for index in indices):
            raise ValueError(f"{label} contains an out-of-range index")


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
    summary: dict = field(default_factory=dict)   # printable planning stats
    windowed_by_operation: dict = field(default_factory=dict)
    batch_preceding_idle_rounds_by_operation: dict = field(default_factory=dict)


# ------------------------------------------------------ window interaction

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
    absorbed_window_keys: tuple
    restart_window_key: Optional[tuple]
    restart_buffer_lo: Optional[int]
    restart_seam_fault_owner: Optional[SeamFaultOwner]


# ------------------------------------------------------------------- decode

@dataclass
class DecodeJob:
    """One unit of decoder work."""

    op_id: int                               # operation the window belongs to
    window_id: int                           # window index within that op
    n_rounds: int                            # syndrome rounds in the window
    dem: Optional[Any] = None                # window detector error model (data-path decoders)
    payloads: list = field(default_factory=list)   # SyndromePayloads with the window's bits
    ready_time: int = 0                      # tick the job was enqueued (queue-wait accounting)
    deadline: int = 0                        # tick stamped by the DeadlinePolicy (EDF)
    on_done: Optional[Callable[[], None]] = None   # completion callback
    label: str = ""                          # log label
    spatial_nodes: Optional[int] = None      # decoding-graph nodes per round (latency models)
    code: Optional[str] = None               # code name, drives CodeRouter routing
    attempt: int = 0                         # 0 = first (weak) decode, 1 = strong redo
    hint: Optional[str] = None               # routing override, e.g. "strong"
    pool: Optional[str] = None               # unit pool assigned at enqueue
    window: Optional[Window] = None          # back-reference to the source window
    strong_decode_for: Optional[tuple] = None      # (op_id, window_id) this strong job re-decodes
    awaiting_strong_result: bool = False     # weak result held non-final until the strong sibling lands
    cancelled: bool = False                  # set when a speculative sibling is cancelled;
                                             # the completion callback then discards the result
    completed: bool = False                  # guards against duplicate completion delivery
    submitted: bool = False                  # set once the pool admits the job; a job holds one
                                             # queue slot and one unit, so it is enqueued once


@dataclass
class DecodeResult:
    """Decoder output for one window. Timing-only decoders leave every
    optional field None; data-path decoders fill what they compute."""

    op_id: int
    window_id: int
    correction: Optional[Any] = None         # correction operator (None = timing-only)
    logical_observables: Optional[tuple[int, ...]] = None
    # Complete predicted logical-observable vector; None is timing-only.
    soft_output: Optional[float] = None      # decoder confidence; below the Switching threshold escalates
    boundary_defects: Optional[dict] = None  # defects on window seams (cross-window matching)
    boundary_data: Optional[Any] = None      # optional richer interaction payload


@dataclass
class DecodeOutcome:
    """Joint decode outcome delivered to the strategy hook."""

    job: DecodeJob
    result: DecodeResult


@dataclass
class SoftOutput:
    """Soft-output confidence for one window; a smaller gap means the
    decoder is less sure of its logical value."""

    logical_value: int    # predicted logical bit from the best decoding
    gap: float            # confidence, e.g. complementary gap |w_comp - w_min|
    w_min: float = 0.0    # weight of the unconstrained min-weight decoding
    w_comp: float = 0.0   # weight of the best decoding forced to the other class


# ---------------------------------------------------------------- resources

@dataclass(frozen=True)
class ResourceClaim:
    """Typed exclusivity claim on shared hardware. Only kind="qubits" is
    used today (layouts derive one claim from an op's qubit tuple)."""

    kind: str
    ids: frozenset


# ----------------------------------------------------------------- feedback

@dataclass(frozen=True)
class IntrinsicMeasurement:
    """Supplied intrinsic measurement with stable trajectory provenance."""

    operation_id: Any
    trajectory_id: Any
    value: int
    source: str

    def __post_init__(self) -> None:
        if not is_stable_identity(self.operation_id):
            raise TypeError(
                "intrinsic measurement operation_id must be a stable "
                "built-in int, str, or recursive tuple")
        if not is_stable_identity(self.trajectory_id):
            raise TypeError(
                "intrinsic measurement trajectory_id must be a stable "
                "built-in int, str, or recursive tuple")
        if type(self.value) is not int:
            raise TypeError(
                "intrinsic measurement value must be an exact int bit")
        if self.value not in (0, 1):
            raise ValueError(
                f"intrinsic measurement value must be 0 or 1, got "
                f"{self.value}")
        if type(self.source) is not str:
            raise TypeError("intrinsic measurement source must be a string")
        if not self.source.strip():
            raise ValueError(
                "intrinsic measurement source must be nonempty")


@dataclass(frozen=True)
class FeedbackEffect:
    """One complete functional consequence selected from a decode vector."""

    logical_observable_index: int
    decoded_value: int
    intrinsic_measurement: Optional[IntrinsicMeasurement]
    correction_value: int
    basis: str
    pauli: str
    apply_s: bool

    def __post_init__(self) -> None:
        if type(self.logical_observable_index) is not int:
            raise TypeError(
                "logical_observable_index must be an exact int")
        if self.logical_observable_index < 0:
            raise ValueError(
                "logical_observable_index must be nonnegative")
        for field_name in ("decoded_value", "correction_value"):
            value = getattr(self, field_name)
            if type(value) is not int:
                raise TypeError(f"{field_name} must be an exact int bit")
            if value not in (0, 1):
                raise ValueError(f"{field_name} must be 0 or 1, got {value}")
        if (
            self.intrinsic_measurement is not None
            and type(self.intrinsic_measurement) is not IntrinsicMeasurement
        ):
            raise TypeError(
                "intrinsic_measurement must be IntrinsicMeasurement or None")
        if type(self.basis) is not str or not self.basis:
            raise ValueError("basis must be a nonempty string")
        if type(self.pauli) is not str or not self.pauli:
            raise ValueError("pauli must be a nonempty string")
        if type(self.apply_s) is not bool:
            raise TypeError("apply_s must be an exact bool")


@dataclass(frozen=True)
class Decision:
    """Feedback timing plus an optional functional effect."""

    target_operation_id: int
    effect: Optional[FeedbackEffect] = None
    releases_operation: bool = True   # False = result-return only, no op waiting
    strong_committed: bool = False    # mirrors op.requires_strong_commit (marker)


# ----------------------------------------------------------------- workload

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
    predecessors: tuple = ()          # op ids whose windows must decode first
    has_successor: bool = False       # a later op consumes this op's boundary
    # Decode stream this segment's rounds fold into. Seeded StimDevice runs
    # require an exact built-in int/str; unseeded and non-Stim devices may use
    # another identity type accepted by that device.
    stream_id: Optional[Any] = None
    stream_offset: Optional[int] = None  # global-round offset of the segment in its stream
    blocked_by: Optional[int] = None  # op id whose Decision must release this op
    feedback_boundary_mode: Optional[str] = None  # per-op override of the RunSpec mode
    requires_result_return_to_chip: bool = False  # decision must travel back to the QPU
    requires_strong_commit: bool = False  # marker only; release stays unconditional
    byproduct_pauli: str = "X"        # Pauli applied to the successor on measurement 1
    measurement_basis: str = "Z"
    logical_observable_index: Optional[int] = None
    intrinsic_measurement: Optional[IntrinsicMeasurement] = None
    kind: OpKind = OpKind.GENERIC     # rounds-policy vocabulary (see OpKind)

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
    has_successor: bool
    stream_id: Optional[int]
    stream_offset: Optional[int]
    blocked_by: Optional[int]
    feedback_boundary_mode: str
    requires_result_return_to_chip: bool
    requires_strong_commit: bool
    byproduct_pauli: str
    measurement_basis: str
    logical_observable_index: Optional[int]
    intrinsic_measurement: Optional[IntrinsicMeasurement]
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
            has_successor=operation.has_successor,
            stream_id=operation.stream_id,
            stream_offset=operation.stream_offset,
            blocked_by=operation.blocked_by,
            feedback_boundary_mode=(
                operation.feedback_boundary_mode
                if operation.feedback_boundary_mode is not None
                else default_feedback_boundary_mode
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

    @property
    def needs_magic_state(self) -> bool:
        if self.consumes_magic_state is not None:
            return self.consumes_magic_state
        return not self.clifford
