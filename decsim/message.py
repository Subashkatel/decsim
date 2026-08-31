"""The vocabulary every component speaks: the frozen values that travel
between the QPU, the controller, Buffer 0, the window manager, the decoders
and the Pauli frame (readouts, payloads, packets, windows, plans, jobs,
results, boundaries, requests, seeds), and the stable-identity helpers that
make operation and window keys hashable and orderable across types. Nothing
here has behavior beyond a value's own derived views."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
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


_SEED_PATH_TAG = {"field": b"F", "string_key": b"S"}


@dataclass(frozen=True)
class RunSeedPathSegment:
    """One framed semantic edge in the run-level seed component graph."""

    kind: str
    value: Any

    def canonical_bytes(self) -> bytes:
        """Return the normative typed and length-framed seed-path bytes; an
        unknown kind has no tag and fails here."""
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
        tag = _SEED_PATH_TAG[self.kind]
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
    """Where a completed round goes: a window input or a feedback-memory round."""

    WINDOW_INPUT = auto()
    FEEDBACK_MEMORY_ROUND = auto()


@dataclass(frozen=True)
class SyndromePacketRoute:
    """The route of one round from the controller: window input, or a feedback-memory round of a source operation."""

    kind: SyndromePacketRouteKind
    source_operation_id: Optional[Any] = None

    @classmethod
    def feedback_memory_round(cls, source_operation_id) -> "SyndromePacketRoute":
        return cls(SyndromePacketRouteKind.FEEDBACK_MEMORY_ROUND,
                   source_operation_id)

WINDOW_INPUT_ROUTE = SyndromePacketRoute(SyndromePacketRouteKind.WINDOW_INPUT)


@dataclass(frozen=True)
class QPUReadout:
    """One QPU-side readout awaiting controller front-end handling.

    DECSIM intentionally does not carry an analog waveform. ``bits`` is the
    sampled/classifiable outcome cargo; after the configured physical
    acquisition/discrimination latency, the controller exposes its normalized
    classical-bit tuple. Detection events are formed later from these packets.
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
    bits: Optional[Any] = None        # raw measurement bits (None = timing-only run)
    code: Optional[str] = None        # code name; drives CodeRouter routing
    n_fragments: int = 1              # link-layer fragments the round arrives in
    fragment_index: int = 0           # stable position within the complete round
    size_bits: Optional[int] = None   # wire size, for bandwidth/packing models


def normalize_binary_bits(bits: Any) -> Optional[tuple[int, ...]]:
    """Bits as a tuple of 0/1 ints; a list, tuple or NumPy bool array in."""
    if bits is None:
        return None
    return tuple(int(bit) for bit in bits)


@dataclass(frozen=True)
class RetainedSyndromeFragment:
    """One validated immutable fragment retained after controller packing."""

    operation_id: Any
    patch_id: Any
    round_index: int
    bits: Optional[tuple[int, ...]]
    size_bits: Optional[int]
    fragment_index: int

    @classmethod
    def from_payload(cls, payload: SyndromePayload) -> "RetainedSyndromeFragment":
        return cls(
            operation_id=payload.operation_id,
            patch_id=payload.patch_id,
            round_index=payload.round_index,
            bits=normalize_binary_bits(payload.bits),
            size_bits=payload.size_bits,
            fragment_index=payload.fragment_index,
        )


@dataclass(frozen=True)
class SyndromeRoundPacket:
    """One complete immutable syndrome round in transport-arrival order."""

    operation_id: Any
    round_index: int
    fragments: tuple[RetainedSyndromeFragment, ...]

    def defects_text(self) -> str:
        """The round's cargo for the I/O trace: set detection-event indices
        across the fragments in order, sparse so d=11 lines stay readable."""
        position = 0
        defects = []
        for fragment in self.fragments:
            bits = fragment.bits
            if bits is None:
                return "timing-only"
            for bit in bits:
                if bit:
                    defects.append(position)
                position += 1
        return f"defects {{{', '.join(map(str, defects))}}}" if defects else "no defects"

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
    buffer_filled_by_memory: bool = False  # trailing buffer satisfied by
                                      # memory rounds alone: released on time,
                                      # no syndrome content behind those
                                      # rounds (references decode buffer
                                      # content: LATTE, SWIPER, Skoric/Tan)
    deps: list = field(default_factory=list)        # window keys this one waits on
    dependents: list = field(default_factory=list)  # window keys waiting on this one
    deps_remaining: int = 0           # unfinished deps countdown; 0 = unblocked
    service_began: bool = False       # its decode is past the boundary gate
    committed: bool = False           # result folded into the op's accumulator
    queued: bool = False              # job handed to the decoder cluster
    blocked_logged: bool = False      # log-once flag for the "blocked" trace line
    boundary_in: Any = field(default_factory=dict)  # state owned by the
                                      # configured WindowInteraction
    decode_status: Optional[str] = None  # best-effort status of the committed decode, None = succeeded
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
    window_floor_justification: Optional[str]


@dataclass(frozen=True)
class ResolvedOperationPlanning:
    """Exact immutable planning/control facts for one operation."""

    operation_id: int
    code_geometry: ResolvedCodeGeometry
    round_count: int
    round_ticks: int
    spatial_node_count: int


@dataclass(frozen=True)
class ResolvedPatchPlanning:
    """Exact immutable cadence and idle-work facts for one patch."""

    patch_identity: Any
    code_geometry: ResolvedCodeGeometry
    round_ticks: int
    spatial_node_count: int


@dataclass(frozen=True)
class WindowGeometry:
    """One immutable static window interval."""

    buffer_lo: int
    commit_lo: int
    commit_hi: int
    buffer_hi: int
    closed_temporal_boundaries: bool = False

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


@dataclass(frozen=True)
class SoftOutput:
    """One nonnegative confidence gap with immutable interpretation."""

    gap: float
    source: SoftOutputSource
    w_min: Optional[float] = None
    w_comp: Optional[float] = None


class DecoderTier(Enum):
    """Weak (first, fast) or strong (escalated, slow) decode."""

    WEAK = "weak"
    STRONG = "strong"


@dataclass(frozen=True)
class DecoderRequestKey:
    """Identity of one decode request: window, tier and the run-wide ordinal that keeps retries distinct."""

    operation_id: Any
    window_id: int
    tier: DecoderTier
    run_sequence: int


@dataclass(frozen=True)
class DecoderServiceKey:
    """Identity of one decoder service (a batch of requests served together)."""

    run_sequence: int


@dataclass
class DecodeJob:
    """One unit of decoder work: a window's rounds, its model, its identity
    in the decoder queues, and the timestamps of its life. ``payloads`` is the
    Buffer 0 view of the rounds until the transfer lands them in a unit's
    memory (``decoder_input``); a decoder reads only its unit's memory.
    """

    op_id: int                               # operation the window belongs to
    window_id: int                           # window index within that op
    n_rounds: int                            # syndrome rounds in the window
    dem: Optional[Any] = None                # window detector error model (data-path decoders)
    payloads: list = field(default_factory=list)   # transfer-source view; cleared after materialization
    decoder_input: Optional[Any] = None             # materialized decoder memory value
    input_hold: Optional[Any] = None                # upstream hold released at transfer completion
    reserve_transfer: Optional[Callable[[], int]] = None   # called at dispatch: reserve the input link, return its delay in ticks
    unit: Optional[int] = None                     # decoder unit assigned at dispatch
    memory: Optional[Any] = None                   # that unit's DecoderMemory while it holds this job's input
    ready_time: int = 0                      # tick the job was enqueued (queue-wait accounting)
    on_done: Optional[Callable[[], None]] = None   # completion callback
    label: str = ""                          # log label
    strong_label: Optional[str] = None       # manager-owned label for a strong sibling
    spatial_nodes: Optional[int] = None      # decoding-graph nodes per round (latency models)
    code: Optional[str] = None               # code name, drives CodeRouter routing
    attempt: int = 0                         # 0 = first (weak) decode, 1 = strong redo
    hint: Optional[str] = None               # routing override, e.g. "strong"
    pool: Optional[str] = None               # unit pool assigned at dispatch
    window: Optional[Window] = None          # back-reference to the source window
    strong_decode_for: Optional[tuple] = None      # (op_id, window_id) this strong job re-decodes
    gap_sibling_for: Optional[tuple] = None        # (op_id, window_id) whose split-gap half this job solves
    awaiting_strong_result: bool = False     # weak result held non-final until the strong sibling lands
    cancelled: bool = False                  # cancelled siblings discard completion
    completed: bool = False                  # terminal flag; admission refuses reuse of a completed job
    submitted: bool = False                  # admitted once to one queue slot and unit
    input_landed: bool = False               # the input transfer deposited into unit memory
    service_started: bool = False            # the decode itself began (past the boundary gate)
    request_key: Optional[DecoderRequestKey] = None
    request_created_ticks: Optional[int] = None
    request_admitted_ticks: Optional[int] = None
    service_key: Optional[DecoderServiceKey] = None
    service_original_request_keys: tuple[DecoderRequestKey, ...] = ()
    service_cancelled_request_keys: set[DecoderRequestKey] = field(default_factory=set)
    service_dispatch_ticks: Optional[int] = None


@dataclass(frozen=True)
class LogicalContribution:
    """One decoder prediction owner over an exact inclusive round extent of a stream."""

    owner_key: tuple
    commit_lo: int
    commit_hi: int
    ownership_kind: str
    logical_observables: Optional[tuple[int, ...]]


@dataclass
class DecodeResult:
    """One window result; timing-only decoders leave optional fields unset."""

    op_id: int
    window_id: int
    correction: Optional[Any] = None         # correction operator (None = timing-only)
    logical_observables: Optional[tuple[int, ...]] = None  # full prediction
    soft_output: Optional["SoftOutput"] = None  # source-compatible confidence
    # one forced-class solve's weight, carried to the split-gap join
    # (the gap exists only once both halves have reported)
    gap_half_weight: Optional[float] = None
    boundary_defects: Optional[dict] = None  # round-keyed seam defects (synthetic decoders, recovery lock scenarios)
    boundary_data: Optional[Any] = None      # optional richer interaction payload
    # BackendDecodeStatus of a best-effort correction (nonconverged, low
    # confidence, does not reproduce the syndrome); None when the decode
    # succeeded. The correction is committed either way and the status travels
    # with it, as cudaqx's per-window converged flag does.
    decode_status: Optional[Any] = None


@dataclass
class Submission:
    """One decode job an escalation policy wants enqueued, optionally after a delay.

    A strong redo job gets its ready time when it reaches the queue, so link
    delay is not charged as queue wait.
    """

    job: DecodeJob
    delay_ticks: int = 0


class Directive(Enum):
    """What the core should do with a decode outcome."""
    FINALIZE = auto()          # accept the weak result; cancel any parallel strong
    AWAIT_STRONG = auto()      # hold the weak result; .extra may carry the strong redo
    FINALIZE_STRONG = auto()   # a strong result landed; core applies hold-or-deliver


@dataclass
class OutcomeDirective:
    """An escalation policy's verdict on one decode outcome."""
    directive: Directive
    extra: Optional[Submission] = None
    strong_request_key: Optional[DecoderRequestKey] = None


@dataclass(frozen=True)
class StrongDecodeCompletion:
    """A strong result paired with the request it answers."""

    request_key: DecoderRequestKey
    result: DecodeResult


@dataclass
class DecodeOutcome:
    """Joint decode outcome delivered to the escalation policy hook."""

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


@dataclass(frozen=True)
class RunOperationBody:
    """Immutable controller-to-QPU command for one operation body."""

    operation: Any
    round_ticks: int
    round_count: int
    source_round_count: int
    emits_detector_data: bool = True
    finalizes_stream_round: bool = False

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
    """How many rounds of a dependent operation have arrived, and how many it has."""

    operation_id: int
    rounds_arrived: int
    round_count: int


@dataclass(frozen=True)
class WindowReadiness:
    """What a scheme sees when deciding whether a window has its data."""

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
