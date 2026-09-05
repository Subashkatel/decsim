"""The vocabulary every component speaks: the frozen values that travel
between the QPU, the controller, Buffer 0, the window manager, the decoders
and the Pauli frame (readouts, payloads, packets, windows, plans, jobs,
results, boundaries, requests, seeds), and the stable-identity helpers that
make operation and window keys hashable and orderable across types. Nothing
here has behavior beyond a value's own derived views."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Union


def is_stable_string(value: Any) -> bool:
    return type(value) is str and all(
        not 0xD800 <= ord(character) <= 0xDFFF for character in value
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
        return len(left) == len(right) and all(
            same_stable_identity(left_item, right_item)
            for left_item, right_item in zip(left, right)
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
            len(item).to_bytes(8, "big") + item for item in encoded_items
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
            return b"I" + len(encoded_value).to_bytes(4, "big") + encoded_value
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
    def feedback_memory_round(
        cls, source_operation_id
    ) -> "SyndromePacketRoute":
        return cls(
            SyndromePacketRouteKind.FEEDBACK_MEMORY_ROUND, source_operation_id
        )


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
    def from_readout(cls, readout: QPUReadout) -> "RetainedSyndromeFragment":
        """The readout as controller binary: its bits normalized."""
        bits = normalize_binary_bits(readout.bits)
        return cls(
            operation_id=readout.operation_id,
            patch_id=readout.patch_id,
            round_index=readout.round_index,
            bits=bits,
            size_bits=readout.size_bits,
            fragment_index=readout.fragment_index,
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
        return (
            f"defects {{{', '.join(map(str, defects))}}}"
            if defects
            else "no defects"
        )


@dataclass(frozen=True)
class PackedRound:
    """A finished round as it leaves the assembler.

    The packet, its route, and its size on the wire: the raw measurement
    bits, before detection formation, which is what the links carry.
    """

    packet: SyndromeRoundPacket
    route: SyndromePacketRoute
    wire_bits: Optional[int]

    @property
    def round_key(self) -> tuple:
        """(operation_id, round_index), the store's key."""
        return (self.packet.operation_id, self.packet.round_index)


@dataclass(frozen=True)
class RoundEvent:
    """One recorded transition of one syndrome round through the controller.

    kind is one of EMITTED, BINARY_AVAILABLE, PACKED, STALLED, CWB_SENT,
    PUBLISHED, DROPPED, FEEDBACK_MEMORY_DELIVERED.
    """

    kind: str
    tick: int
    operation_id: object
    round_index: int
    patch_id: object = None
    route: str = ""


@dataclass(frozen=True)
class ControllerOutputEvent:
    """One transition on the controller's digital-to-QPU path.

    payload is the decision or QPU command itself, so a ledger can prove
    that the data whose timing was modeled is the data the QPU received.
    """

    kind: str
    tick: int
    operation_id: object
    payload: object


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

    op_id: int  # operation that owns the stream
    k: int  # window index within the op; key = (op_id, k)
    commit_lo: int  # first round this window commits
    commit_hi: int  # last round this window commits
    buffer_hi: int  # last round it reads (trailing buffer)
    n_rounds: int  # planned rounds from start_round to buffer_hi; the job is priced for the rounds that exist
    buffer_lo: Optional[int] = (
        None  # leading-buffer start (for two-sided A windows)
    )
    closed_temporal_boundaries: bool = False
    batched_preceding_idle_round_count: int = 0
    deps: list = field(default_factory=list)  # window keys this one waits on
    dependents: list = field(
        default_factory=list
    )  # window keys waiting on this one
    deps_remaining: int = 0  # unfinished deps countdown; 0 = unblocked
    service_began: bool = False  # its decode is past the boundary gate
    committed: bool = False  # result folded into the op's accumulator
    # a strong window covers it: the weak chain skips it (never decoded)
    is_absorbed: bool = False
    # the request whose result the window finally published; None until
    # the final one, so a provisional weak commit is still awaiting strong
    published_request_key: Optional["DecoderRequestKey"] = None
    queued: bool = False  # job handed to the decoder cluster
    blocked_logged: bool = False  # log-once flag for the "blocked" trace line
    boundary_in: Any = field(default_factory=dict)  # state owned by the
    # configured WindowInteraction
    decode_status: Optional[str] = (
        None  # best-effort status of the committed decode, None = succeeded
    )
    t_first_round: Optional[int] = None  # tick the first round arrived
    t_data_complete: Optional[int] = (
        None  # tick the last buffered round arrived
    )
    t_queued: Optional[int] = None  # tick the job entered the decode queue
    t_dispatch: Optional[int] = None  # tick a decoder unit started it
    t_done: Optional[int] = None  # tick the decode finished

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
                None if detector_positions is None else dict(detector_positions)
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

    windows: dict  # (op_id, k) -> Window
    window_count: dict  # op_id -> number of windows
    op_windows: dict  # op_id -> [window keys, in k order]
    successors: dict  # op_id -> [op ids listing it as predecessor]
    spatial_nodes: dict  # op_id -> decoding-graph nodes per round
    rounds_by_operation: dict  # op_id -> resolved positive round count
    code_names: dict  # op_id -> exact resolved code name
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


# ---- consumer hold tokens: who keeps rounds in a round store and why


@dataclass(frozen=True)
class PotentialStrong:
    """A hold: a window's rounds, kept in case its weak result escalates."""

    window_key: tuple


@dataclass(frozen=True)
class PendingStrong:
    """A hold: rounds for an admitted, not yet served strong request."""

    request_key: DecoderRequestKey


@dataclass(frozen=True)
class StrongInputInFlight:
    """A hold: rounds in flight to a strong decoder."""

    request_key: DecoderRequestKey


@dataclass(frozen=True)
class DecoderInputHold:
    """A hold: a decode job's rounds until they land in unit memory."""

    request_key: DecoderRequestKey


@dataclass(frozen=True)
class RephaseGuard:
    """A hold: a rephased suffix's rounds while its strong request is live."""

    request_key: DecoderRequestKey


@dataclass(frozen=True)
class DecoderServiceKey:
    """Identity of one decoder service (a batch of requests served together)."""

    run_sequence: int


class RequestProcessingOutcome(Enum):
    """How one decode request ended, for the switching study's records."""

    PRIMARY_FORWARDED_FOR_DELIVERY = "primary_forwarded_for_delivery"
    WEAK_AWAITED_STRONG = "weak_awaited_strong"
    STRONG_FORWARDED_FOR_DELIVERY = "strong_forwarded_for_delivery"
    STRONG_COMPLETED_DISCARDED = "strong_completed_discarded"
    STRONG_CANCELLED_BEFORE_DISPATCH = "strong_cancelled_before_dispatch"
    STRONG_CANCELLED_WHILE_STAGED = "strong_cancelled_while_staged"
    STRONG_CANCELLED_DURING_SERVICE = "strong_cancelled_during_service"
    STRONG_CANCELLED_MEMBER_SERVICE_CONTINUED = (
        "strong_cancelled_member_service_continued"
    )
    WEAK_WITHDRAWN_FOR_STRONG_WINDOW = "weak_withdrawn_for_strong_window"


def distinct_round_count(payloads) -> int:
    """The distinct syndrome rounds a set of payloads carries.

    The number of (operation_id, round_index) identities, the rounds a
    decode job is priced and admitted for, the rounds the decoder reads:
    a sliding-window decoder's work scales with the rounds in its window
    (Skoric et al. 2209.08552, tau_W over n_W), a final window can be
    smaller than a regular one with the whole window as core (Tan et al.
    2209.09219), and no window implementation feeds rounds beyond the
    data (Gong et al. sliding-window decoder; cudaq-qec sliding_window).
    """
    round_identities = set()
    for payload in payloads:
        round_identities.add((payload.operation_id, payload.round_index))
    return len(round_identities)


@dataclass
class DecodeJob:
    """One unit of decoder work: a window's rounds, its model, its identity
    in the decoder queues, and the timestamps of its life. ``payloads`` is the
    Buffer 0 view of the rounds until the transfer lands them in a unit's
    memory (``decoder_input``); a decoder reads only its unit's memory.
    """

    op_id: int  # operation the window belongs to
    window_id: int  # window index within that op
    n_rounds: int  # rounds the decoder processes: the distinct rounds landed in its input, plus batched idle rounds
    dem: Optional[Any] = (
        None  # window detector error model (data-path decoders)
    )
    payloads: list = field(
        default_factory=list
    )  # transfer-source view; cleared after materialization
    decoder_input: Optional[Any] = None  # materialized decoder memory value
    input_hold: Optional[Any] = (
        None  # upstream hold released at transfer completion
    )
    # the WindowInputGate the decoder manager asks before staging, before
    # starting and when masking the landed input; None for a windowless job
    gate: Optional[Any] = None
    # where the result goes: on_decoded(job, result), set at enqueue
    on_decoded: Optional[Callable] = None
    send_input: Optional[Callable[[Callable[[], None]], int]] = (
        None  # called at dispatch: send the input link, call back at the landing, return the expected delay in ticks
    )
    unit: Optional[Any] = None  # the DecoderUnit assigned at dispatch
    memory: Optional[Any] = (
        None  # that unit's DecoderMemory while it holds this job's input
    )
    ready_time: int = 0  # tick the job was enqueued (queue-wait accounting)
    on_done: Optional[Callable[[], None]] = None  # completion callback
    label: str = ""  # log label
    strong_label: Optional[str] = (
        None  # manager-owned label for a strong sibling
    )
    spatial_nodes: Optional[int] = (
        None  # decoding-graph nodes per round (latency models)
    )
    code: Optional[str] = None  # code name, drives CodeRouter routing
    attempt: int = 0  # 0 = first (weak) decode, 1 = strong redo
    hint: Optional[str] = None  # routing override, e.g. "strong"
    pool: Optional[str] = None  # unit pool assigned at dispatch
    window: Optional[Window] = None  # back-reference to the source window
    strong_decode_for: Optional[tuple] = (
        None  # (op_id, window_id) this strong job re-decodes
    )
    gap_sibling_for: Optional[tuple] = (
        None  # (op_id, window_id) whose split-gap half this job solves
    )
    awaiting_strong_result: bool = (
        False  # weak result held non-final until the strong sibling lands
    )
    cancelled: bool = False  # cancelled siblings discard completion
    completed: bool = (
        False  # terminal flag; admission refuses reuse of a completed job
    )
    submitted: bool = False  # admitted once to one queue slot and unit
    input_landed: bool = False  # the input transfer deposited into unit memory
    input_landing_ticks: Optional[int] = (
        None  # tick the staged input lands (set at DMA start)
    )
    service_started: bool = (
        False  # the decode itself began (past the boundary gate)
    )
    # landed in its slot with a boundary still owed: holds the slot,
    # never the unit's compute, until release_parked
    is_parked: bool = False
    request_key: Optional[DecoderRequestKey] = None
    request_created_ticks: Optional[int] = None
    request_admitted_ticks: Optional[int] = None
    service_key: Optional[DecoderServiceKey] = None
    service_original_request_keys: tuple[DecoderRequestKey, ...] = ()
    service_cancelled_request_keys: set[DecoderRequestKey] = field(
        default_factory=set
    )
    service_dispatch_ticks: Optional[int] = None

    def payload_bits(self) -> Optional[int]:
        """The bits of the job's payloads; None when any size is unknown."""
        payloads = self.payloads or ()
        sizes = []
        for payload in payloads:
            sizes.append(payload.size_bits)
        for size in sizes:
            if size is None:
                return None
        return sum(sizes)


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
    correction: Optional[Any] = None  # correction operator (None = timing-only)
    logical_observables: Optional[tuple[int, ...]] = None  # full prediction
    soft_output: Optional["SoftOutput"] = None  # source-compatible confidence
    # one forced-class solve's weight, carried to the split-gap join
    # (the gap exists only once both halves have reported)
    gap_half_weight: Optional[float] = None
    boundary_defects: Optional[dict] = (
        None  # round-keyed seam defects (synthetic decoders, recovery lock scenarios)
    )
    boundary_data: Optional[Any] = None  # optional richer interaction payload
    # BackendDecodeStatus of a best-effort correction (nonconverged, low
    # confidence, does not reproduce the syndrome); None when the decode
    # succeeded. The correction is committed either way and the status travels
    # with it, as cudaqx's per-window converged flag does.
    decode_status: Optional[Any] = None


@dataclass
class Submission:
    """One decode job an escalation policy wants enqueued, with its input send.

    send_input(on_landed) moves the job's input at dispatch and returns
    the delay the link expects; None for a job that carries no input.
    """

    job: DecodeJob
    send_input: Optional[Callable[[Callable[[], None]], int]] = None


class Verdict(Enum):
    """The escalation policy's answer to one weak result.

    KEEP commits the weak result as final; ESCALATE commits it
    provisionally and asks the strong tier to re-decode the window
    (Toshio et al. 2510.25222 Sec. III A, steps 3 and 4).
    """

    KEEP = auto()
    ESCALATE = auto()


@dataclass(frozen=True)
class RunShape:
    """What a run is made of, as the root checks it before planning.

    The escalation policy refuses a run it cannot serve from this
    record, once, in Machine.build. is_double_window is the strong
    window's shape (Toshio et al. 2510.25222 Sec. III C when true, the
    two-sided context of Sec. III A otherwise); is_bulk_strong is the
    decoder manager's merging of queued strong re-decodes; operations
    are the workload's planning views.
    """

    scheme: Any
    boundary_policy: Any
    operations: tuple
    is_double_window: bool
    is_bulk_strong: bool
    has_dynamic_streams: bool
    has_static_decode_plan: bool
    has_frontend: bool


# -------------------------------------------------------------------- links


class LinkPath(str, Enum):
    """The hops of the reaction path, one per pair of components, in the
    order the reports list them. The weak buffer is syndrome buffer 0,
    the strong buffer syndrome buffer 1; the two controller-to-buffer
    hops are optional on a card."""

    QPU_TO_CONTROLLER = "qpu_to_controller"  # a readout
    CONTROLLER_TO_WEAK_BUFFER = "controller_to_weak_buffer"  # a published round
    WEAK_BUFFER_TO_WEAK_DECODER = (
        "weak_buffer_to_weak_decoder"  # a window, or a feedback-memory round
    )
    WEAK_DECODER_TO_STRONG_DECODER = (
        "weak_decoder_to_strong_decoder"  # an escalation
    )
    STRONG_BUFFER_TO_STRONG_DECODER = (
        "strong_buffer_to_strong_decoder"  # the strong window's input
    )
    WEAK_DECODER_TO_FRAME = "weak_decoder_to_frame"  # the weak correction
    DECODER_TO_DECODER = "decoder_to_decoder"  # a committed window boundary
    STRONG_DECODER_TO_FRAME = "strong_decoder_to_frame"  # the strong correction
    FRAME_TO_CONTROLLER = "frame_to_controller"  # the conditional release
    CONTROLLER_TO_QPU = "controller_to_qpu"  # the instruction back
    CONTROLLER_TO_STRONG_BUFFER = (
        "controller_to_strong_buffer"  # the room-side write
    )


@dataclass(frozen=True)
class RequestTransferRelation:
    """Provenance tying one transfer to the decoder request it serves."""

    request_key: DecoderRequestKey


@dataclass(frozen=True)
class BoundaryTransferRelation:
    """Provenance tying one decoder-to-decoder transfer to the boundary it
    delivers: from which window, produced by which request, to which
    window, and both revisions."""

    source_request_key: DecoderRequestKey
    source_window_key: tuple
    destination_window_key: tuple
    source_revision: int
    delivery_revision: int


@dataclass(frozen=True)
class TransferAttribution:
    """Whose transfer this is: the operation, its patches, the window or
    the inclusive round range the bits belong to, and the relation the
    path's rule asks for."""

    # An operation id is whatever the front end chose; the links never
    # look inside it.
    operation_id: Any
    patch_ids: tuple
    window_id: Optional[int]
    first_round: Optional[int]
    last_round: Optional[int]
    relation: Optional[
        Union[RequestTransferRelation, BoundaryTransferRelation]
    ] = None

    @classmethod
    def for_round(
        cls, operation_id, patch_ids: tuple, round_index: int
    ) -> "TransferAttribution":
        """One round of one operation, its patches in stable order."""
        ordered_patch_ids = tuple(
            sorted(patch_ids, key=stable_identity_order_key)
        )
        return cls(
            operation_id=operation_id,
            patch_ids=ordered_patch_ids,
            window_id=None,
            first_round=round_index,
            last_round=round_index,
        )

    @classmethod
    def for_window(
        cls, window: Window, operation: Operation, request_key
    ) -> "TransferAttribution":
        """A window's transfer: the operation's patches, the rounds it reads."""
        ordered_patches = sorted(
            operation.patches, key=stable_identity_order_key
        )
        first_round, last_round = _read_range(window)
        return cls(
            operation_id=operation.id,
            patch_ids=tuple(ordered_patches),
            window_id=window.k,
            first_round=first_round,
            last_round=last_round,
            relation=RequestTransferRelation(request_key),
        )

    @classmethod
    def for_job(cls, job: DecodeJob, request_key) -> "TransferAttribution":
        """A job's transfer: the patches of its payloads, its window's rounds."""
        payloads = job.payloads or ()
        patches = {}
        for payload in payloads:
            patch_id = payload.patch_id
            order_key = stable_identity_order_key(patch_id)
            patches[order_key] = patch_id
        ordered_keys = sorted(patches)
        patch_ids = tuple(patches[key] for key in ordered_keys)
        window = job.window
        assert window is not None, (
            "window-scoped transport requires a DecodeJob window"
        )
        first_round, last_round = _read_range(window)
        return cls(
            operation_id=job.op_id,
            patch_ids=patch_ids,
            window_id=job.window_id,
            first_round=first_round,
            last_round=last_round,
            relation=RequestTransferRelation(request_key),
        )

    @classmethod
    def for_packet(cls, packet: SyndromeRoundPacket) -> "TransferAttribution":
        """The packed round's transfer: every patch of the round."""
        patch_ids = tuple(fragment.patch_id for fragment in packet.fragments)
        return cls.for_round(packet.operation_id, patch_ids, packet.round_index)


def _read_range(window: Window) -> tuple:
    """The inclusive round range a window reads."""
    first_round = window.commit_lo
    if window.buffer_lo is not None:
        first_round = window.buffer_lo
    last_round = window.commit_hi
    if window.buffer_hi is not None:
        last_round = window.buffer_hi
    return first_round, last_round


class PayloadSelection(Enum):
    """Where a transfer's payload size came from."""

    ACTUAL = "actual"
    CONFIGURED_DEFAULT = "configured_default"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Transfer:
    """One transfer's timing on its channel, complete at delivery.

    The sender asked at request_ticks. The setup ended at send_ticks, when
    the transfer reached the wire's queue; the wire took it at
    serializer_start_ticks and let its last bit go at
    serializer_end_ticks; the receiver has it at delivery_ticks, one
    propagation later. total_delay_ticks counts from the request.
    """

    payload_bits: Optional[int]
    request_ticks: int
    setup_ticks: int
    send_ticks: int
    queue_wait_ticks: int
    serialization_ticks: int
    propagation_ticks: int
    serializer_start_ticks: int
    serializer_end_ticks: int
    delivery_ticks: int
    total_delay_ticks: int
    physical_sequence: int


@dataclass(frozen=True)
class TransferRecord:
    """One ledger entry: which path, on which channel, for whom, with
    which payload, and the transfer it got. request_sequence orders the
    ledger by request, whatever order the wires delivered."""

    request_sequence: int
    path: LinkPath
    channel: str
    attribution: TransferAttribution
    payload_selection: PayloadSelection
    payload_source: Optional[str]
    transfer: Transfer


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
    name: str  # human-readable label used in traces
    qubits: tuple  # logical qubit ids the op acts on
    clifford: bool = True  # non-Clifford implies a magic state by default
    circuit: Optional[Any] = (
        None  # stim circuit for real-syndrome (data-path) runs
    )
    consumes_magic_state: Optional[bool] = (
        None  # override; None = infer from clifford
    )
    patches: tuple = ()  # patch ids whose syndrome streams feed the op
    predecessors: tuple = ()  # workload op ids that must complete first
    decoder_boundary_predecessors: tuple = ()  # prior decode streams at a boundary
    # Decode stream this segment's rounds fold into. Seeded StimDevice runs
    # require an exact built-in int/str; unseeded and non-Stim devices may use
    # another identity type accepted by that device.
    stream_id: Optional[Any] = None
    stream_offset: Optional[int] = (
        None  # global-round offset of the segment in its stream
    )
    scheduled_start_round: int = 0
    emits_detector_data: bool = True
    finalizes_stream_round: bool = False
    syndrome_fragment_index: Optional[int] = None
    syndrome_fragment_count: Optional[int] = None
    blocked_by: Optional[int] = (
        None  # op id whose Decision must release this op
    )
    feedback_boundary_mode: Optional[str] = (
        None  # per-op override of the RunSpec mode
    )
    requires_result_return_to_qpu: bool = (
        False  # decision must travel back to the QPU
    )
    kind: OpKind = OpKind.GENERIC  # rounds-policy vocabulary (see OpKind)

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
