"""One decode: the job, the result, and the holds that keep its rounds.

A weak decode runs first and fast; an escalated strong decode re-decodes
the same window (Toshio et al. 2510.25222 Sec. III A). The confidence,
the request outcome and the run's shape live here too, because the
escalation policy reads them with the result.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Union

import numpy

import decsim.records.decoder_evidence as evidence_records
import decsim.records.fault_model_contracts as fault_models
import decsim.records.windows as window_records


@dataclass(frozen=True)
class SoftOutputSource:
    """The signal that computed a gap, by its method's name."""

    method: str


@dataclass(frozen=True)
class SoftOutput:
    """One nonnegative confidence gap with immutable interpretation."""

    gap: float
    source: SoftOutputSource
    decoded_class_weight: Optional[float] = None
    complementary_class_weight: Optional[float] = None


@dataclass(frozen=True)
class SoftOutputComputation:
    """One window's soft output and what computing it cost.

    ticks are the weak tier's clock ticks the signal's own computation took:
    zero for a subtraction, the time of a walk over the decode's growth
    (Meister et al. 2405.07433 Algorithm 2). The unit that produced the
    evidence is charged them, since the evidence and its reader are the same
    hardware (Toshio et al. 2510.25222 lines 152-160).
    """

    soft_output: Optional[SoftOutput]
    ticks: int = 0


@dataclass(frozen=True)
class WindowConfidence:
    """One window's confidence gap as the verdict read it.

    gap_nats is None when the signal gave no gap (no observable pinned, no
    cluster grown), and the policy escalates such a window.
    is_strong_revised says whether the strong decode predicted other
    observables than the weak one, None when it did not escalate or no
    strong answer came: a window has no truth of its own, so this is the
    per-window answer to whether the weak decode was wrong (Toshio et al.
    2510.25222 lines 807-841).
    """

    window_key: tuple
    gap_nats: Optional[float]
    is_escalated: bool
    is_strong_revised: Optional[bool]


# ---- consumer hold tokens: who keeps rounds in a syndrome buffer and why
#
# Every token answers referenced_operation_ids: the operations it keeps
# open beyond the ones the rounds it names belong to. A store asks the
# token rather than reading its type, so a token added later is counted
# by the liveness check like every other one.
#
# Every token also answers operation_ids_read_at_once: the operations
# whose rounds, among those it names, the window side waits to see all
# stored at once before it ends. Every token also answers
# holders_waited_for: the holds that end before it can. A bounded store
# asks both while a round waits for room (syndrome_buffer.py).


@dataclass(frozen=True)
class WindowReads:
    """A hold: the rounds one window's own decode reads, in its store."""

    window_key: tuple

    def referenced_operation_ids(self) -> tuple:
        """None: the rounds held are the reading window's own."""
        return ()

    def operation_ids_read_at_once(self) -> tuple:
        """The window's operation: its decode waits for every round of it.

        A buffer past the operation's end is filled by memory rounds or
        by any one successor (windows/schemes/window_data.py), so the
        rounds of other operations it names are not waited for at once.
        """
        operation_id = self.window_key[0]
        return (operation_id,)

    def holders_waited_for(self) -> tuple:
        """None: its decode waits for rounds, not for another hold."""
        return ()


@dataclass(frozen=True)
class PotentialStrong:
    """A hold: a window's rounds, kept in case its weak result escalates."""

    window_key: tuple

    def referenced_operation_ids(self) -> tuple:
        """None: the rounds held are the held window's own."""
        return ()

    def operation_ids_read_at_once(self) -> tuple:
        """None: the hold ends at a weak verdict that keeps the window."""
        return ()

    def holders_waited_for(self) -> tuple:
        """None: an absorbing strong window may end it unread."""
        return ()


@dataclass(frozen=True)
class PotentialRestart:
    """A hold in the weak syndrome buffer: a window's reads and one before them.

    Under the double window an earlier escalation may re-slice this window
    as its restart window, which re-reads one buffer into the strong region
    (Toshio 2510.25222 Sec. III C); the rounds stay until the window before
    it commits.
    """

    window_key: tuple

    def referenced_operation_ids(self) -> tuple:
        """None: the rounds held are the held window's own."""
        return ()

    def operation_ids_read_at_once(self) -> tuple:
        """None: the hold ends when the window before it commits."""
        return ()

    def holders_waited_for(self) -> tuple:
        """None: the window before it may be absorbed rather than read."""
        return ()


@dataclass(frozen=True)
class LaterStreamReads:
    """A hold: the raw rounds a stream's windows still to come read early.

    A stream's window registers only as its rounds arrive, so the stream
    keeps the raw rounds its strong read will need until then.
    """

    stream_id: Any  # an opaque identity

    def referenced_operation_ids(self) -> tuple:
        """None: the rounds held are the stream's own."""
        return ()

    def operation_ids_read_at_once(self) -> tuple:
        """None: the hold moves on as the stream's windows register."""
        return ()

    def holders_waited_for(self) -> tuple:
        """None: the hold moves on as the stream's windows register."""
        return ()


@dataclass(frozen=True)
class PendingStrong:
    """A hold: rounds for an admitted, not yet served strong request.

    restart_key is the window whose weak commit releases the request, or
    None at the operation's end (escalation/strong_window_shapes.py,
    _far_face_conditions); the request key alone names the hold.
    """

    request_key: window_records.DecoderRequestKey
    restart_key: Optional[tuple] = field(default=None, compare=False)

    def referenced_operation_ids(self) -> tuple:
        """None: the rounds held are the requested window's own."""
        return ()

    def operation_ids_read_at_once(self) -> tuple:
        """The request's operation: its job waits for every round stored."""
        return (self.request_key.operation_id,)

    def holders_waited_for(self) -> tuple:
        """The restart window's read, which ends before its weak commit."""
        if self.restart_key is None:
            return ()
        return (WindowReads(self.restart_key),)


@dataclass(frozen=True)
class StrongInputInFlight:
    """A hold: rounds in flight to a strong decoder."""

    request_key: window_records.DecoderRequestKey

    def referenced_operation_ids(self) -> tuple:
        """None: the rounds held are the requested window's own."""
        return ()

    def operation_ids_read_at_once(self) -> tuple:
        """None: its decode has started on rounds already stored."""
        return ()

    def holders_waited_for(self) -> tuple:
        """None: it ends when its rounds land in unit memory."""
        return ()


@dataclass(frozen=True)
class DecoderInputHold:
    """A hold: a decode job's rounds until they land in unit memory.

    boundary_window_keys are the windows whose boundaries the job waits
    for before its input takes a slot (decode_requests.py, may_stage);
    the request key alone names the hold.
    """

    request_key: window_records.DecoderRequestKey
    boundary_window_keys: tuple = field(default=(), compare=False)

    def referenced_operation_ids(self) -> tuple:
        """None: the rounds held are the job's own."""
        return ()

    def operation_ids_read_at_once(self) -> tuple:
        """None: its decode has started on rounds already stored."""
        return ()

    def holders_waited_for(self) -> tuple:
        """The reads of the windows whose boundaries it waits for."""
        waited = []
        for window_key in self.boundary_window_keys:
            reads = WindowReads(window_key)
            waited.append(reads)
        return tuple(waited)


@dataclass(frozen=True)
class RephaseGuard:
    """A hold: a rephased suffix's rounds while its strong request is live."""

    request_key: window_records.DecoderRequestKey

    def referenced_operation_ids(self) -> tuple:
        """The guarded request's operation, which the suffix may outlive."""
        return (self.request_key.operation_id,)

    def operation_ids_read_at_once(self) -> tuple:
        """None: the hold ends when the strong plan lands."""
        return ()

    def holders_waited_for(self) -> tuple:
        """None: the hold ends when the strong plan lands."""
        return ()


# Every hold a syndrome buffer keeps rounds for; the store asks it the
# three questions above and never reads which one it is.
Hold = Union[
    WindowReads,
    PotentialStrong,
    PotentialRestart,
    LaterStreamReads,
    PendingStrong,
    StrongInputInFlight,
    DecoderInputHold,
    RephaseGuard,
]


@dataclass(frozen=True)
class SyndromeBufferingPlan:
    """The holds every window places on the stores.

    A hold names the rounds a consumer keeps alive.
    """

    weak_holds: tuple
    potential_holds: tuple


@dataclass(frozen=True)
class DecoderServiceKey:
    """Identity of one decoder service (a batch of requests served together)."""

    run_sequence: int


class DecodeJobKind(Enum):
    """What one decode job is, declared once and read by everyone.

    A heterogeneous runtime declares a task's kind rather than inferring it,
    as StarPU declares a codelet per architecture and Legion a processor
    kind per task. STRONG_BATCH is the merged timing-only decode of several
    re-decodes under bulk_strong; SELF_CONTAINED has no window and no
    syndrome (a factory correction or an idle region).
    """

    WINDOW = "window"
    STRONG_REDECODE = "strong_redecode"
    STRONG_BATCH = "strong_batch"
    SELF_CONTAINED = "self_contained"


class RequestProcessingOutcome(Enum):
    """How one decode request ended, for the switching study's records."""

    PRIMARY_FORWARDED_FOR_DELIVERY = "primary_forwarded_for_delivery"
    WEAK_AWAITED_STRONG = "weak_awaited_strong"
    WEAK_FORCED_CLASS_COMPANION = "weak_forced_class_companion"
    STRONG_FORWARDED_FOR_DELIVERY = "strong_forwarded_for_delivery"
    STRONG_COMPLETED_DISCARDED = "strong_completed_discarded"
    STRONG_CANCELLED_BEFORE_DISPATCH = "strong_cancelled_before_dispatch"
    STRONG_CANCELLED_WHILE_STAGED = "strong_cancelled_while_staged"
    STRONG_CANCELLED_DURING_SERVICE = "strong_cancelled_during_service"
    STRONG_CANCELLED_MEMBER_SERVICE_CONTINUED = (
        "strong_cancelled_member_service_continued"
    )
    WEAK_WITHDRAWN_FOR_STRONG_WINDOW = "weak_withdrawn_for_strong_window"


def distinct_round_count(payloads: list) -> int:
    """The distinct (operation_id, round_index) rounds of the payloads.

    A decode job is priced and admitted for these: a sliding-window
    decoder's work scales with its window's rounds (Skoric et al.
    2209.08552), a final window may be smaller (Tan et al. 2209.09219), and
    no window implementation feeds rounds beyond the data.
    """
    round_identities = set()
    for payload in payloads:
        round_identities.add((payload.operation_id, payload.round_index))
    return len(round_identities)


@dataclass
class DecodeJob:
    """One unit of decoder work: a window's rounds and its life.

    payloads is the weak syndrome buffer's view of the rounds until the
    transfer lands them in a unit's memory (decoder_input); a decoder reads
    only its unit's memory.
    """

    operation_id: int  # operation the window belongs to
    window_id: int  # window index within that op
    # rounds the decoder processes: the distinct rounds landed in its
    # input, plus batched idle rounds
    round_count: int
    # the window's error model, which a data-path decoder reads
    detector_error_model: Optional[fault_models.WindowErrorModel] = None
    payloads: list = field(
        default_factory=list
    )  # transfer-source view; cleared after materialization
    decoder_input: Optional[Any] = None  # materialized decoder memory value
    # the raw rounds before the first payload, in round order, read out
    # of the store with them when the tier's decoder forms the events
    # and has not formed that first round (detection_events,
    # rounds_needed_before); held by the former and never decoded,
    # cleared with the payloads
    rounds_before: tuple = ()
    input_hold: Optional[Callable[[], None]] = (
        None  # upstream hold released at transfer completion
    )
    # the WindowInputGate the decoder manager asks before staging, before
    # starting and when masking the landed input; None for a windowless job
    gate: Optional[Any] = None
    # where the result goes: on_decoded(job, result), set at enqueue
    on_decoded: Optional[Callable] = None
    # called at dispatch: send the input link, call back at the landing,
    # return the expected delay in ticks
    send_input: Optional[Callable[[Callable[[], None]], int]] = None
    # the store the input leaves from, stamped by that store's own port
    input_source_name: Optional[str] = None
    unit: Optional[Any] = None  # the DecoderUnit assigned at dispatch
    # tick a unit took this decode, the end of its own queue wait; the
    # window record keeps the last one, this keeps each decode's own
    dispatch_ticks: Optional[int] = None
    # tick this decode first may compute: its input has landed in the
    # unit's memory and its window owes no boundary. What it waits for
    # after this tick is the unit's compute, not a dependency
    ready_ticks: Optional[int] = None
    # the ticks this decode waited inside a strong backend for its
    # dispatcher or a worker, summed over its steps, as a gem5 instruction
    # carries its own stage ticks (src/cpu/o3/dyn_inst.hh:1017-1028);
    # zero on a decoder with no queue of its own
    backend_queue_wait_ticks: int = 0
    # the ticks this decode's input read took in its tier's store, from
    # the dispatch that asked for it to the read's end, its port waits
    # included; zero on a store that prices no read
    store_read_ticks: int = 0
    memory: Optional[Any] = (
        None  # that unit's DecoderMemory while it holds this job's input
    )
    ready_time: int = 0  # tick the job was enqueued (queue-wait accounting)
    on_done: Optional[Callable[[], None]] = None  # completion callback
    label: str = ""  # log label
    strong_label: Optional[str] = (
        None  # manager-owned label for a speculative strong decode
    )
    spatial_nodes: Optional[int] = (
        None  # decoding-graph nodes per round (latency models)
    )
    code: Optional[str] = None  # code name, a latency function may read it
    kind: DecodeJobKind = DecodeJobKind.WINDOW  # what this job is
    # the logical class this decode is pinned to, or None to decode
    # normally; a forced solve reports that class's minimum weight
    # (Gidney et al. 2312.04522, the augmented virtual detector)
    forced_logical_class: Optional[int] = None
    pool: Optional[str] = None  # unit pool assigned at dispatch
    # back-reference to the source window
    window: Optional[window_records.Window] = None
    strong_decode_for: Optional[tuple] = (
        None  # (operation_id, window_id) this strong job re-decodes
    )
    cancelled: bool = False  # cancelled speculative decodes discard completion
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
    request_key: Optional[window_records.DecoderRequestKey] = None
    # the request whose transfer brought the rounds this job reads; the
    # jobs that share one landed input share this key and a unit holds
    # one copy per input, never one per job
    input_key: Optional[window_records.DecoderRequestKey] = None
    request_created_ticks: Optional[int] = None
    request_admitted_ticks: Optional[int] = None
    service_key: Optional[DecoderServiceKey] = None
    service_original_request_keys: tuple[
        window_records.DecoderRequestKey, ...
    ] = ()
    service_dispatch_ticks: Optional[int] = None
    # the unit that ran this decode, by name, kept after the job leaves
    # its slot so the confidence its evidence feeds is attributed to it
    decoding_unit_name: Optional[str] = None
    # ticks of confidence computation charged on that unit after the
    # decode, so the unit stays busy for the signal's own work
    soft_output_ticks: int = 0
    # the rounds this job's tier turns into detection events for it,
    # frozen at the first ask so the formation stage and the dispatcher
    # read one number; None until then, and under controller-side
    # formation nothing ever asks (decoders/detection_events.py)
    detection_event_rounds: Optional[tuple] = None

    def payload_bits(self) -> Optional[int]:
        """The bits of the job's payloads and the rounds before them.

        None when any size is unknown.
        """
        payloads = self.payloads or ()
        carried = self.rounds_before + tuple(payloads)
        sizes = []
        for payload in carried:
            sizes.append(payload.size_bits)
        for size in sizes:
            if size is None:
                return None
        return sum(sizes)


@dataclass(frozen=True)
class LogicalContribution:
    """One decoder prediction's owner over one extent of a stream.

    The extent is an inclusive round range, and the owner is the window
    key whose result predicted those rounds.
    """

    owner_key: tuple
    commit_lo: int
    commit_hi: int
    ownership_kind: str
    logical_observables: Optional[tuple[int, ...]]


class DecoderEvidence(Enum):
    """What a decode can show about itself, beyond its correction.

    Each signal declares what it needs and each decoder row what it
    produces, as a decoder declares its fault representation.
    FORCED_CLASS_WEIGHT is the minimum weight inside a pinned logical
    class, honest only from a decoder that minimises weight inside it (Lee
    et al. 2510.05795 Sec. 2.1.1); CLUSTER_GROWTH is the graph and radii a
    cluster decode grew (Meister et al. 2405.07433 Algorithm 2).
    """

    FORCED_CLASS_WEIGHT = "forced_class_weight"
    CLUSTER_GROWTH = "cluster_growth"


# The named capabilities follow the sets they are built from.
NO_DECODER_EVIDENCE = frozenset()
FORCED_CLASS_SOLVES = frozenset({DecoderEvidence.FORCED_CLASS_WEIGHT})
CLUSTER_GROWTH_EVIDENCE = frozenset({DecoderEvidence.CLUSTER_GROWTH})


class BackendDecodeStatus(Enum):
    """Backend-neutral disposition of one window decode attempt."""

    SUCCEEDED = "succeeded"
    LOW_CONFIDENCE = "low_confidence"
    NONCONVERGED = "nonconverged"
    INVALID_CORRECTION = "invalid_correction"
    EMPTY_MODEL_UNSATISFIABLE = "empty_model_unsatisfiable"
    BACKEND_ERROR = "backend_error"


class BackendFailureReason(Enum):
    """Typed reason a backend attempt could not be committed."""

    SEARCH_LIMIT_EXHAUSTED = "search_limit_exhausted"
    NO_CONVERGED_RELAY_SOLUTION = "no_converged_relay_solution"
    CORRECTION_NOT_BINARY = "correction_not_binary"
    CORRECTION_WRONG_ARITY = "correction_wrong_arity"
    CORRECTION_DOES_NOT_MATCH_SYNDROME = "correction_does_not_match_syndrome"
    NONZERO_SYNDROME_WITHOUT_FAULTS = "nonzero_syndrome_without_faults"
    UPSTREAM_EXCEPTION = "upstream_exception"
    # PyMatching raises on a syndrome no matching explains; the matching
    # rows report it with no correction under INVALID_CORRECTION
    NO_PERFECT_MATCHING = "no_perfect_matching"


@dataclass(frozen=True)
class WindowDecode:
    """What one backend call on one window answers; unproduced fields None."""

    selected_faults: Union[numpy.ndarray, tuple[int, ...]]
    decode_status: Optional[BackendDecodeStatus] = None
    forced_class_weight: Optional[float] = None
    cluster_evidence: Optional[evidence_records.UnionFindHardEvidence] = None
    iterations: Optional[int] = None
    no_correction_reason: Optional[BackendFailureReason] = None


@dataclass(frozen=True)
class Step:
    """One step of a strong decode on its device, as the device states it.

    priced_on names where a zero-tick step's time is counted instead, such
    as the link card's echo round trip, so a trace shows every step and
    counts each tick once.
    """

    name: str
    ticks: int
    resource: Optional[str] = None
    priced_on: Optional[str] = None


@dataclass
class DecodeResult:
    """One window result; timing-only decoders leave optional fields unset."""

    operation_id: int
    window_id: int
    # correction operator (None = timing-only)
    correction: Optional[numpy.ndarray] = None
    logical_observables: Optional[tuple[int, ...]] = None  # full prediction
    soft_output: Optional[SoftOutput] = None  # source-compatible confidence
    # the minimum weight inside the class the job was forced to; None
    # when the decode was not forced or the window pins no observable
    forced_class_weight: Optional[float] = None
    # the growth a cluster-based decode did, what a cluster gap reads;
    # None from a row that grows no clusters
    cluster_evidence: Optional[evidence_records.UnionFindHardEvidence] = None
    # the window's detection events the decode read, in its detector
    # rows' order and read-only; None from a row that reads none
    detection_events: Optional[numpy.ndarray] = None
    # the message-passing iterations the decode ran, what a measured
    # device time law reads; None from a row that runs no iterations
    iterations: Optional[int] = None
    # round-keyed seam defects a decoder may report in place of a
    # residual in boundary_data
    boundary_defects: Optional[dict] = None
    # optional richer interaction payload
    boundary_data: Optional[window_records.DependencyResidual] = None
    # CrossingCommit: the part of the correction that commits faults
    # touching a round before the window's commit region, which the
    # residual's XOR cannot be split into afterwards
    crossing_commit: Optional[window_records.CrossingCommit] = None
    # BackendDecodeStatus of a best-effort correction (nonconverged, low
    # confidence, does not reproduce the syndrome); None when the decode
    # succeeded. The correction is committed either way and the status travels
    # with it, as cudaqx's per-window converged flag does.
    decode_status: Optional[BackendDecodeStatus] = None
    # BackendFailureReason of a backend that produced no correction, whose
    # empty stand-in is the correction above; None when it produced one
    no_correction_reason: Optional[BackendFailureReason] = None


@dataclass(frozen=True)
class Ticket:
    """One submitted strong decode: decsim's answer and its priced ticks."""

    result: DecodeResult
    decode_ticks: int


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

    ESCALATE commits the weak result provisionally and asks the strong tier
    to re-decode (Toshio et al. 2510.25222 Sec. III A, steps 3 and 4).
    """

    KEEP = auto()
    ESCALATE = auto()


@dataclass(frozen=True)
class RunShape:
    """What a run is made of, as the root checks it before planning.

    The escalation policy refuses a run it cannot serve from this, once, in
    Machine.build. is_absorbing_strong_window is the strong window row's
    declaration that its region replaces the weak windows it covers (the
    double window, Toshio et al. 2510.25222 Sec. III C). commit_round_count
    and buffer_round_count size every window.
    """

    scheme: Any
    boundary_policy: Any
    operations: tuple
    commit_round_count: int
    buffer_round_count: int
    strong_window: str
    is_absorbing_strong_window: bool
    is_bulk_strong: bool
    has_dynamic_streams: bool
    has_static_decode_plan: bool
