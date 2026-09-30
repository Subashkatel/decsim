"""One decode: the job, the result, and the holds that keep its rounds.

A weak decode runs first and fast; an escalated strong decode re-decodes
the same window (Toshio et al. 2510.25222 Sec. III A). The confidence a
decoder reports, the outcome one request ended in, and the shape of the
whole run that the root checks before planning are here too, because the
escalation policy reads them together with the result.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

import decsim.records.windows as window_records


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
    decoded_class_weight: Optional[float] = None
    complementary_class_weight: Optional[float] = None


@dataclass(frozen=True)
class SoftOutputComputation:
    """One window's soft output and what computing it cost.

    ticks are the weak tier's clock ticks the signal's own computation
    took: zero for a subtraction, the measured or declared time of a
    walk over the decode's growth (decision D8, Meister et al.
    2405.07433 Algorithm 2). The unit that produced the evidence is
    charged for them, because the evidence and its reader are the same
    hardware (Toshio et al. 2510.25222 lines 152-160).
    """

    soft_output: Optional[SoftOutput]
    ticks: int = 0


@dataclass(frozen=True)
class WindowConfidence:
    """One window's confidence gap as the verdict read it.

    gap_nats is None when the signal gave no gap, a window whose model
    pins no observable or whose decode grew no cluster, and the policy
    escalates such a window (decsim/escalation/policies.py).
    is_strong_revised says whether the strong decode predicted other
    observables than the weak one it replaced, None for a window that
    did not escalate or whose strong answer never came: a window has no
    truth of its own, so this is the one per-window answer to whether
    the weak decode was wrong (Toshio et al. 2510.25222 lines 807-841
    sign the gap by exactly that).
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


@dataclass(frozen=True)
class WindowReads:
    """A hold: the rounds one window's own decode reads, in its store."""

    window_key: tuple

    def referenced_operation_ids(self) -> tuple:
        """None: the rounds held are the reading window's own."""
        return ()


@dataclass(frozen=True)
class PotentialStrong:
    """A hold: a window's rounds, kept in case its weak result escalates."""

    window_key: tuple

    def referenced_operation_ids(self) -> tuple:
        """None: the rounds held are the held window's own."""
        return ()


@dataclass(frozen=True)
class PotentialRestart:
    """A hold in the weak syndrome buffer: a window's reads and one before them.

    Under the double window an earlier escalation may re-slice
    this window as its restart window, whose weak decode re-reads one
    buffer into the strong region (Toshio 2510.25222 Sec. III C); the
    rounds stay past the window's own request and landing, until the
    window before it commits.
    """

    window_key: tuple

    def referenced_operation_ids(self) -> tuple:
        """None: the rounds held are the held window's own."""
        return ()


@dataclass(frozen=True)
class PendingStrong:
    """A hold: rounds for an admitted, not yet served strong request."""

    request_key: window_records.DecoderRequestKey

    def referenced_operation_ids(self) -> tuple:
        """None: the rounds held are the requested window's own."""
        return ()


@dataclass(frozen=True)
class StrongInputInFlight:
    """A hold: rounds in flight to a strong decoder."""

    request_key: window_records.DecoderRequestKey

    def referenced_operation_ids(self) -> tuple:
        """None: the rounds held are the requested window's own."""
        return ()


@dataclass(frozen=True)
class DecoderInputHold:
    """A hold: a decode job's rounds until they land in unit memory."""

    request_key: window_records.DecoderRequestKey

    def referenced_operation_ids(self) -> tuple:
        """None: the rounds held are the job's own."""
        return ()


@dataclass(frozen=True)
class RephaseGuard:
    """A hold: a rephased suffix's rounds while its strong request is live."""

    request_key: window_records.DecoderRequestKey

    def referenced_operation_ids(self) -> tuple:
        """The guarded request's operation, which the suffix may outlive."""
        return (self.request_key.operation_id,)


@dataclass(frozen=True)
class DecoderServiceKey:
    """Identity of one decoder service (a batch of requests served together)."""

    run_sequence: int


class DecodeJobKind(Enum):
    """What one decode job is, declared once and read by everyone.

    The manager, the queue and the request ledger all need to know what
    a job is before they read it, and a declared kind is how a
    heterogeneous runtime says so: StarPU declares one codelet per
    architecture and Legion one processor kind per task, and the
    scheduler reads the declaration rather than inferring it. WINDOW is
    one window's decode on the tier that owns it, forced-class solves
    included; STRONG_REDECODE is one escalated window on the strong
    tier; STRONG_BATCH is the merged timing-only decode that serves
    several of those under bulk_strong; SELF_CONTAINED is a decode with
    no window and no syndrome, a factory correction or an idle region.
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
    """One unit of decoder work: a window's rounds and its life.

    The window's rounds, its detector error model, its identity in the
    decoder queues, and the timestamps of its life. ``payloads`` is the
    weak syndrome buffer's view of the rounds until the transfer lands them
    in a unit's memory (``decoder_input``); a decoder reads only its unit's
    memory.
    """

    operation_id: int  # operation the window belongs to
    window_id: int  # window index within that op
    # rounds the decoder processes: the distinct rounds landed in its
    # input, plus batched idle rounds
    round_count: int
    detector_error_model: Optional[Any] = (
        None  # window detector error model (data-path decoders)
    )
    payloads: list = field(
        default_factory=list
    )  # transfer-source view; cleared after materialization
    decoder_input: Optional[Any] = None  # materialized decoder memory value
    # the raw round before the first payload, read out of the store with
    # them when the tier's decoder forms the events and has not formed
    # that first round (detection_events, needs_the_round_before); held
    # by the former and never decoded, cleared with the payloads
    round_before: tuple = ()
    input_hold: Optional[Any] = (
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
    attempt: int = 0  # 0 = first (weak) decode, 1 = strong redo
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
    # decode, so the unit stays busy for the signal's own work (D8)
    soft_output_ticks: int = 0
    # the rounds this job's tier turns into detection events for it,
    # frozen at the first ask so the formation stage and the dispatcher
    # read one number; None until then, and under controller-side
    # formation nothing ever asks (decoders/detection_events.py)
    detection_event_rounds: Optional[tuple] = None

    def payload_bits(self) -> Optional[int]:
        """The bits of the job's payloads and the round before them.

        None when any size is unknown.
        """
        payloads = self.payloads or ()
        carried = self.round_before + tuple(payloads)
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

    A confidence signal reads one of these off the decode that produced
    the correction, so each signal declares what it needs and each
    decoder row declares what it produces, the way a decoder declares
    its fault representation. FORCED_CLASS_WEIGHT is the minimum weight
    inside a logical class the solve was pinned to, which only a decoder
    that minimises weight inside the class reports honestly (Lee et al.
    2510.05795 Sec. 2.1.1); CLUSTER_GROWTH is the graph and the radii a
    cluster-based decode grew (Meister et al. 2405.07433 Algorithm 2).
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
    """What one backend call on one window answers.

    ``selected_faults`` is the correction and ``decode_status`` a
    best-effort disposition (None when the decode succeeded).
    ``no_correction_reason`` is the backend's BackendFailureReason when
    it produced no correction and ``selected_faults`` is the empty one
    committed in its place; None when the backend produced one. The two
    evidence fields are what a confidence signal reads off the decode
    that produced the correction: the minimum weight inside the class a
    forced solve was pinned to, and the growth a cluster-based decode
    did (Meister et al. 2405.07433 Algorithm 2 lines 518-525 reads the
    graph and the radii). ``iterations`` is the message-passing
    iterations an iterative decode ran, which is what its time on a
    device scales with. A row leaves what it does not produce None.
    """

    selected_faults: Any
    decode_status: Optional[BackendDecodeStatus] = None
    forced_class_weight: Optional[float] = None
    cluster_evidence: Optional[Any] = None
    iterations: Optional[int] = None
    no_correction_reason: Optional[BackendFailureReason] = None


@dataclass(frozen=True)
class Step:
    """One step of a strong decode on its device, as the device states it.

    name says what the step is (notice, launch, copy_in, decode); ticks
    is its time; resource is the one it holds, dispatcher or worker, or
    None for none. priced_on names where a zero-tick step's time is
    counted instead, such as the link card's echo round trip, so a trace
    shows every step and counts each tick once.
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
    correction: Optional[Any] = None  # correction operator (None = timing-only)
    logical_observables: Optional[tuple[int, ...]] = None  # full prediction
    soft_output: Optional[SoftOutput] = None  # source-compatible confidence
    # the minimum weight inside the class the job was forced to; None
    # when the decode was not forced or the window pins no observable
    forced_class_weight: Optional[float] = None
    # the growth a cluster-based decode did, what a cluster gap reads;
    # None from a row that grows no clusters
    cluster_evidence: Optional[Any] = None
    # the message-passing iterations the decode ran, what a measured
    # device time law reads; None from a row that runs no iterations
    iterations: Optional[int] = None
    # round-keyed seam defects (synthetic decoders, recovery lock
    # scenarios)
    boundary_defects: Optional[dict] = None
    boundary_data: Optional[Any] = None  # optional richer interaction payload
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
    record, once, in Machine.build. strong_window is the row of
    STRONG_WINDOW_SHAPES the escalation section named, so a refusal
    names the shape the yaml chose; is_absorbing_strong_window is that
    row's own declaration that its region replaces the weak windows it
    covers (the double window of Toshio et al. 2510.25222 Sec. III C;
    the redo window absorbs nothing); is_bulk_strong is the
    decoder manager's merging of queued strong re-decodes; operations
    are the workload's planning views; commit_round_count and
    buffer_round_count size every window (windows.commit_rounds and
    windows.buffer_rounds, the code distance when the yaml leaves them
    null).
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
