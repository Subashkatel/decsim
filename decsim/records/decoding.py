"""One decode: the job, the result, and the holds that keep its rounds.

A weak decode runs first and fast; an escalated strong decode re-decodes
the same window (Toshio et al. 2510.25222 Sec. III A). The confidence a
decoder reports, the outcome one request ended in, and the shape of the
whole run that the root checks before planning are here too, because the
escalation policy reads them together with the result.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

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


# ---- consumer hold tokens: who keeps rounds in a round store and why


@dataclass(frozen=True)
class PotentialStrong:
    """A hold: a window's rounds, kept in case its weak result escalates."""

    window_key: tuple


@dataclass(frozen=True)
class PotentialRestart:
    """A hold: a window's reads and one buffer before them, in Buffer 0.

    Under the double window an earlier escalation may re-slice the
    window as its restart window, whose weak decode re-reads one buffer
    into the strong region (Toshio 2510.25222 Sec. III C); the rounds
    stay past the window's own request and landing, until the window
    before it commits.
    """

    window_key: tuple


@dataclass(frozen=True)
class PendingStrong:
    """A hold: rounds for an admitted, not yet served strong request."""

    request_key: window_records.DecoderRequestKey


@dataclass(frozen=True)
class StrongInputInFlight:
    """A hold: rounds in flight to a strong decoder."""

    request_key: window_records.DecoderRequestKey


@dataclass(frozen=True)
class DecoderInputHold:
    """A hold: a decode job's rounds until they land in unit memory."""

    request_key: window_records.DecoderRequestKey


@dataclass(frozen=True)
class RephaseGuard:
    """A hold: a rephased suffix's rounds while its strong request is live."""

    request_key: window_records.DecoderRequestKey


@dataclass(frozen=True)
class DecoderServiceKey:
    """Identity of one decoder service (a batch of requests served together)."""

    run_sequence: int


class DecodeJobKind(Enum):
    """What one decode job is, declared once and read by everyone.

    The manager, the queue, the router and the request ledger all need
    to know what a job is before they read it, and a declared kind is
    how a heterogeneous runtime says so: StarPU declares one codelet per
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
    Buffer 0 view of the rounds until the transfer lands them in a
    unit's memory (``decoder_input``); a decoder reads only its unit's
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
    service_cancelled_request_keys: set[window_records.DecoderRequestKey] = (
        field(default_factory=set)
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


@dataclass(frozen=True)
class WindowDecode:
    """What one backend call on one window answers.

    ``selected_faults`` is the correction and ``decode_status`` a
    best-effort disposition (None when the decode succeeded). The two
    evidence fields are what a confidence signal reads off the decode
    that produced the correction: the minimum weight inside the class a
    forced solve was pinned to, and the growth a cluster-based decode
    did (Meister et al. 2405.07433 Algorithm 2 lines 518-525 reads the
    graph and the radii). A row leaves what it does not produce None.
    """

    selected_faults: Any
    decode_status: Optional[Any] = None
    forced_class_weight: Optional[float] = None
    cluster_evidence: Optional[Any] = None


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
    # round-keyed seam defects (synthetic decoders, recovery lock
    # scenarios)
    boundary_defects: Optional[dict] = None
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
    window's shape (the forward window of Toshio et al. 2510.25222
    Sec. III C when true, decsim's own two-sided context otherwise);
    is_bulk_strong is the
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
    is_double_window: bool
    is_bulk_strong: bool
    has_dynamic_streams: bool
    has_static_decode_plan: bool
    has_frontend: bool
