"""One decoder window: its geometry, its plan, its boundaries, its state.

A window is a range of an operation's rounds that one decode covers, so
a request for it is a window key plus the tier that serves it and a
run-wide ordinal; DecoderTier and DecoderRequestKey live here because a
window records which request finally published its correction. The
window itself is the window manager's live bookkeeping, so it and the
plan records that carry live state are the folder's unfrozen ones.
FormationReads says which reads also hold the raw rounds before them;
the plan builder and the planner share it.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class WindowProtocol(Enum):
    """Scientific model-building contract for one operation's window plan."""

    GENERIC = auto()
    TAN_ZERO_SEAM_GRAPHLIKE = auto()


class DecoderTier(Enum):
    """Weak (first, fast) or strong (escalated, slow) decode."""

    WEAK = "weak"
    STRONG = "strong"


# windows.terminal_policy names one of these: how a finite serial stream
# drains its last buffered window. flush ends the last window at the
# stream's last round, which is qLDPC's last window (qLDPC
# src/qldpc/decoders/sinter.py:776-777); lookahead keeps the regular
# stride, so the last window still reads rounds past its own commit.
TERMINAL_POLICIES = ("flush", "lookahead")


@dataclass(frozen=True)
class WindowingSchemeCard:
    """The windows section's keys a windowing scheme row reads.

    One record so every row of WINDOWING_SCHEMES has one constructor
    signature and the root builds a row without asking which geometry it
    lays; a row reads the keys its own layout needs and ignores the
    rest. This is the shape the rows of STRONG_WINDOW_SHAPES have, and
    gem5's params object
    (gem5 src/python/m5/SimObject.py:204-205).
    """

    terminal_policy: str = "flush"


# The card a row is built on when the section names no key of its own.
DEFAULT_SCHEME_CARD = WindowingSchemeCard()


@dataclass(frozen=True)
class DecoderRequestKey:
    """Identity of one decode request: its window, its tier, its ordinal.

    The run-wide ordinal keeps a retry of the same window and tier
    distinct from the request it replaces.
    """

    operation_id: Any
    window_id: int
    tier: DecoderTier
    run_sequence: int


# What a request's name is on a wire. Three kinds of message carry it,
# once in every transfer: the selection, which is the name alone, a
# region transfer, which is the name and its rounds, and the strong
# answer, which is the name and its flips. The rule is gem5's: its
# network sizes a message with no data at control_msg_size, 8 bytes, and
# one with data at its data plus that same size (gem5
# src/mem/ruby/network/Network.cc MessageSizeType_to_int and
# m_data_msg_size, Network.py control_msg_size). gem5's data size is one
# configured number; decsim applies the same sum to a region of any
# size. The name is decsim's own abstraction of whatever identifies the
# request, not a published layout: CUDA-Q QEC spends an int64 on the
# decoder's id at the front of every request payload (cudaqx
# decoder_rpc_wire_format.h EnqueueRequestPayload,
# GetCorrectionsRequestPayload, ResetRequestPayload) and 32 bits on a
# request id in every 24-byte header. A path's header frames each
# transfer beside the name.
REQUEST_KEY_WIRE_BITS = 64


@dataclass
class Window:
    """One decoder window inside an operation's syndrome stream.

    Rounds are 1-based and ranges inclusive: the window reads rounds
    [start_round, buffer_hi] and commits corrections for
    [commit_lo, commit_hi].
    """

    operation_id: int  # operation that owns the stream
    # the window's index within the operation;
    # key = (operation_id, window_index)
    window_index: int
    commit_lo: int  # first round this window commits
    commit_hi: int  # last round this window commits
    buffer_hi: int  # last round it reads (trailing buffer)
    # planned rounds from start_round to buffer_hi; the job is priced for
    # the rounds that exist
    round_count: int
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
    published_request_key: Optional[DecoderRequestKey] = None
    queued: bool = False  # a decode request is pending or admitted
    boundary_in: Any = field(default_factory=dict)  # state owned by the
    # configured WindowInteraction
    decode_status: Optional[str] = (
        None  # best-effort status of the committed decode, None = succeeded
    )
    # the BackendFailureReason value of the committed decode whose
    # backend produced no correction; None when it produced one
    no_correction_reason: Optional[str] = None
    # the same of a provisional commit the strong result later replaced,
    # kept because what that commit fed forward outlives the replacement
    provisional_no_correction_reason: Optional[str] = None
    # what its committed decode owns of the faults crossing behind its
    # commit region; None until it commits, and empty for every window
    # an earlier owner's commit closes
    crossing_commit: Optional["CrossingCommit"] = None
    t_first_round: Optional[int] = None  # tick the first round arrived
    t_data_complete: Optional[int] = (
        None  # tick the last buffered round arrived
    )
    t_queued: Optional[int] = None  # tick the job entered the decode queue
    # The three ticks below are the window's own decodes as they happen,
    # so a window decoded more than once (the two forced-class solves of
    # a complementary gap, decision D2) keeps the last one's dispatch
    # and the last one's compute start, and t_done is the tick its last
    # weak answer arrived. A reader that needs one decode's own ticks
    # reads the stage records, which carry the run ordinals they served
    # and the tick a unit took that decode (decoders/staged_decoder.py);
    # the latency points do (experiments/measure.py).
    t_dispatch: Optional[int] = None  # tick a decoder unit took the job
    # tick the unit began computing the job it took: the input had
    # landed in the unit's memory and the window owed no boundary
    t_compute_start: Optional[int] = None
    t_done: Optional[int] = None  # tick the decode finished

    @property
    def start_round(self) -> int:
        """The first round this window needs.

        The leading buffer's start when the window has one, else the
        first round it commits.
        """
        if self.buffer_lo is None:
            return self.commit_lo
        return self.buffer_lo

    @property
    def key(self) -> tuple:
        """(operation_id, window_index), the key window collections use."""
        return (self.operation_id, self.window_index)


@dataclass(frozen=True)
class WindowInfo:
    """Read-only geometry and topology exposed to interaction policies."""

    operation_id: int
    window_index: int
    commit_lo: int
    commit_hi: int
    buffer_hi: int
    round_count: int
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
        """The window as a policy reads it, with its edges frozen."""
        positions = None
        if detector_positions is not None:
            positions = dict(detector_positions)
        return cls(
            operation_id=window.operation_id,
            window_index=window.window_index,
            commit_lo=window.commit_lo,
            commit_hi=window.commit_hi,
            buffer_hi=window.buffer_hi,
            round_count=window.round_count,
            buffer_lo=window.buffer_lo,
            deps=tuple(window.deps),
            dependents=tuple(window.dependents),
            detector_positions=positions,
        )

    @property
    def start_round(self) -> int:
        """The first round the window needs, as Window.start_round reads it."""
        if self.buffer_lo is None:
            return self.commit_lo
        return self.buffer_lo


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
        """The rounds the interval spans, both buffers included."""
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

    windows: dict  # (operation_id, window_index) -> Window
    window_count: dict  # operation_id -> number of windows
    # operation_id -> [window keys, in window_index order]
    op_windows: dict
    successors: dict  # operation_id -> [op ids listing it as predecessor]
    rounds_by_operation: dict  # operation_id -> resolved positive round count
    total_windows: int
    windowed_by_operation: dict
    batch_preceding_idle_rounds_by_operation: dict
    protocol_by_operation: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DependencyResidual:
    """Complete global detector effect plus its compatibility mask view."""

    detector_ids: tuple[int, ...] = ()
    defects: Optional[dict] = None


@dataclass(frozen=True)
class CrossingCommit:
    """What one decode committed of the faults crossing its near seam.

    A window that owns the faults touching the round before its commit
    region hands that part of its correction on twice: as the boundary
    condition of the region ending there, and, when it escalates, to its
    own strong redo, which owns none of those faults. The residual is
    their complete detector effect, and logical_observables the
    observables they flip, which stay with the window when the strong
    result replaces its prediction.
    """

    residual: DependencyResidual
    logical_observables: tuple[int, ...]


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
class BoundarySeam:
    """What one boundary message updates: the destination's oldest layer.

    A window hands its neighbour the detectors its committed correction
    flips on the one round layer the neighbour starts with: Tan et al.
    2209.09219 lines 936-946 ("the detectors on the oldest layer of the
    next window are updated"), quits `syn_update` sized by one check
    layer (sliding_window.py:164-174) and cuda-q QEC's `syndrome_mods`
    written only between the next window's round bounds
    (sliding_window.cpp:325-344). detector_count is that layer's
    detectors, d*d-1 on a bulk layer of a rotated surface code;
    flip_count is how many of them the message flips.
    """

    detector_count: int
    flip_count: int


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


# The double-window region folds one buffer region into the escalated
# window's commit region on each side, so it commits rcom + 2 rbuf
# rounds. Toshio et al. 2510.25222 line 1352 parameterises that two as
# alpha, so a sweep over region width changes this number here and every
# reader of the geometry moves with it: the interaction row that plans
# the region, the build refusal that checks it lands on a stride edge,
# the retention that holds the rounds it would read, and the card that
# provisions the link carrying it.
STRONG_REGION_BUFFER_REGIONS = 2


def strong_region_round_count(
    commit_round_count: int, buffer_round_count: int
) -> int:
    """How many rounds one double-window region covers."""
    buffered = STRONG_REGION_BUFFER_REGIONS * buffer_round_count
    return commit_round_count + buffered


def strong_context_bounds(window: "Window") -> tuple:
    """(context_lo, commit_lo, commit_hi, context_hi) a strong redo reads.

    A strong redo of this window reads its commit rounds and one buffer
    region past them, and no round before them: every row pins its past
    face on the earlier neighbour's committed correction, and a fixed
    boundary condition replaces the buffer that would otherwise open
    that face (Bombin et al. 2303.04846 lines 1456-1458), while the
    future face stays open over one buffer region of raw context (lines
    850-852). The redo window reads exactly this; the double window
    reads from the same first round on, further forward
    (escalation/strong_regions.py). The round retention keeps this span
    for a window the strong tier may redo.
    """
    buffer_span = window.buffer_hi - window.commit_hi
    buffer_rounds = max(0, buffer_span)
    context_hi = window.commit_hi + buffer_rounds
    return window.commit_lo, window.commit_lo, window.commit_hi, context_hi


def restart_reread_round_count(
    reread_buffer_regions: int, buffer_round_count: int
) -> int:
    """How many of the strong region's rounds the restart window re-reads."""
    return reread_buffer_regions * buffer_round_count


@dataclass(frozen=True)
class StrongRegionPlan:
    """Geometry and seam ownership for one deferred strong decode."""

    commit_lo: int
    commit_hi: int
    context_lo: int
    context_hi: int
    restart_buffer_lo: Optional[int]
    restart_seam_fault_owner: Optional[SeamFaultOwner]


@dataclass(frozen=True)
class SuccessorReadiness:
    """How many rounds of a dependent operation have arrived, of how many.

    A window at the end of an operation waits on the operation that
    follows it, so a scheme reads the successor's arrivals too.
    """

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


@dataclass(frozen=True)
class FormationReads:
    """Which reads also hold the raw rounds their rounds' recipes read.

    A seat that forms the events and starts mid-stream is given every
    raw round before its first that its rounds' recipes read
    (FormationTable rounds_read_before_first), so the read that carries
    those rounds holds them, as an HEVC decoder keeps each picture the
    current reference set names (FFmpeg hevc/refs.c:486-517).
    strong_side_forms is a seat past the weak syndrome buffer that
    forms; primary_reader_forms is the decoder the primary store feeds,
    forming. tables maps an operation id to its formation table; an
    operation with none reads nothing before its first round.
    """

    strong_side_forms: bool = False
    primary_reader_forms: bool = False
    tables: Mapping = field(default_factory=dict)

    def strong_read_start(
        self, operation_id: Any, first_round: int, last_round: int
    ) -> int:
        """The first round a strong read of these rounds holds."""
        if not self.strong_side_forms:
            return first_round
        return self._earliest_round_read(operation_id, first_round, last_round)

    def primary_read_start(
        self, operation_id: Any, first_round: int, last_round: int
    ) -> int:
        """The first round a primary read of these rounds holds."""
        if not self.primary_reader_forms:
            return first_round
        return self._earliest_round_read(operation_id, first_round, last_round)

    def _earliest_round_read(
        self, operation_id: Any, first_round: int, last_round: int
    ) -> int:
        table = self.tables.get(operation_id)
        if table is None:
            return first_round
        stop_round = last_round + 1
        read_rounds = range(first_round, stop_round)
        reach_count = table.rounds_read_before_first(first_round, read_rounds)
        return first_round - reach_count


# a run whose reading seats form nothing holds no round before a read
NO_FORMING_READER = FormationReads()
