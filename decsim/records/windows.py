"""One decoder window: its geometry, its plan, its boundaries, its state.

A window is a range of an operation's rounds that one decode covers, so
a request for it is a window key plus the tier that serves it and a
run-wide ordinal; DecoderTier and DecoderRequestKey live here because a
window records which request finally published its correction. The
window itself is the window manager's live bookkeeping, and is the one
record in the folder that is not frozen.
"""

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


@dataclass
class Window:
    """One decoder window inside an operation's syndrome stream.

    Rounds are 1-based and ranges inclusive: the window reads rounds
    [start_round, buffer_hi] and commits corrections for
    [commit_lo, commit_hi].
    """

    op_id: int  # operation that owns the stream
    k: int  # window index within the op; key = (op_id, k)
    commit_lo: int  # first round this window commits
    commit_hi: int  # last round this window commits
    buffer_hi: int  # last round it reads (trailing buffer)
    # planned rounds from start_round to buffer_hi; the job is priced for
    # the rounds that exist
    n_rounds: int
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
        """The first round this window needs.

        The leading buffer's start when the window has one, else the
        first round it commits.
        """
        if self.buffer_lo is None:
            return self.commit_lo
        return self.buffer_lo

    @property
    def key(self) -> tuple:
        """(op_id, k), the key every window collection is keyed by."""
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
        """The window as a policy reads it, with its edges frozen."""
        positions = None
        if detector_positions is not None:
            positions = dict(detector_positions)
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


@dataclass(frozen=True)
class DependencyResidual:
    """Complete global detector effect plus its compatibility mask view."""

    detector_ids: tuple[int, ...] = ()
    defects: Optional[dict] = None


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
