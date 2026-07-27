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


@dataclass
class WindowPlan:
    """Compile-time window layout handed to the window manager."""

    windows: dict         # (op_id, k) -> Window
    window_count: dict    # op_id -> number of windows
    op_windows: dict      # op_id -> [window keys, in k order]
    successors: dict      # op_id -> [op ids listing it as predecessor]
    spatial_nodes: dict   # op_id -> decoding-graph nodes per round
    total_windows: int
    summary: dict = field(default_factory=dict)   # printable planning stats


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
    logical_value: Optional[int] = None      # predicted logical observable bit
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

@dataclass
class Decision:
    """Feedback decision sent orchestrator -> controller -> chip after a
    non-Clifford measurement decodes: steer the successor's basis and fold
    the byproduct into its Pauli frame."""

    target_operation_id: int          # blocked op this decision addresses
    basis: str                        # measurement basis chosen for the successor
    releases_operation: bool = True   # False = result-return only, no op waiting
    pauli: str = "I"                  # byproduct Pauli folded into the frame
    apply_s: bool = False             # also fold an S gate (X -> Y) into the frame
    correction_value: int = 0         # the decoded logical measurement bit
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
    kind: OpKind = OpKind.GENERIC     # rounds-policy vocabulary (see OpKind)

    @property
    def needs_magic_state(self) -> bool:
        """True when this operation draws a distilled magic state from the factory."""
        if self.consumes_magic_state is not None:
            return self.consumes_magic_state
        return not self.clifford
