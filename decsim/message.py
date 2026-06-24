"""Shared message and plan records passed between simulator components."""

from dataclasses import dataclass, field
from typing import Callable, Optional, Any


@dataclass
class SyndromePayload:
    """One measured syndrome round for one logical operation."""

    operation_id: int
    patch_id: int
    round_index: int
    bits: Optional[Any] = None
    code: Optional[str] = None
    n_fragments: int = 1

    def __post_init__(self) -> None:
        """Reject impossible fragment counts early."""
        if self.n_fragments < 1:
            raise ValueError(f"n_fragments must be >= 1 (got {self.n_fragments})")


@dataclass
class MagicState:
    """A distilled resource state produced by a factory."""

    state_id: int
    fidelity: float = 1.0
    payload: Optional[Any] = None


@dataclass
class DecodeJob:
    """One unit of decoder work."""

    op_id: int
    window_id: int
    n_rounds: int
    dem: Optional[Any] = None
    payloads: list = field(default_factory=list)
    ready_time: int = 0
    deadline: int = 0
    on_done: Optional[Callable[[], None]] = None
    label: str = ""
    spatial_nodes: Optional[int] = None
    code: Optional[str] = None
    attempt: int = 0
    hint: Optional[str] = None
    pool: Optional[str] = None
    window: Optional["Window"] = None
    strong_decode_for: Optional[tuple] = None
    awaiting_strong_result: bool = False
    cancelled: bool = False


@dataclass
class Window:
    """One decoder window inside an operation's syndrome stream."""

    op_id: int
    k: int
    commit_lo: int
    commit_hi: int
    buffer_hi: int
    n_rounds: int
    buffer_lo: Optional[int] = None
    deps: list = field(default_factory=list)
    dependents: list = field(default_factory=list)
    deps_remaining: int = 0
    committed: bool = False
    queued: bool = False
    blocked_logged: bool = False
    boundary_in: dict = field(default_factory=dict)
    t_first_round: Optional[int] = None   # this window's first round arrived
    t_data_complete: Optional[int] = None # all needed rounds present
    t_queued: Optional[int] = None        # data and dependencies ready
    t_dispatch: Optional[int] = None      # decoder service begins
    t_done: Optional[int] = None          # decode committed

    @property
    def start_round(self) -> int:
        """First round this window needs (leading buffer if present, else commit start)."""
        return self.commit_lo if self.buffer_lo is None else self.buffer_lo


@dataclass
class WindowPlan:
    """Compile-time window layout sent to the decoder cluster."""

    windows: dict
    window_count: dict
    op_windows: dict
    successors: dict
    spatial_nodes: dict
    total_windows: int
    summary: dict = field(default_factory=dict)


@dataclass
class DecodeResult:
    """Decoder output for one window."""

    op_id: int
    window_id: int
    correction: Optional[Any] = None
    logical_value: Optional[int] = None
    soft_output: Optional[float] = None
    boundary_defects: Optional[dict] = None


@dataclass
class Decision:
    """Feedback decision sent from the orchestrator back to the chip."""

    target_operation_id: int
    basis: str
    releases_operation: bool = True


@dataclass
class Operation:
    """One logical operation in the circuit."""

    id: int
    name: str
    qubits: tuple
    clifford: bool = True
    circuit: Optional[Any] = None
    consumes_magic_state: Optional[bool] = None
    patches: tuple = ()
    predecessors: tuple = ()
    has_successor: bool = False
    stream_id: Optional[Any] = None
    stream_offset: Optional[int] = None
    blocked_by: Optional[int] = None
    feedback_boundary_mode: Optional[str] = None
    requires_result_return_to_chip: bool = False

    @property
    def needs_magic_state(self) -> bool:
        """True when this operation draws a distilled magic state from the factory."""
        if self.consumes_magic_state is not None:
            return self.consumes_magic_state
        return not self.clifford
