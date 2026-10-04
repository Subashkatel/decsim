"""The program the front end hands the machine, one operation at a time.

Every operation, the resolved facts planning needs, and the commands and
claims around them. An operation is the unit a workload is written in
and the unit the QPU runs; the planning views are the immutable facts
resolved once per run, so a planning collaborator reads a frozen record
rather than the live operation with its Stim circuit.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


@dataclass(frozen=True)
class ResolvedCodeGeometry:
    """Canonical planning/control geometry resolved once for one run."""

    code_name: str
    distance: int
    commit_round_count: int
    buffer_round_count: int


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
class ResourceClaim:
    """Typed exclusivity claim on shared hardware.

    Only kind="qubits" is used today: a layout derives one claim from an
    operation's qubit tuple.
    """

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


@dataclass(frozen=True)
class ProtectedRegion:
    """One owner patch group protected between inclusive operation endpoints."""

    stream_id: int
    start_operation_id: int
    end_operation_id: int


class OpKind(Enum):
    """Logical-op kind vocabulary for a rounds policy.

    It lets a RoundsPolicy tell a measurement (1 round) from a merge
    (m·d rounds) from an injection (O(1)).
    """

    IDLE = auto()
    MEMORY = auto()
    MERGE = auto()
    MEASURE = auto()
    INJECT = auto()
    GENERIC = auto()


@dataclass(frozen=True)
class Operation:
    """One logical operation in the circuit."""

    id: int
    name: str  # human-readable label used in traces
    qubits: tuple  # logical qubit ids the op acts on
    clifford: bool = True  # non-Clifford implies a magic state by default
    # Stim circuit for real-syndrome (data-path) runs. A Stim circuit has
    # no hash, and equal operations hold equal circuits, so the hash
    # leaves it out and still agrees with equality.
    circuit: Optional[Any] = field(default=None, hash=False)
    consumes_magic_state: Optional[bool] = (
        None  # override; None = infer from clifford
    )
    patches: tuple = ()  # patch ids whose syndrome streams feed the op
    predecessors: tuple = ()  # workload op ids that must complete first
    # prior decode streams at a boundary
    decoder_boundary_predecessors: tuple = ()
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
    # First slot of this operation's contiguous readout group in the round.
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
        """Whether the operation draws a distilled state from the factory."""
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
        feedback_boundary_mode = operation.feedback_boundary_mode
        if feedback_boundary_mode is None:
            feedback_boundary_mode = default_feedback_boundary_mode
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
            feedback_boundary_mode=feedback_boundary_mode,
            requires_result_return_to_qpu=(
                operation.requires_result_return_to_qpu
            ),
            kind=operation.kind,
        )


def patches_of(operation: Operation) -> tuple:
    """The patches an operation occupies; its first qubit stands in for none."""
    if operation.patches:
        return tuple(operation.patches)
    if operation.qubits:
        return (operation.qubits[0],)
    return (0,)


def decode_identity(operation: Operation) -> Any:
    """The decode stream an operation's rounds fold into.

    A stream segment names its stream; a standalone operation is its own
    stream. Every syndrome source keys its draws and its readouts by it.
    """
    if operation.stream_id is not None:
        return operation.stream_id
    return operation.id


def global_round(operation: Operation, round_index: int) -> int:
    """A round's index in its decode stream: the segment's offset added."""
    stream_offset = operation.stream_offset
    if stream_offset is None:
        stream_offset = 0
    return round_index + stream_offset
