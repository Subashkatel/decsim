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

import decsim.records.identity as identity_records


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
            sorted(patch_ids, key=identity_records.stable_identity_order_key)
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
            operation.patches, key=identity_records.stable_identity_order_key
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
            order_key = identity_records.stable_identity_order_key(patch_id)
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
