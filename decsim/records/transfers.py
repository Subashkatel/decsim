"""One transfer on one link: its hop, its bits, its timing, its ledger.

The hops are the reaction path, one per pair of components. A transfer's
attribution says whose bits moved, so the traffic report can tie a
delivery back to the operation, the window and the decode request that
asked for it.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import decsim.records.decoding as decoding_records
import decsim.records.identity as identity_records
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records


class LinkPath(str, Enum):
    """The hops of the reaction path, one per pair of components.

    In the order the reports list them. The weak buffer is syndrome
    buffer 0, the strong buffer syndrome buffer 1; a fabric card prices
    every hop here, so none of them is ever free.
    """

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

    request_key: window_records.DecoderRequestKey


@dataclass(frozen=True)
class BoundaryTransferRelation:
    """Provenance tying one decoder-to-decoder transfer to its boundary.

    From which window, produced by which request, to which window, and
    both revisions.
    """

    source_request_key: window_records.DecoderRequestKey
    source_window_key: tuple
    destination_window_key: tuple
    source_revision: int
    delivery_revision: int


@dataclass(frozen=True)
class TransferAttribution:
    """Whose transfer this is.

    The operation, its patches, the window or the inclusive round range
    the bits belong to, and the relation the path's rule asks for.
    """

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
        cls,
        window: window_records.Window,
        operation: program_records.Operation,
        request_key,
    ) -> "TransferAttribution":
        """A window's transfer: the operation's patches, the rounds it reads."""
        ordered_patches = sorted(
            operation.patches, key=identity_records.stable_identity_order_key
        )
        first_round, last_round = _read_range(window)
        relation = RequestTransferRelation(request_key)
        return cls(
            operation_id=operation.id,
            patch_ids=tuple(ordered_patches),
            window_id=window.window_index,
            first_round=first_round,
            last_round=last_round,
            relation=relation,
        )

    @classmethod
    def for_job(
        cls, job: decoding_records.DecodeJob, request_key
    ) -> "TransferAttribution":
        """A job's transfer: its payloads' patches, its window's rounds."""
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
        relation = RequestTransferRelation(request_key)
        return cls(
            operation_id=job.operation_id,
            patch_ids=patch_ids,
            window_id=job.window_id,
            first_round=first_round,
            last_round=last_round,
            relation=relation,
        )

    @classmethod
    def for_packet(
        cls, packet: round_records.SyndromeRoundPacket
    ) -> "TransferAttribution":
        """The packed round's transfer: every patch of the round."""
        patch_ids = tuple(fragment.patch_id for fragment in packet.fragments)
        return cls.for_round(packet.operation_id, patch_ids, packet.round_index)


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
    """One ledger entry: which path, on which channel, for whom.

    With which payload, and the transfer it got. request_sequence orders
    the ledger by request, whatever order the wires delivered.
    """

    request_sequence: int
    path: LinkPath
    channel: str
    attribution: TransferAttribution
    payload_selection: PayloadSelection
    payload_source: Optional[str]
    transfer: Transfer


def _read_range(window: window_records.Window) -> tuple:
    """The inclusive round range a window reads."""
    first_round = window.commit_lo
    if window.buffer_lo is not None:
        first_round = window.buffer_lo
    last_round = window.commit_hi
    if window.buffer_hi is not None:
        last_round = window.buffer_hi
    return first_round, last_round
