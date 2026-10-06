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

    In report order; a fabric card prices every hop, so none is free.
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
    STRONG_DECODER_TO_WEAK_DECODER = (
        "strong_decoder_to_weak_decoder"  # a strong answer to the chip
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

    The range counts past the operation's end the way its window does;
    round_keys names every round the bits carry, a lookahead window's
    next-operation rounds under their own operation.
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
    round_keys: tuple = ()

    @classmethod
    def for_round(
        cls,
        operation_id: Any,  # an opaque identity
        patch_ids: tuple,
        round_index: int,
    ) -> "TransferAttribution":
        """One round of one operation, its patches in stable order."""
        ordered_patch_ids = tuple(
            sorted(patch_ids, key=identity_records.stable_identity_bytes)
        )
        return cls(
            operation_id=operation_id,
            patch_ids=ordered_patch_ids,
            window_id=None,
            first_round=round_index,
            last_round=round_index,
            round_keys=((operation_id, round_index),),
        )

    @classmethod
    def for_window(
        cls,
        window: window_records.Window,
        operation: program_records.Operation,
        request_key: window_records.DecoderRequestKey,
        round_keys: tuple,
    ) -> "TransferAttribution":
        """A window's transfer: the operation's patches, the rounds it reads."""
        ordered_patches = sorted(
            operation.patches, key=identity_records.stable_identity_bytes
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
            round_keys=round_keys,
        )

    @classmethod
    def for_job(
        cls,
        job: decoding_records.DecodeJob,
        request_key: window_records.DecoderRequestKey,
        round_keys: tuple,
    ) -> "TransferAttribution":
        """A job's transfer: its payloads' patches, its window's rounds."""
        payloads = job.payloads or ()
        patches = round_records.fragment_patch_ids(payloads)
        ordered_patches = sorted(
            patches, key=identity_records.stable_identity_bytes
        )
        patch_ids = tuple(ordered_patches)
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
            round_keys=round_keys,
        )

    @classmethod
    def for_region(
        cls, region: round_records.EscalatedRegion
    ) -> "TransferAttribution":
        """An escalated region's transfer: its rounds, in its request's name.

        The footprint covers every packet, the rounds before the strong
        window's first among them; the round range names the first and
        last packets.
        """
        carried = region.carried_packets
        first_packet = carried[0]
        last_packet = carried[-1]
        fragments = []
        round_keys = []
        for packet in carried:
            fragments.extend(packet.fragments)
            round_keys.append((packet.operation_id, packet.round_index))
        patch_ids = round_records.fragment_patch_ids(fragments)
        ordered_patch_ids = tuple(
            sorted(patch_ids, key=identity_records.stable_identity_bytes)
        )
        relation = RequestTransferRelation(region.request_key)
        return cls(
            operation_id=first_packet.operation_id,
            patch_ids=ordered_patch_ids,
            window_id=region.request_key.window_id,
            first_round=first_packet.round_index,
            last_round=last_packet.round_index,
            relation=relation,
            round_keys=tuple(round_keys),
        )

    @classmethod
    def for_packet(
        cls, packet: round_records.SyndromeRoundPacket
    ) -> "TransferAttribution":
        """The packed round's transfer: every patch of the round."""
        patch_ids = round_records.fragment_patch_ids(packet.fragments)
        return cls.for_round(packet.operation_id, patch_ids, packet.round_index)


class PayloadSelection(Enum):
    """Where a transfer's payload size came from."""

    ACTUAL = "actual"
    CONFIGURED_DEFAULT = "configured_default"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Transfer:
    """One transfer's timing on its channel, complete at delivery.

    total_delay_ticks counts from the request and is the sum of the two
    setup spans, the queue wait, the serialization and the propagation.
    header_bits is the path's framing, serialized with the payload and no
    part of it. On a packet protocol serialization_ticks is the time the
    frames held the wire, lost and resent ones included, and
    queue_wait_ticks the rest of the span from first frame start to last
    frame end.
    """

    payload_bits: Optional[int]
    header_bits: int
    request_ticks: int
    setup_wait_ticks: int
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


@dataclass(frozen=True)
class FramedPayload:
    """One transfer's payload, framed by its path's header.

    The ledger counts the payload and the wire serializes both. A payload of
    unknown size rides an unbounded channel only.
    """

    payload_bits: Optional[int]
    header_bits: int = 0


@dataclass(frozen=True)
class FrameTiming:
    """One frame's trip across a wire, in ticks.

    bits is None for a payload of unknown size on an unbounded wire.
    """

    bits: Optional[int]
    credit_wait_ticks: int
    start_ticks: int
    end_ticks: int
    landed_ticks: int


@dataclass(frozen=True)
class FrameRecord:
    """One frame on one channel, reported when its message is delivered.

    A reliable channel also reports each lost frame and each retransmission.
    """

    channel: str
    transfer_sequence: int
    frame_index: int
    timing: FrameTiming
    is_lost: bool = False
    is_retransmission: bool = False


def _read_range(window: window_records.Window) -> tuple:
    """The inclusive round range a window reads."""
    first_round = window.commit_lo
    if window.buffer_lo is not None:
        first_round = window.buffer_lo
    last_round = window.commit_hi
    if window.buffer_hi is not None:
        last_round = window.buffer_hi
    return first_round, last_round
