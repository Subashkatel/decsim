"""One syndrome round on its way out of the QPU and through the controller.

The bits are raw measurements; detection events are formed later,
downstream of these records.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional, Union

import numpy

import decsim.records.windows as window_records


class SyndromePacketRouteKind(Enum):
    """Where a completed round goes: into a window, or feedback memory."""

    WINDOW_INPUT = auto()
    FEEDBACK_MEMORY_ROUND = auto()


@dataclass(frozen=True)
class SyndromePacketRoute:
    """The route of one round from the controller.

    A window input, or a feedback-memory round of a source operation.
    """

    kind: SyndromePacketRouteKind
    source_operation_id: Optional[Any] = None  # an opaque identity

    @classmethod
    def feedback_memory_round(
        cls,
        source_operation_id: Any,  # an opaque identity
    ) -> "SyndromePacketRoute":
        """The route of a round that feeds a source operation's memory."""
        return cls(
            SyndromePacketRouteKind.FEEDBACK_MEMORY_ROUND, source_operation_id
        )


WINDOW_INPUT_ROUTE = SyndromePacketRoute(SyndromePacketRouteKind.WINDOW_INPUT)


@dataclass(frozen=True)
class QPUReadout:
    """One readout of one round as it leaves the QPU for the controller.

    bits are raw outcomes, None from a timing-only source. operation_id is
    the decode identity (a segment's stream id; an idle patch's memory round
    names the patch), round_index one-based. size_bits is the wire width,
    None when the source states none. event_bits is the width once formed,
    stated by a source with no circuit (qpu/syndrome_devices.py); None when
    a formation table sizes them, or nothing does.
    """

    operation_id: Any  # an opaque identity
    patch_ids: tuple
    round_index: int
    bits: Optional[Union[list, tuple, numpy.ndarray]] = None
    fragment_count: int = 1
    fragment_index: int = 0
    size_bits: Optional[int] = None
    event_bits: Optional[int] = None


@dataclass(frozen=True)
class RetainedSyndromeFragment:
    """One validated immutable fragment retained after controller packing.

    event_bits is the readout's, until a seat forms the fragment.
    """

    operation_id: Any  # an opaque identity
    patch_ids: tuple
    round_index: int
    bits: Optional[tuple[int, ...]]
    size_bits: Optional[int]
    fragment_index: int
    event_bits: Optional[int] = None

    @classmethod
    def from_readout(cls, readout: QPUReadout) -> "RetainedSyndromeFragment":
        """The readout as controller binary: its bits normalized.

        A list, a tuple or a NumPy bool array becomes a tuple of 0/1
        ints; a timing-only readout keeps None.
        """
        bits = None
        if readout.bits is not None:
            bits = tuple(int(bit) for bit in readout.bits)
        return cls(
            operation_id=readout.operation_id,
            patch_ids=readout.patch_ids,
            round_index=readout.round_index,
            bits=bits,
            size_bits=readout.size_bits,
            fragment_index=readout.fragment_index,
            event_bits=readout.event_bits,
        )


@dataclass(frozen=True)
class SyndromeRoundPacket:
    """One complete immutable syndrome round in declared measurement order."""

    operation_id: Any  # an opaque identity
    round_index: int
    fragments: tuple[RetainedSyndromeFragment, ...]

    def defects_text(self) -> str:
        """The round's cargo for the I/O trace.

        The set detection-event indices across the fragments in order,
        sparse so d=11 lines stay readable.
        """
        round_bits = []
        for fragment in self.fragments:
            bits = fragment.bits
            if bits is None:
                return "timing-only"
            round_bits.extend(bits)
        defects = []
        for position, bit in enumerate(round_bits):
            if bit:
                defects.append(position)
        if not defects:
            return "no defects"
        indices = ", ".join(str(defect) for defect in defects)
        return f"defects {{{indices}}}"


@dataclass(frozen=True)
class EscalatedRegion:
    """The rounds of a strong window as they leave the chip for the strong side.

    It carries the rounds the strong syndrome buffer lacks, read out of the
    weak one: Toshio et al. 2510.25222 lines 1247-1250 assign r_strong
    rounds to the strong decoder at the switch, and CUDA-Q QEC's
    enqueue_syndromes names the decoder and carries them
    (cudaqx_realtime_decoding.h lines 27-35). wire_bits is what the strong
    buffer is written, None when a fragment has no size. message_bits is
    one message behind one name however many rounds, as a gem5 DMA request
    covers its range (src/dev/dma_device.cc:195-207). rounds_before are the
    raw rounds before the first, read for its detectors and never formed.
    """

    request_key: window_records.DecoderRequestKey
    packets: tuple[SyndromeRoundPacket, ...]
    wire_bits: Optional[int]
    rounds_before: tuple[SyndromeRoundPacket, ...] = ()

    @classmethod
    def of(
        cls,
        request_key: window_records.DecoderRequestKey,
        packets: tuple,
        first_round: int = 1,
    ) -> "EscalatedRegion":
        """The region of these packets, its width summed from the fragments.

        first_round is the strong window's first round; a packet of the
        strong window's operation before it is a round before.
        """
        wire_bits = 0
        for packet in packets:
            packet_bits = fragment_wire_bits(packet.fragments)
            if packet_bits is None:
                wire_bits = None
                break
            wire_bits += packet_bits
        operation_id = request_key.operation_id
        rounds_before, window_packets = _split_at_the_first_round(
            packets, operation_id, first_round
        )
        return cls(
            request_key=request_key,
            packets=window_packets,
            wire_bits=wire_bits,
            rounds_before=rounds_before,
        )

    @property
    def carried_packets(self) -> tuple:
        """Every packet the region carries: the rounds before, then its own."""
        return self.rounds_before + self.packets

    def message_bits(self) -> Optional[int]:
        """The region on the wire: the request's name, then the rounds.

        gem5 sizes a data message as data plus control size
        (src/mem/ruby/network/Network.cc m_data_msg_size). None as wire_bits.
        """
        if self.wire_bits is None:
            return None
        return window_records.REQUEST_KEY_WIRE_BITS + self.wire_bits

    @property
    def round_keys(self) -> tuple:
        """(operation_id, round_index) of every round carried, in order."""
        keys = []
        for packet in self.carried_packets:
            keys.append((packet.operation_id, packet.round_index))
        return tuple(keys)


@dataclass(frozen=True)
class PackedRound:
    """A finished round as it leaves the assembler.

    wire_bits is the detection events where the controller forms them, the
    raw outcomes where a later seat does.
    """

    packet: SyndromeRoundPacket
    route: SyndromePacketRoute
    wire_bits: Optional[int]

    @property
    def round_key(self) -> tuple:
        """(operation_id, round_index), the store's key."""
        return (self.packet.operation_id, self.packet.round_index)


@dataclass(frozen=True)
class RoundEvent:
    """One recorded transition of one syndrome round through the controller.

    kind is one of EMITTED, BINARY_AVAILABLE, PACKED, STALLED, RELEASED,
    CWB_SENT, PUBLISHED, FEEDBACK_MEMORY_DELIVERED.
    """

    kind: str
    tick: int
    operation_id: object
    round_index: int
    patch_ids: tuple = ()
    route: str = ""

    @classmethod
    def of(
        cls,
        kind: str,
        tick: int,
        operation_id: object,
        round_index: int,
        route: SyndromePacketRoute,
        patch_ids: tuple = (),
    ) -> "RoundEvent":
        """One transition on a route, named by the route's kind."""
        return cls(
            kind, tick, operation_id, round_index, patch_ids, route.kind.name
        )


@dataclass(frozen=True)
class ControllerOutputEvent:
    """One transition on the controller's digital-to-QPU path.

    payload is the decision or QPU command itself, so a ledger can prove
    that the data whose timing was modeled is the data the QPU received.
    """

    kind: str
    tick: int
    operation_id: object
    payload: object


def fragment_wire_bits(
    fragments: Sequence[RetainedSyndromeFragment],
) -> Optional[int]:
    """The fragments' wire size, None when any fragment has no known size."""
    fragment_sizes = [fragment.size_bits for fragment in fragments]
    if None in fragment_sizes:
        return None
    return sum(fragment_sizes)


def fragment_patch_ids(
    fragments: Sequence[RetainedSyndromeFragment],
) -> tuple:
    """The contributing footprint, once per patch in first appearance order."""
    patches = []
    for fragment in fragments:
        patches.extend(fragment.patch_ids)
    distinct = dict.fromkeys(patches)
    return tuple(distinct)


def stated_bits(bits: Optional[int]) -> int:
    """A stated width as a number to add up; an unstated one counts zero.

    A memory bounded in bits refuses rounds of unknown width before they
    land, so only an unbounded memory ever adds one up.
    """
    if bits is None:
        return 0
    return bits


def _split_at_the_first_round(
    packets: tuple,
    operation_id: Any,  # an opaque identity
    first_round: int,
) -> tuple:
    """(the rounds before first_round, the rest), each in the order given.

    A successor operation's rounds carry small round numbers too, so a
    round before is one of the strong window's own operation.
    """
    rounds_before = []
    window_packets = []
    for packet in packets:
        is_before = packet.round_index < first_round
        if is_before and packet.operation_id == operation_id:
            rounds_before.append(packet)
            continue
        window_packets.append(packet)
    return tuple(rounds_before), tuple(window_packets)
