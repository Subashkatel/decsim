"""One syndrome round on its way out of the QPU and through the controller.

A readout arrives per patch, becomes a retained fragment, joins the other
fragments of its round into a packet, and leaves the assembler as a
packed round on a route. The two events at the end are what the ledgers
record; the bits themselves are raw measurements, and detection events
are formed later, downstream of these records.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional


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
    source_operation_id: Optional[Any] = None

    @classmethod
    def feedback_memory_round(
        cls, source_operation_id
    ) -> "SyndromePacketRoute":
        """The route of a round that feeds a source operation's memory."""
        return cls(
            SyndromePacketRouteKind.FEEDBACK_MEMORY_ROUND, source_operation_id
        )


WINDOW_INPUT_ROUTE = SyndromePacketRoute(SyndromePacketRouteKind.WINDOW_INPUT)


@dataclass(frozen=True)
class QPUReadout:
    """One QPU-side readout awaiting controller front-end handling.

    DECSIM intentionally does not carry an analog waveform. ``bits`` is the
    sampled/classifiable outcome cargo; after the configured physical
    acquisition/discrimination latency, the controller exposes its normalized
    classical-bit tuple. Detection events are formed later from these packets.
    """

    operation_id: Any
    patch_id: Any
    round_index: int
    bits: Optional[Any] = None
    code: Optional[str] = None
    fragment_count: int = 1
    fragment_index: int = 0
    size_bits: Optional[int] = None


@dataclass(frozen=True)
class RetainedSyndromeFragment:
    """One validated immutable fragment retained after controller packing."""

    operation_id: Any
    patch_id: Any
    round_index: int
    bits: Optional[tuple[int, ...]]
    size_bits: Optional[int]
    fragment_index: int

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
            patch_id=readout.patch_id,
            round_index=readout.round_index,
            bits=bits,
            size_bits=readout.size_bits,
            fragment_index=readout.fragment_index,
        )


@dataclass(frozen=True)
class SyndromeRoundPacket:
    """One complete immutable syndrome round in transport-arrival order."""

    operation_id: Any
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
class PackedRound:
    """A finished round as it leaves the assembler.

    The packet, its route, and its size on the wire: what leaves the
    controller, which is the detection events where the controller forms
    them and the raw measurement outcomes where the decoder does
    (controller.detection_events_formed_at).
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
    CWB_SENT, PUBLISHED, DROPPED, FEEDBACK_MEMORY_DELIVERED.
    """

    kind: str
    tick: int
    operation_id: object
    round_index: int
    patch_id: object = None
    route: str = ""

    @classmethod
    def of(
        cls,
        kind: str,
        tick: int,
        operation_id,
        round_index: int,
        route: SyndromePacketRoute,
        patch_id=None,
    ) -> "RoundEvent":
        """One transition on a route, named by the route's kind."""
        return cls(
            kind, tick, operation_id, round_index, patch_id, route.kind.name
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
