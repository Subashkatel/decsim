"""The strong syndrome buffer's receiving end: room, then landing.

Two hops land here. A strong-primary run's controller writes every
round over controller_to_strong_buffer, the one transport such a run
has (Caune 2410.05202 Fig. 1a stage F). A switching run's strong
redecode carries an escalated window's rounds over
weak_decoder_to_strong_decoder, read out of the weak syndrome buffer at
the switch (Toshio 2510.25222 lines 1247 to 1250: the syndrome data of
r_strong rounds is assigned to the strong decoder when the soft output
is small), so the rounds a cold weak tier keeps for itself never cross
the cryostat (Battistel 2303.00054 lines 342 to 347). Either sender
executes its own crossing, being the end the data leaves by (OMNeT++
refuses a module that sends a message it does not own,
tmp/resources/omnetpp/src/sim/csimplemodule.cc:333-334; gem5 bills a
transfer to the port it left by, coherent_xbar.cc:354-357). This end owns the
room and the landing: it answers has_room counting the bits still in
flight, reserves those bits before a crossing starts, gem5's packet
store counting its reserved bytes as taken (`avail() = _maxsize -
_size - _reserved` and `reserve(len)`, src/dev/net/pktfifo.hh), and
stores each round at its landing; the store's holds and lifetime are
SyndromeBuffer's. A landing whose operation closed while its bits
crossed is dropped at the door instead of stored, since no reader can
ever name it (_drop_landing): a strong-primary run's last rounds, or
the region of a parallel strong sibling that a confident weak result
cancelled while the region was on the wire.

Controller packets retain their canonical route. Window input wakes the
window manager; timing-only idle input occupies a slot until the store's
own decoder hop delivers it, without creating a window.
"""

import dataclasses
import functools
from typing import Optional

import decsim.ports as ports
import decsim.records.log_sources as log_sources
import decsim.records.rounds as round_records
import decsim.trace_source as trace_source


@dataclasses.dataclass(frozen=True)
class _Hop:
    """One of the two hops that land here: its row, and where its bits sat."""

    path_name: str
    source_name: str


CONTROLLER_WRITE = _Hop("controller_to_strong_buffer", "controller assembler")
ESCALATION = _Hop("weak_decoder_to_strong_decoder", "weak syndrome buffer")


class StrongSyndromeRoundReceiver:
    """The room, the writes in flight, and the landing into the store.

    Trace source: copy_made(round_key, bits, source, "strong syndrome
    buffer") at every landing, the crossing's copy (data_path.md hops 3
    and 5); the source is the controller assembler or the weak syndrome
    buffer, whichever the round left. round_event reports
    FEEDBACK_MEMORY_DELIVERED at the idle round's decoder-side delivery.
    """

    # a receiver built with no window side stores its rounds for a reader
    # that never asks
    windows = ports.Port(ports.WindowInput, optional=True)
    store = ports.Port(ports.SyndromeBuffer)
    output = ports.Port(ports.SyndromeBufferOutput)
    memory_arrivals = ports.Port(ports.MemoryRoundArrivals)

    def __init__(self, engine) -> None:
        self.engine = engine
        self.writes_in_flight = 0
        # the bits the crossing rounds will take, held against the store
        self.reserved_bits = 0
        self.trace = _TraceSources()

    def has_room(self, bits: Optional[int]) -> bool:
        """A write can land: the store weighs it against what is reserved."""
        return self.store.has_room(bits, self.reserved_bits)

    def reserve_write(self, bits: Optional[int]) -> None:
        """Take the bits one crossing round will need, before it leaves."""
        assert self.has_room(bits), (
            "a round was written into a full strong store"
        )
        self.writes_in_flight += 1
        self.reserved_bits += round_records.stated_bits(bits)

    def receive_round(self, packed: round_records.PackedRound) -> None:
        """Take one round that landed here: its room is now its slot."""
        self.writes_in_flight -= 1
        self.reserved_bits -= round_records.stated_bits(packed.wire_bits)
        window_input = round_records.SyndromePacketRouteKind.WINDOW_INPUT
        if packed.route.kind is window_input:
            self._land(packed.packet, packed.wire_bits, CONTROLLER_WRITE)
            return
        self._forward_memory_round(packed)

    def reserve_region(self, round_count: int, bits: Optional[int]) -> None:
        """Take the room an escalated region needs, before it leaves the chip.

        An escalation cannot wait: the weak result is already given up,
        so a strong store with no room for the region stops the run
        rather than holding the chip. The yaml sizes the store.
        """
        if not self.has_room(bits):
            self._refuse_region(round_count, bits)
        self.writes_in_flight += round_count
        self.reserved_bits += round_records.stated_bits(bits)

    def receive_region(self, region: round_records.EscalatedRegion) -> None:
        """Take an escalated region that landed here: every round its slot."""
        self.reserved_bits -= round_records.stated_bits(region.wire_bits)
        for packet in region.packets:
            self.writes_in_flight -= 1
            packet_bits = round_records.fragment_wire_bits(packet.fragments)
            self._land(packet, packet_bits, ESCALATION)

    def _refuse_region(self, round_count: int, bits: Optional[int]) -> None:
        """An escalated region that does not fit stops the run, by the yaml."""
        capacity = self.store.capacity_bits()
        raise RuntimeError(
            f"the strong syndrome buffer has no room for the {bits} bits "
            f"of an escalated region's {round_count} rounds: "
            f"{self.store.occupied_bits} bits stored and "
            f"{self.reserved_bits} reserved against "
            f"strong_syndrome_buffer.bits {capacity}"
        )

    def _forward_memory_round(self, packed: round_records.PackedRound) -> None:
        """A timing-only round holds a slot until its decoder hop delivers."""
        self.store.accept_packed_round(packed.packet, publication_tick=None)
        self.trace.copy_made.fire(
            packed.round_key,
            packed.wire_bits,
            CONTROLLER_WRITE.source_name,
            "strong syndrome buffer",
        )
        self.engine.log_io(
            log_sources.STRONG_BUFFER,
            lambda: self._received_text(packed.packet, CONTROLLER_WRITE),
        )
        delivered = functools.partial(self._deliver_memory_round, packed)
        self.output.send_memory_round(packed, delivered)

    def _deliver_memory_round(self, packed: round_records.PackedRound) -> None:
        source_operation_id = packed.route.source_operation_id
        self.memory_arrivals.receive_memory_round(source_operation_id)
        operation_id, round_index = packed.round_key
        event = round_records.RoundEvent.of(
            "FEEDBACK_MEMORY_DELIVERED",
            self.engine.now,
            operation_id,
            round_index,
            packed.route,
        )
        self.trace.round_event.fire(event)

    def _land(
        self,
        packet: round_records.SyndromeRoundPacket,
        packet_bits: Optional[int],
        hop: _Hop,
    ) -> None:
        """Store the round, or drop it when its operation already closed."""
        if self.store.has_operation(packet.operation_id):
            self._store_landing(packet, packet_bits, hop)
            return
        self._drop_landing(packet, packet_bits, hop)

    def _store_landing(
        self,
        packet: round_records.SyndromeRoundPacket,
        packet_bits: Optional[int],
        hop: _Hop,
    ) -> None:
        self.store.accept_packed_round(packet, publication_tick=self.engine.now)
        round_key = (packet.operation_id, packet.round_index)
        self.trace.copy_made.fire(
            round_key, packet_bits, hop.source_name, "strong syndrome buffer"
        )
        self.engine.log_io(
            log_sources.STRONG_BUFFER,
            lambda: self._received_text(packet, hop),
        )
        if self.windows is not None:
            self.windows.accept_room_round(
                packet.operation_id, packet.round_index
            )
        # Arrival can create a live window's first hold. Publish before
        # reclamation, as gem5 services targets before freeing an entry
        # (src/mem/cache/base.cc:637-656).
        self.store.release_round_if_unheld(round_key)

    def _drop_landing(
        self,
        packet: round_records.SyndromeRoundPacket,
        packet_bits: Optional[int],
        hop: _Hop,
    ) -> None:
        """The operation closed while the round crossed: it has no reader.

        The same rule as the unheld landing above, one step later. A
        store closes an operation only when no hold and no stored round
        names it (syndrome_buffer.py close_operation), and a hold on a closed
        operation cannot be registered afterwards, so a round of a closed
        operation provably has no reader ever and must not enter the
        store. The copy still fires: the bits did cross the hop, and
        the link's traffic ledger charged the transfer, so a suppressed
        copy would leave the two accounts of the same hop disagreeing.
        """
        round_key = (packet.operation_id, packet.round_index)
        self.trace.copy_made.fire(
            round_key, packet_bits, hop.source_name, "strong syndrome buffer"
        )
        self.engine.log_io(
            log_sources.STRONG_BUFFER, lambda: self._dropped_text(packet, hop)
        )

    def _received_text(
        self, packet: round_records.SyndromeRoundPacket, hop: _Hop
    ) -> str:
        defects = packet.defects_text()
        holds = self.store.held_rounds_description()
        return (
            f"received round {packet.round_index} of op {packet.operation_id} "
            f"from {hop.path_name}; {defects}; holds {holds}"
        )

    def _dropped_text(
        self, packet: round_records.SyndromeRoundPacket, hop: _Hop
    ) -> str:
        holds = self.store.held_rounds_description()
        return (
            f"dropped round {packet.round_index} of op {packet.operation_id} "
            f"from {hop.path_name}: op {packet.operation_id} "
            f"closed while the round crossed; holds {holds}"
        )

    def check_settled(self) -> None:
        """At the end of a run nothing may be in flight or held."""
        if self.writes_in_flight:
            raise RuntimeError(
                f"strong syndrome buffer ended with {self.writes_in_flight} "
                f"writes in flight"
            )
        self.store.check_settled()


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the strong syndrome round receiver reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (tmp/resources/gem5/src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    copy_made: trace_source.TraceSource = trace_source.new_source()
    round_event: trace_source.TraceSource = trace_source.new_source()
