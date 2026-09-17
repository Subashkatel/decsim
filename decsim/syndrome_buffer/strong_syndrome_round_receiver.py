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
transfer to the port it left by, packet.hh:424-431). This end owns the
room and the landing: it answers has_room counting the writes still in
flight, reserves the room before a crossing starts, gem5's queue
counting its reserved entries as taken (src/mem/cache/queue.hh:150-152
isFull, src/mem/cache/base.cc allocateWriteBuffer), and stores each
round at its landing; the store's holds and lifetime are
SyndromeBuffer's. A landing whose operation closed while its bits
crossed is dropped at the door instead of stored, since no reader can
ever name it (_drop_landing): a strong-primary run's last rounds, or
the region of a parallel strong sibling that a confident weak result
cancelled while the region was on the wire.
"""

import dataclasses
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
    buffer, whichever the round left.
    """

    # a receiver built with no window side stores its rounds for a reader
    # that never asks
    windows = ports.Port(ports.WindowInput, optional=True)
    store = ports.Port(ports.SyndromeBuffer)

    def __init__(self, engine) -> None:
        self.engine = engine
        self.writes_in_flight = 0
        self.trace = _TraceSources()

    def has_room(self) -> bool:
        """A write can land: capacity counts the rounds stored and in flight."""
        return self._has_room_for(1)

    def reserve_write(self) -> None:
        """Take the room one crossing round will need, before it leaves."""
        assert self.has_room(), "a round was written into a full strong store"
        self.writes_in_flight += 1

    def receive_round(
        self,
        packet: round_records.SyndromeRoundPacket,
        packet_bits: Optional[int],
    ) -> None:
        """Take one round that landed here: its room is now its slot."""
        self.writes_in_flight -= 1
        self._land(packet, packet_bits, CONTROLLER_WRITE)

    def reserve_region(self, round_count: int) -> None:
        """Take the room an escalated region needs, before it leaves the chip.

        An escalation cannot wait: the weak result is already given up,
        so a strong store with no room for the region stops the run
        rather than holding the chip. The yaml sizes the store.
        """
        if not self._has_room_for(round_count):
            capacity = self.store.capacity_rounds()
            raise RuntimeError(
                f"the strong syndrome buffer has no room for the "
                f"{round_count} rounds of an escalated region: "
                f"{self.store.occupancy} stored and {self.writes_in_flight} "
                f"in flight against strong_syndrome_buffer.rounds {capacity}"
            )
        self.writes_in_flight += round_count

    def receive_region(self, region: round_records.EscalatedRegion) -> None:
        """Take an escalated region that landed here: every round its slot."""
        for packet in region.packets:
            self.writes_in_flight -= 1
            packet_bits = round_records.fragment_wire_bits(packet.fragments)
            self._land(packet, packet_bits, ESCALATION)

    def _has_room_for(self, round_count: int) -> bool:
        capacity = self.store.capacity_rounds()
        if capacity is None:
            return True
        taken = self.store.occupancy + self.writes_in_flight
        return taken + round_count <= capacity

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
        # a round whose every reader resolved while it crossed the link
        # is dropped at the door: nobody can ever read it
        self.store.release_round_if_unheld(round_key)
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
