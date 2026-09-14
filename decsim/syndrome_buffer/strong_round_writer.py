"""The room-side end of controller_to_strong_buffer: room, then landing.

Every packed round is carried out of the fridge exactly once and stored
on the room side in parallel with its Buffer 0 publication, so the
strong tier's context lives at room temperature. The controller
executes that crossing, being the end the round leaves by (OMNeT++
refuses a module that sends a message it does not own,
tmp/resources/omnetpp/src/sim/csimplemodule.cc:333-334; gem5 bills a
transfer to the port it left by, packet.hh:424-431). This end owns the
room and the landing: it answers has_room counting the writes still in
flight, reserves one before the crossing starts, gem5's queue counting
its reserved entries as taken (src/mem/cache/queue.hh:150-152 isFull,
src/mem/cache/base.cc allocateWriteBuffer), and stores each round at its
landing; the store's holds and lifetime are RoundStore's. A landing
whose operation closed while the round crossed is dropped at the door
instead of stored, since no reader can ever name it (_drop_landing).
"""

import dataclasses
from typing import Callable, Optional

import decsim.records.log_sources as log_sources
import decsim.records.rounds as round_records
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.trace_source as trace_source


class StrongRoundWriter:
    """The room, the writes in flight, and the landing into the store.

    Trace source: copy_made(round_key, bits, "controller assembler",
    "Buffer 1") at every landing, the dual write's copy (data_path.md
    hop 3).
    """

    def __init__(
        self,
        engine,
        store: round_store_module.RoundStore,
        *,
        on_round_stored: Optional[Callable] = None,
    ) -> None:
        self.engine = engine
        self.store = store
        self.writes_in_flight = 0
        # hears (operation_id, round_index) once a round is stored
        self.on_round_stored = on_round_stored
        self.trace = _TraceSources()

    def has_room(self) -> bool:
        """A write can land: capacity counts the rounds stored and in flight."""
        capacity = self.store.capacity_rounds()
        if capacity is None:
            return True
        return self.store.occupancy + self.writes_in_flight < capacity

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
        self._land(packet, packet_bits)

    def _land(
        self,
        packet: round_records.SyndromeRoundPacket,
        packet_bits: Optional[int],
    ) -> None:
        """Store the round, or drop it when its operation already closed."""
        if self.store.has_operation(packet.operation_id):
            self._store_landing(packet, packet_bits)
            return
        self._drop_landing(packet, packet_bits)

    def _store_landing(
        self,
        packet: round_records.SyndromeRoundPacket,
        packet_bits: Optional[int],
    ) -> None:
        self.store.accept_packed_round(packet, publication_tick=self.engine.now)
        round_key = (packet.operation_id, packet.round_index)
        # a round whose every reader resolved while it crossed the link
        # is dropped at the door: nobody can ever read it
        self.store.release_round_if_unheld(round_key)
        self.trace.copy_made.fire(
            round_key, packet_bits, "controller assembler", "Buffer 1"
        )
        self.engine.log_io(
            log_sources.STRONG_BUFFER, lambda: self._received_text(packet)
        )
        if self.on_round_stored is not None:
            self.on_round_stored(packet.operation_id, packet.round_index)

    def _drop_landing(
        self,
        packet: round_records.SyndromeRoundPacket,
        packet_bits: Optional[int],
    ) -> None:
        """The operation closed while the round crossed: it has no reader.

        The same rule as the unheld landing above, one step later. A
        store closes an operation only when no hold and no stored round
        names it (round_store.py close_operation), and a hold on a closed
        operation cannot be registered afterwards, so a round of a closed
        operation provably has no reader ever and must not enter the
        store. The copy still fires: the bits did cross
        controller_to_strong_buffer, and the link's traffic ledger
        charged the transfer, so a suppressed copy would leave the two
        accounts of the same hop disagreeing.
        """
        round_key = (packet.operation_id, packet.round_index)
        self.trace.copy_made.fire(
            round_key, packet_bits, "controller assembler", "Buffer 1"
        )
        self.engine.log_io(
            log_sources.STRONG_BUFFER, lambda: self._dropped_text(packet)
        )

    def _received_text(self, packet: round_records.SyndromeRoundPacket) -> str:
        defects = packet.defects_text()
        holds = self.store.held_rounds_description()
        return (
            f"received round {packet.round_index} of op {packet.operation_id} "
            f"from controller_to_strong_buffer; {defects}; holds {holds}"
        )

    def _dropped_text(self, packet: round_records.SyndromeRoundPacket) -> str:
        holds = self.store.held_rounds_description()
        return (
            f"dropped round {packet.round_index} of op {packet.operation_id} "
            f"from controller_to_strong_buffer: op {packet.operation_id} "
            f"closed while the round crossed; holds {holds}"
        )

    def check_settled(self) -> None:
        """At the end of a run nothing may be in flight or held."""
        if self.writes_in_flight:
            raise RuntimeError(
                f"syndrome buffer 1 ended with {self.writes_in_flight} "
                f"controller_to_strong_buffer writes in flight"
            )
        self.store.check_settled()


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the strong round writer reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (tmp/resources/gem5/src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    copy_made: trace_source.TraceSource = trace_source.new_source()
