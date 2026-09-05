"""The priced controller_to_strong_buffer crossing into a round store.

Every packed round is written out of the fridge exactly once over the
crossing (priced when the card wires it, free otherwise) and stored on
the room side in parallel with its Buffer 0 publication, so the strong
tier's context lives at room temperature. The writer answers has_room
counting the writes still in flight, gem5's queue counting its reserved
entries (src/mem/cache/queue.hh isFull), and stores each round at its
landing; the store's holds and lifetime are RoundStore's.
"""

from typing import Callable, Optional

import decsim.message as message
import decsim.observe.round_events as round_events
import decsim.syndrome_buffer.round_store as round_store_module


class StrongRoundWriter:
    """The crossing, the writes in flight, and the landing into the store."""

    def __init__(
        self,
        engine,
        link,
        store: round_store_module.RoundStore,
        *,
        on_round_stored: Optional[Callable] = None,
        recorder=None,
    ) -> None:
        self.engine = engine
        self.link = link
        self.store = store
        self.writes_in_flight = 0
        # hears (operation_id, round_index) once a round is stored
        self.on_round_stored = on_round_stored
        if recorder is None:
            recorder = round_events.NoRoundEvents()
        self.recorder = recorder

    def has_room(self) -> bool:
        """A write can land: capacity counts the rounds stored and in flight."""
        capacity = self.store.settings.rounds
        if capacity is None:
            return True
        return self.store.occupancy + self.writes_in_flight < capacity

    def write(
        self,
        packet: message.SyndromeRoundPacket,
        *,
        packet_bits: Optional[int],
        attribution: message.TransferAttribution,
    ) -> None:
        """Carry the round over the crossing and store it at landing."""
        assert self.has_room(), "a round was written into a full strong store"
        is_priced = self.link.is_wired(
            message.LinkPath.CONTROLLER_TO_STRONG_BUFFER
        )
        if not is_priced:
            self._land(packet)
            return
        self.writes_in_flight += 1

        def landed(_transfer) -> None:
            self.writes_in_flight -= 1
            self._land(packet)

        self.link.send(
            message.LinkPath.CONTROLLER_TO_STRONG_BUFFER,
            packet_bits,
            self.engine.now,
            attribution,
            landed,
        )

    def _land(self, packet: message.SyndromeRoundPacket) -> None:
        self.store.accept_packed_round(packet, publication_tick=self.engine.now)
        round_key = (packet.operation_id, packet.round_index)
        # a round whose every reader resolved while it crossed the link
        # is dropped at the door: nobody can ever read it
        self.store.release_round_if_unheld(round_key)
        self.recorder.round_stored(packet.operation_id, packet.round_index)
        self.engine.log_io(
            "SyndromeBuffer1", lambda: self._received_text(packet)
        )
        if self.on_round_stored is not None:
            self.on_round_stored(packet.operation_id, packet.round_index)

    def _received_text(self, packet: message.SyndromeRoundPacket) -> str:
        defects = packet.defects_text()
        holds = self.store.held_rounds_description()
        return (
            f"received round {packet.round_index} of op {packet.operation_id} "
            f"from controller_to_strong_buffer; {defects}; holds {holds}"
        )

    def check_settled(self) -> None:
        """At the end of a run nothing may be in flight or held."""
        if self.writes_in_flight:
            raise RuntimeError(
                f"syndrome buffer 1 ended with {self.writes_in_flight} "
                f"controller_to_strong_buffer writes in flight"
            )
        self.store.check_settled()
