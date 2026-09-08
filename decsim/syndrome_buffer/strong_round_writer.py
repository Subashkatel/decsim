"""The priced controller_to_strong_buffer crossing into a round store.

Every packed round is written out of the fridge exactly once over the
crossing and stored on the room side in parallel with its Buffer 0
publication, so the strong tier's context lives at room temperature. The
writer answers has_room counting the writes still in flight, gem5's
queue counting its reserved entries (src/mem/cache/queue.hh isFull), and
stores each round at its landing; the store's holds and lifetime are
RoundStore's.
"""

import dataclasses
from typing import Callable, Optional

import decsim.records.log_sources as log_sources
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.trace_source as trace_source


class StrongRoundWriter:
    """The crossing, the writes in flight, and the landing into the store.

    Trace source: copy_made(round_key, bits, "controller assembler",
    "Buffer 1") at every landing, the dual write's copy (data_path.md
    hop 4).
    """

    def __init__(
        self,
        engine,
        link,
        store: round_store_module.RoundStore,
        *,
        on_round_stored: Optional[Callable] = None,
    ) -> None:
        self.engine = engine
        self.link = link
        self.store = store
        self.writes_in_flight = 0
        # hears (operation_id, round_index) once a round is stored
        self.on_round_stored = on_round_stored
        self.trace = _TraceSources()

    def has_room(self) -> bool:
        """A write can land: capacity counts the rounds stored and in flight."""
        capacity = self.store.settings.rounds
        if capacity is None:
            return True
        return self.store.occupancy + self.writes_in_flight < capacity

    def write(
        self,
        packet: round_records.SyndromeRoundPacket,
        *,
        packet_bits: Optional[int],
        attribution: transfer_records.TransferAttribution,
    ) -> None:
        """Carry the round over the crossing and store it at landing."""
        assert self.has_room(), "a round was written into a full strong store"
        self.writes_in_flight += 1

        def landed(_transfer) -> None:
            self.writes_in_flight -= 1
            self._land(packet, packet_bits)

        self.link.send(
            transfer_records.LinkPath.CONTROLLER_TO_STRONG_BUFFER,
            packet_bits,
            self.engine.now,
            attribution,
            landed,
        )

    def _land(
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

    def _received_text(self, packet: round_records.SyndromeRoundPacket) -> str:
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


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the strong round writer reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (tmp/resources/gem5/src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    copy_made: trace_source.TraceSource = trace_source.new_source()
