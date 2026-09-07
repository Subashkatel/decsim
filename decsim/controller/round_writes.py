"""The writer: a finished round into every store it must reach, or held.

The backpressure law of the readout path: a store answers has_room before
any write; a finished round that finds no room in either store waits in
HeldRounds, upstream of the stores, and enters when a slot frees, in the
order the rounds were completed. gem5's blocked port keeps the request at
the requester and re-sends it when clearBlocked schedules the retry
(src/mem/cache/base.cc, setBlocked, clearBlocked, processSendRetry);
Ciw's Type I blocking keeps the customer at the upstream node and
releases the longest blocked one when the destination has capacity
(ciw/node.py, block_individual, release_blocked_individual). Nothing is
reordered; under the stall policy nothing is dropped. The written round
leaves on its route at the write (RoundTransmitter).
"""

from typing import Callable

import decsim.controller.settings as controller_settings
import decsim.observe.trace_source as trace_source
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records


class HeldRounds:
    """The waiting line in front of the stores, and what a full store does.

    Trace source: round_event(RoundEvent) with kind STALLED when a round
    is held for room, DROPPED when the policy drops it.
    """

    def __init__(
        self,
        engine,
        on_full: controller_settings.PackingOverflowPolicy,
    ) -> None:
        self.engine = engine
        # (held round, the admission it retries), in completion order
        self.waiting: list = []
        self.on_full = on_full
        self.round_event = trace_source.TraceSource()

    def refuse(
        self,
        packed: round_records.PackedRound,
        admit: Callable[[round_records.PackedRound], bool],
    ) -> bool:
        """A round found no room: hold it for a retry, or drop it.

        False either way, so the admission that called reports the
        refusal; a held round is recorded STALLED once.
        """
        operation_id, round_index = packed.round_key
        drop = controller_settings.PackingOverflowPolicy.DROP_ROUND
        if self.on_full is drop:
            dropped = round_records.RoundEvent.of(
                "DROPPED",
                self.engine.now,
                operation_id,
                round_index,
                packed.route,
            )
            self.round_event.fire(dropped)
            return False
        if self._is_holding(packed):
            return False
        self.waiting.append((packed, admit))
        stalled = round_records.RoundEvent.of(
            "STALLED", self.engine.now, operation_id, round_index, packed.route
        )
        self.round_event.fire(stalled)
        return False

    def retry(self) -> None:
        """A slot freed: admit from the head, stop at the first refused."""
        while self.waiting:
            packed, admit = self.waiting[0]
            admitted = admit(packed)
            if not admitted:
                return
            self.waiting.pop(0)

    @property
    def count(self) -> int:
        """How many rounds wait."""
        return len(self.waiting)

    def _is_holding(self, packed: round_records.PackedRound) -> bool:
        for held, _admit in self.waiting:
            if held is packed:
                return True
        return False


class RoundWriter:
    """Writes a finished round into Buffer 0 and the strong store, or holds.

    Trace sources: round_event(RoundEvent) with kind PUBLISHED when the
    round is published as it is stored; copy_made(round_key, bits,
    "controller assembler", "Buffer 0") for the write (data_path.md hop
    3). It narrates Buffer 0's intake on the engine's io_line.
    """

    def __init__(
        self,
        engine,
        weak_store,
        strong_writer,
        *,
        publishes_from_strong_store: bool,
        held_rounds: HeldRounds,
        transmitter,
    ) -> None:
        self.engine = engine
        self.weak_store = weak_store
        self.strong_writer = strong_writer
        # a strong-primary plan reads its windows from the room side, so a
        # window-input round takes one hop, into the strong store; a
        # feedback-memory round still crosses Buffer 0
        self.publishes_from_strong_store = publishes_from_strong_store
        self.held_rounds = held_rounds
        self.transmitter = transmitter
        self.round_event = trace_source.TraceSource()
        self.copy_made = trace_source.TraceSource()

    def admit(self, packed: round_records.PackedRound) -> bool:
        """Write the round where it belongs; False when it had to wait."""
        if self._takes_the_strong_hop_only(packed):
            if not self.strong_writer.has_room():
                return self.held_rounds.refuse(packed, self.admit)
            self._write_strong(packed)
            return True
        if not self.weak_store.has_room():
            return self.held_rounds.refuse(packed, self.admit)
        if not self._strong_has_room():
            return self.held_rounds.refuse(packed, self.admit)
        publication_tick = self.transmitter.publication_tick_at_storage(
            packed.route
        )
        self.weak_store.accept_packed_round(
            packed.packet, publication_tick=publication_tick
        )
        operation_id, round_index = packed.round_key
        self.copy_made.fire(
            packed.round_key,
            packed.wire_bits,
            "controller assembler",
            "Buffer 0",
        )
        if publication_tick is not None:
            published = round_records.RoundEvent.of(
                "PUBLISHED",
                publication_tick,
                operation_id,
                round_index,
                packed.route,
            )
            self.round_event.fire(published)
        self.engine.log_io(
            "Buffer 0", lambda: _received_text(packed.packet, self.weak_store)
        )
        if self.strong_writer is not None:
            # the dual write: the same round leaves for the room side in
            # parallel with its Buffer 0 publication
            self._write_strong(packed)
        self.transmitter.send(packed)
        return True

    def check_settled(self) -> None:
        """At the end of a run no round may still wait for room."""
        if self.held_rounds.count:
            raise RuntimeError(
                f"run ended with {self.held_rounds.count} rounds held for "
                f"store room"
            )

    def _takes_the_strong_hop_only(
        self, packed: round_records.PackedRound
    ) -> bool:
        window_input = round_records.SyndromePacketRouteKind.WINDOW_INPUT
        on_window_route = packed.route.kind is window_input
        return on_window_route and self.publishes_from_strong_store

    def _strong_has_room(self) -> bool:
        if self.strong_writer is None:
            return True
        return self.strong_writer.has_room()

    def _write_strong(self, packed: round_records.PackedRound) -> None:
        attribution = transfer_records.TransferAttribution.for_packet(
            packed.packet
        )
        self.strong_writer.write(
            packed.packet, packet_bits=packed.wire_bits, attribution=attribution
        )


def _received_text(
    packet: round_records.SyndromeRoundPacket, weak_store
) -> str:
    defects = packet.defects_text()
    holds = weak_store.held_rounds_description()
    return (
        f"received round {packet.round_index} of "
        f"op {packet.operation_id} from packing; {defects}; holds {holds}"
    )
