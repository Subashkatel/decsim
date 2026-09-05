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
import decsim.message as message


class HeldRounds:
    """The waiting line in front of the stores, and what a full store does."""

    def __init__(
        self,
        on_full: controller_settings.PackingOverflowPolicy,
        recorder,
    ) -> None:
        # (held round, the admission it retries), in completion order
        self.waiting: list = []
        self.on_full = on_full
        self.recorder = recorder

    def refuse(
        self,
        packed: message.PackedRound,
        admit: Callable[[message.PackedRound], bool],
    ) -> bool:
        """A round found no room: hold it for a retry, or drop it.

        False either way, so the admission that called reports the
        refusal; a held round is recorded STALLED once.
        """
        operation_id, round_index = packed.round_key
        drop = controller_settings.PackingOverflowPolicy.DROP_ROUND
        if self.on_full is drop:
            self.recorder.round_dropped(operation_id, round_index, packed.route)
            return False
        if self._is_holding(packed):
            return False
        self.waiting.append((packed, admit))
        self.recorder.record("STALLED", operation_id, round_index, packed.route)
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

    def _is_holding(self, packed: message.PackedRound) -> bool:
        for held, _admit in self.waiting:
            if held is packed:
                return True
        return False


class RoundWriter:
    """Writes a finished round into Buffer 0 and the strong store, or holds."""

    def __init__(
        self,
        weak_store,
        strong_writer,
        *,
        publishes_from_strong_store: bool,
        held_rounds: HeldRounds,
        transmitter,
        recorder,
    ) -> None:
        self.weak_store = weak_store
        self.strong_writer = strong_writer
        # a strong-primary plan reads its windows from the room side, so a
        # window-input round takes one hop, into the strong store; a
        # feedback-memory round still crosses Buffer 0
        self.publishes_from_strong_store = publishes_from_strong_store
        self.held_rounds = held_rounds
        self.transmitter = transmitter
        self.recorder = recorder

    def admit(self, packed: message.PackedRound) -> bool:
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
        if publication_tick is not None:
            self.recorder.record(
                "PUBLISHED",
                operation_id,
                round_index,
                packed.route,
                tick=publication_tick,
            )
        self.recorder.weak_store_received(packed.packet, self.weak_store)
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

    def _takes_the_strong_hop_only(self, packed: message.PackedRound) -> bool:
        window_input = message.SyndromePacketRouteKind.WINDOW_INPUT
        on_window_route = packed.route.kind is window_input
        return on_window_route and self.publishes_from_strong_store

    def _strong_has_room(self) -> bool:
        if self.strong_writer is None:
            return True
        return self.strong_writer.has_room()

    def _write_strong(self, packed: message.PackedRound) -> None:
        attribution = message.TransferAttribution.for_packet(packed.packet)
        self.strong_writer.write(
            packed.packet, packet_bits=packed.wire_bits, attribution=attribution
        )
