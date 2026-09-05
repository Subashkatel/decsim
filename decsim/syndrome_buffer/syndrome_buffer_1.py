"""Syndrome buffer 1: the room-side round store that feeds the strong tier.

Every packed round is written out of the fridge exactly once over the
controller_to_strong_buffer hop (priced when the card wires it, free
otherwise) and stored here in parallel with its Buffer 0 publication.
Strong jobs point into this store and their strong_buffer_to_strong_decoder
input is assembled from it, so the two-sided strong context lives at room
temperature and Buffer 0 keeps only what the weak lane reads.

Buffer semantics (refcounted holds, orphan until arrival, tombstoned late
writes, refuse before mutation) have one owner: the composed
SyndromeBuffer. This class adds only the crossing, the arrival gate, and
the accounting.
"""

from typing import Callable, Optional

import decsim.message as message
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module


class SyndromeBuffer1:
    """Own the crossing and the room-side retention of every round."""

    def __init__(
        self,
        engine,
        links,
        *,
        capacity_rounds: Optional[int] = None,
        memory_model=None,
        on_round_stored: Optional[Callable] = None,
    ) -> None:
        self.engine = engine
        self.links = links
        self.store = syndrome_buffer_module.SyndromeBuffer(
            capacity=capacity_rounds, memory_model=memory_model
        )
        self.capacity_rounds = capacity_rounds
        # room-side arrival counter per operation: the strong tier's
        # data-readiness gates read this, never Buffer 0's counter
        self.rounds_arrived: dict = {}
        self.copied_bits_total = 0
        self.on_round_stored = on_round_stored
        # flight-recorder rows: (tick, operation_id, round_index) per store
        self.stored_log: list = []
        self._in_flight_writes = 0
        self._written: set = set()

    def has_room(self) -> bool:
        """A write can land: capacity counts the rounds stored and in flight."""
        if self.capacity_rounds is None:
            return True
        occupied = self._occupied_rounds()
        return occupied + self._in_flight_writes < self.capacity_rounds

    def _occupied_rounds(self) -> int:
        metrics = self.store.metrics()
        return metrics.live_allocations

    # ------------------------------------------------------------- writes

    def write(
        self,
        packet: message.SyndromeRoundPacket,
        *,
        packet_bits: Optional[int],
        attribution: message.TransferAttribution,
    ) -> None:
        """The dual write: one crossing, then the round is stored.

        Capacity is checked before the send, counting writes still in
        flight, so a refused write leaves no trace on the wire or the
        store.
        """
        operation_id = packet.operation_id
        identity = (operation_id, packet.round_index)
        if identity in self._written:
            raise ValueError(f"round {identity!r} was already written")
        if not self.store.has_operation(operation_id):
            self.store.open_operation(operation_id)
        if self.capacity_rounds is not None:
            occupied = self._occupied_rounds()
            after_write = occupied + self._in_flight_writes + 1
            if after_write > self.capacity_rounds:
                raise RuntimeError(
                    f"syndrome buffer 1 over capacity: {after_write} rounds "
                    f"exceed {self.capacity_rounds}"
                )
        self._written.add(identity)
        self.copied_bits_total += packet_bits or 0
        is_priced = self.links.is_wired(
            message.LinkPath.CONTROLLER_TO_STRONG_BUFFER
        )
        if not is_priced:
            self._store(packet)
            return
        self._in_flight_writes += 1

        def land(_transfer) -> None:
            self._in_flight_writes -= 1
            self._store(packet)

        self.links.send(
            message.LinkPath.CONTROLLER_TO_STRONG_BUFFER,
            packet_bits,
            self.engine.now,
            attribution,
            land,
        )

    def _store(self, packet: message.SyndromeRoundPacket) -> None:
        admission = self.store.accept_packed_round(
            packet, publication_tick=self.engine.now
        )
        if admission.refused:
            identity = (packet.operation_id, packet.round_index)
            raise RuntimeError(
                f"syndrome buffer 1 refused round {identity!r} at landing"
            )
        # a round whose every reader resolved while it crossed the link
        # is dropped at the door: nobody can ever read it
        self.store.release_round_if_unheld(admission.round_identity)
        operation_id = packet.operation_id
        self.stored_log.append(
            (self.engine.now, operation_id, packet.round_index)
        )
        arrived = self.rounds_arrived.get(operation_id, 0)
        self.rounds_arrived[operation_id] = max(arrived, packet.round_index)
        self.engine.log_io(
            "SyndromeBuffer1", lambda: self._received_text(packet)
        )
        if self.on_round_stored is not None:
            self.on_round_stored(operation_id)

    def _received_text(self, packet: message.SyndromeRoundPacket) -> str:
        defects = packet.defects_text()
        holds = self.store.held_rounds_description()
        return (
            f"received round {packet.round_index} of op {packet.operation_id} "
            f"from controller_to_strong_buffer; {defects}; holds {holds}"
        )

    # -------------------------------------------------------------- reads

    def retained_fragments(self, round_identity) -> Optional[tuple]:
        """The retained fragments, or None before and after retention."""
        return self.store.retained_fragments(round_identity)

    def publication_tick(self, round_identity) -> Optional[int]:
        """The tick the round landed here, or None."""
        return self.store.publication_tick(round_identity)

    def check_rounds_stored(self, round_identities) -> None:
        """A strong input reads only rounds that have landed here.

        A round not stored raises; it is never silently served early.
        """
        for identity in round_identities:
            tick = self.store.publication_tick(identity)
            if tick is None:
                raise RuntimeError(
                    f"round {identity!r} is not stored in syndrome buffer 1"
                )

    # -------------------------------------------------------------- holds

    def register_hold(self, holder, round_identities) -> None:
        """Keep the listed rounds alive for one consumer token."""
        self._open_referenced_operations(round_identities)
        self.store.register_hold(holder, round_identities)

    def replace_hold(self, holder, round_identities) -> None:
        """Re-point a live hold at a new set of rounds."""
        self._open_referenced_operations(round_identities)
        self.store.replace_hold(holder, round_identities)

    def _open_referenced_operations(self, round_identities) -> None:
        # holds pre-register future rounds at plan load, before any write
        # has opened their operation on this store
        operation_ids = {identity[0] for identity in round_identities}
        for operation_id in operation_ids:
            if not self.store.has_operation(operation_id):
                self.store.open_operation(operation_id)

    def transfer_hold(self, old_holder, new_holder) -> None:
        """Move a live hold to a new token without freeing its rounds."""
        self.store.transfer_hold(old_holder, new_holder)

    def release_hold(self, holder) -> None:
        """Drop a hold; rounds with no remaining holder become releasable."""
        self.store.release_hold(holder)

    def has_hold(self, holder) -> bool:
        """True while this holder token is live."""
        return self.store.has_hold(holder)

    def hold_round_identities(self, holder) -> tuple:
        """The rounds a live holder keeps."""
        return self.store.hold_round_identities(holder)

    def has_operation(self, operation_id) -> bool:
        """True while the operation may still receive rounds."""
        return self.store.has_operation(operation_id)

    def has_live_operation_reference(self, operation_id) -> bool:
        """True while any live hold refers to this operation."""
        return self.store.has_live_operation_reference(operation_id)

    def open_operation(self, operation_id) -> None:
        """Admit rounds of this operation."""
        self.store.open_operation(operation_id)

    def close_operation(self, operation_id) -> None:
        """Retire an operation once none of its rounds or holds are live."""
        self.store.close_operation(operation_id)

    # -------------------------------------------------------------- close

    def peak_occupancy_rounds(self) -> int:
        """The most rounds the store held at once."""
        metrics = self.store.metrics()
        return metrics.peak_live_allocations

    def check_settled(self) -> None:
        """At the end of a run nothing may still be held: a leak is a bug."""
        if self._in_flight_writes:
            raise RuntimeError(
                f"syndrome buffer 1 ended with {self._in_flight_writes} "
                f"controller_to_strong_buffer writes in flight"
            )
        snapshot = self.store.snapshot()
        if snapshot.occupancy:
            held = [identity for identity, _slot in snapshot.identity_to_slot]
            raise RuntimeError(
                f"syndrome buffer 1 still holds rounds {held} at the end"
            )
        if snapshot.hold_counts:
            held = [identity for identity, _count in snapshot.hold_counts]
            raise RuntimeError(
                f"syndrome buffer 1 has unresolved holds on {held} "
                f"(rounds expected but never written, or holders never "
                f"released)"
            )
