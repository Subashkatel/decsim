"""Syndrome buffer 1: the room-side round store that feeds the strong tier.

Every packed round is written out of the fridge exactly once over the
csb hop (priced when the card wires ``LinkPath.CSB``, free
otherwise) and stored here in parallel with its Buffer 0 publication.
Strong jobs point into this store and their SBD input is assembled from it,
so the two-sided strong context lives at room temperature and Buffer 0
keeps only what the weak lane reads.

Buffer semantics (refcounted holds, orphan until arrival, tombstoned late
writes, refuse before mutation) have one owner: the composed
``SyndromeBuffer``. This class adds only the crossing, the arrival gate,
and the accounting.
"""

from __future__ import annotations

from typing import Callable, Optional

from ..links.links import LinkPath, TrafficAttribution
from ..message import SyndromeRoundPacket
from .syndrome_buffer import SyndromeBuffer


class SyndromeBuffer1:
    """Own the csb crossing and the room-side retention of every round."""

    def __init__(
        self, engine, links, *, capacity_rounds: Optional[int] = None,
        memory_model=None,
        on_round_stored: Optional[Callable] = None,
    ) -> None:
        self.engine = engine
        self.links = links
        self.store = SyndromeBuffer(
            capacity=capacity_rounds, memory_model=memory_model)
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

    # ------------------------------------------------------------- writes

    def write(self, packet: SyndromeRoundPacket, *, packet_bits: Optional[int],
              attribution: TrafficAttribution) -> None:
        """The dual write: one csb crossing, then the round is stored.

        Capacity is checked before the link reservation, counting writes
        still in flight, so a refused write leaves no trace on the
        serializer or the store."""
        operation_id = packet.operation_id
        identity = (operation_id, packet.round_index)
        if identity in self._written:
            raise ValueError(f"round {identity!r} was already written")
        if not self.store.has_operation(operation_id):
            self.store.open_operation(operation_id)
        if self.capacity_rounds is not None:
            occupied = self.store.metrics().live_allocations
            if occupied + self._in_flight_writes + 1 > self.capacity_rounds:
                raise RuntimeError(
                    f"syndrome buffer 1 over capacity: "
                    f"{occupied + self._in_flight_writes + 1} rounds exceed "
                    f"{self.capacity_rounds}")
        self._written.add(identity)
        csb_is_priced = LinkPath.CSB in self.links.paths
        if not csb_is_priced:
            self.copied_bits_total += packet_bits or 0
            self._store(packet)
            return
        reservation = self.links.reserve(
            LinkPath.CSB, payload_bits=packet_bits,
            now_ticks=self.engine.now, attribution=attribution)
        self.copied_bits_total += reservation.payload_bits or 0
        delay_ticks = reservation.total_delay_ticks
        if delay_ticks == 0:
            self._store(packet)
            return
        self._in_flight_writes += 1

        def land() -> None:
            self._in_flight_writes -= 1
            self._store(packet)

        self.engine.schedule(delay_ticks, land,
                             label="csb -> syndrome buffer 1")

    def _store(self, packet: SyndromeRoundPacket) -> None:
        admission = self.store.accept_packed_round(
            packet, publication_tick=self.engine.now)
        if admission.refused:
            raise RuntimeError(
                f"syndrome buffer 1 refused round "
                f"{(packet.operation_id, packet.round_index)!r} at landing")
        # a round whose every reader resolved while it crossed the csb
        # is dropped at the door: nobody can ever read it
        self.store.release_round_if_unheld(admission.round_identity)
        operation_id = packet.operation_id
        self.stored_log.append(
            (self.engine.now, operation_id, packet.round_index))
        arrived = self.rounds_arrived.get(operation_id, 0)
        self.rounds_arrived[operation_id] = max(arrived, packet.round_index)
        self.engine.log_io(
            "SyndromeBuffer1",
            lambda: f"received round {packet.round_index} of op {operation_id} "
                    f"from csb; {packet.defects_text()}; holds "
                    f"{self.store.held_rounds_description()}")
        if self.on_round_stored is not None:
            self.on_round_stored(operation_id)

    # -------------------------------------------------------------- reads

    def retained_fragments(self, round_identity) -> Optional[tuple]:
        return self.store.retained_fragments(round_identity)

    def publication_tick(self, round_identity) -> Optional[int]:
        return self.store.publication_tick(round_identity)

    def ready_tick(self, round_identities) -> int:
        """Latest room-side arrival of the listed rounds; a round not stored
        raises, it is never silently served early."""
        latest = 0
        for identity in round_identities:
            tick = self.store.publication_tick(identity)
            if tick is None:
                raise RuntimeError(
                    f"round {identity!r} is not stored in syndrome buffer 1")
            latest = max(latest, tick)
        return latest

    # -------------------------------------------------------------- holds

    def register_hold(self, holder, round_identities) -> None:
        self._open_referenced_operations(round_identities)
        self.store.register_hold(holder, round_identities)

    def replace_hold(self, holder, round_identities) -> None:
        self._open_referenced_operations(round_identities)
        self.store.replace_hold(holder, round_identities)

    def _open_referenced_operations(self, round_identities) -> None:
        # holds pre-register future rounds at plan load, before any write
        # has opened their operation on this store
        for operation_id in {identity[0] for identity in round_identities}:
            if not self.store.has_operation(operation_id):
                self.store.open_operation(operation_id)

    def transfer_hold(self, old_holder, new_holder) -> None:
        self.store.transfer_hold(old_holder, new_holder)

    def release_hold(self, holder) -> None:
        self.store.release_hold(holder)

    def has_hold(self, holder) -> bool:
        return self.store.has_hold(holder)

    def hold_round_identities(self, holder) -> tuple:
        return self.store.hold_round_identities(holder)

    def has_operation(self, operation_id) -> bool:
        return self.store.has_operation(operation_id)

    def has_live_operation_reference(self, operation_id) -> bool:
        return self.store.has_live_operation_reference(operation_id)

    def open_operation(self, operation_id) -> None:
        self.store.open_operation(operation_id)

    def close_operation(self, operation_id) -> None:
        self.store.close_operation(operation_id)

    # -------------------------------------------------------------- close

    def peak_occupancy_rounds(self) -> int:
        return self.store.metrics().peak_live_allocations

    def check_settled(self) -> None:
        """At the end of a run nothing may still be held: a leak is a bug."""
        if self._in_flight_writes:
            raise RuntimeError(
                f"syndrome buffer 1 ended with {self._in_flight_writes} "
                f"csb writes in flight")
        snapshot = self.store.snapshot()
        if snapshot.occupancy:
            raise RuntimeError(
                f"syndrome buffer 1 still holds rounds "
                f"{[identity for identity, _ in snapshot.identity_to_slot]} "
                f"at the end")
        if snapshot.hold_counts:
            held = [identity for identity, _count in snapshot.hold_counts]
            raise RuntimeError(
                f"syndrome buffer 1 has unresolved holds on {held} "
                f"(rounds expected but never written, or holders never "
                f"released)")
