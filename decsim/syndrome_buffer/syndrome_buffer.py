"""Store finished syndrome rounds until every consumer is done with them.

The controller's syndrome packing assembles and forms each round; this
store accepts only the finished packed round (accept_packed_round, its
one intake), gives it a slot, and consumer holds keep it alive. The slot
is freed after its last hold is released.

This module owns data lifetime only. It does not assemble fragments,
schedule events, model links, or manage decoder queues.
"""

import dataclasses
from typing import Optional, Protocol, runtime_checkable

import decsim.message as message


@runtime_checkable
class MemoryModel(Protocol):
    """An observer of retained payload storage: store and evict per fragment."""

    def store(self, key, payload) -> None:
        """One fragment was retained under the key."""

    def evict(self, key) -> None:
        """The fragment under the key was released."""


# ---- consumer hold tokens: who keeps rounds in Buffer 0 and why


@dataclasses.dataclass(frozen=True)
class PotentialStrong:
    """Buffer 0 hold: a window's rounds, kept in case its result escalates."""

    window_key: tuple


@dataclasses.dataclass(frozen=True)
class PendingStrong:
    """Buffer 0 hold: rounds for an admitted, not yet served strong request."""

    request_key: message.DecoderRequestKey


@dataclasses.dataclass(frozen=True)
class CsdInput:
    """Buffer 0 hold: rounds in flight to a strong decoder over SBD."""

    request_key: message.DecoderRequestKey


@dataclasses.dataclass(frozen=True)
class DecoderInputHold:
    """Buffer 0 hold: a decode job's rounds until they land in unit memory."""

    request_key: message.DecoderRequestKey


@dataclasses.dataclass(frozen=True)
class RephaseGuard:
    """Buffer 0 hold: a rephased suffix's rounds while its request is live."""

    request_key: message.DecoderRequestKey


@dataclasses.dataclass(frozen=True)
class FragmentAdmission:
    """Outcome of offering one round: refused when no slot is free."""

    round_identity: tuple
    received_fragments: int
    expected_fragments: int
    round_complete: bool
    refused: bool = False


@dataclasses.dataclass(frozen=True)
class SyndromeBufferSnapshot:
    """Immutable observation of buffer occupancy, states, and holds."""

    capacity: Optional[int]
    occupancy: int
    free_slot_indices: tuple
    identity_to_slot: tuple
    assembling_identities: tuple
    packing_identities: tuple
    retained_identities: tuple
    hold_counts: tuple
    tombstoned_identities: tuple


@dataclasses.dataclass(frozen=True)
class SyndromeBufferMetrics:
    """Counters over the buffer's whole life; one allocation per round."""

    allocations_total: int
    live_allocations: int
    peak_live_allocations: int


@dataclasses.dataclass
class _RoundSlot:
    identity: tuple
    slot_index: int
    packet: message.SyndromeRoundPacket


@dataclasses.dataclass(frozen=True)
class _HoldRecord:
    round_identities: tuple
    referenced_operation_ids: frozenset


class SyndromeBuffer:
    """Own upstream round allocations and their consumer holds."""

    def __init__(
        self, *, capacity: Optional[int] = None, memory_model=None
    ) -> None:
        self.capacity = capacity
        # a store that frees a slot tells its writer, so a round stalled
        # for room can be admitted; None when nobody waits on this store
        self.on_round_released = None
        self.memory_model = memory_model
        self._publication_ticks: dict = {}
        self._slots: list = []
        self._free_slot_indices: set = set()
        if capacity is not None:
            self._slots = [None] * capacity
            self._free_slot_indices = set(range(capacity))
        self._rounds: dict = {}
        self._open_operations: set = set()
        self._closed_operations: set = set()
        self._tombstones: set = set()
        self._live_holds: dict = {}
        self._released_holds: dict = {}
        self._holders_by_round: dict = {}
        self._allocations_total = 0
        self._peak_live_allocations = 0

    # ------------------------------------------------------ operation scope

    def open_operation(self, operation_id) -> None:
        """Admit rounds of this operation; a closed identity never reopens."""
        if operation_id in self._closed_operations:
            raise RuntimeError("closed operation identities cannot be reused")
        self._open_operations.add(operation_id)

    def has_operation(self, operation_id) -> bool:
        """True while the operation may still receive rounds."""
        return operation_id in self._open_operations

    def close_operation(self, operation_id) -> None:
        """Retire an operation once none of its rounds or holds are live."""
        live_rounds = []
        for identity in self._rounds:
            if message.same_stable_identity(identity[0], operation_id):
                live_rounds.append(identity)
        if live_rounds:
            ordered = sorted(live_rounds, key=message.stable_identity_order_key)
            raise RuntimeError(
                f"operation {operation_id!r} has live buffer rounds {ordered!r}"
            )
        if self.has_live_operation_reference(operation_id):
            raise RuntimeError(
                f"operation {operation_id!r} has live consumer holds"
            )
        self._open_operations.discard(operation_id)
        self._closed_operations.add(operation_id)
        self._tombstones = {
            identity
            for identity in self._tombstones
            if not message.same_stable_identity(identity[0], operation_id)
        }
        stale = []
        for holder, referenced in self._released_holds.items():
            if referenced.isdisjoint(self._open_operations):
                stale.append(holder)
        for holder in stale:
            del self._released_holds[holder]

    # -------------------------------------------------------------- intake

    def accept_packed_round(
        self,
        packet: message.SyndromeRoundPacket,
        *,
        publication_tick: Optional[int],
    ) -> FragmentAdmission:
        """Admit one complete packed round: the store's only intake.

        The controller's syndrome packing merged the fragments and formed
        the detection events; this store gives the finished round a slot
        and retains it for its consumers. If the buffer is full, the
        write is refused before anything is modified, so a refusal leaves
        no trace.
        """
        identity = (packet.operation_id, packet.round_index)
        if packet.operation_id not in self._open_operations:
            raise RuntimeError(f"operation {packet.operation_id!r} is not open")
        if identity in self._tombstones:
            raise ValueError(
                f"late write: round {identity!r} was already released"
            )
        if identity in self._rounds:
            raise ValueError(f"round {identity!r} was already written")
        if self.capacity is not None and not self._free_slot_indices:
            return FragmentAdmission(identity, 0, 1, False, refused=True)
        slot = self._take_slot(identity, packet)
        self._store_in_memory_model(slot)
        self._publication_ticks[identity] = publication_tick
        return FragmentAdmission(identity, 1, 1, True)

    def _take_slot(self, identity: tuple, packet) -> _RoundSlot:
        """Give the round the lowest free slot, or a new one when unbounded."""
        if self._free_slot_indices:
            slot_index = min(self._free_slot_indices)
            self._free_slot_indices.remove(slot_index)
        else:
            slot_index = len(self._slots)
            self._slots.append(None)
        slot = _RoundSlot(identity, slot_index, packet)
        self._slots[slot_index] = slot
        self._rounds[identity] = slot
        self._allocations_total += 1
        live = len(self._rounds)
        self._peak_live_allocations = max(self._peak_live_allocations, live)
        return slot

    def _store_in_memory_model(self, slot: _RoundSlot) -> None:
        """Offer every fragment to the memory model; a refusal undoes it."""
        if self.memory_model is None:
            return
        packet = slot.packet
        stored_keys = []
        try:
            for fragment in packet.fragments:
                key = _fragment_key(packet, fragment)
                self.memory_model.store(key, fragment)
                stored_keys.append(key)
        except BaseException:
            for key in reversed(stored_keys):
                self.memory_model.evict(key)
            self._slots[slot.slot_index] = None
            self._free_slot_indices.add(slot.slot_index)
            del self._rounds[slot.identity]
            raise

    def retained_fragments(self, round_identity) -> Optional[tuple]:
        """The retained fragments, or None before and after retention."""
        slot = self._rounds.get(round_identity)
        if slot is None:
            return None
        return slot.packet.fragments

    def mark_publication_tick(
        self, round_identity, publication_tick: int
    ) -> None:
        """Stamp a retained round when its publication reaches Buffer 0."""
        slot = self._rounds.get(round_identity)
        if slot is None:
            raise RuntimeError(f"round {round_identity!r} is not retained")
        if self._publication_ticks[round_identity] is not None:
            raise RuntimeError(
                f"round {round_identity!r} was already published"
            )
        self._publication_ticks[round_identity] = publication_tick

    def publication_tick(self, round_identity) -> Optional[int]:
        """The tick a retained round was published to the windows, or None."""
        return self._publication_ticks.get(round_identity)

    # ------------------------------------------------------- consumer holds

    def has_live_operation_reference(self, operation_id) -> bool:
        """True while any live hold refers to this operation."""
        for record in self._live_holds.values():
            if operation_id in record.referenced_operation_ids:
                return True
        return False

    def register_hold(self, holder, round_identities) -> None:
        """Keep the listed rounds alive for one consumer token."""
        self._check_new_holder(holder)
        record = self._hold_record(holder, round_identities)
        self._live_holds[holder] = record
        for identity in record.round_identities:
            self._attach_holder(identity, holder)

    def replace_hold(self, holder, round_identities) -> None:
        """Re-point a live hold at a new set of rounds."""
        old = self._live_holds.get(holder)
        if old is None:
            raise RuntimeError("consumer hold is not live")
        new = self._hold_record(holder, round_identities)
        if new == old:
            return
        self._live_holds[holder] = new
        for identity in new.round_identities:
            if identity not in old.round_identities:
                self._attach_holder(identity, holder)
        for identity in old.round_identities:
            if identity not in new.round_identities:
                self._detach_holder(identity, holder)

    def transfer_hold(self, old_holder, new_holder) -> None:
        """Move a live hold to a new token without freeing its rounds."""
        self._check_new_holder(new_holder)
        record = self._live_holds.pop(old_holder, None)
        if record is None:
            raise RuntimeError("consumer hold is not live")
        self._live_holds[new_holder] = record
        for identity in record.round_identities:
            holders = self._holders_by_round[identity]
            holders.discard(old_holder)
            holders.add(new_holder)
        self._released_holds[old_holder] = record.referenced_operation_ids

    def release_hold(self, holder) -> None:
        """Drop a hold; rounds with no remaining holder become releasable."""
        if holder in self._released_holds:
            return
        record = self._live_holds.pop(holder, None)
        if record is None:
            raise RuntimeError("consumer hold was never registered")
        for identity in record.round_identities:
            self._detach_holder(identity, holder)
        self._released_holds[holder] = record.referenced_operation_ids

    def has_hold(self, holder) -> bool:
        """True while this holder token is live."""
        return holder in self._live_holds

    def hold_round_identities(self, holder) -> tuple:
        """The rounds a live holder keeps."""
        return self._live_holds[holder].round_identities

    def _hold_record(self, holder, round_identities) -> _HoldRecord:
        unique = dict.fromkeys(round_identities)
        identities = tuple(unique)
        for identity in identities:
            self._check_hold_identity(identity)
        references = {identity[0] for identity in identities}
        if type(holder) is RephaseGuard:
            references.add(holder.request_key.operation_id)
        closed = references - self._open_operations
        if closed:
            ordered = sorted(closed, key=message.stable_identity_order_key)
            raise RuntimeError(
                f"hold {holder!r} references closed operation {ordered!r}"
            )
        referenced = frozenset(references)
        return _HoldRecord(identities, referenced)

    def _check_hold_identity(self, identity: tuple) -> None:
        if identity[0] not in self._open_operations:
            raise RuntimeError(
                f"hold references closed operation {identity[0]!r}"
            )
        if identity in self._tombstones:
            raise ValueError(f"hold references released round {identity!r}")

    def _check_new_holder(self, holder) -> None:
        if holder in self._live_holds or holder in self._released_holds:
            raise ValueError(f"duplicate consumer hold token {holder!r}")

    def _attach_holder(self, identity: tuple, holder) -> None:
        holders = self._holders_by_round.setdefault(identity, set())
        holders.add(holder)

    def _detach_holder(self, identity: tuple, holder) -> None:
        holders = self._holders_by_round.get(identity)
        if holders is None:
            return
        holders.discard(holder)
        if holders:
            return
        del self._holders_by_round[identity]
        slot = self._rounds.get(identity)
        if slot is not None:
            self._free_round(slot)

    # ------------------------------------------------- release/cancellation

    def release_round_if_unheld(self, round_identity) -> bool:
        """Free an existing round only when no consumer holds it."""
        slot = self._rounds.get(round_identity)
        if slot is None:
            return False
        if self._holders_by_round.get(round_identity):
            return False
        self._free_round(slot)
        return True

    def release_round(self, round_identity) -> None:
        """Free one unheld round."""
        slot = self._rounds.get(round_identity)
        if slot is None:
            raise RuntimeError(
                f"round {round_identity!r} holds no live allocation"
            )
        if self._holders_by_round.get(round_identity):
            raise RuntimeError(
                f"round {round_identity!r} has live consumer holds"
            )
        self._free_round(slot)

    def _free_round(self, slot: _RoundSlot) -> None:
        if self.memory_model is not None:
            packet = slot.packet
            for fragment in packet.fragments:
                key = _fragment_key(packet, fragment)
                self.memory_model.evict(key)
        self._publication_ticks.pop(slot.identity, None)
        del self._rounds[slot.identity]
        self._slots[slot.slot_index] = None
        self._free_slot_indices.add(slot.slot_index)
        self._tombstones.add(slot.identity)
        if self.on_round_released is not None:
            self.on_round_released()

    # -------------------------------------------------------- observability

    def held_rounds_description(self) -> str:
        """One compact line of the store's contents, for the I/O trace."""
        round_indices_by_operation: dict = {}
        for operation_id, round_index in self._rounds:
            indices = round_indices_by_operation.setdefault(operation_id, [])
            indices.append(round_index)
        if not round_indices_by_operation:
            return "empty"
        parts = []
        for operation_id in sorted(round_indices_by_operation, key=str):
            round_indices = round_indices_by_operation[operation_id]
            ordered = sorted(round_indices)
            ranges = _round_ranges_text(ordered)
            count = len(round_indices)
            parts.append(f"op {operation_id} rounds {ranges} ({count})")
        return "; ".join(parts)

    def snapshot(self) -> SyndromeBufferSnapshot:
        """Frozen view of every slot, hold and tombstone."""
        slots = self._rounds.values()
        ordered = sorted(slots, key=_slot_index)
        free_slot_indices = ()
        if self.capacity is not None:
            free_slot_indices = tuple(sorted(self._free_slot_indices))
        identity_to_slot = tuple(
            (slot.identity, slot.slot_index) for slot in ordered
        )
        retained_identities = tuple(slot.identity for slot in ordered)
        hold_counts = []
        for identity, holders in self._holders_by_round.items():
            hold_counts.append((identity, len(holders)))
        hold_counts.sort(key=_hold_count_order_key)
        tombstoned = sorted(
            self._tombstones, key=message.stable_identity_order_key
        )
        return SyndromeBufferSnapshot(
            capacity=self.capacity,
            occupancy=len(self._rounds),
            free_slot_indices=free_slot_indices,
            identity_to_slot=identity_to_slot,
            assembling_identities=(),
            packing_identities=(),
            retained_identities=retained_identities,
            hold_counts=tuple(hold_counts),
            tombstoned_identities=tuple(tombstoned),
        )

    def metrics(self) -> SyndromeBufferMetrics:
        """Allocation and occupancy counters."""
        return SyndromeBufferMetrics(
            allocations_total=self._allocations_total,
            live_allocations=len(self._rounds),
            peak_live_allocations=self._peak_live_allocations,
        )


def _fragment_key(packet: message.SyndromeRoundPacket, fragment) -> tuple:
    return (packet.operation_id, packet.round_index, fragment.patch_id)


def _slot_index(slot: _RoundSlot) -> int:
    return slot.slot_index


def _hold_count_order_key(item: tuple):
    identity, _count = item
    return message.stable_identity_order_key(identity)


def _round_ranges_text(sorted_round_indices: list) -> str:
    """[1, 2, 3, 7, 8] -> "1..3, 7..8"."""
    ranges = []
    range_start = sorted_round_indices[0]
    previous = range_start
    for round_index in sorted_round_indices[1:]:
        if round_index != previous + 1:
            ranges.append((range_start, previous))
            range_start = round_index
        previous = round_index
    ranges.append((range_start, previous))
    texts = [_range_text(low, high) for low, high in ranges]
    return ", ".join(texts)


def _range_text(low: int, high: int) -> str:
    if low == high:
        return f"{low}"
    return f"{low}..{high}"
