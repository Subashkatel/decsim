"""A round store: finished rounds held until their last hold releases.

Table row round_store. The controller's packing stage writes each
finished round once (accept_packed_round) after asking has_room, the
window side keeps it alive with holds (RoundHolds), and the slot is
freed when the last hold releases; on_slot_freed then tells the writer,
so a round held for room enters in order. That is gem5's cache queue
(src/mem/cache/queue.hh isFull, then allocate) and its blocked port
(src/mem/cache/base.cc clearBlocked schedules the retry): the store
never refuses a write, it answers room first. A round's status (its
packet, its publication tick) lives on its record, gem5's CacheBlk.

The store reports through trace sources (observe/trace_source.py) and
runs with no listener: round_stored(round_key, packet) when a slot is
taken, round_published(round_key, tick) when the round's data is ready
for the windows, round_released(round_key) when the slot frees;
hold_registered(holder, round_keys), hold_transferred(old_holder,
new_holder) and hold_released(holder) for the consumers' tokens.
"""

from typing import Callable, Optional

import decsim.observe.trace_source as trace_source
import decsim.records.decoding as decoding_records
import decsim.records.identity as identity_records
import decsim.records.rounds as round_records
import decsim.syndrome_buffer.round_holds as round_holds
import decsim.syndrome_buffer.settings as round_store_settings


class _StoredRound:
    """One stored round: its packet and, once published, its tick."""

    def __init__(self, packet: round_records.SyndromeRoundPacket) -> None:
        self.packet = packet
        self.publication_tick: Optional[int] = None


class RoundStore:
    """The store: rounds by key, their holds, and the operations it serves.

    Eleven attributes, six of them trace sources (the module docstring
    names them); the state is the settings, the rounds, the holds, the
    operations and the slot-freed callback.
    """

    def __init__(
        self,
        settings: round_store_settings.RoundStoreSettings,
        *,
        on_slot_freed: Optional[Callable[[], None]] = None,
    ) -> None:
        self.settings = settings
        self.round_by_key: dict = {}
        self.holds = round_holds.RoundHolds()
        # operation id -> True while it may receive rounds, False once
        # closed; a closed identity never reopens
        self.operations: dict = {}
        self.on_slot_freed = on_slot_freed
        self.round_stored = trace_source.TraceSource()
        self.round_published = trace_source.TraceSource()
        self.round_released = trace_source.TraceSource()
        self.hold_registered = trace_source.TraceSource()
        self.hold_transferred = trace_source.TraceSource()
        self.hold_released = trace_source.TraceSource()

    # ---- the port

    def has_room(self) -> bool:
        """Whether one more round fits; asked before every write."""
        capacity = self.settings.rounds
        if capacity is None:
            return True
        return len(self.round_by_key) < capacity

    def accept_packed_round(
        self,
        packet: round_records.SyndromeRoundPacket,
        *,
        publication_tick: Optional[int],
    ) -> None:
        """Keep one finished round; the writer asked has_room first."""
        assert self.has_room(), "a round was written into a full store"
        round_key = (packet.operation_id, packet.round_index)
        assert round_key not in self.round_by_key, (
            f"round {round_key!r} was written twice"
        )
        self._open(packet.operation_id)
        stored = _StoredRound(packet)
        stored.publication_tick = publication_tick
        self.round_by_key[round_key] = stored
        self.round_stored.fire(round_key, packet)
        if publication_tick is not None:
            self.round_published.fire(round_key, publication_tick)

    def release_round(self, round_key) -> None:
        """Free one unheld round; its consumers are done with it."""
        if round_key not in self.round_by_key:
            raise RuntimeError(f"round {round_key!r} is not stored")
        if self.holds.is_held(round_key):
            raise RuntimeError(f"round {round_key!r} has live consumer holds")
        self._free_round(round_key)

    # ---- reads

    @property
    def occupancy(self) -> int:
        """The rounds stored now."""
        return len(self.round_by_key)

    def retained_fragments(self, round_key) -> Optional[tuple]:
        """The stored round's fragments, or None before and after storage."""
        stored = self.round_by_key.get(round_key)
        if stored is None:
            return None
        return stored.packet.fragments

    def is_round_held(self, round_key) -> bool:
        """Whether a consumer keeps this round, stored or still expected.

        A hold names the rounds its holder reads from the moment it is
        placed, so a round with a hold and no fragments is one the store
        expects and has not received.
        """
        return self.holds.is_held(round_key)

    def publication_tick(self, round_key) -> Optional[int]:
        """The tick the round was published to the windows, or None."""
        stored = self.round_by_key.get(round_key)
        if stored is None:
            return None
        return stored.publication_tick

    def mark_publication_tick(self, round_key, publication_tick: int) -> None:
        """Stamp a stored round when its priced publication arrives."""
        stored = self.round_by_key.get(round_key)
        if stored is None:
            raise RuntimeError(f"round {round_key!r} is not stored")
        if stored.publication_tick is not None:
            raise RuntimeError(f"round {round_key!r} was already published")
        stored.publication_tick = publication_tick
        self.round_published.fire(round_key, publication_tick)

    # ---- operation scope

    def open_operation(self, operation_id) -> None:
        """Admit rounds and holds of this operation."""
        self._open(operation_id)

    def has_operation(self, operation_id) -> bool:
        """True while the operation may still receive rounds."""
        return self.operations.get(operation_id, False)

    def close_operation(self, operation_id) -> None:
        """Retire an operation once none of its rounds or holds are live."""
        live_rounds = []
        for round_key in self.round_by_key:
            if identity_records.same_stable_identity(
                round_key[0], operation_id
            ):
                live_rounds.append(round_key)
        if live_rounds:
            ordered = sorted(
                live_rounds, key=identity_records.stable_identity_order_key
            )
            raise RuntimeError(
                f"operation {operation_id!r} has live buffer rounds {ordered!r}"
            )
        if self.holds.references(operation_id):
            raise RuntimeError(
                f"operation {operation_id!r} has live consumer holds"
            )
        self.operations[operation_id] = False
        open_ids = set()
        for candidate, is_open in self.operations.items():
            if is_open:
                open_ids.add(candidate)
        self.holds.forget_released_outside(open_ids)

    def has_live_operation_reference(self, operation_id) -> bool:
        """True while any live hold refers to this operation."""
        return self.holds.references(operation_id)

    # ---- holds

    def register_hold(self, holder, round_keys) -> None:
        """Keep the listed rounds alive for one consumer token."""
        record = self._hold_record(holder, round_keys)
        self.holds.register(holder, record)
        self.hold_registered.fire(holder, record.round_keys)

    def replace_hold(self, holder, round_keys) -> None:
        """Re-point a live hold at a new set of rounds."""
        record = self._hold_record(holder, round_keys)
        orphaned = self.holds.replace(holder, record)
        self._free_stored(orphaned)

    def transfer_hold(self, old_holder, new_holder) -> None:
        """Move a live hold to a new token without freeing its rounds."""
        self.holds.transfer(old_holder, new_holder)
        self.hold_transferred.fire(old_holder, new_holder)

    def release_hold(self, holder) -> None:
        """Drop a hold; rounds with no remaining holder are freed."""
        orphaned = self.holds.release(holder)
        self.hold_released.fire(holder)
        self._free_stored(orphaned)

    def has_hold(self, holder) -> bool:
        """True while this holder token is live."""
        return self.holds.is_live(holder)

    def hold_round_identities(self, holder) -> tuple:
        """The rounds a live holder keeps."""
        return self.holds.round_keys_of(holder)

    def release_round_if_unheld(self, round_key) -> bool:
        """Free a stored round only when no consumer holds it."""
        if round_key not in self.round_by_key:
            return False
        if self.holds.is_held(round_key):
            return False
        self._free_round(round_key)
        return True

    # ---- settlement and the trace

    def check_settled(self) -> None:
        """At the end of a run nothing may still be held: a leak is a bug."""
        if self.round_by_key:
            held = list(self.round_by_key)
            raise RuntimeError(
                f"the round store still holds rounds {held} at the end"
            )
        held_keys = self.holds.held_round_keys()
        if held_keys:
            raise RuntimeError(
                f"the round store has unresolved holds on {held_keys} "
                f"(rounds expected but never written, or holders never "
                f"released)"
            )

    def held_rounds_description(self) -> str:
        """One compact line of the store's contents, for the I/O trace."""
        round_indices_by_operation: dict = {}
        for operation_id, round_index in self.round_by_key:
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

    # ---- private

    def _open(self, operation_id) -> None:
        is_open = self.operations.get(operation_id)
        if is_open is False:
            raise RuntimeError("closed operation identities cannot be reused")
        self.operations[operation_id] = True

    def _hold_record(self, holder, round_keys) -> round_holds.HoldRecord:
        """The record of a hold; a hold may name rounds not yet written."""
        unique = dict.fromkeys(round_keys)
        keys = tuple(unique)
        references = set()
        for round_key in keys:
            references.add(round_key[0])
        if type(holder) is decoding_records.RephaseGuard:
            references.add(holder.request_key.operation_id)
        for operation_id in references:
            self._open(operation_id)
        referenced = frozenset(references)
        return round_holds.HoldRecord(keys, referenced)

    def _free_stored(self, round_keys) -> None:
        for round_key in round_keys:
            if round_key in self.round_by_key:
                self._free_round(round_key)

    def _free_round(self, round_key) -> None:
        del self.round_by_key[round_key]
        self.round_released.fire(round_key)
        if self.on_slot_freed is not None:
            self.on_slot_freed()


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
