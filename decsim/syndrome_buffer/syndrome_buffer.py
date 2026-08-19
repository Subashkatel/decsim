"""Store syndrome rounds until every decoder request is finished with them.

A round uses one buffer slot. Fragments fill that slot, packing marks the round
ready, and consumer holds keep it alive. The slot is freed after its last hold
is released.

This module owns data lifetime only. It does not schedule events, model links,
or manage decoder queues.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Optional

from ..message import (
    RetainedSyndromeFragment,
    SyndromeRoundPacket,
    same_stable_identity,
    stable_identity_order_key,
)


@dataclass(frozen=True)
class SyndromeBufferingConfig:
    """Optional capacity of the upstream syndrome buffer, in rounds.

    ``None`` means unbounded. Decoder-side storage is configured separately and
    in different units by ``RunSpec.decoder_memory``.
    """

    upstream_packet_slots: Optional[int] = None


class SyndromeBufferRoundState(Enum):
    """Current state of one upstream round allocation."""

    ASSEMBLING = auto()
    PACKING = auto()
    PACKED_RETAINED = auto()


# ---- consumer hold tokens: who keeps rounds in Buffer 0 and why

@dataclass(frozen=True)
class PotentialStrong:
    """Buffer 0 hold: a window's rounds kept in case its weak result escalates."""

    window_key: tuple


@dataclass(frozen=True)
class PendingStrong:
    """Buffer 0 hold: rounds kept for a strong request that is admitted but not yet served."""

    request_key: DecoderRequestKey


@dataclass(frozen=True)
class CsdInput:
    """Buffer 0 hold: rounds in flight to a strong decoder over CSD."""

    request_key: DecoderRequestKey


@dataclass(frozen=True)
class DecoderInputHold:
    """Buffer 0 hold: rounds a decode job needs until they land in unit memory."""

    request_key: DecoderRequestKey


@dataclass(frozen=True)
class Replay:
    """Buffer 0 hold: rounds a window needs again for a speculative replay of one boundary generation."""

    window_key: tuple
    boundary_generation: int


@dataclass(frozen=True)
class RephaseGuard:
    """Buffer 0 hold: rounds a rephased suffix keeps while its strong request is live."""

    request_key: DecoderRequestKey


@dataclass(frozen=True)
class FragmentAdmission:
    """Outcome of offering one fragment: refused when the buffer has no free
    slot for a new round, otherwise how far the round's assembly has come."""

    round_identity: tuple
    received_fragments: int
    expected_fragments: int
    round_complete: bool
    refused: bool = False


@dataclass(frozen=True)
class SyndromeBufferSnapshot:
    """Immutable observation of buffer occupancy, states, and holds."""

    capacity: Optional[int]
    occupancy: int
    free_slot_indices: tuple[int, ...]
    identity_to_slot: tuple[tuple[tuple, int], ...]
    assembling_identities: tuple[tuple, ...]
    packing_identities: tuple[tuple, ...]
    retained_identities: tuple[tuple, ...]
    hold_counts: tuple[tuple[tuple, int], ...]
    tombstoned_identities: tuple[tuple, ...]


@dataclass(frozen=True)
class SyndromeBufferMetrics:
    """Counters over the buffer's whole life; one allocation per round."""

    allocations_total: int
    live_allocations: int
    peak_live_allocations: int
    released_rounds: int
    fragments_accepted: int


@dataclass
class _RoundSlot:
    identity: tuple
    expected_fragments: int
    slot_index: int
    state: SyndromeBufferRoundState = SyndromeBufferRoundState.ASSEMBLING
    fragments: list = field(default_factory=list)
    packet: Optional[SyndromeRoundPacket] = None
    published_tick: Optional[int] = None


@dataclass(frozen=True)
class _HoldRecord:
    round_identities: tuple
    referenced_operation_ids: frozenset


def _merge_fragments_by_patch(fragments) -> tuple:
    """Order fragments by index, merging parts from the same patch.

    ``SyndromeRoundPacket`` requires distinct patch identities, so parts of
    one patch concatenate bits and sizes in fragment-index order. Distinct
    patches keep their own immutable fragments untouched.
    """
    merged: list = []
    for fragment in sorted(fragments, key=lambda item: item.fragment_index):
        prior_index = next(
            (
                index
                for index, prior in enumerate(merged)
                if same_stable_identity(prior.patch_id, fragment.patch_id)
            ),
            None,
        )
        if prior_index is None:
            merged.append(fragment)
            continue
        prior = merged[prior_index]
        bits = (
            prior.bits + fragment.bits
            if prior.bits is not None and fragment.bits is not None
            else None
        )
        size_bits = (
            prior.size_bits + fragment.size_bits
            if prior.size_bits is not None and fragment.size_bits is not None
            else None
        )
        merged[prior_index] = replace(prior, bits=bits, size_bits=size_bits)
    return tuple(merged)


class SyndromeBuffer:
    """Own upstream round allocations and their consumer holds."""

    def __init__(
        self, *, capacity: Optional[int] = None, memory_model=None,
    ) -> None:
        self.capacity = capacity
        self.memory_model = memory_model
        self.payloads_held = 0
        self.peak_payloads = 0
        self._publication_ticks: dict[tuple, Optional[int]] = {}
        self._slots: list[Optional[_RoundSlot]] = (
            [None] * capacity if capacity is not None else []
        )
        self._free_slot_indices: set[int] = (
            set(range(capacity)) if capacity is not None else set()
        )
        self._rounds: dict[tuple, _RoundSlot] = {}
        self._open_operations: set = set()
        self._closed_operations: set = set()
        self._tombstones: set[tuple] = set()
        self._orphan_on_publish: set[tuple] = set()
        self._live_holds: dict = {}
        self._released_holds: dict = {}
        self._holders_by_round: dict[tuple, set] = {}
        self._allocations_total = 0
        self._peak_live_allocations = 0
        self._released_rounds = 0
        self._fragments_accepted = 0

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
        live_rounds = [
            identity
            for identity in self._rounds
            if same_stable_identity(identity[0], operation_id)
        ]
        if live_rounds:
            ordered = sorted(live_rounds, key=stable_identity_order_key)
            raise RuntimeError(
                f"operation {operation_id!r} has live buffer rounds {ordered!r}"
            )
        live_holders = [
            holder
            for holder, record in self._live_holds.items()
            if operation_id in record.referenced_operation_ids
        ]
        if live_holders:
            raise RuntimeError(
                f"operation {operation_id!r} has live consumer holds"
            )
        self._open_operations.discard(operation_id)
        self._closed_operations.add(operation_id)
        self._tombstones = {
            identity
            for identity in self._tombstones
            if not same_stable_identity(identity[0], operation_id)
        }
        stale = [
            holder
            for holder, referenced in self._released_holds.items()
            if referenced.isdisjoint(self._open_operations)
        ]
        for holder in stale:
            del self._released_holds[holder]

    # ---------------------------------------------------- fragment assembly

    def accept_fragment(
        self,
        fragment: RetainedSyndromeFragment,
        *,
        expected_fragments: int,
    ) -> FragmentAdmission:
        """Add one fragment. The first fragment allocates the round slot; a
        full buffer refuses the first fragment of a new round."""
        if fragment.operation_id not in self._open_operations:
            raise RuntimeError(
                f"operation {fragment.operation_id!r} is not open"
            )
        identity = (fragment.operation_id, fragment.round_index)
        if identity in self._tombstones:
            raise ValueError(
                f"late fragment: round {identity!r} was already released"
            )
        slot = self._rounds.get(identity)
        if slot is None:
            if self.capacity is not None and not self._free_slot_indices:
                return FragmentAdmission(identity, 0, expected_fragments, False, refused=True)
            slot = self._allocate(identity, expected_fragments)
        elif slot.state is not SyndromeBufferRoundState.ASSEMBLING:
            raise ValueError(
                f"fragment admission for round {identity!r} was retired at "
                f"packing"
            )
        if expected_fragments != slot.expected_fragments:
            raise ValueError("all fragments must declare the same count")
        if fragment.fragment_index >= slot.expected_fragments:
            raise ValueError("fragment index exceeds the declared count")
        if any(
            fragment.fragment_index == held.fragment_index
            for held in slot.fragments
        ):
            raise ValueError("duplicate syndrome fragment index")
        slot.fragments.append(fragment)
        self._fragments_accepted += 1
        if len(slot.fragments) == slot.expected_fragments:
            slot.state = SyndromeBufferRoundState.PACKING
        return FragmentAdmission(
            round_identity=identity,
            received_fragments=len(slot.fragments),
            expected_fragments=slot.expected_fragments,
            round_complete=slot.state is SyndromeBufferRoundState.PACKING,
        )

    def _allocate(self, identity, expected_fragments: int) -> _RoundSlot:
        if self._free_slot_indices:
            slot_index = min(self._free_slot_indices)
            self._free_slot_indices.remove(slot_index)
        else:
            slot_index = len(self._slots)
            self._slots.append(None)
        slot = _RoundSlot(identity, expected_fragments, slot_index)
        self._slots[slot_index] = slot
        self._rounds[identity] = slot
        self._allocations_total += 1
        self._peak_live_allocations = max(
            self._peak_live_allocations, len(self._rounds)
        )
        return slot

    # -------------------------------------------------------------- packing

    def finish_packing(
        self, round_identity, *, publication_tick: Optional[int] = None,
    ) -> SyndromeRoundPacket:
        """Finish packing and make the retained round readable."""
        slot = self._rounds.get(round_identity)
        if slot is None:
            raise RuntimeError(
                f"round {round_identity!r} holds no live allocation"
            )
        if slot.state is not SyndromeBufferRoundState.PACKING:
            raise RuntimeError(
                f"round {round_identity!r} is {slot.state.name}, not PACKING"
            )
        packet = SyndromeRoundPacket(
            operation_id=slot.identity[0],
            round_index=slot.identity[1],
            fragments=_merge_fragments_by_patch(slot.fragments),
        )
        stored_keys = []
        try:
            if self.memory_model is not None:
                for fragment in packet.fragments:
                    key = (
                        packet.operation_id, packet.round_index,
                        fragment.patch_id,
                    )
                    self.memory_model.store(key, fragment)
                    stored_keys.append(key)
        except BaseException:
            for key in reversed(stored_keys):
                self.memory_model.evict(key)
            raise
        slot.packet = packet
        slot.fragments = []
        slot.state = SyndromeBufferRoundState.PACKED_RETAINED
        self._publication_ticks[slot.identity] = publication_tick
        self.payloads_held += len(packet.fragments)
        self.peak_payloads = max(self.peak_payloads, self.payloads_held)
        if slot.identity in self._orphan_on_publish:
            self._orphan_on_publish.remove(slot.identity)
            self._free_round(slot)
        return packet

    def read_retained_round(self, round_identity) -> SyndromeRoundPacket:
        """The packed packet of a retained round."""
        slot = self._rounds.get(round_identity)
        if slot is None or slot.state is not (
            SyndromeBufferRoundState.PACKED_RETAINED
        ):
            raise RuntimeError(
                f"round {round_identity!r} is not packed and retained"
            )
        return slot.packet

    def retained_fragments(self, round_identity) -> Optional[tuple]:
        """Return retained fragments, or ``None`` before/after retention."""
        slot = self._rounds.get(round_identity)
        if slot is None or slot.state is not SyndromeBufferRoundState.PACKED_RETAINED:
            return None
        return slot.packet.fragments

    def mark_publication_tick(self, round_identity, publication_tick: int) -> None:
        """Stamp a retained round when its priced publication reaches Buffer 0."""
        identity = round_identity
        slot = self._rounds.get(identity)
        if slot is None or slot.state is not SyndromeBufferRoundState.PACKED_RETAINED:
            raise RuntimeError(f"round {identity!r} is not packed and retained")
        if self._publication_ticks[identity] is not None:
            raise RuntimeError(f"round {identity!r} was already published")
        self._publication_ticks[identity] = publication_tick

    def publication_tick(self, round_identity) -> Optional[int]:
        """Tick a retained round was published to the window manager, or None."""
        return self._publication_ticks.get(round_identity)

    def round_state(self, round_identity) -> Optional[SyndromeBufferRoundState]:
        """Slot state of a round, or None when it holds no live allocation."""
        slot = self._rounds.get(round_identity)
        return None if slot is None else slot.state

    # ------------------------------------------------------- consumer holds

    def _hold_record(self, holder, round_identities) -> _HoldRecord:
        identities = tuple(dict.fromkeys(round_identities))
        for identity in identities:
            if identity[0] not in self._open_operations:
                raise RuntimeError(
                    f"hold references closed operation {identity[0]!r}"
                )
            if identity in self._tombstones:
                raise ValueError(
                    f"hold references released round {identity!r}"
                )
        references = {identity[0] for identity in identities}
        if type(holder) is Replay:
            references.add(holder.window_key[0])
        elif type(holder) is RephaseGuard:
            references.add(holder.request_key.operation_id)
        closed = references - self._open_operations
        if closed:
            ordered = sorted(closed, key=stable_identity_order_key)
            raise RuntimeError(
                f"hold {holder!r} references closed operation {ordered!r}"
            )
        return _HoldRecord(identities, frozenset(references))

    def has_live_operation_reference(self, operation_id) -> bool:
        """True while any live hold refers to this operation."""
        return any(
            operation_id in record.referenced_operation_ids
            for record in self._live_holds.values()
        )

    def _validate_new_holder(self, holder) -> None:
        if holder is None:
            raise TypeError("consumer hold tokens cannot be None")
        if holder in self._live_holds or holder in self._released_holds:
            raise ValueError(f"duplicate consumer hold token {holder!r}")

    def register_hold(self, holder, round_identities) -> None:
        """Keep the listed rounds alive for one consumer token."""
        self._validate_new_holder(holder)
        record = self._hold_record(holder, round_identities)
        self._live_holds[holder] = record
        for identity in record.round_identities:
            self._holders_by_round.setdefault(identity, set()).add(holder)

    def replace_hold(self, holder, round_identities) -> None:
        """Re-point a live hold at a new set of rounds."""
        try:
            old = self._live_holds[holder]
        except KeyError as error:
            raise RuntimeError("consumer hold is not live") from error
        new = self._hold_record(holder, round_identities)
        if new == old:
            return
        self._live_holds[holder] = new
        for identity in new.round_identities:
            if identity not in old.round_identities:
                self._holders_by_round.setdefault(identity, set()).add(holder)
        for identity in old.round_identities:
            if identity not in new.round_identities:
                self._detach_holder(identity, holder)

    def transfer_hold(self, old_holder, new_holder) -> None:
        """Move a live hold to a new token without freeing its rounds."""
        self._validate_new_holder(new_holder)
        try:
            record = self._live_holds.pop(old_holder)
        except KeyError as error:
            raise RuntimeError("consumer hold is not live") from error
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
        try:
            record = self._live_holds.pop(holder)
        except KeyError as error:
            raise RuntimeError(
                "consumer hold was never registered"
            ) from error
        for identity in record.round_identities:
            self._detach_holder(identity, holder)
        self._released_holds[holder] = record.referenced_operation_ids

    def _detach_holder(self, identity, holder) -> None:
        holders = self._holders_by_round.get(identity)
        if holders is None:
            return
        holders.discard(holder)
        if holders:
            return
        del self._holders_by_round[identity]
        slot = self._rounds.get(identity)
        if slot is None:
            return
        if slot.state is SyndromeBufferRoundState.PACKED_RETAINED:
            self._free_round(slot)
        else:
            self._orphan_on_publish.add(identity)

    def has_hold(self, holder) -> bool:
        """True while this holder token is live."""
        return holder in self._live_holds

    def hold_round_identities(self, holder) -> tuple:
        """The rounds a live holder keeps."""
        return self._live_holds[holder].round_identities

    # ------------------------------------------------- release/cancellation

    def release_round_if_unheld(self, round_identity) -> bool:
        """Free an existing round only when no consumer holds it."""
        identity = round_identity
        slot = self._rounds.get(identity)
        if slot is None or self._holders_by_round.get(identity):
            return False
        self._free_round(slot)
        return True

    def release_round(self, round_identity) -> None:
        """Free one unheld round, including a partially assembled round."""
        identity = round_identity
        slot = self._rounds.get(identity)
        if slot is None:
            raise RuntimeError(
                f"round {round_identity!r} holds no live allocation"
            )
        if self._holders_by_round.get(identity):
            raise RuntimeError(
                f"round {round_identity!r} has live consumer holds"
            )
        self._free_round(slot)

    def _free_round(self, slot: _RoundSlot) -> None:
        if slot.packet is not None:
            self.payloads_held -= len(slot.packet.fragments)
            if self.memory_model is not None:
                for fragment in slot.packet.fragments:
                    self.memory_model.evict((
                        slot.packet.operation_id, slot.packet.round_index,
                        fragment.patch_id,
                    ))
        self._publication_ticks.pop(slot.identity, None)
        del self._rounds[slot.identity]
        self._slots[slot.slot_index] = None
        self._free_slot_indices.add(slot.slot_index)
        self._tombstones.add(slot.identity)
        self._orphan_on_publish.discard(slot.identity)
        self._released_rounds += 1

    # -------------------------------------------------------- observability

    def snapshot(self) -> SyndromeBufferSnapshot:
        """Frozen view of every slot, hold and tombstone."""
        ordered = sorted(
            self._rounds.values(), key=lambda slot: slot.slot_index
        )
        by_state = {
            state: tuple(
                slot.identity for slot in ordered if slot.state is state
            )
            for state in SyndromeBufferRoundState
        }
        return SyndromeBufferSnapshot(
            capacity=self.capacity,
            occupancy=len(self._rounds),
            free_slot_indices=(
                tuple(sorted(self._free_slot_indices))
                if self.capacity is not None
                else ()
            ),
            identity_to_slot=tuple(
                (slot.identity, slot.slot_index) for slot in ordered
            ),
            assembling_identities=by_state[
                SyndromeBufferRoundState.ASSEMBLING
            ],
            packing_identities=by_state[SyndromeBufferRoundState.PACKING],
            retained_identities=by_state[
                SyndromeBufferRoundState.PACKED_RETAINED
            ],
            hold_counts=tuple(
                sorted(
                    (
                        (identity, len(holders))
                        for identity, holders in self._holders_by_round.items()
                    ),
                    key=lambda item: stable_identity_order_key(item[0]),
                )
            ),
            tombstoned_identities=tuple(
                sorted(self._tombstones, key=stable_identity_order_key)
            ),
        )

    def metrics(self) -> SyndromeBufferMetrics:
        """Allocation and occupancy counters."""
        return SyndromeBufferMetrics(
            allocations_total=self._allocations_total,
            live_allocations=len(self._rounds),
            peak_live_allocations=self._peak_live_allocations,
            released_rounds=self._released_rounds,
            fragments_accepted=self._fragments_accepted,
        )
