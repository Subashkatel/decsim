"""Bounded SB0/SB1 storage over one immutable syndrome-packet backing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from .message import (
    EndpointRole,
    EndpointState,
    Replay,
    RephaseGuard,
    SyndromeRoundPacket,
    stable_identity_order_key,
)
from .protocols import EndpointCapacityChangeReceiver


@dataclass(frozen=True)
class SyndromeBufferingConfig:
    """Aggregate packet capacities; fragment and bit volume stay separate."""
    controller_ingress_packet_slots: Optional[int] = None
    sb0_packet_slots: Optional[int] = None
    sb1_packet_slots: Optional[int] = None

    def __post_init__(self) -> None:
        capacities = (self.controller_ingress_packet_slots,
                      self.sb0_packet_slots, self.sb1_packet_slots)
        if any(capacity is None for capacity in capacities):
            if capacities != (None, None, None):
                raise ValueError(
                    "syndrome buffering capacities are all finite or all unbounded")
            return
        if any(type(capacity) is not int or capacity < 1 for capacity in capacities):
            raise TypeError("syndrome buffering capacities must be positive ints")


@dataclass(frozen=True)
class EndpointSnapshot:
    capacity: Optional[int]
    free_slot_indices: tuple[int, ...]
    prepared_identities: tuple[tuple, ...]
    resident_identities: tuple[tuple, ...]
    identity_to_slot: tuple[tuple[tuple, int], ...]
    states_by_identity: tuple[tuple[tuple, EndpointState], ...]


class EndpointCapacityExhaustion(RuntimeError):
    """A valid owner or packed window input cannot acquire endpoint slots."""

    status = "endpoint_capacity_exhaustion"

    def __init__(
        self, *, phase, roles, packet_or_owner_identity, required_slots,
        free_slots, endpoint_snapshots, controller_staging_snapshot=None,
    ) -> None:
        if phase not in ("owner_acquisition", "window_input_admission"):
            raise ValueError("invalid endpoint-capacity exhaustion phase")
        self.phase = phase
        self.endpoint_roles = tuple(roles)
        self.packet_or_owner_identity = packet_or_owner_identity
        self.required_slots = tuple(required_slots)
        self.free_slots = tuple(free_slots)
        self.endpoint_snapshots = tuple(endpoint_snapshots)
        self.controller_staging_snapshot = controller_staging_snapshot
        role_names = ", ".join(role.name for role in self.endpoint_roles)
        super().__init__(f"{role_names} endpoint capacity exhausted during {phase}")


@dataclass(frozen=True)
class _OwnerRecord:
    packet_identities: tuple
    referenced_operation_ids: frozenset


class _PairState(Enum):
    PREPARED = auto()
    COMMITTED_UNPUBLISHED = auto()
    PUBLISHED = auto()
    CANCELLED = auto()


class _TransportOwner(Enum):
    CRYO_IN_FLIGHT = auto()


class _EndpointLedger:
    def __init__(self, role, capacity):
        self.role = role
        self.capacity = capacity
        self.slots = [] if capacity is None else [None] * capacity
        self.free = set() if capacity is None else set(range(capacity))
        self.indices = {}
        self.states = {}
        self.owners = {}

    def reserve(self, identity):
        if identity in self.indices:
            raise ValueError(f"duplicate endpoint identity {identity!r}")
        if self.capacity is not None and not self.free:
            return None
        index = min(self.free) if self.free else len(self.slots)
        if self.free:
            self.free.remove(index)
        else:
            self.slots.append(None)
        self.slots[index] = identity
        self.indices[identity] = index
        self.states[identity] = EndpointState.PREPARED
        return index

    def publish_transport(self, identity, owners):
        if self.states.get(identity) is not EndpointState.PREPARED:
            raise RuntimeError("endpoint identity is not prepared")
        self.states[identity] = (
            EndpointState.RESIDENT
            if self.role is EndpointRole.SB0
            else EndpointState.CRYO_IN_FLIGHT
        )
        self.owners[identity] = set(owners)

    def activate_resident(self, identity, owners):
        if self.states.get(identity) is not EndpointState.PREPARED:
            raise RuntimeError("endpoint identity is not prepared")
        self.states[identity] = EndpointState.RESIDENT
        self.owners[identity] = set(owners)

    def remove(self, identity):
        index = self.indices.pop(identity, None)
        if index is None:
            return False
        self.states.pop(identity, None)
        self.owners.pop(identity, None)
        self.slots[index] = None
        self.free.add(index)
        return True

    def release(self, identity, owner):
        owners = self.owners.get(identity)
        if owners is None or owner not in owners:
            raise RuntimeError("endpoint owner is not live")
        owners.remove(owner)
        return not owners and self.remove(identity)

    def snapshot(self):
        by_slot = tuple(sorted(self.indices.items(), key=lambda item: item[1]))
        prepared = tuple(identity for identity, _ in by_slot
                         if self.states[identity] is EndpointState.PREPARED)
        resident = tuple(identity for identity, _ in by_slot
                         if self.states[identity] is EndpointState.RESIDENT)
        states = tuple((identity, self.states[identity]) for identity, _ in by_slot)
        free = tuple(sorted(self.free)) if self.capacity is not None else ()
        return EndpointSnapshot(
            self.capacity, free, prepared, resident, by_slot, states)


class _PairReservation:
    def __init__(self, store, identity, packet, completion_tick):
        self.store = store
        self.identity = identity
        self.packet = packet
        self.completion_tick = completion_tick
        self.state = _PairState.PREPARED

    def commit_unpublished(self):
        if self.state is not _PairState.PREPARED:
            raise RuntimeError("pair is not prepared")
        self.state = _PairState.COMMITTED_UNPUBLISHED

    def set_completion_tick(self, completion_tick):
        if self.state is not _PairState.PREPARED:
            raise RuntimeError("pair completion changes only before commit")
        self.completion_tick = completion_tick

    def publish(self):
        if self.state is not _PairState.COMMITTED_UNPUBLISHED:
            raise RuntimeError("pair is not committed")
        try:
            self.store._publish_pair(self)
        except BaseException:
            self.cancel()
            raise
        self.state = _PairState.PUBLISHED

    def cancel(self):
        if self.state is _PairState.PUBLISHED:
            raise RuntimeError("published pair cannot be cancelled")
        if self.state is not _PairState.CANCELLED:
            self.store._cancel_pair(self)
            self.state = _PairState.CANCELLED


class PayloadStore:
    """Own packet backing, bounded endpoint slots, and typed owner lifetime."""

    def __init__(self, memory_model=None, *, sb0_capacity=None,
                 sb1_capacity=None) -> None:
        self.memory_model = memory_model
        self.payloads_held = 0
        self.peak_payloads = 0
        self._ledgers = {
            EndpointRole.SB0: _EndpointLedger(EndpointRole.SB0, sb0_capacity),
            EndpointRole.SB1: _EndpointLedger(EndpointRole.SB1, sb1_capacity),
        }
        self._future_owners = {}
        self._released_owners = {}
        self._backing = {}
        self._pairs = set()
        self._open_operations = set()
        self._cryo_owner = _TransportOwner.CRYO_IN_FLIGHT
        self._capacity_change_receiver = None

    def register_op(self, operation_id) -> None:
        self._open_operations.add(operation_id)

    def has_op(self, operation_id) -> bool:
        return operation_id in self._open_operations

    def _owner_record(self, owner, packet_identities) -> _OwnerRecord:
        identities = tuple(dict.fromkeys(packet_identities))
        references = {identity[0] for identity in identities}
        if type(owner) is Replay:
            references.add(owner.window_key[0])
        elif type(owner) is RephaseGuard:
            references.add(owner.request_key.operation_id)
        return _OwnerRecord(identities, frozenset(references))

    def _validate_new_token(self, role, owner) -> None:
        if type(role) is not EndpointRole:
            raise TypeError("endpoint role must be EndpointRole")
        token = (role, owner)
        if token in self._future_owners or token in self._released_owners:
            raise ValueError("duplicate endpoint owner")
        if type(owner) is Replay and (
            type(owner.boundary_generation) is not int
            or owner.boundary_generation < 0
        ):
            raise TypeError("Replay generation must be a nonnegative int")

    def _require_referenced_operations_open(self, owner, record) -> None:
        closed = record.referenced_operation_ids - self._open_operations
        if closed:
            ordered = sorted(closed, key=stable_identity_order_key)
            raise RuntimeError(
                f"endpoint owner {owner!r} references closed operation {ordered!r}"
            )

    def _register_records(self, records) -> None:
        reservations = {role: [] for role in EndpointRole}
        for role, _, record in records:
            other = EndpointRole.SB1 if role is EndpointRole.SB0 else EndpointRole.SB0
            for identity in record.packet_identities:
                if identity in self._backing and identity not in self._ledgers[role].indices:
                    if identity not in self._ledgers[other].indices:
                        raise RuntimeError("backing has no live endpoint role")
                    reservations[role].append(identity)
        failures = [
            (role, len(identities), len(self._ledgers[role].free))
            for role, identities in reservations.items()
            if self._ledgers[role].capacity is not None
            and len(identities) > len(self._ledgers[role].free)
        ]
        if failures:
            raise EndpointCapacityExhaustion(
                phase="owner_acquisition", roles=tuple(row[0] for row in failures),
                packet_or_owner_identity=records[0][1],
                required_slots=tuple(row[1] for row in failures),
                free_slots=tuple(row[2] for row in failures),
                endpoint_snapshots=tuple(
                    (role, self.endpoint_snapshot(role)) for role in EndpointRole),
            )
        reserved = []
        attached = []
        try:
            for role in EndpointRole:
                for identity in reservations[role]:
                    if self._ledgers[role].reserve(identity) is None:
                        raise RuntimeError("preflighted endpoint reservation failed")
                    reserved.append((role, identity))
            for role, owner, record in records:
                ledger = self._ledgers[role]
                for identity in record.packet_identities:
                    owners = ledger.owners.get(identity)
                    if owners is not None:
                        owners.add(owner)
                        attached.append((role, identity, owner))
                for identity in reservations[role]:
                    ledger.activate_resident(identity, (owner,))
                self._future_owners[(role, owner)] = record
        except BaseException:
            for role, identity, owner in reversed(attached):
                self._ledgers[role].owners[identity].discard(owner)
            for role, identity in reversed(reserved):
                self._ledgers[role].remove(identity)
            for role, owner, _ in records:
                self._future_owners.pop((role, owner), None)
            raise

    def register_owner(self, role, owner, packet_identities) -> None:
        self._validate_new_token(role, owner)
        record = self._owner_record(owner, packet_identities)
        self._require_referenced_operations_open(owner, record)
        self._register_records(((role, owner, record),))

    def register_rephase_guard(self, guard, sb0_ids, sb1_ids) -> None:
        if type(guard) is not RephaseGuard:
            raise TypeError("rephase guard must be RephaseGuard")
        rows = []
        for role, identities in (
            (EndpointRole.SB0, tuple(sb0_ids)),
            (EndpointRole.SB1, tuple(sb1_ids)),
        ):
            if identities:
                self._validate_new_token(role, guard)
                rows.append((role, guard, self._owner_record(guard, identities)))
        if not rows:
            raise ValueError("rephase guard cannot be empty")
        combined_references = frozenset().union(
            *(record.referenced_operation_ids for _, _, record in rows)
        )
        self._require_referenced_operations_open(
            guard, _OwnerRecord((), combined_references))
        self._register_records(tuple(rows))

    def replace_owner_membership(self, role, owner, packet_identities) -> None:
        token = (role, owner)
        try:
            old = self._future_owners[token]
        except KeyError as error:
            raise RuntimeError("endpoint owner is not live") from error
        new = self._owner_record(owner, packet_identities)
        self._require_referenced_operations_open(owner, new)
        if new == old:
            return
        additions = tuple(identity for identity in new.packet_identities
                          if identity not in old.packet_identities)
        self._future_owners.pop(token)
        try:
            self._register_records(((role, owner,
                                     self._owner_record(owner, additions)),))
        except BaseException:
            self._future_owners[token] = old
            raise
        self._future_owners[token] = new
        ledger = self._ledgers[role]
        for identity in old.packet_identities:
            if identity not in new.packet_identities and identity in ledger.owners:
                if ledger.release(identity, owner):
                    self._drop_backing_if_unowned(identity)
                    self._notify_capacity_change()

    def owner_packet_identities(self, role, owner) -> tuple:
        try:
            return self._future_owners[(role, owner)].packet_identities
        except KeyError as error:
            raise KeyError((role, owner)) from error

    def has_owner(self, role, owner) -> bool:
        return (role, owner) in self._future_owners

    def has_live_operation_reference(self, operation_id) -> bool:
        return any(operation_id in record.referenced_operation_ids
                   for record in self._future_owners.values())

    def release_owner(self, role, owner) -> None:
        token = (role, owner)
        if token in self._released_owners:
            return
        try:
            record = self._future_owners.pop(token)
        except KeyError as error:
            raise RuntimeError("endpoint owner was never registered") from error
        ledger = self._ledgers[role]
        for identity in record.packet_identities:
            if identity in ledger.owners and ledger.release(identity, owner):
                self._drop_backing_if_unowned(identity)
                self._notify_capacity_change()
        self._released_owners[token] = record

    def close_operation(self, operation_id) -> None:
        for token, record in self._future_owners.items():
            if operation_id in record.referenced_operation_ids:
                raise RuntimeError(
                    f"operation {operation_id!r} has live endpoint owner {token!r}"
                )
        if any(pair.identity[0] == operation_id for pair in self._pairs):
            raise RuntimeError(f"operation {operation_id!r} has a prepared pair")
        if any(identity[0] == operation_id for identity in self._backing):
            raise RuntimeError(f"operation {operation_id!r} has live endpoint backing")
        self._open_operations.discard(operation_id)
        stale = [token for token, record in self._released_owners.items()
                 if record.referenced_operation_ids.isdisjoint(
                     self._open_operations)]
        for token in stale:
            del self._released_owners[token]

    def endpoint_capacity(self, role) -> Optional[int]:
        return self._ledgers[role].capacity

    def endpoint_snapshot(self, role):
        return self._ledgers[role].snapshot()

    @property
    def backing_identities(self):
        return tuple(sorted(self._backing, key=stable_identity_order_key))

    def connect_capacity_change_receiver(
        self, receiver: EndpointCapacityChangeReceiver) -> None:
        if self._capacity_change_receiver is not None:
            raise RuntimeError("capacity change receiver is already connected")
        self._capacity_change_receiver = receiver

    def _notify_capacity_change(self) -> None:
        if self._capacity_change_receiver is not None:
            self._capacity_change_receiver.on_endpoint_capacity_changed()

    def prepare_pair(self, packet, completion_tick):
        if type(packet) is not SyndromeRoundPacket:
            raise TypeError("prepare_pair requires a SyndromeRoundPacket")
        if type(completion_tick) is not int or completion_tick < 0:
            raise TypeError("completion_tick must be a nonnegative int")
        identity = (packet.operation_id, packet.round_index)
        if packet.operation_id not in self._open_operations:
            raise RuntimeError("packet operation storage is not open")
        if identity in self._backing:
            raise ValueError("duplicate packet backing identity")
        reserved = []
        for role in EndpointRole:
            if self._ledgers[role].reserve(identity) is None:
                for prior_role in reserved:
                    self._ledgers[prior_role].remove(identity)
                return None
            reserved.append(role)
        pair = _PairReservation(self, identity, packet, completion_tick)
        self._pairs.add(pair)
        return pair

    def _publish_pair(self, pair) -> None:
        identity = pair.identity
        try:
            sb0_owners = self._owners_for(EndpointRole.SB0, identity)
            sb1_owners = self._owners_for(EndpointRole.SB1, identity)
            self._ledgers[EndpointRole.SB0].publish_transport(identity, sb0_owners)
            self._after_sb0_activation(pair)
            self._ledgers[EndpointRole.SB1].publish_transport(
                identity, sb1_owners + (self._cryo_owner,))
            self._after_sb1_activation(pair)
            self._store_fragments(pair.packet)
            self._backing[identity] = (pair.packet, pair.completion_tick)
            if not sb0_owners:
                self._ledgers[EndpointRole.SB0].remove(identity)
            self._pairs.remove(pair)
        except BaseException:
            for ledger in self._ledgers.values():
                ledger.remove(identity)
            if identity in self._backing:
                self._discard_backing(identity)
            raise

    def _after_sb0_activation(self, pair) -> None:
        return None

    def _after_sb1_activation(self, pair) -> None:
        return None

    def _owners_for(self, role, identity):
        return tuple(owner for (candidate_role, owner), record
                     in self._future_owners.items()
                     if candidate_role is role
                     and identity in record.packet_identities)

    def _cancel_pair(self, pair) -> None:
        for ledger in self._ledgers.values():
            ledger.remove(pair.identity)
        self._pairs.discard(pair)

    def complete_cryo(self, identity) -> None:
        ledger = self._ledgers[EndpointRole.SB1]
        if ledger.states.get(identity) is not EndpointState.CRYO_IN_FLIGHT:
            raise RuntimeError("SB1 identity is not CRYO in flight")
        ledger.states[identity] = EndpointState.RESIDENT
        if ledger.release(identity, self._cryo_owner):
            self._drop_backing_if_unowned(identity)
            self._notify_capacity_change()

    def fragments(self, operation_id, round_index: int) -> Optional[tuple]:
        backing = self._backing.get((operation_id, round_index))
        return None if backing is None else backing[0].fragments

    def round_complete_tick(self, operation_id, round_index: int) -> Optional[int]:
        backing = self._backing.get((operation_id, round_index))
        return None if backing is None else backing[1]

    def _store_fragments(self, packet) -> None:
        stored_keys = []
        try:
            if self.memory_model is not None:
                for fragment in packet.fragments:
                    key = (packet.operation_id, packet.round_index, fragment.patch_id)
                    self.memory_model.store(key, fragment)
                    stored_keys.append(key)
        except BaseException:
            for key in reversed(stored_keys):
                self.memory_model.evict(key)
            raise
        self.payloads_held += len(packet.fragments)
        self.peak_payloads = max(self.peak_payloads, self.payloads_held)

    def _drop_backing_if_unowned(self, identity) -> None:
        if any(identity in ledger.indices for ledger in self._ledgers.values()):
            return
        self._discard_backing(identity)

    def _discard_backing(self, identity) -> None:
        packet, _ = self._backing.pop(identity)
        self.payloads_held -= len(packet.fragments)
        if self.memory_model is not None:
            for fragment in packet.fragments:
                self.memory_model.evict(
                    (packet.operation_id, packet.round_index, fragment.patch_id))
