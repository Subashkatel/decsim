"""Payload store: refcounted retention, idempotent release, replay, peak accounting."""
import pytest

from decsim.message import (
    DecoderRequestKey,
    DecoderTier,
    EndpointRole,
    EndpointState,
    PotentialStrong,
    Replay,
    RephaseGuard,
    RetainedSyndromeFragment,
    SyndromeRoundPacket,
)
from decsim.payload_store import PayloadStore


def _packet(op_id, round_index, fragments=((0, (0,)),)):
    return SyndromeRoundPacket(
        operation_id=op_id,
        round_index=round_index,
        fragments=tuple(
            RetainedSyndromeFragment(
                operation_id=op_id,
                patch_id=patch_id,
                round_index=round_index,
                bits=bits,
                code=None,
                size_bits=len(bits),
                fragment_index=fragment_index,
            )
            for fragment_index, (patch_id, bits) in enumerate(fragments)
        ),
    )


def _store_state(store):
    ledgers = []
    for role in EndpointRole:
        ledger = store._ledgers[role]
        ledgers.append((
            role, tuple(ledger.slots), frozenset(ledger.free),
            tuple(sorted(ledger.indices.items(), key=repr)),
            tuple(sorted(ledger.states.items(), key=repr)),
            tuple(sorted(
                ((identity, frozenset(owners))
                 for identity, owners in ledger.owners.items()),
                key=repr,
            )),
        ))
    return (
        tuple(ledgers),
        tuple(sorted(store._future_owners.items(), key=repr)),
        tuple(sorted(store._released_owners.items(), key=repr)),
        tuple(sorted(store._backing.items(), key=repr)),
        tuple(sorted(
            ((pair.identity, pair.state, pair.completion_tick)
             for pair in store._pairs),
            key=repr,
        )),
        frozenset(store._open_operations),
        store.payloads_held,
        store.peak_payloads,
    )


def test_endpoint_slots_release_out_of_order_reuse_exact_index_and_bound_backing():
    store = PayloadStore(sb0_capacity=3, sb1_capacity=3)

    def publish(identity):
        key = (identity, 1)
        owner = ("consumer", identity)
        store.register_op(identity)
        store.register_owner(EndpointRole.SB0, owner, (key,))
        store.register_owner(EndpointRole.SB1, owner, (key,))
        pair = store.prepare_pair(_packet(identity, 1), completion_tick=1)
        assert pair is not None
        pair.commit_unpublished()
        pair.publish()
        store.complete_cryo(key)
        return owner

    owners = {identity: publish(identity) for identity in "ABC"}
    before = store.endpoint_snapshot(EndpointRole.SB0)
    slots = dict(before.identity_to_slot)
    assert set(slots) == {(identity, 1) for identity in "ABC"}
    assert before.free_slot_indices == ()

    store.release_owner(EndpointRole.SB0, owners["B"])
    after_sb0 = store.endpoint_snapshot(EndpointRole.SB0)
    assert dict(after_sb0.identity_to_slot) == {
        ("A", 1): slots[("A", 1)], ("C", 1): slots[("C", 1)]}
    assert after_sb0.free_slot_indices == (slots[("B", 1)],)
    assert store.backing_identities == (("A", 1), ("B", 1), ("C", 1))

    store.release_owner(EndpointRole.SB1, owners["B"])
    assert store.backing_identities == (("A", 1), ("C", 1))
    owner_d = publish("D")
    assert dict(store.endpoint_snapshot(
        EndpointRole.SB0).identity_to_slot)[("D", 1)] == slots[("B", 1)]
    assert dict(store.endpoint_snapshot(
        EndpointRole.SB1).identity_to_slot)[("D", 1)] == slots[("B", 1)]
    assert set(store.backing_identities) == {
        ("A", 1), ("C", 1), ("D", 1)}

    for identity, owner in (*owners.items(), ("D", owner_d)):
        if identity != "B":
            store.release_owner(EndpointRole.SB0, owner)
            assert (identity, 1) in store.backing_identities
            store.release_owner(EndpointRole.SB1, owner)
    assert store.backing_identities == ()


def test_sb1_is_cryo_in_flight_until_transport_completion():
    store = PayloadStore(sb0_capacity=1, sb1_capacity=1)
    key = ("packet", 3)
    store.register_op(key[0])
    owner = ("consumer", key)
    for role in EndpointRole:
        store.register_owner(role, owner, (key,))
    pair = store.prepare_pair(_packet(*key), completion_tick=4)
    pair.commit_unpublished()
    pair.publish()

    sb0_states = dict(
        store.endpoint_snapshot(EndpointRole.SB0).states_by_identity)
    sb1_states = dict(
        store.endpoint_snapshot(EndpointRole.SB1).states_by_identity)
    assert sb0_states[key] is EndpointState.RESIDENT
    assert sb1_states[key] is EndpointState.CRYO_IN_FLIGHT

    store.complete_cryo(key)

    assert dict(store.endpoint_snapshot(
        EndpointRole.SB1).states_by_identity)[key] is EndpointState.RESIDENT


@pytest.mark.parametrize(
    "fault_hook", ("_after_sb0_activation", "_after_sb1_activation"))
def test_pair_publication_fault_is_atomic_at_either_endpoint(
    monkeypatch, fault_hook,
):
    store = PayloadStore(sb0_capacity=1, sb1_capacity=1)
    key = ("packet", 7)
    store.register_op(key[0])
    owner = ("consumer", key)
    for role in EndpointRole:
        store.register_owner(role, owner, (key,))
    before = {role: store.endpoint_snapshot(role) for role in EndpointRole}
    pair = store.prepare_pair(_packet(*key), completion_tick=11)
    pair.commit_unpublished()

    def injected_fault(_pair):
        raise RuntimeError(f"fault at {fault_hook}")

    monkeypatch.setattr(store, fault_hook, injected_fault)
    with pytest.raises(RuntimeError, match="fault at"):
        pair.publish()

    assert pair.state.name == "CANCELLED"
    assert store.backing_identities == ()
    assert store.payloads_held == 0
    assert {role: store.endpoint_snapshot(role)
            for role in EndpointRole} == before


def test_pair_memory_publication_fault_rolls_back_all_physical_effects():
    class FaultingMemory:
        def __init__(self):
            self.live = {}

        def store(self, key, value):
            if len(self.live) == 1:
                raise RuntimeError("memory publication fault")
            self.live[key] = value

        def evict(self, key):
            self.live.pop(key)

    memory = FaultingMemory()
    store = PayloadStore(
        memory_model=memory, sb0_capacity=1, sb1_capacity=1)
    key = ("packet", 9)
    store.register_op(key[0])
    owner = ("consumer", key)
    for role in EndpointRole:
        store.register_owner(role, owner, (key,))
    before = {role: store.endpoint_snapshot(role) for role in EndpointRole}
    pair = store.prepare_pair(
        _packet("packet", 9, (("north", (0,)), ("south", (1,)))),
        completion_tick=13)
    pair.commit_unpublished()

    with pytest.raises(RuntimeError, match="memory publication fault"):
        pair.publish()

    assert pair.state.name == "CANCELLED"
    assert memory.live == {}
    assert store.payloads_held == store.peak_payloads == 0
    assert store.backing_identities == ()
    assert {role: store.endpoint_snapshot(role)
            for role in EndpointRole} == before


def test_operation_close_rejects_live_owners_and_purges_released_history():
    store = PayloadStore(sb0_capacity=1, sb1_capacity=1)
    owner = PotentialStrong((0, 0))
    store.register_op(0)
    store.register_owner(EndpointRole.SB1, owner, ((0, 1),))

    with pytest.raises(RuntimeError, match="live endpoint owner"):
        store.close_operation(0)

    store.release_owner(EndpointRole.SB1, owner)
    store.close_operation(0)

    store.register_op(1)
    store.register_owner(EndpointRole.SB1, owner, ((1, 1),))


def test_close_keeps_cross_operation_release_history_until_all_ops_close():
    store = PayloadStore()
    owner = PotentialStrong((0, 0))
    store.register_op(0)
    store.register_op(1)
    store.register_owner(EndpointRole.SB1, owner, ((0, 1), (1, 1)))
    store.release_owner(EndpointRole.SB1, owner)

    store.close_operation(0)
    store.release_owner(EndpointRole.SB1, owner)
    with pytest.raises(ValueError, match="duplicate endpoint owner"):
        store.register_owner(EndpointRole.SB1, owner, ((1, 1),))
    store.close_operation(1)
    store.register_op(2)
    store.register_owner(EndpointRole.SB1, owner, ((2, 1),))


def test_closed_replay_source_is_rejected_before_backed_slot_reservation(
    monkeypatch,
):
    store = PayloadStore(sb0_capacity=1, sb1_capacity=1)
    store.register_op("source")
    store.register_op("packet")
    packet_identity = ("packet", 1)
    sb0_owner = ("weak", packet_identity)
    store.register_owner(EndpointRole.SB0, sb0_owner, (packet_identity,))
    pair = store.prepare_pair(_packet(*packet_identity), completion_tick=1)
    pair.commit_unpublished()
    pair.publish()
    store.complete_cryo(packet_identity)
    store.close_operation("source")

    before = _store_state(store)
    reserve_calls = 0
    ledger = store._ledgers[EndpointRole.SB1]
    original_reserve = ledger.reserve

    def count_reservation(identity):
        nonlocal reserve_calls
        reserve_calls += 1
        return original_reserve(identity)

    monkeypatch.setattr(ledger, "reserve", count_reservation)
    owner = Replay(("source", 0), 0)
    with pytest.raises(RuntimeError, match="closed operation.*source"):
        store.register_owner(EndpointRole.SB1, owner, (packet_identity,))

    assert reserve_calls == 0
    assert not store.has_owner(EndpointRole.SB1, owner)
    assert before == _store_state(store)


def test_replay_replacement_refuses_closed_packet_before_release_or_reserve(
    monkeypatch,
):
    store = PayloadStore(sb0_capacity=1, sb1_capacity=1)
    for operation_id in ("source", "old-packet", "closed-packet"):
        store.register_op(operation_id)
    old = Replay(("source", 0), 0)
    replacement = Replay(("source", 0), 1)
    store.register_owner(
        EndpointRole.SB1, old, (("old-packet", 1),))
    store.close_operation("closed-packet")
    before = _store_state(store)
    reserve_calls = 0
    release_calls = 0

    for ledger in store._ledgers.values():
        reserve = ledger.reserve

        def count_reserve(identity, *, _reserve=reserve):
            nonlocal reserve_calls
            reserve_calls += 1
            return _reserve(identity)

        monkeypatch.setattr(ledger, "reserve", count_reserve)
    release_owner = store.release_owner

    def count_release(role, owner):
        nonlocal release_calls
        release_calls += 1
        return release_owner(role, owner)

    monkeypatch.setattr(store, "release_owner", count_release)
    with pytest.raises(RuntimeError, match="closed operation.*closed-packet"):
        store.register_owner(
            EndpointRole.SB1,
            replacement,
            (("old-packet", 1), ("closed-packet", 1)),
        )

    assert reserve_calls == release_calls == 0
    assert store.has_owner(EndpointRole.SB1, old)
    assert not store.has_owner(EndpointRole.SB1, replacement)
    assert _store_state(store) == before


def test_two_role_guard_refuses_closed_request_without_mutation_or_reserve(
    monkeypatch,
):
    store = PayloadStore(sb0_capacity=1, sb1_capacity=1)
    store.register_op("request")
    store.register_op("packet")
    identity = ("packet", 1)
    store.register_owner(EndpointRole.SB0, "weak", (identity,))
    pair = store.prepare_pair(_packet(*identity), completion_tick=2)
    pair.commit_unpublished()
    pair.publish()
    store.complete_cryo(identity)
    store.close_operation("request")
    before = _store_state(store)
    reserve_calls = 0

    for ledger in store._ledgers.values():
        reserve = ledger.reserve

        def count_reserve(packet_identity, *, _reserve=reserve):
            nonlocal reserve_calls
            reserve_calls += 1
            return _reserve(packet_identity)

        monkeypatch.setattr(ledger, "reserve", count_reserve)
    request_key = DecoderRequestKey(
        "request", 0, DecoderTier.STRONG, 0)
    guard = RephaseGuard(request_key)
    with pytest.raises(RuntimeError, match="closed operation.*request"):
        store.register_rephase_guard(guard, (identity,), (identity,))

    assert reserve_calls == 0
    assert not any(store.has_owner(role, guard) for role in EndpointRole)
    assert _store_state(store) == before


def test_membership_replacement_acquires_before_releasing_old_packets(
    monkeypatch,
):
    store = PayloadStore(sb0_capacity=3, sb1_capacity=3)
    for operation_id in ("source", "a", "b", "c"):
        store.register_op(operation_id)
    owner = RephaseGuard(
        DecoderRequestKey("source", 0, DecoderTier.STRONG, 0))
    a, b, c = ((identity, 1) for identity in "abc")
    store.register_owner(EndpointRole.SB1, owner, (a, b))
    store.register_owner(EndpointRole.SB0, "c-backing", (c,))
    for identity in (a, b, c):
        pair = store.prepare_pair(_packet(*identity), completion_tick=1)
        assert pair is not None
        pair.commit_unpublished()
        pair.publish()
        store.complete_cryo(identity)

    ledger = store._ledgers[EndpointRole.SB1]
    activate_resident = ledger.activate_resident

    def assert_old_owner_is_live(identity, owners):
        if identity == c:
            assert a in ledger.indices
            assert owner in ledger.owners[a]
            assert store.fragments(*a) is not None
        activate_resident(identity, owners)

    monkeypatch.setattr(ledger, "activate_resident", assert_old_owner_is_live)
    store.replace_owner_membership(EndpointRole.SB1, owner, (b, c))

    assert store.owner_packet_identities(EndpointRole.SB1, owner) == (b, c)
    assert store.fragments(*a) is None
    assert store.fragments(*b) is not None
    assert store.fragments(*c) is not None


@pytest.mark.parametrize(
    ("retained_role", "late_role"),
    ((EndpointRole.SB0, EndpointRole.SB1),
     (EndpointRole.SB1, EndpointRole.SB0)),
)
def test_late_other_role_activation_is_resident_without_transport(
    retained_role, late_role,
):
    store = PayloadStore(sb0_capacity=2, sb1_capacity=2)
    store.register_op("operation")
    identity = ("operation", 1)
    retained_owner = ("retained", retained_role)
    store.register_owner(retained_role, retained_owner, (identity,))
    pair = store.prepare_pair(_packet(*identity), completion_tick=7)
    pair.commit_unpublished()
    pair.publish()
    store.complete_cryo(identity)
    other_role = EndpointRole.SB1 if retained_role is EndpointRole.SB0 \
        else EndpointRole.SB0
    if other_role is not retained_role:
        transient_owners = tuple(
            owner for owner in store._ledgers[other_role].owners.get(identity, ())
            if owner != retained_owner)
        for owner in transient_owners:
            store._ledgers[other_role].release(identity, owner)
    if identity in store._ledgers[late_role].owners:
        store.release_owner(late_role, retained_owner)

    late_owner = ("late", late_role)
    store.register_owner(late_role, late_owner, (identity,))

    snapshot = store.endpoint_snapshot(late_role)
    assert dict(snapshot.states_by_identity)[identity] is EndpointState.RESIDENT
    assert store._cryo_owner not in store._ledgers[late_role].owners[identity]


def test_same_role_attachment_preserves_in_flight_state():
    store = PayloadStore(sb0_capacity=1, sb1_capacity=1)
    store.register_op("operation")
    identity = ("operation", 1)
    first = ("strong", 1)
    store.register_owner(EndpointRole.SB1, first, (identity,))
    pair = store.prepare_pair(_packet(*identity), completion_tick=3)
    pair.commit_unpublished()
    pair.publish()

    second = ("strong", 2)
    store.register_owner(EndpointRole.SB1, second, (identity,))

    state = dict(store.endpoint_snapshot(
        EndpointRole.SB1).states_by_identity)[identity]
    assert state is EndpointState.CRYO_IN_FLIGHT


def test_last_typed_owner_release_frees_one_backing_and_preserves_peak():
    store = PayloadStore(sb0_capacity=1, sb1_capacity=1)
    store.register_op("operation")
    identity = ("operation", 1)
    packet = _packet("operation", 1, ((0, (0,)), (1, (1,))))
    owners = {role: (role.name, identity) for role in EndpointRole}
    for role, owner in owners.items():
        store.register_owner(role, owner, (identity,))
    pair = store.prepare_pair(packet, completion_tick=5)
    assert pair is not None
    pair.commit_unpublished()
    pair.publish()
    store.complete_cryo(identity)

    assert store.fragments(*identity) == packet.fragments
    assert store.round_complete_tick(*identity) == 5
    assert store.payloads_held == store.peak_payloads == 2
    store.release_owner(EndpointRole.SB0, owners[EndpointRole.SB0])
    assert store.fragments(*identity) is not None
    store.release_owner(EndpointRole.SB1, owners[EndpointRole.SB1])
    assert store.fragments(*identity) is None
    assert store.payloads_held == 0 and store.peak_payloads == 2
