"""Payload store: refcounted retention, idempotent release, replay, peak accounting."""
import pytest

from decsim.message import RetainedSyndromeFragment, SyndromeRoundPacket
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


def test_store_retains_one_complete_packet_and_its_exact_tick():
    store = PayloadStore()
    store.register_op(("operation", 1))
    packet = _packet(
        ("operation", 1),
        3,
        (("north", (0, 1)), ((2, "south"), (1, 0))),
    )

    store.store_round(packet, completion_tick=91)

    assert store.fragments(("operation", 1), 3) == packet.fragments
    assert store.round_complete_tick(("operation", 1), 3) == 91
    assert store.payloads_held == 2
    with pytest.raises(ValueError, match="already retained"):
        store.store_round(packet, completion_tick=92)


def _store_rounds(ps, op_id, rounds):
    ps.register_op(op_id)
    for r in rounds:
        ps.store_round(_packet(op_id, r), completion_tick=r)


def test_weak_release_frees_unshared_rounds():
    ps = PayloadStore()
    ps.lease("w0", [(0, 1), (0, 2), (0, 3)])
    _store_rounds(ps, 0, [1, 2, 3])
    assert ps.payloads_held == 3
    ps.release("w0")
    assert ps.payloads_held == 0 and ps.fragments(0, 2) is None


def test_shared_round_survives_until_last_lease():
    ps = PayloadStore()
    ps.lease("w0", [(0, 1), (0, 2)])
    ps.lease("w1", [(0, 2), (0, 3)])       # round 2 shared (buffer overlap)
    _store_rounds(ps, 0, [1, 2, 3])
    ps.release("w0")
    assert ps.fragments(0, 1) is None      # only w0 needed it
    assert ps.fragments(0, 2) is not None  # w1 still holds it
    ps.release("w1")
    assert ps.payloads_held == 0


def test_strong_lease_replay_then_release():
    ps = PayloadStore()
    ps.lease("w0", [(0, 2), (0, 3)])
    ps.lease("w0.strong", [(0, 1), (0, 4)])   # strong context beyond weak reads
    _store_rounds(ps, 0, [1, 2, 3, 4])
    got = ps.replay([(0, 1), (0, 2), (0, 3), (0, 4)])
    assert [(op, r) for op, r, _ in got] == [(0, 1), (0, 2), (0, 3), (0, 4)]
    ps.release("w0.strong")                    # released at weak commit (parity)
    assert ps.fragments(0, 1) is None and ps.fragments(0, 2) is not None
    ps.release("w0.strong")                    # idempotent: no error, no change
    assert ps.payloads_held == 2


def test_replay_skips_missing_rounds():
    ps = PayloadStore()
    ps.lease("w0", [(0, 1)])
    _store_rounds(ps, 0, [1])
    assert [(op, r) for op, r, _ in ps.replay([(0, 1), (0, 9)])] == [(0, 1)]


def test_peak_and_fragment_accounting():
    ps = PayloadStore()
    ps.lease("w0", [(0, 1)])
    ps.register_op(0)
    packet = _packet(0, 1, ((0, (0,)), (1, (1,))))
    ps.store_round(packet, completion_tick=1)
    assert ps.payloads_held == 2 and ps.peak_payloads == 2
    with pytest.raises(ValueError, match="already retained"):
        ps.store_round(packet, completion_tick=2)
    assert ps.payloads_held == 2 and ps.peak_payloads == 2
    ps.release("w0")
    assert ps.payloads_held == 0 and ps.peak_payloads == 2


def test_duplicate_lease_rejected():
    ps = PayloadStore()
    ps.lease("w0", [(0, 1)])
    with pytest.raises(ValueError):
        ps.lease("w0", [(0, 2)])
    ps.replace("w0", [(0, 2)])                 # replace is the sanctioned path
    _store_rounds(ps, 0, [1, 2])
    assert ps.fragments(0, 1) is None or ps.payloads_held >= 1
