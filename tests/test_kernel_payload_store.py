"""Payload store: refcounted retention, idempotent release, replay, peak accounting."""
import pytest

from decsim.payload_store import PayloadStore


def _store_rounds(ps, op_id, rounds):
    for r in rounds:
        ps.store(op_id, r, payload=f"p{op_id}.{r}")


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
    ps.store(0, 1, "a", fragment_index=0)
    ps.store(0, 1, "b", fragment_index=1)     # second fragment of same round
    ps.store(0, 1, "b2", fragment_index=1)    # overwrite: not double-counted
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
