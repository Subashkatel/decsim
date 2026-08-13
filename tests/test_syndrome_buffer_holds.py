"""Syndrome buffer holds: retention fanout, closed-op rejection, memory faults."""
import pytest

from decsim.message import (
    DecoderRequestKey,
    DecoderTier,
    PotentialStrong,
    Replay,
    RephaseGuard,
    RetainedSyndromeFragment,
)
from decsim.syndrome_buffer import SyndromeBuffer, SyndromeBufferRoundState


def _fragments(op_id, round_index, fragments=((0, (0,)),)):
    return tuple(
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
    )


def _retain(buffer, op_id, round_index, fragments=((0, (0,)),), *,
            publication_tick=None):
    """Assemble and pack one round through the one upstream allocation."""
    if not buffer.has_operation(op_id):
        buffer.open_operation(op_id)
    parts = _fragments(op_id, round_index, fragments)
    for fragment in parts:
        buffer.accept_fragment(fragment, expected_fragments=len(parts))
    return buffer.finish_packing(
        (op_id, round_index), publication_tick=publication_tick)


class RecordingMemory:
    def __init__(self):
        self.live = {}
        self.stored = []
        self.evicted = []

    def store(self, key, payload):
        self.live[key] = payload
        self.stored.append(key)

    def evict(self, key):
        self.live.pop(key)
        self.evicted.append(key)


def test_memory_model_observes_exactly_the_retained_fragments_once():
    memory = RecordingMemory()
    buffer = SyndromeBuffer(memory_model=memory)
    packet = _retain(
        buffer, "operation", 1, (("north", (0,)), ("south", (1,))),
        publication_tick=5)

    keys = {("operation", 1, fragment.patch_id)
            for fragment in packet.fragments}
    assert set(memory.live) == keys
    assert buffer.payloads_held == buffer.peak_payloads == 2
    assert buffer.publication_tick(("operation", 1)) == 5

    buffer.register_hold("weak", (("operation", 1),))
    buffer.release_hold("weak")

    assert memory.live == {}
    assert sorted(memory.stored) == sorted(memory.evicted)
    assert buffer.payloads_held == 0 and buffer.peak_payloads == 2
    assert buffer.retained_fragments(("operation", 1)) is None
    assert buffer.publication_tick(("operation", 1)) is None


def test_memory_publication_fault_rolls_back_all_stored_fragments():
    class FaultingMemory(RecordingMemory):
        def store(self, key, payload):
            if len(self.live) == 1:
                raise RuntimeError("memory publication fault")
            super().store(key, payload)

    memory = FaultingMemory()
    buffer = SyndromeBuffer(memory_model=memory)
    buffer.open_operation("operation")
    parts = _fragments("operation", 1, (("north", (0,)), ("south", (1,))))
    for fragment in parts:
        buffer.accept_fragment(fragment, expected_fragments=len(parts))

    with pytest.raises(RuntimeError, match="memory publication fault"):
        buffer.finish_packing(("operation", 1))

    assert memory.live == {}
    assert buffer.payloads_held == buffer.peak_payloads == 0
    assert buffer.retained_fragments(("operation", 1)) is None
    assert buffer.round_state(("operation", 1)) is (
        SyndromeBufferRoundState.PACKING)


def test_closed_replay_source_is_rejected_without_mutation():
    buffer = SyndromeBuffer()
    _retain(buffer, "packet", 1)
    buffer.register_hold("weak", (("packet", 1),))
    before = buffer.snapshot()

    replay = Replay(("source", 0), 0)
    with pytest.raises(RuntimeError, match="closed operation.*source"):
        buffer.register_hold(replay, (("packet", 1),))

    assert not buffer.has_hold(replay)
    assert buffer.snapshot() == before


def test_replay_replacement_refuses_closed_packet_before_release():
    buffer = SyndromeBuffer()
    for operation_id in ("source", "old-packet"):
        buffer.open_operation(operation_id)
    old = Replay(("source", 0), 0)
    buffer.register_hold(old, (("old-packet", 1),))

    replacement = Replay(("source", 0), 1)
    with pytest.raises(RuntimeError, match="closed operation.*closed-packet"):
        buffer.register_hold(
            replacement, (("old-packet", 1), ("closed-packet", 1)))

    assert buffer.has_hold(old)
    assert not buffer.has_hold(replacement)
    assert buffer.hold_round_identities(old) == (("old-packet", 1),)


def test_rephase_guard_refuses_closed_request_without_mutation():
    buffer = SyndromeBuffer()
    _retain(buffer, "packet", 1)
    buffer.register_hold("weak", (("packet", 1),))
    before = buffer.snapshot()

    request_key = DecoderRequestKey("request", 0, DecoderTier.STRONG, 0)
    guard = RephaseGuard(request_key)
    with pytest.raises(RuntimeError, match="closed operation.*request"):
        buffer.register_hold(guard, (("packet", 1),))

    assert not buffer.has_hold(guard)
    assert buffer.snapshot() == before


def test_close_keeps_cross_operation_release_history_until_all_ops_close():
    buffer = SyndromeBuffer()
    owner = PotentialStrong((0, 0))
    buffer.open_operation(0)
    buffer.open_operation(1)
    buffer.register_hold(owner, ((0, 1), (1, 1)))
    buffer.release_hold(owner)

    buffer.close_operation(0)
    buffer.release_hold(owner)
    with pytest.raises(ValueError, match="duplicate consumer hold"):
        buffer.register_hold(owner, ((1, 1),))
    buffer.close_operation(1)
    buffer.open_operation(2)
    buffer.register_hold(owner, ((2, 1),))


def test_operation_reference_tracks_replay_window_source():
    buffer = SyndromeBuffer()
    for operation_id in ("source", "packet"):
        buffer.open_operation(operation_id)
    replay = Replay(("source", 0), 0)
    buffer.register_hold(replay, (("packet", 1),))

    assert buffer.has_live_operation_reference("source")
    with pytest.raises(RuntimeError, match="live consumer holds"):
        buffer.close_operation("source")

    buffer.release_hold(replay)
    assert not buffer.has_live_operation_reference("source")
    buffer.close_operation("source")


def test_last_hold_release_frees_one_backing_and_preserves_peak():
    buffer = SyndromeBuffer(capacity=1)
    packet = _retain(
        buffer, "operation", 1, ((0, (0,)), (1, (1,))), publication_tick=5)
    holders = ("weak", "potential-strong")
    for holder in holders:
        buffer.register_hold(holder, (("operation", 1),))

    assert buffer.retained_fragments(("operation", 1)) == packet.fragments
    assert buffer.publication_tick(("operation", 1)) == 5
    assert buffer.payloads_held == buffer.peak_payloads == 2
    buffer.release_hold(holders[0])
    assert buffer.retained_fragments(("operation", 1)) is not None
    buffer.release_hold(holders[1])
    assert buffer.retained_fragments(("operation", 1)) is None
    assert buffer.payloads_held == 0 and buffer.peak_payloads == 2


def test_release_round_if_unheld_frees_only_unconsumed_rounds():
    buffer = SyndromeBuffer()
    _retain(buffer, "operation", 1)
    _retain(buffer, "operation", 2)
    buffer.register_hold("weak", (("operation", 2),))

    assert buffer.release_round_if_unheld(("operation", 1))
    assert not buffer.release_round_if_unheld(("operation", 2))
    assert buffer.retained_fragments(("operation", 1)) is None
    assert buffer.retained_fragments(("operation", 2)) is not None
