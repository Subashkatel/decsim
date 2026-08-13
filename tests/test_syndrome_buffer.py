"""SyndromeBuffer core: one allocation per round, holds, capacity."""
import pytest

from decsim.message import RetainedSyndromeFragment
from decsim.syndrome_buffer import (
    SyndromeBuffer,
    SyndromeBufferCapacityExhaustion,
    SyndromeBufferRoundState,
)


def _fragment(op_id="op", patch_id=0, round_index=1, bits=(0, 1),
              fragment_index=0):
    return RetainedSyndromeFragment(
        operation_id=op_id,
        patch_id=patch_id,
        round_index=round_index,
        bits=tuple(bits),
        code=None,
        size_bits=len(bits),
        fragment_index=fragment_index,
    )


def _buffer_with_round(buffer=None, *, op_id="op", round_index=1,
                       fragments=((0, (0, 1)), (1, (1, 0)))):
    """Open op, assemble all fragments, and finish packing one round."""
    if buffer is None:
        buffer = SyndromeBuffer()
    if not buffer.has_operation(op_id):
        buffer.open_operation(op_id)
    for index, (patch_id, bits) in enumerate(fragments):
        buffer.accept_fragment(
            _fragment(op_id, patch_id, round_index, bits, index),
            expected_fragments=len(fragments),
        )
    buffer.finish_packing((op_id, round_index))
    return buffer


def test_conservation_of_identities_and_bits_through_packing():
    buffer = SyndromeBuffer()
    buffer.open_operation("op")
    source = [(0, (0, 1, 1)), (1, (1, 0, 0)), (2, (1, 1, 0))]
    # Deliver out of order to check fragment-index ordering, not arrival.
    for index, (patch_id, bits) in ((2, source[2]), (0, source[0]),
                                    (1, source[1])):
        admission = buffer.accept_fragment(
            _fragment("op", patch_id, 1, bits, index), expected_fragments=3)
        assert admission.round_identity == ("op", 1)
    packet = buffer.finish_packing(("op", 1))
    assert packet.operation_id == "op"
    assert packet.round_index == 1
    assert [f.fragment_index for f in packet.fragments] == [0, 1, 2]
    assert [f.patch_id for f in packet.fragments] == [0, 1, 2]
    assert [f.bits for f in packet.fragments] == [bits for _, bits in source]
    assert sum(f.size_bits for f in packet.fragments) == 9
    assert buffer.read_retained_round(("op", 1)) is packet


def test_same_patch_fragments_merge_bits_in_fragment_index_order():
    buffer = SyndromeBuffer()
    buffer.open_operation("op")
    buffer.accept_fragment(
        _fragment("op", 7, 1, (1, 1), fragment_index=1), expected_fragments=2)
    buffer.accept_fragment(
        _fragment("op", 7, 1, (0, 0), fragment_index=0), expected_fragments=2)
    packet = buffer.finish_packing(("op", 1))
    assert len(packet.fragments) == 1
    assert packet.fragments[0].patch_id == 7
    assert packet.fragments[0].bits == (0, 0, 1, 1)
    assert packet.fragments[0].size_bits == 4


def test_one_physical_allocation_spans_assembly_through_retained():
    buffer = SyndromeBuffer(capacity=2)
    buffer.open_operation("op")
    first = buffer.accept_fragment(
        _fragment("op", 0, 1, (0,), 0), expected_fragments=2)
    assert buffer.metrics().allocations_total == 1
    second = buffer.accept_fragment(
        _fragment("op", 1, 1, (1,), 1), expected_fragments=2)
    assert second.round_complete
    assert second.slot_index == first.slot_index
    assert buffer.round_state(("op", 1)) is SyndromeBufferRoundState.PACKING
    buffer.finish_packing(("op", 1))
    snapshot = buffer.snapshot()
    assert dict(snapshot.identity_to_slot)[("op", 1)] == first.slot_index
    assert snapshot.retained_identities == (("op", 1),)
    assert buffer.metrics().allocations_total == 1


def test_fragment_admission_retires_at_packing_before_publication():
    buffer = SyndromeBuffer()
    buffer.open_operation("op")
    admission = buffer.accept_fragment(
        _fragment("op", 0, 1, (0,), 0), expected_fragments=1)
    assert admission.round_complete
    # PACKING, publication has not happened yet: admission already retired.
    with pytest.raises(ValueError, match="retired at packing"):
        buffer.accept_fragment(
            _fragment("op", 1, 1, (1,), 0), expected_fragments=1)
    buffer.finish_packing(("op", 1))
    with pytest.raises(ValueError, match="retired at packing"):
        buffer.accept_fragment(
            _fragment("op", 1, 1, (1,), 0), expected_fragments=1)


def test_late_fragment_after_release_is_rejected_by_tombstone():
    buffer = _buffer_with_round()
    buffer.register_hold("consumer", (("op", 1),))
    buffer.release_hold("consumer")
    assert ("op", 1) in buffer.snapshot().tombstoned_identities
    with pytest.raises(ValueError, match="late fragment"):
        buffer.accept_fragment(
            _fragment("op", 0, 1, (0,), 0), expected_fragments=2)


def test_mismatched_and_duplicate_fragments_are_rejected():
    buffer = SyndromeBuffer()
    buffer.open_operation("op")
    buffer.accept_fragment(
        _fragment("op", 0, 1, (0,), 0), expected_fragments=3)
    with pytest.raises(ValueError, match="same count"):
        buffer.accept_fragment(
            _fragment("op", 1, 1, (1,), 1), expected_fragments=2)
    with pytest.raises(ValueError, match="duplicate"):
        buffer.accept_fragment(
            _fragment("op", 1, 1, (1,), 0), expected_fragments=3)
    with pytest.raises(ValueError, match="exceeds"):
        buffer.accept_fragment(
            _fragment("op", 1, 1, (1,), 3), expected_fragments=3)
    with pytest.raises(RuntimeError, match="not open"):
        buffer.accept_fragment(
            _fragment("other", 0, 1, (0,), 0), expected_fragments=1)


def test_overlapping_consumers_fan_out_and_last_release_frees():
    buffer = _buffer_with_round()
    buffer.register_hold("window-a", (("op", 1),))
    buffer.register_hold("window-b", (("op", 1),))
    buffer.release_hold("window-a")
    packet = buffer.read_retained_round(("op", 1))
    assert packet.round_index == 1
    assert buffer.snapshot().retained_identities == (("op", 1),)
    buffer.release_hold("window-b")
    buffer.release_hold("window-b")  # idempotent
    assert buffer.snapshot().occupancy == 0
    with pytest.raises(RuntimeError, match="not packed and retained"):
        buffer.read_retained_round(("op", 1))


def test_replace_hold_frees_rounds_it_was_the_last_holder_of():
    buffer = _buffer_with_round(round_index=1)
    buffer = _buffer_with_round(buffer, round_index=2)
    buffer.register_hold("consumer", (("op", 1), ("op", 2)))
    buffer.replace_hold("consumer", (("op", 2),))
    snapshot = buffer.snapshot()
    assert snapshot.retained_identities == (("op", 2),)
    assert ("op", 1) in snapshot.tombstoned_identities
    assert buffer.hold_round_identities("consumer") == (("op", 2),)
    with pytest.raises(RuntimeError, match="not live"):
        buffer.replace_hold("stranger", (("op", 2),))


def test_transfer_hold_moves_membership_without_a_free_window():
    buffer = _buffer_with_round()
    buffer.register_hold("old-owner", (("op", 1),))
    buffer.transfer_hold("old-owner", "new-owner")
    assert buffer.snapshot().retained_identities == (("op", 1),)
    assert not buffer.has_hold("old-owner")
    assert buffer.hold_round_identities("new-owner") == (("op", 1),)
    with pytest.raises(ValueError, match="duplicate"):
        buffer.register_hold("old-owner", (("op", 1),))
    buffer.release_hold("new-owner")
    assert buffer.snapshot().occupancy == 0


def test_cancellation_releases_an_assembling_round_and_blocks_late_fragments():
    buffer = SyndromeBuffer()
    buffer.open_operation("op")
    buffer.accept_fragment(
        _fragment("op", 0, 1, (0,), 0), expected_fragments=2)
    buffer.release_round(("op", 1))
    assert buffer.round_state(("op", 1)) is None
    with pytest.raises(ValueError, match="late fragment"):
        buffer.accept_fragment(
            _fragment("op", 1, 1, (1,), 1), expected_fragments=2)


def test_release_round_refuses_rounds_with_live_holds():
    buffer = _buffer_with_round()
    buffer.register_hold("consumer", (("op", 1),))
    with pytest.raises(RuntimeError, match="live consumer holds"):
        buffer.release_round(("op", 1))
    buffer.release_hold("consumer")
    with pytest.raises(RuntimeError, match="no live allocation"):
        buffer.release_round(("op", 1))


def test_close_operation_requires_no_live_rounds_or_holds():
    buffer = _buffer_with_round()
    with pytest.raises(RuntimeError, match="live buffer rounds"):
        buffer.close_operation("op")
    buffer.register_hold("consumer", (("op", 1),))
    buffer.release_hold("consumer")
    buffer.register_hold("late-consumer-check", ())
    buffer.close_operation("op")
    assert not buffer.has_operation("op")
    assert buffer.snapshot().tombstoned_identities == ()
    with pytest.raises(RuntimeError, match="not open"):
        buffer.accept_fragment(
            _fragment("op", 0, 2, (0,), 0), expected_fragments=1)


def test_close_operation_refuses_while_holds_reference_it():
    buffer = _buffer_with_round()
    buffer.register_hold("consumer", (("op", 1),))
    buffer.release_hold("consumer")
    # A future hold over a round that never allocated still pins the op open.
    buffer.register_hold("future", (("op", 9),))
    with pytest.raises(RuntimeError, match="live consumer holds"):
        buffer.close_operation("op")
    buffer.release_hold("future")
    buffer.close_operation("op")


def test_finite_capacity_rejects_then_reuses_the_freed_slot():
    buffer = SyndromeBuffer(capacity=1)
    buffer.open_operation("op")
    first = buffer.accept_fragment(
        _fragment("op", 0, 1, (0,), 0), expected_fragments=1)
    with pytest.raises(SyndromeBufferCapacityExhaustion) as exc_info:
        buffer.accept_fragment(
            _fragment("op", 0, 2, (0,), 0), expected_fragments=1)
    assert exc_info.value.capacity == 1
    assert exc_info.value.incoming_identity == ("op", 2)
    buffer.finish_packing(("op", 1))
    buffer.register_hold("consumer", (("op", 1),))
    buffer.release_hold("consumer")
    reused = buffer.accept_fragment(
        _fragment("op", 0, 2, (0,), 0), expected_fragments=1)
    assert reused.slot_index == first.slot_index
    metrics = buffer.metrics()
    assert metrics.allocations_total == 2
    assert metrics.peak_live_allocations == 1
    assert metrics.released_rounds == 1


def test_capacity_none_is_unbounded():
    buffer = SyndromeBuffer(capacity=None)
    buffer.open_operation("op")
    for round_index in range(1, 6):
        buffer.accept_fragment(
            _fragment("op", 0, round_index, (0,), 0), expected_fragments=1)
    snapshot = buffer.snapshot()
    assert snapshot.capacity is None
    assert snapshot.occupancy == 5
    assert buffer.metrics().peak_live_allocations == 5


def test_finish_packing_requires_the_packing_state():
    buffer = SyndromeBuffer()
    buffer.open_operation("op")
    with pytest.raises(RuntimeError, match="no live allocation"):
        buffer.finish_packing(("op", 1))
    buffer.accept_fragment(
        _fragment("op", 0, 1, (0,), 0), expected_fragments=2)
    with pytest.raises(RuntimeError, match="ASSEMBLING, not PACKING"):
        buffer.finish_packing(("op", 1))
    buffer.accept_fragment(
        _fragment("op", 1, 1, (1,), 1), expected_fragments=2)
    buffer.finish_packing(("op", 1))
    with pytest.raises(RuntimeError, match="PACKED_RETAINED, not PACKING"):
        buffer.finish_packing(("op", 1))


def test_hold_registration_validates_identities_and_tokens():
    buffer = _buffer_with_round()
    with pytest.raises(TypeError, match="cannot be None"):
        buffer.register_hold(None, (("op", 1),))
    with pytest.raises(TypeError, match="round identities"):
        buffer.register_hold("consumer", (("op", 0),))
    with pytest.raises(RuntimeError, match="closed operation"):
        buffer.register_hold("consumer", (("ghost-op", 1),))
    buffer.register_hold("consumer", (("op", 1),))
    buffer.release_hold("consumer")
    with pytest.raises(ValueError, match="released round"):
        buffer.register_hold("late", (("op", 1),))


def test_release_future_hold_before_packing_frees_on_publication():
    buffer = SyndromeBuffer()
    buffer.open_operation("op")
    buffer.register_hold("consumer", (("op", 1),))
    buffer.accept_fragment(_fragment("op", 0, 1, (1,), 0), expected_fragments=1)
    buffer.release_hold("consumer")
    packet = buffer.finish_packing(("op", 1))
    assert packet.round_index == 1
    assert buffer.round_state(("op", 1)) is None
    assert buffer.metrics().live_allocations == 0


def test_closed_operation_identity_cannot_be_reopened():
    buffer = SyndromeBuffer()
    buffer.open_operation("op")
    buffer.close_operation("op")
    with pytest.raises(RuntimeError, match="cannot be reused"):
        buffer.open_operation("op")
