"""Behavior contracts for decoder-local materialized input and its slots.

Covers: capacity refusal, the exact reserve/deposit/take/discard lifecycle,
separate decoder-local materialization (no upstream container aliasing,
leaf bit tuples shared), and duplicate or invalid transitions.
"""

import pytest

from decsim.decoder_local import (
    DecoderInput,
    DecoderLocalCapacityExhaustion,
    MaterializedSyndromeRound,
    DecoderLocalMemory,
    materialize_decoder_input,
)
from decsim.message import (
    DecodeJob,
    DecoderRequestKey,
    DecoderTier,
    RetainedSyndromeFragment,
)


def _fragment(round_index: int, patch_id: str = "north",
              fragment_index: int = 0) -> RetainedSyndromeFragment:
    return RetainedSyndromeFragment(
        operation_id=7,
        patch_id=patch_id,
        round_index=round_index,
        bits=(0, 1, 1, 0),
        code="surface",
        size_bits=4,
        fragment_index=fragment_index,
    )


def _request_key(window_id: int = 3) -> DecoderRequestKey:
    return DecoderRequestKey(
        operation_id=7, window_id=window_id,
        tier=DecoderTier.WEAK, run_sequence=1)


def _job(window_id: int = 3, rounds: tuple = (1, 2)) -> DecodeJob:
    payloads = [_fragment(round_index) for round_index in rounds]
    return DecodeJob(
        op_id=7, window_id=window_id, n_rounds=len(rounds),
        payloads=payloads, label=f"w{window_id}",
        request_key=_request_key(window_id))


# ------------------------------------------------------------- lifecycle

def test_reserve_deposit_take_returns_materialized_input_and_frees_slot():
    memory = DecoderLocalMemory()
    job = _job()

    memory.reserve("k")
    assert memory.slots_in_use == 1
    deposited = memory.deposit("k", job)
    taken = memory.take("k")

    assert taken is deposited
    assert type(taken) is DecoderInput
    assert (taken.op_id, taken.window_id) == (7, 3)
    assert taken.request_key == job.request_key
    assert taken.round_count == 2
    assert [entry.round_index for entry in taken.rounds] == [1, 2]
    assert memory.slots_in_use == 0


def test_discard_frees_reserved_and_deposited_slots():
    memory = DecoderLocalMemory()
    memory.reserve("reserved-only")
    memory.reserve("deposited")
    memory.deposit("deposited", _job())

    memory.discard("reserved-only")
    memory.discard("deposited")
    assert memory.slots_in_use == 0
    # both keys are reusable after discard
    memory.reserve("reserved-only")
    memory.reserve("deposited")


def test_taken_slot_key_is_reusable_for_a_new_decode():
    memory = DecoderLocalMemory()
    memory.reserve("k")
    memory.deposit("k", _job())
    memory.take("k")
    memory.reserve("k")
    second = memory.deposit("k", _job(window_id=4))
    assert second.window_id == 4


# -------------------------------------------------------------- capacity

def test_default_capacity_is_unbounded():
    memory = DecoderLocalMemory()
    assert memory.capacity is None
    for index in range(100):
        memory.reserve(index)
    assert memory.slots_in_use == 100


def test_bounded_capacity_refuses_reserve_when_full():
    memory = DecoderLocalMemory(capacity=2)
    memory.reserve("a")
    memory.reserve("b")
    with pytest.raises(DecoderLocalCapacityExhaustion, match="2 decoder-local"):
        memory.reserve("c")
    assert memory.slots_in_use == 2  # refused reserve left state untouched

    memory.discard("a")
    memory.reserve("c")  # freed slot admits the next reserve
    assert memory.slots_in_use == 2


def test_capacity_must_be_none_or_positive_int():
    for bad in (0, -1, 1.0, True, "2"):
        with pytest.raises(TypeError, match="capacity"):
            DecoderLocalMemory(capacity=bad)


# ------------------------------------------------- invalid transitions

def test_duplicate_reserve_is_refused_in_both_slot_states():
    memory = DecoderLocalMemory()
    memory.reserve("k")
    with pytest.raises(RuntimeError, match="already reserved"):
        memory.reserve("k")
    memory.deposit("k", _job())
    with pytest.raises(RuntimeError, match="already deposited"):
        memory.reserve("k")


def test_deposit_requires_a_reservation_and_happens_once():
    memory = DecoderLocalMemory()
    with pytest.raises(RuntimeError, match="unreserved"):
        memory.deposit("k", _job())
    memory.reserve("k")
    memory.deposit("k", _job())
    with pytest.raises(RuntimeError, match="already holds a deposit"):
        memory.deposit("k", _job())


def test_take_requires_a_deposit_and_discard_requires_a_live_slot():
    memory = DecoderLocalMemory()
    with pytest.raises(RuntimeError, match="unknown"):
        memory.take("k")
    memory.reserve("k")
    with pytest.raises(RuntimeError, match="before any deposit"):
        memory.take("k")
    memory.discard("k")
    with pytest.raises(RuntimeError, match="unknown"):
        memory.take("k")
    with pytest.raises(RuntimeError, match="unknown"):
        memory.discard("k")


def test_failed_deposit_leaves_the_reservation_intact():
    memory = DecoderLocalMemory()
    memory.reserve("k")
    with pytest.raises(TypeError, match="DecodeJob"):
        memory.deposit("k", "not a job")
    memory.deposit("k", _job())  # reservation survived the refused deposit


# ------------------------------------------------ materialization/alias

def test_materialization_builds_new_containers_not_upstream_aliases():
    job = _job(rounds=(1, 2))
    upstream_fragments = list(job.payloads)

    decoder_input = materialize_decoder_input(job)

    materialized = [fragment for entry in decoder_input.rounds
                    for fragment in entry.fragments]
    assert len(materialized) == len(upstream_fragments)
    for local, upstream in zip(materialized, upstream_fragments):
        assert local is not upstream          # new immutable fragment value
        assert local == upstream              # identical field content
        assert local.bits is upstream.bits    # leaf bit tuple may be shared
    assert decoder_input.rounds is not job.payloads
    assert all(type(entry) is MaterializedSyndromeRound
               for entry in decoder_input.rounds)


def test_deposited_input_survives_upstream_payload_list_mutation():
    memory = DecoderLocalMemory()
    job = _job(rounds=(1, 2))
    memory.reserve("k")
    deposited = memory.deposit("k", job)

    job.payloads.clear()  # upstream container mutates after the capture

    taken = memory.take("k")
    assert taken is deposited
    assert taken.round_count == 2
    assert taken.rounds[0].fragments[0].bits == (0, 1, 1, 0)


def test_materialization_groups_fragments_into_ascending_rounds():
    payloads = [
        _fragment(2, patch_id="north", fragment_index=0),
        _fragment(1, patch_id="north", fragment_index=0),
        _fragment(2, patch_id="south", fragment_index=1),
    ]
    job = DecodeJob(op_id=7, window_id=3, n_rounds=2, payloads=payloads,
                    label="w3", request_key=_request_key())

    decoder_input = materialize_decoder_input(job)

    assert [entry.round_index for entry in decoder_input.rounds] == [1, 2]
    assert len(decoder_input.rounds[1].fragments) == 2


def test_materialization_rejects_non_fragment_payloads():
    job = DecodeJob(op_id=7, window_id=3, n_rounds=1,
                    payloads=[object()], label="w3")
    with pytest.raises(TypeError, match="RetainedSyndromeFragment"):
        materialize_decoder_input(job)


def test_decoder_input_is_immutable():
    decoder_input = materialize_decoder_input(_job())
    with pytest.raises(Exception):
        decoder_input.op_id = 9
    with pytest.raises(Exception):
        decoder_input.rounds = ()


def test_decoder_input_rejects_unordered_or_duplicate_rounds():
    round_two = MaterializedSyndromeRound(
        operation_id=7, round_index=2, fragments=(_fragment(2),))
    round_one = MaterializedSyndromeRound(
        operation_id=7, round_index=1, fragments=(_fragment(1),))
    with pytest.raises(ValueError, match="ordered by operation and round"):
        DecoderInput(op_id=7, window_id=3, label="w3", request_key=None,
                     rounds=(round_two, round_one))
    with pytest.raises(ValueError, match="ordered by operation and round"):
        DecoderInput(op_id=7, window_id=3, label="w3", request_key=None,
                     rounds=(round_one, round_one))


def test_materialized_round_rejects_mismatched_fragments():
    with pytest.raises(ValueError, match="share round_index"):
        MaterializedSyndromeRound(
            operation_id=7, round_index=1, fragments=(_fragment(2),))
    with pytest.raises(ValueError, match="share operation identity"):
        MaterializedSyndromeRound(
            operation_id=8, round_index=1, fragments=(_fragment(1),))


# Evidence footer
# ---------------
# Grounding: tmp/validation/combined_syndrome_buffer_implementation_audit_v1/
# REPORT.md. These tests pin the least-claim invariant shared by the audited
# systems: decode input is captured into decoder-owned immutable state with
# an exact slot lifetime, without asserting any DMA/ring/placement topology.


def test_materialization_groups_cross_operation_round_identities():
    first = _fragment(1)
    second = RetainedSyndromeFragment(
        operation_id=8, patch_id="south", round_index=1,
        bits=(1, 0), code="surface", size_bits=2, fragment_index=0,
    )
    job = DecodeJob(
        op_id=7, window_id=3, n_rounds=2,
        payloads=[second, first], label="cross-op",
        request_key=_request_key(),
    )
    materialized = materialize_decoder_input(job)
    assert tuple(
        (row.operation_id, row.round_index) for row in materialized.rounds
    ) == ((7, 1), (8, 1))
