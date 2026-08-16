from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError, dataclass, fields

import pytest

import decsim.decoder_local as decoder_local
from decsim.decoder_local import (
    DecoderInput,
    DecoderLocalCapacityExhaustion,
    DecoderLocalMemory,
    MaterializedSyndromeRound,
    materialize_decoder_input,
)
from decsim.message import DecodeJob, RetainedSyndromeFragment


@dataclass(frozen=True)
class CompatibleRetainedFragment:
    operation_id: object
    patch_id: object
    round_index: int
    bits: tuple[int, ...]
    code: str
    size_bits: int
    fragment_index: int


@dataclass
class CompatibleJob:
    op_id: object
    window_id: object
    request_key: object
    payloads: object


class OneShotPayloads:
    def __init__(self, payloads: list[object]) -> None:
        self._payloads = payloads
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("one-shot payloads were traversed more than once")
        yield from self._payloads


def make_fragment(
    operation_id: object = 1,
    round_index: int = 0,
    fragment_index: int = 0,
    *,
    patch_id: object | None = None,
    bits: tuple[int, ...] = (0, 1),
) -> RetainedSyndromeFragment:
    return RetainedSyndromeFragment(
        operation_id=operation_id,
        patch_id=patch_id if patch_id is not None else f"patch-{fragment_index}",
        round_index=round_index,
        bits=bits,
        code="surface",
        size_bits=len(bits),
        fragment_index=fragment_index,
    )


def make_job(
    payloads: list[object] | None = None,
    *,
    op_id: object = 41,
    window_id: object = 7,
    request_key: object | None = None,
) -> DecodeJob:
    return DecodeJob(
        op_id=op_id,
        window_id=window_id,
        n_rounds=len(payloads or []),
        payloads=list(payloads or []),
        request_key=request_key,
    )


def test_module_imports_only_kept_dependencies_and_has_no_stale_helper() -> None:
    """The module keeps only its narrow dependency set and no retired helper."""
    tree = ast.parse(inspect.getsource(decoder_local))
    imports = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            imports.append(
                (node.level, node.module, tuple(alias.name for alias in node.names))
            )
    assert imports == [
        (0, "__future__", ("annotations",)),
        (0, "dataclasses", ("dataclass",)),
        (0, "enum", ("Enum",)),
        (0, "typing", ("Any", "Optional")),
        (
            1,
            "message",
            (
                "DecodeJob",
                "DecoderRequestKey",
                "RetainedSyndromeFragment",
                "same_stable_identity",
                "stable_identity_order_key",
            ),
        ),
    ]
    assert not hasattr(decoder_local, "_rematerialized_fragment")


def test_materialization_orders_rounds_and_preserves_within_round_order() -> None:
    """Materialization canonically orders rounds without permuting their fragments."""
    first_in_round = make_fragment(1, 4, 8, bits=(1, 0))
    second_in_round = make_fragment(1, 4, 2, bits=(0, 1))
    payloads = [
        make_fragment(2, 3, 0),
        first_in_round,
        make_fragment(1, 2, 0),
        second_in_round,
        make_fragment(2, 1, 0),
    ]

    decoder_input = materialize_decoder_input(make_job(payloads))

    assert [
        (round_input.operation_id, round_input.round_index)
        for round_input in decoder_input.rounds
    ] == [(1, 2), (1, 4), (2, 1), (2, 3)]
    assert decoder_input.rounds[1].fragments == (
        first_in_round,
        second_in_round,
    )


def test_empty_job_materializes_to_self_identifying_input() -> None:
    """A timing-only job produces an empty frozen input with only request identity fields."""
    request_marker = object()
    decoder_input = materialize_decoder_input(
        make_job([], op_id="operation", window_id="window", request_key=request_marker)
    )

    assert tuple(field.name for field in fields(DecoderInput)) == (
        "op_id",
        "window_id",
        "request_key",
        "rounds",
    )
    assert decoder_input.op_id == "operation"
    assert decoder_input.window_id == "window"
    assert decoder_input.request_key is request_marker
    assert decoder_input.rounds == ()
    assert not hasattr(decoder_input, "label")
    assert not hasattr(decoder_input, "round_count")
    with pytest.raises(FrozenInstanceError):
        decoder_input.rounds = ()


def test_materialization_rejects_a_structurally_compatible_fragment() -> None:
    """Exact payload admission rejects a structurally compatible noncanonical fragment."""
    sibling = CompatibleRetainedFragment(
        operation_id=1,
        patch_id="patch",
        round_index=0,
        bits=(0, 1),
        code="surface",
        size_bits=2,
        fragment_index=0,
    )

    with pytest.raises(TypeError, match="every job payload"):
        materialize_decoder_input(make_job([make_fragment(), sibling]))


def test_compatible_job_consumes_one_shot_payloads_once() -> None:
    """A compatible job consumes one-shot payloads once while enforcing exact admission."""
    fragment = make_fragment(operation_id=3, round_index=2)
    accepted_payloads = OneShotPayloads([fragment])
    accepted_job = CompatibleJob(
        op_id=3,
        window_id=8,
        request_key=None,
        payloads=accepted_payloads,
    )

    decoder_input = materialize_decoder_input(accepted_job)

    assert accepted_payloads.iterations == 1
    assert len(decoder_input.rounds) == 1
    assert decoder_input.rounds[0].fragments == (fragment,)

    rejected_payloads = OneShotPayloads(
        [
            CompatibleRetainedFragment(
                operation_id=3,
                patch_id="patch",
                round_index=2,
                bits=(0, 1),
                code="surface",
                size_bits=2,
                fragment_index=0,
            )
        ]
    )
    rejected_job = CompatibleJob(
        op_id=3,
        window_id=8,
        request_key=None,
        payloads=rejected_payloads,
    )
    with pytest.raises(TypeError, match="every job payload"):
        materialize_decoder_input(rejected_job)
    assert rejected_payloads.iterations == 1


def test_materialized_fragments_remain_safe_aliases_after_upstream_clear() -> None:
    """Frozen fragment aliases remain valid after the mutable upstream payload list is cleared."""
    fragment = make_fragment(bits=(1, 1, 0))
    job = make_job([fragment])
    decoder_input = materialize_decoder_input(job)

    job.payloads.clear()

    retained = decoder_input.rounds[0].fragments[0]
    assert retained is fragment
    assert retained.bits is fragment.bits
    assert retained.bits == (1, 1, 0)
    with pytest.raises(FrozenInstanceError):
        retained.bits = (0,)


@pytest.mark.parametrize("round_index", [True, 1.0, "1"])
def test_materialized_round_requires_exact_builtin_integer(round_index: object) -> None:
    """A materialized round rejects every nonexact built-in integer index."""
    with pytest.raises(TypeError, match="exact built-in int"):
        MaterializedSyndromeRound(
            operation_id=1,
            round_index=round_index,
            fragments=(make_fragment(1, 1),),
        )


def test_materialized_round_rejects_cross_type_operation_identity() -> None:
    """A materialized round compares operation identities without cross-type equality."""
    with pytest.raises(ValueError, match="share operation identity"):
        MaterializedSyndromeRound(
            operation_id=1,
            round_index=0,
            fragments=(make_fragment(True, 0),),
        )


def test_direct_values_accept_deleted_duplicate_and_round_agreement_checks() -> None:
    """Direct values retain duplicates and mismatched fragment rounds without redundant checks."""
    fragment = make_fragment(1, 99, 3, patch_id="same-patch")
    round_input = MaterializedSyndromeRound(
        operation_id=1,
        round_index=4,
        fragments=(fragment, fragment),
    )
    request_marker = object()

    decoder_input = DecoderInput(
        op_id="different-operation",
        window_id="window",
        request_key=request_marker,
        rounds=(round_input, round_input),
    )

    assert round_input.fragments == (fragment, fragment)
    assert tuple(item.round_index for item in decoder_input.rounds) == (4, 4)
    assert decoder_input.op_id == "different-operation"
    assert decoder_input.request_key is request_marker


def test_materialized_round_is_frozen_with_field_dependent_hashing() -> None:
    """Materialized rounds are frozen and hash only when all field values are hashable."""
    round_input = MaterializedSyndromeRound(
        operation_id=1,
        round_index=0,
        fragments=(make_fragment(),),
    )
    assert isinstance(hash(round_input), int)
    with pytest.raises(FrozenInstanceError):
        round_input.round_index = 1

    unhashable_round = MaterializedSyndromeRound(
        operation_id=[],
        round_index=0,
        fragments=(),
    )
    with pytest.raises(TypeError):
        hash(unhashable_round)


def test_materialization_rejects_identity_outside_stable_domain() -> None:
    """Canonical round ordering naturally rejects an unsupported operation identity."""
    with pytest.raises(TypeError):
        materialize_decoder_input(make_job([make_fragment(object(), 0)]))


@pytest.mark.parametrize("capacity", [0, -1, True, 1.0])
def test_memory_capacity_requires_none_or_positive_exact_integer(capacity: object) -> None:
    """Decoder-local capacity accepts only unbounded or positive exact integers."""
    with pytest.raises(TypeError, match="positive built-in int"):
        DecoderLocalMemory(capacity)


def test_unbounded_memory_tracks_reserved_and_deposited_slots() -> None:
    """Unbounded memory counts both reserved and deposited live slots."""
    memory = DecoderLocalMemory()
    memory.reserve("reserved")
    memory.reserve("deposited")
    memory.deposit("deposited", make_job())

    assert memory.capacity is None
    assert memory.slots_in_use == 2


@pytest.mark.parametrize("deposited", [False, True])
def test_duplicate_reserve_reports_current_state_without_mutation(deposited: bool) -> None:
    """Duplicate reservation reports the current slot state without changing occupancy."""
    memory = DecoderLocalMemory(2)
    memory.reserve("request")
    if deposited:
        memory.deposit("request", make_job())
    expected_state = "deposited" if deposited else "reserved"

    with pytest.raises(RuntimeError, match=rf"request.*{expected_state}"):
        memory.reserve("request")

    assert memory.slots_in_use == 1


def test_equal_slot_keys_collide_loudly_without_mutation() -> None:
    """Python-equal slot keys collide loudly rather than sharing a live slot."""
    memory = DecoderLocalMemory(2)
    memory.reserve(1)

    with pytest.raises(RuntimeError, match="already reserved"):
        memory.reserve(True)

    assert memory.slots_in_use == 1


def test_capacity_exhaustion_is_typed_and_releases_after_take() -> None:
    """A full store raises its typed error and accepts work after a successful release."""
    memory = DecoderLocalMemory(1)
    memory.reserve("first")
    stored = memory.deposit("first", make_job([make_fragment()]))

    with pytest.raises(
        DecoderLocalCapacityExhaustion,
        match="all 1 decoder-local slots",
    ) as error:
        memory.reserve("second")
    assert isinstance(error.value, RuntimeError)
    assert memory.slots_in_use == 1

    assert memory.take("first") is stored
    memory.reserve("second")
    assert memory.slots_in_use == 1


def test_deposit_rejects_overwrite_without_mutation() -> None:
    """A second deposit raises a state error and preserves the original input."""
    memory = DecoderLocalMemory()
    memory.reserve("request")
    original = memory.deposit("request", make_job([make_fragment(1, 0)]))

    with pytest.raises(RuntimeError, match="already holds a deposit"):
        memory.deposit("request", make_job([make_fragment(1, 1)]))

    assert memory.slots_in_use == 1
    assert memory.take("request") is original


def test_take_rejects_reserved_slot_without_mutation() -> None:
    """Taking before deposit raises a state error and leaves the reservation live."""
    memory = DecoderLocalMemory()
    memory.reserve("request")

    with pytest.raises(RuntimeError, match="before any deposit"):
        memory.take("request")

    assert memory.slots_in_use == 1
    memory.discard("request")
    assert memory.slots_in_use == 0


@pytest.mark.parametrize("action", ["deposit", "take"])
def test_unknown_deposit_and_take_raise_natural_key_error(action: str) -> None:
    """Unknown deposit and take operations expose the mapping's natural key error."""
    memory = DecoderLocalMemory()

    with pytest.raises(KeyError) as error:
        if action == "deposit":
            memory.deposit("missing", make_job())
        else:
            memory.take("missing")

    assert error.value.args == ("missing",)
    assert memory.slots_in_use == 0


def test_discard_rejects_unknown_key_and_preserves_live_slot() -> None:
    """Discarding an unknown key raises a state error without touching live slots."""
    memory = DecoderLocalMemory()
    memory.reserve("live")

    with pytest.raises(RuntimeError, match="discard of unknown"):
        memory.discard("missing")

    assert memory.slots_in_use == 1
    memory.discard("live")


@pytest.mark.parametrize("deposited", [False, True])
def test_discard_releases_reserved_and_deposited_slots(deposited: bool) -> None:
    """Discard frees either live slot state without returning its contents."""
    memory = DecoderLocalMemory(1)
    memory.reserve("request")
    if deposited:
        memory.deposit("request", make_job())

    assert memory.discard("request") is None
    assert memory.slots_in_use == 0
    memory.reserve("replacement")
    assert memory.slots_in_use == 1


def test_successful_deposit_and_take_return_one_identical_input() -> None:
    """A successful reserve-deposit-take lifecycle returns one identical frozen input."""
    memory = DecoderLocalMemory(1)
    memory.reserve("request")
    deposited = memory.deposit("request", make_job([make_fragment()]))

    taken = memory.take("request")

    assert taken is deposited
    assert memory.slots_in_use == 0


def test_failed_materialization_leaves_slot_reserved() -> None:
    """A failed materialization leaves its reserved slot unchanged and recoverable."""
    memory = DecoderLocalMemory(1)
    memory.reserve("request")

    with pytest.raises(TypeError):
        memory.deposit(
            "request",
            make_job(
                [
                    CompatibleRetainedFragment(
                        operation_id=1,
                        patch_id="patch",
                        round_index=0,
                        bits=(0,),
                        code="surface",
                        size_bits=1,
                        fragment_index=0,
                    )
                ]
            ),
        )

    assert memory.slots_in_use == 1
    with pytest.raises(RuntimeError, match="before any deposit"):
        memory.take("request")
    memory.discard("request")


def test_capacity_attribute_can_change_without_revalidating_live_slots() -> None:
    """Changing the public capacity does not retroactively validate existing occupancy."""
    memory = DecoderLocalMemory(2)
    memory.reserve("first")
    memory.reserve("second")

    memory.capacity = 1

    assert memory.capacity == 1
    assert memory.slots_in_use == 2
    with pytest.raises(DecoderLocalCapacityExhaustion):
        memory.reserve("third")
