from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError, dataclass, fields

import pytest

import decsim.decoder_input_store as decoder_input_store
from decsim.decoder_input_store import (
    DecoderInput,
    DecoderInputStore,
    DecoderInputStoreCapacityExhaustion,
    DecoderInputStoreConfig,
    DecoderInputStoreOverflowPolicy,
    DecoderInputStoreSnapshot,
    DecoderInputStoreUnsatisfiableDemand,
    MaterializedSyndromeRound,
    materialize_decoder_input,
)
from decsim.detector_error_model.fault_model_contracts import (
    GRAPHLIKE_FAULT_MODEL_REQUIRED,
    WindowErrorModel,
)
from decsim.detector_error_model.window_model_builders import (
    build_window_error_models,
)
from decsim.adapters.stim_device import StimDevice
from decsim.decoders import PerRoundDecoder
from decsim.message import DecodeJob, Operation, RetainedSyndromeFragment
from decsim.mwpm_decoder.decoder import PyMatchingDecoder
from decsim.rounds import FixedRounds
from decsim.run_spec import RunSpec


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
    bits: tuple[int, ...] | None = (0, 1),
) -> RetainedSyndromeFragment:
    return RetainedSyndromeFragment(
        operation_id=operation_id,
        patch_id=patch_id if patch_id is not None else f"patch-{fragment_index}",
        round_index=round_index,
        bits=bits,
        code="surface",
        size_bits=None if bits is None else len(bits),
        fragment_index=fragment_index,
    )


def make_job(
    payloads: list[object] | None = None,
    *,
    op_id: object = 41,
    window_id: object = 7,
    request_key: object | None = None,
    dem: object = None,
) -> DecodeJob:
    return DecodeJob(
        op_id=op_id,
        window_id=window_id,
        n_rounds=len(payloads or []),
        dem=dem,
        payloads=list(payloads or []),
        request_key=request_key,
    )


def test_module_imports_only_kept_dependencies_and_has_no_stale_helper() -> None:
    """The module keeps only its narrow dependency set and no retired helper."""
    tree = ast.parse(inspect.getsource(decoder_input_store))
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
        (0, "types", ("MappingProxyType",)),
        (0, "typing", ("Any", "Callable", "Mapping", "Optional")),
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
    assert not hasattr(decoder_input_store, "_rematerialized_fragment")


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


def test_real_window_model_accepts_matching_operation_round_layout() -> None:
    """A real model accepts matching operation, round order, and dense counts."""
    stim = pytest.importorskip("stim")
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.01,
    )
    window_model = build_window_error_models(
        circuit,
        [(1, 3, 3)],
        round_count=3,
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
    )[0]
    detector_count_by_round: dict[int, int] = {}
    for detector_id in window_model.detector_ids:
        round_index, position_in_round = window_model.defect_positions[detector_id]
        assert position_in_round == detector_count_by_round.get(round_index, 0)
        detector_count_by_round[round_index] = position_in_round + 1
    fragments = tuple(
        make_fragment(
            operation_id=9,
            round_index=round_index,
            bits=tuple(index % 2 for index in range(detector_count)),
        )
        for round_index, detector_count in detector_count_by_round.items()
    )

    decoder_input = materialize_decoder_input(
        make_job(list(reversed(fragments)), op_id=9, dem=window_model)
    )

    assert tuple(
        (round_input.operation_id, round_input.round_index)
        for round_input in decoder_input.rounds
    ) == tuple((9, round_index) for round_index in detector_count_by_round)
    assert sum(
        len(fragment.bits)
        for round_input in decoder_input.rounds
        for fragment in round_input.fragments
    ) == len(window_model.detector_ids)


@pytest.mark.parametrize(
    "store_config",
    [
        None,
        DecoderInputStoreConfig({"default": 8}),
        DecoderInputStoreConfig({"default": 16}),
    ],
    ids=["unset", "finite-tight", "finite-roomy"],
)
def test_real_stim_run_reaches_model_backed_materialization_and_completes(
    monkeypatch: pytest.MonkeyPatch,
    store_config: DecoderInputStoreConfig | None,
) -> None:
    """A real StimDevice run materializes a window model and completes."""
    stim = pytest.importorskip("stim")
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
        before_round_data_depolarization=0.01,
    )
    operation = Operation(
        id=1,
        name="repetition memory",
        qubits=(0,),
        patches=(0,),
        circuit=circuit,
    )
    model_backed_jobs = []
    original_materialize = decoder_input_store.materialize_decoder_input

    def record_model_backed_materialization(job):
        model = getattr(job, "dem", None)
        if model is not None:
            model_backed_jobs.append(job)
        return original_materialize(job)

    monkeypatch.setattr(
        decoder_input_store,
        "materialize_decoder_input",
        record_model_backed_materialization,
    )

    completed = RunSpec(
        ops=[operation],
        d=3,
        rounds_policy=FixedRounds(3),
        device=StimDevice(),
        decoder=PyMatchingDecoder(PerRoundDecoder(tau_us=0.1)),
        decoder_input_store=store_config,
        seed=7,
    ).build()

    assert completed.result.terminal_status == "complete"
    assert completed.result.event_queue_empty
    assert completed.result.execution_workload_complete
    assert model_backed_jobs
    assert all(isinstance(job.dem, WindowErrorModel) for job in model_backed_jobs)
    logical_result = completed.result.operation_results[0]
    assert logical_result.logical_observables == logical_result.observable_truth


def test_same_size_divergent_model_row_layout_order_raises() -> None:
    """A same-size round-layout permutation fails instead of moving syndrome bits."""
    window_model = WindowErrorModel(
        detector_ids=(20, 10),
        detector_coordinates=None,
        defect_positions={10: (1, 0), 20: (2, 0)},
        graphlike_faults=None,
        physical_faults=None,
    )
    payloads = [
        make_fragment(round_index=2, bits=(0,)),
        make_fragment(round_index=1, bits=(1,)),
    ]

    with pytest.raises(ValueError, match="row layout"):
        materialize_decoder_input(make_job(payloads, op_id=1, dem=window_model))


def test_successor_identity_cannot_replace_predecessor_model_rows() -> None:
    """A sorting-first successor cannot silently occupy predecessor model rows."""
    window_model = WindowErrorModel(
        detector_ids=(10, 20),
        detector_coordinates=None,
        defect_positions={10: (1, 0), 20: (2, 0)},
        graphlike_faults=None,
        physical_faults=None,
    )
    predecessor_fragment = make_fragment(
        operation_id=2,
        round_index=2,
        bits=(0,),
    )
    successor_fragment = make_fragment(
        operation_id=1,
        round_index=1,
        bits=(1,),
    )

    with pytest.raises(ValueError, match="round operation 1.*job operation 2"):
        materialize_decoder_input(
            make_job(
                [predecessor_fragment, successor_fragment],
                op_id=2,
                dem=window_model,
            )
        )


def test_unknown_model_detector_identity_raises() -> None:
    """A detector identity absent from defect positions raises loudly."""
    window_model = WindowErrorModel(
        detector_ids=(10, 99),
        detector_coordinates=None,
        defect_positions={10: (1, 0)},
        graphlike_faults=None,
        physical_faults=None,
    )

    with pytest.raises(KeyError) as error:
        materialize_decoder_input(
            make_job(
                [make_fragment(round_index=1, bits=(1, 0))],
                op_id=1,
                dem=window_model,
            )
        )

    assert error.value.args == (99,)


def test_detector_row_length_divergence_raises() -> None:
    """A model and materialized bit vector with different lengths fail loudly."""
    window_model = WindowErrorModel(
        detector_ids=(10,),
        detector_coordinates=None,
        defect_positions={10: (1, 0)},
        graphlike_faults=None,
        physical_faults=None,
    )

    with pytest.raises(ValueError, match="row layout"):
        materialize_decoder_input(
            make_job(
                [make_fragment(round_index=1, bits=(1, 0))],
                op_id=1,
                dem=window_model,
            )
        )


def test_custom_model_without_row_layout_materializes_unchanged() -> None:
    """A non-None custom model without row-layout attributes remains accepted."""
    class CustomModelWithoutRowLayout:
        """Represent a decoder model whose contract has no row-layout view."""

    custom_model = CustomModelWithoutRowLayout()
    fragment = make_fragment(operation_id=3, round_index=2, bits=(1, 0))

    decoder_input = materialize_decoder_input(
        make_job([fragment], op_id=3, dem=custom_model)
    )

    assert decoder_input.rounds[0].fragments == (fragment,)


def test_timing_only_fragments_need_no_detector_model_or_bits() -> None:
    """Timing-only fragments with no model and no bits still materialize."""
    fragment = make_fragment(round_index=3, bits=None)

    decoder_input = materialize_decoder_input(make_job([fragment], dem=None))

    assert decoder_input.rounds[0].fragments == (fragment,)
    assert decoder_input.rounds[0].fragments[0].bits is None


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

    assert not hasattr(accepted_job, "dem")
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
def test_store_capacity_requires_unbounded_or_positive_exact_integer(capacity: object) -> None:
    """Store capacity accepts only unbounded or positive exact integers."""
    with pytest.raises(TypeError, match="positive built-in int"):
        DecoderInputStore(pool="weak", round_capacity=capacity)


def test_store_pool_requires_a_string_label() -> None:
    """A store pool label must be an exact string-compatible label."""
    with pytest.raises(TypeError, match="str label"):
        DecoderInputStore(pool=1, round_capacity=None)


def test_unbounded_store_tracks_exact_round_credits_and_immutable_snapshot() -> None:
    """An unbounded store accounts for reserved rounds in immutable snapshots."""
    input_store = DecoderInputStore(pool="weak", round_capacity=None)
    input_store.reserve("reserved", 3)
    input_store.reserve("deposited", 1)
    input_store.deposit("deposited", make_job([make_fragment()]))

    snapshot = input_store.snapshot()
    assert snapshot == DecoderInputStoreSnapshot(
        pool="weak",
        capacity_rounds=None,
        occupied_rounds=4,
        peak_occupied_rounds=4,
        slots_in_use=2,
        admissions=2,
    )
    assert input_store.rounds_in_use == 4
    with pytest.raises(FrozenInstanceError):
        snapshot.occupied_rounds = 0


@pytest.mark.parametrize("deposited", [False, True])
def test_duplicate_reserve_reports_current_state_without_mutation(deposited: bool) -> None:
    """Duplicate reservation reports the current state without changing credits."""
    input_store = DecoderInputStore(pool="weak", round_capacity=2)
    input_store.reserve("request", 1)
    if deposited:
        input_store.deposit("request", make_job([make_fragment()]))
    expected_state = "deposited" if deposited else "reserved"
    before = input_store.snapshot()

    with pytest.raises(RuntimeError, match=rf"request.*{expected_state}"):
        input_store.reserve("request", 1)

    assert input_store.snapshot() == before


def test_equal_slot_keys_collide_loudly_without_mutation() -> None:
    """Python-equal keys collide loudly rather than sharing round credits."""
    input_store = DecoderInputStore(pool="weak", round_capacity=2)
    input_store.reserve(1, 1)
    before = input_store.snapshot()

    with pytest.raises(RuntimeError, match="already reserved"):
        input_store.reserve(True, 1)

    assert input_store.snapshot() == before


def test_round_credit_reserve_deposit_take_and_discard_are_exact() -> None:
    """Every lifecycle operation charges and returns exactly its request's rounds."""
    input_store = DecoderInputStore(pool="strong", round_capacity=5)
    first_job = make_job([make_fragment(round_index=0), make_fragment(round_index=1)])
    second_job = make_job([make_fragment(round_index=2)])
    input_store.reserve("first", 2)
    first_input = input_store.deposit("first", first_job)
    input_store.reserve("second", 1)
    input_store.deposit("second", second_job)
    assert input_store.snapshot().occupied_rounds == 3

    assert input_store.take("first") is first_input
    assert input_store.snapshot().occupied_rounds == 1
    input_store.discard("second")
    assert input_store.snapshot() == DecoderInputStoreSnapshot(
        pool="strong",
        capacity_rounds=5,
        occupied_rounds=0,
        peak_occupied_rounds=3,
        slots_in_use=0,
        admissions=2,
    )


def test_capacity_exhaustion_is_enriched_and_snapshot_is_frozen() -> None:
    """Aggregate overflow raises a typed diagnostic with its immutable snapshot."""
    input_store = DecoderInputStore(pool="strong", round_capacity=3)
    input_store.reserve("first", 2)
    before = input_store.snapshot()

    with pytest.raises(DecoderInputStoreCapacityExhaustion) as caught:
        input_store.reserve("second", 2)

    error = caught.value
    assert isinstance(error, RuntimeError)
    assert (error.pool, error.requested_rounds, error.capacity_rounds) == (
        "strong", 2, 3)
    assert error.snapshot == before
    assert input_store.snapshot() == before
    with pytest.raises(FrozenInstanceError):
        error.snapshot.occupied_rounds = 0


def test_unsatisfiable_demand_is_distinct_and_does_not_reserve() -> None:
    """A request larger than the whole budget fails distinctly without mutation."""
    input_store = DecoderInputStore(pool="weak", round_capacity=2)

    with pytest.raises(DecoderInputStoreUnsatisfiableDemand) as caught:
        input_store.reserve("too-large", 3)

    error = caught.value
    assert not isinstance(error, DecoderInputStoreCapacityExhaustion)
    assert (error.pool, error.requested_rounds, error.capacity_rounds) == (
        "weak", 3, 2)
    assert error.snapshot.occupied_rounds == 0
    assert input_store.snapshot().slots_in_use == 0


@pytest.mark.parametrize(
    ("round_demand", "error_type"),
    [(True, TypeError), (1.0, TypeError), (-1, ValueError)],
)
def test_round_demand_requires_a_nonnegative_exact_integer(
    round_demand: object, error_type: type[Exception]
) -> None:
    """Round reservations reject nonexact or negative demand without mutation."""
    input_store = DecoderInputStore(pool="weak", round_capacity=2)
    with pytest.raises(error_type):
        input_store.reserve("request", round_demand)
    assert input_store.snapshot().occupied_rounds == 0


def test_deposit_rejects_overwrite_without_mutation() -> None:
    """A second deposit preserves the original input and its credit charge."""
    input_store = DecoderInputStore(pool="weak", round_capacity=None)
    input_store.reserve("request", 1)
    original = input_store.deposit("request", make_job([make_fragment(1, 0)]))
    before = input_store.snapshot()

    with pytest.raises(RuntimeError, match="already holds a deposit"):
        input_store.deposit("request", make_job([make_fragment(1, 1)]))

    assert input_store.snapshot() == before
    assert input_store.take("request") is original


def test_take_rejects_reserved_slot_without_mutation() -> None:
    """Taking before deposit leaves both reservation and credits live."""
    input_store = DecoderInputStore(pool="weak", round_capacity=None)
    input_store.reserve("request", 2)
    before = input_store.snapshot()

    with pytest.raises(RuntimeError, match="before any deposit"):
        input_store.take("request")

    assert input_store.snapshot() == before
    input_store.discard("request")
    assert input_store.rounds_in_use == 0


@pytest.mark.parametrize("action", ["deposit", "take"])
def test_unknown_deposit_and_take_raise_natural_key_error(action: str) -> None:
    """Unknown deposit and take expose the mapping's natural key error."""
    input_store = DecoderInputStore(pool="weak", round_capacity=None)

    with pytest.raises(KeyError) as error:
        if action == "deposit":
            input_store.deposit("missing", make_job())
        else:
            input_store.take("missing")

    assert error.value.args == ("missing",)
    assert input_store.snapshot().occupied_rounds == 0


def test_discard_rejects_unknown_key_and_preserves_live_slot() -> None:
    """Discarding an unknown key leaves every live credit untouched."""
    input_store = DecoderInputStore(pool="weak", round_capacity=None)
    input_store.reserve("live", 2)
    before = input_store.snapshot()

    with pytest.raises(RuntimeError, match="discard of unknown"):
        input_store.discard("missing")

    assert input_store.snapshot() == before
    input_store.discard("live")


@pytest.mark.parametrize("deposited", [False, True])
def test_discard_releases_reserved_and_deposited_credits(deposited: bool) -> None:
    """Discard returns all rounds from either live slot state exactly once."""
    input_store = DecoderInputStore(pool="weak", round_capacity=2)
    input_store.reserve("request", 2)
    if deposited:
        input_store.deposit(
            "request",
            make_job([make_fragment(round_index=0), make_fragment(round_index=1)]),
        )

    assert input_store.discard("request") is None
    assert input_store.rounds_in_use == 0
    with pytest.raises(RuntimeError, match="discard of unknown"):
        input_store.discard("request")


def test_failed_materialization_keeps_reserved_credits_for_caller_unwind() -> None:
    """Failed materialization keeps its reservation until the caller unwinds it."""
    input_store = DecoderInputStore(pool="weak", round_capacity=1)
    input_store.reserve("request", 1)

    with pytest.raises(TypeError):
        input_store.deposit(
            "request",
            make_job([
                CompatibleRetainedFragment(
                    operation_id=1,
                    patch_id="patch",
                    round_index=0,
                    bits=(0,),
                    code="surface",
                    size_bits=1,
                    fragment_index=0,
                )
            ]),
        )

    assert input_store.snapshot().occupied_rounds == 1
    input_store.discard("request")
    assert input_store.snapshot().occupied_rounds == 0


def test_deposit_rejects_materialized_round_count_drift() -> None:
    """Deposit fails loudly if reserved demand differs from materialized rounds."""
    input_store = DecoderInputStore(pool="weak", round_capacity=3)
    input_store.reserve("request", 2)

    with pytest.raises(RuntimeError, match="reserved 2 rounds but materialized 1"):
        input_store.deposit("request", make_job([make_fragment(round_index=0)]))

    assert input_store.snapshot().occupied_rounds == 2
    input_store.discard("request")


def test_config_is_frozen_and_copies_its_mapping_read_only() -> None:
    """Configuration cannot change through its source, view, or frozen fields."""
    capacities = {"weak": 2, "strong": 5}
    config = DecoderInputStoreConfig(capacities)
    capacities["weak"] = 99

    assert dict(config.round_capacity_by_pool) == {"weak": 2, "strong": 5}
    with pytest.raises(TypeError):
        config.round_capacity_by_pool["weak"] = 3
    with pytest.raises(FrozenInstanceError):
        config.overflow_policy = DecoderInputStoreOverflowPolicy.FAIL_STOP


@pytest.mark.parametrize(
    "capacities",
    [
        {1: 2},
        {"weak": 0},
        {"weak": -1},
        {"weak": True},
        {"weak": 1.0},
    ],
)
def test_config_requires_string_pools_and_positive_exact_integer_budgets(
    capacities: object,
) -> None:
    """Finite configuration validates exact pool labels and round budgets."""
    with pytest.raises(TypeError):
        DecoderInputStoreConfig(capacities)


def test_config_rejects_duplicate_pool_entries_and_wrong_policy_type() -> None:
    """Duplicate pool entries and nonpolicy overflow values fail loudly."""
    with pytest.raises(ValueError, match="duplicate"):
        DecoderInputStoreConfig([("weak", 1), ("weak", 2)])
    with pytest.raises(TypeError, match="overflow_policy"):
        DecoderInputStoreConfig({"weak": 1}, overflow_policy="stall")
