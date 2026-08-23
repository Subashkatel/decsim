from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError, dataclass, fields

import pytest

import decsim.decoders.decoder_memory as decoder_memory
from decsim.decoders.decoder_memory import (
    DecoderInput,
    DecoderMemory,
    DecoderMemoryCapacityExhaustion,
    DecoderMemoryConfig,
    DecoderMemorySnapshot,
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
from decsim.qpu.stim_device import StimDevice
from decsim.decoders.decoders import PerRoundDecoder
from decsim.message import DecodeJob, Operation, RetainedSyndromeFragment
from decsim.decoders.mwpm.decoder import PyMatchingDecoder
from decsim.qpu.round_policies import FixedRounds
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
    tree = ast.parse(inspect.getsource(decoder_memory))
    imports = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            imports.append(
                (node.level, node.module, tuple(alias.name for alias in node.names))
            )
    assert imports == [
        (0, "__future__", ("annotations",)),
        (0, "dataclasses", ("dataclass",)),
        (0, "types", ("MappingProxyType",)),
        (0, "typing", ("Any", "Mapping", "Optional")),
        (
            2,
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
    assert not hasattr(decoder_memory, "_rematerialized_fragment")


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
        DecoderMemoryConfig({"default": 8}),
        DecoderMemoryConfig({"default": 16}),
    ],
    ids=["unset", "finite-tight", "finite-roomy"],
)
def test_real_stim_run_reaches_model_backed_materialization_and_completes(
    monkeypatch: pytest.MonkeyPatch,
    store_config: DecoderMemoryConfig | None,
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
    original_materialize = decoder_memory.materialize_decoder_input

    def record_model_backed_materialization(job):
        model = getattr(job, "dem", None)
        if model is not None:
            model_backed_jobs.append(job)
        return original_materialize(job)

    monkeypatch.setattr(
        decoder_memory,
        "materialize_decoder_input",
        record_model_backed_materialization,
    )

    completed = RunSpec(
        ops=[operation],
        d=3,
        rounds_policy=FixedRounds(3),
        device=StimDevice(),
        decoder=PyMatchingDecoder(PerRoundDecoder(tau_us=0.1)),
        decoder_memory=store_config,
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


def test_compatible_job_consumes_one_shot_payloads_once() -> None:
    """A compatible job consumes one-shot payloads once."""
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


# ---------------------------------------------------------------- per-unit memory

def _unit_job(label, rounds, request_key=None):
    payloads = [make_fragment(round_index=r, bits=(0, 1)) for r in rounds]
    job = make_job(payloads, op_id=1, window_id=len(rounds), request_key=request_key)
    job.label = label
    return job


def test_config_is_rounds_per_unit_by_pool_and_absent_pools_are_unbounded() -> None:
    config = DecoderMemoryConfig({"default": 6})
    assert config.capacity_for("default") == 6
    assert config.capacity_for("strong") is None
    with pytest.raises(ValueError):
        DecoderMemoryConfig({"default": 0})


def test_unit_memory_holds_one_job_input_and_frees_it_exactly() -> None:
    memory = DecoderMemory("default", 0, capacity_rounds=4)
    job = _unit_job("w0", (0, 1, 2))
    decoder_input = memory.deposit(job)
    assert isinstance(decoder_input, DecoderInput)
    assert memory.occupied_rounds == 3 and memory.peak_occupied_rounds == 3
    assert memory.snapshot() == DecoderMemorySnapshot("default", 0, 4, 3, 3, 1)
    with pytest.raises(RuntimeError, match="already holds"):
        memory.deposit(job)
    memory.take(job)
    assert memory.occupied_rounds == 0
    memory.take(job)                       # a job it no longer holds is ignored


def test_window_larger_than_the_unit_memory_stops_the_run() -> None:
    memory = DecoderMemory("default", 1, capacity_rounds=2)
    with pytest.raises(DecoderMemoryCapacityExhaustion) as caught:
        memory.deposit(_unit_job("big", (0, 1, 2)))
    assert (caught.value.pool, caught.value.unit, caught.value.requested_rounds,
            caught.value.capacity_rounds) == ("default", 1, 3, 2)
    assert memory.occupied_rounds == 0


def test_run_gives_every_unit_its_own_memory_and_frees_it_at_completion() -> None:
    from decsim.decoders.decoders import PerRoundDecoder
    from decsim.message import Operation
    from decsim.qpu.round_policies import FixedRounds
    from decsim.run_spec import RunSpec

    operations = [Operation(id=i, name=f"op {i}", qubits=(i,), patches=(i,)) for i in (1, 2)]
    completed = RunSpec(ops=operations, d=3, rounds_policy=FixedRounds(3),
                        decoder=PerRoundDecoder(tau_us=1.0), num_units=2,
                        decoder_memory=DecoderMemoryConfig({"default": 6})).build()
    memories = completed.decoder_manager.decoder_memories
    assert sorted(memories) == [("default", 0), ("default", 1)]
    assert all(m.occupied_rounds == 0 for m in memories.values())
    assert sum(m.admissions for m in memories.values()) >= 2
    completed.decoder_manager.check_decode_work_settled()


def test_a_unit_too_small_for_its_window_fails_the_run_loudly() -> None:
    from decsim.decoders.decoders import PerRoundDecoder
    from decsim.message import Operation
    from decsim.qpu.round_policies import FixedRounds
    from decsim.run_spec import RunSpec

    with pytest.raises(DecoderMemoryCapacityExhaustion):
        RunSpec(ops=[Operation(id=1, name="op", qubits=(1,), patches=(1,))], d=3,
                rounds_policy=FixedRounds(3), decoder=PerRoundDecoder(tau_us=1.0),
                decoder_memory=DecoderMemoryConfig({"default": 1})).build()
