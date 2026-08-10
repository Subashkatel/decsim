"""Deterministic gates for the Skoric block A/B dependency runtime."""
from __future__ import annotations

import numpy as np
import pymatching
import pytest
import stim

from decsim.adapters.stim_device import StimDevice
from decsim.adapters.window_decode_results import result_from_selected_faults
from decsim.codes import SurfaceCodeModel
from decsim.detector_error_model import (
    FaultRepresentation,
    GRAPHLIKE_FAULT_MODEL_REQUIRED,
    LINKED_FAULT_MODELS_REQUIRED,
    NO_FAULT_MODEL_REQUIRED,
    build_window_error_models,
)
from decsim.devices import TimingOnlyDevice
from decsim.message import (
    DecodeJob,
    DecodeResult,
    DependencyResidual,
    Operation,
    SyndromePayload,
)
from decsim.mwpm_decoder import PyMatchingDecoder
from decsim.planner import FixedRounds
from decsim.run_spec import RunSpec, simulate
from decsim.schemes import ParallelWindowScheme


class _OneTick:
    def latency(self, job):
        return 1


def _memory_circuit(distance=3, rounds=18, probability=0.003):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=probability,
        after_reset_flip_probability=probability,
        before_measure_flip_probability=probability,
        before_round_data_depolarization=probability,
    )


def _detector_rounds(circuit, round_count):
    raw = {
        detector_id: int(circuit.get_detector_coordinates()[detector_id][-1])
        for detector_id in range(circuit.num_detectors)
    }
    return {
        detector_id: round_count if layer == round_count else layer + 1
        for detector_id, layer in raw.items()
    }


def test_right_A_owns_leading_seam_fault_and_emits_backward_residual():
    distance, rounds = 3, 18
    circuit = _memory_circuit(distance, rounds)
    round_of = _detector_rounds(circuit, rounds)
    plan = ParallelWindowScheme().plan_operation(
        0,
        rounds,
        commit_round_count=distance,
        buffer_round_count=distance,
    )
    entries = [
        (window.buffer_lo, window.commit_lo, window.commit_hi, window.buffer_hi)
        for window in plan.windows
    ]
    models = build_window_error_models(
        circuit,
        entries,
        round_count=rounds,
        detector_rounds=round_of,
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
        dependency_edges=plan.internal_dependencies,
        closed_temporal_boundary_windows=tuple(
            window_index
            for window_index, window in enumerate(plan.windows)
            if window.closed_temporal_boundaries
        ),
    )
    b_model = models[1]
    right_a_model = models[2]
    b_detector_ids = set(b_model.detector_ids)
    right_a_faults = right_a_model.require_faults(FaultRepresentation.GRAPHLIKE)

    crossing_column = next(
        column_index
        for column_index, owned in enumerate(right_a_faults.owned)
        if owned
        and set(right_a_faults.boundary_flips[column_index]) & b_detector_ids
    )
    source_fault_id = right_a_faults.source_fault_ids[crossing_column]
    b_faults = b_model.require_faults(FaultRepresentation.GRAPHLIKE)
    assert source_fault_id not in b_faults.source_fault_ids

    selected = np.zeros(right_a_faults.check.shape[1], dtype=np.uint8)
    selected[crossing_column] = 1
    result = result_from_selected_faults(
        DecodeJob(0, 2, right_a_model.commit_hi - right_a_model.buffer_lo + 1,
                  dem=right_a_model, label="right A"),
        right_a_model,
        right_a_faults,
        selected,
    )

    assert result.boundary_defects is None  # no forward endpoint at the final A
    assert isinstance(result.boundary_data, DependencyResidual)
    assert 15 in result.boundary_data.defects
    assert any(result.boundary_data.defects[15])


def test_parallel_partition_covers_both_fault_domains_and_both_A_edges():
    distance, rounds = 3, 18
    circuit = _memory_circuit(distance, rounds)
    round_of = _detector_rounds(circuit, rounds)
    plan = ParallelWindowScheme().plan_operation(
        0, rounds, commit_round_count=distance, buffer_round_count=distance)
    entries = [
        (window.buffer_lo, window.commit_lo, window.commit_hi, window.buffer_hi)
        for window in plan.windows
    ]
    models = build_window_error_models(
        circuit,
        entries,
        round_count=rounds,
        detector_rounds=round_of,
        fault_model_requirement=LINKED_FAULT_MODELS_REQUIRED,
        fault_exclusion_ranges=(),
        dependency_edges=plan.internal_dependencies,
        closed_temporal_boundary_windows=tuple(
            window_index
            for window_index, window in enumerate(plan.windows)
            if window.closed_temporal_boundaries
        ),
    )

    for model in models:
        assert set(model.detector_ids) <= set(model.defect_positions)

    for representation in (
        FaultRepresentation.GRAPHLIKE,
        FaultRepresentation.PHYSICAL,
    ):
        placed_ids = {
            source_fault_id
            for model in models
            for source_fault_id in model.require_faults(
                representation).source_fault_ids
        }
        assert placed_ids == set(range(max(placed_ids) + 1))
        owned_ids = [
            source_fault_id
            for model in models
            for faults in [model.require_faults(representation)]
            for source_fault_id, owned in zip(
                faults.source_fault_ids,
                faults.owned,
            )
            if owned
        ]
        assert set(owned_ids) == placed_ids
        assert len(owned_ids) == len(set(owned_ids))
        for source_index, destination_index in plan.internal_dependencies:
            source = models[source_index].require_faults(representation)
            destination_faults = models[destination_index].require_faults(
                representation
            )
            source_owned_ids = {
                source_fault_id
                for source_fault_id, owned in zip(
                    source.source_fault_ids,
                    source.owned,
                )
                if owned
            }
            assert source_owned_ids.isdisjoint(
                destination_faults.source_fault_ids
            )
            destination_ids = set(models[destination_index].detector_ids)
            applicable = {
                detector_id
                for column_index, owned in enumerate(source.owned)
                if owned
                for detector_id in source.boundary_flips.get(column_index, ())
            } & destination_ids
            assert applicable, (
                representation,
                source_index,
                destination_index,
            )


class _FixedRoundDevice(TimingOnlyDevice):
    def __init__(self, raw_by_round):
        self.raw_by_round = dict(raw_by_round)

    def round_payloads(self, op, round_index):
        return [SyndromePayload(
            op.id,
            op.qubits[0],
            round_index,
            bits=self.raw_by_round.get(round_index, (0, 0)),
        )]


class _ABDecoder:
    fault_model_requirement = NO_FAULT_MODEL_REQUIRED

    def __init__(self, window_id, latency, residual=None, logical=None):
        self.window_id = window_id
        self.decode_latency = latency
        self.residual = residual or {}
        self.logical = (window_id & 1,) if logical is None else tuple(logical)
        self.jobs = []

    def latency(self, job):
        assert job.window_id == self.window_id
        return self.decode_latency

    def decode(self, job):
        self.jobs.append(job)
        return DecodeResult(
            job.op_id,
            job.window_id,
            logical_observables=self.logical,
            boundary_data=DependencyResidual(defects=dict(self.residual)),
        )


class _ABRouter:
    def __init__(self, decoders):
        self.decoders = decoders

    def route(self, job):
        return self.decoders[job.window_id]

    def fault_model_requirement_for(self, code):
        return NO_FAULT_MODEL_REQUIRED


@pytest.mark.parametrize(
    ("left_latency", "right_latency", "expected_first"),
    [(1, 20_000_000, 0), (20_000_000, 1, 2)],
)
def test_B_waits_for_both_A_residuals_and_decodes_raw_xor_left_xor_right(
    left_latency,
    right_latency,
    expected_first,
):
    raw = {10: (1, 0)}
    decoders = {
        0: _ABDecoder(0, left_latency, {10: [1, 1]}),
        1: _ABDecoder(1, 1),
        2: _ABDecoder(2, right_latency, {10: [1, 0]}),
    }
    operation = Operation(0, "memory", (0,), clifford=True)
    run = simulate(RunSpec(
        ops=[operation],
        d=3,
        rounds_policy=FixedRounds(18),
        round_us=1.1,
        scheme=ParallelWindowScheme(),
        device=_FixedRoundDevice(raw),
        router=_ABRouter(decoders),
        num_units=3,
    ), verbose=False)

    a0 = run.window_manager.windows[(0, 0)]
    b0 = run.window_manager.windows[(0, 1)]
    a1 = run.window_manager.windows[(0, 2)]
    assert b0.t_dispatch >= max(a0.t_done, a1.t_done)
    assert b0.deps_remaining == 0
    assert len(decoders[1].jobs) == 1
    payload_by_round = {
        payload.round_index: tuple(payload.bits)
        for payload in decoders[1].jobs[0].payloads
    }
    assert payload_by_round[10] == (1, 1)
    assert {
        source[1]
        for source in b0.boundary_in.contributions
    } == {0, 2}
    assert run.window_manager._released_boundary_dependencies == {
        ((0, 0), (0, 1)),
        ((0, 2), (0, 1)),
    }
    first_done = 0 if a0.t_done < a1.t_done else 2
    assert first_done == expected_first


def test_fanout_A_logical_correction_is_committed_once_not_per_delivery():
    decoders = {
        0: _ABDecoder(0, 1, logical=(0,)),
        1: _ABDecoder(1, 1, logical=(0,)),
        2: _ABDecoder(2, 1, logical=(1,)),
        3: _ABDecoder(3, 1, logical=(0,)),
    }
    operation = Operation(0, "memory", (0,), clifford=True)
    run = simulate(RunSpec(
        ops=[operation],
        d=3,
        rounds_policy=FixedRounds(23),
        round_us=1.1,
        scheme=ParallelWindowScheme(),
        device=_FixedRoundDevice({}),
        router=_ABRouter(decoders),
        num_units=4,
    ), verbose=False)

    assert run.window_manager.windows[(0, 2)].dependents == [(0, 1), (0, 3)]
    assert run.window_manager.op_results[0] == (1,)
    assert set(run.window_manager.logical_contributions) == {
        (0, 0), (0, 1), (0, 2), (0, 3),
    }
    assert all(len(decoder.jobs) == 1 for decoder in decoders.values())


class _SentinelDecoderFailure(RuntimeError):
    pass


class _FailingABDecoder(_ABDecoder):
    def decode(self, job):
        self.jobs.append(job)
        raise _SentinelDecoderFailure(f"window {job.window_id} failed")


@pytest.mark.parametrize("failing_window", [0, 2, 1])
def test_parallel_decoder_failure_propagates_without_fabricating_join(
    failing_window,
):
    decoders = {
        window_id: (
            _FailingABDecoder(window_id, 1)
            if window_id == failing_window
            else _ABDecoder(window_id, 1)
        )
        for window_id in range(3)
    }
    operation = Operation(0, "memory", (0,), clifford=True)
    with pytest.raises(
        _SentinelDecoderFailure,
        match=f"window {failing_window} failed",
    ):
        simulate(RunSpec(
            ops=[operation],
            d=3,
            rounds_policy=FixedRounds(18),
            round_us=1.1,
            scheme=ParallelWindowScheme(),
            device=_FixedRoundDevice({}),
            router=_ABRouter(decoders),
            num_units=3,
        ), verbose=False)

    if failing_window in (0, 2):
        assert not decoders[1].jobs


class _ForceRightASeamDecoder:
    fault_model_requirement = GRAPHLIKE_FAULT_MODEL_REQUIRED

    def __init__(self, *, both_sides=False):
        self.both_sides = both_sides
        self.jobs = {}
        self.results = {}

    def latency(self, job):
        return 1

    def decode(self, job):
        self.jobs[job.window_id] = job
        faults = job.dem.require_faults(FaultRepresentation.GRAPHLIKE)
        selected = np.zeros(faults.check.shape[1], dtype=np.uint8)
        if job.window_id == 2:
            leading = next(
                column_index
                for column_index, owned in enumerate(faults.owned)
                if owned
                and any(
                    job.dem.defect_positions[detector_id][0] < job.dem.commit_lo
                    for detector_id in faults.boundary_flips[column_index]
                )
            )
            selected[leading] = 1
            if self.both_sides:
                trailing = next(
                    column_index
                    for column_index, owned in enumerate(faults.owned)
                    if owned
                    and column_index != leading
                    and any(
                        job.dem.defect_positions[detector_id][0] > job.dem.commit_hi
                        for detector_id in faults.boundary_flips[column_index]
                    )
                )
                selected[trailing] = 1
        result = result_from_selected_faults(job, job.dem, faults, selected)
        self.results[job.window_id] = result
        return result


def test_runtime_delivers_forced_right_A_correction_by_global_detector_identity():
    distance, rounds = 3, 18
    circuit = _memory_circuit(distance, rounds)
    syndrome = np.zeros(circuit.num_detectors, dtype=np.bool_)
    decoder = _ForceRightASeamDecoder()
    operation = Operation(1, "memory", (0,), clifford=True, circuit=circuit)
    run = simulate(RunSpec(
        ops=[operation],
        num_units=3,
        rounds_policy=FixedRounds(rounds),
        code=SurfaceCodeModel(d=distance),
        scheme=ParallelWindowScheme(),
        device=_FixedSyndromeDevice(syndrome),
        decoder=decoder,
        seed=None,
    ), verbose=False)

    assert set(decoder.jobs) == {0, 1, 2}
    assert decoder.results[2].boundary_defects is None
    residual = decoder.results[2].boundary_data
    destination_positions = decoder.jobs[1].dem.defect_positions
    expected_ids = set(residual.detector_ids) & set(destination_positions)
    assert expected_ids
    assert {
        destination_positions[detector_id][0]
        for detector_id in expected_ids
    } == {15}
    b_round_15 = next(
        payload
        for payload in decoder.jobs[1].payloads
        if payload.round_index == 15
    )
    assert sum(b_round_15.bits) % 2 == 1
    assert run.window_manager._released_boundary_dependencies == {
        ((1, 0), (1, 1)),
        ((1, 2), (1, 1)),
    }


def test_one_A_full_residual_is_intersected_separately_by_left_and_right_B():
    distance, rounds = 3, 23
    circuit = _memory_circuit(distance, rounds)
    syndrome = np.zeros(circuit.num_detectors, dtype=np.bool_)
    decoder = _ForceRightASeamDecoder(both_sides=True)
    operation = Operation(1, "memory", (0,), clifford=True, circuit=circuit)
    run = simulate(RunSpec(
        ops=[operation],
        num_units=4,
        rounds_policy=FixedRounds(rounds),
        code=SurfaceCodeModel(d=distance),
        scheme=ParallelWindowScheme(),
        device=_FixedSyndromeDevice(syndrome),
        decoder=decoder,
        seed=None,
    ), verbose=False)

    assert set(decoder.jobs) == {0, 1, 2, 3}
    residual_ids = set(decoder.results[2].boundary_data.detector_ids)
    left_positions = decoder.jobs[1].dem.defect_positions
    right_positions = decoder.jobs[3].dem.defect_positions
    left_ids = residual_ids & set(left_positions)
    right_ids = residual_ids & set(right_positions)
    assert left_ids and right_ids and left_ids.isdisjoint(right_ids)
    assert {left_positions[detector_id][0] for detector_id in left_ids} == {15}
    assert {right_positions[detector_id][0] for detector_id in right_ids} == {19}

    left_payloads = {
        payload.round_index: tuple(payload.bits)
        for payload in decoder.jobs[1].payloads
    }
    right_payloads = {
        payload.round_index: tuple(payload.bits)
        for payload in decoder.jobs[3].payloads
    }
    assert sum(left_payloads[15]) % 2 == 1
    assert sum(right_payloads[19]) % 2 == 1
    assert all(not any(bits) for round_index, bits in left_payloads.items()
               if round_index != 15)
    assert all(not any(bits) for round_index, bits in right_payloads.items()
               if round_index != 19)
    assert ((1, 2), (1, 1)) in run.window_manager._released_boundary_dependencies
    assert ((1, 2), (1, 3)) in run.window_manager._released_boundary_dependencies


class _FixedSyndromeDevice(StimDevice):
    def __init__(self, syndrome):
        super().__init__()
        self.syndrome = np.asarray(syndrome, dtype=np.bool_)

    def begin_operation(self, op, segment_round_count, source_round_count):
        super().begin_operation(op, segment_round_count, source_round_count)
        key = self._key(op)
        self._dets[key] = self.syndrome.copy()
        self._dets[op.id] = self._dets[key]


def test_repaired_parallel_runtime_matches_global_on_frozen_seam_counterexample():
    distance, rounds = 3, 12
    circuit = _memory_circuit(distance, rounds)
    syndrome = np.zeros(circuit.num_detectors, dtype=np.bool_)
    syndrome[[18, 21, 26]] = 1
    device = _FixedSyndromeDevice(syndrome)
    operation = Operation(1, "memory", (0,), clifford=True, circuit=circuit)
    run = simulate(RunSpec(
        ops=[operation],
        num_units=4,
        rounds_policy=FixedRounds(rounds),
        code=SurfaceCodeModel(d=distance),
        scheme=ParallelWindowScheme(),
        device=device,
        decoder=PyMatchingDecoder(_OneTick()),
        seed=None,
    ), verbose=False)

    global_matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True)
    )
    expected = tuple(int(bit) for bit in global_matching.decode(syndrome))
    assert expected == (1,)
    assert run.window_manager.op_results[1] == expected
