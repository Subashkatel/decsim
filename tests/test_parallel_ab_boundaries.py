"""Independent graph-theoretic gates for Skoric A/B time boundaries."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import stim

from decsim.adapters.stim_device import StimDevice
from decsim.codes import SurfaceCodeModel
from decsim.detector_error_model import (
    FaultRepresentation,
    GRAPHLIKE_FAULT_MODEL_REQUIRED,
    build_window_error_models,
    detector_error_model_to_faults,
)
from decsim.message import Operation
from decsim.mwpm_decoder import PyMatchingDecoder
from decsim.planner import FixedRounds
from decsim.run_spec import RunSpec, simulate
from decsim.schemes import ParallelWindowScheme


class _OneTick:
    def latency(self, job):
        return 1


@dataclass(frozen=True)
class _ABModels:
    circuit: object
    round_count: int
    round_of: dict
    plan: object
    global_detector_sets: tuple
    global_observable_sets: tuple
    global_priors: tuple
    models: tuple


def _memory_circuit(distance, rounds, probability=0.003):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=probability,
        after_reset_flip_probability=probability,
        before_measure_flip_probability=probability,
        before_round_data_depolarization=probability,
    )


def _rounds_from_stim_coordinates(circuit, round_count):
    raw = {
        detector_id: int(circuit.get_detector_coordinates()[detector_id][-1])
        for detector_id in range(circuit.num_detectors)
    }
    return {
        detector_id: round_count if layer == round_count else layer + 1
        for detector_id, layer in raw.items()
    }


def _build_ab_models(distance, rounds):
    circuit = _memory_circuit(distance, rounds)
    round_of = _rounds_from_stim_coordinates(circuit, rounds)
    plan = ParallelWindowScheme().plan_operation(
        0,
        rounds,
        commit_round_count=distance,
        buffer_round_count=distance,
    )
    entries = tuple(
        (window.buffer_lo, window.commit_lo, window.commit_hi, window.buffer_hi)
        for window in plan.windows
    )
    detector_sets, observable_sets, priors = detector_error_model_to_faults(
        circuit.detector_error_model(decompose_errors=True)
    )
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
    return _ABModels(
        circuit=circuit,
        round_count=rounds,
        round_of=round_of,
        plan=plan,
        global_detector_sets=tuple(tuple(ids) for ids in detector_sets),
        global_observable_sets=tuple(tuple(ids) for ids in observable_sets),
        global_priors=tuple(priors),
        models=tuple(models),
    )


class _SevenFaultCircuit:
    """Minimal exact DEM oracle with physical and artificial boundary edges."""

    num_detectors = 6
    num_observables = 1

    def __init__(self):
        self.dem = stim.DetectorErrorModel("""
            error(0.01) D0 L0
            error(0.02) D0 D1
            error(0.03) D1 D2
            error(0.04) D2 D3
            error(0.05) D3 D4
            error(0.06) D4 D5
            error(0.07) D5 L0
        """)

    def detector_error_model(self, *, decompose_errors):
        return self.dem

    def get_detector_coordinates(self):
        return {detector_id: [float(detector_id), 0.0]
                for detector_id in range(self.num_detectors)}


def _seven_fault_models(dependency_edges=None):
    circuit = _SevenFaultCircuit()
    plan = ParallelWindowScheme().plan_operation(
        0, 6, commit_round_count=1, buffer_round_count=1)
    edges = (
        plan.internal_dependencies
        if dependency_edges is None
        else dependency_edges
    )
    entries = tuple(
        (window.buffer_lo, window.commit_lo, window.commit_hi, window.buffer_hi)
        for window in plan.windows
    )
    models = build_window_error_models(
        circuit,
        entries,
        round_count=6,
        detector_rounds={detector_id: detector_id + 1
                         for detector_id in range(6)},
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
        dependency_edges=edges,
        closed_temporal_boundary_windows=(1,),
    )
    return circuit, plan, tuple(models)


def _artificial_temporal_source_ids(circuit, model):
    global_sets, _, _ = detector_error_model_to_faults(
        circuit.detector_error_model(decompose_errors=True))
    local_ids = set(model.detector_ids)
    return {
        source_fault_id
        for column_index, source_fault_id in enumerate(
            model.graphlike_faults.source_fault_ids
        )
        if len(global_sets[source_fault_id]) == 2
        and int(model.graphlike_faults.check[:, column_index].sum()) == 1
        and set(global_sets[source_fault_id]) - local_ids
    }


def test_seven_fault_oracle_pins_exact_A_open_and_B_closed_graphs():
    circuit, plan, models = _seven_fault_models()
    assert plan.internal_dependencies == ((0, 1), (2, 1))
    expected = (
        {
            "detectors": (0, 1, 2),
            "sources": (0, 1, 2, 3),
            "owned": (1, 1, 1, 0),
            "check": ((1, 1, 0, 0),
                      (0, 1, 1, 0),
                      (0, 0, 1, 1)),
            "artificial": {3},
            "boundary_rows": {0, 2},
        },
        {
            "detectors": (2, 3, 4),
            "sources": (3, 4),
            "owned": (1, 1),
            "check": ((1, 0),
                      (1, 1),
                      (0, 1)),
            "artificial": set(),
            "boundary_rows": set(),
        },
        {
            "detectors": (4, 5),
            "sources": (4, 5, 6),
            "owned": (0, 1, 1),
            "check": ((1, 1, 0),
                      (0, 1, 1)),
            "artificial": {4},
            "boundary_rows": {0, 1},
        },
    )
    for model, oracle in zip(models, expected):
        faults = model.graphlike_faults
        assert model.detector_ids == oracle["detectors"]
        assert faults.source_fault_ids == oracle["sources"]
        assert tuple(int(bit) for bit in faults.owned) == oracle["owned"]
        assert tuple(tuple(int(bit) for bit in row) for row in faults.check) == (
            oracle["check"]
        )
        assert _artificial_temporal_source_ids(circuit, model) == (
            oracle["artificial"]
        )
        matching = PyMatchingDecoder(_OneTick())._matching_for_model(faults)
        actual_boundary_rows = {
            node
            for left, right, _ in matching.edges()
            for node, other in ((left, right), (right, left))
            if other in matching.boundary and node not in matching.boundary
        }
        assert actual_boundary_rows == oracle["boundary_rows"]


@pytest.mark.parametrize(
    ("edges", "fault_id"),
    [(((0, 1),), 5), (((2, 1),), 2)],
)
def test_seven_fault_oracle_rejects_each_missing_B_dependency(edges, fault_id):
    with pytest.raises(
        ValueError,
        match=(
            rf"closed temporal boundary window 1 truncates global fault "
            rf"{fault_id}"
        ),
    ):
        _seven_fault_models(edges)


class _NonlocalFaultCircuit:
    num_detectors = 10
    num_observables = 0

    def detector_error_model(self, *, decompose_errors):
        return stim.DetectorErrorModel("error(0.1) D3 D9")

    def get_detector_coordinates(self):
        return {detector_id: [float(detector_id), 0.0]
                for detector_id in range(self.num_detectors)}


def test_nonlocal_fault_that_would_open_B_is_rejected_fail_closed():
    circuit = _NonlocalFaultCircuit()
    plan = ParallelWindowScheme().plan_operation(
        0, 10, commit_round_count=1, buffer_round_count=1)
    entries = tuple(
        (window.buffer_lo, window.commit_lo, window.commit_hi, window.buffer_hi)
        for window in plan.windows
    )
    assert plan.internal_dependencies == (
        (0, 1), (2, 1), (2, 3), (4, 3))
    with pytest.raises(
        ValueError,
        match=(
            r"closed temporal boundary window 1 truncates global fault 0"
        ),
    ):
        build_window_error_models(
            circuit,
            entries,
            round_count=10,
            detector_rounds={detector_id: detector_id + 1
                             for detector_id in range(10)},
            fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
            fault_exclusion_ranges=(),
            dependency_edges=plan.internal_dependencies,
            closed_temporal_boundary_windows=(1, 3),
        )


def _independent_dag_depths(window_count, edges):
    incoming = [set() for _ in range(window_count)]
    for source, destination in edges:
        incoming[destination].add(source)
    depths = [None] * window_count
    while any(depth is None for depth in depths):
        ready = [
            index
            for index, predecessors in enumerate(incoming)
            if depths[index] is None
            and all(depths[source] is not None for source in predecessors)
        ]
        assert ready, "independent oracle received a cyclic graph"
        for index in ready:
            depths[index] = (
                0 if not incoming[index]
                else 1 + max(depths[source] for source in incoming[index])
            )
    return tuple(depths), incoming


def _independent_owner_and_ancestors(data):
    windows = data.plan.windows
    depths, incoming = _independent_dag_depths(
        len(windows), data.plan.internal_dependencies)
    ancestors = [set() for _ in windows]
    for destination in sorted(range(len(windows)), key=depths.__getitem__):
        for source in incoming[destination]:
            ancestors[destination].add(source)
            ancestors[destination].update(ancestors[source])

    owners = []
    for source_fault_id, detector_ids in enumerate(data.global_detector_sets):
        candidates = [
            window_index
            for window_index, window in enumerate(windows)
            if any(
                window.commit_lo <= data.round_of[detector_id] <= window.commit_hi
                for detector_id in detector_ids
            )
        ]
        assert candidates, source_fault_id
        minimum_depth = min(depths[index] for index in candidates)
        causal_candidates = [
            index for index in candidates if depths[index] == minimum_depth
        ]
        assert len(causal_candidates) == 1, (source_fault_id, candidates)
        owners.append(causal_candidates[0])
    return tuple(owners), tuple(frozenset(value) for value in ancestors)


@pytest.mark.parametrize(("distance", "rounds"), [(3, 23), (5, 38)])
def test_local_check_matrices_equal_independent_global_crop_and_causal_exclusion(
    distance,
    rounds,
):
    data = _build_ab_models(distance, rounds)
    owners, ancestors = _independent_owner_and_ancestors(data)

    for window_index, (geometry, model) in enumerate(
        zip(data.plan.windows, data.models)
    ):
        local_detector_ids = set(model.detector_ids)
        prior_fault_ids = {
            source_fault_id
            for source_fault_id, owner in enumerate(owners)
            if owner in ancestors[window_index]
        }
        expected_source_fault_ids = tuple(
            source_fault_id
            for source_fault_id, detector_ids in enumerate(
                data.global_detector_sets
            )
            if set(detector_ids) & local_detector_ids
            and source_fault_id not in prior_fault_ids
        )
        faults = model.require_faults(FaultRepresentation.GRAPHLIKE)
        assert faults.source_fault_ids == expected_source_fault_ids

        row_index = {
            detector_id: index
            for index, detector_id in enumerate(model.detector_ids)
        }
        for column_index, source_fault_id in enumerate(
            faults.source_fault_ids
        ):
            expected_rows = {
                row_index[detector_id]
                for detector_id in data.global_detector_sets[source_fault_id]
                if detector_id in row_index
            }
            actual_rows = set(np.flatnonzero(faults.check[:, column_index]))
            assert actual_rows == expected_rows
            assert faults.priors[column_index] == pytest.approx(
                data.global_priors[source_fault_id], rel=0, abs=0)
            expected_observables = set(
                data.global_observable_sets[source_fault_id]
            )
            actual_observables = set(
                np.flatnonzero(faults.observables[:, column_index])
            )
            assert actual_observables == expected_observables
            assert bool(faults.owned[column_index]) == (
                owners[source_fault_id] == window_index
            )


@pytest.mark.parametrize(
    ("distance", "rounds"),
    [(3, 18), (3, 23), (5, 30), (5, 38)],
)
def test_A_is_temporally_open_and_B_is_temporally_closed_in_actual_matching_graphs(
    distance,
    rounds,
):
    data = _build_ab_models(distance, rounds)
    incoming = {destination for _, destination in data.plan.internal_dependencies}

    for window_index, (geometry, model) in enumerate(
        zip(data.plan.windows, data.models)
    ):
        faults = model.require_faults(FaultRepresentation.GRAPHLIKE)
        local_detector_ids = set(model.detector_ids)
        truncated = []
        for column_index, source_fault_id in enumerate(
            faults.source_fault_ids
        ):
            global_ids = set(data.global_detector_sets[source_fault_id])
            local_ids = global_ids & local_detector_ids
            outside_ids = global_ids - local_detector_ids
            if outside_ids:
                truncated.append(
                    (column_index, source_fault_id, local_ids, outside_ids)
                )

        if window_index in incoming:  # B: no artificial temporal sink.
            assert not truncated
            # Degree-one spatial/physical boundary edges are still legitimate;
            # closure means no globally multi-ended edge was cut in time.
            assert any(
                len(data.global_detector_sets[source_fault_id]) == 1
                and int(faults.check[:, column_index].sum()) == 1
                for column_index, source_fault_id in enumerate(
                    faults.source_fault_ids
                )
            )
            continue

        expected_sides = set()
        if geometry.buffer_lo > 1:
            expected_sides.add("left")
        if geometry.buffer_hi < rounds:
            expected_sides.add("right")
        actual_sides = set()
        matching = PyMatchingDecoder(_OneTick())._matching_for_model(faults)
        for column_index, source_fault_id, local_ids, outside_ids in truncated:
            assert len(data.global_detector_sets[source_fault_id]) == 2
            assert len(local_ids) == len(outside_ids) == 1
            assert int(faults.check[:, column_index].sum()) == 1
            local_id = next(iter(local_ids))
            outside_id = next(iter(outside_ids))
            local_round = data.round_of[local_id]
            outside_round = data.round_of[outside_id]
            if outside_round < geometry.buffer_lo:
                actual_sides.add("left")
            elif outside_round > geometry.buffer_hi:
                actual_sides.add("right")
            else:
                pytest.fail(
                    f"fault {source_fault_id} was truncated without crossing "
                    "an artificial temporal boundary"
                )
            row = model.detector_ids.index(local_id)
            assert any(
                (left == row and right in matching.boundary)
                or (right == row and left in matching.boundary)
                for left, right, _ in matching.edges()
            ), (
                window_index,
                source_fault_id,
                local_round,
                outside_round,
            )
        assert actual_sides == expected_sides


class _InjectedSyndromeDevice(StimDevice):
    def __init__(self, syndrome):
        super().__init__()
        self.syndrome = np.asarray(syndrome, dtype=np.bool_)

    def begin_operation(self, op, segment_round_count, source_round_count):
        super().begin_operation(op, segment_round_count, source_round_count)
        key = self._key(op)
        self._dets[key] = self.syndrome.copy()
        self._dets[op.id] = self._dets[key]


class _LatencyByAOrder:
    def __init__(self, right_first):
        self.right_first = right_first

    def latency(self, job):
        slow_window = 0 if self.right_first else 2
        return 20_000_000 if job.window_id == slow_window else 1


class _RecordingPyMatchingDecoder(PyMatchingDecoder):
    def __init__(self, latency):
        super().__init__(latency)
        self.completed = []

    def decode(self, job):
        result = super().decode(job)
        self.completed.append((job, result))
        return result


def _run_and_reconstruct_global_correction(data, detector_ids, right_first):
    syndrome = np.zeros(data.circuit.num_detectors, dtype=np.bool_)
    syndrome[list(detector_ids)] = 1
    decoder = _RecordingPyMatchingDecoder(_LatencyByAOrder(right_first))
    operation = Operation(
        1, "memory", (0,), clifford=True, circuit=data.circuit)
    run = simulate(RunSpec(
        ops=[operation],
        num_units=3,
        rounds_policy=FixedRounds(data.round_count),
        code=SurfaceCodeModel(d=3),
        scheme=ParallelWindowScheme(),
        device=_InjectedSyndromeDevice(syndrome),
        decoder=decoder,
        seed=None,
    ), verbose=False)

    selected_source_fault_ids = []
    for job, result in decoder.completed:
        faults = job.dem.require_faults(FaultRepresentation.GRAPHLIKE)
        selected_source_fault_ids.extend(
            source_fault_id
            for source_fault_id, selected in zip(
                faults.source_fault_ids,
                result.correction,
            )
            if selected
        )
    reconstructed_detectors = set()
    reconstructed_observables = set()
    for source_fault_id in selected_source_fault_ids:
        reconstructed_detectors.symmetric_difference_update(
            data.global_detector_sets[source_fault_id]
        )
        reconstructed_observables.symmetric_difference_update(
            data.global_observable_sets[source_fault_id]
        )
    return run, reconstructed_detectors, reconstructed_observables


@pytest.mark.parametrize("right_first", [False, True])
def test_low_weight_seams_and_logical_counterexample_satisfy_global_H_and_O(
    right_first,
):
    data = _build_ab_models(distance=3, rounds=18)
    patterns = [
        set(detector_ids)
        for detector_ids in data.global_detector_sets
        if {data.round_of[detector_id] for detector_id in detector_ids}
        in ({6, 7}, {15, 16})
    ]
    assert sum(
        {data.round_of[detector_id] for detector_id in detector_ids} == {6, 7}
        for detector_ids in patterns
    ) == 18
    assert sum(
        {data.round_of[detector_id] for detector_id in detector_ids} == {15, 16}
        for detector_ids in patterns
    ) == 18
    patterns.append({18, 21, 26})

    for detector_ids in patterns:
        run, reconstructed_detectors, reconstructed_observables = (
            _run_and_reconstruct_global_correction(
                data,
                detector_ids,
                right_first,
            )
        )
        assert reconstructed_detectors == detector_ids  # H * kappa == y
        assert run.window_manager.op_results[1] == tuple(
            int(observable_id in reconstructed_observables)
            for observable_id in range(data.circuit.num_observables)
        )  # O * kappa
    assert run.window_manager.op_results[1] == (1,)
