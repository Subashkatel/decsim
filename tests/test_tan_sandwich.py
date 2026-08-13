"""Exact Tan zero-seam sandwich geometry and correction-edge ownership."""

import math

import pytest
import stim

from decsim.detector_error_model import (
    GRAPHLIKE_FAULT_MODEL_REQUIRED,
    PHYSICAL_FAULT_MODEL_REQUIRED,
    build_window_error_models,
)
from decsim.detector_error_model import NO_FAULT_MODEL_REQUIRED
from decsim.adapters.stim_device import StimDevice
from decsim.codes import SurfaceCodeModel
from decsim.decoders import PresetLatencyDecoder
from decsim.devices import TimingOnlyDevice
from decsim.message import (
    DecodeResult,
    DependencyResidual,
    Operation,
    QPUReadout, SyndromePayload,
    WindowGeometry,
    WindowProtocol,
)
from decsim.mwpm_decoder.decoder import PyMatchingDecoder
from decsim.planner import FixedRounds
from decsim.run_spec import RunSpec, simulate
from decsim.schemes import TanSandwichScheme


class _TanEdgeCircuit:
    num_detectors = 18
    num_observables = 1

    def __init__(self, extra_error=""):
        errors = []
        for detector_id in range(17):
            errors.append(
                f"error({0.001 + detector_id * 0.0001}) "
                f"D{detector_id} D{detector_id + 1}"
            )
        # Spatial-boundary edges on the six zero-offset seam layers.
        for detector_id in (3, 5, 7, 9, 11, 13):
            errors.append(f"error(0.003) D{detector_id}")
        if extra_error:
            errors.append(extra_error)
        self.dem = stim.DetectorErrorModel("\n".join(errors))

    def detector_error_model(self, *, decompose_errors):
        return self.dem

    def get_detector_coordinates(self):
        return {
            detector_id: [0.0, 0.0, float(detector_id + 1)]
            for detector_id in range(self.num_detectors)
        }


def _plan(round_count=18, step=2, buffer=2):
    return TanSandwichScheme().plan_operation(
        0,
        round_count,
        commit_round_count=step,
        buffer_round_count=buffer,
    )


def _entries(plan):
    return tuple(
        (window.buffer_lo, window.commit_lo,
         window.commit_hi, window.buffer_hi)
        for window in plan.windows
    )


def _models(circuit=None):
    circuit = circuit or _TanEdgeCircuit()
    plan = _plan()
    models = build_window_error_models(
        circuit,
        _entries(plan),
        round_count=18,
        detector_rounds={detector_id: detector_id + 1 for detector_id in range(18)},
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
        dependency_edges=plan.internal_dependencies,
        closed_temporal_boundary_windows=tuple(range(1, len(plan.windows), 2)),
        window_protocol=plan.protocol,
    )
    return circuit, plan, tuple(models)


def test_tan_published_zero_seam_geometry_is_exact():
    plan = _plan()
    assert plan.windows == (
        WindowGeometry(1, 1, 3, 6),
        WindowGeometry(4, 4, 4, 4, True),
        WindowGeometry(3, 5, 5, 8),
        WindowGeometry(6, 6, 6, 6, True),
        WindowGeometry(5, 7, 7, 10),
        WindowGeometry(8, 8, 8, 8, True),
        WindowGeometry(7, 9, 9, 12),
        WindowGeometry(10, 10, 10, 10, True),
        WindowGeometry(9, 11, 11, 14),
        WindowGeometry(12, 12, 12, 12, True),
        WindowGeometry(11, 13, 13, 16),
        WindowGeometry(14, 14, 14, 14, True),
        WindowGeometry(13, 15, 18, 18),
    )
    assert plan.entry_window_indices == tuple(range(0, 13, 2))
    assert plan.exit_window_indices == tuple(range(1, 13, 2))
    assert plan.internal_dependencies == tuple(
        edge
        for seam in range(1, 13, 2)
        for edge in ((seam - 1, seam), (seam + 1, seam))
    )


def test_tan_endpoint_sweep_partitions_core_vertices_and_seams_once():
    for step, buffer in ((2, 1), (2, 2), (3, 3), (4, 2)):
        width = step + 2 * buffer
        for round_count in range(1, 3 * width + 1):
            plan = _plan(round_count, step, buffer)
            expected_type_1_count = max(
                1, math.ceil((round_count - width) / step) + 1
            )
            assert len(plan.windows) == 2 * expected_type_1_count - 1
            assert plan.protocol is WindowProtocol.TAN_ZERO_SEAM_GRAPHLIKE
            assert all(
                window.buffer_lo == window.commit_lo
                == window.commit_hi == window.buffer_hi
                and window.closed_temporal_boundaries
                for window in plan.windows[1::2]
            )
            committed = tuple(
                detector_round
                for window in plan.windows
                for detector_round in range(window.commit_lo, window.commit_hi + 1)
            )
            assert committed == tuple(range(1, round_count + 1))
            assert all(
                1 <= window.buffer_lo <= window.commit_lo
                <= window.commit_hi <= window.buffer_hi <= round_count
                for window in plan.windows
            )


def test_tan_edge_cores_and_seams_own_every_global_edge_exactly_once():
    circuit, plan, models = _models()
    owned_source_ids = []
    for model in models:
        faults = model.graphlike_faults
        owned_source_ids.extend(
            source_id
            for source_id, owned in zip(faults.source_fault_ids, faults.owned)
            if owned
        )
    assert sorted(owned_source_ids) == list(range(23))
    assert len(set(owned_source_ids)) == 23

    # Every one-detector spatial seam edge belongs to its type-2 task.
    for seam_number, model_index in enumerate(range(1, len(models), 2)):
        faults = models[model_index].graphlike_faults
        expected_source_id = 17 + seam_number
        assert expected_source_id in {
            source_id
            for source_id, owned in zip(faults.source_fault_ids, faults.owned)
            if owned
        }

    # A correction incident on a core vertex keeps its complete seam effect.
    first_a = models[0].graphlike_faults
    source_id_to_column = {
        source_id: column
        for column, source_id in enumerate(first_a.source_fault_ids)
    }
    crossing_column = source_id_to_column[2]  # D2--D3, rounds 3--4.
    assert first_a.owned[crossing_column]
    assert first_a.boundary_flips[crossing_column] == (2, 3)
    assert 3 in models[1].detector_ids


def test_tan_fails_closed_on_cross_core_or_unroutable_edges_and_wrong_domain():
    with pytest.raises(ValueError, match="straddles independent commit regions"):
        _models(_TanEdgeCircuit("error(0.004) D2 D4"))
    with pytest.raises(
        ValueError,
        match="no direct dependency route|closed temporal boundary",
    ):
        _models(_TanEdgeCircuit("error(0.004) D2 D5"))

    plan = _plan()
    with pytest.raises(ValueError, match="graphlike correction-edge"):
        build_window_error_models(
            _TanEdgeCircuit(),
            _entries(plan),
            round_count=18,
            detector_rounds={i: i + 1 for i in range(18)},
            fault_model_requirement=PHYSICAL_FAULT_MODEL_REQUIRED,
            fault_exclusion_ranges=(),
            dependency_edges=plan.internal_dependencies,
            closed_temporal_boundary_windows=tuple(range(1, len(plan.windows), 2)),
            window_protocol=plan.protocol,
        )


class _FixedRoundDevice(TimingOnlyDevice):
    def __init__(self, raw_by_round):
        self.raw_by_round = dict(raw_by_round)

    def round_payloads(self, op, round_index):
        return [QPUReadout(
            op.id,
            op.qubits[0],
            round_index,
            bits=self.raw_by_round.get(round_index, (0, 0)),
        )]


class _TanDecoder:
    fault_model_requirement = NO_FAULT_MODEL_REQUIRED

    def __init__(self, window_id, latency=1, residual=None):
        self.window_id = window_id
        self.decode_latency = latency
        self.residual = residual or {}
        self.jobs = []

    def latency(self, job):
        return self.decode_latency

    def decode(self, job):
        self.jobs.append(job)
        return DecodeResult(
            job.op_id,
            job.window_id,
            logical_observables=(0,),
            boundary_data=DependencyResidual(defects=dict(self.residual)),
        )


class _TanRouter:
    def __init__(self, decoders):
        self.decoders = decoders

    def route(self, job):
        return self.decoders[job.window_id]

    def fault_model_requirement_for(self, code):
        return NO_FAULT_MODEL_REQUIRED


def test_tan_runtime_waits_for_both_type_1_tasks_and_xors_their_seam_residuals():
    # For d=s=b=3 and R=18, the first zero-offset seam is round 6.
    decoders = {
        index: _TanDecoder(index)
        for index in range(7)
    }
    decoders[0] = _TanDecoder(0, latency=1, residual={6: [1, 1]})
    decoders[2] = _TanDecoder(2, latency=20_000_000, residual={6: [1, 0]})
    operation = Operation(0, "memory", (0,), clifford=True)
    run = simulate(RunSpec(
        ops=[operation],
        d=3,
        rounds_policy=FixedRounds(18),
        round_us=1.1,
        scheme=TanSandwichScheme(),
        device=_FixedRoundDevice({6: (1, 0)}),
        router=_TanRouter(decoders),
        num_units=4,
    ), verbose=False)

    left = run.window_manager.windows[(0, 0)]
    seam = run.window_manager.windows[(0, 1)]
    right = run.window_manager.windows[(0, 2)]
    assert not left.closed_temporal_boundaries
    assert seam.closed_temporal_boundaries
    assert seam.t_dispatch >= max(left.t_done, right.t_done)
    payload_by_round = {
        payload.round_index: tuple(payload.bits)
        for payload in decoders[1].jobs[0].payloads
    }
    assert payload_by_round[6] == (1, 1)
    assert {
        source[1] for source in seam.boundary_in.contributions
    } == {0, 2}


@pytest.mark.parametrize("seed", range(20))
def test_tan_paper_parameters_run_end_to_end_on_surface_memory(seed):
    """Exercise s=b=(d+1)/2 with graphlike MWPM and exact seam checks."""
    round_count = 18
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=round_count,
        after_clifford_depolarization=0.003,
        after_reset_flip_probability=0.003,
        before_measure_flip_probability=0.003,
        before_round_data_depolarization=0.003,
    )
    operation = Operation(
        0,
        "rotated-memory-z",
        (0,),
        clifford=True,
        circuit=circuit,
    )
    run = simulate(RunSpec(
        ops=[operation],
        code=SurfaceCodeModel(
            d=3,
            commit_rounds_override=2,
            buffer_rounds_override=2,
        ),
        rounds_policy=FixedRounds(round_count),
        scheme=TanSandwichScheme(),
        device=StimDevice(),
        decoder=PyMatchingDecoder(PresetLatencyDecoder(0.1)),
        seed=seed,
        num_units=8,
    ), verbose=False)
    assert run.window_manager.total_windows == 13
    assert len(run.window_manager.committed_windows) == 13
