"""Exact detector-model and boundary gates for forward Sliding windows."""
from __future__ import annotations

import itertools

import numpy as np
import pymatching
import pytest
import stim

from decsim.detector_error_model import (
    FaultRepresentation,
    GRAPHLIKE_FAULT_MODEL_REQUIRED,
    build_window_error_models,
    decode_windowed,
    detector_error_model_to_faults,
)
from decsim.mwpm_decoder import PyMatchingDecoder, matching_window_decoder
from decsim.schemes import SlidingWindowScheme


class _OneTick:
    def latency(self, job):
        return 1


class _NineFaultCircuit:
    num_detectors = 8
    num_observables = 1

    def __init__(self):
        self.dem = stim.DetectorErrorModel("""
            error(0.01) D0 L0
            error(0.02) D0 D1
            error(0.03) D1 D2
            error(0.04) D2 D3
            error(0.05) D3 D4
            error(0.06) D4 D5
            error(0.07) D5 D6
            error(0.08) D6 D7
            error(0.09) D7 L0
        """)

    def detector_error_model(self, *, decompose_errors):
        return self.dem

    def get_detector_coordinates(self):
        return {detector_id: [float(detector_id), 0.0]
                for detector_id in range(self.num_detectors)}


def _sliding_models():
    circuit = _NineFaultCircuit()
    plan = SlidingWindowScheme().plan_operation(
        0, 8, commit_round_count=2, buffer_round_count=2)
    entries = tuple(
        (window.buffer_lo, window.commit_lo, window.commit_hi, window.buffer_hi)
        for window in plan.windows
    )
    models = build_window_error_models(
        circuit,
        entries,
        round_count=8,
        detector_rounds={detector_id: detector_id + 1
                         for detector_id in range(8)},
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
        dependency_edges=plan.internal_dependencies,
    )
    return circuit, plan, tuple(models)


def _boundary_rows(faults):
    matching = PyMatchingDecoder(_OneTick())._matching_for_model(faults)
    return {
        node
        for left, right, _ in matching.edges()
        for node, other in ((left, right), (right, left))
        if other in matching.boundary and node not in matching.boundary
    }


def test_nine_fault_oracle_pins_exact_forward_sliding_models():
    circuit, plan, models = _sliding_models()
    assert plan.internal_dependencies == ((0, 1), (1, 2))
    expected = (
        {
            "detectors": (0, 1, 2, 3),
            "sources": (0, 1, 2, 3, 4),
            "owned": (1, 1, 1, 0, 0),
            "check": ((1, 1, 0, 0, 0),
                      (0, 1, 1, 0, 0),
                      (0, 0, 1, 1, 0),
                      (0, 0, 0, 1, 1)),
            "future": {2: (2,)},
            "boundary_rows": {0, 3},
        },
        {
            "detectors": (2, 3, 4, 5),
            "sources": (3, 4, 5, 6),
            "owned": (1, 1, 0, 0),
            "check": ((1, 0, 0, 0),
                      (1, 1, 0, 0),
                      (0, 1, 1, 0),
                      (0, 0, 1, 1)),
            "future": {1: (4,)},
            "boundary_rows": {3},
        },
        {
            "detectors": (4, 5, 6, 7),
            "sources": (5, 6, 7, 8),
            "owned": (1, 1, 1, 1),
            "check": ((1, 0, 0, 0),
                      (1, 1, 0, 0),
                      (0, 1, 1, 0),
                      (0, 0, 1, 1)),
            "future": {},
            "boundary_rows": {3},
        },
    )
    global_sets, _, _ = detector_error_model_to_faults(
        circuit.detector_error_model(decompose_errors=True))
    prior_owned = set()
    for model, oracle in zip(models, expected):
        faults = model.require_faults(FaultRepresentation.GRAPHLIKE)
        assert model.detector_ids == oracle["detectors"]
        assert faults.source_fault_ids == oracle["sources"]
        assert tuple(int(bit) for bit in faults.owned) == oracle["owned"]
        assert tuple(tuple(int(bit) for bit in row) for row in faults.check) == (
            oracle["check"]
        )
        assert faults.future_flips == oracle["future"]
        assert _boundary_rows(faults) == oracle["boundary_rows"]
        assert prior_owned.isdisjoint(faults.source_fault_ids)
        prior_owned.update(
            source_fault_id
            for source_fault_id, owned in zip(
                faults.source_fault_ids, faults.owned)
            if owned
        )
        for column_index, source_fault_id in enumerate(faults.source_fault_ids):
            expected_rows = {
                model.detector_ids.index(detector_id)
                for detector_id in global_sets[source_fault_id]
                if detector_id in model.detector_ids
            }
            assert set(np.flatnonzero(faults.check[:, column_index])) == (
                expected_rows
            )
    assert prior_owned == set(range(9))


def test_forward_sliding_commits_reconstruct_H_and_O_for_all_weight_two_patterns():
    circuit, _, models = _sliding_models()
    detector_sets, observable_sets, _ = detector_error_model_to_faults(
        circuit.detector_error_model(decompose_errors=True))
    global_matching = pymatching.Matching.from_detector_error_model(circuit.dem)
    inner_decode = matching_window_decoder()

    for pattern_size in (0, 1, 2):
        for source_fault_ids in itertools.combinations(
            range(len(detector_sets)), pattern_size
        ):
            input_detectors = set()
            for source_fault_id in source_fault_ids:
                input_detectors.symmetric_difference_update(
                    detector_sets[source_fault_id]
                )
            detection_events = np.zeros(8, dtype=np.uint8)
            detection_events[list(input_detectors)] = 1
            selections = []

            def recording_decode(model, syndrome):
                selected = inner_decode(model, syndrome)
                selections.append((model, selected.copy()))
                return selected

            prediction = decode_windowed(
                list(models),
                detection_events,
                recording_decode,
                selected_fault_representation=FaultRepresentation.GRAPHLIKE,
            )
            committed_source_fault_ids = []
            for model, selected in selections:
                faults = model.graphlike_faults
                committed_source_fault_ids.extend(
                    source_fault_id
                    for source_fault_id, selected_bit, owned in zip(
                        faults.source_fault_ids,
                        selected,
                        faults.owned,
                    )
                    if selected_bit and owned
                )
            reconstructed_detectors = set()
            reconstructed_observables = set()
            for source_fault_id in committed_source_fault_ids:
                reconstructed_detectors.symmetric_difference_update(
                    detector_sets[source_fault_id]
                )
                reconstructed_observables.symmetric_difference_update(
                    observable_sets[source_fault_id]
                )
            assert reconstructed_detectors == input_detectors
            assert tuple(int(bit) for bit in prediction) == tuple(
                int(observable_id in reconstructed_observables)
                for observable_id in range(circuit.num_observables)
            )
            assert tuple(int(bit) for bit in prediction) == tuple(
                int(bit) for bit in global_matching.decode(detection_events)
            )


class _LongFaultCircuit:
    num_detectors = 3
    num_observables = 1

    def __init__(self):
        self.dem = stim.DetectorErrorModel("""
            error(0.1) D0 D2 L0
            error(0.2) D2 L0
        """)

    def detector_error_model(self, *, decompose_errors):
        return self.dem

    def get_detector_coordinates(self):
        return {detector_id: [float(detector_id), 0.0]
                for detector_id in range(self.num_detectors)}


def test_long_fault_residual_persists_through_intermediate_buffer():
    models = build_window_error_models(
        _LongFaultCircuit(),
        [(1, 1, 1, 2), (2, 2, 2, 3), (3, 3, 3, 3)],
        round_count=3,
        detector_rounds={0: 1, 1: 2, 2: 3},
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
        dependency_edges=((0, 1), (1, 2)),
    )
    observed_syndromes = []
    inner_decode = matching_window_decoder()

    def decode(model, syndrome):
        observed_syndromes.append(tuple(int(value) for value in syndrome))
        return inner_decode(model, syndrome)

    prediction = decode_windowed(
        models,
        np.array([1, 0, 1], dtype=np.uint8),
        decode,
        selected_fault_representation=FaultRepresentation.GRAPHLIKE,
    )
    assert tuple(prediction) == (1,)
    assert observed_syndromes == [(1, 0), (0, 0), (0,)]
