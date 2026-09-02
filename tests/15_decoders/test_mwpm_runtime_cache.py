"""Runtime PyMatching decoder: one matching graph per live window model, warmed
on syndromes every graph can satisfy."""

import gc

import numpy as np
import pytest

from decsim.decoders.decoders import PresetLatencyDecoder
from decsim.decoders.mwpm.decoder import PyMatchingDecoder
from decsim.detector_error_model.fault_model_contracts import (
    FaultRepresentation,
    PlacedFaultModel,
)


def _placed(check):
    check = np.asarray(check, dtype=np.uint8)
    column_count = check.shape[1]
    return PlacedFaultModel(
        representation=FaultRepresentation.GRAPHLIKE,
        check=check,
        priors=np.full(column_count, 0.1),
        observables=np.zeros((1, column_count), dtype=np.uint8),
        owned=np.ones(column_count, dtype=bool),
        source_fault_ids=tuple(range(column_count)),
        boundary_flips={},
    )


def test_warm_up_survives_boundaryless_components():
    """Two disjoint edges without a boundary (a toric-like graph): an arbitrary
    detector pair has no perfect matching, a column's own detector set does."""
    decoder = PyMatchingDecoder(PresetLatencyDecoder(0.0))
    matching = decoder._matching_for_model(_placed([[1, 0], [0, 1], [1, 0], [0, 1]]))
    assert matching.decode(np.array([1, 0, 1, 0], dtype=np.uint8)).tolist() == [1, 0]


def test_cache_entry_leaves_with_its_model():
    decoder = PyMatchingDecoder(PresetLatencyDecoder(0.0))
    faults = _placed([[1, 1, 0], [0, 1, 1]])
    first = decoder._matching_for_model(faults)
    assert decoder._matching_for_model(faults) is first
    assert len(decoder._matchings) == 1
    del faults
    gc.collect()
    assert decoder._matchings == {}


def test_parallel_fault_columns_combine_as_independent_errors():
    """Two faults with the same detector endpoints are one edge whose
    probability is p1(1-p2) + p2(1-p1): the convention of Stim detector
    error models and of PyMatching's DEM loader, and what a window
    restriction produces when distinct faults fold onto the same in-window
    endpoints. Here two parallel A-B faults at 0.05 each (combined 0.095,
    weight 2.254) beat the A-C-B path at 0.2 per edge (weight 2.773) only
    when combined; keeping the lighter parallel edge alone (weight 2.944)
    would pick the path and flip the wrong observable."""
    check = np.asarray([[1, 1, 1, 0],
                        [1, 1, 0, 1],
                        [0, 0, 1, 1]], dtype=np.uint8)
    faults = PlacedFaultModel(
        representation=FaultRepresentation.GRAPHLIKE,
        check=check,
        priors=np.asarray([0.05, 0.05, 0.2, 0.2]),
        observables=np.asarray([[1, 1, 0, 0]], dtype=np.uint8),
        owned=np.ones(4, dtype=bool),
        source_fault_ids=(0, 1, 2, 3),
        boundary_flips={},
    )
    decoder = PyMatchingDecoder(PresetLatencyDecoder(0.0))
    matching = decoder._matching_for_model(faults)
    selected = matching.decode(np.array([1, 1, 0], dtype=np.uint8))
    assert selected[2] == 0 and selected[3] == 0
    assert selected[0] + selected[1] == 1


def test_gap_metric_minimum_weight_equals_the_window_decoders():
    """The complementary gap is measured from the window decoder's own
    minimum: its base matching merges parallel faults as independent errors
    exactly as the decoder does, so w_min is the decoder's weight and the
    gap is the decoder's confidence, not a differently built graph's.
    Faults that flip the observable are boundary faults here, as they are
    for a logical operator supported on the code boundary (Stim's memory
    circuits), which the augmented-detector solve requires."""
    from decsim.confidence.complementary import ComplementaryGapMetric
    from decsim.decoders.mwpm.weights import matching_weights
    # A: two parallel boundary faults flipping the observable (0.05 each,
    # combined 0.095, weight 2.254); A-B and B's boundary at 0.2 (2.773)
    check = np.asarray([[1, 1, 1, 0],
                        [0, 0, 1, 1]], dtype=np.uint8)
    priors = np.asarray([0.05, 0.05, 0.2, 0.2])
    observables = np.asarray([[1, 1, 0, 0]], dtype=np.uint8)
    faults = PlacedFaultModel(
        representation=FaultRepresentation.GRAPHLIKE, check=check, priors=priors,
        observables=observables, owned=np.ones(4, dtype=bool),
        source_fault_ids=(0, 1, 2, 3), boundary_flips={})
    decoder_matching = PyMatchingDecoder(PresetLatencyDecoder(0.0))._matching_for_model(faults)
    metric = ComplementaryGapMetric(check, observables, matching_weights(priors))
    syndrome = np.array([1, 0], dtype=np.uint8)
    _, decoder_weight = decoder_matching.decode(syndrome, return_weight=True)
    soft = metric.evaluate(syndrome)
    assert decoder_weight == pytest.approx(np.log((1 - 0.095) / 0.095))
    assert soft.w_min == pytest.approx(decoder_weight)
    assert soft.w_comp == pytest.approx(2 * np.log(0.8 / 0.2))
    assert soft.gap == pytest.approx(soft.w_comp - soft.w_min)
