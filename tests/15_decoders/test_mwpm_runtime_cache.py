"""Runtime PyMatching decoder: one matching graph per live window model, warmed
on syndromes every graph can satisfy."""

import gc

import numpy as np

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
