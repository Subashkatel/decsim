"""Union-Find on a residual syndrome the window cannot explain: best effort,
as PECOS (grow until no progress, then peel) and ldpc (return the decoding)
do; nothing raises, the unmatched detectors are reported in the evidence."""

import numpy as np

from decsim.decoders.union_find.window_decoder import (
    _decode_graph,
    _graph_from_model,
)
from decsim.detector_error_model.fault_model_contracts import (
    FaultRepresentation,
    PlacedFaultModel,
)


def _placed(check, priors, observables=None):
    check = np.asarray(check, dtype=np.uint8)
    column_count = check.shape[1]
    if observables is None:
        observables = np.zeros((1, column_count), dtype=np.uint8)
    return PlacedFaultModel(
        representation=FaultRepresentation.GRAPHLIKE,
        check=check,
        priors=np.asarray(priors, dtype=float),
        observables=np.asarray(observables, dtype=np.uint8),
        owned=np.ones(column_count, dtype=bool),
        source_fault_ids=tuple(range(column_count)),
        boundary_flips={},
    )


def _decode(check, priors, syndrome):
    graph = _graph_from_model(_placed(check, priors), location="test", weight_step=0.1)
    return _decode_graph(graph, np.asarray(syndrome, dtype=np.uint8))


def test_isolated_defect_is_left_unmatched_without_raising():
    # detector 0 has a boundary edge, detector 1 has no edge at all
    evidence = _decode([[1], [0]], [0.1], [0, 1])
    assert evidence.selected_faults == (0,)
    assert evidence.unmatched_detectors == (1,)


def test_satisfiable_syndrome_is_still_exact():
    evidence = _decode([[1], [0]], [0.1], [1, 0])
    assert evidence.selected_faults == (1,)
    assert evidence.unmatched_detectors == ()


def test_odd_clusters_without_boundary_stop_growing_and_peel_partially():
    # two disjoint edges, no boundary: (0,2) and (1,3); one defect on each
    check = [[1, 0], [0, 1], [1, 0], [0, 1]]
    evidence = _decode(check, [0.1, 0.1], [1, 1, 0, 0])
    assert evidence.selected_faults == (0, 0)
    assert evidence.unmatched_detectors == (0, 1)


def test_satisfiable_pair_on_boundaryless_edge_is_matched():
    check = [[1, 0], [0, 1], [1, 0], [0, 1]]
    evidence = _decode(check, [0.1, 0.1], [1, 0, 1, 0])
    assert evidence.selected_faults == (1, 0)
    assert evidence.unmatched_detectors == ()
