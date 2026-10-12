"""The union-find graph of decsim/records/decoder_evidence.py.

The graph lays its edges out as the flat arrays union_find.h reads
(union_find_decode: endpoint_a, endpoint_b, length_half_ticks, and the
logical parity of union_find_cluster_gap), once per graph, since every
decode on one placed model reads the same arrays.
"""

import numpy
import pytest

import decsim.records.decoder_evidence as evidence_records

BOUNDARY = -1


def two_edge_graph() -> evidence_records.UnionFindGraph:
    """A detector between the boundary and a second detector."""
    first = evidence_records.UnionFindEdge(
        fault_index=0,
        detector_a=0,
        detector_b=BOUNDARY,
        logical_observables=(1,),
        length_half_ticks=6,
    )
    second = evidence_records.UnionFindEdge(
        fault_index=2,
        detector_a=0,
        detector_b=1,
        logical_observables=(0,),
        length_half_ticks=4,
    )
    return evidence_records.UnionFindGraph(
        detector_count=2,
        fault_count=3,
        edges=(first, second),
        baseline_faults=(),
        baseline_syndrome=(0, 0),
    )


def test_the_edge_arrays_hold_each_edge_in_edge_order():
    graph = two_edge_graph()
    endpoint_a, endpoint_b, lengths = graph.edge_arrays
    assert endpoint_a.tolist() == [0, 0]
    assert endpoint_b.tolist() == [BOUNDARY, 1]
    assert lengths.tolist() == [6, 4]
    assert graph.logical_parities.tolist() == [1, 0]


def test_the_edge_arrays_take_the_types_the_compiled_growth_reads():
    graph = two_edge_graph()
    endpoint_a, endpoint_b, lengths = graph.edge_arrays
    assert endpoint_a.dtype == numpy.int32
    assert endpoint_b.dtype == numpy.int32
    assert lengths.dtype == numpy.int64
    assert graph.logical_parities.dtype == numpy.uint8


def test_a_graph_lays_out_its_edge_arrays_once():
    graph = two_edge_graph()
    assert graph.edge_arrays is graph.edge_arrays
    assert graph.logical_parities is graph.logical_parities


def test_the_edge_arrays_every_decode_shares_are_read_only():
    graph = two_edge_graph()
    endpoint_a, _endpoint_b, _lengths = graph.edge_arrays
    with pytest.raises(ValueError, match="read-only"):
        endpoint_a[0] = 1


def test_the_laid_out_arrays_leave_two_equal_graphs_equal():
    graph = two_edge_graph()
    laid_out = two_edge_graph()
    _arrays = laid_out.edge_arrays
    assert graph == laid_out
    assert hash(graph) == hash(laid_out)
