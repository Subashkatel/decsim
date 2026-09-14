"""The compiled growth, forest and peeling: the binding to union_find.c.

The decisions are the C file's; this module lays one graph and one
residual syndrome out as flat arrays, makes one call, and reads the
outcome back as the records the evidence carries. ctypes rather than
cffi because ctypes is in the standard library, so a checkout that
compiles the C needs nothing else, and one call carries a whole decode,
so the per-call cost of either binding is beside the point.
"""

import ctypes
import dataclasses
import functools
import os
import pathlib

import numpy

import decsim.records.decoder_evidence as evidence_records

LIBRARY_FILE = "union_find.so"
BUILD_COMMAND = "tools/build_union_find.sh"
LIBRARY_VARIABLE = "DECSIM_UNION_FIND_LIBRARY"

_OK = 0
_OUT_OF_MEMORY = 1
_INTERVAL_ORDER_LOST = 2
_NO_POSITIVE_GROWTH = 3
_GROWTH_BOUND_EXCEEDED = 4

_MESSAGE_BY_STATUS = {
    _OUT_OF_MEMORY: "weighted Union-Find could not take its workspace",
    _INTERVAL_ORDER_LOST: (
        "weighted Union-Find edge update lost represented interval order"
    ),
    _NO_POSITIVE_GROWTH: (
        "weighted Union-Find event has no positive represented growth"
    ),
    _GROWTH_BOUND_EXCEEDED: (
        "weighted Union-Find growth exceeded its finite graph bound"
    ),
}

_INDEX_ARRAY = numpy.ctypeslib.ndpointer(
    dtype=numpy.int32, ndim=1, flags="C_CONTIGUOUS"
)
_TICK_ARRAY = numpy.ctypeslib.ndpointer(
    dtype=numpy.int64, ndim=1, flags="C_CONTIGUOUS"
)
_BIT_ARRAY = numpy.ctypeslib.ndpointer(
    dtype=numpy.uint8, ndim=1, flags="C_CONTIGUOUS"
)
_ARGUMENT_TYPES = [
    ctypes.c_int32,
    ctypes.c_int32,
    _INDEX_ARRAY,
    _INDEX_ARRAY,
    _TICK_ARRAY,
    _BIT_ARRAY,
    _BIT_ARRAY,
    _BIT_ARRAY,
    _TICK_ARRAY,
    _TICK_ARRAY,
    _INDEX_ARRAY,
    _INDEX_ARRAY,
    _INDEX_ARRAY,
    _INDEX_ARRAY,
]


@dataclasses.dataclass(frozen=True)
class GrowthOutcome:
    """What one compiled decode leaves behind, in edge indices."""

    selected_edges: tuple
    edge_intervals: tuple
    contact_edges: tuple
    forest_edges: tuple


def decode(graph: evidence_records.UnionFindGraph, residual_syndrome):
    """Grow, take the forest and peel one residual syndrome on one graph."""
    endpoint_a, endpoint_b, lengths = _graph_arrays(graph)
    edge_count = len(graph.edges)
    selected = numpy.zeros(edge_count, dtype=numpy.uint8)
    is_closed = numpy.zeros(edge_count, dtype=numpy.uint8)
    lower_tick = numpy.zeros(edge_count, dtype=numpy.int64)
    upper_tick = numpy.zeros(edge_count, dtype=numpy.int64)
    contacts = numpy.zeros(edge_count, dtype=numpy.int32)
    contact_count = numpy.zeros(1, dtype=numpy.int32)
    forest = numpy.zeros(edge_count, dtype=numpy.int32)
    forest_count = numpy.zeros(1, dtype=numpy.int32)
    decode_window = entry_point()
    status = decode_window(
        graph.detector_count,
        edge_count,
        endpoint_a,
        endpoint_b,
        lengths,
        residual_syndrome,
        selected,
        is_closed,
        lower_tick,
        upper_tick,
        contacts,
        contact_count,
        forest,
        forest_count,
    )
    _refuse_failure(status)
    selected_edges = _selected_edges(selected)
    edge_intervals = _intervals(is_closed, lower_tick, upper_tick)
    contact_edges = _prefix(contacts, contact_count)
    forest_edges = _prefix(forest, forest_count)
    return GrowthOutcome(
        selected_edges=selected_edges,
        edge_intervals=edge_intervals,
        contact_edges=contact_edges,
        forest_edges=forest_edges,
    )


@functools.cache
def entry_point():
    """The one exported function, bound once per process."""
    path = library_path()
    if not path.exists():
        raise RuntimeError(
            "the Union-Find decoder needs its compiled library at "
            f"{path}; build it with {BUILD_COMMAND}"
        )
    library = ctypes.CDLL(str(path))
    decode_window = library.union_find_decode
    decode_window.argtypes = _ARGUMENT_TYPES
    decode_window.restype = ctypes.c_int32
    return decode_window


def library_path() -> pathlib.Path:
    """The built library, or the one the environment names instead."""
    named = os.environ.get(LIBRARY_VARIABLE)
    if named:
        return pathlib.Path(named)
    here = pathlib.Path(__file__)
    folder = here.parent
    return folder / LIBRARY_FILE


def _graph_arrays(graph: evidence_records.UnionFindGraph) -> tuple:
    """The graph as the three flat arrays the C reads, in edge order."""
    endpoint_a = []
    endpoint_b = []
    lengths = []
    for edge in graph.edges:
        endpoint_a.append(edge.detector_a)
        endpoint_b.append(edge.detector_b)
        lengths.append(edge.length_half_ticks)
    first = numpy.asarray(endpoint_a, dtype=numpy.int32)
    second = numpy.asarray(endpoint_b, dtype=numpy.int32)
    third = numpy.asarray(lengths, dtype=numpy.int64)
    return first, second, third


def _refuse_failure(status: int) -> None:
    if status == _OK:
        return
    message = _MESSAGE_BY_STATUS[status]
    raise RuntimeError(message)


def _selected_edges(selected) -> tuple:
    found = numpy.nonzero(selected)
    indices = []
    for edge_index in found[0]:
        indices.append(int(edge_index))
    return tuple(indices)


def _intervals(is_closed, lower_tick, upper_tick) -> tuple:
    """One interval per edge, shared between the edges that carry it.

    Open and Closed are frozen and carry no identity: every reader tests
    the type and reads the bounds, so one instance stands for every edge
    with the same interval. A window has tens of thousands of edges and
    a few hundred distinct intervals, and building one object each was
    three quarters of a decode's Python time. The arrays are read whole
    because element by element indexing of numpy costs more than the
    list does.
    """
    closed = evidence_records.Closed()
    closed_flags = is_closed.tolist()
    lower_ticks = lower_tick.tolist()
    upper_ticks = upper_tick.tolist()
    open_by_bounds = {}
    intervals = []
    for edge_index, closed_flag in enumerate(closed_flags):
        if closed_flag:
            intervals.append(closed)
            continue
        lower = lower_ticks[edge_index]
        upper = upper_ticks[edge_index]
        bounds = (lower, upper)
        interval = open_by_bounds.get(bounds)
        if interval is None:
            interval = evidence_records.Open(lower, upper)
            open_by_bounds[bounds] = interval
        intervals.append(interval)
    return tuple(intervals)


def _prefix(values, count) -> tuple:
    taken = values[: count[0]]
    indices = []
    for value in taken:
        indices.append(int(value))
    return tuple(indices)
