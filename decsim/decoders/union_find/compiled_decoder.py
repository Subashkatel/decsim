"""The compiled Union-Find row: the binding to its two C sources.

The decisions are the C files' (union_find.c grows, takes the contact
forest and peels; cluster_gap.c walks the quotient graph of a growth);
this module lays a graph, a residual syndrome and a growth out as flat
arrays, makes one call, and reads the outcome back as the records the
evidence carries. ctypes rather than cffi because ctypes is in the
standard library, so a checkout that compiles the C needs nothing else,
and one call carries a whole window, so the per-call cost of either
binding is beside the point.
"""

import ctypes
import dataclasses
import functools
import math
import os
import pathlib
from typing import Union

import numpy

import decsim.records.decoder_evidence as evidence_records

LIBRARY_FILE = "union_find.so"
BUILD_COMMAND = "tools/build_union_find.sh"
LIBRARY_VARIABLE = "DECSIM_UNION_FIND_LIBRARY"
_DECODE_SYMBOL = "union_find_decode"
_CLUSTER_GAP_SYMBOL = "union_find_cluster_gap"

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
_DECODE_ARGUMENT_TYPES = [
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
    _INDEX_ARRAY,
    _INDEX_ARRAY,
    _TICK_ARRAY,
    _BIT_ARRAY,
    _INDEX_ARRAY,
    _INDEX_ARRAY,
]
_CLUSTER_GAP_ARGUMENT_TYPES = [
    ctypes.c_int32,
    ctypes.c_int32,
    _INDEX_ARRAY,
    _INDEX_ARRAY,
    _TICK_ARRAY,
    _BIT_ARRAY,
    _BIT_ARRAY,
    _TICK_ARRAY,
    _TICK_ARRAY,
    _TICK_ARRAY,
]
# the gap the C writes when the growth admits no odd closed walk
_UNREACHABLE_GAP = -1


@dataclasses.dataclass(frozen=True)
class GrowthOutcome:
    """What one compiled decode leaves behind, in edge indices."""

    selected_edges: tuple
    edge_intervals: tuple
    contact_edges: tuple
    forest_edges: tuple
    # one GrowthStep per growth step, the cycle count's input
    growth_steps: tuple
    # the deepest parent chain of the trees the peel walked
    forest_depth: int


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
    step_edge_counts = numpy.zeros(edge_count, dtype=numpy.int32)
    step_hop_counts = numpy.zeros(edge_count, dtype=numpy.int32)
    step_growth_ticks = numpy.zeros(edge_count, dtype=numpy.int64)
    step_odd_fusions = numpy.zeros(edge_count, dtype=numpy.uint8)
    step_count = numpy.zeros(1, dtype=numpy.int32)
    forest_depth = numpy.zeros(1, dtype=numpy.int32)
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
        step_edge_counts,
        step_hop_counts,
        step_growth_ticks,
        step_odd_fusions,
        step_count,
        forest_depth,
    )
    _refuse_failure(status)
    selected_edges = _selected_edges(selected)
    edge_intervals = _intervals(is_closed, lower_tick, upper_tick)
    contact_edges = _prefix(contacts, contact_count)
    forest_edges = _prefix(forest, forest_count)
    growth_steps = _growth_steps(
        step_edge_counts,
        step_hop_counts,
        step_growth_ticks,
        step_odd_fusions,
        step_count,
    )
    depth = int(forest_depth[0])
    return GrowthOutcome(
        selected_edges=selected_edges,
        edge_intervals=edge_intervals,
        contact_edges=contact_edges,
        forest_edges=forest_edges,
        growth_steps=growth_steps,
        forest_depth=depth,
    )


def cluster_gap(
    graph: evidence_records.UnionFindGraph, edge_intervals: tuple
) -> Union[int, float]:
    """The shortest odd closed walk of one growth, in half ticks.

    Infinite when the growth admits no such walk, which is what a
    window whose logical row the growth never crosses leaves behind.
    """
    edge_count = len(graph.edges)
    _require_one_interval_per_edge(edge_count, edge_intervals)
    endpoint_a, endpoint_b, lengths = _graph_arrays(graph)
    parities = _logical_parities(graph)
    is_closed, lower_tick, upper_tick = _interval_arrays(edge_intervals)
    gap = numpy.zeros(1, dtype=numpy.int64)
    walk = cluster_gap_entry_point()
    status = walk(
        graph.detector_count,
        edge_count,
        endpoint_a,
        endpoint_b,
        lengths,
        parities,
        is_closed,
        lower_tick,
        upper_tick,
        gap,
    )
    _refuse_failure(status)
    half_ticks = int(gap[0])
    if half_ticks == _UNREACHABLE_GAP:
        return math.inf
    return half_ticks


@functools.cache
def entry_point():
    """The decode, bound once per process."""
    return _bound(_DECODE_SYMBOL, _DECODE_ARGUMENT_TYPES)


@functools.cache
def cluster_gap_entry_point():
    """The cluster gap walk, bound once per process."""
    return _bound(_CLUSTER_GAP_SYMBOL, _CLUSTER_GAP_ARGUMENT_TYPES)


def library_path() -> pathlib.Path:
    """The built library, or the one the environment names instead."""
    named = os.environ.get(LIBRARY_VARIABLE)
    if named:
        return pathlib.Path(named)
    here = pathlib.Path(__file__)
    folder = here.parent
    return folder / LIBRARY_FILE


def _bound(symbol: str, argument_types: list):
    """One exported function of the library, with its call declared."""
    path = library_path()
    if not path.exists():
        raise RuntimeError(
            "the Union-Find decoder needs its compiled library at "
            f"{path}; build it with {BUILD_COMMAND}"
        )
    library = ctypes.CDLL(str(path))
    function = getattr(library, symbol)
    function.argtypes = argument_types
    function.restype = ctypes.c_int32
    return function


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


def _require_one_interval_per_edge(
    edge_count: int, edge_intervals: tuple
) -> None:
    """The C reads one interval per edge and sizes nothing itself."""
    interval_count = len(edge_intervals)
    if interval_count == edge_count:
        return
    raise ValueError(
        "the Union-Find cluster gap walks one interval per edge: the "
        f"graph has {edge_count} edges and the growth left "
        f"{interval_count} intervals"
    )


def _logical_parities(graph: evidence_records.UnionFindGraph):
    """Each edge's logical row, the parity its first segment carries."""
    parities = []
    for edge in graph.edges:
        parities.append(edge.logical_observables[0])
    return numpy.asarray(parities, dtype=numpy.uint8)


def _interval_arrays(edge_intervals: tuple) -> tuple:
    """The growth as the three flat arrays the C reads, in edge order."""
    closed_flags = []
    lower_ticks = []
    upper_ticks = []
    for interval in edge_intervals:
        is_closed = isinstance(interval, evidence_records.Closed)
        closed_flags.append(is_closed)
        lower, upper = _open_bounds(interval, is_closed)
        lower_ticks.append(lower)
        upper_ticks.append(upper)
    first = numpy.asarray(closed_flags, dtype=numpy.uint8)
    second = numpy.asarray(lower_ticks, dtype=numpy.int64)
    third = numpy.asarray(upper_ticks, dtype=numpy.int64)
    return first, second, third


def _open_bounds(interval, is_closed: bool) -> tuple:
    """A closed edge has no uncovered span, and its bounds go unread."""
    if is_closed:
        return 0, 0
    return interval.lower_tick, interval.upper_tick


def _growth_steps(
    step_edge_counts,
    step_hop_counts,
    step_growth_ticks,
    step_odd_fusions,
    step_count,
) -> tuple:
    edge_counts = _prefix(step_edge_counts, step_count)
    hop_counts = _prefix(step_hop_counts, step_count)
    growth_ticks = _prefix(step_growth_ticks, step_count)
    odd_fusions = _prefix(step_odd_fusions, step_count)
    rows = zip(edge_counts, hop_counts, growth_ticks, odd_fusions)
    steps = []
    for edge_count, hop_count, ticks, odd_fusion in rows:
        step = evidence_records.GrowthStep(
            edge_count, hop_count, ticks, bool(odd_fusion)
        )
        steps.append(step)
    return tuple(steps)


def _prefix(values, count) -> tuple:
    taken = values[: count[0]]
    indices = []
    for value in taken:
        indices.append(int(value))
    return tuple(indices)
