"""The evidence a decode returns beside its correction, for a signal to read.

A confidence signal reads the decoder's own numbers rather than running
a second decode, so the record that carries them crosses the Decoder
port (decoding.CLUSTER_GROWTH_EVIDENCE) and lives here rather than
inside a decoder package. Lampson's "don't hide power" is the reason the
window exists at all: "the client of an abstraction ... should not be
denied the ability to use the power of the implementation"
(lampson1983.txt 298-303); STYLE.md rule 6 is the reason it lives here.

The weighted union-find growth is the one family so far. Its shape is
Delfosse and Nickerson 1709.06218 as weighted by Huang, Newman and Brown
2004.04693: an edge of log-odds weight w has integer length round(w /
weight_step), and a growing front covers it half a tick at a time, so
what a decode leaves behind is an interval per edge, open where the
growth did not reach and closed where it did. The algorithm that grows
them stays in decsim/decoders/union_find/window_decoder.py.
"""

import dataclasses
import math
from numbers import Real
from typing import Union

# one tick of an edge length is this many natural-log units of weight
DEFAULT_WEIGHT_STEP = 0.1


@dataclasses.dataclass(frozen=True)
class Open:
    """The uncovered interval between two growing edge fronts."""

    lower_tick: int
    upper_tick: int


@dataclasses.dataclass(frozen=True)
class Closed:
    """A fully covered edge."""


@dataclasses.dataclass(frozen=True)
class UnionFindEdge:
    """One graphlike residual fault column in the weighted graph."""

    fault_index: int
    detector_a: int
    detector_b: int
    logical_observables: tuple[int, ...]
    length_half_ticks: int


@dataclasses.dataclass(frozen=True)
class UnionFindGraph:
    """Immutable graph state shared by decodes of one placed fault model."""

    detector_count: int
    fault_count: int
    edges: tuple[UnionFindEdge, ...]
    baseline_faults: tuple[int, ...]
    baseline_syndrome: tuple[int, ...]
    logical_observables_by_fault: tuple[tuple[int, ...], ...] = ()
    logical_observable_count: int = 0
    # the absolute natural-log weight one tick of an edge length is, so
    # a reader of the growth knows what its distances mean
    weight_step: float = DEFAULT_WEIGHT_STEP


@dataclasses.dataclass(frozen=True)
class UnionFindHardEvidence:
    """Immutable weighted growth and peeling evidence from one hard decode."""

    graph: UnionFindGraph
    syndrome: tuple[int, ...]
    residual_syndrome: tuple[int, ...]
    selected_faults: tuple[int, ...]
    contact_faults: tuple[int, ...]
    edge_intervals: tuple[Union[Open, Closed], ...]
    erasure_forest_faults: tuple[int, ...]
    logical_observables: tuple[int, ...]
    # detectors the best-effort correction leaves unexplained: an odd
    # cluster that ran out of edges before reaching another defect or the
    # boundary
    unmatched_detectors: tuple[int, ...] = ()


def normalized_weight_step(weight_step) -> float:
    """The weight step as a positive finite float; anything else is refused."""
    if isinstance(weight_step, bool) or not isinstance(weight_step, Real):
        raise TypeError("Union-Find weight_step must be a real number")
    normalized = float(weight_step)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError("Union-Find weight_step must be finite and positive")
    return normalized
