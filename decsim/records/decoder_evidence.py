"""The evidence a decode returns beside its correction, for a signal to read.

A confidence signal reads the decoder's own numbers rather than running
a second decode, so the record crosses the Decoder port and lives here,
not inside a decoder package: "the client of an abstraction ... should
not be denied the ability to use the power of the implementation"
(Lampson 1983).

The weighted union-find growth is Delfosse and Nickerson 1709.06218 as
weighted by Huang, Newman and Brown 2004.04693: an edge of log-odds
weight w has integer length round(w / weight_step), and a front covers
it half a tick at a time, so a decode leaves an interval per edge.
"""

import dataclasses
import math
import numbers
from typing import Union

# one tick of an edge length is this many natural-log units of weight
DEFAULT_WEIGHT_STEP = 0.1

# what the strongest fusion of one growth step changed
FUSION_NONE = "none"
FUSION_ROOTS = "roots"
FUSION_PARITY = "parity"
FUSION_KINDS = (FUSION_NONE, FUSION_ROOTS, FUSION_PARITY)


@dataclasses.dataclass(frozen=True)
class Open:
    """The uncovered interval between two growing edge fronts."""

    lower_tick: int
    upper_tick: int


@dataclasses.dataclass(frozen=True)
class Closed:
    """A fully covered edge."""


@dataclasses.dataclass(frozen=True)
class GrowthStep:
    """One growth step's inputs to the cycle count.

    edge_count is the step's work. hop_count is its critical path, the
    deepest flood over closed edges from the lowest detector of any
    cluster the step fused, the boundary left out: the stages a cluster
    identifier and parity take to cross it (Helios 2301.08419 lines
    623-629). growth_ticks is the one-unit growth iterations a unit
    spends (lines 1242-1254). fusion is the strongest kind among the
    step's fusions: none, roots (only roots and the boundary flag moved),
    or parity (an odd absorbed root's parity must cross the fused
    cluster).
    """

    edge_count: int
    hop_count: int
    growth_ticks: int
    fusion: str


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
    """The evidence one hard union-find decode leaves.

    It covers the weighted growth and the peel.
    """

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
    # one GrowthStep per growth step, in order, for the cycle count
    growth_steps: tuple[GrowthStep, ...] = ()
    # the deepest parent chain of the trees the peel walked, a root at
    # zero: the levels the peel's flags and completions cross
    forest_depth: int = 0


def normalized_weight_step(
    weight_step: object, key: str = "Union-Find weight_step"
) -> float:
    """The weight step as a positive finite float; anything else is refused.

    key is the name the refusal gives the value.
    """
    if isinstance(weight_step, bool) or not isinstance(
        weight_step, numbers.Real
    ):
        raise ValueError(f"{key} must be a real number")
    normalized = float(weight_step)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{key} must be finite and positive")
    return normalized
