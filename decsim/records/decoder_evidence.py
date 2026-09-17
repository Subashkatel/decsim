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
    """One growth step's cycle count inputs: its work and its critical path.

    edge_count is the boundary edges the step advanced. hop_count is the
    deepest flood over closed edges from the root of any cluster the step
    fused, the stages a cluster identifier and its parity take to cross
    the cluster (Helios 2301.08419 lines 623-629); zero when the step
    fused nothing. growth_ticks is the ticks the step spanned, which is
    the one-unit growth iterations a unit that grows one unit of weight
    at a time spends on it (lines 1053-1063, latency growing with the
    weight resolution). fusion is the strongest kind among the step's
    own fusions, one of FUSION_KINDS: none when the closing edges united
    no two clusters, roots when clusters united and only roots and the
    touching-boundary flag moved, parity when the survivor's parity took
    an odd absorbed root's and has to cross the cluster it fused.
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
    # one GrowthStep per growth step, in order, for the cycle count
    growth_steps: tuple[GrowthStep, ...] = ()
    # the deepest parent chain of the trees the peel walked, a root at
    # zero: the levels the peel's flags and completions cross
    forest_depth: int = 0


def normalized_weight_step(weight_step) -> float:
    """The weight step as a positive finite float; anything else is refused."""
    if isinstance(weight_step, bool) or not isinstance(weight_step, Real):
        raise TypeError("Union-Find weight_step must be a real number")
    normalized = float(weight_step)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError("Union-Find weight_step must be finite and positive")
    return normalized
