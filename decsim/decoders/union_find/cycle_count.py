"""The Union-Find decoder's cycle count per growth step, on one clock.

A hardware union-find decoder repeats one step, grow then merge, until
no cluster is odd; its time is the steps it took, each priced as the
larger of its critical path and its work spread over the unit's width,
between a setup that scales with the graph and a drain (Helios
2301.08419 lines 623-629 and 694-699: growing takes one cycle, and the
flood that merges a cluster takes as many stages as the cluster is
deep; AFS 2001.06598 lines 536 and 1110: a row that prices its memory
access charges the read's latency per unit of work). The steps come
from the decode itself (records/decoder_evidence.py, GrowthStep); the
row's constants come from the yaml's cycle_count block, one row per
chip, and the cycles end on the named clock's edge as every stage of
the decoder unit does.
"""

import dataclasses
import math
from collections.abc import Mapping
from numbers import Real
from typing import Optional

import decsim.config as config
import decsim.records.decoder_evidence as evidence_records

CYCLE_FIELDS = (
    "setup_cycles",
    "setup_cycles_per_vertex",
    "setup_cycles_per_edge",
    "cycles_per_step",
    "cycles_per_hop",
    "drain_cycles",
)


@dataclasses.dataclass(frozen=True)
class CycleCount:
    """The yaml's `cycle_count` block under a tier that names union_find.

    cycles = setup_cycles + setup_cycles_per_vertex x detectors
           + setup_cycles_per_edge x edges
           + sum over steps of max(cycles_per_step + cycles_per_hop x hops,
                                   ceil(cycles_per_edge x edges of the step))
           + drain_cycles.

    The setup is the structural floor: the vertices reset and the edge
    intervals laid before the first step, which on the traced C is most
    of an empty decode; the rounds loaded are the unit's fetch stage
    and not counted here. cycles_per_edge is zero for a unit with a
    processing element per vertex, where the whole step's work runs in
    its critical path, and the cycles per boundary edge for a unit that
    walks its edges through a memory port.

    A step is one growth event of the weighted growth, which advances
    every growing cluster to the next edge that closes; a unit that
    grows one unit of weight per iteration takes as many iterations as
    the event spans (Helios 2301.08419 lines 1053-1063, latency growing
    with the weight resolution), which this count prices as one step.
    On unweighted graphs, where every interior event is one iteration,
    the count meets Helios's published points within three percent; on
    weighted graphs it runs short.
    """

    clock: config.Clock
    setup_cycles: int = 0
    setup_cycles_per_vertex: int = 0
    setup_cycles_per_edge: int = 0
    cycles_per_step: int = 0
    cycles_per_hop: int = 0
    cycles_per_edge: float = 0.0
    drain_cycles: int = 0

    def __post_init__(self) -> None:
        for name in CYCLE_FIELDS:
            value = getattr(self, name)
            config.check_cycles(f"cycle_count.{name}", value)
        _check_cycles_per_edge(self.cycles_per_edge)

    @classmethod
    def from_yaml(
        cls, section: Mapping, clocks: config.ClockSettings
    ) -> "CycleCount":
        """The block's fields, its clock resolved to that domain's Clock."""
        _check_keys(section)
        clock = clocks.clock(section["clock"])
        fields = {}
        for name in CYCLE_FIELDS:
            fields[name] = section.get(name, 0)
        cycles_per_edge = section.get("cycles_per_edge", 0.0)
        return cls(clock, cycles_per_edge=cycles_per_edge, **fields)

    def cycles(
        self, evidence: Optional[evidence_records.UnionFindHardEvidence]
    ) -> int:
        """The whole decode's cycles; a decode with no graph costs nothing."""
        if evidence is None:
            return 0
        graph = evidence.graph
        edge_count = len(graph.edges)
        vertex_cycles = self.setup_cycles_per_vertex * graph.detector_count
        edge_cycles = self.setup_cycles_per_edge * edge_count
        setup = self.setup_cycles + vertex_cycles + edge_cycles
        steps = 0
        for step in evidence.growth_steps:
            step_cycles = self._step_cycles(step)
            steps += step_cycles
        return setup + steps + self.drain_cycles

    def ticks(
        self,
        evidence: Optional[evidence_records.UnionFindHardEvidence],
        now: int,
    ) -> int:
        """The ticks from now to the clock edge the decode's cycles end on."""
        cycles = self.cycles(evidence)
        if cycles == 0:
            return 0
        edge = self.clock.edge(cycles, now)
        return edge - now

    def _step_cycles(self, step: evidence_records.GrowthStep) -> int:
        hop_cycles = self.cycles_per_hop * step.hop_count
        critical_path = self.cycles_per_step + hop_cycles
        spread = self.cycles_per_edge * step.edge_count
        work = math.ceil(spread)
        return max(critical_path, work)


def _check_keys(section: Mapping) -> None:
    known = set(CYCLE_FIELDS)
    known.add("clock")
    known.add("cycles_per_edge")
    unknown = set(section) - known
    if unknown:
        listed = sorted(unknown)
        raise ValueError(
            f"cycle_count has no key {listed}; its keys are clock, "
            f"cycles_per_edge and {list(CYCLE_FIELDS)}"
        )


def _check_cycles_per_edge(value) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("cycle_count.cycles_per_edge must be a number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            "cycle_count.cycles_per_edge must be a finite nonnegative number"
        )
