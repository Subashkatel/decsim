"""The Union-Find decoder's cycle count per growth tick, on one clock.

A hardware union-find decoder repeats one iteration, grow then merge,
until no cluster is odd, and then peels; its time is what that loop and
that peel cost on its own clock. The law below is traced through
Helios's register transfer level, github.com/yale-paragon/
Helios_scalable_QEC at 2dda998, the design behind arXiv 2301.08419v2:
control_node_single_FPGA.v lines 74-75 and 82-85 (the registered busy
and the counter), 152-223 (loading, GROW, MERGE, PEELING);
processing_unit_single_FPGA_v2.v lines 104-105, 122, 140-184 (an
element's stage, its growth and its merge), 236-266 and 313-333 (the
peel and busy); neighbor_link_internal_v2.v lines 78-102 and 130 (a
front covers one half tick per tick per growing end, and an edge is
grown at its weight). The steps come from the decode itself
(records/decoder_evidence.py, GrowthStep); the row's constants come
from the yaml's cycle_count block, one row per chip, and the cycles end
on the named clock's edge as every stage of the decoder unit does.
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
    "delay_cycles",
)

# the cycle the counter starts at, the loading cycle in which the
# machine takes its last byte of syndrome
COUNTER_START = 1
# one cycle grows every front and one cycle the controller decides in
GROW_AND_DECIDE_CYCLES = 2
# the first peeling cycle, in which every element reports busy until its
# parity flag is set, and the cycle the controller decides in
PEEL_BUSY_AND_DECIDE_CYCLES = 2
# the flags descend one level a cycle and completion climbs back
PEEL_CYCLES_PER_LEVEL = 2
# the cycle in which the controller asks whether the two boundaries
# share a root, once an iteration of the extra growth (Kishi
# 2602.03336 Algorithm 1 tests the join after every iteration); Helios
# has no such compare, so the one cycle is this design's own
JOIN_TEST_CYCLES = 1


@dataclasses.dataclass(frozen=True)
class CycleCount:
    """The yaml's `cycle_count` block under a tier that names union_find.

    cycles = 1 + setup_cycles + setup_cycles_per_vertex x detectors
           + setup_cycles_per_edge x edges
           + (2 + delay_cycles) x max(1, the growth ticks of every step)
           + sum over steps of max(the step's changes,
                                   ceil(cycles_per_edge x its edges
                                        x its growth ticks))
           + delay_cycles + 2 + 2 x the peel's depth.

    The leading 1 is the cycle the counter starts at. An iteration of
    the loop is one cycle in which every front grows, one in which the
    controller decides whether to grow again, and delay_cycles between
    them: the clocked handoffs an element's change crosses before the
    controller sees it, which on Helios are three, the element's stage
    register, its own busy register and the controller's. A unit grows
    one unit of weight an iteration, so a step that spans several growth
    ticks is that many iterations, and one quiet iteration runs even for
    an empty syndrome.

    A step's changes are what its fusions ripple through the cluster
    they fused, a level a cycle: nothing when its closing edges united
    no two clusters, its deepest flood when only roots and the touching
    flag move, and one for the root plus two a level when a parity moves
    (the parity climbs while the odd flag descends). A step's hop count
    is the deepest flood among the clusters it fused and its fusion the
    strongest of them, so a step that joins a deep even cluster and
    flips a parity in a shallow one is charged at that ceiling.
    cycles_per_edge is zero for a unit with a processing element per
    vertex, where the whole step's work runs in its critical path, and
    the cycles per boundary edge for a unit that walks its edges through
    a memory port; the step then costs the larger of the two.

    The peel pays the same delay, a cycle for the controller's decision
    and a cycle in which every element reports busy until its parity
    flag is set, then two cycles a level, the flags descending and
    completion climbing back.

    The setup terms are the structural floor of a unit that lays its
    graph out before it grows, which Helios, holding its graph in
    registers, does not pay; the rounds loaded are the unit's fetch
    stage and not counted here.

    The extra growth of the extra-cluster gap (Kishi 2602.03336
    Algorithm 1) runs the same loop on after the
    decode, so its cycles are the iteration's, with one more cycle a
    tick for the join test, plus its steps' changes, and no counter
    start, setup or peel:

    extra growth cycles = (2 + delay_cycles + 1) x the growth ticks
                        + sum over steps of max(the step's changes,
                                                its spread work)

    and one join test cycle when no tick was grown at all.

    The law is checked against Helios's published points outside the
    repo, with nothing fitted.
    """

    clock: config.Clock
    delay_cycles: int = 0
    cycles_per_edge: float = 0.0
    setup_cycles: int = 0
    setup_cycles_per_vertex: int = 0
    setup_cycles_per_edge: int = 0

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
        steps = evidence.growth_steps
        iterations = self._iteration_cycles(steps)
        merges = self._merge_cycles(steps)
        peel = self._peel_cycles(evidence.forest_depth)
        counted = iterations + merges + peel
        return COUNTER_START + setup + counted

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

    def extra_growth_cycles(self, steps: tuple) -> int:
        """The cycles the extra growth's events cost after the decode."""
        growth_ticks = 0
        for step in steps:
            growth_ticks += step.growth_ticks
        if growth_ticks == 0:
            return JOIN_TEST_CYCLES
        iteration = GROW_AND_DECIDE_CYCLES + self.delay_cycles
        iteration += JOIN_TEST_CYCLES
        merges = self._merge_cycles(steps)
        return iteration * growth_ticks + merges

    def extra_growth_ticks(self, steps: tuple) -> int:
        """The ticks those cycles span from the edge the decode ended on."""
        cycles = self.extra_growth_cycles(steps)
        return cycles * self.clock.period_ticks

    def _iteration_cycles(self, steps: tuple) -> int:
        """Grow, wait and decide, once per growth tick the decode spans."""
        growth_ticks = 0
        for step in steps:
            growth_ticks += step.growth_ticks
        iterations = max(1, growth_ticks)
        floor = GROW_AND_DECIDE_CYCLES + self.delay_cycles
        return floor * iterations

    def _merge_cycles(self, steps: tuple) -> int:
        merges = 0
        for step in steps:
            step_cycles = self._step_merge_cycles(step)
            merges += step_cycles
        return merges

    def _step_merge_cycles(self, step: evidence_records.GrowthStep) -> int:
        """The larger of one step's rippling changes and its spread work."""
        changes = _fusion_changes(step)
        edge_work = self.cycles_per_edge * step.edge_count
        spread = edge_work * step.growth_ticks
        work = math.ceil(spread)
        return max(changes, work)

    def _peel_cycles(self, forest_depth: int) -> int:
        level_cycles = PEEL_CYCLES_PER_LEVEL * forest_depth
        floor = self.delay_cycles + PEEL_BUSY_AND_DECIDE_CYCLES
        return floor + level_cycles


def _fusion_changes(step: evidence_records.GrowthStep) -> int:
    """The cycles one step's fusions ripple through the cluster they fused.

    A fusion that moves no parity settles a root and the touching flag a
    level a cycle. One that moves a parity moves the root once and then
    climbs the parity and descends the odd flag a level a cycle, and
    those two floods overlap the root's own descent, so a chain of depth
    D settles in 1 + 2 D changes.
    """
    if step.fusion == evidence_records.FUSION_NONE:
        return 0
    if step.fusion == evidence_records.FUSION_ROOTS:
        return max(1, step.hop_count)
    climbs = 2 * step.hop_count
    return 1 + climbs


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
