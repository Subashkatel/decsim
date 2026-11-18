"""A serial union-find chip's time: from its memory reads, or its loop.

A chip that keeps its graph in memory tables and works through its
clusters one at a time spends its time on memory reads, and two
published designs price it.

ClusterReadCount is the read count of Das et al. 2001.06598 Sec. III C,
Eq. 5 and 6 (the same law in their HPCA 2022 paper, Sec. IV E). A growth
step reads the rows of the memory that cover the cluster, about the
square of its diameter, so growing a cluster to diameter D costs
1 + 4 + ... + D^2 reads; the spanning tree search and the correction
each read the cluster's vertices once. The paper counts on the clusters
the growth ends with, as this does, and holds a read for four clock
cycles.

PairCompareCount is the loop of Barber et al. 2309.05558, which keeps
an entry a defect and compares every pair of them: the loop is run on
the decode's own graph and defects (pair_compare_loop.py), and each of
its operations costs the cycles the run gives it.

Both keep one chip per stabilizer type, the two working side by side,
so a decode is held for the slower of the two (2001.06598 Sec. III: two
decoders per logical qubit, one for each type of error; 2309.05558
Fig. 5: disjoint decoding graphs decode X and Z errors). The read count
reads a detector's type and place off its coordinates: Stim's rotated
memory places stabilizers at even x and y with the two types
alternating, so (x + y) / 2 is even for one type and odd for the other,
one type's stabilizers are one step apart along the diagonals
(x + y) / 4 and (x - y) / 4, and the third coordinate counts rounds.
The loop's chips are the connected parts of the decode's graph.
"""

import dataclasses
import math
from typing import Optional

import numpy

import decsim.config as config
import decsim.decoders.cluster_record as cluster_record
import decsim.decoders.union_find.pair_compare_loop as pair_compare_loop
import decsim.records.decoder_evidence as evidence_records

# the two stabilizer types, a chip each
STABILIZER_TYPE_COUNT = 2
# the spanning tree search and the correction each read a vertex once
# (2001.06598 Eq. 6)
WALK_READS_PER_VERTEX = 2
# neither paper prices growth past the decode
EXTRA_GROWTH_REFUSAL = (
    "a serial chip's time law prices the decode only; declare the time "
    "of the extra-cluster gap's growth (walk_microseconds)"
)
# PairCompareCount's costs, each in cycles for one of its operations
OPERATION_COST_FIELDS = (
    "load_cycles",
    "load_cycles_per_word",
    "load_cycles_per_defect",
    "grow_cycles_per_pass",
    "grow_cycles_per_entry",
    "grow_cycles_per_grown_entry",
    "grow_cycles_per_root_step",
    "grow_cycles_per_boundary_hit",
    "match_cycles_per_step",
    "match_cycles_per_pair",
    "match_cycles_per_first_collision",
    "match_cycles_per_boundary_hit",
    "union_cycles_per_step",
)


@dataclasses.dataclass(frozen=True)
class ClusterReadCount:
    """A union-find unit's time as the memory reads of its final clusters.

    reads = sum over the clusters that hold a detection event of
            (1 + 4 + ... + diameter^2) + 2 x the cluster's detectors

    for each stabilizer type's chip apart, and the unit is held for
    cycles_per_read cycles a read of the chip with more reads. A
    cluster is the detectors the growth joined by fully covered edges
    (cluster_record.detector_roots). The paper says neither what the
    diameter is nor how the three stages add to one decode's time. Here
    the stages follow one another, and the diameter is the cluster's
    widest extent along the grid's three axes plus one, the half edges
    at its two ends. That reading gives the paper's mean of 42 ns for
    its own case (tests/decoders/test_union_find_serial_chip_time.py).

    The law has no fixed cost: a decode with no detection event takes no
    time. It prices the decode only: the extra growth of the
    extra-cluster gap is refused.
    """

    clock: config.Clock
    cycles_per_read: int

    def __post_init__(self) -> None:
        config.check_cycles(
            "cluster_read_count.cycles_per_read", self.cycles_per_read
        )

    def reads(self, evidence: evidence_records.UnionFindHardEvidence) -> list:
        """The reads of each stabilizer type's chip for one decode."""
        coordinates = _coordinates_of(evidence)
        clusters = _clusters(evidence)
        reads = [0] * STABILIZER_TYPE_COUNT
        for cluster in clusters:
            diameter = _diameter(cluster, coordinates)
            growth_reads = _growth_reads(diameter)
            walk_reads = WALK_READS_PER_VERTEX * len(cluster)
            first_detector = cluster[0]
            chip = _stabilizer_type(coordinates[first_detector])
            reads[chip] += growth_reads
            reads[chip] += walk_reads
        return reads

    def decode_ticks(
        self,
        evidence: Optional[evidence_records.UnionFindHardEvidence],
        elapsed_nanoseconds: int,
        now: int,
    ) -> int:
        """The ticks from now to the clock edge the slower chip ends on."""
        del elapsed_nanoseconds
        if evidence is None:
            return 0
        reads = self.reads(evidence)
        cycles = self.cycles_per_read * max(reads)
        if cycles == 0:
            return 0
        return self.clock.ticks_to_edge(cycles, now)

    def extra_growth_ticks(self, steps: tuple, elapsed_nanoseconds: int) -> int:
        """Refused: the law prices no growth past the decode."""
        del steps
        del elapsed_nanoseconds
        raise ValueError(EXTRA_GROWTH_REFUSAL)


@dataclasses.dataclass(frozen=True)
class PairCompareCount:
    """A union-find unit's time as the cycles of its loop over defect pairs.

    The loop (pair_compare_loop.passes_of) runs on the decode's own
    graph and defects, a chip for each connected part of the graph, and
    the unit is held for the cycles of the chip that takes longest,
    rounded up:

    cycles = load_cycles + load_cycles_per_word x the syndrome's words
           + load_cycles_per_defect x defects
           + sum over passes of
               grow_cycles_per_pass
               + grow_cycles_per_entry x entries
               + grow_cycles_per_grown_entry x entries grown
               + grow_cycles_per_root_step x parent-table steps
               + grow_cycles_per_boundary_hit x entries at the boundary
           + sum over the passes that grow of
               match_cycles_per_step + union_cycles_per_step
               + match_cycles_per_pair x pairs compared
               + match_cycles_per_first_collision x pairs newly met
               + match_cycles_per_boundary_hit x entries at the boundary

    The first line is the paper's Init: a chip loads its syndrome a
    word of syndrome_word_bits detectors at a time and pushes an entry
    a defect. A decode with no defect still costs a chip its load and
    one Grow pass. The paper prints no cycles of an operation, so the
    costs have no default: a run writes them, with their source. With
    weights the loop runs more passes and an operation costs what it
    costs without them. The paper measures no weights: its Appendix E
    gives a weighted distance for one weight a direction and leaves
    circuit noise out. The law prices the decode only: the extra growth
    of the extra-cluster gap is refused.
    """

    clock: config.Clock
    syndrome_word_bits: int
    load_cycles: float
    load_cycles_per_word: float
    load_cycles_per_defect: float
    grow_cycles_per_pass: float
    grow_cycles_per_entry: float
    grow_cycles_per_grown_entry: float
    grow_cycles_per_root_step: float
    grow_cycles_per_boundary_hit: float
    match_cycles_per_step: float
    match_cycles_per_pair: float
    match_cycles_per_first_collision: float
    match_cycles_per_boundary_hit: float
    union_cycles_per_step: float

    def __post_init__(self) -> None:
        config.check_whole_count(
            "pair_compare_count.syndrome_word_bits",
            self.syndrome_word_bits,
            "bits",
        )
        for name in OPERATION_COST_FIELDS:
            value = getattr(self, name)
            _check_cost(name, value)

    def cycles(self, evidence: evidence_records.UnionFindHardEvidence) -> list:
        """The cycles of each chip for one decode."""
        chips = pair_compare_loop.chips_of(evidence.graph)
        fired = numpy.asarray(evidence.residual_syndrome, dtype=bool)
        cycles = []
        for chip in chips:
            chip_fired = fired[chip.detectors]
            defects = numpy.flatnonzero(chip_fired)
            passes, _ = pair_compare_loop.passes_of(chip, defects)
            detector_count = len(chip.detectors)
            chip_cycles = _load_cycles(self, len(defects), detector_count)
            for one_pass in passes:
                chip_cycles += _pass_cycles(self, one_pass)
            cycles.append(chip_cycles)
        return cycles

    def decode_ticks(
        self,
        evidence: Optional[evidence_records.UnionFindHardEvidence],
        elapsed_nanoseconds: int,
        now: int,
    ) -> int:
        """The ticks from now to the clock edge the slower chip ends on."""
        del elapsed_nanoseconds
        if evidence is None:
            return 0
        cycles = self.cycles(evidence)
        if not cycles:
            return 0
        slowest = max(cycles)
        whole_cycles = math.ceil(slowest)
        return self.clock.ticks_to_edge(whole_cycles, now)

    def extra_growth_ticks(self, steps: tuple, elapsed_nanoseconds: int) -> int:
        """Refused: the law prices no growth past the decode."""
        del steps
        del elapsed_nanoseconds
        raise ValueError(EXTRA_GROWTH_REFUSAL)


def _coordinates_of(evidence: evidence_records.UnionFindHardEvidence) -> tuple:
    coordinates = evidence.detector_coordinates
    if coordinates is None:
        raise ValueError(
            "a serial chip's time is read off where the detectors lie; "
            "this decode's window carries no detector coordinates"
        )
    return coordinates


def _clusters(evidence: evidence_records.UnionFindHardEvidence) -> list:
    """The detectors of each final cluster that holds a detection event."""
    roots = cluster_record.detector_roots(
        evidence.graph, evidence.edge_intervals
    )
    events_by_root = cluster_record.cluster_events(
        roots, evidence.residual_syndrome
    )
    detectors_by_root = {root: [] for root in events_by_root}
    for detector, root in enumerate(roots):
        if root in detectors_by_root:
            detectors_by_root[root].append(detector)
    clusters = detectors_by_root.values()
    return list(clusters)


def _diameter(cluster: list, coordinates: tuple) -> int:
    """The cluster's widest extent in grid steps, plus one."""
    places = [_grid_place(coordinates[detector]) for detector in cluster]
    widest = 0.0
    for axis_places in zip(*places, strict=True):
        extent = max(axis_places) - min(axis_places)
        widest = max(widest, extent)
    return round(widest) + 1


def _growth_reads(diameter: int) -> int:
    """1 + 4 + ... + diameter^2: the rows each growth step reads."""
    step_count = diameter + 1
    step_reads = [step * step for step in range(1, step_count)]
    return sum(step_reads)


def _stabilizer_type(coordinate: tuple) -> int:
    """Which of the two types a detector at this coordinate is."""
    diagonal = coordinate[0] + coordinate[1]
    half_diagonal = diagonal / 2
    return round(half_diagonal) % STABILIZER_TYPE_COUNT


def _grid_place(coordinate: tuple) -> tuple:
    """The detector's place in steps of its type's grid, and its round."""
    along = (coordinate[0] + coordinate[1]) / 4
    across = (coordinate[0] - coordinate[1]) / 4
    return along, across, coordinate[2]


def _check_cost(name: str, value: object) -> None:
    is_finite = config.is_number(value) and math.isfinite(value)
    if is_finite and value >= 0:
        return
    raise ValueError(
        f"pair_compare_count.{name} must be a finite nonnegative number "
        f"(got {value!r})"
    )


def _load_cycles(
    costs: PairCompareCount, defect_count: int, detector_count: int
) -> float:
    """A chip's cycles before its first pass: its syndrome, word by word."""
    words = detector_count / costs.syndrome_word_bits
    word_count = math.ceil(words)
    cycles = costs.load_cycles
    cycles += costs.load_cycles_per_word * word_count
    cycles += costs.load_cycles_per_defect * defect_count
    return cycles


def _pass_cycles(
    costs: PairCompareCount, one_pass: pair_compare_loop.Pass
) -> float:
    """One pass's cycles: its Grow, and its Match and Union when it grows."""
    cycles = costs.grow_cycles_per_pass
    cycles += costs.grow_cycles_per_entry * one_pass.entry_count
    cycles += costs.grow_cycles_per_grown_entry * one_pass.grown_entry_count
    cycles += costs.grow_cycles_per_root_step * one_pass.root_step_count
    boundary_hit_count = one_pass.boundary_hit_count
    cycles += costs.grow_cycles_per_boundary_hit * boundary_hit_count
    if one_pass.grown_entry_count == 0:
        return cycles
    cycles += costs.match_cycles_per_step + costs.union_cycles_per_step
    cycles += costs.match_cycles_per_pair * one_pass.compared_pair_count
    first_collision_count = one_pass.first_collision_count
    cycles += costs.match_cycles_per_first_collision * first_collision_count
    cycles += costs.match_cycles_per_boundary_hit * boundary_hit_count
    return cycles
