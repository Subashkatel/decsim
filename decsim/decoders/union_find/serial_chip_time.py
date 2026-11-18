"""A serial union-find chip's time: from its memory reads, or its defects.

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

DefectCycleCount is a law in the defects alone,

    cycles = cycles_per_round x rounds + cycles_per_defect x defects
           + cycles_per_squared_defect x defects^2,

after the work Barber et al. 2309.05558 Sec. II A name for their
design: a read and a write per defect a growth step, and a comparison
per pair of defects. The paper prints no law and no cycle counts, so
the three numbers are a fit to its published times, and a run supplies
them with their source.

Both keep one chip per stabilizer type, the two working side by side,
so a decode is held for the slower of the two (2001.06598 Sec. III: two
decoders per logical qubit, one for each type of error; 2309.05558
Fig. 5: disjoint decoding graphs decode X and Z errors). A detector's
type and place are read off its coordinates: Stim's rotated memory
places stabilizers at even x and y with the two types alternating, so
(x + y) / 2 is even for one type and odd for the other, one type's
stabilizers are one step apart along the diagonals (x + y) / 4 and
(x - y) / 4, and the third coordinate counts rounds.
"""

import dataclasses
import math
from typing import Optional

import decsim.config as config
import decsim.decoders.cluster_record as cluster_record
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
class DefectCycleCount:
    """A union-find unit's time as a law in each chip's defects.

    cycles = cycles_per_round x rounds + cycles_per_defect x defects
           + cycles_per_squared_defect x defects^2, rounded up,

    with the defects of the stabilizer type that has more of them, and
    the rounds the window's detectors span. A decode with no defect
    still costs its rounds. It prices the decode only: the extra growth
    of the extra-cluster gap is refused.
    """

    clock: config.Clock
    cycles_per_round: float
    cycles_per_defect: float
    cycles_per_squared_defect: float

    def __post_init__(self) -> None:
        _check_cycle_rate("cycles_per_round", self.cycles_per_round)
        _check_cycle_rate("cycles_per_defect", self.cycles_per_defect)
        _check_cycle_rate(
            "cycles_per_squared_defect", self.cycles_per_squared_defect
        )

    def cycles(self, evidence: evidence_records.UnionFindHardEvidence) -> int:
        """The cycles of the chip with more defects for one decode."""
        coordinates = _coordinates_of(evidence)
        defects = _defects_by_type(evidence.residual_syndrome, coordinates)
        most_defects = max(defects)
        rounds = {coordinate[2] for coordinate in coordinates}
        round_cycles = self.cycles_per_round * len(rounds)
        defect_cycles = self.cycles_per_defect * most_defects
        squared_defects = most_defects * most_defects
        pair_cycles = self.cycles_per_squared_defect * squared_defects
        cycles = round_cycles + defect_cycles + pair_cycles
        return math.ceil(cycles)

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
        if cycles == 0:
            return 0
        return self.clock.ticks_to_edge(cycles, now)

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


def _defects_by_type(residual_syndrome: tuple, coordinates: tuple) -> list:
    defects = [0] * STABILIZER_TYPE_COUNT
    for detector, bit in enumerate(residual_syndrome):
        if not bit:
            continue
        chip = _stabilizer_type(coordinates[detector])
        defects[chip] += 1
    return defects


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


def _check_cycle_rate(name: str, value: object) -> None:
    is_finite = config.is_number(value) and math.isfinite(value)
    if is_finite and value >= 0:
        return
    raise ValueError(
        f"defect_cycle_count.{name} must be finite and not negative "
        f"(got {value!r})"
    )
