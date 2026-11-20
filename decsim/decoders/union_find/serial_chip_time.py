"""A serial union-find chip's time: the cycles of its loop.

PairCompareCount is the loop of Barber et al. 2309.05558, which keeps
an entry a defect and compares every pair of them: the loop is run on
the decode's own graph and defects (pair_compare_loop.py), and each of
its operations costs the cycles the run gives it.

The design keeps one chip per stabilizer type, the two working side by
side, so a decode is held for the slower of the two (2309.05558 Fig. 5:
disjoint decoding graphs decode X and Z errors). The loop's chips are
the connected parts of the decode's graph.
"""

import dataclasses
import math
from typing import Optional

import numpy

import decsim.config as config
import decsim.decoders.union_find.pair_compare_loop as pair_compare_loop
import decsim.records.decoder_evidence as evidence_records

# the paper prices no growth past the decode
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
