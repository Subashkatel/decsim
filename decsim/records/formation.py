"""A detector's recipe, and the table of every recipe of one circuit.

A recipe is the raw bits a detector XORs, addressed as (round, slot) in
the QPU's packet schedule, plus its noiseless reference parity: Stim's
rule (stim.Circuit.compile_m2d_converter). The table is read off the
circuit once (detector_error_model/detector_formation.py
build_formation_table) and read by every seat that forms a round's
detection events.
"""

import dataclasses
import enum
import functools
from collections.abc import Iterable
from typing import Optional


class LayerKind(enum.Enum):
    """What a detector compares, told by how many records it reads."""

    # Compared against the prepared state: one record.
    PREPARATION = "prep"
    # This round against the previous round: two records.
    BULK = "bulk"
    # Rebuilt from the data-qubit readout: three or more records.
    READOUT = "readout"


@dataclasses.dataclass(frozen=True)
class DetectorRecipe:
    """What one detector XORs, starting from its reference parity."""

    detector_index: int
    round_index: int
    kind: LayerKind
    records: tuple[tuple[int, int], ...]
    reference_parity: int
    coordinates: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class ObservableRecipe:
    """What one logical observable XORs, starting from its reference parity."""

    observable_index: int
    records: tuple[tuple[int, int], ...]
    reference_parity: int


@dataclasses.dataclass(frozen=True)
class FormationTable:
    """Every recipe read off one circuit, over its packet layout.

    readout_slot_start is the slot where the folded data readout begins in
    the last packet, or None. live_reach, on a live stream's table, is how
    many rounds back a round not yet run may read (rounds_read_back); None
    on a whole operation's table.
    """

    round_count: int
    packet_width_by_round: dict[int, int]
    readout_slot_start: Optional[int]
    detectors: tuple[DetectorRecipe, ...]
    observables: tuple[ObservableRecipe, ...]
    live_reach: Optional[int] = None

    def detectors_of_round(self, round_index: int) -> list[DetectorRecipe]:
        """The detectors formed when this round's packet arrives."""
        detectors = self._detectors_by_round.get(round_index, ())
        return list(detectors)

    def rounds_read_by(self, round_indices: Iterable[int]) -> set[int]:
        """The rounds whose packets forming these rounds reads, and no more.

        A seat joining mid-stream is given what its rounds read and nothing
        between.
        """
        record_rounds = set()
        for round_index in round_indices:
            detectors = self.detectors_of_round(round_index)
            rounds_of_round = _record_rounds_of(detectors)
            record_rounds.update(rounds_of_round)
        return record_rounds

    def rounds_read_before_first(
        self, first_round: int, round_indices: Iterable[int]
    ) -> int:
        """How many rounds before first_round forming these rounds reads.

        A later round can reach further back than the first (rec[-1] and rec[-4]
        after rounds of rec[-1] alone), so the earliest read of any counts.
        """
        record_rounds = self.rounds_read_by(round_indices)
        record_rounds.add(first_round)
        earliest_round = min(record_rounds)
        return first_round - earliest_round

    def earlier_rounds_read(self, first_round: int) -> tuple[int, ...]:
        """The rounds before first_round that it or any later round reads.

        What a stream's window keeps for the windows after it: on a live table
        the live_reach rounds before it, since a round not yet run may read that
        far.
        """
        reach = self._reach_from(first_round)
        earliest_round = first_round - reach
        earliest_round = max(1, earliest_round)
        read_rounds = range(earliest_round, first_round)
        return tuple(read_rounds)

    def rounds_reading(self, round_index: int) -> tuple[int, ...]:
        """The later rounds whose formation reads this round's packet.

        A round's own formation always has its own packet, so it is not listed.
        """
        return self._later_readers_by_round.get(round_index, ())

    def detector_rounds(self) -> dict[int, int]:
        """Each detector's round, the map resolve_detector_rounds yields."""
        return {
            recipe.detector_index: recipe.round_index
            for recipe in self.detectors
        }

    def observable_slots_of_round(
        self, round_index: int
    ) -> tuple[tuple[int, int], ...]:
        """(observable index, slot) for each observable record in the packet."""
        return self._observable_slots_by_round.get(round_index, ())

    def _reach_from(self, first_round: int) -> int:
        """How far before first_round its or a later detector reads."""
        if self.live_reach is not None:
            return self.live_reach
        earliest_round = self._earliest_detector_read_from.get(
            first_round, first_round
        )
        return first_round - earliest_round

    @functools.cached_property
    def _detectors_by_round(self) -> dict[int, list[DetectorRecipe]]:
        """The detectors grouped by round, once per table.

        A seat asks per round to form it and to size its store's room; a scan
        of a d=15, 100 round table costs 0.4 ms a call.
        """
        by_round: dict[int, list[DetectorRecipe]] = {}
        for recipe in self.detectors:
            group = by_round.setdefault(recipe.round_index, [])
            group.append(recipe)
        return by_round

    @functools.cached_property
    def _earliest_detector_read_from(self) -> dict[int, int]:
        """For each round, the earliest record it or a later detector reads.

        One pass from the last round down, so each window asks in a lookup.
        """
        earliest_by_round = {}
        earliest_round = self.round_count
        for round_index in range(self.round_count, 0, -1):
            detectors = self._detectors_by_round.get(round_index, ())
            record_rounds = _record_rounds_of(detectors)
            record_rounds.append(earliest_round)
            record_rounds.append(round_index)
            earliest_round = min(record_rounds)
            earliest_by_round[round_index] = earliest_round
        return earliest_by_round

    @functools.cached_property
    def _later_readers_by_round(self) -> dict[int, tuple[int, ...]]:
        """Each round's later readers, read off every recipe once."""
        readers: dict[int, set] = {}
        for recipe in self.detectors:
            _note_later_reader(readers, recipe.records, recipe.round_index)
        by_round = {}
        for round_index, reader_rounds in readers.items():
            by_round[round_index] = tuple(sorted(reader_rounds))
        return by_round

    @functools.cached_property
    def _observable_slots_by_round(self) -> dict[int, tuple]:
        """Each round's observable records, read off every recipe once."""
        by_round: dict[int, list] = {}
        for recipe in self.observables:
            for record_round, slot in recipe.records:
                slots = by_round.setdefault(record_round, [])
                slots.append((recipe.observable_index, slot))
        return {
            round_index: tuple(slots) for round_index, slots in by_round.items()
        }


def _note_later_reader(readers: dict, records, reader_round: int) -> None:
    """Add reader_round to the readers of every earlier round it reads."""
    for record_round, _ in records:
        if record_round >= reader_round:
            continue
        reader_rounds = readers.setdefault(record_round, set())
        reader_rounds.add(reader_round)


def _record_rounds_of(recipes) -> list[int]:
    """Every round the recipes' records name."""
    return [
        record_round for recipe in recipes for record_round, _ in recipe.records
    ]
