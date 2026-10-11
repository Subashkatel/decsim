"""A stream's formation table: a short circuit's recipes, moved in time.

A live stream's table read off the circuit run so far grows with the
rounds and is read again every round (qpu/streaming_stim_device.py,
_History.append), so a long stream costs its rounds squared. A repeated
round's detectors are the same from one round to the next but for the
time: each reads the same slots of the same rounds back, against the
same reference parity, one round later. Stim's generated memory says so
in its own text, a REPEAT block whose body ends in SHIFT_COORDS
(stim.Circuit.generated), and IBM's sliding-window decoder decodes every
window with one fixed check matrix (2510.21600, Algorithm 2), the same
translation applied to the model.

So the table here is read off a short circuit once, the program's first
round, then repeated rounds, then (for a finished stream) its final
round, and a round of a long stream is a round of that template moved:
- the first prefix_rounds rounds, the ones a later round can read the
  first round from, are the template's own;
- a later repeated round is the template's first round past them,
  moved by the rounds between;
- the final round is the template's final round, moved likewise.
The template's later rounds are checked to be the first one moved, and a
program whose repeated round is not is refused. A table of the template's
length or shorter is the circuit's own (detector_formation), and a test
holds every table to the one read off the whole circuit.
"""

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Optional, Union

import stim

import decsim.detector_error_model.detector_formation as detector_formation
import decsim.records.circuits as circuit_records
import decsim.records.formation as formation_records

# template rounds past the prefix: the first moved round, one to check
# it against, and a final round past both
EXTRA_TEMPLATE_ROUNDS = 3


class StreamTemplate:
    """One program's short tables, live and finished, and how they move."""

    def __init__(self, program: circuit_records.RepeatedStimCircuit) -> None:
        round_width = program.repeated_round.num_measurements
        repeated_reach = detector_formation.rounds_read_back(
            program.repeated_round, round_width
        )
        final_reach = detector_formation.rounds_read_back(
            program.final_round, round_width
        )
        reach = max(repeated_reach, final_reach, 1)
        # the rounds a later round can read the first round from
        self.prefix_rounds = reach + 1
        template_rounds = self.prefix_rounds + EXTRA_TEMPLATE_ROUNDS
        self.program = program
        self.live = assembled_table(program, template_rounds, is_final=False)
        self.final = assembled_table(program, template_rounds, is_final=True)
        first_moved = self.prefix_rounds + 1
        self.step = _coordinate_step(self.live, first_moved)
        _check_periodic(self)

    @property
    def round_count(self) -> int:
        """The template's rounds."""
        return self.live.round_count

    def table(
        self, round_count: int, *, is_final: bool, live_reach: Optional[int]
    ) -> Union[formation_records.FormationTable, "PeriodicFormationTable"]:
        """The table of a stream that has run round_count rounds."""
        if round_count <= self.round_count:
            return assembled_table(
                self.program, round_count, is_final, live_reach
            )
        return PeriodicFormationTable(self, round_count, is_final, live_reach)

    def detectors_before(self, round_index: int) -> int:
        """The template's detectors formed before this round."""
        return _detectors_before(self.live, round_index)


class PeriodicFormationTable(formation_records.TableReads):
    """A long stream's table, its rounds moved from the template's.

    It answers what a FormationTable answers. Its whole detectors and
    packet widths are made when read, a pass over the rounds; the seats
    read them by round.
    """

    def __init__(
        self,
        template: StreamTemplate,
        round_count: int,
        is_final: bool,
        live_reach: Optional[int],
    ) -> None:
        self.template = template
        self.round_count = round_count
        self.is_final = is_final
        self.live_reach = live_reach

    @property
    def packet_width_by_round(self) -> Mapping:
        """Each round's packet width, by round."""
        return _PacketWidths(self)

    @property
    def readout_slot_start(self) -> Optional[int]:
        """Where the folded readout begins in the final packet, or None."""
        source = self._source_table()
        return source.readout_slot_start

    @property
    def detectors(self) -> tuple:
        """Every detector's recipe, in index order: a pass over the rounds."""
        every_recipe = []
        after_last_round = self.round_count + 1
        for round_index in range(1, after_last_round):
            round_recipes = self.detectors_of_round(round_index)
            every_recipe.extend(round_recipes)
        return tuple(every_recipe)

    @property
    def observables(self) -> tuple:
        """The observables, read in the final rounds, moved to this length."""
        source = self._source_table()
        shift = self.round_count - source.round_count
        return tuple(
            _moved_observable(self.template, recipe, shift)
            for recipe in source.observables
        )

    def detectors_of_round(self, round_index: int) -> list:
        """The detectors formed when this round's packet arrives."""
        if not 1 <= round_index <= self.round_count:
            return []
        source_round, shift = self._source_of(round_index)
        source = self._source_table()
        index_shift = self._first_index(round_index) - _detectors_before(
            source, source_round
        )
        recipes = source.detectors_of_round(source_round)
        step = self.template.step
        return [
            _moved_detector(recipe, index_shift, shift, step)
            for recipe in recipes
        ]

    def rounds_reading(self, round_index: int) -> tuple:
        """The later rounds whose formation reads this round's packet."""
        last_reader = round_index + self.template.round_count
        after_last_reader = min(last_reader, self.round_count) + 1
        readers = []
        first_reader = round_index + 1
        for reader_round in range(first_reader, after_last_reader):
            recipes = self.detectors_of_round(reader_round)
            if _reads_round(recipes, round_index):
                readers.append(reader_round)
        return tuple(readers)

    def observable_slots_of_round(self, round_index: int) -> tuple:
        """(observable index, slot) for each observable record in the packet."""
        slots = []
        for recipe in self.observables:
            recipe_slots = _slots_in_round(recipe, round_index)
            slots.extend(recipe_slots)
        return tuple(slots)

    def stream_template(self) -> StreamTemplate:
        """The template this table is moved from."""
        return self.template

    def check_extends(self, previous: formation_records.TableReads) -> None:
        """Refuse a longer table that changes a round the previous had.

        A table moved from the same live template keeps every round, so
        it is checked in a step; another is checked round by round.
        """
        if self._continues(previous):
            return
        after_last_round = previous.round_count + 1
        for round_index in range(1, after_last_round):
            _check_same_round(previous, self, round_index)

    def _reach_from(self, first_round: int) -> int:
        """How far before first_round its or a later detector reads."""
        if self.live_reach is not None:
            return self.live_reach
        last_moved = first_round + self.template.round_count
        rounds = list(range(first_round, min(last_moved, self.round_count)))
        rounds.append(self.round_count)
        earliest_round = first_round
        for round_index in rounds:
            recipes = self.detectors_of_round(round_index)
            record_rounds = _record_rounds(recipes)
            earliest_round = min([earliest_round, *record_rounds])
        return first_round - earliest_round

    def _continues(self, previous) -> bool:
        if previous.stream_template() is not self.template:
            return False
        if previous.is_final:
            return False
        return previous.round_count <= self.round_count

    def _source_table(self) -> formation_records.FormationTable:
        if self.is_final:
            return self.template.final
        return self.template.live

    def _source_of(self, round_index: int) -> tuple:
        """The template round this round is moved from, and by how much."""
        template = self.template
        if round_index <= template.prefix_rounds:
            return round_index, 0
        if self.is_final and round_index == self.round_count:
            return template.round_count, round_index - template.round_count
        first_moved = template.prefix_rounds + 1
        return first_moved, round_index - first_moved

    def _first_index(self, round_index: int) -> int:
        """The index of the first detector this round forms."""
        template = self.template
        if round_index <= template.prefix_rounds + 1:
            return template.detectors_before(round_index)
        first_moved = template.prefix_rounds + 1
        first_recipes = template.live.detectors_of_round(first_moved)
        per_round = len(first_recipes)
        moved_rounds = round_index - first_moved
        return template.detectors_before(first_moved) + per_round * moved_rounds


def assembled_table(
    program: circuit_records.RepeatedStimCircuit,
    round_count: int,
    is_final: bool,
    live_reach: Optional[int] = None,
) -> formation_records.FormationTable:
    """The table read off the program's first round_count rounds."""
    circuit = stim.Circuit()
    measurement_rounds = {}
    after_last_round = round_count + 1
    for round_index in range(1, after_last_round):
        is_last_final = is_final and round_index == round_count
        fragment = program.round_circuit(round_index, is_last_final)
        first_measurement = circuit.num_measurements
        circuit += fragment
        for measurement in range(first_measurement, circuit.num_measurements):
            measurement_rounds[measurement] = round_index
    return detector_formation.build_formation_table(
        circuit,
        round_count,
        measurement_rounds=measurement_rounds,
        live_reach=live_reach,
    )


def moved_coordinates(coordinates: tuple, shift: int, step: tuple) -> tuple:
    """A detector's coordinates moved shift rounds; unmoved with no step."""
    if not step:
        return coordinates
    moved = []
    for coordinate, delta in zip(coordinates, step, strict=True):
        moved_by = shift * delta
        moved_coordinate = coordinate + moved_by
        moved.append(moved_coordinate)
    return tuple(moved)


class _PacketWidths(Mapping):
    """A periodic table's packet widths, read by round."""

    def __init__(self, table: PeriodicFormationTable) -> None:
        self.table = table

    def __getitem__(self, round_index: int) -> int:
        if not 1 <= round_index <= self.table.round_count:
            raise KeyError(round_index)
        source_round, _shift = self.table._source_of(round_index)
        source = self.table._source_table()
        return source.packet_width_by_round[source_round]

    def __iter__(self) -> Iterable:
        after_last_round = self.table.round_count + 1
        return iter(range(1, after_last_round))

    def __len__(self) -> int:
        return self.table.round_count


def _detectors_before(table, round_index: int) -> int:
    count = 0
    for earlier_round in range(1, round_index):
        recipes = table.detectors_of_round(earlier_round)
        count += len(recipes)
    return count


def _coordinate_step(table, first_moved: int) -> tuple:
    """How a detector's coordinates move from one repeated round to the next.

    Stim's generated body ends in SHIFT_COORDS(0, 0, 1), so time moves by
    one; the step is read off the template rather than assumed.
    """
    first = table.detectors_of_round(first_moved)
    next_round = first_moved + 1
    second = table.detectors_of_round(next_round)
    if not first or len(first) != len(second):
        return ()
    return tuple(
        later - earlier
        for earlier, later in zip(
            first[0].coordinates, second[0].coordinates, strict=True
        )
    )


def _check_periodic(template: StreamTemplate) -> None:
    """Every later template round is the first moved one, moved."""
    first_moved = template.prefix_rounds + 1
    after_last_round = template.round_count + 1
    live_rounds = range(first_moved, after_last_round)
    final_rounds = range(first_moved, template.round_count)
    for table, rounds in (
        (template.live, live_rounds),
        (template.final, final_rounds),
    ):
        _check_rounds_moved(template, table, first_moved, rounds)


def _check_rounds_moved(template, table, first_moved, rounds) -> None:
    first_recipes = template.live.detectors_of_round(first_moved)
    first_index = template.detectors_before(first_moved)
    for round_index in rounds:
        shift = round_index - first_moved
        index_shift = _detectors_before(table, round_index) - first_index
        moved = [
            _moved_detector(recipe, index_shift, shift, template.step)
            for recipe in first_recipes
        ]
        if moved == table.detectors_of_round(round_index):
            continue
        raise ValueError(
            "not implemented yet: a stream moves its repeated round in "
            f"time, and round {round_index} of this program is not round "
            f"{first_moved} moved by {shift} rounds"
        )


def _moved_detector(recipe, index_shift: int, shift: int, step: tuple):
    records = tuple(
        (record_round + shift, slot) for record_round, slot in recipe.records
    )
    coordinates = moved_coordinates(recipe.coordinates, shift, step)
    detector_index = recipe.detector_index + index_shift
    round_index = recipe.round_index + shift
    return dataclasses.replace(
        recipe,
        detector_index=detector_index,
        round_index=round_index,
        records=records,
        coordinates=coordinates,
    )


def _moved_observable(template: StreamTemplate, recipe, shift: int):
    """Records past the prefix move with the final round; the prefix's stay."""
    records = tuple(
        _moved_record(template, record_round, slot, shift)
        for record_round, slot in recipe.records
    )
    return dataclasses.replace(recipe, records=records)


def _moved_record(template, record_round: int, slot: int, shift: int):
    if record_round <= template.prefix_rounds:
        return record_round, slot
    return record_round + shift, slot


def _record_rounds(recipes: list) -> list:
    return [
        record_round for recipe in recipes for record_round, _ in recipe.records
    ]


def _reads_round(recipes: list, round_index: int) -> bool:
    record_rounds = _record_rounds(recipes)
    return round_index in record_rounds


def _slots_in_round(recipe, round_index: int) -> list:
    """(observable index, slot) of the recipe's records in the round."""
    return [
        (recipe.observable_index, slot)
        for record_round, slot in recipe.records
        if record_round == round_index
    ]


def _check_same_round(previous, table, round_index: int) -> None:
    if (
        previous.packet_width_by_round[round_index]
        != (table.packet_width_by_round[round_index])
    ):
        raise RuntimeError("formation extension changes an existing packet")
    if previous.detectors_of_round(round_index) != table.detectors_of_round(
        round_index
    ):
        raise RuntimeError("formation extension changes an existing detector")
