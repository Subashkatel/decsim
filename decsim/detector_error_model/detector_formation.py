"""Turns raw measurement bits into detection events, round by round.

A detector's recipe is the raw measurement bits, addressed as (round,
slot) in the QPU's one-based packet schedule, that XOR into it plus their
noiseless reference parity; that is Stim's own rule
(stim.Circuit.compile_m2d_converter, measurements_to_detection_events).
A front end may declare the packet schedule (measurement_rounds);
without one, the Stim generator layout applies, with the trailing data
readout folded into the last round's packet.

The folding is why a round's detector count is not constant. On a
rotated surface-code memory Stim lays out three kinds of layer: the
preparation layer compares each check against the prepared state and
holds (d*d - 1)/2 detectors, every bulk layer compares a round against
the one before it and holds d*d - 1, and the readout layer rebuilds the
checks from the data-qubit readout and holds another (d*d - 1)/2. The
readout layer folds into the last round, so the rounds this module
forms carry (d*d - 1)/2 events on the first, d*d - 1 in the middle and
3(d*d - 1)/2 on the last: 4, 8 and 12 at d=3, 12, 24 and 36 at d=5,
24, 48 and 72 at d=7, read off stim.Circuit.generated. The raw packet
widths differ again, d*d - 1 per round and d*d - 1 + d*d on the last,
which is where the 240-against-249 and 1200-against-1225 bit counts of
the store hop come from.
"""

import dataclasses
import enum
import functools
from collections.abc import Iterable, Sequence
from typing import Optional

import stim

from decsim.detector_error_model import detector_chronology


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

    `packet_width_by_round` is the raw bit count of each round's packet.
    `readout_slot_start` is the slot where the folded data readout begins
    in the last packet, or None when nothing was folded.
    `max_record_span` is how many rounds back any recipe reaches.
    """

    round_count: int
    packet_width_by_round: dict[int, int]
    readout_slot_start: Optional[int]
    detectors: tuple[DetectorRecipe, ...]
    observables: tuple[ObservableRecipe, ...]
    max_record_span: int

    def detectors_of_round(self, round_index: int) -> list[DetectorRecipe]:
        """The detectors formed when this round's packet arrives."""
        detectors = self._detectors_by_round.get(round_index, ())
        return list(detectors)

    def rounds_read_before(self, round_index: int) -> int:
        """How many rounds before this one its formation reads.

        Forming a round reads the records of its detectors and, on the
        last round, of every observable (StreamingDetectorFormer
        feed_packet), so a former that starts at this round needs the
        raw rounds from the earliest record on. Stim's converter reads
        every record a detector names, however far back
        (stim.Circuit.compile_m2d_converter). Records start at round
        one, so the count never reaches before it.
        """
        recipes_of_round = self.detectors_of_round(round_index)
        if round_index == self.round_count:
            recipes_of_round.extend(self.observables)
        earliest_round = round_index
        for recipe in recipes_of_round:
            for record_round, _ in recipe.records:
                earliest_round = min(earliest_round, record_round)
        return round_index - earliest_round

    def rounds_read_before_first(
        self, first_round: int, round_indices: Iterable[int]
    ) -> int:
        """How many rounds before first_round forming these rounds reads.

        A read forms each of its rounds in turn, and a later round can
        reach further back than the first (a detector of rec[-1] and
        rec[-4] after rounds of rec[-1] alone), so the read reaches back
        to the earliest round any of them reads (rounds_read_before).
        """
        earliest_round = first_round
        for round_index in round_indices:
            reach_count = self.rounds_read_before(round_index)
            earliest_read = round_index - reach_count
            earliest_round = min(earliest_round, earliest_read)
        return first_round - earliest_round

    def rounds_reading(self, round_index: int) -> tuple[int, ...]:
        """The later rounds whose formation reads this round's packet.

        The reference set of the packet: forming any of these rounds
        reads it (rounds_read_before names the same records from the
        reading side), and the observables are read with the last round.
        A round's own formation always comes with its own packet, so it
        is not listed.
        """
        return self._later_readers_by_round.get(round_index, ())

    def detector_rounds(self) -> dict[int, int]:
        """Each detector's round, the map resolve_detector_rounds yields."""
        return {
            recipe.detector_index: recipe.round_index
            for recipe in self.detectors
        }

    @functools.cached_property
    def _detectors_by_round(self) -> dict[int, list[DetectorRecipe]]:
        """The detectors grouped by round, once per table.

        A seat asks for one round's detectors to form it and again to
        size its store's room; scanning every detector of a d=15, 100
        round table costs 0.4 ms a call, a lookup costs nothing.
        """
        by_round: dict[int, list[DetectorRecipe]] = {}
        for recipe in self.detectors:
            group = by_round.setdefault(recipe.round_index, [])
            group.append(recipe)
        return by_round

    @functools.cached_property
    def _later_readers_by_round(self) -> dict[int, tuple[int, ...]]:
        """Each round's later readers, read off every recipe once."""
        readers: dict[int, set] = {}
        for recipe in self.detectors:
            _note_later_reader(readers, recipe.records, recipe.round_index)
        for recipe in self.observables:
            _note_later_reader(readers, recipe.records, self.round_count)
        by_round = {}
        for round_index, reader_rounds in readers.items():
            by_round[round_index] = tuple(sorted(reader_rounds))
        return by_round


def _note_later_reader(readers: dict, records, reader_round: int) -> None:
    """Add reader_round to the readers of every earlier round it reads."""
    for record_round, _ in records:
        if record_round >= reader_round:
            continue
        reader_rounds = readers.setdefault(record_round, set())
        reader_rounds.add(reader_round)


def build_formation_table(
    circuit: stim.Circuit,
    round_count: int,
    *,
    measurement_rounds: Optional[dict[int, int]] = None,
    detector_rounds: Optional[dict[int, int]] = None,
) -> FormationTable:
    """Read the recipe of every detector and observable off the circuit.

    `measurement_rounds` is the QPU's packet schedule as a front end
    declares it: one round per absolute measurement index.
    `detector_rounds` is each detector's declared round, inside
    1..round_count as detector_chronology requires; a detector can only
    be formed once every bit it reads has arrived, so a declared round
    may not precede them either.
    """
    if round_count < 1:
        raise ValueError("round_count must be positive")
    packet_of_measurement, packet_width_by_round, readout_slot_start = (
        _measurement_packets(circuit, round_count, measurement_rounds)
    )
    tables = _circuit_tables(
        circuit, packet_of_measurement, round_count, detector_rounds
    )
    detectors, observables = _read_recipes(circuit, tables)
    max_span = _max_record_span(detectors, observables, round_count)
    return FormationTable(
        round_count=round_count,
        packet_width_by_round=packet_width_by_round,
        readout_slot_start=readout_slot_start,
        detectors=tuple(detectors),
        observables=tuple(observables),
        max_record_span=max_span,
    )


class StreamingDetectorFormer:
    """One seat's former for one operation: a raw packet in, events out.

    It keeps a packet while a round that reads it (FormationTable
    rounds_reading) has not been formed here, as an HEVC decoder keeps
    each picture the current reference set names (FFmpeg
    hevc/refs.c:486-517), so a round formed out of order finds the
    packets it reads. It also keeps the max_record_span + 1 packets
    ending at the table's last round: a live table's next round has no
    recipes yet and may read them. Fed in round order this holds no more
    than a ring of the last max_record_span + 1 packets, the state IBM's
    windowed form keeps as its running syndrome (Maurer 2510.21600
    Algorithm 2, lines 760-770), and on a surface code the same packets.
    Every detector of the arriving round starts at its reference parity
    and XORs in its listed bits. The observables come out with the last
    round.
    """

    def __init__(self, table: FormationTable):
        self.table = table
        self.packets: dict[int, tuple[int, ...]] = {}
        # the rounds formed here, whose reads of a packet are done
        self.formed_rounds: set[int] = set()

    def feed_packet(
        self, round_index: int, bits: Iterable[int]
    ) -> tuple[list[tuple[int, int]], Optional[list[tuple[int, int]]]]:
        """Store one round's packet and form the detectors it completes.

        Returns (events, observables): events as (detector index, bit)
        pairs, observables as (observable index, bit) pairs on the last
        round and None before it.
        """
        self.hold_packet(round_index, bits)
        self.formed_rounds.add(round_index)
        recipes = self.table.detectors_of_round(round_index)
        events = [
            (recipe.detector_index, self._form_parity(recipe))
            for recipe in recipes
        ]
        observables = None
        if round_index == self.table.round_count:
            observables = [
                (recipe.observable_index, self._form_parity(recipe))
                for recipe in self.table.observables
            ]
        return events, observables

    def hold_packet(self, round_index: int, bits: Iterable[int]) -> None:
        """Keep one round's packet, for the rounds after it, forming nothing.

        Every other packet no unformed round reads is let go now.
        """
        packet = tuple(int(bit) for bit in bits)
        expected_bit_count = self.table.packet_width_by_round[round_index]
        if len(packet) != expected_bit_count:
            raise ValueError(
                f"round {round_index}: packet has {len(packet)} bits, "
                f"the formation table expects {expected_bit_count}"
            )
        self.packets[round_index] = packet
        self._let_go_of_unread_packets(round_index)

    def holds_packet(self, round_index: int) -> bool:
        """Whether this round's raw packet is held here."""
        return round_index in self.packets

    def held_bits(self) -> int:
        """The raw bits the former holds."""
        return sum(len(packet) for packet in self.packets.values())

    def extend_table(self, table: FormationTable) -> None:
        """Append recipes without changing earlier rounds or losing history.

        Live circuits and incrementally described recorded streams supply
        the next recipes before their packets. Stim's record lookbacks
        still require every referenced earlier measurement to be retained.
        """
        _check_formation_prefix(self.table, table)
        previous_detector_count = len(self.table.detectors)
        new_detectors = table.detectors[previous_detector_count:]
        new_recipes = new_detectors + table.observables
        _check_retained_recipe_records(new_recipes, self.packets)
        self.table = table

    def _let_go_of_unread_packets(self, held_round: int) -> None:
        unread_rounds = []
        for kept_round in self.packets:
            if kept_round == held_round:
                continue
            if self._is_still_read(kept_round):
                continue
            unread_rounds.append(kept_round)
        for unread_round in unread_rounds:
            del self.packets[unread_round]

    def _is_still_read(self, kept_round: int) -> bool:
        """Whether a round not yet formed here, or not yet known, reads it."""
        tail_start = self.table.round_count - self.table.max_record_span
        if kept_round >= tail_start:
            return True
        reader_rounds = self.table.rounds_reading(kept_round)
        for reader_round in reader_rounds:
            if reader_round not in self.formed_rounds:
                return True
        return False

    def _form_parity(self, recipe) -> int:
        value = recipe.reference_parity
        for record_round, slot in recipe.records:
            packet = self.packets.get(record_round)
            if packet is None:
                raise RuntimeError(
                    f"round {recipe.round_index} reads round {record_round}, "
                    "which this former does not hold: a detector compares "
                    "a round against the one before it (LILLIPUT 2108.06569 "
                    "lines 499-510), so its seat must be given that round"
                )
            value ^= packet[slot]
        return value


def split_measurements_into_packets(
    table: FormationTable, measurement_row: Sequence[int]
) -> dict[int, tuple[int, ...]]:
    """Cut one shot's measurement row into per-round packets (the QPU side)."""
    packets = {}
    cursor = 0
    after_last_round = table.round_count + 1
    for round_index in range(1, after_last_round):
        width = table.packet_width_by_round[round_index]
        packet_end = cursor + width
        packet_bits = measurement_row[cursor:packet_end]
        packets[round_index] = tuple(int(bit) for bit in packet_bits)
        cursor = packet_end
    return packets


def form_shot(
    table: FormationTable, packets_by_round: dict[int, Sequence[int]]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Every detector and observable bit of one shot, in index order."""
    streaming_former = StreamingDetectorFormer(table)
    detector_bits = [0] * len(table.detectors)
    observable_bits = [0] * len(table.observables)
    after_last_round = table.round_count + 1
    for round_index in range(1, after_last_round):
        events, observables = streaming_former.feed_packet(
            round_index, packets_by_round[round_index]
        )
        _store_bits(detector_bits, events)
        if observables is not None:
            _store_bits(observable_bits, observables)
    return tuple(detector_bits), tuple(observable_bits)


@dataclasses.dataclass(frozen=True)
class _CircuitTables:
    """What every recipe is read against."""

    packet_of_measurement: dict
    reference_sample: object
    coordinates: dict
    detector_rounds: Optional[dict]


def _check_formation_prefix(
    previous: FormationTable, table: FormationTable
) -> None:
    for round_index, width in previous.packet_width_by_round.items():
        if table.packet_width_by_round.get(round_index) != width:
            raise RuntimeError("formation extension changes an existing packet")
    previous_detector_count = len(previous.detectors)
    detector_prefix = table.detectors[:previous_detector_count]
    if detector_prefix != previous.detectors:
        raise RuntimeError("formation extension changes an existing detector")


def _check_retained_recipe_records(recipes, packets: dict) -> None:
    last_received_round = max(packets, default=0)
    for recipe in recipes:
        _check_recipe_history(recipe, packets, last_received_round)


def _check_recipe_history(
    recipe, packets: dict, last_received_round: int
) -> None:
    for record_round, _ in recipe.records:
        if record_round > last_received_round:
            continue
        if record_round not in packets:
            raise RuntimeError(
                f"formation extension references discarded round {record_round}"
            )


def _circuit_tables(
    circuit: stim.Circuit,
    packet_of_measurement: dict[int, tuple[int, int]],
    round_count: int,
    detector_rounds: Optional[dict[int, int]],
) -> _CircuitTables:
    """What the recipes are read against, with the declared rounds checked."""
    declared_rounds = None
    if detector_rounds is not None:
        declared_rounds = detector_chronology.checked_detector_round_map(
            detector_rounds, circuit.num_detectors, round_count
        )
    reference_sample = circuit.reference_sample()
    coordinates = circuit.get_detector_coordinates()
    return _CircuitTables(
        packet_of_measurement=packet_of_measurement,
        reference_sample=reference_sample,
        coordinates=coordinates,
        detector_rounds=declared_rounds,
    )


class _MeasurementRoundReader:
    """Gives each measurement its round from one walk over the circuit.

    The measurement blocks between two DETECTOR groups belong to the round
    the following group announces (its time coordinate plus one). Blocks
    after the last group, and groups past round_count (Stim's post-readout
    layer), fold into the last round's packet.
    """

    def __init__(self, round_count: int):
        self.round_count = round_count
        self.coordinate_shift = 0.0
        self.pending_count = 0
        self.rounds: list[int] = []
        self.readout_start: Optional[int] = None
        self.folded_group_count = 0

    def read(self, circuit: stim.Circuit) -> tuple[list[int], Optional[int]]:
        """The round of every measurement, and where the readout starts."""
        for instruction in _flat_instructions(circuit):
            self._note_instruction(instruction)
        self._fold_trailing_measurements()
        if _has_a_backward_step(self.rounds):
            raise ValueError(
                "measurement blocks are not in round order; declare "
                "measurement_rounds"
            )
        after_last_round = self.round_count + 1
        if set(self.rounds) != set(range(1, after_last_round)):
            raise ValueError(
                f"circuit announces rounds {sorted(set(self.rounds))}, "
                f"formation table was asked for {self.round_count}"
            )
        return self.rounds, self.readout_start

    def _note_instruction(self, instruction) -> None:
        measurement_count = _measurement_count(instruction)
        if measurement_count:
            self.pending_count += measurement_count
            return
        if instruction.name == "SHIFT_COORDS":
            self._note_shift(instruction)
            return
        if instruction.name == "DETECTOR" and self.pending_count:
            self._note_detector_group(instruction)

    def _note_shift(self, instruction) -> None:
        arguments = instruction.gate_args_copy()
        if arguments:
            self.coordinate_shift += arguments[-1]

    def _note_detector_group(self, instruction) -> None:
        arguments = instruction.gate_args_copy()
        if arguments:
            shifted_layer = arguments[-1] + self.coordinate_shift
            layer = int(shifted_layer)
        else:
            layer = len(set(self.rounds))
        announced = layer + 1
        if announced > self.round_count:
            self._note_folded_group(announced)
        assigned_round = min(announced, self.round_count)
        assigned_rounds = [assigned_round] * self.pending_count
        self.rounds.extend(assigned_rounds)
        self.pending_count = 0

    def _note_folded_group(self, announced: int) -> None:
        # Only Stim's single post-readout layer may fold; anything more
        # means the declared round count does not fit the circuit.
        self.folded_group_count += 1
        last_foldable_round = self.round_count + 1
        is_too_far = announced > last_foldable_round
        if is_too_far or self.folded_group_count > 1:
            raise ValueError(
                f"circuit announces round {announced}, "
                f"formation table was asked for {self.round_count}"
            )
        if self.readout_start is None:
            self.readout_start = len(self.rounds)

    def _fold_trailing_measurements(self) -> None:
        if not self.pending_count:
            return
        ends_on_last_round = bool(self.rounds) and (
            self.rounds[-1] == self.round_count
        )
        if self.readout_start is None and ends_on_last_round:
            self.readout_start = len(self.rounds)
        trailing_rounds = [self.round_count] * self.pending_count
        self.rounds.extend(trailing_rounds)


def _measurement_count(instruction) -> int:
    """How many records an instruction appends; Stim's count decides."""
    return instruction.num_measurements


def _record_offsets(instruction) -> list[int]:
    """The rec[-k] lookbacks of a DETECTOR or OBSERVABLE_INCLUDE.

    Pauli targets (OBSERVABLE_INCLUDE(k) X5) carry no record and are
    skipped, as Stim skips them.
    """
    return [
        target.value
        for target in instruction.targets_copy()
        if target.is_measurement_record_target
    ]


def _flat_instructions(circuit):
    """Instructions in execution order with REPEAT blocks unrolled."""
    for instruction in circuit:
        if not isinstance(instruction, stim.CircuitRepeatBlock):
            yield instruction
            continue
        body = instruction.body_copy()
        for _ in range(instruction.repeat_count):
            yield from _flat_instructions(body)


def _has_a_backward_step(values: list[int]) -> bool:
    return any(
        later < earlier
        for earlier, later in zip(values, values[1:], strict=False)
    )


def _round_of_each_measurement(
    circuit: stim.Circuit,
    round_count: int,
    measurement_rounds: Optional[dict[int, int]],
) -> tuple[list[int], Optional[int]]:
    """One round per absolute measurement index, and the readout start.

    Declared by the front end, or read off the circuit's shape.
    """
    if measurement_rounds is None:
        reader = _MeasurementRoundReader(round_count)
        return reader.read(circuit)
    measurement_count = 0
    for instruction in _flat_instructions(circuit):
        measurement_count += _measurement_count(instruction)
    if set(measurement_rounds) != set(range(measurement_count)):
        raise ValueError(
            "measurement-round map must cover every measurement exactly"
        )
    rounds = [
        int(measurement_rounds[index]) for index in range(measurement_count)
    ]
    if any(not 1 <= round_index <= round_count for round_index in rounds):
        raise ValueError(
            "declared measurement rounds must lie in 1..round_count"
        )
    if _has_a_backward_step(rounds):
        raise ValueError("declared measurement rounds must be non-decreasing")
    return rounds, None


def _measurement_packets(
    circuit: stim.Circuit,
    round_count: int,
    measurement_rounds: Optional[dict[int, int]],
) -> tuple[dict[int, tuple[int, int]], dict[int, int], Optional[int]]:
    """Each measurement's (round, slot), each packet's width, readout start."""
    rounds, readout_start = _round_of_each_measurement(
        circuit, round_count, measurement_rounds
    )
    packet_of_measurement: dict[int, tuple[int, int]] = {}
    packet_width_by_round: dict[int, int] = {}
    readout_slot_start = None
    for absolute_index, round_index in enumerate(rounds):
        slot = packet_width_by_round.get(round_index, 0)
        if absolute_index == readout_start:
            readout_slot_start = slot
        packet_of_measurement[absolute_index] = (round_index, slot)
        packet_width_by_round[round_index] = slot + 1
    after_last_round = round_count + 1
    for round_index in range(1, after_last_round):
        packet_width_by_round.setdefault(round_index, 0)
    return packet_of_measurement, packet_width_by_round, readout_slot_start


def _read_recipes(circuit, tables: _CircuitTables) -> tuple[list, list]:
    """Every detector recipe and every observable recipe, in index order."""
    detectors: list[DetectorRecipe] = []
    observable_records: dict[int, list] = {}
    observable_parity: dict[int, int] = {}
    measurement_count_so_far = 0
    for instruction in _flat_instructions(circuit):
        if instruction.name == "DETECTOR":
            recipe = _detector_recipe(
                tables, instruction, len(detectors), measurement_count_so_far
            )
            detectors.append(recipe)
            continue
        if instruction.name == "OBSERVABLE_INCLUDE":
            _note_observable(
                tables,
                instruction,
                measurement_count_so_far,
                observable_records,
                observable_parity,
            )
            continue
        measurement_count_so_far += _measurement_count(instruction)
    # every ID below circuit.num_observables is an observable, one the
    # circuit never names always zero, as Stim's converter reports it
    # (compile_m2d_converter, separate_observables)
    observables = []
    for index in range(circuit.num_observables):
        records = observable_records.get(index, [])
        records = tuple(records)
        parity = observable_parity.get(index, 0)
        recipe = ObservableRecipe(index, records, parity)
        observables.append(recipe)
    return detectors, observables


def _absolute_indices(instruction, measurement_count_so_far: int) -> list[int]:
    """The absolute measurement indices an instruction's lookbacks name."""
    return [
        measurement_count_so_far + offset
        for offset in _record_offsets(instruction)
    ]


def _detector_recipe(
    tables: _CircuitTables,
    instruction,
    detector_index: int,
    measurement_count_so_far: int,
) -> DetectorRecipe:
    absolute_indices = _absolute_indices(instruction, measurement_count_so_far)
    records = tuple(
        tables.packet_of_measurement[index] for index in absolute_indices
    )
    arrival_round = max(record_round for record_round, _ in records)
    round_index = arrival_round
    if tables.detector_rounds is not None:
        round_index = int(tables.detector_rounds[detector_index])
        if round_index < arrival_round:
            raise ValueError(
                f"detector {detector_index} is declared in round "
                f"{round_index} but reads a bit that arrives in round "
                f"{arrival_round}"
            )
    reference_bits = tables.reference_sample[absolute_indices]
    reference_sum = reference_bits.sum()
    parity = reference_sum % 2
    reference_parity = int(parity)
    coordinates = tables.coordinates.get(detector_index, ())
    kind = _layer_kind_for(len(records))
    return DetectorRecipe(
        detector_index=detector_index,
        round_index=round_index,
        kind=kind,
        records=records,
        reference_parity=reference_parity,
        coordinates=tuple(coordinates),
    )


def _note_observable(
    tables: _CircuitTables,
    instruction,
    measurement_count_so_far: int,
    observable_records: dict,
    observable_parity: dict,
) -> None:
    """Add one OBSERVABLE_INCLUDE's bits to its observable's recipe."""
    arguments = instruction.gate_args_copy()
    observable_index = int(arguments[0])
    absolute_indices = _absolute_indices(instruction, measurement_count_so_far)
    records = observable_records.setdefault(observable_index, [])
    for index in absolute_indices:
        records.append(tables.packet_of_measurement[index])
    reference_bits = tables.reference_sample[absolute_indices]
    parity_so_far = observable_parity.get(observable_index, 0)
    reference_sum = reference_bits.sum()
    added_parity = int(reference_sum)
    total_parity = parity_so_far + added_parity
    observable_parity[observable_index] = total_parity % 2


def _layer_kind_for(record_count: int) -> LayerKind:
    if record_count == 1:
        return LayerKind.PREPARATION
    if record_count == 2:
        return LayerKind.BULK
    return LayerKind.READOUT


def _max_record_span(
    detectors: list, observables: list, round_count: int
) -> int:
    """How many rounds back any recipe reaches.

    A former keeps the rounds a live table's next round may read by it,
    observables included: a logical readout can span a whole block.
    """
    spans = []
    for recipe in detectors:
        earliest_round = min(record_round for record_round, _ in recipe.records)
        span = recipe.round_index - earliest_round
        spans.append(span)
    for recipe in observables:
        if not recipe.records:
            continue
        earliest_round = min(record_round for record_round, _ in recipe.records)
        span = round_count - earliest_round
        spans.append(span)
    return max(spans, default=0)


def _store_bits(bits: list[int], indexed_values) -> None:
    for index, value in indexed_values:
        bits[index] = value
