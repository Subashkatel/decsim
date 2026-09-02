"""Turns raw measurement bits into detection events, round by round.

A detector asks whether a stabilizer reading differs from what it should
be. Its recipe is the list of raw measurement bits that XOR into it plus
the noiseless reference parity of those bits. That is Stim's own rule
(stim.Circuit.compile_m2d_converter, measurements_to_detection_events),
and the reference term is what keeps a circuit whose expected readings
are not all zero correct.

Every measurement bit is addressed as (round, slot): round is the QPU
round whose packet carries it, slot is its position inside that packet.
Rounds are one-based. Which round a measurement belongs to is the QPU's
schedule, not a property of the circuit text, so a front end may declare
it (measurement_rounds). Without a declaration the Stim generator
convention applies: the measurement blocks before a DETECTOR group belong
to the round that group announces in its time coordinate, and the
trailing data readout folds into the last round's packet after that
round's own bits. Either way the readout detectors land on the last
round, exactly where detector_chronology.resolve_detector_rounds puts
them.

build_formation_table reads the recipes off a circuit once;
StreamingDetectorFormer applies them one packet at a time in the
controller; split_measurements_into_packets and form_shot are the QPU
side and the whole-shot check.
"""

import dataclasses
import enum
from collections.abc import Iterable, Sequence
from typing import Optional

import stim


class LayerKind(enum.Enum):
    """How many records a detector reads, and so what it compares."""

    # Compared against the prepared state: one record.
    PREPARATION = "prep"
    # This round against the previous round: two records.
    BULK = "bulk"
    # Rebuilt from the data-qubit readout: three or more records.
    READOUT = "readout"


@dataclasses.dataclass(frozen=True)
class DetectorRecipe:
    """The bits one detector XORs, and the parity they should give."""

    detector_index: int
    round_index: int
    kind: LayerKind
    records: tuple[tuple[int, int], ...]
    reference_parity: int
    coordinates: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class ObservableRecipe:
    """The bits one logical observable XORs, and their reference parity."""

    observable_index: int
    records: tuple[tuple[int, int], ...]
    reference_parity: int


@dataclasses.dataclass(frozen=True)
class FormationTable:
    """Every recipe of one circuit, and the packet layout they read.

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
        return [
            recipe
            for recipe in self.detectors
            if recipe.round_index == round_index
        ]

    def detector_rounds(self) -> dict[int, int]:
        """Each detector's round, the map resolve_detector_rounds yields."""
        return {
            recipe.detector_index: recipe.round_index
            for recipe in self.detectors
        }


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
    `detector_rounds` is each detector's declared round; a detector can
    only be formed once every bit it reads has arrived, so a declared
    round may not precede them.
    """
    if round_count < 1:
        raise ValueError("round_count must be positive")
    packet_of_measurement, packet_width_by_round, readout_slot_start = (
        _measurement_packets(circuit, round_count, measurement_rounds)
    )
    reference_sample = circuit.reference_sample()
    coordinates = circuit.get_detector_coordinates()
    tables = _CircuitTables(
        packet_of_measurement=packet_of_measurement,
        reference_sample=reference_sample,
        coordinates=coordinates,
        detector_rounds=detector_rounds,
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
    """The controller stage: one raw packet in per round, events out.

    A ring buffer keeps the last max_record_span + 1 packets (two for a
    memory experiment). Every detector of the arriving round starts at its
    reference parity and XORs in its listed bits. The observables come out
    with the last round.
    """

    def __init__(self, table: FormationTable):
        self.table = table
        self.kept_packet_count = table.max_record_span + 1
        self.packets: dict[int, tuple[int, ...]] = {}

    def feed_packet(
        self, round_index: int, bits: Iterable[int]
    ) -> tuple[list[tuple[int, int]], Optional[list[tuple[int, int]]]]:
        """Store one round's packet and form the detectors it completes.

        Returns (events, observables): events as (detector index, bit)
        pairs, observables as (observable index, bit) pairs on the last
        round and None before it.
        """
        packet = tuple(int(bit) for bit in bits)
        expected = self.table.packet_width_by_round[round_index]
        if len(packet) != expected:
            raise ValueError(
                f"round {round_index}: packet has {len(packet)} bits, "
                f"the formation table expects {expected}"
            )
        self.packets[round_index] = packet
        newest_stale_round = round_index - self.kept_packet_count
        self._forget_through(newest_stale_round)
        events = [
            (recipe.detector_index, self._form_parity(recipe))
            for recipe in self.table.detectors_of_round(round_index)
        ]
        observables = None
        if round_index == self.table.round_count:
            observables = [
                (recipe.observable_index, self._form_parity(recipe))
                for recipe in self.table.observables
            ]
        return events, observables

    def _forget_through(self, newest_stale_round: int) -> None:
        stale_rounds = [
            kept_round
            for kept_round in self.packets
            if kept_round <= newest_stale_round
        ]
        for stale_round in stale_rounds:
            del self.packets[stale_round]

    def _form_parity(self, recipe) -> int:
        value = recipe.reference_parity
        for record_round, slot in recipe.records:
            value ^= self.packets[record_round][slot]
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
    former = StreamingDetectorFormer(table)
    detector_bits = [0] * len(table.detectors)
    observable_bits = [0] * len(table.observables)
    after_last_round = table.round_count + 1
    for round_index in range(1, after_last_round):
        events, observables = former.feed_packet(
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

    def read(self, circuit) -> tuple[list[int], Optional[int]]:
        """The round of every measurement, and where the readout starts."""
        for instruction in _flat_instructions(circuit):
            self._note_instruction(instruction)
        self._fold_trailing_measurements()
        if _decreases(self.rounds):
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
        is_too_far = announced > self.round_count + 1
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


def _decreases(values: list[int]) -> bool:
    return any(later < earlier for earlier, later in zip(values, values[1:]))


def _round_of_each_measurement(
    circuit, round_count: int, measurement_rounds
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
    rounds = [
        int(measurement_rounds[index]) for index in range(measurement_count)
    ]
    if any(not 1 <= round_index <= round_count for round_index in rounds):
        raise ValueError(
            "declared measurement rounds must lie in 1..round_count"
        )
    if _decreases(rounds):
        raise ValueError("declared measurement rounds must be non-decreasing")
    return rounds, None


def _measurement_packets(circuit, round_count: int, measurement_rounds):
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
    measurements_so_far = 0
    for instruction in _flat_instructions(circuit):
        if instruction.name == "DETECTOR":
            recipe = _detector_recipe(
                tables, instruction, len(detectors), measurements_so_far
            )
            detectors.append(recipe)
            continue
        if instruction.name == "OBSERVABLE_INCLUDE":
            _note_observable(
                tables,
                instruction,
                measurements_so_far,
                observable_records,
                observable_parity,
            )
            continue
        measurements_so_far += _measurement_count(instruction)
    observables = []
    for index in sorted(observable_records):
        records = tuple(observable_records[index])
        parity = observable_parity[index]
        recipe = ObservableRecipe(index, records, parity)
        observables.append(recipe)
    return detectors, observables


def _absolute_indices(instruction, measurements_so_far: int) -> list[int]:
    """The absolute measurement indices an instruction's lookbacks name."""
    return [
        measurements_so_far + offset for offset in _record_offsets(instruction)
    ]


def _detector_recipe(
    tables: _CircuitTables,
    instruction,
    detector_index: int,
    measurements_so_far: int,
) -> DetectorRecipe:
    absolute_indices = _absolute_indices(instruction, measurements_so_far)
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
    measurements_so_far: int,
    observable_records: dict,
    observable_parity: dict,
) -> None:
    """Add one OBSERVABLE_INCLUDE's bits to its observable's recipe."""
    arguments = instruction.gate_args_copy()
    observable_index = int(arguments[0])
    absolute_indices = _absolute_indices(instruction, measurements_so_far)
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

    The ring buffer must hold every round a recipe reads, observables
    included: a logical readout can span a whole block.
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
