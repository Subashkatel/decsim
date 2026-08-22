"""Detector formation: the table that turns raw measurement bits into
detection events, round by round, in the controller.

A detector asks "did this stabilizer reading differ from what it should be".
Its recipe is the list of raw measurement bits that XOR into it plus the
noiseless reference parity of those bits; that is Stim's own rule
(measurements_to_detection_events), and the reference term is what keeps
circuits whose expected readings are not all zero correct.

Packet coordinates: every measurement bit is addressed as (round, slot),
where round is the QPU round whose packet carries it and slot is the bit's
position inside that packet. Rounds are one-based. Which round a measurement
belongs to is the QPU's schedule, not a property of the circuit text, so a
frontend may declare it (measurement_rounds); without a declaration the
Stim-generator convention applies: one measurement instruction per round,
with one trailing data-readout instruction folded into the last round's
packet after that round's own bits. Either way the readout detectors land
on the last round exactly as resolve_detector_rounds places them.

Leaf module: it imports nothing from this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import stim


def _measurement_count(instruction) -> int:
    """Records an instruction appends. M, MR, MX.., MXX/MYY/MZZ, MPP, MPAD
    and the heralded channels all produce records; Stim's own count is the
    authority, as in measurements_to_detection_events."""
    return instruction.num_measurements


def _record_offsets(instruction) -> list[int]:
    """rec[-k] lookbacks of a DETECTOR or OBSERVABLE_INCLUDE. Pauli targets
    (OBSERVABLE_INCLUDE(k) X5 ...) carry no record and are skipped, as Stim
    skips them."""
    return [target.value for target in instruction.targets_copy()
            if target.is_measurement_record_target]


class LayerKind(Enum):
    PREP = "prep"          # compared against the prepared state: one record
    BULK = "bulk"          # this round against the previous round: two records
    READOUT = "readout"    # rebuilt from the data-qubit readout: three or more records


@dataclass(frozen=True)
class DetectorRecipe:
    detector_index: int
    round_index: int
    kind: LayerKind
    records: tuple[tuple[int, int], ...]   # ((round, slot), ...)
    reference_parity: int
    coordinates: tuple


@dataclass(frozen=True)
class ObservableRecipe:
    observable_index: int
    records: tuple[tuple[int, int], ...]
    reference_parity: int


@dataclass(frozen=True)
class FormationTable:
    round_count: int
    packet_width: dict[int, int]           # round -> raw bits in that round's packet
    readout_slot_start: int | None         # slot where folded readout bits begin in the last packet
    detectors: tuple[DetectorRecipe, ...]  # in detector index order
    observables: tuple[ObservableRecipe, ...]
    max_record_span: int                   # rounds a recipe reaches back

    def detectors_of_round(self, round_index: int) -> list[DetectorRecipe]:
        return [recipe for recipe in self.detectors if recipe.round_index == round_index]

    def detector_rounds(self) -> dict[int, int]:
        """detector index -> round, the same map resolve_detector_rounds yields."""
        return {recipe.detector_index: recipe.round_index for recipe in self.detectors}


def _flat_instructions(circuit):
    """Instructions in execution order with REPEAT blocks unrolled."""
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            body = instruction.body_copy()
            for _ in range(instruction.repeat_count):
                yield from _flat_instructions(body)
        else:
            yield instruction


def _round_of_each_measurement(circuit, round_count: int, measurement_rounds):
    """One round per absolute measurement index, plus where the folded readout
    starts: declared by the frontend, or read off the circuit's shape.

    Circuit rule: the measurement blocks between two DETECTOR groups belong
    to the round the following group announces (its time coordinate plus
    one). Blocks after the last group, and groups past round_count (Stim's
    post-readout layer), fold into the last round's packet.
    """
    measurement_count = sum(
        _measurement_count(instruction) for instruction in _flat_instructions(circuit))
    if measurement_rounds is not None:
        rounds = [int(measurement_rounds[index]) for index in range(measurement_count)]
        if any(not 1 <= r <= round_count for r in rounds):
            raise ValueError("declared measurement rounds must lie in 1..round_count")
        if any(later < earlier for earlier, later in zip(rounds, rounds[1:])):
            raise ValueError("declared measurement rounds must be non-decreasing")
        return rounds, None

    coordinate_shift = 0.0
    pending = 0                 # measurements waiting for their DETECTOR group
    rounds: list[int] = []
    readout_start = None
    folded_groups = 0           # groups announcing a round past round_count
    for instruction in _flat_instructions(circuit):
        if _measurement_count(instruction):
            pending += _measurement_count(instruction)
        elif instruction.name == "SHIFT_COORDS":
            arguments = instruction.gate_args_copy()
            coordinate_shift += arguments[-1] if arguments else 0.0
        elif instruction.name == "DETECTOR" and pending:
            arguments = instruction.gate_args_copy()
            layer = int(arguments[-1] + coordinate_shift) if arguments else len(set(rounds))
            announced = layer + 1
            if announced > round_count:
                folded_groups += 1
                # only Stim's single post-readout layer may fold; anything
                # more means the declared round count does not fit
                if announced > round_count + 1 or folded_groups > 1:
                    raise ValueError(
                        f"circuit announces round {announced}, "
                        f"formation table was asked for {round_count}")
                if readout_start is None:
                    readout_start = len(rounds)
            rounds.extend([min(announced, round_count)] * pending)
            pending = 0
    if pending:
        if readout_start is None and rounds and rounds[-1] == round_count:
            readout_start = len(rounds)
        rounds.extend([round_count] * pending)
    if any(later < earlier for earlier, later in zip(rounds, rounds[1:])):
        raise ValueError("measurement blocks are not in round order; declare measurement_rounds")
    if set(rounds) != set(range(1, round_count + 1)):
        raise ValueError(
            f"circuit announces rounds {sorted(set(rounds))}, "
            f"formation table was asked for {round_count}")
    return rounds, readout_start


def _measurement_packets(circuit, round_count: int, measurement_rounds):
    """Absolute measurement index -> (round, slot), each packet's width, and
    the slot where the folded readout begins in the last packet (or None)."""
    rounds, readout_start = _round_of_each_measurement(circuit, round_count, measurement_rounds)
    packet_of_measurement: dict[int, tuple[int, int]] = {}
    packet_width: dict[int, int] = {}
    readout_slot_start = None
    for absolute, round_index in enumerate(rounds):
        slot = packet_width.get(round_index, 0)
        if absolute == readout_start:
            readout_slot_start = slot
        packet_of_measurement[absolute] = (round_index, slot)
        packet_width[round_index] = slot + 1
    for round_index in range(1, round_count + 1):
        packet_width.setdefault(round_index, 0)
    return packet_of_measurement, packet_width, readout_slot_start


def _kind_of(record_count: int) -> LayerKind:
    if record_count == 1:
        return LayerKind.PREP
    if record_count == 2:
        return LayerKind.BULK
    return LayerKind.READOUT


def build_formation_table(circuit, round_count: int, *,
                          measurement_rounds=None,
                          detector_rounds=None) -> FormationTable:
    """Read the recipe of every detector and observable off the circuit.

    measurement_rounds: optional round per absolute measurement index, the
    QPU's packet schedule as a frontend declares it.
    detector_rounds: optional detector index -> round, the same map
    resolve_detector_rounds yields; a detector can only be formed once every
    bit it reads has arrived, so a declared round may not precede them.
    """
    if round_count < 1:
        raise ValueError("round_count must be positive")
    reference = circuit.reference_sample()
    packet_of_measurement, packet_width, readout_slot_start = _measurement_packets(
        circuit, round_count, measurement_rounds)
    coordinates = circuit.get_detector_coordinates()

    detectors: list[DetectorRecipe] = []
    observable_records: dict[int, list] = {}
    observable_parity: dict[int, int] = {}
    measurements_so_far = 0
    detector_index = 0
    for instruction in _flat_instructions(circuit):
        if instruction.name == "DETECTOR":
            absolute = [measurements_so_far + offset
                        for offset in _record_offsets(instruction)]
            records = tuple(packet_of_measurement[index] for index in absolute)
            arrival_round = max(record_round for record_round, _ in records)
            round_index = arrival_round
            if detector_rounds is not None:
                round_index = int(detector_rounds[detector_index])
                if round_index < arrival_round:
                    raise ValueError(
                        f"detector {detector_index} is declared in round {round_index} "
                        f"but reads a bit that arrives in round {arrival_round}")
            detectors.append(DetectorRecipe(
                detector_index=detector_index,
                round_index=round_index,
                kind=_kind_of(len(records)),
                records=records,
                reference_parity=int(reference[absolute].sum() % 2),
                coordinates=tuple(coordinates.get(detector_index, ())),
            ))
            detector_index += 1
            continue
        if instruction.name == "OBSERVABLE_INCLUDE":
            observable_index = int(instruction.gate_args_copy()[0])
            absolute = [measurements_so_far + offset
                        for offset in _record_offsets(instruction)]
            observable_records.setdefault(observable_index, []).extend(
                packet_of_measurement[index] for index in absolute)
            observable_parity[observable_index] = (
                observable_parity.get(observable_index, 0)
                + int(reference[absolute].sum())) % 2
            continue
        measurements_so_far += _measurement_count(instruction)

    observables = tuple(
        ObservableRecipe(index, tuple(observable_records[index]), observable_parity[index])
        for index in sorted(observable_records))
    # the ring buffer must hold every round a recipe reaches back to,
    # observables included (a logical readout can span a whole block)
    detector_spans = [recipe.round_index - min(r for r, _ in recipe.records) for recipe in detectors]
    observable_spans = [round_count - min(r for r, _ in recipe.records) for recipe in observables if recipe.records]
    max_span = max(detector_spans + observable_spans, default=0)
    return FormationTable(
        round_count=round_count,
        packet_width=packet_width,
        readout_slot_start=readout_slot_start,
        detectors=tuple(detectors),
        observables=observables,
        max_record_span=max_span,
    )


class StreamingDetectorFormer:
    """The controller stage: one raw packet in per round, detection events out.

    A ring buffer keeps the last max_record_span + 1 packets (two for a
    memory experiment). Every detector of the arriving round starts at its
    reference parity and XORs in its listed bits. The observables come out
    with the last round.
    """

    def __init__(self, table: FormationTable):
        self.table = table
        self.depth = table.max_record_span + 1
        self.packets: dict[int, tuple[int, ...]] = {}

    def feed_packet(self, round_index: int, bits):
        packet = tuple(int(bit) for bit in bits)
        expected = self.table.packet_width[round_index]
        if len(packet) != expected:
            raise ValueError(
                f"round {round_index}: packet has {len(packet)} bits, "
                f"the formation table expects {expected}")
        self.packets[round_index] = packet
        for stale_round in [r for r in self.packets if r <= round_index - self.depth]:
            del self.packets[stale_round]

        events = [(recipe.detector_index, self._form(recipe))
                  for recipe in self.table.detectors_of_round(round_index)]
        observables = None
        if round_index == self.table.round_count:
            observables = [(recipe.observable_index, self._form(recipe))
                           for recipe in self.table.observables]
        return events, observables

    def _form(self, recipe) -> int:
        value = recipe.reference_parity
        for record_round, slot in recipe.records:
            value ^= self.packets[record_round][slot]
        return value


def split_measurements_into_packets(table: FormationTable, measurement_row):
    """Cut one shot's measurement row into per-round packets (the QPU side)."""
    packets = {}
    cursor = 0
    for round_index in range(1, table.round_count + 1):
        width = table.packet_width[round_index]
        packets[round_index] = tuple(int(bit) for bit in measurement_row[cursor:cursor + width])
        cursor += width
    return packets


def form_shot(table: FormationTable, packets_by_round):
    """Whole-shot convenience used by the device for truth and by tests."""
    former = StreamingDetectorFormer(table)
    detector_bits = [0] * len(table.detectors)
    observable_bits = [0] * len(table.observables)
    for round_index in range(1, table.round_count + 1):
        events, observables = former.feed_packet(round_index, packets_by_round[round_index])
        for index, value in events:
            detector_bits[index] = value
        if observables is not None:
            for index, value in observables:
                observable_bits[index] = value
    return tuple(detector_bits), tuple(observable_bits)
