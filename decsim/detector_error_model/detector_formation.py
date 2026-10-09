"""Turns raw measurement bits into detection events, round by round.

A detector's recipe is the raw bits it XORs, addressed as (round, slot)
in the QPU's one-based packet schedule, plus their noiseless reference
parity: Stim's rule (stim.Circuit.compile_m2d_converter). A front end
may declare the packet schedule (measurement_rounds); without one, the
Stim generator layout applies, with the data readout folded into the
last round's packet.

The folding is why a round's detector count is not constant. On a
rotated surface-code memory the preparation layer holds (d*d - 1)/2
detectors, each bulk layer d*d - 1, and the readout layer another
(d*d - 1)/2, folded into the last round: 4, 8 and 12 at d=3, 12, 24 and
36 at d=5 (stim.Circuit.generated). The raw packets are d*d - 1 bits per
round and d*d - 1 + d*d on the last, hence the 240-against-249 and
1200-against-1225 bit counts of the store hop.
"""

import dataclasses
from collections.abc import Iterable, Sequence
from typing import Optional

import stim

import decsim.records.formation as formation_records
from decsim.detector_error_model import detector_chronology


def build_formation_table(
    circuit: stim.Circuit,
    round_count: int,
    *,
    measurement_rounds: Optional[dict[int, int]] = None,
    detector_rounds: Optional[dict[int, int]] = None,
    live_reach: Optional[int] = None,
) -> formation_records.FormationTable:
    """Read the recipe of every detector and observable off the circuit.

    measurement_rounds is the front end's packet schedule, one round per
    absolute measurement index. detector_rounds is each detector's declared
    round in 1..round_count; a detector forms only once every bit it reads
    has arrived, so it may not precede them. live_reach marks a live
    stream's table.
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
    return formation_records.FormationTable(
        round_count=round_count,
        packet_width_by_round=packet_width_by_round,
        readout_slot_start=readout_slot_start,
        detectors=tuple(detectors),
        observables=tuple(observables),
        live_reach=live_reach,
    )


def rounds_read_back(fragment: stim.Circuit, round_width: int) -> int:
    """How many rounds back a fragment reads, run after rounds this wide.

    A record k measurements before the fragment lies at most
    ceil(k / round_width) rounds back. The count is fixed by the program's
    text, so a stream keeps that many rounds, as gem5's TAGE keeps its last
    maxHist outcomes (src/cpu/pred/tage_base.cc:310-313). After rounds that
    measure nothing no count bounds it, so that read is refused. Observable
    records are not counted: no seat forms an observable.
    """
    furthest_count = 0
    measured_so_far = 0
    for instruction in _flat_instructions(fragment):
        if instruction.name != "DETECTOR":
            measured_so_far += _measurement_count(instruction)
            continue
        for offset in _record_offsets(instruction):
            before_start = -offset - measured_so_far
            reach_count = _rounds_back(before_start, round_width)
            furthest_count = max(furthest_count, reach_count)
    return furthest_count


class StreamingDetectorFormer:
    """One seat's former for one operation: a raw packet in, events out.

    It keeps a packet while a round that reads it is neither formed here
    nor retired, as an HEVC decoder keeps each picture the reference set
    names (FFmpeg hevc/refs.c:486-517), so a round formed out of order finds
    its packets. On a live table it also keeps the last live_reach packets.
    Fed in order it holds the last k rounds, k the furthest reach back: the
    running syndrome of IBM's windowed form (Maurer 2510.21600 Algorithm 2).
    It forms no observable; form_shot folds them per shot.
    """

    def __init__(
        self,
        table: formation_records.FormationTable,
        done_rounds: Optional["DoneRounds"] = None,
    ) -> None:
        """done_rounds is shared with the seat.

        A seat that forms only some rounds may make its former after rounds
        have already left the store.
        """
        self.table = table
        self.packets: dict[int, tuple[int, ...]] = {}
        if done_rounds is None:
            done_rounds = DoneRounds()
        # the rounds whose reads of a packet are done here: formed here,
        # or retired, which no read forms here again
        self.done_rounds = done_rounds

    def feed_packet(
        self, round_index: int, bits: Iterable[int]
    ) -> list[tuple[int, int]]:
        """Store one round's packet and form the detectors it completes."""
        self.take_packet(round_index, bits)
        return self.form_round(round_index)

    def take_packet(self, round_index: int, bits: Iterable[int]) -> None:
        """Keep one round's packet beside those held, letting none go."""
        packet = tuple(int(bit) for bit in bits)
        expected_bit_count = self.table.packet_width_by_round[round_index]
        if len(packet) != expected_bit_count:
            raise ValueError(
                f"round {round_index}: packet has {len(packet)} bits, "
                f"the formation table expects {expected_bit_count}"
            )
        self.packets[round_index] = packet

    def form_round(self, round_index: int) -> list[tuple[int, int]]:
        """The round's (detector index, bit) pairs; unread packets go."""
        self.done_rounds.add(round_index)
        recipes = self.table.detectors_of_round(round_index)
        events = [
            (recipe.detector_index, self._form_parity(recipe))
            for recipe in recipes
        ]
        self._let_go_of_unread_packets()
        return events

    def hold_packet(self, round_index: int, bits: Iterable[int]) -> None:
        """Keep one round's packet for later rounds, forming nothing."""
        self.take_packet(round_index, bits)
        self._let_go_of_unread_packets()

    def retire_round(self, round_index: int) -> None:
        """No read forms this round here again; let go of what it alone read."""
        self.done_rounds.add(round_index)
        self._let_go_of_unread_packets()

    def holds_packet(self, round_index: int) -> bool:
        """Whether this round's raw packet is held here."""
        return round_index in self.packets

    def held_bits(self) -> int:
        """The raw bits the former holds."""
        return sum(len(packet) for packet in self.packets.values())

    def extend_table(self, table: formation_records.FormationTable) -> None:
        """Take a longer table whose earlier rounds are unchanged.

        A live circuit supplies the next recipes before their packets. A round
        they read that this seat was never given comes with the read
        (rounds_needed_before).
        """
        _check_formation_prefix(self.table, table)
        self.table = table

    def _let_go_of_unread_packets(self) -> None:
        unread_rounds = []
        for kept_round in self.packets:
            if self._is_still_read(kept_round):
                continue
            unread_rounds.append(kept_round)
        for unread_round in unread_rounds:
            del self.packets[unread_round]

    def _is_still_read(self, kept_round: int) -> bool:
        """Whether a round not done here, or not yet run, reads it."""
        if self._is_read_after_the_table(kept_round):
            return True
        reader_rounds = self.table.rounds_reading(kept_round)
        for reader_round in reader_rounds:
            if reader_round not in self.done_rounds:
                return True
        return False

    def _is_read_after_the_table(self, kept_round: int) -> bool:
        """Whether a live stream's rounds still to run may read it."""
        live_reach = self.table.live_reach
        if live_reach is None:
            return False
        return kept_round > self.table.round_count - live_reach

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


class DoneRounds:
    """One operation's rounds done at a seat, in memory that does not grow.

    Every round up to through is done, and those done above it are listed
    in above, as TCP's cumulative ACK and SACK blocks (RFC 2018, section 3).
    Rounds finish in about round order, so above stays small.
    """

    def __init__(self) -> None:
        self.through = 0
        self.above: set[int] = set()

    def add(self, round_index: int) -> None:
        """Mark the round done, moving the watermark over any run it closes."""
        if round_index <= self.through:
            return
        self.above.add(round_index)
        next_round = self.through + 1
        while next_round in self.above:
            self.above.remove(next_round)
            self.through = next_round
            next_round += 1

    def __contains__(self, round_index: int) -> bool:
        if round_index <= self.through:
            return True
        return round_index in self.above


def split_measurements_into_packets(
    table: formation_records.FormationTable, measurement_row: Sequence[int]
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
    table: formation_records.FormationTable,
    packets_by_round: dict[int, Sequence[int]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Every detector and observable bit of one shot, in index order.

    The device calls it once per shot with complete packets. Each observable
    starts at its reference parity and XORs in each round's records of it,
    as Stim's frame simulator does at OBSERVABLE_INCLUDE
    (src/stim/simulators/frame_simulator.inl:233-243, v1.16.0); XOR
    commutes, so folding by round gives the same parity.
    """
    streaming_former = StreamingDetectorFormer(table)
    detector_bits = [0] * len(table.detectors)
    observable_bits = [recipe.reference_parity for recipe in table.observables]
    after_last_round = table.round_count + 1
    for round_index in range(1, after_last_round):
        packet = packets_by_round[round_index]
        events = streaming_former.feed_packet(round_index, packet)
        _store_bits(detector_bits, events)
        _fold_observables(observable_bits, table, round_index, packet)
    return tuple(detector_bits), tuple(observable_bits)


def measurements_forming(
    table: formation_records.FormationTable,
    detection_events: Sequence[int],
    observable_flips: Sequence[int],
) -> tuple[int, ...]:
    """A raw measurement row that form_shot turns into these bits.

    form_shot read backwards: each recipe is one equation over GF(2),
    its reference parity XOR its records equal to its bit. Elimination
    pivots each equation on its latest record, then back substitution
    sets the pivots from the earliest up; a bit no pivot names stays 0,
    one of the rows Stim's sampler could have drawn for the same events,
    since every detector and observable reads only through its recipe.
    """
    offsets = _packet_offsets(table)
    pivots: dict = {}
    equations = _formation_equations(
        table, offsets, detection_events, observable_flips
    )
    for record_mask, parity in equations:
        _eliminate_into(pivots, record_mask, parity)
    packet_widths = table.packet_width_by_round.values()
    row_width = sum(packet_widths)
    row = _back_substituted(pivots, row_width)
    return tuple(row)


@dataclasses.dataclass(frozen=True)
class _CircuitTables:
    """What every recipe is read against."""

    packet_of_measurement: dict
    reference_sample: object
    coordinates: dict
    detector_rounds: Optional[dict]


def _check_formation_prefix(
    previous: formation_records.FormationTable,
    table: formation_records.FormationTable,
) -> None:
    for round_index, width in previous.packet_width_by_round.items():
        if table.packet_width_by_round.get(round_index) != width:
            raise RuntimeError("formation extension changes an existing packet")
    previous_detector_count = len(previous.detectors)
    detector_prefix = table.detectors[:previous_detector_count]
    if detector_prefix != previous.detectors:
        raise RuntimeError("formation extension changes an existing detector")


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

    The measurements before a DETECTOR group belong to the round it
    announces (time coordinate plus one). Those after the last group, and
    groups past round_count (Stim's post-readout layer), fold into the last
    round.
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
    """One round per absolute measurement, declared or read off the circuit."""
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
    _check_declared_rounds(rounds, round_count)
    return rounds, None


def _check_declared_rounds(rounds: list[int], round_count: int) -> None:
    if any(not 1 <= round_index <= round_count for round_index in rounds):
        raise ValueError(
            "declared measurement rounds must lie in 1..round_count"
        )
    if _has_a_backward_step(rounds):
        raise ValueError("declared measurement rounds must be non-decreasing")


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
    detectors: list[formation_records.DetectorRecipe] = []
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
        recipe = formation_records.ObservableRecipe(index, records, parity)
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
) -> formation_records.DetectorRecipe:
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
    return formation_records.DetectorRecipe(
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


def _layer_kind_for(record_count: int) -> formation_records.LayerKind:
    if record_count == 1:
        return formation_records.LayerKind.PREPARATION
    if record_count == 2:
        return formation_records.LayerKind.BULK
    return formation_records.LayerKind.READOUT


def _store_bits(bits: list[int], indexed_values) -> None:
    for index, value in indexed_values:
        bits[index] = value


def _fold_observables(
    parities: list[int],
    table: formation_records.FormationTable,
    round_index: int,
    packet,
) -> None:
    """XOR the round's observable records into the running parities."""
    for observable_index, slot in table.observable_slots_of_round(round_index):
        parities[observable_index] ^= int(packet[slot])


def _rounds_back(before_start: int, round_width: int) -> int:
    """How many rounds back a record before_start measurements back lies."""
    if before_start <= 0:
        return 0
    if round_width == 0:
        raise ValueError(
            "a detector reads a record from before its round, but the "
            "repeated round measures nothing, so the record lies a round "
            "further back every round and no stream can keep it; give the "
            "repeated round a measurement"
        )
    return -(-before_start // round_width)


def _packet_offsets(table: formation_records.FormationTable) -> dict:
    """Each round's first index in the row, as the packets cut it."""
    offsets = {}
    cursor = 0
    after_last_round = table.round_count + 1
    for round_index in range(1, after_last_round):
        offsets[round_index] = cursor
        cursor += table.packet_width_by_round[round_index]
    return offsets


def _formation_equations(
    table: formation_records.FormationTable,
    offsets: dict,
    detection_events: Sequence[int],
    observable_flips: Sequence[int],
) -> list:
    """(record mask, parity) for every detector, then every observable.

    A mask is an int with one bit per row index the recipe reads.
    """
    equations = []
    for recipe in table.detectors:
        event = detection_events[recipe.detector_index]
        equation = _recipe_equation(recipe, offsets, event)
        equations.append(equation)
    for recipe in table.observables:
        flip = observable_flips[recipe.observable_index]
        equation = _recipe_equation(recipe, offsets, flip)
        equations.append(equation)
    return equations


def _recipe_equation(recipe, offsets: dict, bit: int) -> tuple:
    """One recipe's records as a mask, against its bit and reference."""
    record_mask = 0
    for round_index, slot in recipe.records:
        row_index = offsets[round_index] + slot
        record_mask ^= 1 << row_index
    parity = int(bit) ^ recipe.reference_parity
    return record_mask, parity


def _eliminate_into(pivots: dict, record_mask: int, parity: int) -> None:
    """Reduce one equation by the pivots; keep it under its latest record.

    An equation that reduces to nothing must have parity 0: events that
    no row forms are not a shot of this circuit.
    """
    while record_mask:
        latest_record = record_mask.bit_length() - 1
        if latest_record not in pivots:
            pivots[latest_record] = (record_mask, parity)
            return
        pivot_mask, pivot_parity = pivots[latest_record]
        record_mask ^= pivot_mask
        parity ^= pivot_parity
    if parity:
        raise RuntimeError(
            "the detection events and observable flips are formed by no "
            "measurement row of this circuit"
        )


def _back_substituted(pivots: dict, row_width: int) -> list[int]:
    """The row the reduced equations fix, earliest pivot first.

    A pivot's other records are all earlier, so each is set by then.
    """
    row = [0] * row_width
    for latest_record in sorted(pivots):
        record_mask, parity = pivots[latest_record]
        earlier_records = record_mask ^ (1 << latest_record)
        earlier_parity = _parity_of_records(row, earlier_records)
        row[latest_record] = parity ^ earlier_parity
    return row


def _parity_of_records(row: list[int], record_mask: int) -> int:
    """The XOR of the row's bits the mask names."""
    parity = 0
    while record_mask:
        lowest_record = record_mask & -record_mask
        row_index = lowest_record.bit_length() - 1
        parity ^= row[row_index]
        record_mask ^= lowest_record
    return parity
