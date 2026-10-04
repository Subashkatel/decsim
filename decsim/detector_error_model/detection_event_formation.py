"""The detection event former, seated at the points the settings name.

A detection event is a parity of raw outcomes, so its value is the same
wherever it is formed: a seat moves only the width the later hops
carry, the state the seat holds and the clock it is charged on. The
seats are the published placements. At the controller, measurements
"are converted into detections and then streamed to the real-time
decoding software" (Google 2408.13687 lines 474-476). At the decoder
chip's input, a "detector window processing component calculates
parities" (Maurer 2510.21600 lines 234-236), seated at either store's
receiving end. Inside the decoder, "The decoder computes the syndrome
from measurement outcomes" (Caune 2410.05202 lines 1252-1255), as
LILLIPUT's Event Detection Logic (2108.06569 lines 500-509) and cudaqx's
real-time decoder (libs/qec/lib/decoder.cpp:426-432) do.

Every seat keeps its own history, as cudaqx keeps a detector buffer per
decoder instance (decoder.cpp:51-61): the raw packets its recipes still
read, and the rounds it formed, so a round two windows read is formed
once.
"""

import dataclasses
import operator
from typing import Any, Optional

import decsim.detector_error_model.detector_formation as detector_formation
import decsim.detector_error_model.settings as detection_event_settings
import decsim.ports as ports
import decsim.records.rounds as round_records
import decsim.trace_source as trace_source


class SeatedFormation:
    """The run's former at every seat detection_events.formed_at names.

    A seat not listed hands a round on as it came, at no cost. state_held
    reports the raw bits a seat holds per operation, IBM's running syndrome
    (Maurer 2510.21600 Algorithm 2), which no referent sizes, so it is
    reported and never refused.
    """

    def __init__(
        self,
        source: Optional[ports.DetectionEventFormer],
        settings: detection_event_settings.DetectionEventSettings,
    ) -> None:
        _check_charged_cost_has_a_clock(settings)
        self.settings = settings
        self.clock = settings.clock
        # the source's recipes, None when it answers none
        self.recipes = source
        self.trace = _TraceSources()
        self.history_by_seat = {}
        for seat in settings.formed_at:
            history = _SeatHistory(seat, source)
            history.state_held = self.trace.state_held
            self.history_by_seat[seat] = history

    def forms_at(self, seat: str) -> bool:
        """Whether the named seat forms the rounds that cross it."""
        return seat in self.history_by_seat

    def form_at(
        self, seat: str, fragments: tuple, rounds_before: tuple = ()
    ) -> tuple:
        """The fragments as they leave the seat: formed there, or as they came.

        A decoder seat keeps the landed width, since the unit's input memory was
        written the raw round. rounds_before are the raw rounds before the first
        fragment that the seat needs (rounds_needed_before); it holds them and
        neither forms nor returns them.
        """
        history = self.history_by_seat.get(seat)
        if history is None:
            return fragments
        history.hold(rounds_before)
        return history.form(fragments)

    def width_at(self, seat: str, fragments: tuple) -> Optional[int]:
        """The width one round's fragments leave the seat at, forming nothing.

        A store weighs its room before the round lands, as gem5 makes room at
        the compressed size (src/mem/cache/base.cc:1678-1698), so this must not
        move the seat's history.
        """
        history = self.history_by_seat.get(seat)
        if history is None:
            return round_records.fragment_wire_bits(fragments)
        return history.width(fragments)

    def rounds_needed_before(
        self,
        seat: str,
        operation_id: Any,  # an opaque identity
        first_round: int,
        last_round: int,
    ) -> tuple:
        """The raw rounds before a read's first round the seat must be given.

        A detector compares a round against earlier ones, one round back on a
        surface code (LILLIPUT 2108.06569 lines 499-510) and further on others,
        so a read is given exactly the rounds its unformed rounds' recipes read
        and the seat does not already hold.
        """
        history = self.history_by_seat.get(seat)
        if history is None:
            return ()
        return history.rounds_needed_before(
            operation_id, first_round, last_round
        )

    def earlier_rounds_read(
        self,
        operation_id: Any,  # an opaque identity
        first_round: int,
    ) -> tuple[int, ...]:
        """The raw rounds before first_round it or any later round reads.

        What a stream's window holds for a forming seat, since a later window
        registers after it. Empty for a source with no recipes.
        """
        if self.recipes is None:
            return ()
        table = self.recipes.formation_table(operation_id)
        return table.earlier_rounds_read(first_round)

    def retire_round(self, round_key: tuple) -> None:
        """No read forms this round again: every seat lets go of it.

        The round left the store the plan's windows read, so no window or
        escalation reads it after now.
        """
        operation_id, round_index = round_key
        for history in self.history_by_seat.values():
            history.retire(operation_id, round_index)

    def claim_rounds(self, seat: str, round_keys: tuple) -> tuple:
        """The round keys no earlier job of the seat claimed, now claimed.

        A decoder tier charges a job for the rounds it is first to claim. A job
        claims while the store holds its rounds, and a round retires only when
        no read holds it, so a claim never outlives its round.
        """
        history = self.history_by_seat[seat]
        return history.memory.claim(round_keys)

    def return_claim(self, seat: str, round_keys: tuple) -> None:
        """A job that never started gives its claimed round keys back."""
        history = self.history_by_seat[seat]
        history.memory.unclaim(round_keys)

    def check_settled(self) -> None:
        """At the end of a run no seat may still hold a raw round."""
        for history in self.history_by_seat.values():
            history.check_settled()

    def cycles_at(self, seat: str, round_count: int) -> int:
        """What forming round_count rounds together costs the seat, on clock."""
        if seat not in self.history_by_seat:
            return 0
        return self.settings.cycles_for(round_count)


class _SeatHistory:
    """What one seat holds: the packets its recipes read, the rounds formed."""

    def __init__(
        self,
        seat: str,
        source: Optional[ports.DetectionEventFormer],
    ) -> None:
        self.seat = seat
        # the source's recipes, read per operation
        self.recipes = source
        # where the seat reports the raw bits it holds; bound by the
        # placement, silent for a history on its own
        self.state_held = trace_source.SILENT
        self.former_by_operation: dict = {}
        self.memory = _SeatMemory()

    @property
    def keeps_landed_width(self) -> bool:
        """A decoder seat keeps the width a round landed at."""
        return self.seat in detection_event_settings.DECODER_SEATS

    def form(self, fragments: tuple) -> tuple:
        """Every round among the fragments formed, in the order they came.

        A source with no recipes leaves at its stated event width; a timing-only
        round, which carries no bits, leaves as it came.
        """
        if self.recipes is None:
            return self._at_the_stated_width(fragments)
        formed = []
        for round_fragments in _rounds_in_order(fragments):
            leaving = self._form_round(round_fragments)
            formed.extend(leaving)
        return tuple(formed)

    def width(self, fragments: tuple) -> Optional[int]:
        """The width form would give one round's fragments."""
        if self.keeps_landed_width:
            return round_records.fragment_wire_bits(fragments)
        if self.recipes is None:
            stated = self._at_the_stated_width(fragments)
            return round_records.fragment_wire_bits(stated)
        if _joined_bits(fragments) is None:
            return round_records.fragment_wire_bits(fragments)
        first = fragments[0]
        table = self.recipes.formation_table(first.operation_id)
        detectors = table.detectors_of_round(first.round_index)
        return len(detectors)

    def _at_the_stated_width(self, fragments: tuple) -> tuple:
        """Each fragment at its stated event width; a decoder seat keeps it."""
        if self.keeps_landed_width:
            return fragments
        leaving = []
        for fragment in fragments:
            stated = _as_stated_events(fragment)
            leaving.append(stated)
        return tuple(leaving)

    def hold(self, fragments: tuple) -> None:
        """Keep raw rounds for the detectors of the rounds after them."""
        if self.recipes is None:
            return
        for round_fragments in _rounds_in_order(fragments):
            bits = _joined_bits(round_fragments)
            if bits is None:
                continue
            first = round_fragments[0]
            former = self._former_for(first.operation_id)
            former.hold_packet(first.round_index, bits)
            self._report_state(first.operation_id, former)

    def retire(
        self,
        operation_id: Any,  # an opaque identity
        round_index: int,
    ) -> None:
        """No read forms this round here again; held packets may go.

        A seat with no former yet records the round for the former it makes
        later. The former takes the newest table first, so a live stream that
        has run its final round keeps no rounds for one still to run.
        """
        self.memory.retire(operation_id, round_index)
        if operation_id not in self.former_by_operation:
            return
        former = self._former_for(operation_id)
        former.retire_round(round_index)
        self._report_state(operation_id, former)

    def check_settled(self) -> None:
        """No former of this seat holds a raw round."""
        for operation_id, former in self.former_by_operation.items():
            if not former.packets:
                continue
            held_rounds = sorted(former.packets)
            raise RuntimeError(
                f"the {self.seat} seat still holds raw rounds {held_rounds} "
                f"of operation {operation_id!r} at the end of the run, "
                "though every round has formed or left the store"
            )

    def rounds_needed_before(
        self,
        operation_id: Any,  # an opaque identity
        first_round: int,
        last_round: int,
    ) -> tuple:
        """The raw rounds before first_round the read needs and lacks."""
        if self.recipes is None:
            return ()
        table = self.recipes.formation_table(operation_id)
        stop_round = last_round + 1
        unformed_rounds = self._unformed(operation_id, first_round, stop_round)
        read_rounds = table.rounds_read_by(unformed_rounds)
        return self._lacking(operation_id, read_rounds, first_round)

    def _lacking(
        self,
        operation_id: Any,  # an opaque identity
        read_rounds: set,
        first_round: int,
    ) -> tuple:
        """The read rounds before first_round whose packet the seat lacks."""
        former = self.former_by_operation.get(operation_id)
        lacking_rounds = []
        for round_index in sorted(read_rounds):
            if round_index >= first_round:
                continue
            if former is not None and former.holds_packet(round_index):
                continue
            lacking_rounds.append(round_index)
        return tuple(lacking_rounds)

    def _unformed(
        self,
        operation_id: Any,  # an opaque identity
        first_round: int,
        stop_round: int,
    ) -> tuple:
        """The rounds from first_round to before stop_round never formed."""
        done_rounds = self.memory.done_rounds_of(operation_id)
        unformed_rounds = []
        for round_index in range(first_round, stop_round):
            if round_index in done_rounds:
                continue
            unformed_rounds.append(round_index)
        return tuple(unformed_rounds)

    def _form_round(self, fragments: list) -> tuple:
        """One round's fragments as one fragment of its events.

        Stim rec targets span fragments, so the round is formed once, whole, as
        one event vector shared by its patches.
        """
        bits = _joined_bits(fragments)
        if bits is None:
            return tuple(fragments)
        first = fragments[0]
        events = self._events(first.operation_id, first.round_index, bits)
        patches = round_records.fragment_patch_ids(fragments)
        landed_bits = round_records.fragment_wire_bits(fragments)
        formed = dataclasses.replace(
            first, bits=events, size_bits=landed_bits, patch_ids=patches
        )
        if self.keeps_landed_width:
            return (formed,)
        sized = _sized_by_its_bits(formed)
        return (sized,)

    def _events(
        self,
        operation_id: Any,  # an opaque identity
        round_index: int,
        raw_bits,
    ) -> tuple:
        """The round's events, formed here the first time the seat sees it."""
        former = self._former_for(operation_id)
        key = (operation_id, round_index)
        remembered = self.memory.events_by_round.get(key)
        if remembered is not None:
            former.hold_packet(round_index, raw_bits)
            self._report_state(operation_id, former)
            return remembered
        former.take_packet(round_index, raw_bits)
        self._report_state(operation_id, former)
        events = former.form_round(round_index)
        values = tuple(value for _, value in events)
        self.memory.events_by_round[key] = values
        return values

    def _report_state(
        self,
        operation_id: Any,  # an opaque identity
        former: detector_formation.StreamingDetectorFormer,
    ) -> None:
        """The raw bits this seat holds for the operation, now."""
        held_bits = former.held_bits()
        self.state_held.fire(self.seat, operation_id, held_bits)

    def _former_for(
        self,
        operation_id: Any,  # an opaque identity
    ) -> detector_formation.StreamingDetectorFormer:
        """The seat's former for the operation, on the newest table."""
        table = self.recipes.formation_table(operation_id)
        former = self.former_by_operation.get(operation_id)
        if former is None:
            done_rounds = self.memory.done_rounds_of(operation_id)
            former = detector_formation.StreamingDetectorFormer(
                table, done_rounds
            )
            self.former_by_operation[operation_id] = former
            return former
        if former.table is not table:
            former.extend_table(table)
        return former


class _SeatMemory:
    """What a seat remembers of the rounds it is done with."""

    def __init__(self) -> None:
        # per operation, the rounds formed here or retired, kept from
        # the first retirement on, so a former made later knows them
        self.done_by_operation: dict = {}
        # the events of rounds formed here and not yet retired, which a
        # second read of the round is answered from
        self.events_by_round: dict = {}
        # the round keys a decoder tier's jobs claimed, until retired
        self.claimed_keys: set = set()

    def done_rounds_of(
        self,
        operation_id: Any,  # an opaque identity
    ) -> detector_formation.DoneRounds:
        """The operation's rounds formed here or retired."""
        new_record = detector_formation.DoneRounds()
        return self.done_by_operation.setdefault(operation_id, new_record)

    def retire(
        self,
        operation_id: Any,  # an opaque identity
        round_index: int,
    ) -> None:
        """The round is done here; its events and claim are asked no more."""
        done_rounds = self.done_rounds_of(operation_id)
        done_rounds.add(round_index)
        round_key = (operation_id, round_index)
        self.events_by_round.pop(round_key, None)
        self.claimed_keys.discard(round_key)

    def claim(self, round_keys: tuple) -> tuple:
        """The keys not claimed before, now claimed."""
        fresh = []
        for round_key in round_keys:
            if round_key in self.claimed_keys:
                continue
            fresh.append(round_key)
        self.claimed_keys.update(fresh)
        return tuple(fresh)

    def unclaim(self, round_keys: tuple) -> None:
        """The keys are claimed no more."""
        self.claimed_keys.difference_update(round_keys)


def _as_stated_events(
    fragment: round_records.RetainedSyndromeFragment,
) -> round_records.RetainedSyndromeFragment:
    """The fragment at its stated event width; as it came when none is.

    A fake-bit source's events are its first random bits.
    """
    event_bits = fragment.event_bits
    if event_bits is None:
        return fragment
    bits = fragment.bits
    if bits is not None:
        bits = bits[:event_bits]
    return dataclasses.replace(
        fragment, bits=bits, size_bits=event_bits, event_bits=None
    )


def _sized_by_its_bits(
    fragment: round_records.RetainedSyndromeFragment,
) -> round_records.RetainedSyndromeFragment:
    """One fragment sized by the bits it holds, its events included."""
    if fragment.bits is None:
        return fragment
    return dataclasses.replace(fragment, size_bits=len(fragment.bits))


def _joined_bits(fragments) -> Optional[list]:
    """One round's bits in measurement order; None for a timing-only round."""
    bits = []
    for fragment in fragments:
        if fragment.bits is None:
            return None
        bits.extend(fragment.bits)
    return bits


def _rounds_in_order(fragments) -> list:
    """The fragments grouped by round, each group in measurement order."""
    fragments_by_round: dict = {}
    for fragment in fragments:
        key = (fragment.operation_id, fragment.round_index)
        group = fragments_by_round.setdefault(key, [])
        group.append(fragment)
    by_fragment_index = operator.attrgetter("fragment_index")
    ordered = []
    for group in fragments_by_round.values():
        in_measurement_order = sorted(group, key=by_fragment_index)
        ordered.append(in_measurement_order)
    return ordered


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the seated former reports, as one member."""

    state_held: trace_source.TraceSource = trace_source.new_source()


def _check_charged_cost_has_a_clock(
    settings: detection_event_settings.DetectionEventSettings,
) -> None:
    """A charged cost counts its cycles on a clock, its own or the machine's."""
    charged = settings.latency_cycles + settings.cycles_per_round
    if charged == 0 or settings.clock is not None:
        return
    raise ValueError(
        "a charged detection_events cost needs the clock its cycles are "
        "counted on; name one, or give the machine a clock"
    )
