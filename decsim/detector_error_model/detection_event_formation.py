"""The detection event former, seated at the points the yaml names.

A detection event is a parity of raw measurement outcomes
(detector_formation.py), so its value is the same wherever it is formed:
a seat moves the width every hop after it carries, the state the seat
holds and the clock the conversion is charged on, and nothing else
(detection_events.formed_at, detector_error_model/settings.py). The
published placements sit at every seat decsim has. At the controller,
"inside the workstation, measurements are converted into detections
and then streamed to the real-time decoding software via a shared
memory buffer" (Google 2408.13687 lines 474-476). At the decoder chip's
input path, its "detector window processing component calculates
parities of groups of measurements" (Maurer 2510.21600 lines 234-236),
which decsim seats at either store's receiving end. Inside the decoder,
the controller writes the outcomes "sequentially to the decoder" and
"The decoder computes the syndrome from measurement outcomes" (Caune
2410.05202 lines 1252-1255), LILLIPUT's Event Detection Logic block
sits inside it (2108.06569 lines 500-509), and cudaqx's real-time
decoder resolves "every detector whose measurements have now all
arrived" (cudaqx libs/qec/lib/decoder.cpp:426-432).

Every seat keeps its own history, as every real one does: LILLIPUT's
block is inside its decoder and cudaqx keeps a detector buffer per
decoder instance (cudaqx libs/qec/lib/decoder.cpp:51-61, 128-129). A
seat holds the last raw packets its recipes still read, and remembers
the rounds it has formed, so a round two windows of one seat read is
formed once.
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

    A seat not in the list hands a round on as it came and costs
    nothing. The source's recipes are read per operation
    (ports.DetectionEventFormer); each seat forms from its own packets.
    The burst detector, when the run has one, counts the rounds one
    seat forms, the first seat on the path the primary tier reads.

    Trace source: state_held(seat, operation_id, bits) each time a seat
    takes a raw round, bits being the raw packets the seat then holds
    for that operation: the state a former keeps, IBM's running
    syndrome (Maurer 2510.21600 Algorithm 2, lines 760-770), which no
    referent sizes, so it is reported and never refused.
    """

    def __init__(
        self,
        source: Optional[ports.DetectionEventFormer],
        settings: detection_event_settings.DetectionEventSettings,
        observed_seat: Optional[str] = None,
        burst_detector: Optional[ports.BurstDetector] = None,
    ) -> None:
        self.settings = settings
        self.clock = settings.clock
        # the source's recipes, None when it answers none
        self.recipes = source
        self.trace = _TraceSources()
        self.history_by_seat = {}
        for seat in settings.formed_at:
            observer = None
            if seat == observed_seat:
                observer = burst_detector
            history = _SeatHistory(seat, source, observer)
            history.state_held = self.trace.state_held
            self.history_by_seat[seat] = history

    def forms_at(self, seat: str) -> bool:
        """Whether the named seat forms the rounds that cross it."""
        return seat in self.history_by_seat

    def form_at(
        self, seat: str, fragments: tuple, rounds_before: tuple = ()
    ) -> tuple:
        """The fragments as they leave the seat: formed there, or as they came.

        A seat outside a decoder unit sends the events on at their own
        width; a decoder seat keeps the width that landed, since the
        unit's input memory was written the raw round and a memory
        counts what is written into it. rounds_before are the raw rounds
        before the first of the fragments, in order, given to a seat that
        needs them (rounds_needed_before): the seat holds them for that
        round's detectors and neither forms nor returns them.
        """
        history = self.history_by_seat.get(seat)
        if history is None:
            return fragments
        history.hold(rounds_before)
        return history.form(fragments)

    def width_at(self, seat: str, fragments: tuple) -> Optional[int]:
        """The width one round's fragments leave the seat at, forming nothing.

        A store weighs its room at the width it will hold, as gem5 makes
        room for a block at the size its compressor will store it at
        (src/mem/cache/base.cc:1678-1698), so the width is asked before
        the round lands and must not move the seat's history.
        """
        history = self.history_by_seat.get(seat)
        if history is None:
            return round_records.fragment_wire_bits(fragments)
        return history.width(fragments)

    def rounds_needed_before(
        self, seat: str, operation_id: Any, first_round: int, last_round: int
    ) -> tuple:
        """The raw rounds before a read's first round the seat must be given.

        A detector compares a round against earlier ones, one round back
        on a surface code (LILLIPUT 2108.06569 lines 499-510) and further
        on others, so a read of first_round to last_round is given the
        rounds its unformed rounds' recipes read and no round between
        (detector_formation.FormationTable rounds_read_by), the rounds
        its stream keeps for it (earlier_rounds_read). A round the seat
        formed before is answered from what it remembers, and its former
        keeps its packet, or the packet it was given raw, while an
        unformed round reads it
        (detector_formation.StreamingDetectorFormer), so it is not given
        again.
        """
        history = self.history_by_seat.get(seat)
        if history is None:
            return ()
        return history.rounds_needed_before(
            operation_id, first_round, last_round
        )

    def earlier_rounds_read(
        self, operation_id: Any, first_round: int
    ) -> tuple[int, ...]:
        """The raw rounds before first_round it or any later round reads.

        What a stream's window holds before its first round for a seat
        that forms, whatever that seat has formed by the time it reads
        (detector_formation.FormationTable earlier_rounds_read): a later
        window registers after it, so it keeps what that window reads
        too. None for a source with no recipes, which forms nothing.
        """
        if self.recipes is None:
            return ()
        table = self.recipes.formation_table(operation_id)
        return table.earlier_rounds_read(first_round)

    def retire_round(self, round_key: tuple) -> None:
        """No read forms this round again: every seat's reads of it are done.

        The round left the store the plan's windows read, so no window
        or escalation reads it after now, and every seat stops keeping
        packets for it (detector_formation.StreamingDetectorFormer
        retire_round): a round an escalation took to the strong side, a
        sealed stream's last rounds and a clipped tail are among them.
        """
        operation_id, round_index = round_key
        for history in self.history_by_seat.values():
            history.retire(operation_id, round_index)

    def check_settled(self) -> None:
        """At the end of a run no seat may still hold a raw round.

        Every round has formed or left the store by then, so a packet
        still held is one no round would ever have let go of.
        """
        for history in self.history_by_seat.values():
            history.check_settled()

    def cycles_at(self, seat: str, round_count: int) -> int:
        """What forming round_count rounds together costs the seat, on clock."""
        if seat not in self.history_by_seat:
            return 0
        return self.settings.cycles_for(round_count)


class _SeatHistory:
    """What one seat holds: the packets its recipes read, the rounds formed.

    A detector compares this round's outcomes against the round before
    it (LILLIPUT 2108.06569 lines 499-510), so a round is formed from
    the packets the seat holds, each kept while a round that reads it is
    unformed here (detector_formation.StreamingDetectorFormer). A round
    the seat formed before is answered from what it remembers, and its
    packet is held for the rounds after it.
    """

    def __init__(
        self,
        seat: str,
        source: Optional[ports.DetectionEventFormer],
        observer: Optional[ports.BurstDetector],
    ) -> None:
        self.seat = seat
        # the source's recipes, read per operation
        self.recipes = source
        self.observer = observer
        # where the seat reports the raw bits it holds; bound by the
        # placement, silent for a history on its own
        self.state_held = trace_source.SILENT
        self.former_by_operation: dict = {}
        # per operation, the rounds formed here or retired, kept from
        # the first retirement on, so a former made later knows them
        self.done_by_operation: dict = {}
        self.events_by_round: dict = {}

    @property
    def keeps_landed_width(self) -> bool:
        """A decoder seat keeps the width a round landed at."""
        return self.seat in detection_event_settings.DECODER_SEATS

    def form(self, fragments: tuple) -> tuple:
        """Every round among the fragments formed, in the order they came.

        A source with no recipes states the width its events take
        (QPUReadout.event_bits), and the fragments leave at it. The
        fragments as they came when there is nothing to form them with
        or from: a timing-only round of a circuit source, which carries
        no bits.
        """
        if self.recipes is None:
            return self._at_the_stated_width(fragments)
        formed = []
        for round_fragments in _rounds_in_order(fragments):
            leaving = self._form_round(round_fragments)
            formed.extend(leaving)
        return tuple(formed)

    def width(self, fragments: tuple) -> Optional[int]:
        """The width one round's fragments take once formed here.

        The width form gives them: the landed width at a decoder seat or
        for a timing-only round, the stated event width for a source with
        no recipes, and one bit per detector of the round otherwise
        (detector_formation.StreamingDetectorFormer.feed_packet).
        """
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
        """Each fragment at the event width its source stated.

        A decoder seat keeps the width that landed, as for any round.
        """
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

    def retire(self, operation_id: Any, round_index: int) -> None:
        """No read forms this round here again; held packets may go.

        A seat with no former yet records the round, for the former it
        makes later. The former takes the source's newest table first,
        so a live stream that has since run its final round no longer
        keeps rounds for a round still to run.
        """
        done_rounds = self.done_by_operation.setdefault(operation_id, set())
        done_rounds.add(round_index)
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
        self, operation_id: Any, first_round: int, last_round: int
    ) -> tuple:
        """The raw rounds before first_round that forming the read reads.

        Exactly the rounds its unformed rounds' recipes read
        (FormationTable rounds_read_by) that the seat does not hold raw:
        a round given to it or formed there before stays in its former
        while an unformed round reads it. A source with no recipes forms
        nothing, so nothing is missing.
        """
        if self.recipes is None:
            return ()
        table = self.recipes.formation_table(operation_id)
        stop_round = last_round + 1
        unformed_rounds = self._unformed(operation_id, first_round, stop_round)
        read_rounds = table.rounds_read_by(unformed_rounds)
        return self._lacking(operation_id, read_rounds, first_round)

    def _lacking(
        self, operation_id: Any, read_rounds: set, first_round: int
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
        self, operation_id: Any, first_round: int, stop_round: int
    ) -> tuple:
        """The rounds from first_round to before stop_round never formed."""
        unformed_rounds = []
        for round_index in range(first_round, stop_round):
            if (operation_id, round_index) in self.events_by_round:
                continue
            unformed_rounds.append(round_index)
        return tuple(unformed_rounds)

    def _form_round(self, fragments: list) -> tuple:
        """One round's fragments as one fragment of its events.

        Stim rec targets span acquisition fragments, so the round is
        formed once, whole, in measurement order, as one event vector
        shared by all contributing patches.
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

    def _events(self, operation_id: Any, round_index: int, raw_bits) -> tuple:
        """The round's events, formed here the first time the seat sees it."""
        former = self._former_for(operation_id)
        key = (operation_id, round_index)
        remembered = self.events_by_round.get(key)
        if remembered is not None:
            former.hold_packet(round_index, raw_bits)
            self._report_state(operation_id, former)
            return remembered
        former.take_packet(round_index, raw_bits)
        self._report_state(operation_id, former)
        events = former.form_round(round_index)
        values = tuple(value for _, value in events)
        self.events_by_round[key] = values
        if self.observer is not None:
            self.observer.observe_round(operation_id, round_index, values)
        return values

    def _report_state(
        self,
        operation_id: Any,
        former: detector_formation.StreamingDetectorFormer,
    ) -> None:
        """The raw bits this seat holds for the operation, now."""
        held_bits = former.held_bits()
        self.state_held.fire(self.seat, operation_id, held_bits)

    def _former_for(
        self, operation_id: Any
    ) -> detector_formation.StreamingDetectorFormer:
        """This seat's former for the operation, on the source's newest table.

        A live stream's table grows as its rounds execute, and the
        former takes the longer table without losing what it holds.
        """
        table = self.recipes.formation_table(operation_id)
        former = self.former_by_operation.get(operation_id)
        if former is None:
            done_rounds = self.done_by_operation.setdefault(operation_id, set())
            former = detector_formation.StreamingDetectorFormer(
                table, done_rounds
            )
            self.former_by_operation[operation_id] = former
            return former
        if former.table is not table:
            former.extend_table(table)
        return former


def _as_stated_events(
    fragment: round_records.RetainedSyndromeFragment,
) -> round_records.RetainedSyndromeFragment:
    """The fragment formed at its stated width; as it came when none is.

    A fake-bit source's events are the first of its random bits, since
    random bits carry no parity to form.
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
    """Every event the seated former reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (gem5 src/base/stats/group.hh:60-92); a component's events are the
    same shape.
    """

    state_held: trace_source.TraceSource = trace_source.new_source()
