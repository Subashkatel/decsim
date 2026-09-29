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
        self, seat: str, fragments: tuple, round_before: tuple = ()
    ) -> tuple:
        """The fragments as they leave the seat: formed there, or as they came.

        A seat outside a decoder unit sends the events on at their own
        width; a decoder seat keeps the width that landed, since the
        unit's input memory was written the raw round and a memory
        counts what is written into it. round_before is the raw round
        before the first of the fragments, given to a seat that needs it
        (needs_the_round_before): the seat holds it for that round's
        detectors and neither forms nor returns it.
        """
        history = self.history_by_seat.get(seat)
        if history is None:
            return fragments
        history.hold(round_before)
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

    def needs_the_round_before(
        self, seat: str, operation_id: Any, round_index: int
    ) -> bool:
        """Whether the seat must be given the raw round before this one.

        A detector compares a round against the one before it (LILLIPUT
        2108.06569 lines 499-510), so a seat that has not formed this
        round reads the round before it, except on its operation's first
        round, which compares against the reset. A round the seat formed
        before is answered from what it remembers and needs nothing.
        """
        history = self.history_by_seat.get(seat)
        if history is None or round_index <= 1:
            return False
        round_key = (operation_id, round_index)
        return not history.has_formed(round_key)

    def cycles_at(self, seat: str, round_count: int) -> int:
        """What forming round_count rounds together costs the seat, on clock."""
        if seat not in self.history_by_seat:
            return 0
        return self.settings.cycles_for(round_count)


class _SeatHistory:
    """What one seat holds: the packets its recipes read, the rounds formed.

    A detector compares this round's outcomes against the round before
    it (LILLIPUT 2108.06569 lines 499-510), so a round is formed from
    the packets the seat holds, the last max_record_span + 1 it was
    given (detector_formation.StreamingDetectorFormer). A round the
    seat formed before is answered from what it remembers, and its
    packet is held for the round after it.
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

    def has_formed(self, round_key: tuple) -> bool:
        """Whether a source with no recipes, or this seat, formed the round.

        A source with no recipes forms nothing, so nothing is missing.
        """
        if self.recipes is None:
            return True
        return round_key in self.events_by_round

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
        events, _ = former.feed_packet(round_index, raw_bits)
        self._report_state(operation_id, former)
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
            former = detector_formation.StreamingDetectorFormer(table)
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
    ordered = []
    for group in fragments_by_round.values():
        in_measurement_order = sorted(group, key=_fragment_order)
        ordered.append(in_measurement_order)
    return ordered


def _fragment_order(fragment):
    return fragment.fragment_index


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the seated former reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (gem5 src/base/stats/group.hh:60-92); a component's events are the
    same shape.
    """

    state_held: trace_source.TraceSource = trace_source.new_source()
