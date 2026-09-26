"""Where a round's measurement outcomes become its detection events.

The placement is a published choice with three rows, and
controller.detection_events_formed_at names one of them
(controller/settings.py DETECTION_EVENT_FORMATION, ports.py
DetectionEventPlacement). At the controller, "inside the workstation,
measurements are converted into detections and then streamed to the
real-time decoding software via a shared memory buffer" (Google
2408.13687 lines 474-476), so the round leaves the controller at its
detection-event width. At the decoder, the controller writes the
outcomes "sequentially to the decoder" and "The decoder computes the
syndrome from measurement outcomes and decodes it" (Caune et al.
2410.05202 lines 1252-1256), so the store and the tier's input link
carry the wider raw round and each tier forms what it reads. At the weak
syndrome buffer, the round crosses to the weak decoder chip raw and is
formed once as it is stored, so the store, the weak unit's input link
and the escalation hop carry the events. That the decoder's own chip
forms them is published (Maurer 2510.21600 lines 235-237, the decoder
FPGA's "detector window processing component calculates parities of
groups of measurements"); that it does so ahead of its store is this
row's own choice, the papers being silent on the order.

A detection event is a parity of raw measurement outcomes
(detector_formation.py), so its value is the same wherever it is formed:
a row moves the width the round carries and the clock the conversion is
charged on, and nothing else.
"""

import dataclasses
from collections.abc import Sequence
from typing import Any, Optional

import decsim.ports as ports
import decsim.records.rounds as round_records


class ControllerSideFormation:
    """The controller's assembler forms the round before it leaves.

    Google 2408.13687 lines 474-476: the workstation converts the
    measurements into detections and streams those to the decoding
    software, so the round crosses the store and the tier's input link
    at its detection-event width. detection_event_formation_cycles is what that
    conversion costs the controller, charged once per round on the
    controller's clock before the round leaves it.
    """

    forms_at_the_weak_syndrome_buffer = False
    forms_at_the_decoder = False

    def __init__(
        self,
        former: Optional[ports.DetectionEventFormer],
        detection_event_formation_cycles: int,
    ) -> None:
        self.former = former
        self.detection_event_formation_cycles = detection_event_formation_cycles

    def form_before_departure(self, fragments: tuple) -> tuple:
        """The round's fragments as they leave the controller."""
        formed = form_round_fragments(self.former, fragments)
        return _at_their_event_width(formed)

    def form_before_storage(self, fragments: tuple) -> tuple:
        """The round as the weak syndrome buffer stores it: as it landed."""
        return fragments

    def decoder_side_former(self) -> Optional[ports.DetectionEventFormer]:
        """None: the round reaches every tier already formed."""
        return None


class DecoderSideFormation:
    """Each tier forms the rounds it reads.

    Caune et al. 2410.05202 lines 1252-1256: the controller writes the
    outcomes sequentially to the decoder and the decoder computes the
    syndrome from them; LILLIPUT's Event Detection Logic block sits
    inside the decoder (2108.06569 lines 499-510). The round therefore
    leaves the controller raw and crosses the store and the tier's input
    link at its raw measurement width. The controller charges nothing
    for a conversion it does not do, so a controller-side charge under
    this row is refused where the yaml names it; the tier's own charge
    is <tier>_decoder.engine.detection_event_latency_cycles and its
    detection_event_cycles_per_round.

    The two tiers read the same rounds out of two stores and hold the
    same value of every event, so one former forms each round once, in
    round order, and both tiers read it (RememberedDetectionEvents);
    what each tier is charged is its own (decoders/detection_events.py).
    The logic sits in front of each tier's core whatever the source
    sends, so a tier is charged for it under this row even when the
    source carries no outcomes to convert, as the other two rows charge
    theirs.
    """

    forms_at_the_weak_syndrome_buffer = False
    forms_at_the_decoder = True

    def __init__(
        self,
        former: Optional[ports.DetectionEventFormer],
        detection_event_formation_cycles: int,
    ) -> None:
        if detection_event_formation_cycles > 0:
            raise ValueError(
                "controller.detection_event_cycles_per_round charges the "
                "controller for a conversion this run does at the decoder "
                "(controller.detection_events_formed_at); charge the tier "
                "with <tier>_decoder.engine."
                "detection_event_latency_cycles, or write null"
            )
        self.former = _remembered(former)
        self.detection_event_formation_cycles = 0

    def form_before_departure(self, fragments: tuple) -> tuple:
        """The round's fragments as they leave the controller: raw."""
        return fragments

    def form_before_storage(self, fragments: tuple) -> tuple:
        """The round as the weak syndrome buffer stores it: raw."""
        return fragments

    def decoder_side_former(self) -> Optional[ports.DetectionEventFormer]:
        """The former both tiers form through; None when the source has none."""
        return self.former


class WeakSyndromeBufferSideFormation:
    """The weak decoder chip forms the round once, as it stores it.

    Maurer et al. 2510.21600 lines 504-507 and 235-237: the readouts
    reach the decoder FPGA over the serial connection, and its detector
    window processing component calculates the parities in the input
    path. The paper does not say which side of its syndrome FIFO that
    component sits on; this row puts it ahead of the store, so a round
    is formed once however many windows read it. The round leaves the
    controller raw and crosses controller_to_weak_buffer at its raw
    measurement width; the weak syndrome buffer holds the events, so
    every hop after it carries the narrower round: the weak unit's
    input link, and the escalation hop that carries a region to the
    strong side. No tier forms anything.
    The chip's charge is weak_syndrome_buffer.
    detection_event_cycles_per_round, on that buffer's clock; the
    controller charges nothing for a conversion it does not do, so a
    controller-side charge under this row is refused where the yaml
    names it.
    """

    forms_at_the_weak_syndrome_buffer = True
    forms_at_the_decoder = False

    def __init__(
        self,
        former: Optional[ports.DetectionEventFormer],
        detection_event_formation_cycles: int,
    ) -> None:
        if detection_event_formation_cycles > 0:
            raise ValueError(
                "controller.detection_event_cycles_per_round charges the "
                "controller for a conversion this run does at the weak "
                "syndrome buffer (controller.detection_events_formed_at); "
                "charge the chip with weak_syndrome_buffer."
                "detection_event_cycles_per_round, or write null"
            )
        self.former = former
        self.detection_event_formation_cycles = 0

    def form_before_departure(self, fragments: tuple) -> tuple:
        """The round's fragments as they leave the controller: raw."""
        return fragments

    def form_before_storage(self, fragments: tuple) -> tuple:
        """The round as the weak syndrome buffer stores it: its events."""
        formed = form_round_fragments(self.former, fragments)
        return _at_their_event_width(formed)

    def decoder_side_former(self) -> Optional[ports.DetectionEventFormer]:
        """None: the round reaches every tier already formed."""
        return None


class RememberedDetectionEvents:
    """A former that forms each round once and answers every later ask.

    The formation table underneath is stateful: it keeps the packets its
    recipes still read and forms a round from this round's outcomes and
    the round before it (detector_formation.StreamingDetectorFormer), so
    it is fed once per round, in round order. Both decoder tiers ask for
    the same rounds, and an event is a parity of the raw outcomes alone,
    so the value is formed once here and each tier is charged for its own
    event-detection logic. A round asked for out of order is refused
    rather than formed from packets the table no longer holds.
    """

    def __init__(self, former: ports.DetectionEventFormer) -> None:
        self.former = former
        self.events_by_round: dict = {}
        self.newest_round_by_operation: dict = {}

    def form_round(
        self, operation_id: Any, round_index: int, raw_bits: Sequence[int]
    ) -> tuple:
        """The round's detection events, formed here the first time."""
        key = (operation_id, round_index)
        remembered = self.events_by_round.get(key)
        if remembered is not None:
            return remembered
        self._check_in_round_order(operation_id, round_index)
        events = self.former.form_round(operation_id, round_index, raw_bits)
        events = tuple(events)
        self.events_by_round[key] = events
        self.newest_round_by_operation[operation_id] = round_index
        return events

    def _check_in_round_order(
        self, operation_id: Any, round_index: int
    ) -> None:
        """The rounds of one operation are formed in order, or the run stops.

        A detector compares this round's outcomes against the round
        before it (LILLIPUT 2108.06569 lines 499-510), so a former asked
        for a round whose predecessor it never saw would read a packet
        the table has forgotten. Refusing is the only honest answer: the
        prefix it still holds is not this round's syndrome, and a
        silently wrong syndrome is a wrong logical answer.
        """
        newest = self.newest_round_by_operation.get(operation_id, 0)
        expected = newest + 1
        if round_index == expected:
            return
        raise RuntimeError(
            f"detection event formation for operation {operation_id!r} is "
            f"at round {newest} and was asked for round {round_index}; a "
            "detector compares a round against the one before it "
            "(LILLIPUT 2108.06569 lines 499-510), so the rounds of an "
            "operation are formed in order. A windowing shape whose "
            "windows read disjoint round ranges asks out of order: form "
            "its events at the controller "
            "(controller.detection_events_formed_at)"
        )


def sized_by_its_bits(
    fragment: round_records.RetainedSyndromeFragment,
) -> round_records.RetainedSyndromeFragment:
    """One fragment sized by the bits it holds, its events included."""
    if fragment.bits is None:
        return fragment
    return dataclasses.replace(fragment, size_bits=len(fragment.bits))


def form_round_fragments(
    former: Optional[ports.DetectionEventFormer],
    fragments: tuple[round_records.RetainedSyndromeFragment, ...],
) -> tuple[round_records.RetainedSyndromeFragment, ...]:
    """Form one whole round in measurement order, retaining group attribution.

    Stim rec targets span acquisition fragments. The parity conversion must
    consume the complete round once, irrespective of acquisition ownership.
    The result is one event vector shared by all contributing patches,
    sized as the fragments landed: a row that sends the events on
    narrows them to their own width afterwards, and a tier that forms
    them after they land keeps the landed size, since its input memory
    was written the raw round and a memory counts what is written into
    it: "The decoder computes the syndrome from measurement outcomes"
    (Caune et al. 2410.05202 lines 1252-1256). The fragments as they
    landed when there is nothing to form them with or from: no former,
    or a timing-only round that carries no bits.
    """
    if former is None:
        return fragments
    has_bits = all(fragment.bits is not None for fragment in fragments)
    if not has_bits:
        return fragments
    ordered = sorted(fragments, key=_fragment_order)
    raw = ordered[0]
    bits = []
    for fragment in ordered:
        bits.extend(fragment.bits)
    measurements = tuple(bits)
    events = former.form_round(raw.operation_id, raw.round_index, measurements)
    patches = round_records.fragment_patch_ids(ordered)
    landed_bits = round_records.fragment_wire_bits(ordered)
    formed = dataclasses.replace(
        raw, bits=tuple(events), size_bits=landed_bits, patch_ids=patches
    )
    return (formed,)


def _fragment_order(fragment):
    return fragment.fragment_index


def _remembered(
    former: Optional[ports.DetectionEventFormer],
) -> Optional[RememberedDetectionEvents]:
    """The former both tiers share; None when no source forms events."""
    if former is None:
        return None
    return RememberedDetectionEvents(former)


def _at_their_event_width(fragments: tuple) -> tuple:
    """The fragments sized by the events they carry."""
    narrowed = []
    for fragment in fragments:
        sized = sized_by_its_bits(fragment)
        narrowed.append(sized)
    return tuple(narrowed)
