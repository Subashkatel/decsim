"""The strong syndrome buffer's receiving end: room, then landing.

Two hops land here. A strong-primary run's controller writes every round
over controller_to_strong_buffer (Caune 2410.05202 Fig. 1a stage F). A
switching run's strong redecode carries an escalated window's rounds
over weak_decoder_to_strong_decoder, read out of the weak store at the
switch (Toshio 2510.25222 lines 1247-1250), so the rounds a cold weak
tier keeps never cross the cryostat (Battistel 2303.00054 lines
342-347). Each sender executes its own crossing; this end owns the room
and the landing. It reserves a round's bits before the crossing starts,
as gem5's packet store counts reserved bytes as taken
(src/dev/net/pktfifo.hh), and stores each round as the run's detection
event placement says (detection_events.formed_at). A landing whose
operation closed while it crossed is dropped at the door, since no
reader can ever name it.
"""

import dataclasses
import functools
from collections.abc import Callable
from typing import Optional

import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.log_sources as log_sources
import decsim.records.rounds as round_records
import decsim.trace_source as trace_source

# the seat this end is on the path, as detection_events.formed_at names it
_SEAT = "strong_syndrome_buffer"


class StrongSyndromeRoundReceiver:
    """The strong syndrome buffer's receiving end.

    Trace sources: copy_made(round_key, bits, source, "strong syndrome
    buffer") at every landing, the source being the controller assembler
    or the weak syndrome buffer; round_event reports
    FEEDBACK_MEMORY_DELIVERED at the idle round's decoder-side delivery.
    """

    # a receiver built with no window side stores its rounds for a reader
    # that never asks
    windows = ports.Port(ports.WindowInput, optional=True)
    store = ports.Port(ports.SyndromeBuffer)
    output = ports.Port(ports.SyndromeBufferOutput)
    memory_arrivals = ports.Port(ports.MemoryRoundArrivals)
    # the run's placement, which says what the store holds of a landed round
    detection_events = ports.Port(ports.DetectionEventPlacement)

    def __init__(self, engine: engine_module.Engine) -> None:
        self.engine = engine
        # the bits each crossing round will take, by its key, held
        # against the store until it lands
        self.reserved_bits_by_round: dict = {}
        # the tick this seat's former can take the next landing's first
        # round, so landings form in the order they land
        self.former_free_tick = 0
        self.trace = _TraceSources()

    def has_room(self, packed: round_records.PackedRound) -> bool:
        """A write can land: the store weighs it against what is reserved."""
        bits = self._stored_width(packed.packet)
        return self.store.has_room(
            packed.round_key, bits, self.reserved_bits_by_round
        )

    def reserve_write(self, packed: round_records.PackedRound) -> None:
        """Take the bits one crossing round will need, before it leaves."""
        assert self.has_room(packed), (
            "a round was written into a full strong store"
        )
        width = self._stored_width(packed.packet)
        bits = round_records.stated_bits(width)
        self.reserved_bits_by_round[packed.round_key] = bits

    def receive_round(self, packed: round_records.PackedRound) -> None:
        """Take one round that landed here; its room is kept until stored."""
        window_input = round_records.SyndromePacketRouteKind.WINDOW_INPUT
        if packed.route.kind is window_input:
            packets = (packed.packet,)
            packet_bits = (packed.wire_bits,)
            self._form_then_land(
                (), packets, packet_bits, CONTROLLER_WRITE, _nothing
            )
            return
        del self.reserved_bits_by_round[packed.round_key]
        self._forward_memory_round(packed)

    def reserve_region(self, region: round_records.EscalatedRegion) -> None:
        """Take the room an escalated region needs, before it leaves the chip.

        An escalation cannot wait for room, since the weak result is
        already given up, so the store's settings must size it.
        """
        widths = self._stored_widths(region)
        carried = region.carried_packets
        for packet, bits in zip(carried, widths, strict=True):
            round_key = (packet.operation_id, packet.round_index)
            stated = round_records.stated_bits(bits)
            self.reserved_bits_by_round[round_key] = stated

    def receive_region(
        self,
        region: round_records.EscalatedRegion,
        on_stored: Callable[[], None],
    ) -> None:
        """Take an escalated region that landed here: every round its slot.

        on_stored runs once the store holds every round.
        """
        packet_bits = []
        for packet in region.carried_packets:
            bits = round_records.fragment_wire_bits(packet.fragments)
            packet_bits.append(bits)
        self._form_then_land(
            region.rounds_before,
            region.packets,
            packet_bits,
            ESCALATION,
            on_stored,
        )

    def _stored_widths(self, region: round_records.EscalatedRegion) -> list:
        """Each carried round's width as the store will hold it.

        The rounds before land raw and take their raw bits; every other
        round takes the width this seat forms it to.
        """
        widths = []
        for raw_round in region.rounds_before:
            raw_bits = round_records.fragment_wire_bits(raw_round.fragments)
            widths.append(raw_bits)
        for packet in region.packets:
            width = self._stored_width(packet)
            widths.append(width)
        return widths

    def _stored_width(
        self, packet: round_records.SyndromeRoundPacket
    ) -> Optional[int]:
        """The width the store will hold the round at, as this seat forms it."""
        return self.detection_events.width_at(_SEAT, packet.fragments)

    def _forward_memory_round(self, packed: round_records.PackedRound) -> None:
        """A timing-only round holds a slot until its decoder hop delivers."""
        self.store.accept_packed_round(packed.packet, publication_tick=None)
        self.trace.copy_made.fire(
            packed.round_key,
            packed.wire_bits,
            CONTROLLER_WRITE.source_name,
            "strong syndrome buffer",
        )
        self.engine.log_io(
            log_sources.STRONG_BUFFER,
            lambda: self._received_text(packed.packet, CONTROLLER_WRITE),
        )
        delivered = functools.partial(self._deliver_memory_round, packed)
        self.output.send_memory_round(packed, delivered)

    def _deliver_memory_round(self, packed: round_records.PackedRound) -> None:
        source_operation_id = packed.route.source_operation_id
        self.memory_arrivals.receive_memory_round(source_operation_id)
        operation_id, round_index = packed.round_key
        event = round_records.RoundEvent.of(
            "FEEDBACK_MEMORY_DELIVERED",
            self.engine.now,
            operation_id,
            round_index,
            packed.route,
        )
        self.trace.round_event.fire(event)

    def _form_then_land(
        self,
        rounds_before: tuple,
        packets: tuple,
        packet_bits,
        hop: "_Hop",
        on_stored: Callable[[], None],
    ) -> None:
        """Land the rounds once this seat has formed them, if it forms them.

        packet_bits are the carried rounds' bits, the rounds before
        first.
        """
        land = functools.partial(
            self._land_formed,
            rounds_before,
            packets,
            packet_bits,
            hop,
            on_stored,
        )
        formed_tick = self._formed_tick(len(packets))
        delay = formed_tick - self.engine.now
        if delay == 0:
            land()
            return
        self.engine.schedule(delay, land, label="detection event formation")

    def _land_formed(
        self,
        rounds_before: tuple,
        packets: tuple,
        packet_bits,
        hop: "_Hop",
        on_stored: Callable[[], None],
    ) -> None:
        """Each round as the store holds it; its copy reports the hop's bits.

        The rounds before land raw, never formed, and this seat holds
        them for the first round's detectors.
        """
        raw_count = len(rounds_before)
        raw_bits = packet_bits[:raw_count]
        formed_bits = packet_bits[raw_count:]
        held = []
        for raw_round, bits in zip(rounds_before, raw_bits, strict=True):
            self._land(raw_round, bits, hop)
            held.extend(raw_round.fragments)
        held_fragments = tuple(held)
        for packet, bits in zip(packets, formed_bits, strict=True):
            fragments = self.detection_events.form_at(
                _SEAT, packet.fragments, held_fragments
            )
            held_fragments = ()
            stored = dataclasses.replace(packet, fragments=fragments)
            self._land(stored, bits, hop)
        on_stored()

    def _formed_tick(self, round_count: int) -> int:
        """When this seat has formed a landing of round_count rounds.

        The former is one pipelined stage: a fixed latency, then a round
        every cycles_per_round. A landing enters once the one before it
        has entered all its rounds, as gem5's in-order functional unit
        takes the next instruction issueLat cycles after the last
        (src/cpu/minor/func_unit.cc:157-170) and returns results in
        order (src/cpu/minor/buffers.hh:293). A later region's first
        round reads the earlier region's last, so it must not form
        first.
        """
        cycles = self.detection_events.cycles_at(_SEAT, round_count)
        issue_cycles = self._issue_cycles(round_count)
        clock = self.detection_events.clock
        entry_tick = max(self.engine.now, self.former_free_tick)
        if issue_cycles > 0:
            self.former_free_tick = clock.edge(issue_cycles, entry_tick)
        if cycles == 0:
            return entry_tick
        return clock.edge(cycles, entry_tick)

    def _issue_cycles(self, round_count: int) -> int:
        """The cycles the landing's rounds take to enter, one per rate."""
        next_count = round_count + 1
        one_more = self.detection_events.cycles_at(_SEAT, next_count)
        first = self.detection_events.cycles_at(_SEAT, 1)
        return one_more - first

    def _land(
        self,
        packet: round_records.SyndromeRoundPacket,
        packet_bits: Optional[int],
        hop: "_Hop",
    ) -> None:
        """Store the round, or drop it when its operation already closed.

        The reservation is given back here, not at the landing: the bits
        stay taken while this seat forms them, as gem5's packet store
        clears its reserve only in the push that fills the slot
        (src/dev/net/pktfifo.hh).
        """
        round_key = (packet.operation_id, packet.round_index)
        del self.reserved_bits_by_round[round_key]
        if self.store.has_operation(packet.operation_id):
            self._store_landing(packet, packet_bits, hop)
            return
        self._drop_landing(packet, packet_bits, hop)

    def _store_landing(
        self,
        packet: round_records.SyndromeRoundPacket,
        packet_bits: Optional[int],
        hop: "_Hop",
    ) -> None:
        self.store.accept_packed_round(packet, publication_tick=self.engine.now)
        round_key = (packet.operation_id, packet.round_index)
        self.trace.copy_made.fire(
            round_key, packet_bits, hop.source_name, "strong syndrome buffer"
        )
        self.engine.log_io(
            log_sources.STRONG_BUFFER,
            lambda: self._received_text(packet, hop),
        )
        if self.windows is not None:
            self.windows.accept_room_round(
                packet.operation_id, packet.round_index
            )
        # Arrival can create a live window's first hold. Publish before
        # reclamation, as gem5 services targets before freeing an entry
        # (src/mem/cache/base.cc:637-656).
        self.store.release_round_if_unheld(round_key)

    def _drop_landing(
        self,
        packet: round_records.SyndromeRoundPacket,
        packet_bits: Optional[int],
        hop: "_Hop",
    ) -> None:
        """The operation closed while the round crossed: it has no reader.

        A store closes an operation only when nothing names it, and no
        hold on a closed operation is registered afterwards, so the
        round must not enter the store. The copy still fires: the bits
        crossed the hop and the link charged them.
        """
        round_key = (packet.operation_id, packet.round_index)
        self.trace.copy_made.fire(
            round_key, packet_bits, hop.source_name, "strong syndrome buffer"
        )
        self.engine.log_io(
            log_sources.STRONG_BUFFER, lambda: self._dropped_text(packet, hop)
        )

    def _received_text(
        self, packet: round_records.SyndromeRoundPacket, hop: "_Hop"
    ) -> str:
        defects = packet.defects_text()
        holds = self.store.held_rounds_description()
        return (
            f"received round {packet.round_index} of op {packet.operation_id} "
            f"from {hop.path_name}; {defects}; holds {holds}"
        )

    def _dropped_text(
        self, packet: round_records.SyndromeRoundPacket, hop: "_Hop"
    ) -> str:
        holds = self.store.held_rounds_description()
        return (
            f"dropped round {packet.round_index} of op {packet.operation_id} "
            f"from {hop.path_name}: op {packet.operation_id} "
            f"closed while the round crossed; holds {holds}"
        )

    def check_settled(self) -> None:
        """At the end of a run nothing may be in flight or held."""
        if self.reserved_bits_by_round:
            in_flight = len(self.reserved_bits_by_round)
            raise RuntimeError(
                f"strong syndrome buffer ended with {in_flight} "
                f"writes in flight"
            )
        self.store.check_settled()


@dataclasses.dataclass(frozen=True)
class _Hop:
    """One of the two hops that land here.

    path_name is its row; source_name is where its bits sat.
    """

    path_name: str
    source_name: str


CONTROLLER_WRITE = _Hop("controller_to_strong_buffer", "controller assembler")
ESCALATION = _Hop("weak_decoder_to_strong_decoder", "weak syndrome buffer")


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the receiver reports, as one member (gem5's stats Group)."""

    copy_made: trace_source.TraceSource = trace_source.new_source()
    round_event: trace_source.TraceSource = trace_source.new_source()


def _nothing() -> None:
    """A controller write's rounds wake nothing once stored."""
