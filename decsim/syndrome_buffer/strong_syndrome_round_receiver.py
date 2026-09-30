"""The strong syndrome buffer's receiving end: room, then landing.

Two hops land here. A strong-primary run's controller writes every
round over controller_to_strong_buffer, the one transport such a run
has (Caune 2410.05202 Fig. 1a stage F). A switching run's strong
redecode carries an escalated window's rounds over
weak_decoder_to_strong_decoder, read out of the weak syndrome buffer at
the switch (Toshio 2510.25222 lines 1247 to 1250: the syndrome data of
r_strong rounds is assigned to the strong decoder when the soft output
is small), so the rounds a cold weak tier keeps for itself never cross
the cryostat (Battistel 2303.00054 lines 342 to 347). Either sender
executes its own crossing, being the end the data leaves by (OMNeT++
refuses a module that sends a message it does not own,
omnetpp src/sim/csimplemodule.cc:333-334; gem5 bills a
transfer to the port it left by, coherent_xbar.cc:354-357). This end owns the
room and the landing: it answers has_room counting the bits still in
flight, reserves those bits before a crossing starts, gem5's packet
store counting its reserved bytes as taken (`avail() = _maxsize -
_size - _reserved` and `reserve(len)`, src/dev/net/pktfifo.hh), and
stores each round at its landing; the store's holds and lifetime are
SyndromeBuffer's. A landing whose operation closed while its bits
crossed is dropped at the door instead of stored, since no reader can
ever name it (_drop_landing): a strong-primary run's last rounds, or
the region of a speculative strong decode that a confident weak result
cancelled while the region was on the wire.

A landed round is stored as the run's detection event placement says
the store holds it, the landed outcomes or the events formed from them
here (detection_events.formed_at), after the formation's cycles when
this seat forms them. Controller packets retain their canonical route.
Window input wakes the window manager; timing-only idle input occupies a
slot until the store's own decoder hop delivers it, without creating a
window.
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


# _Hop comes before the public class because the two hops below are
# built when the module loads.
@dataclasses.dataclass(frozen=True)
class _Hop:
    """One of the two hops that land here: its row, and where its bits sat."""

    path_name: str
    source_name: str


CONTROLLER_WRITE = _Hop("controller_to_strong_buffer", "controller assembler")
# the seat this end is on the path, as detection_events.formed_at names it
_SEAT = "strong_syndrome_buffer"
ESCALATION = _Hop("weak_decoder_to_strong_decoder", "weak syndrome buffer")


class StrongSyndromeRoundReceiver:
    """The room, the writes in flight, and the landing into the store.

    Trace source: copy_made(round_key, bits, source, "strong syndrome
    buffer") at every landing, the crossing's copy (data_path.md hops 3
    and 5); the source is the controller assembler or the weak syndrome
    buffer, whichever the round left. round_event reports
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
                packets, packet_bits, CONTROLLER_WRITE, False, _nothing
            )
            return
        del self.reserved_bits_by_round[packed.round_key]
        self._forward_memory_round(packed)

    def reserve_region(self, region: round_records.EscalatedRegion) -> None:
        """Take the room an escalated region needs, before it leaves the chip.

        Each round is asked for beside the ones before it, so a store of
        several memories answers for each. An escalation cannot wait:
        the weak result is already given up, so a strong store with no
        room for the region stops the run rather than holding the chip.
        The yaml sizes the store.
        """
        reserved = dict(self.reserved_bits_by_round)
        widths = self._stored_widths(region)
        for packet, bits in zip(region.packets, widths, strict=True):
            round_key = (packet.operation_id, packet.round_index)
            if not self.store.has_room(round_key, bits, reserved):
                self._refuse_region(region, widths)
            reserved[round_key] = round_records.stated_bits(bits)
        self.reserved_bits_by_round = reserved

    def receive_region(
        self,
        region: round_records.EscalatedRegion,
        on_stored: Callable[[], None],
    ) -> None:
        """Take an escalated region that landed here: every round its slot.

        on_stored is called once the store holds every round, after this
        seat has formed them when it forms them.
        """
        packet_bits = []
        for packet in region.packets:
            bits = round_records.fragment_wire_bits(packet.fragments)
            packet_bits.append(bits)
        self._form_then_land(
            region.packets,
            packet_bits,
            ESCALATION,
            region.carries_the_round_before,
            on_stored,
        )

    def _stored_widths(self, region: round_records.EscalatedRegion) -> list:
        """Each round's width as the store will hold it.

        The round before lands raw, as it came (_land_formed); every other
        round at the width this seat forms it to.
        """
        packets = region.packets
        widths = []
        if region.carries_the_round_before:
            round_before = packets[0]
            raw_bits = round_records.fragment_wire_bits(round_before.fragments)
            widths.append(raw_bits)
            packets = packets[1:]
        for packet in packets:
            width = self._stored_width(packet)
            widths.append(width)
        return widths

    def _stored_width(
        self, packet: round_records.SyndromeRoundPacket
    ) -> Optional[int]:
        """The width the store will hold the round at, as this seat forms it."""
        return self.detection_events.width_at(_SEAT, packet.fragments)

    def _refuse_region(
        self, region: round_records.EscalatedRegion, widths: list
    ) -> None:
        """An escalated region that does not fit stops the run, by the yaml."""
        capacity = self.store.capacity_bits()
        round_count = len(region.packets)
        region_bits = sum(widths)
        reserved_widths = self.reserved_bits_by_round.values()
        reserved_bits = sum(reserved_widths)
        raise RuntimeError(
            f"the strong syndrome buffer has no room for the "
            f"{region_bits} bits of an escalated region's {round_count} "
            f"rounds: {self.store.occupied_bits} bits stored and "
            f"{reserved_bits} reserved against "
            f"strong_syndrome_buffer.bits {capacity}"
        )

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
        packets,
        packet_bits,
        hop: _Hop,
        carries_the_round_before: bool,
        on_stored: Callable[[], None],
    ) -> None:
        """Land the rounds once this seat has formed them, if it forms them.

        The rounds of one landing are formed together, a pipelined
        stage's fixed latency once and its rate for every round after
        the first (detection_events, detector_error_model/settings.py).
        A round carried as the round before is not formed.
        """
        round_count = len(packets) - int(carries_the_round_before)
        cycles = self.detection_events.cycles_at(_SEAT, round_count)
        land = functools.partial(
            self._land_formed,
            packets,
            packet_bits,
            hop,
            carries_the_round_before,
            on_stored,
        )
        if cycles == 0:
            land()
            return
        clock = self.detection_events.clock
        edge = clock.edge(cycles, self.engine.now)
        delay = edge - self.engine.now
        self.engine.schedule(delay, land, label="detection event formation")

    def _land_formed(
        self,
        packets,
        packet_bits,
        hop: _Hop,
        carries_the_round_before: bool,
        on_stored: Callable[[], None],
    ) -> None:
        """Each round as the store holds it; its copy reports the hop's bits.

        The round before lands raw, as it came, and this seat holds it
        for the first round's detectors when it forms them.
        """
        round_before = ()
        if carries_the_round_before:
            round_before = packets[0].fragments
            self._land(packets[0], packet_bits[0], hop)
            packets = packets[1:]
            packet_bits = packet_bits[1:]
        for packet, bits in zip(packets, packet_bits, strict=True):
            fragments = self.detection_events.form_at(
                _SEAT, packet.fragments, round_before
            )
            round_before = ()
            stored = dataclasses.replace(packet, fragments=fragments)
            self._land(stored, bits, hop)
        on_stored()

    def _land(
        self,
        packet: round_records.SyndromeRoundPacket,
        packet_bits: Optional[int],
        hop: _Hop,
    ) -> None:
        """Store the round, or drop it when its operation already closed.

        The round's reservation is given back here, not at the landing:
        the bits stay taken while this seat forms them, as gem5's packet
        store clears its reserve only in the push that fills the slot
        (src/dev/net/pktfifo.hh, `push`).
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
        hop: _Hop,
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
        hop: _Hop,
    ) -> None:
        """The operation closed while the round crossed: it has no reader.

        The same rule as the unheld landing above, one step later. A
        store closes an operation only when no hold and no stored round
        names it (syndrome_buffer.py close_operation), and a hold on a closed
        operation cannot be registered afterwards, so a round of a closed
        operation provably has no reader ever and must not enter the
        store. The copy still fires: the bits did cross the hop, and
        the link's traffic ledger charged the transfer, so a suppressed
        copy would leave the two accounts of the same hop disagreeing.
        """
        round_key = (packet.operation_id, packet.round_index)
        self.trace.copy_made.fire(
            round_key, packet_bits, hop.source_name, "strong syndrome buffer"
        )
        self.engine.log_io(
            log_sources.STRONG_BUFFER, lambda: self._dropped_text(packet, hop)
        )

    def _received_text(
        self, packet: round_records.SyndromeRoundPacket, hop: _Hop
    ) -> str:
        defects = packet.defects_text()
        holds = self.store.held_rounds_description()
        return (
            f"received round {packet.round_index} of op {packet.operation_id} "
            f"from {hop.path_name}; {defects}; holds {holds}"
        )

    def _dropped_text(
        self, packet: round_records.SyndromeRoundPacket, hop: _Hop
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
class _TraceSources:
    """Every event the strong syndrome round receiver reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (gem5 src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    copy_made: trace_source.TraceSource = trace_source.new_source()
    round_event: trace_source.TraceSource = trace_source.new_source()


def _nothing() -> None:
    """A controller write's rounds wake nothing once stored."""
