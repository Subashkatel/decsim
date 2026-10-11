"""A syndrome buffer's receiving end: room, then the slot a landing takes.

Every referent writes storage at the destination: ns-3's channel
schedules the destination's Receive
(src/point-to-point/model/point-to-point-channel.cc:88-92), OMNeT++
inserts inside the destination module's handler
(src/sim/csimplemodule.cc:782-799), Ciw counts the individual in the
destination's accept (ciw/node.py:602), and the control papers put the
store on the far side of the wire (Caune 2410.05202 lines 1243-1247;
Google 2408.13687 lines 471-477; Maurer 2510.21600 lines 593-597).

Each sender executes its own crossing; this end owns the room and the
landing. The sender must refuse before it sends, so this end answers for
room with the bits of its writes in flight counted as taken, gem5's
reserved bytes in `avail() = _maxsize - _size - _reserved`
(src/dev/net/pktfifo.hh) and Ruby's `current_size + current_stall_size +
n <= m_max_size` (src/mem/ruby/network/MessageBuffer.cc:181).

A store takes its rounds by one of two intakes, which its place in the
machine sets (INTAKES):

- written: the controller writes each round into the store its first
  pool reads. A round occupies a slot when its write completes and is
  readable then; this seat forms it first, a round at a time, and the
  window manager hears of it once written. A timing-only
  feedback-memory round takes its slot once written and leaves by the
  store's outgoing port.
- landed: rounds land whole. The controller's write of a run whose only
  pool is on the host (Caune 2410.05202 Fig. 1a stage F), and an
  escalated window's rounds, read out of the escalating pool's store at
  the switch (Toshio 2510.25222 lines 1247-1250), so the rounds a cold
  first pool keeps never cross the cryostat (Battistel 2303.00054 lines
  342-347). This seat forms each landing in a pipelined stage, and a
  landing whose operation closed while it crossed is dropped at the
  door, since no reader can ever name it.

Each round is stored as the run's detection event placement says
(detection_events.formed_at).
"""

import dataclasses
import functools
from collections.abc import Callable
from typing import Optional

import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.rounds as round_records
import decsim.trace_source as trace_source

# how a store takes its rounds, as the module docstring says
WRITTEN = "written"
LANDED = "landed"
INTAKES = (WRITTEN, LANDED)


@dataclasses.dataclass(frozen=True)
class Hop:
    """A hop that lands in a store: its row, and where its bits sat."""

    path_name: str
    source_name: str


@dataclasses.dataclass(frozen=True)
class StoreIntake:
    """How one store takes its rounds.

    name is the store's, the line it narrates on and the copy's target;
    seat is its place on the path, as detection_events.formed_at names
    it; shape is one of INTAKES. controller_write is the hop the
    controller writes over, and escalation the hop an escalated region
    lands over, None for a store no escalation reaches.
    """

    name: str
    seat: str
    shape: str
    controller_write: Hop
    escalation: Optional[Hop] = None


class SyndromeRoundReceiver:
    """A syndrome buffer's receiving end: room, in flight, landing.

    Trace sources: copy_made(round_key, bits, source, the store's name) at every
    intake, the write's copy; round_event(RoundEvent) with kind
    PUBLISHED at a written round's announcement, and
    FEEDBACK_MEMORY_DELIVERED at a landed idle round's decoder-side
    delivery.
    """

    store = ports.Port(ports.SyndromeBuffer)
    # the store's outgoing port, which sends what leaves the store
    output = ports.Port(ports.SyndromeBufferOutput)
    # a receiver built with no window side stores its rounds for a reader
    # that never asks
    windows = ports.Port(ports.WindowInput, optional=True)
    # the run's placement, which says what the store holds of a landed round
    detection_events = ports.Port(ports.DetectionEventPlacement)
    # the decoders' end of a landed idle round's route; a written idle
    # round's sender hears its delivery instead
    memory_arrivals = ports.Port(ports.MemoryRoundArrivals, optional=True)

    def __init__(
        self, engine: engine_module.Engine, intake: StoreIntake
    ) -> None:
        self.engine = engine
        self.intake = intake
        # the bits each crossing round will take, by its key, held
        # against the store until it is written or landed
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
            f"a round was written into a full {self.intake.name}"
        )
        width = self._stored_width(packed.packet)
        bits = round_records.stated_bits(width)
        self.reserved_bits_by_round[packed.round_key] = bits

    def receive_written(
        self,
        packed: round_records.PackedRound,
        on_published: Callable[[], None],
    ) -> None:
        """Form one written round, then write it, retaining its reservation.

        The formation cycles come first when this seat forms the round,
        then the write; the window manager hears of the round only once
        it is formed and written. on_published runs after the
        announcement. A round of a closed operation never lands here:
        the window manager refuses a round its plan does not expect.
        """
        formation_cycles = self.detection_events.cycles_at(self.intake.seat, 1)
        if formation_cycles == 0:
            self._write(packed, on_published)
            return
        clock = self.detection_events.clock
        edge = clock.edge(formation_cycles, self.engine.now)
        delay = edge - self.engine.now
        write = functools.partial(self._write, packed, on_published)
        self.engine.schedule(delay, write, label="detection event formation")

    def send_memory_round(
        self,
        packed: round_records.PackedRound,
        on_delivered: Callable[[], None],
    ) -> None:
        """Write one timing-only round, then send it to the decoder.

        It takes its slot once written, and the outgoing port frees the
        slot at the delivery. No window reads it, so it is never
        published.
        """
        stored_bits = round_records.fragment_wire_bits(packed.packet.fragments)
        written_tick = self.store.book_write(packed.round_key, stored_bits)
        send_on = functools.partial(
            self._send_memory_round_on, packed, on_delivered
        )
        if written_tick == self.engine.now:
            send_on()
            return
        delay = written_tick - self.engine.now
        self.engine.schedule(delay, send_on, label="syndrome buffer write")

    def receive_landed(self, packed: round_records.PackedRound) -> None:
        """Take one round that landed here; its room is kept until stored."""
        window_input = round_records.SyndromePacketRouteKind.WINDOW_INPUT
        if packed.route.kind is window_input:
            packets = (packed.packet,)
            packet_bits = (packed.wire_bits,)
            self._form_then_land(
                (), packets, packet_bits, self.intake.controller_write, _nothing
            )
            return
        del self.reserved_bits_by_round[packed.round_key]
        self._forward_memory_round(packed)

    def reserve_region(self, region: round_records.EscalatedRegion) -> None:
        """Take the room an escalated region needs, before it leaves.

        An escalation cannot wait for room, since the escalating result
        is already given up, so the store's settings must size it.
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
            self.intake.escalation,
            on_stored,
        )

    def check_settled(self) -> None:
        """At the end of a run no write is on the wire and no bit is stored.

        A round still stored is a leak or a store too small for its
        widest hold, and the store names which. A written store's holds
        on rounds that never came are not asked: a run that ends with
        nothing delivered is a legal end.
        """
        if self.reserved_bits_by_round:
            in_flight = len(self.reserved_bits_by_round)
            raise RuntimeError(
                f"{self.intake.name} ended with {in_flight} writes in flight"
            )
        if self.intake.shape == LANDED or self.store.occupied_bits:
            self.store.check_settled()

    def _write(
        self,
        landed: round_records.PackedRound,
        on_published: Callable[[], None],
    ) -> None:
        """Ask the store when the formed round's write completes."""
        packed = self._as_stored(landed)
        stored_bits = round_records.fragment_wire_bits(packed.packet.fragments)
        written_tick = self.store.book_write(packed.round_key, stored_bits)
        if written_tick == self.engine.now:
            self._finish_write(packed, on_published)
            return
        delay = written_tick - self.engine.now
        finish = functools.partial(self._finish_write, packed, on_published)
        self.engine.schedule(delay, finish, label="syndrome buffer write")

    def _finish_write(
        self,
        packed: round_records.PackedRound,
        on_published: Callable[[], None],
    ) -> None:
        self._give_back_reservation(packed)
        self._take_slot(packed, self.engine.now)
        self._fire_published(packed)
        self.windows.accept_window_input(packed.packet)
        on_published()

    def _send_memory_round_on(
        self,
        packed: round_records.PackedRound,
        on_delivered: Callable[[], None],
    ) -> None:
        self._give_back_reservation(packed)
        self._take_slot(packed, None)
        self.output.send_memory_round(packed, on_delivered)

    def _forward_memory_round(self, packed: round_records.PackedRound) -> None:
        """A landed timing-only round holds a slot until its hop delivers."""
        self.store.accept_packed_round(packed.packet, publication_tick=None)
        self._narrate_intake(
            packed.round_key,
            packed.wire_bits,
            packed.packet,
            self.intake.controller_write,
        )
        delivered = functools.partial(self._deliver_memory_round, packed)
        self.output.send_memory_round(packed, delivered)

    def _deliver_memory_round(self, packed: round_records.PackedRound) -> None:
        source_operation_id = packed.route.source_operation_id
        self.memory_arrivals.receive_memory_round(source_operation_id)
        self._fire_event("FEEDBACK_MEMORY_DELIVERED", packed)

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
        return self.detection_events.width_at(
            self.intake.seat, packet.fragments
        )

    def _give_back_reservation(self, packed: round_records.PackedRound) -> None:
        """The crossing is over: it gives back exactly what it reserved."""
        del self.reserved_bits_by_round[packed.round_key]

    def _as_stored(
        self, landed: round_records.PackedRound
    ) -> round_records.PackedRound:
        """The written round with the fragments the store holds of it.

        wire_bits stays what crossed the hop: the intake copy reports
        the hop's bits, not the store's.
        """
        fragments = self.detection_events.form_at(
            self.intake.seat, landed.packet.fragments
        )
        packet = dataclasses.replace(landed.packet, fragments=fragments)
        return dataclasses.replace(landed, packet=packet)

    def _take_slot(
        self,
        packed: round_records.PackedRound,
        publication_tick: Optional[int],
    ) -> None:
        """Store the written round, then narrate the copy and the intake."""
        self.store.accept_packed_round(
            packed.packet, publication_tick=publication_tick
        )
        self._narrate_intake(
            packed.round_key,
            packed.wire_bits,
            packed.packet,
            self.intake.controller_write,
        )

    def _narrate_intake(
        self,
        round_key: tuple,
        bits: Optional[int],
        packet: round_records.SyndromeRoundPacket,
        hop: Hop,
    ) -> None:
        """The copy the intake made, and its line in the I/O trace."""
        self.trace.copy_made.fire(
            round_key, bits, hop.source_name, self.intake.name
        )
        self.engine.log_io(
            self.intake.name, lambda: self._received_text(packet, hop)
        )

    def _form_then_land(
        self,
        rounds_before: tuple,
        packets: tuple,
        packet_bits,
        hop: Hop,
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
        hop: Hop,
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
                self.intake.seat, packet.fragments, held_fragments
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
        cycles = self.detection_events.cycles_at(self.intake.seat, round_count)
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
        one_more = self.detection_events.cycles_at(self.intake.seat, next_count)
        first = self.detection_events.cycles_at(self.intake.seat, 1)
        return one_more - first

    def _land(
        self,
        packet: round_records.SyndromeRoundPacket,
        packet_bits: Optional[int],
        hop: Hop,
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
        hop: Hop,
    ) -> None:
        self.store.accept_packed_round(packet, publication_tick=self.engine.now)
        round_key = (packet.operation_id, packet.round_index)
        self._narrate_intake(round_key, packet_bits, packet, hop)
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
        hop: Hop,
    ) -> None:
        """The operation closed while the round crossed: it has no reader.

        A store closes an operation only when nothing names it, and no
        hold on a closed operation is registered afterwards, so the
        round must not enter the store. The copy still fires: the bits
        crossed the hop and the link charged them.
        """
        round_key = (packet.operation_id, packet.round_index)
        self.trace.copy_made.fire(
            round_key, packet_bits, hop.source_name, self.intake.name
        )
        self.engine.log_io(
            self.intake.name, lambda: self._dropped_text(packet, hop)
        )

    def _received_text(
        self, packet: round_records.SyndromeRoundPacket, hop: Hop
    ) -> str:
        defects = packet.defects_text()
        holds = self.store.held_rounds_description()
        return (
            f"received round {packet.round_index} of op {packet.operation_id} "
            f"from {hop.path_name}; {defects}; holds {holds}"
        )

    def _dropped_text(
        self, packet: round_records.SyndromeRoundPacket, hop: Hop
    ) -> str:
        holds = self.store.held_rounds_description()
        return (
            f"dropped round {packet.round_index} of op {packet.operation_id} "
            f"from {hop.path_name}: op {packet.operation_id} "
            f"closed while the round crossed; holds {holds}"
        )

    def _fire_published(self, packed: round_records.PackedRound) -> None:
        self._fire_event("PUBLISHED", packed)

    def _fire_event(self, kind: str, packed: round_records.PackedRound) -> None:
        operation_id, round_index = packed.round_key
        event = round_records.RoundEvent.of(
            kind, self.engine.now, operation_id, round_index, packed.route
        )
        self.trace.round_event.fire(event)


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the receiver reports, as one member (gem5's stats Group)."""

    copy_made: trace_source.TraceSource = trace_source.new_source()
    round_event: trace_source.TraceSource = trace_source.new_source()


def _nothing() -> None:
    """A controller write's landed rounds wake nothing once stored."""
