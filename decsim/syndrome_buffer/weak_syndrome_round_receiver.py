"""The weak syndrome round receiver: room, and the slot a landing takes.

A round occupies a slot when its write completes and is readable then;
the link landing starts the write, and a zero write cost completes at
the landing. Every referent writes storage at the destination: ns-3's
channel schedules the destination's Receive
(src/point-to-point/model/point-to-point-channel.cc:88-92), OMNeT++
inserts inside the destination module's handler
(src/sim/csimplemodule.cc:782-799), Ciw counts the individual in the
destination's accept (ciw/node.py:602), and the control papers put the
store on the far side of the wire (Caune 2410.05202 lines 1243-1247;
Google 2408.13687 lines 471-477; Maurer 2510.21600 lines 593-597).

The sender must refuse before it sends, so this end answers for room
with the bits of its writes in flight counted as taken, gem5's reserved
bytes in `avail() = _maxsize - _size - _reserved`
(src/dev/net/pktfifo.hh) and Ruby's `current_size + current_stall_size +
n <= m_max_size` (src/mem/ruby/network/MessageBuffer.cc:181). The strong
store's end has the same shape.

Two kinds of round land here over controller_to_weak_buffer. A packed
round is stored as the run's detection event placement says
(detection_events.formed_at) and announced to the window manager. A
timing-only feedback-memory round takes its slot once written and leaves
by the store's outgoing port, which frees the slot at the delivery; no
window reads it, so it is never published.
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
_SEAT = "weak_syndrome_buffer"


class WeakSyndromeRoundReceiver:
    """The weak syndrome buffer's receiving end: room, in flight, landing.

    Trace sources: round_event(RoundEvent) with kind PUBLISHED, and
    copy_made(round_key, bits, "controller assembler", "weak syndrome
    buffer") at every intake, the write's copy.
    """

    store = ports.Port(ports.SyndromeBuffer)
    # the store's outgoing port, which sends what leaves the store
    output = ports.Port(ports.SyndromeBufferOutput)
    windows = ports.Port(ports.WindowInput)
    # the run's placement, which says what the store holds of a landed round
    detection_events = ports.Port(ports.DetectionEventPlacement)

    def __init__(self, engine: engine_module.Engine) -> None:
        self.engine = engine
        # the bits each crossing round will take, by its key, held
        # against the store until its write completes
        self.reserved_bits_by_round: dict = {}
        self.trace = _TraceSources()

    def has_room(self, packed: round_records.PackedRound) -> bool:
        """A write can land: the store weighs it against what is reserved."""
        bits = self._stored_width(packed)
        return self.store.has_room(
            packed.round_key, bits, self.reserved_bits_by_round
        )

    def reserve_write(self, packed: round_records.PackedRound) -> None:
        """Take the bits one crossing round will need, before it leaves."""
        assert self.has_room(packed), "a round was written into a full store"
        width = self._stored_width(packed)
        bits = round_records.stated_bits(width)
        self.reserved_bits_by_round[packed.round_key] = bits

    def receive_round(
        self,
        packed: round_records.PackedRound,
        on_published: Callable[[], None],
    ) -> None:
        """Form one landed round, then write it, retaining its reservation.

        The formation cycles come first when this seat forms the round,
        then the write; the window manager hears of the round only once
        it is formed and written. on_published runs after the
        announcement. A round of a closed operation never lands here:
        the window manager refuses a round its plan does not expect.
        """
        formation_cycles = self.detection_events.cycles_at(_SEAT, 1)
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
        """Write one landed timing-only round, then send it to the decoder.

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

    def check_settled(self) -> None:
        """At the end of a run no write is on the wire and no bit is stored.

        A round still stored is a leak or a store too small for its
        widest hold, and the store names which. Holds on rounds that
        never came are not asked here: a run that ends with nothing
        delivered is a legal end.
        """
        if self.reserved_bits_by_round:
            in_flight = len(self.reserved_bits_by_round)
            raise RuntimeError(
                f"weak syndrome buffer ended with {in_flight} "
                f"controller_to_weak_buffer writes in flight"
            )
        if self.store.occupied_bits:
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
        self._take_slot(packed, "controller_to_weak_buffer", self.engine.now)
        self._fire_published(packed)
        self.windows.accept_window_input(packed.packet)
        on_published()

    def _send_memory_round_on(
        self,
        packed: round_records.PackedRound,
        on_delivered: Callable[[], None],
    ) -> None:
        self._give_back_reservation(packed)
        self._take_slot(packed, "controller_to_weak_buffer", None)
        self.output.send_memory_round(packed, on_delivered)

    def _stored_width(self, packed: round_records.PackedRound) -> Optional[int]:
        """The width the store will hold the round at, as this seat forms it."""
        fragments = packed.packet.fragments
        return self.detection_events.width_at(_SEAT, fragments)

    def _give_back_reservation(self, packed: round_records.PackedRound) -> None:
        """The crossing is over: it gives back exactly what it reserved."""
        del self.reserved_bits_by_round[packed.round_key]

    def _as_stored(
        self, landed: round_records.PackedRound
    ) -> round_records.PackedRound:
        """The landed round with the fragments the store holds of it.

        wire_bits stays what crossed the hop: the intake copy reports
        the hop's bits, not the store's.
        """
        fragments = self.detection_events.form_at(
            _SEAT, landed.packet.fragments
        )
        packet = dataclasses.replace(landed.packet, fragments=fragments)
        return dataclasses.replace(landed, packet=packet)

    def _take_slot(
        self,
        packed: round_records.PackedRound,
        arrived_from: str,
        publication_tick: Optional[int],
    ) -> None:
        """Store the round, then narrate the copy and the intake."""
        self.store.accept_packed_round(
            packed.packet, publication_tick=publication_tick
        )
        self.trace.copy_made.fire(
            packed.round_key,
            packed.wire_bits,
            "controller assembler",
            log_sources.WEAK_BUFFER,
        )
        self.engine.log_io(
            log_sources.WEAK_BUFFER,
            lambda: self._received_text(packed.packet, arrived_from),
        )

    def _received_text(
        self, packet: round_records.SyndromeRoundPacket, arrived_from: str
    ) -> str:
        defects = packet.defects_text()
        holds = self.store.held_rounds_description()
        return (
            f"received round {packet.round_index} of op {packet.operation_id} "
            f"from {arrived_from}; {defects}; holds {holds}"
        )

    def _fire_published(self, packed: round_records.PackedRound) -> None:
        operation_id, round_index = packed.round_key
        event = round_records.RoundEvent.of(
            "PUBLISHED",
            self.engine.now,
            operation_id,
            round_index,
            packed.route,
        )
        self.trace.round_event.fire(event)


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the receiver reports, as one member (gem5's stats Group)."""

    copy_made: trace_source.TraceSource = trace_source.new_source()
    round_event: trace_source.TraceSource = trace_source.new_source()
