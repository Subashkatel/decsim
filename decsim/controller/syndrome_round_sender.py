"""The sender: a finished round into every store it must reach, or held.

Backpressure: a store's own end answers has_room for the round, counting
the bits it holds and the bits reserved for writes in flight, and the
room is reserved before the wire is used (gem5 src/dev/net/pktfifo.hh,
avail() and reserve(len)). A round that finds no room waits in
HeldRounds, upstream of the stores, and enters in completion order when
a slot frees: gem5's blocked port and retry (src/mem/cache/base.cc
setBlocked, clearBlocked) and Ciw's Type I blocking (ciw/node.py
block_individual). Nothing is reordered and nothing is dropped.
"""

import dataclasses
import functools
from collections.abc import Callable
from typing import Protocol

import decsim.controller.round_destinations as round_destinations
import decsim.controller.round_transmission as round_transmission
import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.rounds as round_records
import decsim.trace_source as trace_source


class WaitingRound(Protocol):
    """What a waiting line reads of a round."""

    round_key: tuple
    route: round_records.SyndromePacketRoute


class HeldRounds:
    """The waiting line in front of one bounded stage of the readout path.

    One waits in front of the stores and one in front of the packing stage
    (round_assembly.py). Neither has a size: the QPU keeps measuring while a
    round waits, so rounds pile up in the controller (Terhal 1302.3428,
    Quantum Machines 2412.00289), as a Ruby MessageBuffer without a size
    holds any number (gem5 src/mem/ruby/network/MessageBuffer.py). A finite
    line would have to stall the QPU or drop a round, and no source settles
    either for a QEC stream.

    The STALLED and RELEASED round events bound the wait, the back-pressure
    a full stage applies, which no stage's own time carries (gem5
    MessageBuffer.cc stall counters; ns-3 queue-disc.cc sojourn time).
    """

    def __init__(self, engine: engine_module.Engine) -> None:
        self.engine = engine
        # (held round, the admission it retries), in completion order
        self.waiting: list = []
        self.trace = _HeldRoundsTraceSources()

    def refuse(
        self,
        held: WaitingRound,
        admit: Callable[[WaitingRound], bool],
    ) -> bool:
        """Hold a round that found no room; False, so the caller reports it.

        A held round is recorded STALLED once.
        """
        stream_id, round_index = held.round_key
        if self._is_holding(held):
            return False
        self.waiting.append((held, admit))
        stalled = round_records.RoundEvent.of(
            "STALLED", self.engine.now, stream_id, round_index, held.route
        )
        self.trace.round_event.fire(stalled)
        return False

    def retry(self) -> None:
        """A slot freed: admit from the head, stop at the first refused."""
        while self.waiting:
            held, admit = self.waiting[0]
            admitted = admit(held)
            if not admitted:
                return
            self.waiting.pop(0)
            self._released(held)

    @property
    def count(self) -> int:
        """How many rounds wait."""
        return len(self.waiting)

    def waiting_round_keys(self) -> set:
        """The keys of the rounds that wait, the head's among them."""
        return {held.round_key for held, _admit in self.waiting}

    def _released(self, held: WaitingRound) -> None:
        """The round left the waiting line: its wait ends at this tick."""
        stream_id, round_index = held.round_key
        released = round_records.RoundEvent.of(
            "RELEASED",
            self.engine.now,
            stream_id,
            round_index,
            held.route,
        )
        self.trace.round_event.fire(released)

    def _is_holding(self, held: WaitingRound) -> bool:
        for waiting, _admit in self.waiting:
            if waiting is held:
                return True
        return False


class SyndromeRoundSender:
    """Sends a finished round to its stores, or holds it until all have room.

    A round goes to the store of the pool that decodes its stream, and a
    copy to each other entry store a window reads it from
    (round_destinations.py); what a pool escalated to needs goes up with
    the escalation, never as a second copy (Battistel 2303.00054: a cold
    first stage keeps rounds off the cryostat I/O). A store that takes
    its rounds written hears them through the transmitter; one that
    takes them landed is sent each round over its controller hop
    directly. The sender reserves room in every store first, as a gem5
    packet reserves before it leaves (pktfifo.hh reserve); the slot and
    the copy are the receiving end's, at the landing. A round counts in
    flight until its own store publishes it, which is when the windows
    hear of it; a copy rides beside it uncounted.
    """

    # the controller's own fabric: it executes a landed store's crossing
    link = ports.Port(ports.Link)
    # each round's stores; absent on a run with no decoder, whose rounds
    # go nowhere
    destinations = ports.Port(ports.RoundDestinations, optional=True)
    held_rounds = ports.Port(ports.HeldRounds)
    # the line in front of the packing stage, which a landed round's
    # place in the stage frees
    packing_line = ports.Port(ports.HeldRounds)
    transmitter = ports.Port(round_transmission.RoundTransmitter)

    def __init__(self, engine: engine_module.Engine) -> None:
        self.engine = engine
        # rounds sent to a landed store and not yet landed there; the
        # packing stage's bound reads it (RoundsInFlight), as it reads the
        # transmitter's in_flight
        self.crossing_count = 0

    def admit(self, packed: round_records.PackedRound) -> bool:
        """Write the round where it belongs; False when it waits.

        A round joins any held rounds without asking for room, even when it
        would fit: a refused gem5 requester waits for a retry before it sends
        again (src/mem/port.hh:244-255, packet_queue.cc), so a narrow round
        never overtakes a wide one.
        """
        if self.held_rounds.count:
            return self.held_rounds.refuse(packed, self._write)
        return self._write(packed)

    def check_settled(self) -> None:
        """At the end of a run no round may still wait for room."""
        if self.held_rounds.count:
            raise RuntimeError(
                f"run ended with {self.held_rounds.count} rounds held for "
                f"store room"
            )

    def _write(self, packed: round_records.PackedRound) -> bool:
        """Write the round if all its stores have room; False when one has none.

        A run with no decoder has no store, so its round is sent nowhere.
        """
        if self.destinations is None:
            return True
        sides = self.destinations.stores_for(packed)
        if not _have_room(sides, packed):
            return self.held_rounds.refuse(packed, self._write)
        self.destinations.note_written(packed)
        own, *copies = sides
        self._send(packed, own, False)
        for side in copies:
            self._send(packed, side, True)
        return True

    def _send(
        self,
        packed: round_records.PackedRound,
        side: round_destinations.StoreSide,
        is_copy: bool,
    ) -> None:
        """Reserve the round's room in one store, then send it there.

        The round takes its slot when its bits are there: the room is
        reserved here, and the landing stores and publishes it.
        """
        side.receiver.reserve_write(packed)
        if side.is_landed:
            self._write_landed(packed, side, is_copy)
            return
        self.transmitter.send(packed, side, is_copy)

    def _write_landed(
        self,
        packed: round_records.PackedRound,
        side: round_destinations.StoreSide,
        is_copy: bool,
    ) -> None:
        """Carry the round over its link to a store that takes it landed.

        The controller is the end the round leaves by, so it executes the send
        (OMNeT++ csimplemodule.cc:333-334; gem5 coherent_xbar.cc:354-357).
        """
        attribution = round_transmission.attribution_of(packed, side, is_copy)
        if not is_copy:
            self.crossing_count += 1
        landed = functools.partial(self._land, packed, side, is_copy)
        self.link.send(
            side.write_path,
            packed.wire_bits,
            self.engine.now,
            attribution,
            landed,
        )

    def _land(
        self,
        packed: round_records.PackedRound,
        side: round_destinations.StoreSide,
        is_copy: bool,
        _transfer,
    ) -> None:
        """Hand the round to the store's landing.

        The round stops counting in flight one event later, as a written
        round does, so a fragment landing at this tick still counts it.
        """
        side.receiver.receive_landed(packed)
        if is_copy:
            return
        self.engine.schedule(
            0, self._leave_crossing, label="strong round landed"
        )

    def _leave_crossing(self) -> None:
        """The round left the packing stage, so a waiting round may enter."""
        self.crossing_count -= 1
        self.packing_line.retry()


def _have_room(sides: tuple, packed: round_records.PackedRound) -> bool:
    """Whether every store the round goes to has room for it."""
    for side in sides:
        if not side.receiver.has_room(packed):
            return False
    return True


@dataclasses.dataclass(frozen=True)
class _HeldRoundsTraceSources:
    """Every event the held rounds reports, as one member."""

    round_event: trace_source.TraceSource = trace_source.new_source()
