"""The sender: a finished round into every store it must reach, or held.

The backpressure law of the readout path: each store's own end answers
has_room for the round before any round leaves for it, counting
the bits it holds and the bits reserved for the writes in flight, and
the room is reserved before the wire is used (gem5's packet store
answers `avail() = _maxsize - _size - _reserved` against the packet's
own length and reserves it with `reserve(len)`,
src/dev/net/pktfifo.hh). A finished
round that finds no room in either store waits in HeldRounds, upstream
of the stores, and enters when a slot frees, in the order the rounds
were completed. gem5's blocked port keeps the request at the requester
and re-sends it when clearBlocked schedules the retry
(src/mem/cache/base.cc:255-257 setBlocked when the write buffer fills,
:266-271 clearBlocked when it drains, processSendRetry); Ciw's Type I
blocking keeps the customer at the upstream node and releases the
longest blocked one when the destination has capacity
(Ciw ciw/node.py:470-473, block_individual,
release_blocked_individual). Nothing is reordered and nothing is
dropped. The written round leaves on its route at the write
(RoundTransmitter) and takes its slot where it lands.
"""

import dataclasses
import functools
from collections.abc import Callable
from typing import Protocol

import decsim.controller.round_transmission as round_transmission
import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.trace_source as trace_source


class WaitingRound(Protocol):
    """What a waiting line reads of a round: its key and its route.

    A packed round waits in front of the stores and a round in assembly
    in front of the packing stage; the line is this package's own seam.
    """

    round_key: tuple
    route: round_records.SyndromePacketRoute


class HeldRounds:
    """The waiting line in front of one bounded stage of the readout path.

    The run keeps two: one in front of the stores, one in front of the
    packing stage (round_assembly.py); a round waits whole in either,
    and the stage it waits for calls retry when it has room.
    Neither line has a size. The QPU keeps measuring while a round waits,
    so the rounds pile up as a backlog in the controller (Terhal
    1302.3428 lines 3151-3159, Quantum Machines 2412.00289 lines
    478-485), and a Ruby MessageBuffer holds any number of messages
    unless it is given a size (gem5
    src/mem/ruby/network/MessageBuffer.py:58-61,
    MessageBuffer.cc:147-153). A finite line would, when full, have to
    stall the QPU or drop a round, and no source settles either for a
    QEC stream.
    Trace source: round_event(RoundEvent) with kind STALLED when a round
    is held for room and RELEASED when a freed slot admits it. The two
    ends are the wait itself, which is the
    back-pressure a full stage applies to its sender and is measured
    nowhere else: the round waits here, before the stage is asked again,
    so the stage's own time carries none of it. Ruby's MessageBuffer counts that
    wait as the buffer's own statistic, the ticks a message was stalled
    in it (gem5 src/mem/ruby/network/MessageBuffer.cc:76-82
    for the stall counters, :331 where the wait is summed at the
    dequeue), and ns-3's queue disc stamps a packet at the enqueue and
    reads the sojourn time back at the dequeue
    (ns-3 src/traffic-control/model/queue-disc.cc:851
    and :701, the trace source described at queue-disc.h:162-167).
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
        """A round found no room: hold it for a retry.

        False, so the admission that called reports the refusal; a held
        round is recorded STALLED once.
        """
        operation_id, round_index = held.round_key
        if self._is_holding(held):
            return False
        self.waiting.append((held, admit))
        stalled = round_records.RoundEvent.of(
            "STALLED", self.engine.now, operation_id, round_index, held.route
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

    def _released(self, held: WaitingRound) -> None:
        """The round left the waiting line: its wait ends at this tick."""
        operation_id, round_index = held.round_key
        released = round_records.RoundEvent.of(
            "RELEASED",
            self.engine.now,
            operation_id,
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
    """Sends a finished round to the store its tier reads, or holds it.

    A weak-primary run's round goes into the weak syndrome buffer and
    stays there: what the strong tier needs of it goes up with the
    escalation (escalation/strong_redecode.py), never as a second copy
    (Battistel 2303.00054 lines 342 to 347, a cold first stage exists to
    keep the rounds off the cryostat I/O). A strong-primary run's window
    round goes into the strong syndrome buffer over
    controller_to_strong_buffer, its one transport. The sender reserves
    the room the store's own end answers for and hands the round to the
    send; the slot, the copy and the intake line are the receiving
    end's, at the round's landing there
    (syndrome_buffer/weak_syndrome_round_receiver.py and
    syndrome_buffer/strong_syndrome_round_receiver.py).
    """

    # the controller's own fabric: it executes the crossing to the strong
    # syndrome buffer
    link = ports.Port(ports.Link)
    # The weak syndrome buffer's own end: its room, and the landing that takes
    # the slot
    weak_receiver = ports.Port(ports.WeakSyndromeRoundReceiver)
    # The weak syndrome buffer itself, which the window side either reads its
    # windows from or does not
    weak_store = ports.Port(ports.SyndromeBuffer)
    # the room side's end, written by a strong-primary run; absent on a run
    # that never reads from it
    strong_receiver = ports.Port(
        ports.StrongSyndromeRoundReceiver, optional=True
    )
    held_rounds = ports.Port(ports.HeldRounds)
    # the line in front of the packing stage, which a landed round's
    # place in the stage frees
    packing_line = ports.Port(ports.HeldRounds)
    transmitter = ports.Port(round_transmission.RoundTransmitter)
    windows = ports.Port(ports.WindowInput)

    def __init__(self, engine: engine_module.Engine) -> None:
        self.engine = engine
        # strong-primary rounds sent to the strong syndrome buffer and not
        # yet landed there; the packing stage's bound reads it
        # (RoundsInFlight), as it reads the transmitter's in_flight
        self.strong_crossing_count = 0

    def start(self) -> None:
        """Read which store the plan's windows come from, once.

        A strong-primary plan reads its windows from the room side, so
        every round takes one hop into the strong store. The
        window side settles that when the root wires it and never moves it
        again, so the sender reads it here rather than at every admit.
        """
        reads_from_buffer_zero = self.windows.reads_windows_from(
            self.weak_store
        )
        self.publishes_from_strong_store = not reads_from_buffer_zero

    def admit(self, packed: round_records.PackedRound) -> bool:
        """Write the round where it belongs; False when it waits.

        A round that finds rounds already held joins the line behind them
        without asking for room, even when its own bits would fit: gem5's
        requester that was refused "must wait for a recvReqRetry" before
        it sends again (src/mem/port.hh:244-255), and its packet queue
        holds every later packet behind the refused front
        (src/mem/packet_queue.cc:155-162 and 191-217), so a narrow round
        never overtakes a wide one that waits.
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
        """Write the round if its store has room; False when it has none."""
        if self.publishes_from_strong_store:
            if not self.strong_receiver.has_room(packed):
                return self.held_rounds.refuse(packed, self._write)
            self._write_strong(packed)
            return True
        if not self.weak_receiver.has_room(packed):
            return self.held_rounds.refuse(packed, self._write)
        # the round takes its weak syndrome buffer slot when its bits are
        # there: the room is reserved here, and the landing stores and
        # publishes it
        self.weak_receiver.reserve_write(packed)
        self.transmitter.send(packed)
        return True

    def _write_strong(self, packed: round_records.PackedRound) -> None:
        """Carry the round over its link to the strong syndrome buffer.

        The controller is the end this round leaves by, so it executes
        the send (OMNeT++ refuses a module that sends a message it does
        not own, omnetpp src/sim/csimplemodule.cc:333-334;
        gem5 bills a transfer to the port it left by,
        coherent_xbar.cc:354-357).
        The room side takes the room before the round leaves, gem5's
        packet store reserving the packet's bytes before the data lands
        (src/dev/net/pktfifo.hh reserve), and handles the landing itself.
        """
        attribution = transfer_records.TransferAttribution.for_packet(
            packed.packet
        )
        self.strong_receiver.reserve_write(packed)
        self.strong_crossing_count += 1
        landed = functools.partial(self._land_in_strong_store, packed)
        self.link.send(
            transfer_records.LinkPath.CONTROLLER_TO_STRONG_BUFFER,
            packed.wire_bits,
            self.engine.now,
            attribution,
            landed,
        )

    def _land_in_strong_store(
        self, packed: round_records.PackedRound, _transfer
    ) -> None:
        """The strong syndrome buffer took the round and handles the landing.

        The landing publishes the round, so it stops counting in the
        event after, as the transmitter lets a weak round go: a fragment
        landing at this tick still counts it.
        """
        self.strong_receiver.receive_round(packed)
        self.engine.schedule(
            0, self._leave_strong_crossing, label="strong round landed"
        )

    def _leave_strong_crossing(self) -> None:
        """The round left the packing stage, so a waiting round may enter."""
        self.strong_crossing_count -= 1
        self.packing_line.retry()


@dataclasses.dataclass(frozen=True)
class _HeldRoundsTraceSources:
    """Every event the held rounds reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (gem5 src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    round_event: trace_source.TraceSource = trace_source.new_source()
