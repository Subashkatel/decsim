"""Buffer 0's incoming port: its room, and the landing that takes a slot.

A round occupies a slot when its bits are in the store, which is at the
landing of the hop that carried them, and it is readable at that same
instant. Every referent that models storage writes it there: ns-3's
channel schedules the destination device's own Receive after the
transmission and the propagation
(`tmp/resources/l5_buffers/ns3-point-to-point/point-to-point-channel.cc:88-92`,
"Simulator::ScheduleWithContext(m_link[wire].m_dst->GetNode()->GetId(),
txTime + m_delay, &PointToPointNetDevice::Receive, ...)", whose method
is `point-to-point-net-device.cc:324`); OMNeT++ takes ownership into the
destination module and inserts inside that module's handler
(`tmp/resources/omnetpp/src/sim/csimplemodule.cc:782-783` "// get
ownership" then "take(msg);", `:799` "handleMessage(msg);", with
`tmp/resources/l5_buffers/omnetpp-queueing/queueinglib/Queue.cc:84-94`
checking "queue.length() >= capacity" and then "queue.insert( job );"
inside it); Ciw counts the individual in the destination's own accept
(`tmp/resources/l5_buffers/Ciw/ciw/node.py:602` "next_node.accept(
next_individual)" into `:102-103` "self.individuals[...].append(
next_individual)" and "self.number_of_individuals += 1"). The quantum
control papers put the store on the far side of the wire too: Caune
2410.05202 lines 1243-1247 "Stores the outcomes of the latest
measurement round in the decoder sequencer's memory ... a 1.4 microsecond
delay must take place between measurement and buffering"; Google
2408.13687 lines 471-477, measurements "transmitted to a specialized
workstation via low-latency Ethernet. Inside the workstation ... streamed
to the real-time decoding software via a shared memory buffer"; Maurer
2510.21600 lines 593-597, readouts "routed via the high speed serial
connection to the decoder FPGA. Once all syndrome bits are collected,
the syndrome is put into a FIFO".

The sender must still refuse before it sends, so this end answers for the
room with the writes it has in flight counted as taken: gem5's queue
holds reserved entries against its size
(`tmp/resources/gem5/src/mem/cache/queue.hh:150-153` "bool isFull() const
{ return (allocated >= numEntries - numReserve); }", the reserve declared
at `:87-93`), its cache blocks the port the moment the write buffer fills
(`src/mem/cache/base.cc:255-257` "if (writeBuffer.isFull()) { setBlocked(
(BlockedCause)MSHRQueue_WriteBuffer); }" and `:266-271` clearBlocked when
it drains), a refusal being the receiver's answer to the sender
(`src/mem/port.hh:244-255`, "If the send does not succeed ... the sender
must wait for a recvReqRetry"); and Ruby sums the same two counts
(`tmp/resources/gem5/src/mem/ruby/network/MessageBuffer.cc:181`
"if (current_size + current_stall_size + n <= m_max_size)", the two sizes
read at `:155-158`). This is the shape syndrome buffer 1 already has
(strong_round_writer.py:48-53), so both stores now answer the same
question by the same shape of object.

Two calls of the controller arrive here, both about Buffer 0. A packed
round lands over controller_to_weak_buffer: this end stores it, narrates
the copy and the intake, and announces the published round to the window
manager. And a timing-only feedback-memory round is handed over to be
sent: it takes its slot here, because Buffer 0 is where it waits, and it
leaves by the store's own outgoing port (round_output.py), which frees
the slot at the delivery. A timing-only round is never published: no
window reads it.
"""

import dataclasses
from typing import Callable, Optional

import decsim.records.log_sources as log_sources
import decsim.records.rounds as round_records
import decsim.trace_source as trace_source


class RoundStoreInput:
    """Buffer 0's port toward the controller, bound once by the root.

    Trace sources: round_event(RoundEvent) with kind PUBLISHED, and
    copy_made(round_key, bits, "controller assembler", "Buffer 0") at
    every intake, the write's copy (data_path.md hop 2).
    """

    def __init__(self, engine, store, output, windows) -> None:
        self.engine = engine
        self.store = store
        # the store's outgoing port, which sends what leaves the store
        self.output = output
        self.windows = windows
        self.writes_in_flight = 0
        self.trace = _TraceSources()

    def has_room(self) -> bool:
        """A write can land: capacity counts the rounds stored and in flight."""
        capacity = self.store.capacity_rounds()
        if capacity is None:
            return True
        return self.store.occupancy + self.writes_in_flight < capacity

    def reserve_write(self) -> None:
        """Take the room one crossing round will need, before it leaves."""
        assert self.has_room(), "a round was written into a full store"
        self.writes_in_flight += 1

    def receive_round(self, packed: round_records.PackedRound) -> None:
        """Take one round that landed here: its room is now its slot.

        The store and the publication are one call at one tick, because
        the bits become readable when they are here and not before, and
        the announcement follows the record, so the window manager never
        hears of a round the store does not yet call readable
        (validation buffer_contract.md, Buffer 0).

        A round whose operation closed while it crossed is dropped at the
        door on the strong side (strong_round_writer.py _drop_landing).
        It cannot reach this door: the window manager refuses a round of
        an operation whose store closed by raising
        (windows/window_manager.py _refuse_unplanned_round), that being
        the device emitting more rounds than the plan expects, so there
        is no drop to make here.
        """
        self.writes_in_flight -= 1
        self._take_slot(packed, "controller_to_weak_buffer", self.engine.now)
        self._fire_published(packed)
        self.windows.accept_window_input(packed.packet)

    def send_memory_round(
        self,
        packed: round_records.PackedRound,
        on_delivered: Callable[[], None],
    ) -> None:
        """Take one timing-only round and send it to the store's decoder.

        The round waits in Buffer 0 until the wire takes it, so it takes
        its slot here, where the store has it; the outgoing port frees
        that slot at the delivery. It is not published: no window reads
        a timing-only round, so nothing may be told it is readable.
        """
        self.writes_in_flight -= 1
        self._take_slot(packed, "packing", None)
        self.output.send_memory_round(packed, on_delivered)

    def check_settled(self) -> None:
        """At the end of a run no write may still be on the wire.

        What the store itself still holds is not asked here: a run that
        ends with nothing delivered is a legal end, and the law that
        covers it is the callback law (tests/test_callback_law.py).
        """
        if self.writes_in_flight:
            raise RuntimeError(
                f"syndrome buffer 0 ended with {self.writes_in_flight} "
                f"controller_to_weak_buffer writes in flight"
            )

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
    """Every event the store's incoming port reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (tmp/resources/gem5/src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    copy_made: trace_source.TraceSource = trace_source.new_source()
    round_event: trace_source.TraceSource = trace_source.new_source()
