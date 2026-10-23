"""The transmitter: a stored round leaves on its route at the write.

Every round rides the controller's write into a written store, whose
sending end this is. A window-input round is published at its own
store, and a copy of it is stored where another pool's window reads it
(round_destinations.py). A feedback-memory round is written to its own
store too and then rides the store's read hop, sent by the store's own
outgoing port.

The sender never waits for a round to land before sending the next, as
DAQs stream every round (Yang et al. 2605.04892; Google 2408.13687;
Caune et al. 2410.05202), gem5's DmaPort queues behind transmitList
(src/dev/dma_device.cc) and ns-3's point-to-point device starts the next
packet at TransmitComplete. The FIFO channel keeps delivery order.
"""

import dataclasses
import functools

import decsim.controller.round_destinations as round_destinations
import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.trace_source as trace_source


class RoundTransmitter:
    """Sends each stored round on its route, to the store side it is for.

    It counts a round in flight until its own store publishes it; a copy
    rides uncounted, since the windows hear of the round from its own
    store.

    Trace source: round_event(RoundEvent) with kinds CWB_SENT and
    FEEDBACK_MEMORY_DELIVERED, neither for a copy.
    """

    link = ports.Port(ports.Link)
    # the decoders' end of the memory route, which hears its landing
    memory_arrivals = ports.Port(ports.MemoryRoundArrivals)
    # the line in front of the packing stage, which a round that left
    # its route frees a place for
    packing_line = ports.Port(ports.HeldRounds)

    def __init__(self, engine: engine_module.Engine) -> None:
        self.engine = engine
        self.in_flight = 0
        self.trace = _TraceSources()

    def send(
        self,
        packed: round_records.PackedRound,
        side: round_destinations.StoreSide,
        is_copy: bool,
    ) -> None:
        """Send the round on its route to the store side at this tick.

        The departure is its own event, so the writer's step completes before
        the windows hear of the round; rounds sent at one tick depart in
        completion order with no arbitration between routes, as gem5's DmaPort
        and ns-3's device queue them.
        """
        if not is_copy:
            self.in_flight += 1
        depart = functools.partial(self._depart, packed, side, is_copy)
        self.engine.schedule(0, depart, label="round transmission")

    def _depart(
        self,
        packed: round_records.PackedRound,
        side: round_destinations.StoreSide,
        is_copy: bool,
    ) -> None:
        window_input = round_records.SyndromePacketRouteKind.WINDOW_INPUT
        if packed.route.kind is not window_input:
            self._send_feedback_memory(packed, side)
            return
        if is_copy:
            self._send_copy(packed, side)
            return
        self._send_window_input(packed, side)

    def check_settled(self) -> None:
        """At the end of a run no round may still be on its route."""
        if self.in_flight:
            raise RuntimeError(
                f"run ended with {self.in_flight} rounds in flight on their "
                f"route"
            )

    def _send_window_input(
        self,
        packed: round_records.PackedRound,
        side: round_destinations.StoreSide,
    ):
        self._fire("CWB_SENT", packed)
        publish = functools.partial(self._publish, packed, side)
        attribution = attribution_of(packed, side, False)
        self._send(side.write_path, packed, attribution, publish)

    def _send_copy(
        self,
        packed: round_records.PackedRound,
        side: round_destinations.StoreSide,
    ) -> None:
        """Write a copy where another pool's window reads it."""
        stored = functools.partial(self._store_copy, packed, side)
        attribution = attribution_of(packed, side, True)
        self._send(side.write_path, packed, attribution, stored)

    def _publish(
        self,
        packed: round_records.PackedRound,
        side: round_destinations.StoreSide,
    ) -> None:
        """The round's own store's receiver takes the landing.

        This sender hears the publication only for its in_flight count, as
        gem5's port hands the packet to the peer's receive method
        (src/mem/port.hh:603-614).
        """
        side.receiver.receive_written(packed, self._leave_after_publication)

    def _store_copy(
        self,
        packed: round_records.PackedRound,
        side: round_destinations.StoreSide,
    ) -> None:
        """The copy's store takes the landing; nothing here counts it."""
        side.receiver.receive_copy(packed, _nothing)

    def _leave_after_publication(self) -> None:
        """Leave in the event after publication.

        A fragment landing at the publication tick still counts the round in
        flight.
        """
        self.engine.schedule(0, self._leave, label="round publication complete")

    def _leave(self) -> None:
        """The round left the packing stage, so a waiting round may enter."""
        self.in_flight -= 1
        self.packing_line.retry()

    def _send_feedback_memory(
        self, packed, side: round_destinations.StoreSide
    ) -> None:
        """Send a feedback-memory round up to the store, which sends it on.

        It occupies a slot once written, as any round does, so it crosses
        the write hop first; the store's own port sends it to the decoder
        and frees the slot.
        """
        landed = functools.partial(self._write_feedback_memory, packed, side)
        attribution = attribution_of(packed, side, False)
        self._send(side.write_path, packed, attribution, landed)

    def _write_feedback_memory(
        self, packed, side: round_destinations.StoreSide
    ) -> None:
        deliver = functools.partial(self._deliver_feedback_memory, packed)
        side.receiver.send_memory_round(packed, deliver)

    def _deliver_feedback_memory(
        self, packed: round_records.PackedRound
    ) -> None:
        """The memory round reached the decoder side, which takes it."""
        source_stream_id = packed.route.source_stream_id
        self.memory_arrivals.receive_memory_round(source_stream_id)
        self._fire("FEEDBACK_MEMORY_DELIVERED", packed)
        self._leave()

    def _fire(self, kind: str, packed: round_records.PackedRound) -> None:
        stream_id, round_index = packed.round_key
        event = round_records.RoundEvent.of(
            kind, self.engine.now, stream_id, round_index, packed.route
        )
        self.trace.round_event.fire(event)

    def _send(
        self,
        path: transfer_records.LinkPath,
        packed,
        attribution: transfer_records.TransferAttribution,
        on_delivered,
    ) -> None:
        def delivered(_transfer) -> None:
            on_delivered()

        self.link.send(
            path, packed.wire_bits, self.engine.now, attribution, delivered
        )


def attribution_of(
    packed: round_records.PackedRound,
    side: round_destinations.StoreSide,
    is_copy: bool,
) -> transfer_records.TransferAttribution:
    """Whose write this is: the round's, marked when it is a copy.

    A per-round measure reads each round's own write once and skips its
    copies (experiments/measure.py controller_write_delays_us).
    """
    attribution = transfer_records.TransferAttribution.for_packet(packed.packet)
    if not is_copy:
        return attribution
    relation = transfer_records.RoundCopyRelation(side.name)
    return dataclasses.replace(attribution, relation=relation)


def _nothing() -> None:
    """A copy is stored where its readers are; the sender waits on none."""


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the round transmitter reports, as one member."""

    round_event: trace_source.TraceSource = trace_source.new_source()
