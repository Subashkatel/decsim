"""The transmitter: a stored round leaves on its route at the write.

Every round rides controller_to_weak_buffer, whose sending end this is.
A window-input round is published at the weak syndrome buffer. A
feedback-memory round is written there too and then rides
weak_buffer_to_weak_decoder, sent by the store's own outgoing port.

The sender never waits for a round to land before sending the next, as
DAQs stream every round (Yang et al. 2605.04892; Google 2408.13687;
Caune et al. 2410.05202), gem5's DmaPort queues behind transmitList
(src/dev/dma_device.cc) and ns-3's point-to-point device starts the next
packet at TransmitComplete. The FIFO channel keeps delivery order.
"""

import dataclasses
import functools

import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.trace_source as trace_source


class RoundTransmitter:
    """Sends a stored round on its route and counts it until it lands.

    Trace source: round_event(RoundEvent) with kinds CWB_SENT and
    FEEDBACK_MEMORY_DELIVERED.
    """

    link = ports.Port(ports.Link)
    # the decoders' end of the memory route, which hears its landing
    memory_arrivals = ports.Port(ports.MemoryRoundArrivals)
    # The weak syndrome buffer's port toward the controller: it handles what
    # lands there and asks the store to send what leaves it
    weak_receiver = ports.Port(ports.WeakSyndromeRoundReceiver)
    # the line in front of the packing stage, which a round that left
    # its route frees a place for
    packing_line = ports.Port(ports.HeldRounds)

    def __init__(self, engine: engine_module.Engine) -> None:
        self.engine = engine
        self.in_flight = 0
        self.trace = _TraceSources()

    def send(self, packed: round_records.PackedRound) -> None:
        """Send the round on its route at this tick.

        The departure is its own event, so the writer's step completes before
        the windows hear of the round; rounds sent at one tick depart in
        completion order with no arbitration between routes, as gem5's DmaPort
        and ns-3's device queue them.
        """
        self.in_flight += 1
        depart = functools.partial(self._depart, packed)
        self.engine.schedule(0, depart, label="round transmission")

    def _depart(self, packed: round_records.PackedRound) -> None:
        window_input = round_records.SyndromePacketRouteKind.WINDOW_INPUT
        if packed.route.kind is window_input:
            self._send_window_input(packed)
            return
        self._send_feedback_memory(packed)

    def check_settled(self) -> None:
        """At the end of a run no round may still be on its route."""
        if self.in_flight:
            raise RuntimeError(
                f"run ended with {self.in_flight} rounds in flight on their "
                f"route"
            )

    def _send_window_input(self, packed: round_records.PackedRound) -> None:
        self._fire("CWB_SENT", packed)
        publish = functools.partial(self._publish, packed)
        self._send(
            transfer_records.LinkPath.CONTROLLER_TO_WEAK_BUFFER, packed, publish
        )

    def _publish(self, packed: round_records.PackedRound) -> None:
        """The weak syndrome buffer's receiver takes the landing.

        This sender hears the publication only for its in_flight count, as
        gem5's port hands the packet to the peer's receive method
        (src/mem/port.hh:603-614).
        """
        self.weak_receiver.receive_round(packed, self._leave_after_publication)

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

    def _send_feedback_memory(self, packed: round_records.PackedRound) -> None:
        """Send a feedback-memory round up to the store, which sends it on.

        It occupies a slot once written, as any round does, so it crosses
        controller_to_weak_buffer first; the store's own port sends it to the
        weak decoder and frees the slot.
        """
        landed = functools.partial(self._write_feedback_memory, packed)
        self._send(
            transfer_records.LinkPath.CONTROLLER_TO_WEAK_BUFFER, packed, landed
        )

    def _write_feedback_memory(self, packed: round_records.PackedRound) -> None:
        deliver = functools.partial(self._deliver_feedback_memory, packed)
        self.weak_receiver.send_memory_round(packed, deliver)

    def _deliver_feedback_memory(
        self, packed: round_records.PackedRound
    ) -> None:
        """The memory round reached the decoder side, which takes it."""
        source_operation_id = packed.route.source_operation_id
        self.memory_arrivals.receive_memory_round(source_operation_id)
        self._fire("FEEDBACK_MEMORY_DELIVERED", packed)
        self._leave()

    def _fire(self, kind: str, packed: round_records.PackedRound) -> None:
        operation_id, round_index = packed.round_key
        event = round_records.RoundEvent.of(
            kind, self.engine.now, operation_id, round_index, packed.route
        )
        self.trace.round_event.fire(event)

    def _send(
        self, path: transfer_records.LinkPath, packed, on_delivered
    ) -> None:
        attribution = transfer_records.TransferAttribution.for_packet(
            packed.packet
        )

        def delivered(_transfer) -> None:
            on_delivered()

        self.link.send(
            path, packed.wire_bits, self.engine.now, attribution, delivered
        )


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the round transmitter reports, as one member."""

    round_event: trace_source.TraceSource = trace_source.new_source()
