"""The transmitter: a stored round leaves on its route at the write.

A window-input round rides controller_to_weak_buffer and is published to
the windows at delivery, where its publication tick is stamped on the
store; on a card that leaves the path unpriced the publication is the
storage itself. A feedback-memory round rides weak_buffer_to_weak_decoder
and, at delivery, tells the windows and frees its Buffer 0 slot. The
sender never waits for a round to land before sending the next: the DAQs
of Yang et al. (2605.04892) and Google's control electronics
(2408.13687) stream every round, Caune et al. (2410.05202) publish each
classified result as it is produced, gem5's DmaPort queues the next
request behind the front of transmitList (src/dev/dma_device.cc) and
ns-3's point-to-point device starts the next packet at TransmitComplete
(point-to-point-net-device.cc); the FIFO channel keeps delivery order.
in_flight counts the rounds from their send until the windows have heard
of them; the packing stage's bound reads it (RoundsInFlight).
"""

import functools
from typing import Optional

import decsim.observe.trace_source as trace_source
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records


class RoundTransmitter:
    """Sends a stored round on its route and tells the windows at delivery.

    Trace source: round_event(RoundEvent) with kinds CWB_SENT, PUBLISHED
    and FEEDBACK_MEMORY_DELIVERED.
    """

    def __init__(self, engine, link, weak_store, windows) -> None:
        self.engine = engine
        self.link = link
        self.weak_store = weak_store
        self.windows = windows
        self.in_flight = 0
        self.round_event = trace_source.TraceSource()

    def publication_tick_at_storage(
        self, route: round_records.SyndromePacketRoute
    ) -> Optional[int]:
        """The tick a round stored now is published, if known at storage.

        A window-input round on a card without controller_to_weak_buffer
        is published as it is stored; with the path priced it is
        published at delivery; a feedback-memory round is never
        published, its terminal is FEEDBACK_MEMORY_DELIVERED.
        """
        window_input = round_records.SyndromePacketRouteKind.WINDOW_INPUT
        if route.kind is not window_input:
            return None
        if self._publishes_at_delivery():
            return None
        return self.engine.now

    def send(self, packed: round_records.PackedRound) -> None:
        """Send the round on its route at this tick.

        The departure is its own event, so the writer's own step (the
        store, the trace line, the strong write) completes before the
        windows hear of the round; rounds sent at one tick depart in
        completion order, each at its own send, as gem5's DmaPort queues
        each request on transmitList and ns-3's device on its FIFO, with
        no arbitration between the routes.
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

    def _publishes_at_delivery(self) -> bool:
        return self.link.is_wired(
            transfer_records.LinkPath.CONTROLLER_TO_WEAK_BUFFER
        )

    def _send_window_input(self, packed: round_records.PackedRound) -> None:
        if not self._publishes_at_delivery():
            self.windows.accept_window_input(packed.packet)
            self._leave_after_publication()
            return
        self._fire("CWB_SENT", packed)
        publish = functools.partial(self._publish, packed)
        self._send(
            transfer_records.LinkPath.CONTROLLER_TO_WEAK_BUFFER, packed, publish
        )

    def _publish(self, packed: round_records.PackedRound) -> None:
        """The round reached Buffer 0: stamp it, record it, wake the windows."""
        self.weak_store.mark_publication_tick(packed.round_key, self.engine.now)
        self._fire("PUBLISHED", packed)
        self.windows.accept_window_input(packed.packet)
        self._leave_after_publication()

    def _leave_after_publication(self) -> None:
        """The round leaves in the event after its publication.

        A round is in flight until its publication has completed, so a
        fragment landing at the publication tick still counts it.
        """
        self.engine.schedule(0, self._leave, label="round publication complete")

    def _leave(self) -> None:
        self.in_flight -= 1

    def _send_feedback_memory(self, packed: round_records.PackedRound) -> None:
        deliver = functools.partial(self._deliver_feedback_memory, packed)
        self._send(
            transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER,
            packed,
            deliver,
        )

    def _deliver_feedback_memory(
        self, packed: round_records.PackedRound
    ) -> None:
        """The memory round landed: tell the windows and free its slot."""
        source_operation_id = packed.route.source_operation_id
        self.windows.accept_feedback_memory_round(source_operation_id)
        self._fire("FEEDBACK_MEMORY_DELIVERED", packed)
        self.weak_store.release_round(packed.round_key)
        self._leave()

    def _fire(self, kind: str, packed: round_records.PackedRound) -> None:
        operation_id, round_index = packed.round_key
        event = round_records.RoundEvent.of(
            kind, self.engine.now, operation_id, round_index, packed.route
        )
        self.round_event.fire(event)

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
