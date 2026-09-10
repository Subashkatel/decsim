"""One physical channel: a setup engine and a wire, driven by events.

A transfer with a setup cost first waits for the channel's setup engine,
which programs one transfer at a time in request order: gem5 keeps one
transmitList per DmaPort (src/dev/dma_device.cc, dmaAction pushes, the
front is sent first), and the DMA engine of Shao et al. (MICRO 2016,
section III.C) services descriptors one by one while the processor is
free to do other work. A transfer is ready when its setup ends, or at
its request when it has no setup; a request with no setup never waits
for another path's setup. The wire takes ready transfers in the order
they became ready, one at a time, serializes each at the channel's rate
and delivers it one propagation later: ns-3's point-to-point device
(point-to-point-net-device.cc: Send enqueues, TransmitStart runs when
the transmitter is READY, and the receiver has the packet txTime plus
the channel delay later). A fractional tick of serialization rounds up,
because a transfer never ends before its exact time. An unbounded
channel serializes nothing and never queues.
"""

import dataclasses
import fractions
import math
from typing import Callable, Optional

import decsim.config as config
import decsim.engine
import decsim.links.settings as link_settings
import decsim.records.transfers as transfer_records

OnDelivered = Callable[[transfer_records.Transfer], None]


class Channel:
    """One channel at run time: its settings, its setup engine, its wire."""

    def __init__(
        self,
        channel_settings: link_settings.ChannelSettings,
        engine: decsim.engine.Engine,
    ):
        self._settings = channel_settings
        self._engine = engine
        self._setup_free_ticks = 0
        self._wire_free_ticks = 0
        self._last_request_ticks = 0
        self._transfer_count = 0

    def send(
        self,
        payload_bits: Optional[int],
        now_ticks: int,
        setup_ticks: int,
        on_delivered: OnDelivered,
    ) -> None:
        """Send one payload; on_delivered(transfer) runs at its delivery.

        The setup, when the path pays one, is queued on the setup engine
        now and finishes at the ready tick; the wire takes the transfer
        from that tick.
        """
        assert payload_bits is None or payload_bits >= 0, (
            "a payload is never negative"
        )
        assert now_ticks >= self._last_request_ticks, (
            "requests reach a channel in time order"
        )
        self._last_request_ticks = now_ticks
        ready_ticks = self._ready_ticks(now_ticks, setup_ticks)
        if setup_ticks > 0:
            self._setup_free_ticks = ready_ticks
        request = _Request(payload_bits, now_ticks, ready_ticks, on_delivered)
        if ready_ticks == now_ticks:
            self._take_wire(request)
            return
        setup_delay_ticks = ready_ticks - self._engine.now
        self._engine.schedule(
            setup_delay_ticks,
            lambda: self._take_wire(request),
            label="link setup",
        )

    def expected_delay_ticks(
        self, payload_bits: Optional[int], now_ticks: int, setup_ticks: int
    ) -> int:
        """The delay a transfer would pay if nothing else reached the channel.

        The setup engine's and the wire's queues as they stand now, the
        serialization and the propagation. A scheduler's estimate, exact
        whenever no later request overtakes it on the wire.
        """
        ready_ticks = self._ready_ticks(now_ticks, setup_ticks)
        start_ticks, serialization_ticks = self._wire_interval(
            ready_ticks, payload_bits
        )
        end_ticks = start_ticks + serialization_ticks
        delivery_ticks = end_ticks + self._settings.propagation_latency_ticks
        return delivery_ticks - now_ticks

    def _ready_ticks(self, now_ticks: int, setup_ticks: int) -> int:
        """When the transfer reaches the wire's queue: after its setup."""
        if setup_ticks == 0:
            return now_ticks
        setup_start_ticks = max(now_ticks, self._setup_free_ticks)
        return setup_start_ticks + setup_ticks

    def _wire_interval(
        self, ready_ticks: int, payload_bits: Optional[int]
    ) -> tuple[int, int]:
        """The wire's next slot for the payload: its start and its length."""
        capacity = self._settings.capacity
        if capacity is None:
            return ready_ticks, 0
        start_ticks = max(ready_ticks, self._wire_free_ticks)
        serialization_ticks = _serialization_ticks(payload_bits, capacity)
        return start_ticks, serialization_ticks

    def _take_wire(self, request: "_Request") -> None:
        """At the ready tick: take the wire's next slot, schedule delivery."""
        start_ticks, serialization_ticks = self._wire_interval(
            request.ready_ticks, request.payload_bits
        )
        end_ticks = start_ticks + serialization_ticks
        if self._settings.capacity is not None:
            self._wire_free_ticks = end_ticks
        propagation_ticks = self._settings.propagation_latency_ticks
        delivery_ticks = end_ticks + propagation_ticks
        setup_ticks = request.ready_ticks - request.request_ticks
        queue_wait_ticks = start_ticks - request.ready_ticks
        total_delay_ticks = delivery_ticks - request.request_ticks
        transfer = transfer_records.Transfer(
            payload_bits=request.payload_bits,
            request_ticks=request.request_ticks,
            setup_ticks=setup_ticks,
            send_ticks=request.ready_ticks,
            queue_wait_ticks=queue_wait_ticks,
            serialization_ticks=serialization_ticks,
            propagation_ticks=propagation_ticks,
            serializer_start_ticks=start_ticks,
            serializer_end_ticks=end_ticks,
            delivery_ticks=delivery_ticks,
            total_delay_ticks=total_delay_ticks,
            physical_sequence=self._transfer_count,
        )
        self._transfer_count += 1
        delivery_delay_ticks = delivery_ticks - self._engine.now
        self._engine.schedule(
            delivery_delay_ticks,
            lambda: request.on_delivered(transfer),
            label="link delivery",
        )


@dataclasses.dataclass(frozen=True)
class _Request:
    """One send waiting for its setup, then for the wire."""

    payload_bits: Optional[int]
    request_ticks: int
    ready_ticks: int
    on_delivered: OnDelivered


def _serialization_ticks(
    payload_bits: int, capacity: link_settings.CapacitySettings
) -> int:
    """Whole ticks to put the payload on the wire at the channel's rate.

    A fractional tick rounds up: serialization never ends before the exact
    transmission time. Exact Fraction arithmetic keeps the card's rate
    exact, so a whole-tick duration is never inflated by float error.
    """
    rate = capacity.exact_aggregate_bits_per_microsecond()
    payload = fractions.Fraction(payload_bits)
    bits_times_ticks = payload * config.TICKS_PER_MICROSECOND
    exact_ticks = bits_times_ticks / rate
    return math.ceil(exact_ticks)
