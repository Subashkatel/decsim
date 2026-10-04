"""One physical channel: a setup engine and a wire, driven by events.

A transfer with a setup cost first waits for the channel's setup engine,
which programs one transfer at a time in request order: gem5 keeps one
transmitList per DmaPort (src/dev/dma_device.cc, dmaAction pushes, the
front is sent first), and the DMA engine of Shao et al. (MICRO 2016,
section III.C) services descriptors one by one while the processor is
free to do other work. A transfer is ready when its setup ends, or at
its request when it has no setup; a request with no setup never waits
for another path's setup. The wire takes ready transfers in the order
they became ready, one at a time, serializes each at the channel's rate,
its path's header with it, and delivers it one propagation later: ns-3's
point-to-point device (point-to-point-net-device.cc: Send adds the
header and enqueues, TransmitStart runs when the transmitter is READY
and times the whole packet, and the receiver has it txTime plus the
channel delay later). A fractional tick of serialization rounds up,
because a transfer never ends before its exact time. An unbounded
channel serializes nothing and never queues.

This is the protocol row `ideal`: one whole frame, an unbounded receive
buffer and nothing lost, the lossless reference gem5's SimpleNetwork
gives with its default infinite buffers (src/mem/ruby/network/simple/
SimpleNetwork.py:53-57) and ns-3's device with no error model
(point-to-point-net-device.cc:55-59). The packet rows keep the setup
engine and replace the wire (decsim/links/credit_channel.py,
decsim/links/reliable_channel.py).
"""

import collections
import copy
import dataclasses
import fractions
import math
from collections.abc import Callable
from typing import Optional

import decsim.config as config
import decsim.engine
import decsim.links.settings as link_settings
import decsim.records.transfers as transfer_records
import decsim.trace_source as trace_source

OnDelivered = Callable[[transfer_records.Transfer], None]


@dataclasses.dataclass(frozen=True)
class FramedPayload:
    """What one transfer puts on the wire: its payload, and its path's header.

    The two are kept apart because the ledger counts the payload a
    component sent and the wire serializes both. A payload of unknown
    size rides an unbounded channel only, which serializes nothing.
    """

    payload_bits: Optional[int]
    header_bits: int = 0


@dataclasses.dataclass(frozen=True)
class FrameTiming:
    """One frame's trip across a wire.

    The frame waited credit_wait_ticks for a receive-buffer credit, held
    the wire from start_ticks to end_ticks, and landed at the receiver
    at landed_ticks. bits is None for a payload of unknown size on an
    unbounded wire.
    """

    bits: Optional[int]
    credit_wait_ticks: int
    start_ticks: int
    end_ticks: int
    landed_ticks: int


@dataclasses.dataclass(frozen=True)
class FrameRecord:
    """One frame on one channel, reported when its message is delivered.

    transfer_sequence is the channel's count of the message the frame
    belongs to and frame_index its place in the message. A reliable
    channel also reports each lost frame and each retransmission.
    """

    channel: str
    transfer_sequence: int
    frame_index: int
    timing: FrameTiming
    is_lost: bool = False
    is_retransmission: bool = False


class Channel:
    """One channel at run time: a setup engine, then a wire moving each whole.

    Trace source: frame_landed(record), one FrameRecord per frame when
    its message is delivered.
    """

    def __init__(
        self,
        channel_settings: link_settings.ChannelSettings,
        engine: decsim.engine.Engine,
    ):
        self.trace = _TraceSources()
        self._settings = channel_settings
        self._engine = engine
        self._in_setup: collections.deque = collections.deque()
        self._wire = self._new_wire()
        self._last_request_ticks = 0

    def send(
        self,
        framed: FramedPayload,
        now_ticks: int,
        setup_ticks: int,
        on_delivered: OnDelivered,
    ) -> None:
        """Send one framed payload; on_delivered(transfer) runs at delivery.

        The setup, when the path pays one, is queued on the setup engine
        now and finishes at the ready tick; the wire takes the transfer
        from that tick.
        """
        payload_bits = framed.payload_bits
        assert payload_bits is None or payload_bits >= 0, (
            "a payload is never negative"
        )
        assert now_ticks >= self._last_request_ticks, (
            "requests reach a channel in time order"
        )
        self._last_request_ticks = now_ticks
        ready_ticks = self._ready_ticks(now_ticks, setup_ticks)
        request = _Request(
            framed, now_ticks, setup_ticks, ready_ticks, on_delivered
        )
        if ready_ticks == now_ticks:
            self._take_wire(request)
            return
        self._in_setup.append(request)
        setup_delay_ticks = ready_ticks - self._engine.now
        self._engine.schedule(
            setup_delay_ticks,
            lambda: self._finish_setup(request),
            label="link setup",
        )

    def expected_delay_ticks(
        self, framed: FramedPayload, now_ticks: int, setup_ticks: int
    ) -> int:
        """The delay a transfer would pay if nothing else reached the channel.

        The setup engine's and the wire's queues as they stand now, the
        serialization and the propagation. A transfer with a setup is
        ready after every transfer now in setup, so each of those takes
        the wire ahead of it. A scheduler's estimate: on the ideal and
        credit rows it is exact whenever no later request overtakes it
        on the wire; on the reliable row it is a lower bound, because a
        packet's wait for the acknowledgement window is left out.
        """
        ready_ticks = self._ready_ticks(now_ticks, setup_ticks)
        wire = self._wire_as_it_stands()
        if setup_ticks > 0:
            for waiting in self._in_setup:
                wire.cross(waiting.framed, waiting.ready_ticks)
        timings = wire.cross(framed, ready_ticks)
        last_frame = timings[-1]
        return last_frame.landed_ticks - now_ticks

    def _new_wire(self) -> "IdealWire":
        """The wire this row serializes on."""
        return IdealWire(self._settings)

    def _wire_as_it_stands(self):
        """A copy of the wire to price a transfer on without moving it."""
        return self._wire.copy()

    def _ready_ticks(self, now_ticks: int, setup_ticks: int) -> int:
        """When the transfer reaches the wire's queue: after its setup."""
        if setup_ticks == 0:
            return now_ticks
        setup_start_ticks = now_ticks
        if self._in_setup:
            last_in_setup = self._in_setup[-1]
            setup_start_ticks = max(now_ticks, last_in_setup.ready_ticks)
        return setup_start_ticks + setup_ticks

    def _finish_setup(self, request: "_Request") -> None:
        """At the setup's end: the engine frees and the wire takes it."""
        finished = self._in_setup.popleft()
        assert finished is request, "setups finish in request order"
        self._take_wire(request)

    def _take_wire(self, request: "_Request") -> None:
        """At the ready tick: cross the wire, schedule the delivery."""
        transfer_sequence = self._wire.crossing_count
        timings = self._wire.cross(request.framed, request.ready_ticks)
        transfer = transfer_for(request, timings, transfer_sequence)
        delivery_delay_ticks = transfer.delivery_ticks - self._engine.now
        self._engine.schedule(
            delivery_delay_ticks,
            lambda: self._deliver(request, transfer, timings),
            label="link delivery",
        )

    def _deliver(
        self,
        request: "_Request",
        transfer: transfer_records.Transfer,
        timings: tuple,
    ) -> None:
        """At the delivery: the frames are reported, then the caller hears."""
        if self.trace.frame_landed.has_listeners:
            records = frame_records(
                self._settings.name, transfer.physical_sequence, timings
            )
            for record in records:
                self.trace.frame_landed.fire(record)
        request.on_delivered(transfer)


class IdealWire:
    """One wire that serializes each transfer whole, as one frame.

    An unbounded wire serializes nothing and keeps no queue.
    """

    def __init__(self, channel_settings: link_settings.ChannelSettings):
        self._capacity = channel_settings.capacity
        self._propagation_ticks = channel_settings.propagation_latency_ticks
        self._free_ticks = 0
        self.crossing_count = 0

    def copy(self) -> "IdealWire":
        """The same wire in the same state, to price on."""
        return copy.copy(self)

    def cross(
        self, framed: FramedPayload, ready_ticks: int
    ) -> tuple[FrameTiming, ...]:
        """Take the wire's next slot for the whole transfer."""
        self.crossing_count += 1
        if self._capacity is None:
            landed_ticks = ready_ticks + self._propagation_ticks
            timing = FrameTiming(
                None, 0, ready_ticks, ready_ticks, landed_ticks
            )
            return (timing,)
        start_ticks = max(ready_ticks, self._free_ticks)
        wire_bits = framed.payload_bits + framed.header_bits
        serialization = serialization_ticks(wire_bits, self._capacity)
        end_ticks = start_ticks + serialization
        self._free_ticks = end_ticks
        landed_ticks = end_ticks + self._propagation_ticks
        timing = FrameTiming(wire_bits, 0, start_ticks, end_ticks, landed_ticks)
        return (timing,)


def transfer_for(
    request: "_Request", timings: tuple, transfer_sequence: int
) -> transfer_records.Transfer:
    """The transfer's record: its frames' span on the wire, as one move.

    The wire held the transfer from its first frame's start to its last
    frame's end; serialization_ticks is the time its frames held the
    wire, and queue_wait_ticks everything else from the ready tick to
    the last frame's end (the wire's queue, credit waits, recovery from
    a loss), so the total is still the sum of the transfer's parts.
    """
    first_frame = timings[0]
    last_frame = timings[-1]
    serialization = 0
    for timing in timings:
        serialization += timing.end_ticks - timing.start_ticks
    wire_span_ticks = last_frame.end_ticks - request.ready_ticks
    queue_wait_ticks = wire_span_ticks - serialization
    propagation_ticks = last_frame.landed_ticks - last_frame.end_ticks
    setup_span_ticks = request.ready_ticks - request.request_ticks
    setup_wait_ticks = setup_span_ticks - request.setup_ticks
    total_delay_ticks = last_frame.landed_ticks - request.request_ticks
    return transfer_records.Transfer(
        payload_bits=request.framed.payload_bits,
        header_bits=request.framed.header_bits,
        request_ticks=request.request_ticks,
        setup_wait_ticks=setup_wait_ticks,
        setup_ticks=request.setup_ticks,
        send_ticks=request.ready_ticks,
        queue_wait_ticks=queue_wait_ticks,
        serialization_ticks=serialization,
        propagation_ticks=propagation_ticks,
        serializer_start_ticks=first_frame.start_ticks,
        serializer_end_ticks=last_frame.end_ticks,
        delivery_ticks=last_frame.landed_ticks,
        total_delay_ticks=total_delay_ticks,
        physical_sequence=transfer_sequence,
    )


def frame_records(
    channel_name: str, transfer_sequence: int, timings: tuple
) -> tuple[FrameRecord, ...]:
    """One record per frame of one transfer, in sending order."""
    records = []
    for frame_index, timing in enumerate(timings):
        record = FrameRecord(
            channel_name, transfer_sequence, frame_index, timing
        )
        records.append(record)
    return tuple(records)


def serialization_ticks(
    wire_bits: int, capacity: link_settings.CapacitySettings
) -> int:
    """Whole ticks to put the bits on the wire at the channel's rate.

    A fractional tick rounds up: serialization never ends before the exact
    transmission time. Exact Fraction arithmetic keeps the card's rate
    exact, so a whole-tick duration is never inflated by float error.
    """
    rate = capacity.input_bits_per_microsecond
    bits = fractions.Fraction(wire_bits)
    bits_times_ticks = bits * config.TICKS_PER_MICROSECOND
    exact_ticks = bits_times_ticks / rate
    return math.ceil(exact_ticks)


@dataclasses.dataclass(frozen=True)
class _Request:
    """One send waiting for its setup, then for the wire."""

    framed: FramedPayload
    request_ticks: int
    setup_ticks: int
    ready_ticks: int
    on_delivered: OnDelivered


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event a channel reports, as one member."""

    frame_landed: trace_source.TraceSource = trace_source.new_source()
