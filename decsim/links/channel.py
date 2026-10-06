"""One physical channel: a setup engine and a wire, driven by events.

The setup engine programs one transfer at a time in request order, as
gem5's DmaPort keeps one transmitList (src/dev/dma_device.cc) and the
DMA engine of Shao et al. (MICRO 2016, section III.C) services
descriptors one by one. A transfer with no setup is ready at its request
and never waits for another path's setup. The wire takes ready transfers
in order, serializes each with its path's header at the channel's rate,
and delivers it one propagation later, as ns-3's point-to-point device
does (point-to-point-net-device.cc). A fractional tick of serialization
rounds to the nearest tick, as ns-3's Time does. An unbounded channel
serializes nothing and never queues.

This is the protocol row `ideal`: one whole frame, an unbounded receive
buffer and nothing lost, the lossless reference of gem5's SimpleNetwork
with infinite buffers (src/mem/ruby/network/simple/SimpleNetwork.py) and
ns-3's device with no error model. The packet rows keep the setup engine
and replace the wire.
"""

import collections
import copy
import dataclasses
import fractions
import math
from collections.abc import Callable

import decsim.config as config
import decsim.engine
import decsim.links.settings as link_settings
import decsim.records.transfers as transfer_records
import decsim.trace_source as trace_source

OnDelivered = Callable[[transfer_records.Transfer], None]
HALF_TICK = fractions.Fraction(1, 2)


class Channel:
    """One channel at run time: a setup engine, then a whole-frame wire."""

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
        framed: transfer_records.FramedPayload,
        now_ticks: int,
        setup_ticks: int,
        on_delivered: OnDelivered,
    ) -> None:
        """Send one framed payload; on_delivered(transfer) runs at delivery."""
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
        self,
        framed: transfer_records.FramedPayload,
        now_ticks: int,
        setup_ticks: int,
    ) -> int:
        """The delay a transfer would pay if nothing else reached the channel.

        A scheduler's estimate: exact on the ideal and credit rows when no later
        request overtakes it; a lower bound on the reliable row, which leaves
        out the wait for the acknowledgement window.
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
        self, framed: transfer_records.FramedPayload, ready_ticks: int
    ) -> tuple[transfer_records.FrameTiming, ...]:
        """Take the wire's next slot for the whole transfer."""
        self.crossing_count += 1
        if self._capacity is None:
            landed_ticks = ready_ticks + self._propagation_ticks
            timing = transfer_records.FrameTiming(
                None, 0, ready_ticks, ready_ticks, landed_ticks
            )
            return (timing,)
        start_ticks = max(ready_ticks, self._free_ticks)
        wire_bits = framed.payload_bits + framed.header_bits
        serialization = serialization_ticks(wire_bits, self._capacity)
        end_ticks = start_ticks + serialization
        self._free_ticks = end_ticks
        landed_ticks = end_ticks + self._propagation_ticks
        timing = transfer_records.FrameTiming(
            wire_bits, 0, start_ticks, end_ticks, landed_ticks
        )
        return (timing,)


def transfer_for(
    request: "_Request", timings: tuple, transfer_sequence: int
) -> transfer_records.Transfer:
    """The transfer's record: its frames' span on the wire, as one move.

    serialization_ticks is the time its frames held the wire, and
    queue_wait_ticks everything else from the ready tick to the last frame's
    end (queue, credit waits, loss recovery), so the parts still sum.
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
) -> tuple[transfer_records.FrameRecord, ...]:
    """One record per frame of one transfer, in sending order."""
    records = []
    for frame_index, timing in enumerate(timings):
        record = transfer_records.FrameRecord(
            channel_name, transfer_sequence, frame_index, timing
        )
        records.append(record)
    return tuple(records)


def serialization_ticks(
    wire_bits: int, capacity: link_settings.CapacitySettings
) -> int:
    """Whole ticks to put the bits on the wire at the channel's rate.

    The time rounds to the nearest tick and a half tick rounds up, as
    ns-3's Time does (src/core/model/nstime.h lines 237-238 and
    int64x64-128.h lines 267-281, Round). Fraction arithmetic keeps a
    whole-tick duration from float inflation.
    """
    rate = capacity.input_bits_per_microsecond
    bits = fractions.Fraction(wire_bits)
    bits_times_ticks = bits * config.TICKS_PER_MICROSECOND
    exact_ticks = bits_times_ticks / rate
    half_up_ticks = exact_ticks + HALF_TICK
    return math.floor(half_up_ticks)


@dataclasses.dataclass(frozen=True)
class _Request:
    """One send waiting for its setup, then for the wire."""

    framed: transfer_records.FramedPayload
    request_ticks: int
    setup_ticks: int
    ready_ticks: int
    on_delivered: OnDelivered


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event a channel reports, as one member."""

    frame_landed: trace_source.TraceSource = trace_source.new_source()
