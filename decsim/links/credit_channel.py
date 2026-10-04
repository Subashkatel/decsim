"""A channel whose frames need a credit from a finite receive buffer.

The packet protocol `credit`. A message is cut into frames by its framing
row; each frame takes the wire when the wire is free and the sender
holds a credit, one frame at a time in order; the receive buffer holds
receive_buffer_frames frames, the receiver takes each frame as it
lands, and the frame's credit is back at the sender credit_latency
later. So frame k starts at

    s_k = max(ready, e_(k-1), c_(k-C)),   e_k = s_k + ceil(w_k / R),
    c_k = e_k + L + L_c,

and the message is delivered when its last frame lands, e_n + L. This
is Garnet's credit loop: an output virtual channel starts with
buffers_per_data_vc credits (gem5 src/mem/ruby/network/garnet/
OutVcState.cc:56-63), a flit is sent only while it has one and takes one
(NetworkInterface.cc:506, 530), and the receiver returns the credit when
the flit leaves its buffer (InputUnit.cc:140-150, and at once in a
network interface, NetworkInterface.cc:236-274), over a credit link of
its own latency (NetworkLink.cc:92-102). The credits travel on backward
flow-control links of their own, a CreditLink beside each forward link
(GarnetLink.py:60-63, 84-88, 126-142), so a credit takes no time on the
data wire and is never lost, which is how this row returns it.
Aurora's native flow control, where a receiver asks its partner to send
idles within the time of 256 blocks (Xilinx SP011 sections 3.1 and 3.3,
pages 29-30), is this row with that bound as its credit latency: an
equivalence, not Aurora's own mechanism. PCIe's flow-control credits
and NVIDIA's real-time ring, whose producer reuses a slot only when its
flags are clear (cuda-quantum realtime/lib/daemon/dispatcher/
cudaq_realtime_api.cpp:360-364), have the same shape. With unbounded
credits and the whole framing the law is the ideal row's.

Garnet's own loop is this one with two cycles more than its link
latency, which a card that means Garnet writes into
credit_latency_cycles: the input unit schedules the credit link a cycle
after the flit leaves (InputUnit.cc:151), and a network interface
spends a credit the cycle after it lands, reading credits after it has
scheduled its output (NetworkInterface.cc:221 and 286-298). Garnet also
holds a virtual channel for a whole packet, freed only by the tail's
credit (NetworkInterface.cc:293-296 and 477); frames here share the
buffer's credits, so a message does not wait for the one before it to
drain.
"""

import collections
import copy
import dataclasses

import decsim.config as config
import decsim.engine
import decsim.links.channel as channel_module
import decsim.links.framings as framings
import decsim.links.settings as link_settings
import decsim.ports as ports


class CreditChannel(channel_module.Channel):
    """A channel whose frames wait for receive-buffer credits.

    The setup engine is the ideal row's; the wire is a CreditWire.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The framing, the buffer, the credit's return, and their clock.

        receive_buffer_frames is C, the frames the receiver can hold;
        credit_latency_cycles is L_c on clock, the card's clock domain,
        from the receiver taking a frame to its credit being usable at
        the sender. Every field is written, because each is a number of
        the hardware the card describes and carries its source.
        """

        framing: framings.FramingSettings
        receive_buffer_frames: int
        credit_latency_cycles: int
        clock: config.Clock

        def __post_init__(self) -> None:
            check_credit_fields(self)

        def build(
            self,
            channel_settings: link_settings.ChannelSettings,
            engine: decsim.engine.Engine,
        ) -> "CreditChannel":
            """A fresh channel whose protocol is these settings."""
            return CreditChannel(channel_settings, engine)

    def _new_wire(self) -> "CreditWire":
        """The wire this protocol serializes on."""
        return CreditWire(self._settings, self._settings.protocol)


class CreditWire:
    """One wire whose frames need a credit, returned L_c after landing."""

    def __init__(
        self,
        channel_settings: link_settings.ChannelSettings,
        protocol_settings,
    ):
        self._channel_settings = channel_settings
        self._framing = protocol_settings.framing.build()
        self._credit_latency_ticks = credit_latency_ticks(channel_settings)
        # the credit-return tick of each of the last C frames sent; the
        # oldest is the credit the next frame waits for
        self._credit_returns: collections.deque = collections.deque(
            maxlen=protocol_settings.receive_buffer_frames
        )
        self._free_ticks = 0
        self.crossing_count = 0

    @property
    def framing(self) -> ports.Framing:
        """How the wire cuts a message into frames."""
        return self._framing

    def copy(self) -> "CreditWire":
        """The same wire in the same state, to price on."""
        copied = copy.copy(self)
        credit_returns = collections.deque(
            self._credit_returns, maxlen=self._credit_returns.maxlen
        )
        copied._credit_returns = credit_returns
        return copied

    def cross(
        self, framed: channel_module.FramedPayload, ready_ticks: int
    ) -> tuple[channel_module.FrameTiming, ...]:
        """Send the message's frames in order, each when it has a credit."""
        self.crossing_count += 1
        frame_bits = self._framing.frames(
            framed.payload_bits, framed.header_bits
        )
        timings = []
        for bits in frame_bits:
            timing = self.cross_frame(bits, ready_ticks)
            timings.append(timing)
        return tuple(timings)

    def next_start_ticks(self, ready_ticks: int) -> int:
        """When a frame ready at ready_ticks could take the wire."""
        queue_ticks = max(ready_ticks, self._free_ticks)
        credit_ticks = self._credit_ticks()
        return max(queue_ticks, credit_ticks)

    def cross_frame(
        self, bits: int, ready_ticks: int
    ) -> channel_module.FrameTiming:
        """Send one frame: wait for the wire and a credit, then serialize."""
        queue_ticks = max(ready_ticks, self._free_ticks)
        start_ticks = self.next_start_ticks(ready_ticks)
        capacity = self._channel_settings.capacity
        serialization = channel_module.serialization_ticks(bits, capacity)
        end_ticks = start_ticks + serialization
        propagation = self._channel_settings.propagation_latency_ticks
        landed_ticks = end_ticks + propagation
        credit_return_ticks = landed_ticks + self._credit_latency_ticks
        self._credit_returns.append(credit_return_ticks)
        self._free_ticks = end_ticks
        credit_wait_ticks = start_ticks - queue_ticks
        return channel_module.FrameTiming(
            bits,
            credit_wait_ticks,
            start_ticks,
            end_ticks,
            landed_ticks,
        )

    def _credit_ticks(self) -> int:
        """When the sender holds a credit: at once, or when the oldest returns.

        Credits return in the order their frames were sent, because every
        frame lands one propagation after its end and its credit one
        credit latency after that.
        """
        credits_out = len(self._credit_returns)
        if credits_out < self._credit_returns.maxlen:
            return 0
        return self._credit_returns[0]


def check_credit_fields(protocol_settings) -> None:
    """The credit loop's latency, which the reliable protocol shares.

    A credit back before its frame lands would let more than C frames
    fly.
    """
    config.check_cycles(
        "credit_latency_cycles", protocol_settings.credit_latency_cycles
    )


def credit_latency_ticks(
    channel_settings: link_settings.ChannelSettings,
) -> int:
    """L_c in ticks: the protocol's cycles on the card's clock."""
    protocol = channel_settings.protocol
    cycles = protocol.credit_latency_cycles
    return cycles * protocol.clock.period_ticks
