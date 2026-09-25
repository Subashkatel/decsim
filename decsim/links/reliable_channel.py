"""A credit channel that loses frames and recovers them by go-back-N.

The protocol row `reliable`: the credit row's wire and framing, with
every frame a packet of a reliable connection, numbered by a packet
sequence number (PSN), and the recovery Linux's soft-RoCE driver runs,
transcribed step by step:

- the requester stops sending when window_packets packets are
  unacknowledged (rxe_req.c:712-717, RXE_MAX_UNACKED_PSNS);
- a packet asks for an acknowledgement when it ends its message or when
  it is the ack_every_packets-th since the last packet that asked, the
  counter rxe keeps as noack_pkts (rxe_req.c:448-451); rxe asks on the
  packet that finds noack_pkts above RXE_MAX_PKT_PER_ACK, 64, so rxe is
  ack_every_packets 66;
- a frame fails its CRC with probability 1 - (1 - BER)^bits and the
  receiver drops it (rxe_hdr.h:55, the invariant CRC), an
  acknowledgement as well as a data packet;
- the responder takes the PSN it expects; a PSN ahead of it is dropped
  with one sequence NAK carrying the expected PSN, and no second NAK
  until the gap closes; a PSN behind it is a duplicate, answered with
  an ACK of the last PSN taken, so a lost ACK is made good when the
  timer resends (rxe_resp.c:70-95, 1257-1288, 1563-1568; IBA C9-105);
- an ACK or a NAK is a packet of its own, built by prepare_ack_packet
  and sent by rxe_xmit_packet (rxe_resp.c:763-810, 1169-1191): a
  RoCE v2 RC_ACKNOWLEDGE frame (framings.RoceV2.acknowledgement_bits)
  on the channel's reverse direction, serialized at the card's rate
  behind the ACKs before it, one propagation long, with the forward
  direction's credit loop, and lost by the same law;
- an ACK acknowledges its PSN and every one before; a NAK acknowledges
  every PSN before its own and sends the requester back to it, unless a
  retry started by a NAK is already under way (rxe_comp.c:303-322,
  720-790);
- the retransmit timer starts when a packet is sent and none is
  running, restarts at every acknowledgement while packets are out, and
  on expiry sends the requester back to the first unacknowledged PSN
  (rxe_req.c:588-590, 684-692; rxe_comp.c:616-636), except at a NAK
  that starts a retry, which leaves it running (rxe_comp.c:668-669);
  it stops when every packet is acknowledged, where rxe leaves it
  pending with a deadline a later send inherits; the retries never run
  out, where rxe gives up after retry_cnt, at most 7, timeouts.

Going back resends from the first unacknowledged PSN in order,
qp->req.psn = qp->comp.psn (rxe_req.c:38-53), so every message is
delivered once, in order, when its last packet is taken. Frames that
are lost still took the wire and a credit, and their credit returns as
any other: the receiver's buffer took them before the CRC failed. The
row is RoCE's, so its frames are roce_v2 frames: the ACK is a RoCE
packet, and no other framing has one. rxe keeps its send queue as work
requests, a message each here, and this row keeps PSNs: the completer's
checks of the oldest work request's state (get_wqe, check_psn,
rxe_comp.c:137-213) are read as PSN comparisons.
Linux's rxe itself needs a kernel module; the reference this row is
checked against is a transcription of the lines above.
"""

import dataclasses
import math
from collections.abc import Mapping
from typing import Optional

import decsim.config as config
import decsim.engine
import decsim.links.channel as channel_module
import decsim.links.credit_channel as credit_channel
import decsim.links.settings as link_settings
import decsim.seeding as seeding


class ReliableChannel(channel_module.Channel, seeding._RandomSeedConsumer):
    """A channel whose lost frames are resent by go-back-N.

    The setup engine is the ideal row's and the wire the credit row's;
    the requester and the responder here decide what the wire carries.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The reliable row's keys: the credit row's, then the connection's.

        window_packets is the most unacknowledged packets in flight;
        ack_every_packets bounds the packets between two acknowledgement
        requests; retransmit_timeout_cycles is on the card's clock;
        bit_error_rate is the probability that one wire bit is wrong.
        """

        framing: link_settings.FramingSettings
        receive_buffer_frames: int
        credit_latency_cycles: int
        window_packets: int
        ack_every_packets: int
        retransmit_timeout_cycles: int
        bit_error_rate: float

        @classmethod
        def from_yaml(
            cls, section: Mapping, path_name: str
        ) -> "ReliableChannel.Settings":
            """The keys of a card's protocol mapping."""
            credit_keys = credit_channel.read_credit_keys(section, path_name)
            framing = credit_keys[0]
            _require_roce_framing(framing, path_name)
            connection_keys = _read_connection_keys(section, path_name)
            return cls(*credit_keys, *connection_keys)

    def __init__(
        self,
        channel_settings: link_settings.ChannelSettings,
        engine: decsim.engine.Engine,
    ):
        channel_module.Channel.__init__(self, channel_settings, engine)
        self._initialize_run_seed_state(None)
        self._connection = channel_settings.protocol.row_settings
        self._sending = _SendState()
        self._receiving = _ReceiveState()
        self._timer = _TimerState()
        # the reverse direction: the responder's ACKs to the requester
        self._reverse_wire = self._new_wire()
        framing = self._reverse_wire.framing
        self._acknowledgement_bits = framing.acknowledgement_bits()

    def _new_wire(self) -> credit_channel.CreditWire:
        """The credit row's wire."""
        protocol = self._settings.protocol.row_settings
        return credit_channel.CreditWire(self._settings, protocol)

    def _wire_as_it_stands(self) -> credit_channel.CreditWire:
        """The wire once every packet queued now has crossed, none lost."""
        wire = self._wire.copy()
        now_ticks = self._engine.now
        packets = self._sending.packets
        for psn in range(self._sending.next_psn, len(packets)):
            packet = packets[psn]
            wire.cross_frame(packet.bits, now_ticks)
        return wire

    def _take_wire(self, request) -> None:
        """At the ready tick: number the message's frames, then send."""
        sending = self._sending
        framing = self._wire.framing
        frame_bits = framing.frames(
            request.framed.payload_bits, request.framed.header_bits
        )
        message = _Message(request, sending.message_count)
        sending.message_count += 1
        last_index = len(frame_bits) - 1
        for frame_index, bits in enumerate(frame_bits):
            psn = len(sending.packets)
            is_last = frame_index == last_index
            packet = _Packet(psn, bits, message, frame_index, is_last)
            sending.packets.append(packet)
        self._send_next()

    # ---- the requester

    def _send_next(self) -> None:
        """Send the next packet when the window and a credit allow."""
        sending = self._sending
        if sending.next_psn >= len(sending.packets):
            return
        window = self._connection.window_packets
        if sending.next_psn - sending.unacked_psn >= window:
            return
        now_ticks = self._engine.now
        start_ticks = self._wire.next_start_ticks(now_ticks)
        if start_ticks > now_ticks:
            self._wake_at(start_ticks)
            return
        packet = sending.packets[sending.next_psn]
        sending.next_psn += 1
        self._transmit(packet, now_ticks)

    def _transmit(self, packet: "_Packet", now_ticks: int) -> None:
        """Serialize one packet and schedule its landing."""
        timing = self._wire.cross_frame(packet.bits, now_ticks)
        message = packet.message
        timing_index = len(message.timings)
        message.timings.append(timing)
        is_lost = self._draw_loss(packet.bits)
        requests_ack = self._requests_ack(packet)
        is_retransmission = packet.transmission_count > 0
        packet.transmission_count += 1
        transmission = _Transmission(
            packet,
            timing,
            timing_index,
            is_lost,
            requests_ack,
            is_retransmission,
        )
        self._arm_timer_if_idle()
        landing_delay = timing.landed_ticks - now_ticks
        self._engine.schedule(
            landing_delay,
            lambda: self._land(transmission),
            label="link frame landing",
        )
        self._wake_at(timing.end_ticks)

    def _requests_ack(self, packet: "_Packet") -> bool:
        """rxe_req.c:448-451: the message's end, or too many without one."""
        sending = self._sending
        if packet.is_last:
            sending.packets_since_ack_request = 0
            return True
        ack_every = self._connection.ack_every_packets
        sending.packets_since_ack_request += 1
        if sending.packets_since_ack_request < ack_every:
            return False
        sending.packets_since_ack_request = 0
        return True

    def _draw_loss(self, bits: int) -> bool:
        """Whether a CRC fails: 1 - (1 - BER)^bits, drawn from the seed."""
        bit_error_rate = self._connection.bit_error_rate
        if bit_error_rate == 0:
            return False
        self._mark_stochastic_use()
        log_bit_survival = math.log1p(-bit_error_rate)
        log_frame_survival = bits * log_bit_survival
        loss_probability = -math.expm1(log_frame_survival)
        draw = self._rng.random()
        return draw < loss_probability

    def _acknowledged(self, is_nak: bool, psn: int) -> None:
        """An ACK or a NAK reaches the requester (rxe_comp.c:303-322).

        A NAK that starts a retry leaves the timer running: rxe's
        completer goes to done with need_retry set and reaches
        reset_retry_timer only on a later pass (rxe_comp.c:668-669,
        743-749). Any other answer restarts it while packets are out.
        """
        sending = self._sending
        if not is_nak and psn >= sending.unacked_psn:
            sending.unacked_psn = psn + 1
            sending.is_retry_started = False
        if is_nak and psn > sending.unacked_psn:
            sending.unacked_psn = psn
        is_retry = is_nak and self._go_back(is_timeout=False)
        if not is_retry:
            self._restart_timer_if_out()
        self._send_next()

    def _go_back(self, is_timeout: bool) -> bool:
        """Resend from the first unacknowledged PSN (rxe_req.c:38-53).

        A NAK does not start a second retry while one is under way; a
        timeout always does (rxe_comp.c:760-785). Whether it went back.
        """
        sending = self._sending
        if sending.is_retry_started and not is_timeout:
            return False
        if sending.next_psn <= sending.unacked_psn:
            return False
        sending.next_psn = sending.unacked_psn
        sending.is_retry_started = True
        return True

    # ---- the retransmit timer

    def _arm_timer_if_idle(self) -> None:
        """rxe_req.c:588-590: a send starts the timer when none is running."""
        if self._timer.expiry is not None:
            return
        self._start_timer()

    def _restart_timer_if_out(self) -> None:
        """rxe_comp.c:624-636: restart while packets are unacknowledged.

        With every packet acknowledged the timer stops. rxe leaves it
        pending instead, so a send soon after inherits the old deadline
        and can time out before its own round trip; stopping makes the
        timer's deadline one full timeout after the last acknowledgement
        or the first send, whatever order the two take at one tick.
        """
        sending = self._sending
        if sending.next_psn > sending.unacked_psn:
            self._start_timer()
            return
        self._stop_timer()

    def _start_timer(self) -> None:
        """Start the timer afresh from now."""
        self._stop_timer()
        timeout_cycles = self._connection.retransmit_timeout_cycles
        clock = self._settings.protocol.clock
        timeout_ticks = timeout_cycles * clock.period_ticks
        self._timer.expiry = self._engine.schedule(
            timeout_ticks, self._timer_expired, label="link retransmit timer"
        )

    def _stop_timer(self) -> None:
        expiry = self._timer.expiry
        if expiry is None:
            return
        self._engine.deschedule(expiry)
        self._timer.expiry = None

    def _timer_expired(self) -> None:
        """The expiry sends the requester back, then it sends again."""
        self._timer.expiry = None
        self._go_back(is_timeout=True)
        self._send_next()

    # ---- the responder

    def _land(self, transmission: "_Transmission") -> None:
        """A frame reaches the responder, which takes, drops or answers it."""
        self._report(transmission)
        if transmission.is_lost:
            return
        receiving = self._receiving
        psn = transmission.packet.psn
        if psn > receiving.expected_psn:
            self._refuse_out_of_sequence()
            return
        if psn < receiving.expected_psn:
            last_taken_psn = receiving.expected_psn - 1
            self._answer(is_nak=False, psn=last_taken_psn)
            return
        receiving.expected_psn += 1
        receiving.is_nak_sent = False
        if transmission.requests_ack:
            self._answer(is_nak=False, psn=psn)
        if transmission.packet.is_last:
            self._deliver_message(transmission)

    def _refuse_out_of_sequence(self) -> None:
        """One NAK with the expected PSN per gap (rxe_resp.c:76-84)."""
        receiving = self._receiving
        if receiving.is_nak_sent:
            return
        receiving.is_nak_sent = True
        self._answer(is_nak=True, psn=receiving.expected_psn)

    def _answer(self, is_nak: bool, psn: int) -> None:
        """Send an ACK or a NAK back on the reverse direction as a frame."""
        bits = self._acknowledgement_bits
        now_ticks = self._engine.now
        timing = self._reverse_wire.cross_frame(bits, now_ticks)
        is_lost = self._draw_loss(bits)
        if is_lost:
            return
        landing_delay = timing.landed_ticks - now_ticks
        self._engine.schedule(
            landing_delay,
            lambda: self._acknowledged(is_nak, psn),
            label="link acknowledgement",
        )

    def _deliver_message(self, transmission: "_Transmission") -> None:
        """The message's last packet is taken: the caller hears now."""
        message = transmission.packet.message
        timings = message.timings[: transmission.timing_index + 1]
        transfer = channel_module.transfer_for(
            message.request, timings, message.sequence
        )
        message.request.on_delivered(transfer)

    def _report(self, transmission: "_Transmission") -> None:
        """Every frame that lands is reported, lost or resent ones too."""
        if not self.trace.frame_landed.has_listeners:
            return
        packet = transmission.packet
        record = channel_module.FrameRecord(
            channel=self._settings.name,
            transfer_sequence=packet.message.sequence,
            frame_index=packet.frame_index,
            timing=transmission.timing,
            is_lost=transmission.is_lost,
            is_retransmission=transmission.is_retransmission,
        )
        self.trace.frame_landed.fire(record)

    def _wake_at(self, ticks: int) -> None:
        """Try the next send at a tick, once however many ask for it."""
        sending = self._sending
        is_earlier_wake_due = sending.wake_ticks is not None
        if is_earlier_wake_due and sending.wake_ticks <= ticks:
            return
        sending.wake_ticks = ticks
        delay_ticks = ticks - self._engine.now
        self._engine.schedule(delay_ticks, self._wake, label="link transmitter")

    def _wake(self) -> None:
        """A scheduled try: clear the mark, then send if allowed."""
        self._sending.wake_ticks = None
        self._send_next()


@dataclasses.dataclass
class _Message:
    """One transfer as the requester numbers it, and its transmissions."""

    request: object
    sequence: int
    timings: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class _Packet:
    """One frame of a message under its PSN."""

    psn: int
    bits: int
    message: _Message
    frame_index: int
    is_last: bool
    transmission_count: int = 0


@dataclasses.dataclass(frozen=True)
class _Transmission:
    """One trip of one packet across the wire."""

    packet: _Packet
    timing: channel_module.FrameTiming
    timing_index: int
    is_lost: bool
    requests_ack: bool
    is_retransmission: bool


@dataclasses.dataclass
class _SendState:
    """The requester's side: rxe's req.psn, comp.psn and counters."""

    packets: list = dataclasses.field(default_factory=list)
    next_psn: int = 0
    unacked_psn: int = 0
    is_retry_started: bool = False
    packets_since_ack_request: int = 0
    message_count: int = 0
    wake_ticks: Optional[int] = None


@dataclasses.dataclass
class _ReceiveState:
    """The responder's side: rxe's resp.psn and sent_psn_nak."""

    expected_psn: int = 0
    is_nak_sent: bool = False


@dataclasses.dataclass
class _TimerState:
    """The retransmit timer: its queued expiry, or None when stopped."""

    expiry: Optional[decsim.engine.Event] = None


def _require_roce_framing(
    framing: link_settings.FramingSettings, path_name: str
) -> None:
    """This row is RoCE's go-back-N, whose ACKs are RoCE packets.

    PCIe recovers by its data link layer's replay, which needs the base
    specification, not in hand; flits, Aurora blocks and UDP datagrams
    have no acknowledgement packet of their own. Those run on the credit
    row.
    """
    if framing.kind == "roce_v2":
        return
    raise ValueError(
        f"links.{path_name}.protocol runs the reliable row on "
        f"{framing.kind} frames; the row is RoCE's go-back-N and its "
        f"acknowledgements are RoCE packets, so it runs on roce_v2 "
        f"frames, and {framing.kind} runs on the credit row"
    )


def _read_connection_keys(section: Mapping, path_name: str) -> tuple:
    """(window, ack interval, timeout cycles, bit error rate) of a card."""
    section_name = f"links.{path_name}.protocol"
    window_packets = _positive_key(section, "window_packets", section_name)
    ack_every_packets = _positive_key(
        section, "ack_every_packets", section_name
    )
    timeout_cycles = _required(
        section, "retransmit_timeout_cycles", section_name
    )
    timeout_name = f"{section_name}.retransmit_timeout_cycles"
    config.check_cycles(timeout_name, timeout_cycles)
    if timeout_cycles == 0:
        raise ValueError(f"{timeout_name} must be positive")
    bit_error_rate = _bit_error_rate(section, section_name)
    return window_packets, ack_every_packets, timeout_cycles, bit_error_rate


def _bit_error_rate(section: Mapping, section_name: str) -> float:
    """A probability below one: a wire that always errs never delivers."""
    value = _required(section, "bit_error_rate", section_name)
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if is_number and 0 <= value < 1:
        return float(value)
    raise ValueError(
        f"{section_name}.bit_error_rate is {value!r}; it is the probability "
        f"that one bit is wrong, at least 0 and below 1"
    )


def _positive_key(section: Mapping, key: str, section_name: str) -> int:
    """A key the connection needs: a positive whole number."""
    value = _required(section, key, section_name)
    is_whole = isinstance(value, int) and not isinstance(value, bool)
    if is_whole and value > 0:
        return value
    raise ValueError(
        f"{section_name}.{key} is {value!r}; it is a positive whole number"
    )


def _required(section: Mapping, key: str, section_name: str):
    """A key the protocol needs, refused by name when missing."""
    if key not in section:
        raise ValueError(f"{section_name} needs {key}")
    return section[key]
