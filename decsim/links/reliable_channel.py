"""A credit channel that loses frames and recovers them by go-back-N.

The packet protocol `reliable`: the credit protocol's wire and framing,
every frame a packet of a reliable connection numbered by a PSN, and the
recovery of Linux's soft-RoCE driver (rxe), transcribed step by step:

- the requester stops when window_packets packets are unacknowledged
  (rxe_req.c:712-717);
- a packet asks for an acknowledgement when it ends its message or is
  the ack_every_packets-th since the last that asked (noack_pkts,
  rxe_req.c:448-451); rxe's RXE_MAX_PKT_PER_ACK of 64 is
  ack_every_packets 66;
- a frame fails its CRC with probability 1 - (1 - BER)^bits and is
  dropped (rxe_hdr.h:55), an acknowledgement as well as a data packet;
- the responder takes the PSN it expects; one ahead is dropped with one
  sequence NAK, and no second until the gap closes; one behind is a
  duplicate, answered with an ACK of the last PSN taken
  (rxe_resp.c:70-95, 1257-1288, 1563-1568; IBA C9-105);
- an ACK or NAK is a RoCE v2 RC_ACKNOWLEDGE frame of its own
  (rxe_resp.c:763-810, 1169-1191) on the reverse direction, serialized
  behind the ACKs before it, one propagation long, lost by the same law;
- an ACK acknowledges its PSN and every one before; a NAK every PSN
  before its own, and sends the requester back to it unless a NAK retry
  is under way (rxe_comp.c:303-322, 720-790);
- the retransmit timer restarts at every pass of rxe's send task while
  packets are out, so at every send, message taken and answer
  (rxe_req.c:831-846; rxe_comp.c:161-162, 742-749, 624-636); on expiry
  the requester goes back to the first unacknowledged PSN
  (rxe_req.c:684-692). A NAK that starts a retry leaves it running
  (rxe_comp.c:668-669);
- each retry spends one of retry_count, refilled by every ACK that
  moves the first unacknowledged PSN; a retry with none left fails the
  run, naming the frame (rxe_comp.c:165-170, 752-792). A count of 7
  is never spent (rxe_comp.c:772-773);
- while nothing is out, an answer that arrives is dropped (get_wqe,
  rxe_comp.c:150-151).

Going back resends from the first unacknowledged PSN in order
(rxe_req.c:38-53), so every message is delivered once, in order. Lost
frames still took the wire and a credit, which returns as any other.
rxe keeps work requests and this row keeps PSNs: the two readings part
only when a NAK carries a PSN past the requester's next one. The
reference this row is checked against is a transcription of these lines,
since rxe itself needs a kernel module.
"""

import dataclasses
import math
from typing import Optional

import decsim.config as config
import decsim.engine
import decsim.links.channel as channel_module
import decsim.links.credit_channel as credit_channel
import decsim.links.settings as link_settings
import decsim.records.transfers as transfer_records
import decsim.seeding as seeding

# rxe's completer never spends a retry count of 7, the largest a card
# may write (rxe_comp.c:772-773), so a card of 7 retries for ever
UNSPENT_RETRY_COUNT = 7


class ReliableChannel(channel_module.Channel, seeding._RandomSeedConsumer):
    """A channel whose lost frames are resent by go-back-N.

    The setup engine is the ideal row's and the wire the credit row's;
    the requester and the responder here decide what the wire carries.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The credit protocol's fields, the connection's, and their clock.

        window_packets is the most unacknowledged packets in flight;
        ack_every_packets bounds the packets between acknowledgement requests;
        retransmit_timeout_cycles is on clock, the card's domain; retry_count is
        rxe's retry_cnt, 0 to 7; bit_error_rate is the probability one wire bit
        is wrong.
        """

        framing: link_settings.FramingSettings
        receive_buffer_frames: int
        credit_latency_cycles: int
        window_packets: int
        ack_every_packets: int
        retransmit_timeout_cycles: int
        retry_count: int
        bit_error_rate: float
        clock: config.Clock

        def __post_init__(self) -> None:
            credit_channel.check_credit_fields(self)
            _require_roce_framing(self.framing)
            link_settings.check_positive_count(
                "window_packets", self.window_packets
            )
            link_settings.check_positive_count(
                "ack_every_packets", self.ack_every_packets
            )
            _check_timeout_cycles(self.retransmit_timeout_cycles)
            _check_retry_count(self.retry_count)
            bit_error_rate = _bit_error_rate(self.bit_error_rate)
            object.__setattr__(self, "bit_error_rate", bit_error_rate)

        def build(
            self,
            channel_settings: link_settings.ChannelSettings,
            engine: decsim.engine.Engine,
        ) -> "ReliableChannel":
            """A fresh channel whose protocol is these settings."""
            return ReliableChannel(channel_settings, engine)

    def __init__(
        self,
        channel_settings: link_settings.ChannelSettings,
        engine: decsim.engine.Engine,
    ):
        channel_module.Channel.__init__(self, channel_settings, engine)
        seeding._RandomSeedConsumer.__init__(self, None)
        self._connection = channel_settings.protocol
        self._sending = _SendState(retries_left=self._connection.retry_count)
        self._receiving = _ReceiveState()
        self._timer = _TimerState()
        # the reverse direction: the responder's ACKs to the requester
        self._reverse_wire = self._new_wire()
        framing = self._reverse_wire.framing
        self._acknowledgement_bits = framing.acknowledgement_bits()

    def _new_wire(self) -> credit_channel.CreditWire:
        """The credit row's wire."""
        protocol = self._settings.protocol
        return credit_channel.CreditWire(self._settings, protocol)

    def _wire_as_it_stands(self) -> credit_channel.CreditWire:
        """The wire once every packet queued now has crossed, none lost.

        No packet waits here for an acknowledgement, so a message wider
        than the window lands later than this wire says.
        """
        wire = self._wire.copy()
        now_ticks = self._engine.now
        sending = self._sending
        for psn in range(sending.next_psn, sending.packet_count):
            packet = sending.packet_by_psn[psn]
            wire.cross_frame(packet.bits, now_ticks)
        return wire

    def _take_wire(self, request) -> None:
        """At the ready tick: number the message's frames, then send.

        Taking a message is a pass of rxe's send task, so it restarts
        the timer while packets are out, sent or not (rxe_comp.c:742-749).
        """
        sending = self._sending
        framing = self._wire.framing
        frame_bits = framing.frames(
            request.framed.payload_bits, request.framed.header_bits
        )
        message = _Message(request, sending.message_count)
        sending.message_count += 1
        last_index = len(frame_bits) - 1
        for frame_index, bits in enumerate(frame_bits):
            psn = sending.packet_count
            is_last = frame_index == last_index
            packet = _Packet(psn, bits, message, frame_index, is_last)
            sending.packet_by_psn[psn] = packet
            sending.packet_count += 1
        self._send_next()
        self._restart_timer_if_out()

    # ---- the requester

    def _send_next(self) -> None:
        """Send the next packet when the window and a credit allow."""
        sending = self._sending
        if sending.next_psn >= sending.packet_count:
            return
        window = self._connection.window_packets
        if sending.next_psn - sending.unacked_psn >= window:
            return
        now_ticks = self._engine.now
        start_ticks = self._wire.next_start_ticks(now_ticks)
        if start_ticks > now_ticks:
            self._wake_at(start_ticks)
            return
        packet = sending.packet_by_psn[sending.next_psn]
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
        # the completer pass that follows this send resets the timer
        # (rxe_req.c:837-840, rxe_comp.c:742-749)
        self._start_timer()
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

        A NAK that starts a retry leaves the timer running (rxe_comp.c:668-669,
        743-749); any other answer restarts it while packets are out.
        """
        sending = self._sending
        is_waiting_to_resend = sending.next_psn <= sending.unacked_psn
        if is_waiting_to_resend:
            self._restart_timer_if_out()
            return
        if not is_nak and psn >= sending.unacked_psn:
            sending.unacked_psn = psn + 1
            sending.is_retry_started = False
            sending.retries_left = self._connection.retry_count
        if is_nak and psn > sending.unacked_psn:
            sending.unacked_psn = psn
        is_retry = is_nak and self._error_retry(is_timeout=False)
        if not is_retry:
            self._restart_timer_if_out()
        self._send_next()
        self._retire_acknowledged()

    def _retire_acknowledged(self) -> None:
        """Let go of the packets no send can reach again.

        rxe retires an acknowledged work request (rxe_comp.c do_complete). The
        requester may still stand behind the first unacknowledged PSN after a
        NAK overtook a go-back, so what is kept starts at the lower of the two.
        """
        sending = self._sending
        first_needed_psn = min(sending.unacked_psn, sending.next_psn)
        for psn in range(sending.first_kept_psn, first_needed_psn):
            del sending.packet_by_psn[psn]
        sending.first_kept_psn = first_needed_psn

    def _error_retry(self, is_timeout: bool) -> bool:
        """Resend from the first unacknowledged PSN, or give up; True if resent.

        rxe's COMPST_ERROR_RETRY (rxe_comp.c:752-792): nothing out is nothing
        to retry; a NAK does not start a second retry, a timeout always does;
        each retry spends one of the count, and one with none left fails.
        """
        sending = self._sending
        if sending.next_psn <= sending.unacked_psn:
            return False
        if sending.is_retry_started and not is_timeout:
            return False
        if sending.retries_left == 0:
            self._give_up()
        if sending.retries_left != UNSPENT_RETRY_COUNT:
            sending.retries_left -= 1
        sending.next_psn = sending.unacked_psn
        sending.is_retry_started = True
        return True

    def _give_up(self) -> None:
        """IB_WC_RETRY_EXC_ERR: the link has failed, and the run stops."""
        sending = self._sending
        psn = sending.unacked_psn
        packet = sending.packet_by_psn[psn]
        retry_count = self._connection.retry_count
        raise RuntimeError(
            f"channel {self._settings.name!r} gave up on frame "
            f"{packet.frame_index} of transfer {packet.message.sequence} "
            f"(PSN {psn}) after {retry_count} retries without an "
            f"acknowledgement: the link failed"
        )

    # ---- the retransmit timer

    def _restart_timer_if_out(self) -> None:
        """Restart the timer while any packet is out (rxe_comp.c:624-636)."""
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
        self._error_retry(is_timeout=True)
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
        record = transfer_records.FrameRecord(
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
    timing: transfer_records.FrameTiming
    timing_index: int
    is_lost: bool
    requests_ack: bool
    is_retransmission: bool


@dataclasses.dataclass
class _SendState:
    """The requester's side: rxe's req.psn, comp.psn and counters."""

    retries_left: int
    # the packets a send can still reach, by PSN, from first_kept_psn
    packet_by_psn: dict = dataclasses.field(default_factory=dict)
    packet_count: int = 0
    first_kept_psn: int = 0
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


def _require_roce_framing(framing: link_settings.FramingSettings) -> None:
    """This protocol is RoCE's go-back-N, whose ACKs are RoCE packets.

    PCIe's data link replay needs the base specification, not in hand;
    flits, Aurora blocks and UDP datagrams have no acknowledgement packet.
    Those run on the credit protocol.
    """
    if framing.has_acknowledgement_packet:
        return
    raise ValueError(
        "the reliable protocol runs on frames with no acknowledgement "
        "packet of their own; it is RoCE's go-back-N and its "
        "acknowledgements are RoCE packets, so it runs on roce_v2 frames, "
        "and any other framing runs on the credit row"
    )


def _check_timeout_cycles(timeout_cycles: int) -> None:
    """A retransmit timer that fires at once would resend for ever."""
    name = "retransmit_timeout_cycles"
    config.check_cycles(name, timeout_cycles)
    if timeout_cycles > 0:
        return
    raise ValueError(f"{name} must be positive")


def _check_retry_count(retry_count) -> None:
    """Rxe's retry_cnt: a whole number, 0 to 7."""
    is_whole = isinstance(retry_count, int) and not isinstance(
        retry_count, bool
    )
    if is_whole and 0 <= retry_count <= UNSPENT_RETRY_COUNT:
        return
    raise ValueError(
        f"retry_count is {retry_count!r}; it is a whole number from 0 to "
        f"{UNSPENT_RETRY_COUNT}"
    )


def _bit_error_rate(value) -> float:
    """A probability below one: a wire that always errs never delivers."""
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if is_number and 0 <= value < 1:
        return float(value)
    raise ValueError(
        f"bit_error_rate is {value!r}; it is the probability that one bit "
        f"is wrong, at least 0 and below 1"
    )
