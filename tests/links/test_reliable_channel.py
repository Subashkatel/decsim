"""A reliable channel recovers every lost frame by go-back-N, in order.

Sources: Linux drivers/infiniband/sw/rxe, rxe_req.c:38-53 (a retry goes
back to the first unacknowledged PSN), 448-451 (the acknowledgement
request), 588-590 and 684-717 (the timer, the retry, the window);
rxe_resp.c:70-95 and 1563-1568 (one sequence NAK per gap), 1257-1288 (a
duplicate re-acknowledged), 763-810 and 1169-1191 (an ACK is a packet
on the wire); rxe_comp.c:303-322 and 720-790 (a NAK acknowledges what
is before it; one retry at a time unless the timer fires). The
hand-traced timelines below follow those lines; with nothing lost the
row is the credit row (tests/links/test_credit_channel.py).

The frames are RoCE v2 writes at a 256-byte path MTU on a wire of one
byte per tick: a message's first packet is 38 + 20 + 8 + 12 + 16 + 256
+ 4 = 354 ticks, a later full one 338, an ACK 38 + 20 + 8 + 12 + 4 + 4
= 86 (tests/links/test_framings.py).
"""

import math
import random

import pytest

import decsim.config as config
import decsim.engine
import decsim.links.channel as channel_module
import decsim.links.credit_channel as credit_channel
import decsim.links.framings as framings
import decsim.links.reliable_channel as reliable_channel
import decsim.links.settings as link_settings

PATH_MTU_BYTES = 256
RATE = 8_000_000  # bits per microsecond: one byte a tick
FIRST_PACKET_TICKS = 354
LATER_PACKET_TICKS = 338
ACK_TICKS = 86
PROPAGATION = 300
CLOCK = config.Clock(50)
TIMEOUT_TICKS = 200 * 50
# a bit error rate that loses an ACK with probability 0.098 and a first
# packet with 0.346, so one draw below the first loses any frame and one
# above the second keeps any
BIT_ERROR_RATE = 0.00015


def reliable_settings(
    bit_error_rate,
    window_packets=16,
    buffer_frames=16,
    ack_every_packets=66,
):
    roce_settings = framings.RoceV2.Settings(PATH_MTU_BYTES)
    framing_settings = link_settings.FramingSettings("roce_v2", roce_settings)
    row = reliable_channel.ReliableChannel.Settings(
        framing=framing_settings,
        receive_buffer_frames=buffer_frames,
        credit_latency_cycles=2,
        window_packets=window_packets,
        ack_every_packets=ack_every_packets,
        retransmit_timeout_cycles=200,
        bit_error_rate=bit_error_rate,
    )
    protocol = link_settings.ProtocolSettings("reliable", row, CLOCK)
    capacity = link_settings.CapacitySettings(RATE, "test")
    return link_settings.ChannelSettings(
        "test", PROPAGATION, capacity, "test", protocol
    )


def seeded_channel(engine, settings, seed):
    channel = reliable_channel.ReliableChannel(settings, engine)
    reservation = channel.reserve_run_seed(seed)
    channel.commit_run_seed(reservation)
    return channel


def loss_probability(frame_ticks):
    """1 - (1 - BER)^bits for a frame of that many one-byte ticks."""
    bits = frame_ticks * 8
    log_bit_survival = math.log1p(-BIT_ERROR_RATE)
    log_frame_survival = bits * log_bit_survival
    return -math.expm1(log_frame_survival)


def seed_losing(pattern):
    """The first seed whose draws lose exactly the pattern's frames.

    The pattern is one flag per frame in the order the channel draws,
    data packets and ACKs alike. A draw below the ACK's loss probability
    loses any frame, and one above the first packet's keeps any.
    """
    loses_any_below = loss_probability(ACK_TICKS)
    keeps_any_above = loss_probability(FIRST_PACKET_TICKS)
    for seed in range(2_000_000):
        generator = random.Random(seed)
        draws = [generator.random() for _ in pattern]
        pairs = list(zip(draws, pattern))
        lost_draws = [draw for draw, is_lost in pairs if is_lost]
        kept_draws = [draw for draw, is_lost in pairs if not is_lost]
        highest_lost = max(lost_draws, default=0.0)
        lowest_kept = min(kept_draws, default=1.0)
        if highest_lost < loses_any_below and lowest_kept > keeps_any_above:
            return seed
    raise AssertionError("no seed draws the pattern")


def send_at(engine, channel, tick, payload_bytes, delivered):
    delay = tick - engine.now
    payload_bits = payload_bytes * 8
    framed = channel_module.FramedPayload(payload_bits)

    def send():
        channel.send(framed, tick, 0, delivered.append)

    engine.schedule(delay, send)


def test_a_lost_frame_is_resent_after_the_sequence_nak_hand_traced():
    """p1 is lost; p2 lands at 1330 and sends NAK(1), back at 1716.

    p0 0-354, p1 354-692 (lost), p2 692-1030, p3 1030-1368; the NAK takes
    the reverse wire 1330-1416 and lands at 1716, when the requester
    goes back to p1: 1716-2054, 2054-2392, 2392-2730, and p3 lands at
    3030, when the message is delivered. The draws: four packets, the
    NAK, three packets, the ACK.
    """
    pattern = (False, True, False, False, False, False, False, False, False)
    seed = seed_losing(pattern)
    engine = decsim.engine.Engine()
    settings = reliable_settings(BIT_ERROR_RATE)
    channel = seeded_channel(engine, settings, seed)
    frames = []
    channel.trace.frame_landed.connect(frames.append)
    delivered = []
    send_at(engine, channel, 0, 1024, delivered)

    engine.run()

    starts = [record.timing.start_ticks for record in frames]
    lost = [record.is_lost for record in frames]
    resent = [record.is_retransmission for record in frames]
    transfer = delivered[0]
    assert starts == [0, 354, 692, 1030, 1716, 2054, 2392]
    assert lost == [False, True, False, False, False, False, False]
    assert resent == [False] * 4 + [True] * 3
    assert transfer.delivery_ticks == 3030
    assert transfer.serialization_ticks == 354 + 6 * LATER_PACKET_TICKS
    assert transfer.queue_wait_ticks == 2730 - 354 - 6 * LATER_PACKET_TICKS


def test_a_nak_that_starts_a_retry_leaves_the_timer_running():
    """The timer set at the first send still expires at 10_000.

    rxe's completer leaves a retry started by a NAK without resetting
    the timer (rxe_comp.c:668-669 and 743-749). p1 is lost, NAK(1) comes
    back at 1716 and the resent p1 is lost too; p2 and p3 then arrive
    out of sequence, but one NAK per gap was sent already, so only the
    timer recovers: back to p1 at 10_000, p3 ends at 10_676 + 338 and
    lands one propagation on.
    """
    pattern = (False, True, False, False, False, True) + (False,) * 6
    seed = seed_losing(pattern)
    engine = decsim.engine.Engine()
    settings = reliable_settings(BIT_ERROR_RATE)
    channel = seeded_channel(engine, settings, seed)
    frames = []
    channel.trace.frame_landed.connect(frames.append)
    delivered = []
    send_at(engine, channel, 0, 1024, delivered)

    engine.run()

    starts = [record.timing.start_ticks for record in frames]
    transfer = delivered[0]
    assert starts[4:] == [1716, 2054, 2392, 10_000, 10_338, 10_676]
    assert transfer.delivery_ticks == 11_014 + PROPAGATION


def test_a_lost_last_frame_is_resent_when_the_timer_expires():
    """No later frame reports the gap, so the 10_000-tick timer does.

    A 200-byte message is one packet of 98 + 200 = 298 ticks.
    """
    seed = seed_losing((True, False, False))
    engine = decsim.engine.Engine()
    settings = reliable_settings(BIT_ERROR_RATE)
    channel = seeded_channel(engine, settings, seed)
    delivered = []
    send_at(engine, channel, 0, 200, delivered)

    engine.run()

    transfer = delivered[0]
    assert transfer.serializer_start_ticks == 0
    assert transfer.delivery_ticks == TIMEOUT_TICKS + 298 + PROPAGATION


def test_a_lost_ack_is_made_good_by_the_timer_and_a_duplicate_ack():
    """The duplicate is answered with an ACK, and delivered only once.

    p0 0-298 lands at 598 and is delivered; its ACK is lost. The timer
    resends p0 at 10_000; it lands at 10_598 as a duplicate, which the
    responder answers with ACK(0) (rxe_resp.c:1274-1284, IBA C9-105),
    10_598-10_684 on the reverse wire, back at 10_984.
    """
    seed = seed_losing((False, True, False, False))
    engine = decsim.engine.Engine()
    settings = reliable_settings(BIT_ERROR_RATE)
    channel = seeded_channel(engine, settings, seed)
    frames = []
    channel.trace.frame_landed.connect(frames.append)
    delivered = []
    send_at(engine, channel, 0, 200, delivered)

    engine.run()

    starts = [record.timing.start_ticks for record in frames]
    assert [transfer.delivery_ticks for transfer in delivered] == [598]
    assert starts == [0, TIMEOUT_TICKS]
    assert engine.now == TIMEOUT_TICKS + 298 + PROPAGATION + 86 + PROPAGATION


def test_the_run_ends_at_the_last_acknowledgement_not_at_the_timer():
    """The timer stops when every packet is acknowledged.

    One packet: sent 0-298, lands 598, its ACK takes the reverse wire
    598-684 and is back at 984; the 10_000-tick timer never fires.
    """
    engine = decsim.engine.Engine()
    settings = reliable_settings(0.0)
    channel = seeded_channel(engine, settings, 1)
    delivered = []
    send_at(engine, channel, 0, 200, delivered)

    engine.run()

    assert delivered[0].delivery_ticks == 598
    assert engine.now == 984


def test_with_nothing_lost_the_reliable_row_is_the_credit_row_property():
    generator = random.Random(11)
    for _trace in range(60):
        count = generator.randrange(1, 20)
        arrivals = sorted(generator.randrange(0, 6000) for _ in range(count))
        sizes = [generator.choice([0, 50, 256, 700]) for _ in range(count)]
        engine = decsim.engine.Engine()
        settings = reliable_settings(0.0, window_packets=10_000)
        reliable = seeded_channel(engine, settings, 1)
        credit_row = credit_channel.CreditChannel.Settings(
            settings.protocol.row_settings.framing, 16, 2
        )
        credit_protocol = link_settings.ProtocolSettings(
            "credit", credit_row, CLOCK
        )
        credit_settings = link_settings.ChannelSettings(
            "test", PROPAGATION, settings.capacity, "test", credit_protocol
        )
        credit = credit_channel.CreditChannel(credit_settings, engine)
        by_reliable = []
        by_credit = []
        for arrival, payload in zip(arrivals, sizes):
            send_at(engine, reliable, arrival, payload, by_reliable)
            send_at(engine, credit, arrival, payload, by_credit)
        engine.run()
        reliable_deliveries = [item.delivery_ticks for item in by_reliable]
        credit_deliveries = [item.delivery_ticks for item in by_credit]
        assert reliable_deliveries == credit_deliveries


def test_every_message_arrives_once_in_order_under_loss_property():
    generator = random.Random(13)
    for trace in range(40):
        count = generator.randrange(1, 25)
        arrivals = sorted(generator.randrange(0, 20_000) for _ in range(count))
        sizes = [generator.randrange(1, 1200) for _ in range(count)]
        window = generator.choice([1, 2, 4, 128])
        buffer_frames = generator.choice([1, 3, 16])
        engine = decsim.engine.Engine()
        settings = reliable_settings(
            0.00003, window_packets=window, buffer_frames=buffer_frames
        )
        channel = seeded_channel(engine, settings, trace)
        delivered = []
        for arrival, payload in zip(arrivals, sizes):
            send_at(engine, channel, arrival, payload, delivered)
        engine.run()
        sequences = [transfer.physical_sequence for transfer in delivered]
        deliveries = [transfer.delivery_ticks for transfer in delivered]
        assert sequences == list(range(count))
        assert deliveries == sorted(deliveries)


def test_a_window_of_one_waits_for_each_acknowledgement():
    """Stop and wait: the next packet goes when the ACK is back.

    Each packet asks for an ACK; p0 0-354 lands at 654, its ACK takes
    the reverse wire 654-740 and is back at 1040; p1, 82 + 256 = 338
    ticks, goes 1040-1378 and lands at 1678.
    """
    engine = decsim.engine.Engine()
    settings = reliable_settings(0.0, window_packets=1, ack_every_packets=1)
    channel = seeded_channel(engine, settings, 1)
    delivered = []
    send_at(engine, channel, 0, 512, delivered)

    engine.run()

    assert delivered[0].delivery_ticks == 1678


def test_one_seed_draws_the_same_losses_twice():
    runs = []
    for _run in range(2):
        engine = decsim.engine.Engine()
        settings = reliable_settings(0.0001)
        channel = seeded_channel(engine, settings, 99)
        delivered = []
        send_at(engine, channel, 0, 4000, delivered)
        engine.run()
        runs.append(delivered)

    assert runs[0] == runs[1]


def _reliable_section(framing: dict, bit_error_rate: float) -> dict:
    return {
        "framing": framing,
        "receive_buffer_frames": 16,
        "credit_latency_cycles": 2,
        "window_packets": 128,
        "ack_every_packets": 66,
        "retransmit_timeout_cycles": 1000,
        "bit_error_rate": bit_error_rate,
    }


def test_a_bit_error_rate_of_one_is_refused():
    framing = {"kind": "roce_v2", "path_mtu_bytes": 1024}
    section = _reliable_section(framing, 1.0)

    with pytest.raises(ValueError, match="below 1"):
        reliable_channel.ReliableChannel.Settings.from_yaml(section, "path")


def test_a_reliable_card_on_frames_other_than_roce_v2_is_refused():
    """The ACK is a RoCE packet; flits have no acknowledgement of theirs."""
    framing = {"kind": "flits", "flit_bits": 64}
    section = _reliable_section(framing, 0.0)
    settings_class = reliable_channel.ReliableChannel.Settings

    with pytest.raises(ValueError, match="flits runs on the credit row"):
        settings_class.from_yaml(section, "p")
