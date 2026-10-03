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

import dataclasses
import functools
import gc
import math
import random
import weakref

import pytest

import decsim.config as config
import decsim.engine
import decsim.links.channel as channel_module
import decsim.links.credit_channel as credit_channel
import decsim.links.framings as framings
import decsim.links.link_profiles as link_profiles
import decsim.links.reliable_channel as reliable_channel
import decsim.links.settings as link_settings
import decsim.machine as machine_module
import decsim.settings as machine_settings
import examples.two_tiers as two_tiers

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
    retry_count=7,
    credit_latency_cycles=2,
    timeout_cycles=200,
):
    framing_settings = framings.RoceV2.Settings(PATH_MTU_BYTES)
    protocol = reliable_channel.ReliableChannel.Settings(
        framing=framing_settings,
        receive_buffer_frames=buffer_frames,
        credit_latency_cycles=credit_latency_cycles,
        window_packets=window_packets,
        ack_every_packets=ack_every_packets,
        retransmit_timeout_cycles=timeout_cycles,
        retry_count=retry_count,
        bit_error_rate=bit_error_rate,
        clock=CLOCK,
    )
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
        pairs = list(zip(draws, pattern, strict=True))
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


def test_the_retransmit_timer_restarts_at_every_send():
    """The timer expires one timeout after the last resend, at 12_392.

    rxe's send task runs the completer after every requester pass
    (rxe_req.c:831-846); with no answer to read, the completer ends at
    COMPST_EXIT, which resets the timer while packets are out
    (rxe_comp.c:161-162, 742-749, 624-636). p1 is lost, NAK(1) comes
    back at 1716 and the resent p1 is lost too; p2 and p3 then arrive
    out of sequence, but one NAK per gap was sent already, so only the
    timer recovers. The resends at 1716, 2054 and 2392 each restart it:
    back to p1 at 2392 + 10_000, p3 ends at 13_068 + 338 and lands one
    propagation on.
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
    assert starts[4:] == [1716, 2054, 2392, 12_392, 12_730, 13_068]
    assert transfer.delivery_ticks == 13_406 + PROPAGATION


def test_a_message_taken_while_packets_are_out_restarts_the_timer():
    """A second message at 5000 moves the lost packet's timeout to 15_000.

    Posting a message runs rxe's send task, whose completer pass resets
    the timer while packets are out (rxe_comp.c:742-749, 624-636), even
    when the window keeps the requester from sending. A window of one:
    p0, 298 ticks, is lost; the message sent at 5000 waits behind it.
    The timer expires at 15_000, p0 is resent 15_000-15_298 and lands
    at 15_598.
    """
    seed = seed_losing((True, False, False, False, False))
    engine = decsim.engine.Engine()
    settings = reliable_settings(BIT_ERROR_RATE, window_packets=1)
    channel = seeded_channel(engine, settings, seed)
    delivered = []
    send_at(engine, channel, 0, 200, delivered)
    send_at(engine, channel, 5000, 200, delivered)

    engine.run()

    first_transfer = delivered[0]
    assert first_transfer.delivery_ticks == 15_598


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


def test_a_retry_with_none_left_fails_the_run_naming_the_frame():
    """A count of one: the first timeout retries, the second gives up.

    The lone packet is lost at 0 and again at 10_000, when the timer
    resent it; at 20_000 no retry is left (rxe_comp.c:771-792).
    """
    seed = seed_losing((True, True))
    engine = decsim.engine.Engine()
    settings = reliable_settings(BIT_ERROR_RATE, retry_count=1)
    channel = seeded_channel(engine, settings, seed)
    send_at(engine, channel, 0, 200, [])

    with pytest.raises(RuntimeError):
        engine.run()
    assert engine.now == 2 * TIMEOUT_TICKS


def test_an_ack_that_moves_the_psn_fills_the_retry_count_again():
    """Each of two messages is lost once and retried once on a count of 1.

    The first is lost at 0, resent at 10_000 and ACKed at 10_984, which
    fills the count (rxe_comp.c:165-170, 303-306); the second, sent at
    20_000, is lost, resent at 30_000 and delivered.
    """
    pattern = (True, False, False, True, False, False)
    seed = seed_losing(pattern)
    engine = decsim.engine.Engine()
    settings = reliable_settings(BIT_ERROR_RATE, retry_count=1)
    channel = seeded_channel(engine, settings, seed)
    delivered = []
    send_at(engine, channel, 0, 200, delivered)
    send_at(engine, channel, 20_000, 200, delivered)

    engine.run()

    deliveries = [transfer.delivery_ticks for transfer in delivered]
    assert deliveries == [10_598, 30_598]


def test_an_answer_while_the_message_waits_to_be_resent_is_dropped():
    """get_wqe drops it (rxe_comp.c:150-151), so it fills no count.

    One 298-tick packet, one frame of buffer, credits back 500 ticks
    after landing, a 600-tick timer and one retry. The packet lands at
    598 and its credit is back at 1098; the timer fires at 600 and
    goes back, but the resend waits for the credit, and the ACK that
    lands at 984 finds the message waiting and is dropped. The resend
    at 1098 starts a timer that fires at 1698, before the duplicate's
    ACK is back at 2082, with no retry left.
    """
    engine = decsim.engine.Engine()
    settings = reliable_settings(
        0.0,
        buffer_frames=1,
        retry_count=1,
        credit_latency_cycles=10,
        timeout_cycles=12,
    )
    channel = seeded_channel(engine, settings, 1)
    delivered = []
    send_at(engine, channel, 0, 200, delivered)

    with pytest.raises(RuntimeError):
        engine.run()
    assert engine.now == 1698
    assert [transfer.delivery_ticks for transfer in delivered] == [598]


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


class _Delivery:
    """A delivery callback, which in a run closes over a job's state."""

    def __init__(self, delivered: list) -> None:
        self.delivered = delivered

    def __call__(self, transfer) -> None:
        self.delivered.append(transfer)


def test_an_acknowledged_message_is_let_go_under_loss():
    """A message is let go once acknowledged, as rxe_comp.c retires one.

    Two 200-byte messages at 0, the first packet lost: m0 goes 0-298, m1
    298-596 and lands out of order at 896, and its NAK, 896-982, is back
    at 1282. The go-back resends m0 1282-1580 and m1 1580-1878, which
    land at 1880 and 2178 and are acknowledged; the channel then keeps
    neither callback.
    """
    seed = seed_losing((True, False, False, False, False, False))
    engine = decsim.engine.Engine()
    settings = reliable_settings(BIT_ERROR_RATE)
    channel = seeded_channel(engine, settings, seed)
    delivered = []
    first_callback = _Delivery(delivered)
    second_callback = _Delivery(delivered)
    first_reference = weakref.ref(first_callback)
    second_reference = weakref.ref(second_callback)
    first_payload = channel_module.FramedPayload(1600)
    second_payload = channel_module.FramedPayload(1600)
    first_send = functools.partial(
        channel.send, first_payload, 0, 0, first_callback
    )
    second_send = functools.partial(
        channel.send, second_payload, 0, 0, second_callback
    )
    engine.schedule(0, first_send)
    engine.schedule(0, second_send)
    del first_callback, second_callback, first_send, second_send

    engine.run()
    gc.collect()

    delivery_ticks = [transfer.delivery_ticks for transfer in delivered]
    assert delivery_ticks == [1880, 2178]
    assert first_reference() is None
    assert second_reference() is None


def test_with_nothing_lost_the_reliable_row_is_the_credit_row_property():
    generator = random.Random(11)
    for _trace in range(60):
        count = generator.randrange(1, 20)
        arrivals = sorted(generator.randrange(0, 6000) for _ in range(count))
        sizes = [generator.choice([0, 50, 256, 700]) for _ in range(count)]
        engine = decsim.engine.Engine()
        settings = reliable_settings(0.0, window_packets=10_000)
        reliable = seeded_channel(engine, settings, 1)
        credit_protocol = credit_channel.CreditChannel.Settings(
            settings.protocol.framing, 16, 2, CLOCK
        )
        credit_settings = link_settings.ChannelSettings(
            "test", PROPAGATION, settings.capacity, "test", credit_protocol
        )
        credit = credit_channel.CreditChannel(credit_settings, engine)
        by_reliable = []
        by_credit = []
        for arrival, payload in zip(arrivals, sizes, strict=True):
            send_at(engine, reliable, arrival, payload, by_reliable)
            send_at(engine, credit, arrival, payload, by_credit)
        engine.run()
        reliable_deliveries = [item.delivery_ticks for item in by_reliable]
        credit_deliveries = [item.delivery_ticks for item in by_credit]
        assert reliable_deliveries == credit_deliveries


def _off_board_escalation(protocol) -> machine_settings.MachineSettings:
    """The two-tier machine with its strong node one chassis hop away.

    The escalation hop is the reference card's off-board one: 100 Gb/s,
    400 bits a cycle at 250 MHz, 65 cycles of latency.
    """
    machine = two_tiers.points[0].machine
    card = link_profiles.path_card(
        machine.links,
        "weak_decoder_to_strong_decoder",
        clock=machine_settings.ROOM_CLOCK,
        latency_cycles=65,
        bits_per_cycle=400.0,
        source="the reference card's off-board escalation hop",
        protocol=protocol,
    )
    links = dataclasses.replace(
        machine.links, weak_decoder_to_strong_decoder=card
    )
    return dataclasses.replace(machine, links=links)


def test_a_reliable_link_at_no_errors_runs_as_its_credit_link_in_a_machine():
    """RoCE v2 frames on the escalation hop of a whole shot, nothing lost."""
    framing = framings.RoceV2.Settings(path_mtu_bytes=1024)
    room = machine_settings.ROOM_CLOCK
    credit = credit_channel.CreditChannel.Settings(
        framing=framing,
        receive_buffer_frames=8,
        credit_latency_cycles=65,
        clock=room,
    )
    reliable = reliable_channel.ReliableChannel.Settings(
        framing=framing,
        receive_buffer_frames=8,
        credit_latency_cycles=65,
        window_packets=128,
        ack_every_packets=66,
        retransmit_timeout_cycles=100_000,
        retry_count=7,
        bit_error_rate=0.0,
        clock=room,
    )
    by_credit_settings = _off_board_escalation(credit)
    by_reliable_settings = _off_board_escalation(reliable)
    by_credit = machine_module.Machine.build(by_credit_settings, 0)
    by_reliable = machine_module.Machine.build(by_reliable_settings, 0)
    frames = []
    by_reliable.links.trace.frame_landed.connect(frames.append)

    credit_result = by_credit.run()
    reliable_result = by_reliable.run()

    assert reliable_result.terminal_status == "complete"
    assert len(frames) > 0
    assert reliable_result.operation_results == credit_result.operation_results
    assert reliable_result.link_traffic == credit_result.link_traffic


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
        for arrival, payload in zip(arrivals, sizes, strict=True):
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


def test_the_expected_delay_leaves_out_the_window_wait():
    """The forecast is the unstalled wire, a lower bound on stop and wait.

    Without the window's wait p1 would follow p0 on the wire, 354 +
    338 = 692 ticks, and land at 992; the delivery waits for the ACK
    and lands at 1678 (the stop and wait timeline above).
    """
    engine = decsim.engine.Engine()
    settings = reliable_settings(0.0, window_packets=1, ack_every_packets=1)
    channel = seeded_channel(engine, settings, 1)
    payload_bits = 512 * 8
    framed = channel_module.FramedPayload(payload_bits)
    expected = channel.expected_delay_ticks(framed, 0, 0)
    delivered = []
    send_at(engine, channel, 0, 512, delivered)

    engine.run()

    assert expected == 992
    assert delivered[0].delivery_ticks == 1678


def test_the_expected_delay_is_the_delivery_within_the_window():
    engine = decsim.engine.Engine()
    settings = reliable_settings(0.0)
    channel = seeded_channel(engine, settings, 1)
    payload_bits = 512 * 8
    framed = channel_module.FramedPayload(payload_bits)
    expected = channel.expected_delay_ticks(framed, 0, 0)
    delivered = []
    send_at(engine, channel, 0, 512, delivered)

    engine.run()

    assert delivered[0].delivery_ticks == expected


def _reliable_protocol(framing, bit_error_rate: float, retry_count: int = 7):
    return reliable_channel.ReliableChannel.Settings(
        framing=framing,
        receive_buffer_frames=16,
        credit_latency_cycles=2,
        window_packets=128,
        ack_every_packets=66,
        retransmit_timeout_cycles=1000,
        retry_count=retry_count,
        bit_error_rate=bit_error_rate,
        clock=CLOCK,
    )


def test_a_bit_error_rate_of_one_is_refused():
    framing = framings.RoceV2.Settings(path_mtu_bytes=1024)

    with pytest.raises(ValueError):
        _reliable_protocol(framing, 1.0)


def test_a_reliable_protocol_on_frames_other_than_roce_v2_is_refused():
    """The ACK is a RoCE packet; flits have no acknowledgement of theirs."""
    framing = framings.Flits.Settings(flit_bits=64)

    with pytest.raises(ValueError):
        _reliable_protocol(framing, 0.0)


def test_a_retry_count_above_seven_is_refused():
    framing = framings.RoceV2.Settings(path_mtu_bytes=1024)

    with pytest.raises(ValueError):
        _reliable_protocol(framing, 0.0, retry_count=8)
