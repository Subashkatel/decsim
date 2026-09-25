"""A reliable channel recovers every lost frame by go-back-N, in order.

Sources: Linux drivers/infiniband/sw/rxe, rxe_req.c:38-53 (a retry goes
back to the first unacknowledged PSN), 448-451 (the acknowledgement
request), 588-590 and 684-717 (the timer, the retry, the window);
rxe_resp.c:70-95 and 1563-1568 (one sequence NAK per gap, a duplicate
re-acknowledged); rxe_comp.c:303-322 and 720-790 (a NAK acknowledges
what is before it; one retry at a time unless the timer fires). The
hand-traced timelines below follow those lines; with nothing lost the
row is the credit row (tests/links/test_credit_channel.py).
"""

import random

import pytest

import decsim.config as config
import decsim.engine
import decsim.links.channel as channel_module
import decsim.links.credit_channel as credit_channel
import decsim.links.framings as framings
import decsim.links.reliable_channel as reliable_channel
import decsim.links.settings as link_settings

# 100-bit flits at 1_000_000 bits per microsecond take 100 ticks each.
FLIT_BITS = 100
RATE = 1_000_000
PROPAGATION = 300
CLOCK = config.Clock(50)
# about one half: 1 - (1 - 0.0069)^100
HALF_LOSS_BIT_ERROR_RATE = 0.0069


def reliable_settings(
    bit_error_rate,
    window_packets=16,
    buffer_frames=16,
    timeout_cycles=200,
    ack_every_packets=66,
):
    flit_settings = framings.Flits.Settings(FLIT_BITS)
    framing_settings = link_settings.FramingSettings("flits", flit_settings)
    row = reliable_channel.ReliableChannel.Settings(
        framing=framing_settings,
        receive_buffer_frames=buffer_frames,
        credit_latency_cycles=2,
        window_packets=window_packets,
        ack_every_packets=ack_every_packets,
        retransmit_timeout_cycles=timeout_cycles,
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


def seed_losing(pattern):
    """The first seed whose draws lose exactly the pattern's transmissions.

    A draw below 0.45 loses a half-loss frame (0.4996) and one above 0.55
    keeps it, so the pattern holds whatever the last digits.
    """
    for seed in range(100_000):
        generator = random.Random(seed)
        draws = [generator.random() for _ in pattern]
        pairs = list(zip(draws, pattern))
        lost_draws = [draw for draw, is_lost in pairs if is_lost]
        kept_draws = [draw for draw, is_lost in pairs if not is_lost]
        highest_lost = max(lost_draws, default=0.0)
        lowest_kept = min(kept_draws, default=1.0)
        if highest_lost < 0.45 and lowest_kept > 0.55:
            return seed
    raise AssertionError("no seed draws the pattern")


def send_at(engine, channel, tick, payload_bits, delivered):
    delay = tick - engine.now
    framed = channel_module.FramedPayload(payload_bits)

    def send():
        channel.send(framed, tick, 0, delivered.append)

    engine.schedule(delay, send)


def test_a_lost_frame_is_resent_after_the_sequence_nak_hand_traced():
    """p1 is lost; p2's arrival at 600 sends NAK(1), back at 700.

    p0 0-100, p1 100-200 (lost), p2 200-300, p3 300-400; at 700 the
    requester goes back to p1: 700-800, 800-900, 900-1000, and p3 lands
    at 1300, when the message is delivered.
    """
    pattern = (False, True, False, False, False, False, False)
    seed = seed_losing(pattern)
    engine = decsim.engine.Engine()
    settings = reliable_settings(HALF_LOSS_BIT_ERROR_RATE)
    channel = seeded_channel(engine, settings, seed)
    frames = []
    channel.trace.frame_landed.connect(frames.append)
    delivered = []
    send_at(engine, channel, 0, 400, delivered)

    engine.run()

    starts = [record.timing.start_ticks for record in frames]
    lost = [record.is_lost for record in frames]
    resent = [record.is_retransmission for record in frames]
    transfer = delivered[0]
    assert starts == [0, 100, 200, 300, 700, 800, 900]
    assert lost == list(pattern)
    assert resent == [False] * 4 + [True] * 3
    assert transfer.delivery_ticks == 1300
    assert transfer.serialization_ticks == 700
    assert transfer.queue_wait_ticks == 300


def test_a_nak_that_starts_a_retry_leaves_the_timer_running():
    """The timer set at the first send still expires at 10_000.

    rxe's completer leaves a retry started by a NAK without resetting
    the timer (rxe_comp.c:668-669 and 743-749). p1 is lost, NAK(1) comes
    back at 700 and the resent p1 is lost too; p2 and p3 then arrive out
    of sequence, but one NAK per gap was sent already, so only the timer
    recovers: back to p1 at 10_000, p3 ends at 10_300 and lands 300 on.
    """
    pattern = (False, True, False, False, True) + (False,) * 5
    seed = seed_losing(pattern)
    engine = decsim.engine.Engine()
    settings = reliable_settings(HALF_LOSS_BIT_ERROR_RATE)
    channel = seeded_channel(engine, settings, seed)
    frames = []
    channel.trace.frame_landed.connect(frames.append)
    delivered = []
    send_at(engine, channel, 0, 400, delivered)

    engine.run()

    starts = [record.timing.start_ticks for record in frames]
    transfer = delivered[0]
    assert starts[4:] == [700, 800, 900, 10_000, 10_100, 10_200]
    assert transfer.delivery_ticks == 10_300 + PROPAGATION


def test_a_lost_last_frame_is_resent_when_the_timer_expires():
    """No later frame reports the gap, so the 10_000-tick timer does."""
    seed = seed_losing((True, False))
    engine = decsim.engine.Engine()
    settings = reliable_settings(HALF_LOSS_BIT_ERROR_RATE)
    channel = seeded_channel(engine, settings, seed)
    delivered = []
    send_at(engine, channel, 0, FLIT_BITS, delivered)

    engine.run()

    transfer = delivered[0]
    assert transfer.serializer_start_ticks == 0
    assert transfer.delivery_ticks == 10_000 + 100 + PROPAGATION


def test_the_run_ends_at_the_last_acknowledgement_not_at_the_timer():
    """The timer stops when every packet is acknowledged.

    One flit: sent 0-100, lands 400, its ACK is back at 500; the
    10_000-tick timer it started never fires.
    """
    engine = decsim.engine.Engine()
    settings = reliable_settings(0.0)
    channel = seeded_channel(engine, settings, 1)
    delivered = []
    send_at(engine, channel, 0, FLIT_BITS, delivered)

    engine.run()

    assert delivered[0].delivery_ticks == 400
    assert engine.now == 500


def test_with_nothing_lost_the_reliable_row_is_the_credit_row_property():
    generator = random.Random(11)
    for _trace in range(60):
        count = generator.randrange(1, 20)
        arrivals = sorted(generator.randrange(0, 3000) for _ in range(count))
        bits = [generator.choice([0, 50, 100, 1000]) for _ in range(count)]
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
        for arrival, payload in zip(arrivals, bits):
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
        arrivals = sorted(generator.randrange(0, 5000) for _ in range(count))
        flit_counts = [generator.randrange(1, 6) for _ in range(count)]
        window = generator.choice([1, 2, 4, 128])
        buffer_frames = generator.choice([1, 3, 16])
        engine = decsim.engine.Engine()
        settings = reliable_settings(
            0.003, window_packets=window, buffer_frames=buffer_frames
        )
        channel = seeded_channel(engine, settings, trace)
        delivered = []
        for arrival, flit_count in zip(arrivals, flit_counts):
            message_bits = flit_count * FLIT_BITS
            send_at(engine, channel, arrival, message_bits, delivered)
        engine.run()
        sequences = [transfer.physical_sequence for transfer in delivered]
        deliveries = [transfer.delivery_ticks for transfer in delivered]
        assert sequences == list(range(count))
        assert deliveries == sorted(deliveries)


def test_a_window_of_one_waits_for_each_acknowledgement():
    """Stop and wait: the next packet goes when the ACK is back.

    Each packet asks for an ACK; p0 lands at 400, its ACK is back at
    500, p1 goes 500-600 and lands at 900.
    """
    engine = decsim.engine.Engine()
    settings = reliable_settings(0.0, window_packets=1, ack_every_packets=1)
    channel = seeded_channel(engine, settings, 1)
    delivered = []
    send_at(engine, channel, 0, 200, delivered)

    engine.run()

    assert delivered[0].delivery_ticks == 900


def test_one_seed_draws_the_same_losses_twice():
    runs = []
    for _run in range(2):
        engine = decsim.engine.Engine()
        settings = reliable_settings(0.004)
        channel = seeded_channel(engine, settings, 99)
        delivered = []
        send_at(engine, channel, 0, 4000, delivered)
        engine.run()
        runs.append(delivered)

    assert runs[0] == runs[1]


def test_a_bit_error_rate_of_one_is_refused():
    section = {
        "framing": {"kind": "flits", "flit_bits": 64},
        "receive_buffer_frames": 4,
        "credit_latency_cycles": 1,
        "window_packets": 128,
        "ack_every_packets": 66,
        "retransmit_timeout_cycles": 1000,
        "bit_error_rate": 1.0,
    }

    with pytest.raises(ValueError, match="below 1"):
        reliable_channel.ReliableChannel.Settings.from_yaml(section, "path")
