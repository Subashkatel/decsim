"""A channel delivers when the point-to-point law says, and setups queue.

Sources: ns-3 point-to-point-net-device.cc (Send enqueues, TransmitStart
runs when the transmitter is READY, one packet on the wire at a time, the
receiver has it txTime plus the channel delay later; a fractional tick of
serialization rounds up); gem5 src/dev/dma_device.cc (one transmitList
per DmaPort, so a setup queue belongs to a channel and two channels'
setups are independent); Shao et al., MICRO 2016, section III.C (the DMA
engine services descriptors one by one while the processor is free, so a
request with no setup never waits for another's setup); the closed form
of that queue, run here over random traces. One microsecond is
1_000_000 ticks.
"""

import fractions
import math
import random

import decsim.config as config
import decsim.engine
import decsim.links.channel as channel_module
import decsim.links.settings as link_settings
import decsim.records.transfers as transfer_records


def bounded_channel(engine, bits_per_microsecond, latency_ticks):
    capacity = link_settings.CapacitySettings(bits_per_microsecond, "test")
    settings = link_settings.ChannelSettings(
        "test", latency_ticks, capacity, "test"
    )
    return channel_module.Channel(settings, engine)


def unbounded_channel(engine, latency_ticks):
    settings = link_settings.ChannelSettings(
        "test", latency_ticks, None, "test"
    )
    return channel_module.Channel(settings, engine)


def send_at(engine, channel, tick, payload_bits, setup_ticks, delivered):
    """Send at the tick; the transfer is appended to delivered."""
    delay = tick - engine.now

    framed = transfer_records.FramedPayload(payload_bits)

    def send():
        channel.send(framed, tick, setup_ticks, delivered.append)

    engine.schedule(delay, send)


def closed_form(arrivals, bits, rate_bits_per_us, propagation_ticks):
    """The point-to-point closed form: start = max(arrival, previous end)."""
    deliveries = []
    serializer_free = 0
    rate_text = str(rate_bits_per_us)
    rate = fractions.Fraction(rate_text)
    for arrival, payload in zip(arrivals, bits, strict=True):
        start = max(arrival, serializer_free)
        payload_fraction = fractions.Fraction(payload)
        exact = payload_fraction * config.TICKS_PER_MICROSECOND / rate
        serialization = math.ceil(exact)
        end = start + serialization
        serializer_free = end
        delivery = end + propagation_ticks
        deliveries.append(delivery)
    return deliveries


def test_random_traces_match_the_point_to_point_closed_form_property():
    generator = random.Random(1)
    for _trace in range(200):
        count = generator.randrange(2, 40)
        arrivals = sorted(generator.randrange(0, 5000) for _ in range(count))
        bits = [generator.choice([1, 17, 64, 289, 4096]) for _ in range(count)]
        rate = generator.choice([0.5, 1.0, 7.0, 64.0, 1000.0, 12345.0])
        propagation = generator.randrange(0, 300)
        engine = decsim.engine.Engine()
        channel = bounded_channel(engine, rate, propagation)
        delivered = []
        for arrival, payload in zip(arrivals, bits, strict=True):
            send_at(engine, channel, arrival, payload, 0, delivered)
        engine.run()
        deliveries = [transfer.delivery_ticks for transfer in delivered]
        assert deliveries == closed_form(arrivals, bits, rate, propagation)


def test_a_header_is_serialized_with_the_payload_and_counted_apart():
    """ns-3 times the packet with its header (net-device.cc:243 and :528)."""
    engine = decsim.engine.Engine()
    rate_bits_per_microsecond = 1000.0
    channel = bounded_channel(engine, rate_bits_per_microsecond, 300)
    delivered = []
    payload_bits = 8
    header_bits = 24
    framed = transfer_records.FramedPayload(
        payload_bits=payload_bits, header_bits=header_bits
    )

    channel.send(framed, 0, 0, delivered.append)
    engine.run()

    wire_bits = payload_bits + header_bits
    # the closed form with no propagation is the serialization alone
    (expected_ticks,) = closed_form(
        [0], [wire_bits], rate_bits_per_microsecond, 0
    )
    transfer = delivered[0]
    assert transfer.serialization_ticks == expected_ticks
    assert transfer.payload_bits == payload_bits
    assert transfer.header_bits == header_bits


def test_a_setup_waits_for_the_previous_setup_on_the_channel():
    engine = decsim.engine.Engine()
    channel = unbounded_channel(engine, 0)
    delivered = []
    send_at(engine, channel, 10, 8, 5, delivered)
    send_at(engine, channel, 11, 8, 5, delivered)
    engine.run()
    first = delivered[0]
    second = delivered[1]
    assert (first.setup_wait_ticks, first.setup_ticks) == (0, 5)
    assert first.send_ticks == 15
    assert (second.setup_wait_ticks, second.setup_ticks) == (4, 5)
    assert second.send_ticks == 20
    assert second.delivery_ticks == 20


def test_setups_on_two_channels_are_independent():
    engine = decsim.engine.Engine()
    slow = unbounded_channel(engine, 0)
    fast = unbounded_channel(engine, 0)
    delivered_slow = []
    delivered_fast = []
    send_at(engine, slow, 0, 8, 50, delivered_slow)
    send_at(engine, fast, 0, 8, 5, delivered_fast)
    engine.run()
    slow_transfer = delivered_slow[0]
    fast_transfer = delivered_fast[0]
    assert (slow_transfer.setup_ticks, slow_transfer.send_ticks) == (50, 50)
    assert (fast_transfer.setup_ticks, fast_transfer.send_ticks) == (5, 5)


def test_a_zero_setup_request_goes_to_the_wire_ahead_of_anothers_setup():
    engine = decsim.engine.Engine()
    channel = bounded_channel(engine, 1000.0, 0)
    delivered = []
    send_at(engine, channel, 10, 8, 5, delivered)
    send_at(engine, channel, 12, 8, 0, delivered)
    engine.run()
    first_delivered = delivered[0]
    second_delivered = delivered[1]
    assert first_delivered.request_ticks == 12
    assert first_delivered.serializer_start_ticks == 12
    assert first_delivered.physical_sequence == 0
    assert second_delivered.request_ticks == 10
    assert second_delivered.serializer_start_ticks == 8012
    assert second_delivered.queue_wait_ticks == 7997
    assert second_delivered.physical_sequence == 1


def test_a_setup_ending_as_a_zero_setup_request_arrives_follows_event_order():
    """A tie within one tick runs in the order the events were enqueued.

    That is ns-3's rule: the Scheduler serves same-time events in
    insertion order (src/core/model/scheduler.h). The zero-setup send at
    15 was enqueued before the setup that ends at 15, so it takes the
    wire first.
    """
    engine = decsim.engine.Engine()
    channel = bounded_channel(engine, 1000.0, 0)
    delivered = []
    send_at(engine, channel, 15, 8, 0, delivered)
    send_at(engine, channel, 10, 8, 5, delivered)
    engine.run()
    without_setup = delivered[0]
    with_setup = delivered[1]
    assert without_setup.request_ticks == 15
    assert without_setup.serializer_start_ticks == 15
    assert without_setup.physical_sequence == 0
    assert with_setup.send_ticks == 15
    assert with_setup.serializer_start_ticks == 8015
    assert with_setup.physical_sequence == 1


def test_setup_serialization_and_propagation_add_up():
    engine = decsim.engine.Engine()
    channel = bounded_channel(engine, 1000.0, 300)
    delivered = []
    send_at(engine, channel, 0, 8, 5, delivered)
    send_at(engine, channel, 0, 8, 5, delivered)
    engine.run()
    first = delivered[0]
    second = delivered[1]
    assert first.setup_ticks == 5
    assert first.serializer_start_ticks == 5
    assert first.serializer_end_ticks == 8005
    assert first.total_delay_ticks == 5 + 8000 + 300
    assert (second.setup_wait_ticks, second.setup_ticks) == (5, 5)
    assert second.queue_wait_ticks == 7995
    assert second.serializer_start_ticks == 8005
    assert second.total_delay_ticks == 5 + 5 + 7995 + 8000 + 300


def test_a_transfer_with_no_payload_size_rides_an_unbounded_channel():
    engine = decsim.engine.Engine()
    channel = unbounded_channel(engine, 9)
    delivered = []
    send_at(engine, channel, 7, None, 0, delivered)
    engine.run()
    transfer = delivered[0]
    assert transfer.payload_bits is None
    assert transfer.delivery_ticks == 16


def test_the_expected_delay_is_the_delivery_when_nothing_overtakes():
    engine = decsim.engine.Engine()
    channel = bounded_channel(engine, 1000.0, 300)
    delivered = []
    expected = []

    framed = transfer_records.FramedPayload(8)

    def ask_then_send():
        delay = channel.expected_delay_ticks(framed, 10, 5)
        expected.append(delay)
        channel.send(framed, 10, 5, delivered.append)

    send_at(engine, channel, 0, 8, 5, delivered)
    engine.schedule(10, ask_then_send)
    engine.run()
    second = delivered[1]
    assert expected == [5 + 7990 + 8000 + 300]
    assert second.total_delay_ticks == 16295


def test_the_expected_delay_counts_a_transfer_still_in_setup_ahead():
    """A setup queued behind another's reaches the wire after it.

    The first send's setup ends at 20000 and its 30 bits hold the wire
    until 50000; the second, asked at 5, is ready at 40000 and starts at
    50000, so it delivers at 58000, not at 48000.
    """
    engine = decsim.engine.Engine()
    channel = bounded_channel(engine, 1000.0, 0)
    delivered = []
    expected = []

    framed = transfer_records.FramedPayload(8)

    def ask_then_send():
        delay = channel.expected_delay_ticks(framed, 5, 20000)
        expected.append(delay)
        channel.send(framed, 5, 20000, delivered.append)

    send_at(engine, channel, 0, 30, 20000, delivered)
    engine.schedule(5, ask_then_send)
    engine.run()
    second = delivered[1]
    assert second.delivery_ticks == 58000
    assert expected == [second.total_delay_ticks]


def test_the_expected_delay_leaves_the_channel_untouched():
    engine = decsim.engine.Engine()
    channel = bounded_channel(engine, 1000.0, 0)
    framed = transfer_records.FramedPayload(8)
    channel.expected_delay_ticks(framed, 10, 5)
    delivered = []
    send_at(engine, channel, 10, 8, 5, delivered)
    engine.run()
    transfer = delivered[0]
    assert transfer.serializer_start_ticks == 15
    assert transfer.physical_sequence == 0


def test_a_channel_reports_each_transfer_as_one_whole_frame():
    """The ideal row's inside is one frame: the transfer's own interval."""
    engine = decsim.engine.Engine()
    channel = bounded_channel(engine, 1000.0, 300)
    frames = []
    channel.trace.frame_landed.connect(frames.append)
    delivered = []
    send_at(engine, channel, 0, 8, 0, delivered)

    engine.run()

    record = frames[0]
    transfer = delivered[0]
    assert len(frames) == 1
    assert record.timing.bits == 8
    assert record.timing.start_ticks == transfer.serializer_start_ticks
    assert record.timing.landed_ticks == transfer.delivery_ticks
