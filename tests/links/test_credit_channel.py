"""A credit channel sends a frame only while it holds a receive-buffer credit.

Sources: gem5 Garnet, src/mem/ruby/network/garnet/OutVcState.cc:56-63
(a virtual channel starts with buffers_per_data_vc credits),
NetworkInterface.cc:506 and 530 (a flit goes only while its VC has a
credit, and takes one), NetworkInterface.cc:236-274 (a network interface
returns the credit as the flit arrives), NetworkLink.cc:92-102 (a link,
and the credit link, deliver after their latency). The closed form of
that loop, frame k starting at max(ready, the previous end, the credit
of frame k - C), is the oracle the random traces run against; ns-3's
point-to-point law (point-to-point-net-device.cc:243) is the ideal row
it reduces to. One microsecond is 1_000_000 ticks.
"""

import random

import pytest

import decsim.config as config
import decsim.engine
import decsim.links.channel as channel_module
import decsim.links.credit_channel as credit_channel
import decsim.links.framings as framings
import decsim.links.settings as link_settings


def credit_settings(
    buffer_frames,
    credit_latency_cycles,
    framing,
    bits_per_microsecond,
    latency_ticks,
    clock_period_ticks=1,
):
    clock = config.Clock(clock_period_ticks)
    protocol = credit_channel.CreditChannel.Settings(
        framing, buffer_frames, credit_latency_cycles, clock
    )
    capacity = link_settings.CapacitySettings(bits_per_microsecond, "test")
    return link_settings.ChannelSettings(
        "test", latency_ticks, capacity, "test", protocol
    )


def flits(flit_bits):
    return framings.Flits.Settings(flit_bits)


def whole():
    return framings.Whole.Settings()


def send_at(engine, channel, tick, payload_bits, setup_ticks, delivered):
    """Send at the tick; the transfer is appended to delivered."""
    delay = tick - engine.now

    framed = channel_module.FramedPayload(payload_bits)

    def send():
        channel.send(framed, tick, setup_ticks, delivered.append)

    engine.schedule(delay, send)


def test_a_frame_waits_for_the_credit_of_the_frame_c_places_before_it():
    """Hand-traced: 100-tick flits, L 300, L_c 100, two credits.

    f0 0-100 lands 400, credit 500; f1 100-200 lands 500, credit 600;
    f2 waits to 500, 500-600 lands 900; f3 600-700 lands 1000; f4 waits
    for f2's credit at 1000, 1000-1100, lands 1400.
    """
    engine = decsim.engine.Engine()
    hundred_bit_flits = flits(100)
    settings = credit_settings(2, 2, hundred_bit_flits, 1_000_000, 300, 50)
    channel = credit_channel.CreditChannel(settings, engine)
    frames = []
    channel.trace.frame_landed.connect(frames.append)
    delivered = []
    send_at(engine, channel, 0, 500, 0, delivered)

    engine.run()

    starts = [record.timing.start_ticks for record in frames]
    credit_waits = [record.timing.credit_wait_ticks for record in frames]
    transfer = delivered[0]
    assert starts == [0, 100, 500, 600, 1000]
    assert credit_waits == [0, 0, 300, 0, 300]
    assert transfer.delivery_ticks == 1400
    assert transfer.serialization_ticks == 500
    assert transfer.queue_wait_ticks == 600
    assert transfer.serializer_end_ticks == 1100


def test_a_credit_protocol_on_an_unbounded_wire_stops_at_its_first_frame():
    """A frame has no wire time without a rate, so the first send stops."""
    engine = decsim.engine.Engine()
    clock = config.Clock(50)
    hundred_bit_flits = flits(100)
    protocol = credit_channel.CreditChannel.Settings(
        hundred_bit_flits, 2, 2, clock
    )
    settings = link_settings.ChannelSettings(
        "test", 300, None, "test", protocol
    )
    channel = credit_channel.CreditChannel(settings, engine)
    delivered = []
    send_at(engine, channel, 0, 500, 0, delivered)

    with pytest.raises(AttributeError, match="input_bits_per_microsecond"):
        engine.run()


def test_with_credits_to_spare_whole_frames_cross_as_the_ideal_row_property():
    """Random traces with setups: the credit row reduces to channel.py."""
    generator = random.Random(3)
    for _trace in range(100):
        count = generator.randrange(2, 30)
        arrivals = sorted(generator.randrange(0, 5000) for _ in range(count))
        bits = [generator.choice([0, 1, 17, 64, 4096]) for _ in range(count)]
        setups = [generator.choice([0, 0, 40, 300]) for _ in range(count)]
        rate = generator.choice([0.5, 7.0, 1000.0, 12345.0])
        propagation = generator.randrange(0, 300)
        engine = decsim.engine.Engine()
        whole_frames = whole()
        settings = credit_settings(count, 0, whole_frames, rate, propagation)
        credit = credit_channel.CreditChannel(settings, engine)
        ideal_settings = link_settings.ChannelSettings(
            "test", propagation, settings.capacity, "test"
        )
        ideal = channel_module.Channel(ideal_settings, engine)
        by_credit = []
        by_ideal = []
        for arrival, payload, setup in zip(arrivals, bits, setups, strict=True):
            send_at(engine, credit, arrival, payload, setup, by_credit)
            send_at(engine, ideal, arrival, payload, setup, by_ideal)
        engine.run()
        assert by_credit == by_ideal


def test_no_c_frames_cross_in_less_than_one_credit_round_trip_property():
    """Frame k starts one round trip after frame k - C at the earliest.

    So the delivered rate is at most min(R, C x F / RTT), with RTT the
    serialization, the propagation and the credit latency.
    """
    generator = random.Random(5)
    for _trace in range(100):
        buffer_frames = generator.randrange(1, 17)
        flit_bits = generator.choice([8, 64, 128, 576])
        rate = generator.choice([100.0, 1000.0, 64_000.0])
        propagation = generator.randrange(0, 2000)
        credit_latency = generator.randrange(0, 2000)
        frame_count = generator.randrange(1, 80)
        engine = decsim.engine.Engine()
        framing = flits(flit_bits)
        settings = credit_settings(
            buffer_frames, credit_latency, framing, rate, propagation
        )
        channel = credit_channel.CreditChannel(settings, engine)
        frames = []
        channel.trace.frame_landed.connect(frames.append)
        delivered = []
        message_bits = frame_count * flit_bits
        send_at(engine, channel, 0, message_bits, 0, delivered)
        engine.run()
        serialization = channel_module.serialization_ticks(
            flit_bits, settings.capacity
        )
        round_trip = serialization + propagation + credit_latency
        last_index = len(frames) - 1
        last_start = frames[last_index].timing.start_ticks
        assert last_start >= last_index * serialization
        assert last_start >= (last_index // buffer_frames) * round_trip


def test_the_expected_delay_is_the_delay_when_nothing_else_arrives():
    engine = decsim.engine.Engine()
    hundred_bit_flits = flits(100)
    settings = credit_settings(2, 2, hundred_bit_flits, 1_000_000, 300, 50)
    channel = credit_channel.CreditChannel(settings, engine)
    delivered = []
    send_at(engine, channel, 0, 300, 0, delivered)
    engine.schedule(10, lambda: None)
    engine.run()
    framed = channel_module.FramedPayload(500)
    expected = channel.expected_delay_ticks(framed, engine.now, 0)
    send_at(engine, channel, engine.now, 500, 0, delivered)

    engine.run()

    second = delivered[1]
    assert expected == second.total_delay_ticks
