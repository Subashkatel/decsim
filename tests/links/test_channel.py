"""A channel delivers when the point-to-point law says, and setups queue.

Sources: ns-3 point-to-point-net-device.cc (Send enqueues, TransmitStart
runs when the transmitter is READY, one packet on the wire at a time, the
receiver has it txTime plus the channel delay later; a fractional tick of
serialization rounds up); gem5 src/dev/dma_device.cc (one transmitList
per DmaPort, so a setup queue belongs to a channel and two channels'
setups are independent); Shao et al., MICRO 2016, section III.C (the DMA
engine services descriptors one by one while the processor is free, so a
request with no setup never waits for another's setup); the closed form
of the row L1 harness in the sandbox validation folder, run here over
random traces. One microsecond is 1_000_000 ticks.
"""

import fractions
import math
import random

import decsim.config as config
import decsim.engine
import decsim.links.channel as channel_module
import decsim.links.settings as link_settings

AGGREGATE = link_settings.QuantityBasis.AGGREGATE


def bounded_channel(engine, bits_per_microsecond, latency_ticks):
    capacity = link_settings.CapacitySettings(
        bits_per_microsecond, AGGREGATE, None, "test"
    )
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

    def send():
        channel.send(payload_bits, tick, setup_ticks, delivered.append)

    engine.schedule(delay, send)


def closed_form(arrivals, bits, rate_bits_per_us, propagation_ticks):
    """The row L1 closed form: start = max(arrival, end of the previous)."""
    deliveries = []
    serializer_free = 0
    rate_text = str(rate_bits_per_us)
    rate = fractions.Fraction(rate_text)
    for arrival, payload in zip(arrivals, bits):
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
        engine = decsim.engine.Engine(verbose=False)
        channel = bounded_channel(engine, rate, propagation)
        delivered = []
        for arrival, payload in zip(arrivals, bits):
            send_at(engine, channel, arrival, payload, 0, delivered)
        engine.run()
        deliveries = [transfer.delivery_ticks for transfer in delivered]
        assert deliveries == closed_form(arrivals, bits, rate, propagation)


def test_delivery_is_send_plus_serialization_plus_latency():
    engine = decsim.engine.Engine(verbose=False)
    channel = bounded_channel(engine, 1000.0, 300)
    delivered = []
    send_at(engine, channel, 10, 8, 0, delivered)
    engine.run()
    transfer = delivered[0]
    assert transfer.request_ticks == 10
    assert transfer.serializer_start_ticks == 10
    assert transfer.serialization_ticks == 8000
    assert transfer.serializer_end_ticks == 8010
    assert transfer.propagation_ticks == 300
    assert transfer.delivery_ticks == 8310
    assert transfer.total_delay_ticks == 8300
    assert engine.now == 8310


def test_an_unbounded_channel_delivers_at_send_plus_latency():
    engine = decsim.engine.Engine(verbose=False)
    channel = unbounded_channel(engine, 7)
    delivered = []
    send_at(engine, channel, 10, 1000, 0, delivered)
    engine.run()
    transfer = delivered[0]
    assert transfer.serialization_ticks == 0
    assert transfer.queue_wait_ticks == 0
    assert transfer.delivery_ticks == 17


def test_two_sends_serialize_on_the_wire():
    engine = decsim.engine.Engine(verbose=False)
    channel = bounded_channel(engine, 1000.0, 0)
    delivered = []
    send_at(engine, channel, 10, 8, 0, delivered)
    send_at(engine, channel, 20, 8, 0, delivered)
    engine.run()
    first = delivered[0]
    second = delivered[1]
    assert first.serializer_end_ticks == 8010
    assert second.serializer_start_ticks == 8010
    assert second.queue_wait_ticks == 7990
    assert second.delivery_ticks == 16010
    assert (first.physical_sequence, second.physical_sequence) == (0, 1)


def test_an_unbounded_channel_never_queues():
    engine = decsim.engine.Engine(verbose=False)
    channel = unbounded_channel(engine, 7)
    delivered = []
    send_at(engine, channel, 10, 1000, 0, delivered)
    send_at(engine, channel, 10, 1000, 0, delivered)
    engine.run()
    second = delivered[1]
    assert second.queue_wait_ticks == 0
    assert second.delivery_ticks == 17


def test_a_request_at_the_wires_free_tick_waits_zero():
    engine = decsim.engine.Engine(verbose=False)
    channel = bounded_channel(engine, 1000.0, 0)
    delivered = []
    send_at(engine, channel, 10, 8, 0, delivered)
    send_at(engine, channel, 8010, 8, 0, delivered)
    engine.run()
    second = delivered[1]
    assert second.queue_wait_ticks == 0
    assert second.serializer_start_ticks == 8010


def test_a_fractional_tick_of_serialization_rounds_up():
    engine = decsim.engine.Engine(verbose=False)
    channel = bounded_channel(engine, 3.0, 0)
    delivered = []
    send_at(engine, channel, 0, 1, 0, delivered)
    engine.run()
    transfer = delivered[0]
    assert transfer.serialization_ticks == 333334


def test_serialization_uses_the_decimal_rate_written_on_the_card():
    engine = decsim.engine.Engine(verbose=False)
    channel = bounded_channel(engine, 0.20846, 0)
    delivered = []
    send_at(engine, channel, 0, 400_310_292, 0, delivered)
    engine.run()
    transfer = delivered[0]
    assert transfer.serialization_ticks == 1_920_321_845_917_683


def test_a_per_lane_rate_serializes_at_the_aggregate_rate():
    engine = decsim.engine.Engine(verbose=False)
    capacity = link_settings.CapacitySettings(
        0.7, link_settings.QuantityBasis.PER_LANE, 3, "test"
    )
    settings = link_settings.ChannelSettings("test", 0, capacity, "test")
    channel = channel_module.Channel(settings, engine)
    delivered = []
    send_at(engine, channel, 0, 21, 0, delivered)
    engine.run()
    transfer = delivered[0]
    assert transfer.serialization_ticks == 10_000_000


def test_a_setup_waits_for_the_previous_setup_on_the_channel():
    engine = decsim.engine.Engine(verbose=False)
    channel = unbounded_channel(engine, 0)
    delivered = []
    send_at(engine, channel, 10, 8, 5, delivered)
    send_at(engine, channel, 11, 8, 5, delivered)
    engine.run()
    first = delivered[0]
    second = delivered[1]
    assert (first.setup_ticks, first.send_ticks) == (5, 15)
    assert (second.setup_ticks, second.send_ticks) == (9, 20)
    assert second.delivery_ticks == 20


def test_setups_on_two_channels_are_independent():
    engine = decsim.engine.Engine(verbose=False)
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


def test_a_setup_after_the_engine_went_idle_costs_only_its_own_ticks():
    engine = decsim.engine.Engine(verbose=False)
    channel = unbounded_channel(engine, 0)
    delivered = []
    send_at(engine, channel, 10, 8, 5, delivered)
    send_at(engine, channel, 100, 8, 5, delivered)
    engine.run()
    later = delivered[1]
    assert (later.setup_ticks, later.send_ticks) == (5, 105)


def test_two_paths_with_setups_on_one_channel_take_the_wire_in_setup_order():
    """The row L1 shared-channel case.

    A at 10, A at 11, B at 12, five-tick setups, 8 bits on a 1000 bits
    per microsecond wire. The setups serialize on the channel's one
    engine, so the wire starts are 15, 8015 and 16015 and B's setup ends
    at 25.
    """
    engine = decsim.engine.Engine(verbose=False)
    channel = bounded_channel(engine, 1000.0, 0)
    delivered = []
    send_at(engine, channel, 10, 8, 5, delivered)
    send_at(engine, channel, 11, 8, 5, delivered)
    send_at(engine, channel, 12, 8, 5, delivered)
    engine.run()
    starts = [transfer.serializer_start_ticks for transfer in delivered]
    sends = [transfer.send_ticks for transfer in delivered]
    assert starts == [15, 8015, 16015]
    assert sends == [15, 20, 25]


def test_a_zero_setup_request_goes_to_the_wire_ahead_of_anothers_setup():
    engine = decsim.engine.Engine(verbose=False)
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


def test_a_request_with_no_setup_leaves_the_setup_engine_untouched():
    engine = decsim.engine.Engine(verbose=False)
    channel = unbounded_channel(engine, 0)
    delivered = []
    send_at(engine, channel, 10, 8, 5, delivered)
    send_at(engine, channel, 12, 8, 0, delivered)
    send_at(engine, channel, 13, 8, 5, delivered)
    engine.run()
    without_setup = delivered[0]
    second_with_setup = delivered[2]
    assert without_setup.send_ticks == 12
    assert (second_with_setup.setup_ticks, second_with_setup.send_ticks) == (
        7,
        20,
    )


def test_a_setup_ending_as_a_zero_setup_request_arrives_follows_event_order():
    """A tie within one tick runs in the order the events were enqueued.

    That is ns-3's rule: the Scheduler serves same-time events in
    insertion order (src/core/model/scheduler.h). The zero-setup send at
    15 was enqueued before the setup that ends at 15, so it takes the
    wire first.
    """
    engine = decsim.engine.Engine(verbose=False)
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
    engine = decsim.engine.Engine(verbose=False)
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
    assert second.setup_ticks == 10
    assert second.queue_wait_ticks == 7995
    assert second.serializer_start_ticks == 8005
    assert second.total_delay_ticks == 10 + 7995 + 8000 + 300


def test_an_empty_payload_pays_setup_and_latency_only():
    engine = decsim.engine.Engine(verbose=False)
    channel = bounded_channel(engine, 1000.0, 7)
    delivered = []
    send_at(engine, channel, 10, 0, 5, delivered)
    engine.run()
    transfer = delivered[0]
    assert transfer.serialization_ticks == 0
    assert transfer.total_delay_ticks == 12


def test_a_transfer_with_no_payload_size_rides_an_unbounded_channel():
    engine = decsim.engine.Engine(verbose=False)
    channel = unbounded_channel(engine, 9)
    delivered = []
    send_at(engine, channel, 7, None, 0, delivered)
    engine.run()
    transfer = delivered[0]
    assert transfer.payload_bits is None
    assert transfer.delivery_ticks == 16


def test_the_expected_delay_is_the_delivery_when_nothing_overtakes():
    engine = decsim.engine.Engine(verbose=False)
    channel = bounded_channel(engine, 1000.0, 300)
    delivered = []
    expected = []

    def ask_then_send():
        delay = channel.expected_delay_ticks(8, 10, 5)
        expected.append(delay)
        channel.send(8, 10, 5, delivered.append)

    send_at(engine, channel, 0, 8, 5, delivered)
    engine.schedule(10, ask_then_send)
    engine.run()
    second = delivered[1]
    assert expected == [5 + 7990 + 8000 + 300]
    assert second.total_delay_ticks == 16295


def test_the_expected_delay_leaves_the_channel_untouched():
    engine = decsim.engine.Engine(verbose=False)
    channel = bounded_channel(engine, 1000.0, 0)
    channel.expected_delay_ticks(8, 10, 5)
    delivered = []
    send_at(engine, channel, 10, 8, 5, delivered)
    engine.run()
    transfer = delivered[0]
    assert transfer.serializer_start_ticks == 15
    assert transfer.physical_sequence == 0
