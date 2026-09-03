"""The fabric sends on the path's channel with the path's payload rule.

Sources: ns-3 point-to-point-net-device.cc for the shared wire two paths
queue on; gem5 src/dev/dma_device.cc for the setup engine two paths
share; the component shape of gem5 src/sim/sim_object.hh (a component
owns its settings and its children and is observed through a callback,
so it runs with no observer at all).
"""

import pytest

import decsim.engine
import decsim.links.fabric as fabric_module
import decsim.links.settings as link_settings
import decsim.message as message
import decsim.ports as ports

PATH = message.LinkPath
AGGREGATE = link_settings.QuantityBasis.AGGREGATE
FREE_CHANNEL = link_settings.ChannelSettings("free", 0, None, "test")
FREE_PATH = link_settings.PathSettings(FREE_CHANNEL, None, "test payload")


def bounded_path(name, bits_per_microsecond, latency_ticks, setup_ticks=0):
    capacity = link_settings.CapacitySettings(
        bits_per_microsecond, AGGREGATE, None, "test"
    )
    channel = link_settings.ChannelSettings(
        name, latency_ticks, capacity, "test"
    )
    return link_settings.PathSettings(
        channel, None, "test payload", setup_ticks
    )


def unbounded_path(name, latency_ticks, setup_ticks=0):
    channel = link_settings.ChannelSettings(name, latency_ticks, None, "test")
    return link_settings.PathSettings(
        channel, None, "test payload", setup_ticks
    )


def default_path(name, bits):
    channel = link_settings.ChannelSettings(name, 0, None, "test")
    payload = link_settings.PayloadSettings(
        bits, AGGREGATE, None, "test default"
    )
    return link_settings.PathSettings(channel, payload, None)


def fabric_with(engine, listener=None, **paths):
    """A fabric whose paths are free unless the test wires them itself."""
    wiring = dict.fromkeys(
        (
            "qpu_to_controller",
            "weak_buffer_to_weak_decoder",
            "weak_decoder_to_strong_decoder",
            "strong_buffer_to_strong_decoder",
            "weak_decoder_to_frame",
            "decoder_to_decoder",
            "strong_decoder_to_frame",
            "frame_to_controller",
            "controller_to_qpu",
        ),
        FREE_PATH,
    )
    wiring.update(paths)
    settings = link_settings.FabricSettings(profile_name="test", **wiring)
    return fabric_module.LinkFabric(settings, engine, listener)


def round_attribution(round_index):
    return message.TransferAttribution(
        operation_id=1,
        patch_ids=(0,),
        window_id=None,
        first_round=round_index,
        last_round=round_index,
    )


def operation_attribution(relation=None):
    return message.TransferAttribution(
        operation_id=1,
        patch_ids=(0,),
        window_id=None,
        first_round=None,
        last_round=None,
        relation=relation,
    )


def request_relation_for(window_id):
    request_key = message.DecoderRequestKey(
        operation_id=1,
        window_id=window_id,
        tier=message.DecoderTier.WEAK,
        run_sequence=window_id,
    )
    return message.RequestTransferRelation(request_key)


def window_attribution(window_id, relation=None):
    return message.TransferAttribution(
        operation_id=1,
        patch_ids=(0,),
        window_id=window_id,
        first_round=window_id,
        last_round=window_id,
        relation=relation,
    )


def requested_window(window_id):
    relation = request_relation_for(window_id)
    return window_attribution(window_id, relation)


class Listener:
    def __init__(self):
        self.records = []

    def on_transfer(self, record):
        self.records.append(record)


def send_on(engine, fabric, path, payload_bits, tick, attribution, delivered):
    """Send on the path at the tick; the transfer is appended to delivered."""
    delay = tick - engine.now

    def send():
        fabric.send(path, payload_bits, tick, attribution, delivered.append)

    engine.schedule(delay, send)


def test_two_paths_on_one_channel_share_its_queue():
    engine = decsim.engine.Engine(verbose=False)
    shared = bounded_path("shared", 1000.0, 300)
    fabric = fabric_with(
        engine, qpu_to_controller=shared, controller_to_weak_buffer=shared
    )
    delivered = []
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    send_on(
        engine, fabric, PATH.QPU_TO_CONTROLLER, 8, 0, first_round, delivered
    )
    send_on(
        engine,
        fabric,
        PATH.CONTROLLER_TO_WEAK_BUFFER,
        4,
        0,
        second_round,
        delivered,
    )
    engine.run()
    on_weak_buffer_path = delivered[1]
    assert on_weak_buffer_path.serializer_start_ticks == 8000
    assert on_weak_buffer_path.queue_wait_ticks == 8000
    assert on_weak_buffer_path.physical_sequence == 1


def test_two_channels_with_the_same_numbers_and_different_names_are_two_wires():
    engine = decsim.engine.Engine(verbose=False)
    first_wire = bounded_path("first", 1000.0, 0)
    second_wire = bounded_path("second", 1000.0, 0)
    fabric = fabric_with(
        engine,
        qpu_to_controller=first_wire,
        controller_to_weak_buffer=second_wire,
    )
    delivered = []
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    send_on(
        engine, fabric, PATH.QPU_TO_CONTROLLER, 8, 0, first_round, delivered
    )
    send_on(
        engine,
        fabric,
        PATH.CONTROLLER_TO_WEAK_BUFFER,
        8,
        0,
        second_round,
        delivered,
    )
    engine.run()
    on_weak_buffer_path = delivered[1]
    assert on_weak_buffer_path.queue_wait_ticks == 0
    assert on_weak_buffer_path.physical_sequence == 0


def test_two_paths_with_setups_on_one_channel_share_its_setup_engine():
    engine = decsim.engine.Engine(verbose=False)
    shared = unbounded_path("shared", 0, setup_ticks=5)
    fabric = fabric_with(
        engine,
        weak_buffer_to_weak_decoder=shared,
        strong_buffer_to_strong_decoder=shared,
    )
    delivered = []
    weak_window = requested_window(1)
    strong_window = requested_window(2)
    send_on(
        engine,
        fabric,
        PATH.WEAK_BUFFER_TO_WEAK_DECODER,
        8,
        10,
        weak_window,
        delivered,
    )
    send_on(
        engine,
        fabric,
        PATH.STRONG_BUFFER_TO_STRONG_DECODER,
        8,
        11,
        strong_window,
        delivered,
    )
    engine.run()
    second = delivered[1]
    assert (second.setup_ticks, second.send_ticks) == (9, 20)


def test_a_path_with_no_setup_never_waits_for_another_paths_setup():
    engine = decsim.engine.Engine(verbose=False)
    with_setup = unbounded_path("shared", 0, setup_ticks=5)
    without_setup = unbounded_path("shared", 0)
    fabric = fabric_with(
        engine,
        weak_buffer_to_weak_decoder=with_setup,
        strong_buffer_to_strong_decoder=without_setup,
    )
    delivered = []
    weak_window = requested_window(1)
    strong_window = requested_window(2)
    send_on(
        engine,
        fabric,
        PATH.WEAK_BUFFER_TO_WEAK_DECODER,
        8,
        10,
        weak_window,
        delivered,
    )
    send_on(
        engine,
        fabric,
        PATH.STRONG_BUFFER_TO_STRONG_DECODER,
        8,
        12,
        strong_window,
        delivered,
    )
    engine.run()
    first_delivered = delivered[0]
    assert first_delivered.request_ticks == 12
    assert first_delivered.delivery_ticks == 12


def test_a_free_path_costs_nothing():
    engine = decsim.engine.Engine(verbose=False)
    fabric = fabric_with(engine)
    delivered = []
    first_round = round_attribution(1)
    send_on(
        engine, fabric, PATH.QPU_TO_CONTROLLER, 1000, 42, first_round, delivered
    )
    engine.run()
    transfer = delivered[0]
    assert transfer.total_delay_ticks == 0
    assert transfer.delivery_ticks == 42


def test_an_actual_payload_is_priced_and_named_by_its_source():
    engine = decsim.engine.Engine(verbose=False)
    listener = Listener()
    fabric = fabric_with(engine, listener)
    delivered = []
    first_round = round_attribution(1)
    send_on(
        engine, fabric, PATH.QPU_TO_CONTROLLER, 7, 0, first_round, delivered
    )
    engine.run()
    record = listener.records[0]
    transfer = delivered[0]
    assert transfer.payload_bits == 7
    assert record.payload_selection is message.PayloadSelection.ACTUAL
    assert record.payload_source == "test payload"


def test_a_missing_payload_takes_the_cards_default():
    engine = decsim.engine.Engine(verbose=False)
    listener = Listener()
    bus_word = default_path("bus", 32)
    fabric = fabric_with(engine, listener, frame_to_controller=bus_word)
    delivered = []
    attribution = operation_attribution()
    send_on(
        engine,
        fabric,
        PATH.FRAME_TO_CONTROLLER,
        None,
        0,
        attribution,
        delivered,
    )
    engine.run()
    record = listener.records[0]
    transfer = delivered[0]
    assert transfer.payload_bits == 32
    assert (
        record.payload_selection is message.PayloadSelection.CONFIGURED_DEFAULT
    )
    assert record.payload_source == "test default"


def test_an_unsized_transfer_rides_an_unbounded_channel_unresolved():
    engine = decsim.engine.Engine(verbose=False)
    listener = Listener()
    fabric = fabric_with(engine, listener)
    delivered = []
    first_round = round_attribution(1)
    send_on(
        engine, fabric, PATH.QPU_TO_CONTROLLER, None, 0, first_round, delivered
    )
    engine.run()
    record = listener.records[0]
    transfer = delivered[0]
    assert transfer.payload_bits is None
    assert record.payload_selection is message.PayloadSelection.UNRESOLVED


def test_a_bounded_channel_refuses_a_transfer_with_no_payload_size():
    engine = decsim.engine.Engine(verbose=False)
    bounded = bounded_path("bounded", 1000.0, 0)
    fabric = fabric_with(engine, controller_to_weak_buffer=bounded)
    attribution = round_attribution(1)
    with pytest.raises(
        RuntimeError, match="a bounded wire needs a size to serialize"
    ):
        fabric.send(PATH.CONTROLLER_TO_WEAK_BUFFER, None, 0, attribution, print)


def test_an_actual_payload_on_a_path_without_a_source_is_refused():
    engine = decsim.engine.Engine(verbose=False)
    default_only = default_path("default", 100)
    fabric = fabric_with(engine, qpu_to_controller=default_only)
    attribution = round_attribution(1)
    with pytest.raises(RuntimeError, match="actual payload source"):
        fabric.send(PATH.QPU_TO_CONTROLLER, 3, 0, attribution, print)


def test_the_listener_sees_every_transfer_once_in_order():
    engine = decsim.engine.Engine(verbose=False)
    listener = Listener()
    fabric = fabric_with(engine, listener)
    delivered = []
    first_round = round_attribution(1)
    instruction = operation_attribution()
    send_on(
        engine, fabric, PATH.QPU_TO_CONTROLLER, 8, 0, first_round, delivered
    )
    send_on(
        engine, fabric, PATH.CONTROLLER_TO_QPU, 128, 5, instruction, delivered
    )
    engine.run()
    paths = [record.path for record in listener.records]
    sequences = [record.request_sequence for record in listener.records]
    assert paths == [PATH.QPU_TO_CONTROLLER, PATH.CONTROLLER_TO_QPU]
    assert sequences == [0, 1]
    assert len(delivered) == 2


def test_the_listener_sees_the_transfer_before_the_caller():
    engine = decsim.engine.Engine(verbose=False)
    listener = Listener()
    fabric = fabric_with(engine, listener)
    seen_by_caller = []

    def delivered(_transfer):
        record_count = len(listener.records)
        seen_by_caller.append(record_count)

    attribution = round_attribution(1)
    fabric.send(PATH.QPU_TO_CONTROLLER, 8, 0, attribution, delivered)
    engine.run()
    assert seen_by_caller == [1]


def test_a_fabric_with_no_listener_runs():
    engine = decsim.engine.Engine(verbose=False)
    fabric = fabric_with(engine)
    delivered = []
    first_round = round_attribution(1)
    send_on(
        engine, fabric, PATH.QPU_TO_CONTROLLER, 8, 0, first_round, delivered
    )
    engine.run()
    assert len(delivered) == 1


def test_an_optional_path_is_wired_only_when_the_card_names_it():
    engine = decsim.engine.Engine(verbose=False)
    store = unbounded_path("store", 3)
    with_store = fabric_with(engine, controller_to_strong_buffer=store)
    without_store = fabric_with(engine)
    assert with_store.is_wired(PATH.CONTROLLER_TO_STRONG_BUFFER)
    assert not without_store.is_wired(PATH.CONTROLLER_TO_STRONG_BUFFER)
    assert without_store.is_wired(PATH.QPU_TO_CONTROLLER)


def test_the_expected_delay_prices_the_paths_payload_rule():
    engine = decsim.engine.Engine(verbose=False)
    capacity = link_settings.CapacitySettings(1000.0, AGGREGATE, None, "test")
    channel = link_settings.ChannelSettings("bounded", 300, capacity, "test")
    payload = link_settings.PayloadSettings(8, AGGREGATE, None, "test default")
    bus_word = link_settings.PathSettings(channel, payload, None)
    fabric = fabric_with(engine, frame_to_controller=bus_word)
    assert (
        fabric.expected_delay_ticks(PATH.FRAME_TO_CONTROLLER, None, 0) == 8300
    )


def test_the_expected_delay_includes_the_paths_setup():
    engine = decsim.engine.Engine(verbose=False)
    with_setup = unbounded_path("store", 3, setup_ticks=5)
    fabric = fabric_with(engine, controller_to_strong_buffer=with_setup)
    assert (
        fabric.expected_delay_ticks(PATH.CONTROLLER_TO_STRONG_BUFFER, 8, 0) == 8
    )


def test_the_fabric_fills_the_link_port():
    engine = decsim.engine.Engine(verbose=False)
    fabric = fabric_with(engine)
    assert isinstance(fabric, ports.Link)
