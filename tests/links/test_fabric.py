"""The fabric sends on the path's channel with the path's payload rule.

Sources: ns-3 point-to-point-net-device.cc for the shared wire two paths
queue on; gem5 src/dev/dma_device.cc for the setup engine two paths
share; the component shape of gem5 src/sim/sim_object.hh (a component
owns its settings and its children and is observed through a callback,
so it runs with no observer at all).
"""

import dataclasses

import pytest

import decsim.config as config
import decsim.engine
import decsim.links.channel as channel_module
import decsim.links.fabric as fabric_module
import decsim.links.settings as link_settings
import decsim.ports as ports
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records

PATH = transfer_records.LinkPath
FREE_CHANNEL = link_settings.ChannelSettings("free", 0, None, "test")
FREE_PATH = link_settings.PathSettings(
    FREE_CHANNEL, None, "test payload", excludes_receiver_processing=False
)


def bounded_path(name, bits_per_microsecond, latency_ticks, setup_ticks=0):
    capacity = link_settings.CapacitySettings(bits_per_microsecond, "test")
    channel = link_settings.ChannelSettings(
        name, latency_ticks, capacity, "test"
    )
    return link_settings.PathSettings(
        channel,
        None,
        "test payload",
        setup_ticks,
        excludes_receiver_processing=False,
    )


def unbounded_path(name, latency_ticks, setup_ticks=0):
    channel = link_settings.ChannelSettings(name, latency_ticks, None, "test")
    return link_settings.PathSettings(
        channel,
        None,
        "test payload",
        setup_ticks,
        excludes_receiver_processing=False,
    )


def default_path(name, bits):
    channel = link_settings.ChannelSettings(name, 0, None, "test")
    payload = link_settings.PayloadSettings(bits, "test default")
    return link_settings.PathSettings(
        channel, payload, None, excludes_receiver_processing=False
    )


def every_path_free():
    """One free path per hop, keyed by the hop's name.

    A card names every hop, so a test that cares about one path wires
    that one and leaves the rest on the shared free channel.
    """
    names = []
    for path in transfer_records.LinkPath:
        names.append(path.value)
    return dict.fromkeys(names, FREE_PATH)


def fabric_with(engine, listener=None, **paths):
    """A fabric whose paths are free unless the test wires them itself."""
    wiring = every_path_free()
    wiring.update(paths)
    settings = link_settings.FabricSettings(profile_name="test", **wiring)
    fabric = fabric_module.LinkFabric(settings, engine)
    if listener is not None:
        fabric.trace.transfer_delivered.connect(listener.on_transfer)
    return fabric


def round_attribution(round_index):
    return transfer_records.TransferAttribution(
        operation_id=1,
        patch_ids=(0,),
        window_id=None,
        first_round=round_index,
        last_round=round_index,
    )


def operation_attribution(relation=None):
    return transfer_records.TransferAttribution(
        operation_id=1,
        patch_ids=(0,),
        window_id=None,
        first_round=None,
        last_round=None,
        relation=relation,
    )


def request_relation_for(window_id):
    request_key = window_records.DecoderRequestKey(
        operation_id=1,
        window_id=window_id,
        tier=window_records.DecoderTier.WEAK,
        run_sequence=window_id,
    )
    return transfer_records.RequestTransferRelation(request_key)


def window_attribution(window_id, relation=None):
    return transfer_records.TransferAttribution(
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
    engine = decsim.engine.Engine()
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


def test_readout_footprints_select_independent_channel_queues() -> None:
    engine = decsim.engine.Engine()
    left = bounded_path("left-wire", 1000.0, 300)
    right = bounded_path("right-wire", 2000.0, 100)
    routes = (
        link_settings.ReadoutRoute(("left",), left),
        link_settings.ReadoutRoute(("right",), right),
    )
    fabric = fabric_with(engine, readout_routes=routes)
    first = transfer_records.TransferAttribution.for_round(1, ("left",), 1)
    second = transfer_records.TransferAttribution.for_round(1, ("right",), 1)
    delivered = []
    send_on(engine, fabric, PATH.QPU_TO_CONTROLLER, 8, 0, first, delivered)
    send_on(engine, fabric, PATH.QPU_TO_CONTROLLER, 8, 0, second, delivered)

    engine.run()

    delivery_ticks = [transfer.delivery_ticks for transfer in delivered]
    assert delivery_ticks == [4100, 8300]
    queue_wait_ticks = [transfer.queue_wait_ticks for transfer in delivered]
    assert queue_wait_ticks == [0, 0]


def test_readout_routes_with_one_channel_name_share_the_wire() -> None:
    engine = decsim.engine.Engine()
    shared = bounded_path("shared-readout", 1000.0, 300)
    routes = (
        link_settings.ReadoutRoute(("left",), shared),
        link_settings.ReadoutRoute(("right",), shared),
    )
    fabric = fabric_with(engine, readout_routes=routes)
    first = transfer_records.TransferAttribution.for_round(1, ("left",), 1)
    second = transfer_records.TransferAttribution.for_round(1, ("right",), 1)
    delivered = []
    send_on(engine, fabric, PATH.QPU_TO_CONTROLLER, 8, 0, first, delivered)
    send_on(engine, fabric, PATH.QPU_TO_CONTROLLER, 4, 0, second, delivered)

    engine.run()

    delivery_ticks = [transfer.delivery_ticks for transfer in delivered]
    assert delivery_ticks == [8300, 12300]
    assert delivered[1].queue_wait_ticks == 8000


def test_a_joint_footprint_uses_its_whole_route_or_the_default() -> None:
    engine = decsim.engine.Engine()
    left = unbounded_path("left-wire", 100)
    joint = unbounded_path("joint-wire", 200)
    default = unbounded_path("default-wire", 300)
    routes = (
        link_settings.ReadoutRoute(("left",), left),
        link_settings.ReadoutRoute(("right", "left"), joint),
    )
    fabric = fabric_with(
        engine, qpu_to_controller=default, readout_routes=routes
    )
    pair = transfer_records.TransferAttribution.for_round(
        1, ("left", "right"), 1
    )
    other_pair = transfer_records.TransferAttribution.for_round(
        2, ("left", "other"), 1
    )
    delivered = []
    send_on(engine, fabric, PATH.QPU_TO_CONTROLLER, 8, 0, pair, delivered)
    send_on(engine, fabric, PATH.QPU_TO_CONTROLLER, 8, 0, other_pair, delivered)

    engine.run()

    delivery_ticks = [transfer.delivery_ticks for transfer in delivered]
    assert delivery_ticks == [200, 300]


def test_two_channels_with_the_same_numbers_and_different_names_are_two_wires():
    engine = decsim.engine.Engine()
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
    engine = decsim.engine.Engine()
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
    assert (second.setup_wait_ticks, second.setup_ticks) == (4, 5)
    assert second.send_ticks == 20


def test_a_paths_header_is_framed_onto_the_transfer_it_sends():
    """448 bits is CUDA-Q's enqueue framing, 24 plus 32 bytes.

    The 24 byte RPCHeader in front of every request is cudaqx
    decoder_rpc_wire_format.h lines 41-43, and the 32 bytes of fields in
    front of an enqueue's syndromes are lines 62-69.
    """
    engine = decsim.engine.Engine()
    rate_bits_per_microsecond = 1000.0
    capacity = link_settings.CapacitySettings(rate_bits_per_microsecond, "test")
    channel = link_settings.ChannelSettings("framed", 0, capacity, "test")
    payload_bits = 360
    header_bits = 448
    framed_path = link_settings.PathSettings(
        channel,
        None,
        "test payload",
        header_bits_per_transfer=header_bits,
        excludes_receiver_processing=False,
    )
    fabric = fabric_with(engine, weak_decoder_to_strong_decoder=framed_path)
    delivered = []
    region = operation_attribution()
    send_on(
        engine,
        fabric,
        PATH.WEAK_DECODER_TO_STRONG_DECODER,
        payload_bits,
        0,
        region,
        delivered,
    )
    engine.run()
    wire_bits = payload_bits + header_bits
    wire_microseconds = wire_bits / rate_bits_per_microsecond
    expected_ticks = config.microseconds_to_ticks(wire_microseconds)
    transfer = delivered[0]
    assert transfer.payload_bits == payload_bits
    assert transfer.header_bits == header_bits
    assert transfer.serialization_ticks == expected_ticks


def test_an_unsized_transfer_rides_an_unbounded_channel_unresolved():
    engine = decsim.engine.Engine()
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
    assert (
        record.payload_selection is transfer_records.PayloadSelection.UNRESOLVED
    )


def test_a_bounded_channel_refuses_a_transfer_with_no_payload_size():
    engine = decsim.engine.Engine()
    bounded = bounded_path("bounded", 1000.0, 0)
    fabric = fabric_with(engine, controller_to_weak_buffer=bounded)
    attribution = round_attribution(1)
    with pytest.raises(
        RuntimeError, match="a bounded wire needs a size to serialize"
    ):
        fabric.send(PATH.CONTROLLER_TO_WEAK_BUFFER, None, 0, attribution, print)


def test_an_actual_payload_on_a_path_without_a_source_is_refused():
    engine = decsim.engine.Engine()
    default_only = default_path("default", 100)
    fabric = fabric_with(engine, qpu_to_controller=default_only)
    attribution = round_attribution(1)
    with pytest.raises(RuntimeError):
        fabric.send(PATH.QPU_TO_CONTROLLER, 3, 0, attribution, print)


def test_the_expected_delay_includes_the_paths_setup():
    engine = decsim.engine.Engine()
    with_setup = unbounded_path("store", 3, setup_ticks=5)
    fabric = fabric_with(engine, controller_to_strong_buffer=with_setup)
    assert (
        fabric.expected_delay_ticks(PATH.CONTROLLER_TO_STRONG_BUFFER, 8, 0) == 8
    )


def test_the_fabric_fills_the_link_port():
    engine = decsim.engine.Engine()
    fabric = fabric_with(engine)
    assert isinstance(fabric, ports.Link)


class _CountingChannel:
    """A channel written outside the links package: it notes what it carries."""

    def __init__(self, channel_settings, engine):
        self.name = channel_settings.name
        self.carried_bits = []
        self.inner = channel_module.Channel(channel_settings, engine)
        self.trace = self.inner.trace

    def send(self, framed, now_ticks, setup_ticks, on_delivered):
        self.carried_bits.append(framed.payload_bits)
        self.inner.send(framed, now_ticks, setup_ticks, on_delivered)

    def expected_delay_ticks(self, framed, now_ticks, setup_ticks):
        return self.inner.expected_delay_ticks(framed, now_ticks, setup_ticks)


@dataclasses.dataclass(frozen=True)
class _CountingProtocol:
    """A protocol record written outside the links package."""

    built: list = dataclasses.field(compare=False)

    def build(self, channel_settings, engine):
        channel = _CountingChannel(channel_settings, engine)
        self.built.append(channel)
        return channel


def test_the_fabric_builds_one_channel_of_the_protocols_per_channel_name():
    """The Channel port: a protocol's channel carries every path it names."""
    engine = decsim.engine.Engine()
    built = []
    shared = bounded_path("shared", 1000.0, 300)
    counting_protocol = _CountingProtocol(built)
    counting_channel = dataclasses.replace(
        shared.channel, protocol=counting_protocol
    )
    shared = dataclasses.replace(shared, channel=counting_channel)
    wiring = every_path_free()
    wiring.update(qpu_to_controller=shared, controller_to_weak_buffer=shared)
    settings = link_settings.FabricSettings(profile_name="test", **wiring)
    fabric = fabric_module.LinkFabric(settings, engine)
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
    assert len(built) == 1
    assert isinstance(built[0], ports.Channel)
    assert built[0].carried_bits == [8, 4]
    assert delivered[1].queue_wait_ticks == 8000


# The paths a whole run uses, read off the ledger the fabric feeds.
