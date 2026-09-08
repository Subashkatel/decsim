"""The fabric sends on the path's channel with the path's payload rule.

Sources: ns-3 point-to-point-net-device.cc for the shared wire two paths
queue on; gem5 src/dev/dma_device.cc for the setup engine two paths
share; the component shape of gem5 src/sim/sim_object.hh (a component
owns its settings and its children and is observed through a callback,
so it runs with no observer at all).

The law at the end of the file reads the ledger of a whole run on the
declared card of tests/declared_run.py: which of the wired paths a
round actually crosses when only one decoder tier exists.
"""

import pytest

import decsim.engine
import decsim.links.fabric as fabric_module
import decsim.links.settings as link_settings
import decsim.ports as ports
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import tests.declared_run as declared_run

PATH = transfer_records.LinkPath
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
    assert (second.setup_ticks, second.send_ticks) == (9, 20)


def test_a_path_with_no_setup_never_waits_for_another_paths_setup():
    engine = decsim.engine.Engine()
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
    engine = decsim.engine.Engine()
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
    engine = decsim.engine.Engine()
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
    assert record.payload_selection is transfer_records.PayloadSelection.ACTUAL
    assert record.payload_source == "test payload"


def test_a_missing_payload_takes_the_cards_default():
    engine = decsim.engine.Engine()
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
        record.payload_selection
        is transfer_records.PayloadSelection.CONFIGURED_DEFAULT
    )
    assert record.payload_source == "test default"


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
    with pytest.raises(RuntimeError, match="actual payload source"):
        fabric.send(PATH.QPU_TO_CONTROLLER, 3, 0, attribution, print)


def test_the_listener_sees_every_transfer_once_in_order():
    engine = decsim.engine.Engine()
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
    engine = decsim.engine.Engine()
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
    engine = decsim.engine.Engine()
    fabric = fabric_with(engine)
    delivered = []
    first_round = round_attribution(1)
    send_on(
        engine, fabric, PATH.QPU_TO_CONTROLLER, 8, 0, first_round, delivered
    )
    engine.run()
    assert len(delivered) == 1


def test_the_expected_delay_prices_the_paths_payload_rule():
    engine = decsim.engine.Engine()
    capacity = link_settings.CapacitySettings(1000.0, AGGREGATE, None, "test")
    channel = link_settings.ChannelSettings("bounded", 300, capacity, "test")
    payload = link_settings.PayloadSettings(8, AGGREGATE, None, "test default")
    bus_word = link_settings.PathSettings(channel, payload, None)
    fabric = fabric_with(engine, frame_to_controller=bus_word)
    assert (
        fabric.expected_delay_ticks(PATH.FRAME_TO_CONTROLLER, None, 0) == 8300
    )


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


# The paths a whole run uses, read off the ledger the fabric feeds.


def transfer_counts_by_path(machine):
    """How many transfers the run put on each wired path."""
    snapshot = machine.observation.traffic.snapshot()
    counts = {}
    for path_snapshot in snapshot.paths:
        counts[path_snapshot.path] = path_snapshot.counters.transfer_count
    return counts


def test_a_strong_primary_round_crosses_one_path_and_only_that_one():
    """One tier means one hop: the room-side path carries every round.

    A single-tier system streams its readout to its decoder over one
    path (LILLIPUT's readout-to-decoder FIFO, Das et al. 2108.06569;
    Google's streaming decoder, 2408.13687), and under StrongOnly
    (decsim/escalation/policies.py) readiness listens to the strong
    store, so the weak store's path is wired by the card and never
    used while each of the six rounds crosses
    controller_to_strong_buffer once.
    """
    machine = declared_run.strong_only_run(rounds=6)

    transfers_by_path = transfer_counts_by_path(machine)

    assert transfers_by_path[PATH.CONTROLLER_TO_WEAK_BUFFER] == 0
    assert transfers_by_path[PATH.CONTROLLER_TO_STRONG_BUFFER] == 6
