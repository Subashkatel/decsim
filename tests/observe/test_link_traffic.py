"""The traffic ledger counts what the fabric delivered and writes the JSON.

The regression lock pins every key and value of result.link_traffic;
the laws here are the ones its shots do not state: a channel's counters
are the sum of its paths' counters (one counter stream, folded twice),
and the order the transfers and channels are listed in.
"""

import decsim.engine
import decsim.links.fabric as fabric_module
import decsim.links.settings as link_settings
import decsim.observe.link_traffic as link_traffic
import decsim.records.identity as identity_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records

PATH = transfer_records.LinkPath
FREE_CHANNEL = link_settings.ChannelSettings("free", 0, None, "test")
FREE_PATH = link_settings.PathSettings(
    FREE_CHANNEL, None, "test payload", excludes_receiver_processing=False
)
OPERATION_ID = ("experiment", 7)


def _edges_on(report: dict, path_name: str) -> list:
    """The report's semantic edges on one path, in report order."""
    edges = []
    for edge in report["semantic_edges"]:
        if edge["path"] == path_name:
            edges.append(edge)
    return edges


def bounded_path(
    name, bits_per_microsecond, latency_ticks, setup_ticks=0, header_bits=0
):
    capacity = link_settings.CapacitySettings(bits_per_microsecond, "test")
    channel = link_settings.ChannelSettings(
        name, latency_ticks, capacity, "test"
    )
    return link_settings.PathSettings(
        channel,
        None,
        "test payload",
        setup_ticks,
        header_bits,
        excludes_receiver_processing=False,
    )


def unbounded_path(name, latency_ticks):
    channel = link_settings.ChannelSettings(name, latency_ticks, None, "test")
    return link_settings.PathSettings(
        channel, None, "test payload", excludes_receiver_processing=False
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


def edge_of(report, path_name):
    """The semantic edge row of one path, found by name not by position."""
    for edge in report["semantic_edges"]:
        if edge["path"] == path_name:
            return edge
    raise AssertionError(f"{path_name} is not in the report")


class Run:
    """An engine, a ledger and a fabric wired to it, from one card."""

    def __init__(self, **paths):
        wiring = every_path_free()
        wiring.update(paths)
        settings = link_settings.FabricSettings(profile_name="test", **wiring)
        self.engine = decsim.engine.Engine()
        self.ledger = link_traffic.TrafficLedger(settings)
        self.fabric = fabric_module.LinkFabric(settings, self.engine)
        self.fabric.trace.transfer_delivered.connect(self.ledger.on_transfer)

    def send(self, path, payload_bits, tick, attribution):
        delay = tick - self.engine.now

        def send():
            self.fabric.send(path, payload_bits, tick, attribution, ignore)

        self.engine.schedule(delay, send)


def ignore(_transfer):
    """A caller that wants nothing at delivery."""


def round_attribution(round_index):
    return transfer_records.TransferAttribution.for_round(
        OPERATION_ID, (1, 2), round_index
    )


def request_key_for(window_id):
    return window_records.DecoderRequestKey(
        operation_id=OPERATION_ID,
        window_id=window_id,
        tier=window_records.DecoderTier.STRONG,
        run_sequence=0,
    )


def window_attribution(window_id, relation):
    return transfer_records.TransferAttribution(
        operation_id=OPERATION_ID,
        patch_ids=(1, 2),
        window_id=window_id,
        first_round=1,
        last_round=2,
        relation=relation,
        round_keys=((OPERATION_ID, 1), (OPERATION_ID, 2)),
    )


def requested_window(window_id):
    request_key = request_key_for(window_id)
    relation = transfer_records.RequestTransferRelation(request_key)
    return window_attribution(window_id, relation)


def test_counters_reconcile_with_the_transfer_list():
    shared = unbounded_path("shared", 300)
    run = Run(qpu_to_controller=shared, controller_to_weak_buffer=shared)
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    third_round = round_attribution(3)
    run.send(PATH.QPU_TO_CONTROLLER, 8, 0, first_round)
    run.send(PATH.CONTROLLER_TO_WEAK_BUFFER, 4, 0, second_round)
    run.send(PATH.QPU_TO_CONTROLLER, None, 0, third_round)
    run.engine.run()
    snapshot = run.ledger.snapshot()
    counters_by_path = {path.path: path.counters for path in snapshot.paths}
    shared_channel = snapshot.channels[0]
    first_transfer = snapshot.transfers[0].transfer
    second_transfer = snapshot.transfers[1].transfer
    third_transfer = snapshot.transfers[2].transfer
    summed = link_traffic.TrafficCounters()
    summed = summed.plus_transfer(first_transfer)
    summed = summed.plus_transfer(second_transfer)
    summed = summed.plus_transfer(third_transfer)
    assert summed == shared_channel.counters
    assert shared_channel.counters == link_traffic.TrafficCounters(
        transfer_count=3,
        known_payload_bits=12,
        unknown_payload_transfer_count=1,
        serialization_ticks=0,
        propagation_ticks=900,
        queue_wait_ticks=0,
    )
    on_qpu_path = counters_by_path[PATH.QPU_TO_CONTROLLER]
    on_weak_buffer_path = counters_by_path[PATH.CONTROLLER_TO_WEAK_BUFFER]
    assert on_qpu_path.plus(on_weak_buffer_path) == shared_channel.counters


def test_readout_bindings_reconcile_once_per_shared_channel() -> None:
    joint = unbounded_path("joint", 7)
    shared = unbounded_path("shared", 11)
    left_route = link_settings.ReadoutRoute((1,), shared)
    right_route = link_settings.ReadoutRoute((2,), shared)
    run = Run(qpu_to_controller=joint, readout_routes=(left_route, right_route))
    joint_attribution = round_attribution(1)
    left = transfer_records.TransferAttribution.for_round(OPERATION_ID, (1,), 2)
    right = transfer_records.TransferAttribution.for_round(
        OPERATION_ID, (2,), 2
    )
    run.send(PATH.QPU_TO_CONTROLLER, 8, 0, joint_attribution)
    run.send(PATH.QPU_TO_CONTROLLER, 3, 0, left)
    run.send(PATH.QPU_TO_CONTROLLER, 5, 0, right)
    run.engine.run()

    report = run.ledger.traffic_json_value()
    readout_edges = _edges_on(report, "qpu_to_controller")
    transfer_counts = [
        edge["counters"]["transfer_count"] for edge in readout_edges
    ]
    payload_bits = [
        edge["counters"]["known_payload_bits"] for edge in readout_edges
    ]
    assert transfer_counts == [1, 2]
    assert payload_bits == [8, 8]
    assert len(report["transfers"]) == 3
    shared_channel = report["reconciliation"][2]
    assert shared_channel["member_paths"] == ["qpu_to_controller"]
    assert shared_channel["physical_counters"]["transfer_count"] == 2
    assert shared_channel["physical_counters"]["propagation_ticks"] == 22
    assert (
        shared_channel["semantic_counter_sum"]
        == shared_channel["physical_counters"]
    )
    assert all(row["reconciles"] for row in report["reconciliation"])


def test_the_transfers_are_listed_in_request_order_whatever_delivered_first():
    slow = unbounded_path("slow", 500)
    fast = unbounded_path("fast", 5)
    run = Run(qpu_to_controller=slow, controller_to_weak_buffer=fast)
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    run.send(PATH.QPU_TO_CONTROLLER, 8, 0, first_round)
    run.send(PATH.CONTROLLER_TO_WEAK_BUFFER, 8, 1, second_round)
    run.engine.run()
    snapshot = run.ledger.snapshot()
    paths = [record.path for record in snapshot.transfers]
    assert paths == [PATH.QPU_TO_CONTROLLER, PATH.CONTROLLER_TO_WEAK_BUFFER]


def test_channels_are_aliased_in_the_order_the_paths_first_meet_them():
    shared = unbounded_path("shared", 0)
    own = unbounded_path("own", 0)
    run = Run(
        qpu_to_controller=shared,
        weak_buffer_to_weak_decoder=own,
        weak_decoder_to_strong_decoder=shared,
    )
    report = run.ledger.traffic_json_value()
    readout = edge_of(report, "qpu_to_controller")
    weak_store = edge_of(report, "controller_to_weak_buffer")
    weak_input = edge_of(report, "weak_buffer_to_weak_decoder")
    escalation = edge_of(report, "weak_decoder_to_strong_decoder")
    strong_input = edge_of(report, "strong_buffer_to_strong_decoder")
    assert readout["physical_alias"] == "channel-0"
    assert weak_store["physical_alias"] == "channel-1"
    assert weak_input["physical_alias"] == "channel-2"
    assert escalation["physical_alias"] == "channel-0"
    assert strong_input["physical_alias"] == "channel-1"
    assert report["physical_channels"][0]["member_paths"] == [
        "qpu_to_controller",
        "weak_decoder_to_strong_decoder",
    ]


def test_a_window_reading_into_the_next_operation_names_each_ones_rounds():
    """Window 1:1 reads rounds 4 to 6 of operation 1 and 1 to 3 of 2."""
    run = Run()
    request_key = request_key_for(1)
    relation = transfer_records.RequestTransferRelation(request_key)
    attribution = transfer_records.TransferAttribution(
        operation_id=1,
        patch_ids=(0,),
        window_id=1,
        first_round=4,
        last_round=9,
        relation=relation,
        round_keys=((1, 4), (1, 5), (1, 6), (2, 1), (2, 2), (2, 3)),
    )
    run.send(PATH.WEAK_BUFFER_TO_WEAK_DECODER, 4, 0, attribution)
    run.engine.run()
    report = run.ledger.traffic_json_value()
    transfer = report["transfers"][0]
    first_json = identity_records.stable_identity_json(1)
    second_json = identity_records.stable_identity_json(2)
    assert transfer["attribution"]["rounds_by_operation"] == [
        {"operation_id": first_json, "round_lo": 4, "round_hi": 6},
        {"operation_id": second_json, "round_lo": 1, "round_hi": 3},
    ]


def test_a_paths_header_bits_and_setup_wait_are_its_transfers_sums():
    # one setup engine: the second window waits out the first one's setup
    bounded = bounded_path("bounded", 1.0, 10, setup_ticks=50, header_bits=8)
    run = Run(strong_buffer_to_strong_decoder=bounded)
    first_window = requested_window(3)
    second_window = requested_window(4)
    run.send(PATH.STRONG_BUFFER_TO_STRONG_DECODER, 4, 100, first_window)
    run.send(PATH.STRONG_BUFFER_TO_STRONG_DECODER, 4, 100, second_window)
    run.engine.run()
    report = run.ledger.traffic_json_value()
    semantic_edge = edge_of(report, "strong_buffer_to_strong_decoder")
    channel = report["physical_channels"][1]
    header_bits = [row["header_bits"] for row in report["transfers"]]
    assert header_bits == [8, 8]
    assert semantic_edge["counters"]["header_bits"] == 16
    assert channel["counters"]["header_bits"] == 16
    assert semantic_edge["setup_ticks"] == 100
    assert semantic_edge["setup_wait_ticks"] == 50


def test_the_ledger_sums_the_queue_wait_per_transfer_and_per_channel():
    shared = bounded_path("shared", 1000.0, 0)
    run = Run(qpu_to_controller=shared, controller_to_weak_buffer=shared)
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    third_round = round_attribution(3)
    run.send(PATH.QPU_TO_CONTROLLER, 8, 0, first_round)
    run.send(PATH.CONTROLLER_TO_WEAK_BUFFER, 8, 0, second_round)
    run.send(PATH.QPU_TO_CONTROLLER, 8, 0, third_round)
    run.engine.run()
    report = run.ledger.traffic_json_value()
    transfers = report["transfers"]
    on_qpu_path = report["semantic_edges"][0]
    on_weak_buffer_path = report["semantic_edges"][1]
    shared_channel = report["physical_channels"][0]
    assert transfers[0]["queue_wait_ticks"] == 0
    assert transfers[1]["queue_wait_ticks"] == 8000
    assert transfers[2]["queue_wait_ticks"] == 16000
    assert on_qpu_path["counters"]["queue_wait_ticks"] == 16000
    assert on_weak_buffer_path["counters"]["queue_wait_ticks"] == 8000
    assert shared_channel["counters"]["queue_wait_ticks"] == 24000
