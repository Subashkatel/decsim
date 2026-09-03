"""The traffic ledger counts what the fabric delivered and writes the JSON.

Sources: the gate's golden (validation/responsibility_audit_2026_08_30)
pins every key and value of result.link_traffic, so the key sets here are
the pinned ones written out; a channel's counters are the sum of its
paths' counters by construction (one counter stream, folded twice).
"""

import json

import decsim.engine
import decsim.links.settings as link_settings
import decsim.message as message
import decsim.observe.link_traffic as link_traffic

PATH = message.LinkPath
AGGREGATE = link_settings.QuantityBasis.AGGREGATE
FREE_CHANNEL = link_settings.ChannelSettings("free", 0, None, "test")
FREE_PATH = link_settings.PathSettings(FREE_CHANNEL, None, "test payload")
OPERATION_ID = ("experiment", 7)

TRAFFIC_KEYS = {
    "schema_version",
    "path_order",
    "semantic_edges",
    "physical_channels",
    "transfers",
    "reconciliation",
}
COUNTER_KEYS = {
    "transfer_count",
    "known_payload_bits",
    "unknown_payload_transfer_count",
    "serialization_ticks",
    "propagation_ticks",
    "queue_wait_ticks",
}
SEMANTIC_EDGE_KEYS = {"path", "physical_alias", "counters", "setup_ticks"}
PHYSICAL_CHANNEL_KEYS = {"physical_alias", "member_paths", "counters"}
RECONCILIATION_KEYS = {
    "physical_alias",
    "member_paths",
    "semantic_counter_sum",
    "physical_counters",
    "reconciles",
}
TRANSFER_KEYS = {
    "path",
    "physical_alias",
    "attribution",
    "payload_bits",
    "payload_selection",
    "payload_source",
    "setup_ticks",
    "send_ticks",
    "serializer_start_ticks",
    "serializer_end_ticks",
    "delivery_ticks",
    "queue_wait_ticks",
    "serialization_ticks",
    "propagation_ticks",
    "total_delay_ticks",
    "physical_sequence",
}
ATTRIBUTION_KEYS = {
    "operation_id",
    "patch_ids",
    "window_id",
    "round_lo",
    "round_hi",
    "relation",
}
REQUEST_KEY_KEYS = {"operation_id", "window_id", "tier", "run_sequence"}
BOUNDARY_RELATION_KEYS = {
    "request_key",
    "source_window_key",
    "destination_window_key",
    "source_revision",
    "delivery_revision",
}


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


def unbounded_path(name, latency_ticks):
    channel = link_settings.ChannelSettings(name, latency_ticks, None, "test")
    return link_settings.PathSettings(channel, None, "test payload")


class Run:
    """An engine, a ledger and a fabric wired to it, from one card."""

    def __init__(self, **paths):
        wiring = dict.fromkeys(
            ("qc", "wbd", "wsd", "sbd", "wdo", "dd", "do", "oc", "cq"),
            FREE_PATH,
        )
        wiring.update(paths)
        settings = link_settings.FabricSettings(profile_name="test", **wiring)
        self.engine = decsim.engine.Engine(verbose=False)
        self.ledger = link_traffic.TrafficLedger(settings)
        self.fabric = settings.build(self.engine, self.ledger)

    def send(self, path, payload_bits, tick, attribution):
        delay = tick - self.engine.now

        def send():
            self.fabric.send(path, payload_bits, tick, attribution, ignore)

        self.engine.schedule(delay, send)


def ignore(_transfer):
    """A caller that wants nothing at delivery."""


def round_attribution(round_index):
    return message.TransferAttribution(
        operation_id=OPERATION_ID,
        patch_ids=(1, 2),
        window_id=None,
        first_round=round_index,
        last_round=round_index,
    )


def request_key_for(window_id):
    return message.DecoderRequestKey(
        operation_id=OPERATION_ID,
        window_id=window_id,
        tier=message.DecoderTier.STRONG,
        run_sequence=0,
    )


def window_attribution(window_id, relation):
    return message.TransferAttribution(
        operation_id=OPERATION_ID,
        patch_ids=(1, 2),
        window_id=window_id,
        first_round=1,
        last_round=2,
        relation=relation,
    )


def requested_window(window_id):
    request_key = request_key_for(window_id)
    relation = message.RequestTransferRelation(request_key)
    return window_attribution(window_id, relation)


def boundary_attribution(window_id):
    request_key = request_key_for(window_id)
    relation = message.BoundaryTransferRelation(
        source_request_key=request_key,
        source_window_key=(OPERATION_ID, window_id),
        destination_window_key=(OPERATION_ID, 4),
        source_revision=2,
        delivery_revision=5,
    )
    return window_attribution(window_id, relation)


def test_counters_reconcile_with_the_transfer_list():
    shared = unbounded_path("shared", 300)
    run = Run(qc=shared, cwb=shared)
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    third_round = round_attribution(3)
    run.send(PATH.QC, 8, 0, first_round)
    run.send(PATH.CWB, 4, 0, second_round)
    run.send(PATH.QC, None, 0, third_round)
    run.engine.run()
    snapshot = run.ledger.snapshot()
    counters_by_path = {path.path: path.counters for path in snapshot.paths}
    shared_channel = snapshot.channels[0]
    summed = link_traffic.TrafficCounters()
    for record in snapshot.transfers:
        summed = summed.plus_transfer(record.transfer)
    assert summed == shared_channel.counters
    assert shared_channel.counters == link_traffic.TrafficCounters(
        transfer_count=3,
        known_payload_bits=12,
        unknown_payload_transfer_count=1,
        serialization_ticks=0,
        propagation_ticks=900,
        queue_wait_ticks=0,
    )
    on_qpu_path = counters_by_path[PATH.QC]
    on_weak_buffer_path = counters_by_path[PATH.CWB]
    assert on_qpu_path.plus(on_weak_buffer_path) == shared_channel.counters


def test_the_transfers_are_listed_in_request_order_whatever_delivered_first():
    slow = unbounded_path("slow", 500)
    fast = unbounded_path("fast", 5)
    run = Run(qc=slow, cwb=fast)
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    run.send(PATH.QC, 8, 0, first_round)
    run.send(PATH.CWB, 8, 1, second_round)
    run.engine.run()
    snapshot = run.ledger.snapshot()
    paths = [record.path for record in snapshot.transfers]
    assert paths == [PATH.QC, PATH.CWB]


def test_channels_are_aliased_in_the_order_the_paths_first_meet_them():
    shared = unbounded_path("shared", 0)
    own = unbounded_path("own", 0)
    run = Run(qc=shared, wbd=own, wsd=shared)
    report = run.ledger.traffic_json_value()
    aliases = {}
    for edge in report["semantic_edges"]:
        aliases[edge["path"]] = edge["physical_alias"]
    assert aliases["qc"] == "channel-0"
    assert aliases["wbd"] == "channel-1"
    assert aliases["wsd"] == "channel-0"
    assert aliases["sbd"] == "channel-2"
    assert report["physical_channels"][0]["member_paths"] == ["qc", "wsd"]


def test_the_traffic_json_carries_the_pinned_keys():
    boundary = unbounded_path("boundary", 0)
    run = Run(dd=boundary)
    attribution = boundary_attribution(3)
    run.send(PATH.DD, 9, 30, attribution)
    run.engine.run()
    report = run.ledger.traffic_json_value()
    transfer = report["transfers"][0]
    relation = transfer["attribution"]["relation"]
    assert set(report) == TRAFFIC_KEYS
    assert report["schema_version"] == 1
    assert set(report["semantic_edges"][0]) == SEMANTIC_EDGE_KEYS
    assert set(report["semantic_edges"][0]["counters"]) == COUNTER_KEYS
    assert set(report["physical_channels"][0]) == PHYSICAL_CHANNEL_KEYS
    assert set(report["reconciliation"][0]) == RECONCILIATION_KEYS
    assert set(transfer) == TRANSFER_KEYS
    assert set(transfer["attribution"]) == ATTRIBUTION_KEYS
    assert set(relation) == BOUNDARY_RELATION_KEYS
    assert set(relation["request_key"]) == REQUEST_KEY_KEYS
    text = json.dumps(report)
    assert json.loads(text) == report


def test_the_traffic_json_writes_a_transfers_timing_and_identity():
    bounded = bounded_path("bounded", 1.0, 10, setup_ticks=50)
    run = Run(sbd=bounded)
    attribution = requested_window(3)
    run.send(PATH.SBD, 4, 100, attribution)
    run.engine.run()
    report = run.ledger.traffic_json_value()
    transfer = report["transfers"][0]
    operation_json = message.stable_identity_json(OPERATION_ID)
    first_patch_json = message.stable_identity_json(1)
    second_patch_json = message.stable_identity_json(2)
    assert transfer["path"] == "sbd"
    assert transfer["physical_alias"] == "channel-1"
    assert transfer["payload_bits"] == 4
    assert transfer["payload_selection"] == "actual"
    assert transfer["payload_source"] == "test payload"
    assert transfer["setup_ticks"] == 50
    assert transfer["send_ticks"] == 150
    assert transfer["serializer_start_ticks"] == 150
    assert transfer["serializer_end_ticks"] == 4_000_150
    assert transfer["delivery_ticks"] == 4_000_160
    assert transfer["queue_wait_ticks"] == 0
    assert transfer["serialization_ticks"] == 4_000_000
    assert transfer["propagation_ticks"] == 10
    assert transfer["total_delay_ticks"] == 4_000_060
    assert transfer["physical_sequence"] == 0
    assert transfer["attribution"] == {
        "operation_id": operation_json,
        "patch_ids": [first_patch_json, second_patch_json],
        "window_id": 3,
        "round_lo": 1,
        "round_hi": 2,
        "relation": {
            "request_key": {
                "operation_id": operation_json,
                "window_id": 3,
                "tier": "strong",
                "run_sequence": 0,
            }
        },
    }


def test_the_traffic_json_itemizes_the_setup_per_path():
    bounded = bounded_path("bounded", 1.0, 10, setup_ticks=50)
    run = Run(sbd=bounded)
    attribution = requested_window(3)
    run.send(PATH.SBD, 4, 100, attribution)
    run.engine.run()
    report = run.ledger.traffic_json_value()
    semantic_edge = report["semantic_edges"][3]
    assert semantic_edge == {
        "path": "sbd",
        "physical_alias": "channel-1",
        "counters": {
            "transfer_count": 1,
            "known_payload_bits": 4,
            "unknown_payload_transfer_count": 0,
            "serialization_ticks": 4_000_000,
            "propagation_ticks": 10,
            "queue_wait_ticks": 0,
        },
        "setup_ticks": 50,
    }


def test_the_traffic_json_writes_a_boundary_relation():
    run = Run()
    attribution = boundary_attribution(3)
    run.send(PATH.DD, None, 30, attribution)
    run.engine.run()
    report = run.ledger.traffic_json_value()
    relation = report["transfers"][0]["attribution"]["relation"]
    source_json = message.stable_identity_json((OPERATION_ID, 3))
    destination_json = message.stable_identity_json((OPERATION_ID, 4))
    assert relation["source_window_key"] == source_json
    assert relation["destination_window_key"] == destination_json
    assert relation["source_revision"] == 2
    assert relation["delivery_revision"] == 5


def test_the_reconciliation_row_states_the_sum_and_the_channel():
    shared = unbounded_path("shared", 0)
    run = Run(qc=shared, wbd=shared)
    first_round = round_attribution(1)
    second_round = round_attribution(2)
    run.send(PATH.QC, 3, 1, first_round)
    run.send(PATH.WBD, 5, 1, second_round)
    run.engine.run()
    report = run.ledger.traffic_json_value()
    row = report["reconciliation"][0]
    assert row["member_paths"] == ["qc", "wbd"]
    assert row["reconciles"] is True
    assert row["semantic_counter_sum"] == row["physical_counters"]
    assert row["physical_counters"]["transfer_count"] == 2
    assert row["physical_counters"]["known_payload_bits"] == 8
    assert report["path_order"] == [
        "qc",
        "wbd",
        "wsd",
        "sbd",
        "wdo",
        "dd",
        "do",
        "oc",
        "cq",
    ]


def test_an_empty_ledger_reports_every_path_with_zero_counters():
    store = unbounded_path("store", 0)
    run = Run(csb=store)
    report = run.ledger.traffic_json_value()
    counts = [
        edge["counters"]["transfer_count"] for edge in report["semantic_edges"]
    ]
    assert counts == [0] * 10
    assert report["transfers"] == []
    assert report["path_order"][-1] == "csb"
