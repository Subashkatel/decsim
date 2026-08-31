"""Behavior tests for typed timing-only reaction-path links."""

from dataclasses import fields
import json
import math
from types import SimpleNamespace

import pytest

import decsim.links.links as links_module
from decsim.config import us
from decsim.links.link_profiles import (
    bandwidth_limited_profile,
    logical_reference_profile,
)
from decsim.links.links import (
    BoundaryTransferRelation,
    Link,
    LinkCapacityConfig,
    LinkConfig,
    LinkEdgeConfig,
    LinkModelConfig,
    LinkPath,
    LinkQuantityBasis,
    LinkReservation,
    PayloadSelectionSource,
    PayloadSizeConfig,
    RequestTransferRelation,
    SemanticTransferRecord,
    TrafficAttribution,
    TrafficCounters,
)
from decsim.message import (
    DecoderRequestKey,
    DecoderTier,
    stable_identity_json,
)
from decsim.links.link_traffic_report import topology_json_value, traffic_json_value


OPERATION_ID = ("experiment", 7)
PATCH_IDS = (1, 2)
PATH_ORDER = ["qc", "cwb", "wbd", "wsd", "sbd", "wdo", "dd", "do", "oc", "cq",
              "csb"]
# A shipped profile wires the nine required paths; CWB and csb are
# optional and unset.
REQUIRED_PATH_ORDER = [p for p in PATH_ORDER if p not in ("cwb", "csb")]


def make_channel(*, capacity=None, propagation_ticks=7, source="test channel"):
    return LinkConfig(propagation_ticks, capacity, source)


def make_actual_edge(*, channel=None, source="measured payload"):
    return LinkEdgeConfig(channel or make_channel(), None, source)


def make_default_edge(*, channel=None, bits=12, source="configured payload"):
    payload = PayloadSizeConfig(
        bits,
        LinkQuantityBasis.DIRECT_AGGREGATE,
        None,
        source,
    )
    return LinkEdgeConfig(channel or make_channel(), payload, None)


def make_model_config(overrides=None):
    overrides = overrides or {}
    edges = {
        path.value: overrides.get(path, make_actual_edge())
        for path in LinkPath
    }
    return LinkModelConfig(
        **edges,
        profile_name="test fabric",
        qc_excludes_controller_processing=False,
    )


def request_relation(*, tier, operation_id=OPERATION_ID, window_id=3, sequence=0):
    key = DecoderRequestKey(operation_id, window_id, tier, sequence)
    return RequestTransferRelation(key)


def valid_attribution(path):
    if path in (LinkPath.QC, LinkPath.CWB, LinkPath.CSB):
        return TrafficAttribution(OPERATION_ID, PATCH_IDS, None, 1, 2)
    if path is LinkPath.WBD:
        relation = request_relation(tier=DecoderTier.WEAK)
        return TrafficAttribution(OPERATION_ID, PATCH_IDS, 3, 1, 2, relation)
    if path in (LinkPath.WSD, LinkPath.SBD, LinkPath.DO):
        relation = request_relation(tier=DecoderTier.STRONG)
        return TrafficAttribution(OPERATION_ID, PATCH_IDS, 3, 1, 2, relation)
    if path is LinkPath.WDO:
        relation = request_relation(tier=DecoderTier.WEAK)
        return TrafficAttribution(OPERATION_ID, PATCH_IDS, 3, 1, 2, relation)
    if path is LinkPath.DD:
        key = DecoderRequestKey(OPERATION_ID, 3, DecoderTier.STRONG, 0)
        relation = BoundaryTransferRelation(
            key,
            (OPERATION_ID, 3),
            (OPERATION_ID, 4),
            2,
            5,
        )
        return TrafficAttribution(OPERATION_ID, PATCH_IDS, 3, 1, 2, relation)
    return TrafficAttribution(OPERATION_ID, PATCH_IDS, None, None, None)


def counters_from(report, path):
    return next(
        edge["counters"]
        for edge in report["semantic_edges"]
        if edge["path"] == path.value
    )


def test_quantity_basis_derives_aggregate_capacity_and_payload():
    """Direct quantities stay raw while per-channel quantities multiply by count."""
    direct_capacity = LinkCapacityConfig(
        8.0, LinkQuantityBasis.DIRECT_AGGREGATE, None, "direct"
    )
    parallel_capacity = LinkCapacityConfig(
        8.0, LinkQuantityBasis.PER_CHANNEL, 4, "parallel"
    )
    direct_payload = PayloadSizeConfig(
        9, LinkQuantityBasis.DIRECT_AGGREGATE, None, "direct"
    )
    parallel_payload = PayloadSizeConfig(
        9, LinkQuantityBasis.PER_CHANNEL, 4, "parallel"
    )

    assert direct_capacity.aggregate_bits_per_us == 8.0
    assert parallel_capacity.aggregate_bits_per_us == 32.0
    assert direct_payload.aggregate_bits == 9
    assert parallel_payload.aggregate_bits == 36
    assert parallel_capacity.to_json_value() == {
        "basis": "per_channel",
        "input_bits_per_us": 8.0,
        "channel_count": 4,
        "source": "parallel",
        "aggregate_bits_per_us": 32.0,
    }
    assert parallel_payload.to_json_value()["aggregate_bits"] == 36


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_capacity_rejects_nonpositive_or_nonfinite_values(value):
    """Capacity rejects values that cannot define finite positive service."""
    with pytest.raises(ValueError):
        LinkCapacityConfig(
            value, LinkQuantityBasis.DIRECT_AGGREGATE, None, "capacity"
        )


def test_capacity_guards_basis_count_and_aggregate_overflow():
    """Capacity guards basis-count coherence and finite derived aggregate service."""
    with pytest.raises(ValueError):
        LinkCapacityConfig(1.0, "per_channel", 2, "capacity")
    with pytest.raises(ValueError):
        LinkCapacityConfig(
            1.0, LinkQuantityBasis.DIRECT_AGGREGATE, 1, "capacity"
        )
    with pytest.raises(TypeError):
        LinkCapacityConfig(
            1.0, LinkQuantityBasis.PER_CHANNEL, None, "capacity"
        )
    normalized_bool = LinkCapacityConfig(
        1.0, LinkQuantityBasis.PER_CHANNEL, True, "capacity"
    )
    assert normalized_bool.channel_count == 1
    assert type(normalized_bool.channel_count) is int
    with pytest.raises(ValueError):
        LinkCapacityConfig(1.0, LinkQuantityBasis.PER_CHANNEL, 0, "capacity")
    with pytest.raises(ValueError):
        LinkCapacityConfig(1e308, LinkQuantityBasis.PER_CHANNEL, 2, "capacity")


def test_payload_guards_nonnegative_basis_and_count():
    """Payload configuration keeps nonnegative and basis-count coherence guards."""
    zero = PayloadSizeConfig(
        0, LinkQuantityBasis.DIRECT_AGGREGATE, None, "zero"
    )
    assert zero.aggregate_bits == 0
    with pytest.raises(ValueError):
        PayloadSizeConfig(-1, LinkQuantityBasis.DIRECT_AGGREGATE, None, "payload")
    with pytest.raises(ValueError):
        PayloadSizeConfig(1, "direct_aggregate", None, "payload")
    with pytest.raises(ValueError):
        PayloadSizeConfig(1, LinkQuantityBasis.DIRECT_AGGREGATE, 1, "payload")
    with pytest.raises(TypeError):
        PayloadSizeConfig(1, LinkQuantityBasis.PER_CHANNEL, None, "payload")
    normalized_bool = PayloadSizeConfig(
        1, LinkQuantityBasis.PER_CHANNEL, True, "payload"
    )
    assert normalized_bool.channel_count == 1
    assert type(normalized_bool.channel_count) is int
    with pytest.raises(ValueError):
        PayloadSizeConfig(1, LinkQuantityBasis.PER_CHANNEL, 0, "payload")


def test_removed_provenance_checks_and_kept_whole_normalization():
    """Provenance stays unchecked while semantically whole payloads normalize."""
    capacity = LinkCapacityConfig(
        2, LinkQuantityBasis.DIRECT_AGGREGATE, None, None
    )
    with pytest.raises(ValueError, match="input_bits"):
        PayloadSizeConfig(
            1.5, LinkQuantityBasis.DIRECT_AGGREGATE, None, object()
        )
    payload = PayloadSizeConfig(
        1, LinkQuantityBasis.DIRECT_AGGREGATE, None, object()
    )
    channel = LinkConfig(0, capacity, None)
    edge = LinkEdgeConfig(channel, payload, "")

    assert capacity.aggregate_bits_per_us == 2
    assert payload.aggregate_bits == 1
    assert edge.actual_payload_source == ""
    with pytest.raises(TypeError):
        PayloadSizeConfig(
            object(), LinkQuantityBasis.DIRECT_AGGREGATE, None, "payload"
        )


def test_physical_and_edge_configuration_keep_only_corruption_guards():
    """Physical and edge configuration retain timing and basis corruption guards."""
    normalized_bool = LinkConfig(True, None, "channel")
    assert normalized_bool.propagation_latency_ticks == 1
    assert type(normalized_bool.propagation_latency_ticks) is int
    with pytest.raises(ValueError):
        LinkConfig(-1, None, "channel")
    with pytest.raises(ValueError):
        LinkEdgeConfig(make_channel(), None, None)

    capacity = LinkCapacityConfig(
        2.0, LinkQuantityBasis.PER_CHANNEL, 2, "capacity"
    )
    channel = make_channel(capacity=capacity)
    direct_payload = PayloadSizeConfig(
        4, LinkQuantityBasis.DIRECT_AGGREGATE, None, "payload"
    )
    wrong_count = PayloadSizeConfig(
        4, LinkQuantityBasis.PER_CHANNEL, 3, "payload"
    )
    with pytest.raises(ValueError):
        LinkEdgeConfig(channel, direct_payload, None)
    with pytest.raises(ValueError):
        LinkEdgeConfig(channel, wrong_count, None)

    unchecked_nested = LinkEdgeConfig(
        SimpleNamespace(capacity=None), SimpleNamespace(), "actual"
    )
    assert unchecked_nested.actual_payload_source == "actual"


def test_attribution_deliberately_does_not_prove_uniqueness_or_provenance():
    """Attribution leaves patch uniqueness and operation provenance unproved."""
    unrelated = RequestTransferRelation(
        DecoderRequestKey("different operation", 9, DecoderTier.WEAK, 0)
    )
    attribution = TrafficAttribution(
        OPERATION_ID, (1, 1), 3, 1, 2, unrelated
    )
    assert attribution.patch_ids == (1, 1)
    assert attribution.relation is unrelated


def test_request_relation_defers_duck_snapshot_to_model_admission():
    """Request relations defer duck-record snapshotting to model admission."""
    key = DecoderRequestKey(OPERATION_ID, 3, DecoderTier.WEAK, 0)
    assert RequestTransferRelation(key).request_key is key
    duck = SimpleNamespace(
        operation_id=OPERATION_ID,
        window_id=3,
        tier=DecoderTier.WEAK,
        run_sequence=0,
    )
    assert RequestTransferRelation(duck).request_key is duck


def test_finite_fifo_reservations_obey_exact_timing_equations():
    """Finite service serializes in call order without occupying propagation time."""
    capacity = LinkCapacityConfig(
        1_000_000.0,
        LinkQuantityBasis.DIRECT_AGGREGATE,
        None,
        "one bit per tick",
    )
    link = Link(make_channel(capacity=capacity, propagation_ticks=7))

    first = link.reserve(payload_bits=10, now_ticks=100)
    second = link.reserve(payload_bits=10, now_ticks=105)
    third = link.reserve(payload_bits=10, now_ticks=117)

    assert first == LinkReservation(10, 100, 0, 10, 7, 100, 110, 17, 0)
    assert second == LinkReservation(10, 105, 5, 10, 7, 110, 120, 22, 1)
    assert third == LinkReservation(10, 117, 3, 10, 7, 120, 130, 20, 2)
    assert first.send_ticks + first.total_delay_ticks == 117
    assert second.serializer_start_ticks == first.serializer_end_ticks
    assert third.serializer_start_ticks < second.send_ticks + second.total_delay_ticks


def test_unlimited_reservations_charge_only_propagation_and_still_account():
    """Unlimited links skip serialization while retaining sequence and accounting."""
    link = Link(make_channel(capacity=None, propagation_ticks=9))
    first = link.reserve(payload_bits=None, now_ticks=4)
    second = link.reserve(payload_bits=5, now_ticks=4)

    assert first == LinkReservation(None, 4, 0, 0, 9, 4, 4, 9, 0)
    assert second == LinkReservation(5, 4, 0, 0, 9, 4, 4, 9, 1)
    assert link.counters_snapshot() == TrafficCounters(
        transfer_count=2,
        known_payload_bits=5,
        unknown_payload_transfer_count=1,
        propagation_ticks=18,
    )


def test_failed_reservations_do_not_advance_time_sequence_or_counters():
    """Only successful reservations advance monotonic time and physical state."""
    capacity = LinkCapacityConfig(
        1_000_000.0,
        LinkQuantityBasis.DIRECT_AGGREGATE,
        None,
        "finite",
    )
    link = Link(make_channel(capacity=capacity))
    first = link.reserve(payload_bits=2, now_ticks=10)

    with pytest.raises(ValueError, match="payload_bits"):
        link.reserve(payload_bits=1.5, now_ticks=20)
    with pytest.raises(TypeError):
        link.reserve(payload_bits=None, now_ticks=18)
    second = link.reserve(payload_bits=2, now_ticks=15)
    with pytest.raises(ValueError):
        link.reserve(payload_bits=2, now_ticks=14)
    third = link.reserve(payload_bits=2, now_ticks=15)

    assert [first.physical_sequence, second.physical_sequence, third.physical_sequence] == [0, 1, 2]
    assert link.counters_snapshot().transfer_count == 3
    assert link.counters_snapshot().known_payload_bits == 6


def test_counter_operations_are_exact_immutable_fieldwise_sums():
    """Counter updates return exact additive snapshots without mutating inputs."""
    original = TrafficCounters(1, 3, 0, 2, 5, 7)
    reservation = LinkReservation(None, 0, 4, 6, 8, 4, 10, 18, 1)
    updated = original.plus_reservation(reservation)
    combined = updated.plus(TrafficCounters(2, 11, 1, 13, 17, 19))

    assert original == TrafficCounters(1, 3, 0, 2, 5, 7)
    assert updated == TrafficCounters(2, 3, 1, 8, 13, 11)
    assert combined == TrafficCounters(4, 14, 2, 21, 30, 30)
    assert combined.to_json_value() == {
        "transfer_count": 4,
        "known_payload_bits": 14,
        "unknown_payload_transfer_count": 2,
        "serialization_ticks": 21,
        "propagation_ticks": 30,
        "queue_wait_ticks": 30,
    }


@pytest.mark.parametrize("path", list(LinkPath))
def test_every_semantic_path_accepts_its_real_attribution_shape(path):
    """Every semantic path accepts its documented real-record attribution shape."""
    model = make_model_config().resolve()
    reservation = model.reserve(
        path,
        payload_bits=4,
        now_ticks=10,
        attribution=valid_attribution(path),
    )
    assert reservation.physical_sequence == 0
    assert counters_from(traffic_json_value(model.snapshot()), path)["transfer_count"] == 1


@pytest.mark.parametrize(
    ("path", "attribution"),
    [
        (LinkPath.QC, TrafficAttribution(OPERATION_ID, PATCH_IDS, 3, 1, 2)),
        (LinkPath.WBD, TrafficAttribution(OPERATION_ID, PATCH_IDS, None, None, None)),
        (LinkPath.WSD, TrafficAttribution(OPERATION_ID, PATCH_IDS, None, None, None)),
        (LinkPath.SBD, TrafficAttribution(OPERATION_ID, PATCH_IDS, None, None, None)),
        (LinkPath.WDO, TrafficAttribution(OPERATION_ID, PATCH_IDS, None, None, None)),
        (LinkPath.DD, TrafficAttribution(OPERATION_ID, PATCH_IDS, None, None, None)),
        (LinkPath.DO, TrafficAttribution(OPERATION_ID, PATCH_IDS, None, None, None)),
        (LinkPath.OC, TrafficAttribution(OPERATION_ID, PATCH_IDS, None, 1, 2)),
        (LinkPath.CQ, TrafficAttribution(OPERATION_ID, PATCH_IDS, None, 1, 2)),
    ],
)
def test_semantic_paths_reject_wrong_geometry(path, attribution):
    """Each semantic path rejects attribution with the wrong window-round geometry."""
    model = make_model_config().resolve()
    with pytest.raises(ValueError):
        model.reserve(path, payload_bits=1, now_ticks=0, attribution=attribution)
    assert counters_from(traffic_json_value(model.snapshot()), path)["transfer_count"] == 0


def test_request_paths_enforce_relation_kind_and_identity():
    """Request routes need a request relation that names the attributed window."""
    model = make_model_config().resolve()
    missing = TrafficAttribution(OPERATION_ID, PATCH_IDS, 3, 1, 2)
    wrong_identity = TrafficAttribution(
        OPERATION_ID,
        PATCH_IDS,
        3,
        1,
        2,
        request_relation(
            tier=DecoderTier.STRONG,
            operation_id="another operation",
        ),
    )
    for attribution in (missing, wrong_identity):
        with pytest.raises(ValueError):
            model.reserve(
                LinkPath.WSD,
                payload_bits=1,
                now_ticks=0,
                attribution=attribution,
            )
    assert counters_from(traffic_json_value(model.snapshot()), LinkPath.WSD)["transfer_count"] == 0


def test_payload_selection_records_actual_default_and_unresolved_sources():
    """Routing records actual precedence, configured defaults, and unresolved payloads."""
    actual_channel = make_channel()
    actual_default = PayloadSizeConfig(
        99, LinkQuantityBasis.DIRECT_AGGREGATE, None, "fallback"
    )
    edges = {
        LinkPath.QC: LinkEdgeConfig(actual_channel, actual_default, "measured"),
        LinkPath.WBD: make_default_edge(bits=12, source="default"),
        LinkPath.OC: make_actual_edge(source="optional measurement"),
    }
    model = make_model_config(edges).resolve()
    model.reserve(
        LinkPath.QC,
        payload_bits=7,
        now_ticks=0,
        attribution=valid_attribution(LinkPath.QC),
    )
    model.reserve(
        LinkPath.WBD,
        payload_bits=None,
        now_ticks=0,
        attribution=valid_attribution(LinkPath.WBD),
    )
    model.reserve(
        LinkPath.OC,
        payload_bits=None,
        now_ticks=0,
        attribution=valid_attribution(LinkPath.OC),
    )

    transfers = traffic_json_value(model.snapshot())["transfers"]
    assert [row["payload_bits"] for row in transfers] == [7, 12, None]
    assert [row["payload_selection"] for row in transfers] == [
        PayloadSelectionSource.ACTUAL.value,
        PayloadSelectionSource.CONFIGURED_DEFAULT.value,
        PayloadSelectionSource.UNRESOLVED.value,
    ]
    assert [row["payload_source"] for row in transfers] == [
        "measured",
        "default",
        "optional measurement",
    ]


def test_payload_admission_failures_leave_semantic_and_physical_state_untouched():
    """Payload policy failures do not create reservations or semantic ledger rows."""
    capacity = LinkCapacityConfig(
        1_000_000.0,
        LinkQuantityBasis.DIRECT_AGGREGATE,
        None,
        "finite",
    )
    default_only = make_default_edge(bits=4)
    unresolved_finite = LinkEdgeConfig(
        make_channel(capacity=capacity), None, "optional actual"
    )
    model = make_model_config(
        {LinkPath.QC: default_only, LinkPath.OC: unresolved_finite}
    ).resolve()

    with pytest.raises(ValueError):
        model.reserve(
            LinkPath.QC,
            payload_bits=3,
            now_ticks=20,
            attribution=valid_attribution(LinkPath.QC),
        )
    with pytest.raises(TypeError):
        model.reserve(
            LinkPath.OC,
            payload_bits=None,
            now_ticks=20,
            attribution=valid_attribution(LinkPath.OC),
        )
    accepted = model.reserve(
        LinkPath.QC,
        payload_bits=None,
        now_ticks=10,
        attribution=valid_attribution(LinkPath.QC),
    )

    report = traffic_json_value(model.snapshot())
    assert accepted.physical_sequence == 0
    assert len(report["transfers"]) == 1
    assert counters_from(report, LinkPath.OC)["transfer_count"] == 0


def test_config_identity_controls_sharing_and_each_resolve_is_run_owned():
    """Only identical channel objects share FIFO state and resolves stay independent."""
    capacity = LinkCapacityConfig(
        1_000_000.0,
        LinkQuantityBasis.DIRECT_AGGREGATE,
        None,
        "finite",
    )
    shared = make_channel(capacity=capacity)
    equal_but_distinct = make_channel(capacity=capacity)
    config = make_model_config(
        {
            LinkPath.QC: make_actual_edge(channel=shared),
            LinkPath.WBD: make_actual_edge(channel=shared),
            LinkPath.OC: make_actual_edge(channel=equal_but_distinct),
        }
    )
    first_run = config.resolve()
    second_run = config.resolve()

    qc = first_run.reserve(
        LinkPath.QC,
        payload_bits=10,
        now_ticks=0,
        attribution=valid_attribution(LinkPath.QC),
    )
    wbd = first_run.reserve(
        LinkPath.WBD,
        payload_bits=10,
        now_ticks=0,
        attribution=valid_attribution(LinkPath.WBD),
    )
    oc = first_run.reserve(
        LinkPath.OC,
        payload_bits=10,
        now_ticks=0,
        attribution=valid_attribution(LinkPath.OC),
    )
    fresh = second_run.reserve(
        LinkPath.QC,
        payload_bits=10,
        now_ticks=0,
        attribution=valid_attribution(LinkPath.QC),
    )

    assert qc.physical_sequence == 0
    assert wbd.physical_sequence == 1
    assert wbd.queue_wait_ticks == qc.serialization_ticks
    assert oc.physical_sequence == 0
    assert fresh.physical_sequence == 0
    topology = topology_json_value(first_run.snapshot())
    aliases = {edge["path"]: edge["physical_alias"] for edge in topology["edges"]}
    assert aliases["qc"] == aliases["wbd"]
    assert aliases["qc"] != aliases["oc"]


def test_shared_fifo_counters_reconcile_across_member_paths():
    """Shared physical counters equal the fieldwise sum of their semantic paths."""
    shared = make_channel()
    model = make_model_config(
        {
            LinkPath.QC: make_actual_edge(channel=shared),
            LinkPath.WBD: make_actual_edge(channel=shared),
        }
    ).resolve()
    model.reserve(
        LinkPath.QC,
        payload_bits=3,
        now_ticks=1,
        attribution=valid_attribution(LinkPath.QC),
    )
    model.reserve(
        LinkPath.WBD,
        payload_bits=5,
        now_ticks=1,
        attribution=valid_attribution(LinkPath.WBD),
    )

    report = traffic_json_value(model.snapshot())
    shared_row = next(
        row for row in report["reconciliation"]
        if row["member_paths"] == ["qc", "wbd"]
    )
    assert shared_row["reconciles"] is True
    assert shared_row["semantic_counter_sum"] == shared_row["physical_counters"]
    assert shared_row["physical_counters"]["transfer_count"] == 2
    assert shared_row["physical_counters"]["known_payload_bits"] == 8


def test_reconciliation_guard_rejects_silent_counter_divergence():
    """Traffic reporting rejects divergence between semantic and physical totals."""
    model = make_model_config().resolve()
    model.reserve(
        LinkPath.QC,
        payload_bits=3,
        now_ticks=0,
        attribution=valid_attribution(LinkPath.QC),
    )
    model._semantic_counters[LinkPath.QC] = TrafficCounters()
    with pytest.raises(RuntimeError, match="do not reconcile"):
        traffic_json_value(model.snapshot())


def test_topology_json_reports_stable_fabric():
    """Topology evidence reports ordered routing, channel policy, and provenance."""
    capacity = LinkCapacityConfig(
        4.0, LinkQuantityBasis.PER_CHANNEL, 2, "capacity source"
    )
    payload = PayloadSizeConfig(
        6, LinkQuantityBasis.PER_CHANNEL, 2, "payload source"
    )
    shared = make_channel(capacity=capacity, propagation_ticks=11, source="wire")
    edge = LinkEdgeConfig(shared, payload, "actual source")
    model = make_model_config({LinkPath.QC: edge, LinkPath.WBD: edge}).resolve()

    topology = topology_json_value(model.snapshot())
    assert topology["schema_version"] == 1
    assert topology["profile_name"] == "test fabric"
    assert topology["path_order"] == PATH_ORDER
    assert topology["cancellation_semantics"] == "non_preemptive_irrevocable"
    qc_edge = topology["edges"][0]
    assert qc_edge["actual_payload_source"] == "actual source"
    assert qc_edge["default_payload"]["aggregate_bits"] == 12
    channel = topology["physical_channels"][0]
    assert channel["member_paths"] == ["qc", "wbd"]
    assert channel["propagation_latency_ticks"] == 11
    assert channel["capacity"]["aggregate_bits_per_us"] == 8.0
    assert channel["configuration_source"] == "wire"
    assert channel["service_scope"] == "aggregate_fifo"


def test_traffic_json_preserves_typed_identity_and_timing_boundaries():
    """Traffic evidence serializes typed request and boundary identities with timing."""
    model = make_model_config().resolve()
    request = valid_attribution(LinkPath.WSD)
    boundary = valid_attribution(LinkPath.DD)
    first = model.reserve(
        LinkPath.WSD,
        payload_bits=8,
        now_ticks=20,
        attribution=request,
    )
    second = model.reserve(
        LinkPath.DD,
        payload_bits=9,
        now_ticks=30,
        attribution=boundary,
    )

    report = traffic_json_value(model.snapshot())
    assert report["schema_version"] == 1
    assert report["path_order"] == PATH_ORDER
    assert [row["path"] for row in report["transfers"]] == ["wsd", "dd"]
    request_json, boundary_json = report["transfers"]
    assert request_json["attribution"]["operation_id"] == stable_identity_json(OPERATION_ID)
    assert request_json["attribution"]["patch_ids"] == [
        stable_identity_json(1), stable_identity_json(2)
    ]
    assert request_json["attribution"]["relation"]["request_key"] == {
        "operation_id": stable_identity_json(OPERATION_ID),
        "window_id": 3,
        "tier": "strong",
        "run_sequence": 0,
    }
    relation_json = boundary_json["attribution"]["relation"]
    assert relation_json["source_window_key"] == stable_identity_json((OPERATION_ID, 3))
    assert relation_json["destination_window_key"] == stable_identity_json((OPERATION_ID, 4))
    assert relation_json["source_revision"] == 2
    assert relation_json["delivery_revision"] == 5
    assert request_json["payload_bits"] == first.payload_bits
    assert boundary_json["payload_bits"] == second.payload_bits
    assert request_json["delivery_ticks"] == first.send_ticks + first.total_delay_ticks
    assert json.loads(json.dumps(report)) == report


def test_reference_profile_has_the_exact_timing_only_project_metadata():
    """The reference profile preserves its nine timing and payload configuration choices."""
    config = logical_reference_profile()
    model = config.resolve()
    topology = topology_json_value(model.snapshot())
    expected_propagation = {
        "qc": us(0.15),
        "wbd": us(2.0),
        "wsd": us(0.5),
        "sbd": us(2.0),
        "wdo": us(1.0),
        "dd": us(0.5),
        "do": us(1.0),
        "oc": us(4.0),
        "cq": us(0.15),
    }
    expected_defaults = {
        "wdo": 5_000_000,
        "dd": 100,
        "do": 5_000_000,
        "oc": 20_000_000,
        "cq": 5_000_000,
    }
    actual_sources = {
        "qc": "SyndromePayload.size_bits",
        "wbd": "SyndromeRoundPacket.fragment_size_sum",
        "wsd": "switching decision payload_bits",
        "sbd": "DecodeJob.retained_payload_size_bits",
    }

    assert config.profile_name == "logical_reference"
    assert config.qc_excludes_controller_processing is False
    assert topology["path_order"] == REQUIRED_PATH_ORDER
    assert len(topology["physical_channels"]) == 9
    assert all(channel["capacity"] is None for channel in topology["physical_channels"])
    propagation = {
        channel["member_paths"][0]: channel["propagation_latency_ticks"]
        for channel in topology["physical_channels"]
    }
    defaults = {
        edge["path"]: edge["default_payload"]["aggregate_bits"]
        for edge in topology["edges"]
        if edge["default_payload"] is not None
    }
    sources = {
        edge["path"]: edge["actual_payload_source"]
        for edge in topology["edges"]
        if edge["actual_payload_source"] is not None
    }
    assert propagation == expected_propagation
    assert defaults == expected_defaults
    assert sources == actual_sources


def test_config_and_record_validation_nonchecks_remain_at_their_boundaries():
    """Fabric and ledger records retain deliberate constructor nonchecks."""
    loose_config = LinkModelConfig(
        qc=None,
        wbd=None,
        wsd=None,
        sbd=None,
        wdo=None,
        dd=None,
        do=None,
        oc=None,
        cq=None,
        profile_name=None,
        qc_excludes_controller_processing=False,
    )
    reservation = LinkReservation(None, -1, -2, -3, -4, -5, -6, -7, -8)
    counters = TrafficCounters(-1, -2, -3, -4, -5, -6)
    record = SemanticTransferRecord(
        path="not a path",
        physical_alias=None,
        attribution=None,
        payload_selection="not a selection",
        payload_source=None,
        reservation=reservation,
    )

    assert loose_config.profile_name is None
    assert counters.transfer_count == -1
    assert record.reservation is reservation
    permissive_flag = LinkModelConfig(
        **{path.value: make_actual_edge() for path in LinkPath},
        profile_name="profile",
        qc_excludes_controller_processing=1,
    )
    assert permissive_flag.qc_excludes_controller_processing == 1


def test_deleted_helpers_factories_fields_and_shims_are_absent():
    """Removed construction conveniences, fields, and accessors have no shims."""
    assert not hasattr(links_module, "_require_nonempty_stable_string")
    assert not hasattr(LinkEdgeConfig, "from_per_channel_transaction")
    assert not hasattr(make_model_config().resolve(), "config")
    assert "payload_bits" not in {field.name for field in fields(SemanticTransferRecord)}
    assert hasattr(Link, "config")


def test_links_expose_timing_only_without_scheduler_or_reclamation_ownership():
    """Links return timing records and expose no scheduling or reclamation operations."""
    link = Link(make_channel())
    reservation = link.reserve(payload_bits=1, now_ticks=0)
    assert type(reservation) is LinkReservation
    for name in (
        "schedule",
        "send",
        "deliver",
        "acknowledge",
        "cancel",
        "release",
        "preempt",
        "set_priority",
    ):
        assert not hasattr(link, name)



def test_bandwidth_profile_declares_finite_calibrated_capacities():
    """The bandwidth profile exposes all calibrated capacities and fallback payloads."""
    config = bandwidth_limited_profile()
    topology = topology_json_value(config.resolve().snapshot())
    expected_capacities = {
        "qc": (24.0, "direct_aggregate", None, 24.0),
        "wbd": (48.0, "direct_aggregate", None, 48.0),
        "wsd": (24.0, "direct_aggregate", None, 24.0),
        "sbd": (72.0, "direct_aggregate", None, 72.0),
        "wdo": (10_000.0, "per_channel", 100, 1_000_000.0),
        "dd": (24.0, "direct_aggregate", None, 24.0),
        "do": (10_000.0, "per_channel", 100, 1_000_000.0),
        "oc": (4_000.0, "per_channel", 1_000, 4_000_000.0),
        "cq": (0.2, "per_channel", 5_000_000, 1_000_000.0),
    }
    expected_fallbacks = {
        "qc": (24, "direct_aggregate", None, 24),
        "wbd": (240, "direct_aggregate", None, 240),
        "wsd": (1, "direct_aggregate", None, 1),
        "sbd": (360, "direct_aggregate", None, 360),
        "wdo": (50_000, "per_channel", 100, 5_000_000),
        "dd": (100, "direct_aggregate", None, 100),
        "do": (50_000, "per_channel", 100, 5_000_000),
        "oc": (20_000, "per_channel", 1_000, 20_000_000),
        "cq": (1, "per_channel", 5_000_000, 5_000_000),
    }
    capacities = {
        channel["member_paths"][0]: (
            channel["capacity"]["input_bits_per_us"],
            channel["capacity"]["basis"],
            channel["capacity"]["channel_count"],
            channel["capacity"]["aggregate_bits_per_us"],
        )
        for channel in topology["physical_channels"]
    }
    fallbacks = {
        edge["path"]: (
            edge["default_payload"]["input_bits"],
            edge["default_payload"]["basis"],
            edge["default_payload"]["channel_count"],
            edge["default_payload"]["aggregate_bits"],
        )
        for edge in topology["edges"]
    }

    assert config.profile_name == "bandwidth_limited"
    assert config.qc_excludes_controller_processing is False
    assert topology["path_order"] == REQUIRED_PATH_ORDER
    assert len(topology["physical_channels"]) == 9
    assert capacities == expected_capacities
    assert fallbacks == expected_fallbacks
    assert all(math.isfinite(values[0]) for values in capacities.values())
    assert json.loads(json.dumps(topology)) == topology


def test_bandwidth_profile_preserves_reference_latency_and_semantic_parameters():
    """The reference profile stays pure latency while shared semantic parameters match."""
    reference = logical_reference_profile()
    bandwidth = bandwidth_limited_profile()
    reference_topology = topology_json_value(reference.resolve().snapshot())
    bandwidth_topology = topology_json_value(bandwidth.resolve().snapshot())

    def propagation_by_path(topology):
        return {
            channel["member_paths"][0]: channel["propagation_latency_ticks"]
            for channel in topology["physical_channels"]
        }

    reference_edges = {edge["path"]: edge for edge in reference_topology["edges"]}
    bandwidth_edges = {edge["path"]: edge for edge in bandwidth_topology["edges"]}
    configured_default_paths = ("wdo", "dd", "do", "oc", "cq")
    actual_payload_paths = ("qc", "wbd", "wsd", "sbd")

    assert reference.profile_name == "logical_reference"
    assert reference.qc_excludes_controller_processing is False
    assert len(reference_topology["physical_channels"]) == 9
    assert all(
        channel["capacity"] is None
        for channel in reference_topology["physical_channels"]
    )
    assert propagation_by_path(reference_topology) == propagation_by_path(
        bandwidth_topology
    )
    payload_parameters = ("input_bits", "basis", "channel_count", "aggregate_bits")
    assert {
        path: tuple(
            reference_edges[path]["default_payload"][parameter]
            for parameter in payload_parameters
        )
        for path in configured_default_paths
    } == {
        path: tuple(
            bandwidth_edges[path]["default_payload"][parameter]
            for parameter in payload_parameters
        )
        for path in configured_default_paths
    }
    assert {
        path: reference_edges[path]["actual_payload_source"]
        for path in actual_payload_paths
    } == {
        path: bandwidth_edges[path]["actual_payload_source"]
        for path in actual_payload_paths
    }


def test_bandwidth_profile_serializes_and_queues_in_physical_fifo_order():
    """Finite QC transfers serialize FIFO and WSD uses its configured fallback."""
    model = bandwidth_limited_profile().resolve()
    first = model.reserve(
        LinkPath.QC,
        payload_bits=24,
        now_ticks=0,
        attribution=valid_attribution(LinkPath.QC),
    )
    second = model.reserve(
        LinkPath.QC,
        payload_bits=24,
        now_ticks=0,
        attribution=valid_attribution(LinkPath.QC),
    )
    wsd = model.reserve(
        LinkPath.WSD,
        payload_bits=None,
        now_ticks=0,
        attribution=valid_attribution(LinkPath.WSD),
    )
    traffic = traffic_json_value(model.snapshot())
    qc_transfers = [
        transfer for transfer in traffic["transfers"]
        if transfer["path"] == "qc"
    ]
    wsd_transfer = next(
        transfer for transfer in traffic["transfers"]
        if transfer["path"] == "wsd"
    )
    qc_channel = next(
        channel for channel in traffic["physical_channels"]
        if channel["member_paths"] == ["qc"]
    )

    assert first.serialization_ticks == us(1.0)
    assert first.queue_wait_ticks == 0
    assert first.serializer_start_ticks == 0
    assert first.propagation_ticks == us(0.15)
    assert first.total_delay_ticks == us(1.15)
    assert second.serializer_start_ticks == first.serializer_end_ticks
    assert second.queue_wait_ticks == us(1.0)
    assert second.serialization_ticks == us(1.0)
    assert second.total_delay_ticks == us(2.15)
    assert second.physical_sequence == 1
    assert wsd.payload_bits == 1
    assert wsd.serialization_ticks == us(1 / 24.0)
    assert wsd_transfer["payload_selection"] == "configured_default"
    assert qc_transfers[1]["serializer_start_ticks"] >= qc_transfers[0][
        "serializer_end_ticks"
    ]
    assert qc_channel["counters"]["transfer_count"] == 2
    assert qc_channel["counters"]["known_payload_bits"] == 48
    assert qc_channel["counters"]["serialization_ticks"] == sum(
        transfer["serialization_ticks"] for transfer in qc_transfers
    )
    assert qc_channel["counters"]["queue_wait_ticks"] == sum(
        transfer["queue_wait_ticks"] for transfer in qc_transfers
    )
    assert all(row["reconciles"] is True for row in traffic["reconciliation"])
    assert json.loads(json.dumps(traffic)) == traffic


def test_bandwidth_profile_capacity_scale_moves_the_contention_regime():
    """Capacity scaling changes service time and rejects invalid scale values."""
    base_capacities = {
        "qc": 24.0,
        "wbd": 48.0,
        "wsd": 24.0,
        "sbd": 72.0,
        "wdo": 1_000_000.0,
        "dd": 24.0,
        "do": 1_000_000.0,
        "oc": 4_000_000.0,
        "cq": 1_000_000.0,
    }

    def aggregate_capacities(scale):
        topology = topology_json_value(
            bandwidth_limited_profile(capacity_scale=scale).resolve().snapshot())
        return {
            channel["member_paths"][0]: channel["capacity"][
                "aggregate_bits_per_us"
            ]
            for channel in topology["physical_channels"]
        }

    slow_model = bandwidth_limited_profile(
        capacity_scale=0.5
    ).resolve()
    fast_model = bandwidth_limited_profile(
        capacity_scale=2.0
    ).resolve()
    slow = slow_model.reserve(
        LinkPath.QC,
        payload_bits=24,
        now_ticks=0,
        attribution=valid_attribution(LinkPath.QC),
    )
    fast = fast_model.reserve(
        LinkPath.QC,
        payload_bits=24,
        now_ticks=0,
        attribution=valid_attribution(LinkPath.QC),
    )

    assert aggregate_capacities(0.5) == {
        path: capacity * 0.5 for path, capacity in base_capacities.items()
    }
    assert aggregate_capacities(2.0) == {
        path: capacity * 2.0 for path, capacity in base_capacities.items()
    }
    assert slow.serialization_ticks == us(2.0)
    assert fast.serialization_ticks == us(0.5)
    for invalid_scale in (0.0, -1.0, math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError):
            bandwidth_limited_profile(
                capacity_scale=invalid_scale
            )


# ---- per-transfer setup overhead (gem5-Aladdin engine-side rule) ----------

def _overhead_edge(overhead_ticks, *, channel=None):
    from decsim.links.links import TransferOverheadConfig
    return LinkEdgeConfig(
        channel or make_channel(), None, "measured payload",
        transfer_overhead=TransferOverheadConfig(overhead_ticks, "test setup"))


def test_transfer_overhead_delays_delivery_without_occupying_the_wire():
    from decsim.links.links import LinkCapacityConfig, LinkQuantityBasis
    capacity = LinkCapacityConfig(
        1.0, LinkQuantityBasis.DIRECT_AGGREGATE, None, "test rate")
    channel = make_channel(capacity=capacity, propagation_ticks=10)
    model = make_model_config({LinkPath.SBD: _overhead_edge(
        50, channel=channel)}).resolve()
    first = model.reserve(LinkPath.SBD, payload_bits=4, now_ticks=100,
                          attribution=valid_attribution(LinkPath.SBD))
    # setup 50 ticks (engine-side) + serialization 4 bits / 1 bit-per-us
    # + propagation 10 ticks
    assert first.send_ticks == 150            # the wire sees the shifted send
    assert first.serialization_ticks == us(4.0)
    assert first.total_delay_ticks == 50 + us(4.0) + 10


def test_setups_serialize_like_aladdins_single_dma_event():
    model = make_model_config(
        {LinkPath.SBD: _overhead_edge(50)}).resolve()
    first = model.reserve(LinkPath.SBD, payload_bits=1, now_ticks=100,
                          attribution=valid_attribution(LinkPath.SBD))
    second = model.reserve(LinkPath.SBD, payload_bits=1, now_ticks=100,
                           attribution=valid_attribution(LinkPath.SBD))
    # one CPU programs the engine: the second setup starts when the first ends
    assert first.total_delay_ticks == 50 + 7
    assert second.total_delay_ticks == 100 + 7


def test_zero_overhead_edges_reserve_identically_to_plain_edges():
    plain = make_model_config().resolve()
    with_field = make_model_config(
        {LinkPath.SBD: _overhead_edge(0)}).resolve()
    a = plain.reserve(LinkPath.SBD, payload_bits=4, now_ticks=10,
                      attribution=valid_attribution(LinkPath.SBD))
    b = with_field.reserve(LinkPath.SBD, payload_bits=4, now_ticks=10,
                           attribution=valid_attribution(LinkPath.SBD))
    assert a == b


def test_shared_channel_with_mixed_overhead_is_refused():
    shared = make_channel()
    overrides = {LinkPath.SBD: _overhead_edge(50, channel=shared),
                 LinkPath.WBD: make_actual_edge(channel=shared)}
    with pytest.raises(ValueError, match="transfer overhead differs"):
        make_model_config(overrides).resolve()


def test_with_transfer_overhead_helper_covers_the_dma_paths():
    from decsim.links.link_profiles import (logical_reference_profile,
                                            with_transfer_overhead)
    card = with_transfer_overhead(
        logical_reference_profile(), overhead_us=0.4,
        source="Shao MICRO 2016 measured 400 ns per transaction")
    assert card.wbd.transfer_overhead is not None
    assert card.sbd.transfer_overhead is not None
    assert card.qc.transfer_overhead is None
    assert card.wbd.transfer_overhead.overhead_ticks == us(0.4)


def test_yaml_card_key_reaches_the_edge(tmp_path):
    from experiments.build_run import link_model
    from experiments.experiment_config import load_experiment
    yaml_text = (
        "mode: weak_baseline\n"
        "code_task: surface_code:rotated_memory_z\n"
        "rounds_per_shot: 15\n"
        "windowing: {scheme: sliding, commit_rounds: null, buffer_rounds: null}\n"
        "sweep: [{physical_error_probability: [0.001], distance: [3],\n"
        "         round_period_us: [1.0], shots: 1}]\n"
        "controller: {clock: fridge, t_binary_availability_cycles: 0, t_pack_cycles: 0}\n"
        "clocks: {fridge: 250.0, room: 250.0}\n"
        "links:\n"
        "  qc:  {latency_cycles: 250, clock: fridge, bits_per_cycle: null}\n"
        "  cwb: {latency_cycles: 125, clock: fridge, bits_per_cycle: 400.0}\n"
        "  wbd: {latency_cycles: 250, clock: fridge, bits_per_cycle: null,\n"
        "        transfer_overhead_cycles: 100}\n"
        "  dd:  {latency_cycles: 125, clock: fridge, bits_per_cycle: null}\n"
        "  wdo: {latency_cycles: 250, clock: fridge, bits_per_cycle: null}\n"
        "buffers: {buffer_0_size: null, buffer_1_size: null,\n"
        "          packing_workspace_size: null}\n"
        "decoder:\n"
        "  weak:\n"
        "    algorithm: 0.028\n"
        "    units: 1\n"
        "    unit_buffer_size: null\n"
        "    engine: {clock: fridge, fetch_cycles_per_round: 1, release_cycles_per_job: 1}\n"
        "pauli_frame: {clock: fridge, commit_cycles: 1}\n")
    (tmp_path / "overhead_card.yaml").write_text(yaml_text)
    card = link_model(load_experiment(tmp_path / "overhead_card.yaml"))
    assert card.wbd.transfer_overhead is not None
    assert card.wbd.transfer_overhead.overhead_ticks == us(0.4)
    assert card.dd.transfer_overhead is None


def test_serialization_never_ends_before_the_exact_transmission_time():
    """LINK-002 causality: a fractional-tick serialization rounds UP, never
    down. One bit at 3 bits/us takes 333333.33... ticks; a reservation that
    ends at 333333 finishes before the bit has fully left the serializer."""
    from fractions import Fraction

    from decsim.config import TICKS_PER_US

    cases = [
        (1, 3.0, 333_334),           # fractional: must round up
        (1, 7.0, 142_858),           # fractional: must round up
        (10, 1_000_000.0, 10),       # exact: must NOT inflate
        (8, 2.0, 4_000_000),         # exact: must NOT inflate
        (1, 10.0, 100_000),          # decimal-exact rate: must NOT inflate
    ]
    for payload_bits, rate, expected_ticks in cases:
        capacity = LinkCapacityConfig(
            rate, LinkQuantityBasis.DIRECT_AGGREGATE, None, "causality case")
        link = Link(make_channel(capacity=capacity, propagation_ticks=0))
        reservation = link.reserve(payload_bits=payload_bits, now_ticks=0)
        exact_ticks = Fraction(payload_bits) * TICKS_PER_US / Fraction(rate)
        assert reservation.serialization_ticks >= exact_ticks, \
            f"{payload_bits} bits at {rate} bits/us ends early"
        assert reservation.serialization_ticks - exact_ticks < 1, \
            f"{payload_bits} bits at {rate} bits/us inflated by a full tick"
        assert reservation.serialization_ticks == expected_ticks
