from dataclasses import replace
import json
import math
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from decsim.config import us
from decsim.links import (
    BoundaryTransferRelation,
    Link,
    LinkCapacityConfig,
    LinkConfig,
    LinkEdgeConfig,
    LinkModel,
    LinkModelConfig,
    LinkPath,
    LinkQuantityBasis,
    PayloadSelectionSource,
    PayloadSizeConfig,
    RequestTransferRelation,
    TrafficAttribution,
)
from decsim.message import DecoderRequestKey, DecoderTier, Operation
from decsim.run_spec import RunSpec, simulate
from decsim.planner import FixedRounds
from decsim.decoders import (
    PerRoundDecoder,
    SAMPLED_CONFIDENCE_SOURCE,
    SampledConfidenceDecoder,
    SwitchingRouter,
)
from decsim.schemes import SlidingWindowScheme
from decsim.switching import Switching


def _attribution(relation=None) -> TrafficAttribution:
    return TrafficAttribution(
        operation_id=("stream", 2),
        patch_ids=(1, "north"),
        window_id=0,
        round_lo=1,
        round_hi=3,
        relation=relation,
    )


REQUEST_KEY = DecoderRequestKey(("stream", 2), 0, DecoderTier.STRONG, 0)
REQUEST_RELATION = RequestTransferRelation(REQUEST_KEY)
BOUNDARY_RELATION = BoundaryTransferRelation(REQUEST_KEY, (("stream", 2), 0),
                                             (("stream", 2), 1), 1, 1)
WRONG_RELATION = RequestTransferRelation(DecoderRequestKey("other", 0, DecoderTier.STRONG, 1))


def test_request_and_boundary_relations_serialize_exact_identity_and_revisions():
    links = LinkModelConfig.reference_fixed_latency_profile().resolve()
    links.reserve(LinkPath.WSD, payload_bits=None, now_ticks=3,
                  attribution=_attribution(REQUEST_RELATION))
    links.reserve(LinkPath.DD, payload_bits=None, now_ticks=5,
                  attribution=_attribution(BOUNDARY_RELATION))

    request, boundary = [row["attribution"]["relation"]
                         for row in links.traffic_json_value()["transfers"]]
    assert request == {"request_key": {
        "operation_id": {"kind": "tuple", "value": None, "items": [
            {"kind": "string", "value": "stream", "items": None},
            {"kind": "integer", "value": "2", "items": None}]},
        "window_id": 0, "tier": "strong", "run_sequence": 0}}
    assert boundary["request_key"] == request["request_key"]
    assert boundary["source_window_key"] == (("stream", 2), 0)
    assert boundary["destination_window_key"] == (("stream", 2), 1)
    assert (boundary["source_revision"], boundary["delivery_revision"]) == (1, 1)
    json.dumps(links.traffic_json_value())


def _round_attribution() -> TrafficAttribution:
    return TrafficAttribution(
        operation_id=("stream", 2),
        patch_ids=(1, "north"),
        window_id=None,
        round_lo=2,
        round_hi=2,
    )


def _operation_attribution() -> TrafficAttribution:
    return TrafficAttribution(
        operation_id=("stream", 2),
        patch_ids=(1, "north"),
        window_id=None,
        round_lo=None,
        round_hi=None,
    )


def test_default_profile_has_complete_distinct_topology():
    config = LinkModelConfig.reference_fixed_latency_profile()
    links = config.resolve()
    topology = links.topology_json_value(
        controller_link_integration_assurance="shipped_controller"
    )

    assert topology["path_order"] == [path.value for path in LinkPath]
    assert [record["propagation_latency_ticks"] for record in
            topology["physical_channels"]] == [
        us(0.15), us(2.0), us(0.5), us(2.0), us(1.0),
        us(0.5), us(1.0), us(4.0), us(0.15),
    ]
    assert [record["physical_alias"] for record in topology["edges"]] == [
        f"channel-{index}" for index in range(9)
    ]
    assert topology["edges"][4]["default_payload"]["aggregate_bits"] == 5_000_000
    assert topology["cancellation_semantics"] == "non_preemptive_irrevocable"


def test_infinite_link_accepts_unknown_size_without_contention():
    link = Link(LinkConfig(us(2.0), None, "hand oracle"))

    first = link.reserve(payload_bits=None, now_ticks=0)
    second = link.reserve(payload_bits=None, now_ticks=0)

    assert first.total_delay_ticks == second.total_delay_ticks == us(2.0)
    assert first.queue_wait_ticks == second.queue_wait_ticks == 0
    assert link.counters_snapshot().unknown_payload_transfer_count == 2


@pytest.mark.parametrize(
    "make",
    [
        lambda: LinkConfig(True, None, "source"),
        lambda: LinkConfig(-1, None, "source"),
        lambda: LinkConfig(0, None, ""),
        lambda: LinkCapacityConfig(1, LinkQuantityBasis.DIRECT_AGGREGATE, None, "x"),
        lambda: LinkCapacityConfig(math.nan, LinkQuantityBasis.DIRECT_AGGREGATE, None, "x"),
        lambda: LinkCapacityConfig(1.0, LinkQuantityBasis.PER_CHANNEL, True, "x"),
        lambda: PayloadSizeConfig(True, LinkQuantityBasis.DIRECT_AGGREGATE, None, "x"),
        lambda: PayloadSizeConfig(1, LinkQuantityBasis.PER_CHANNEL, 0, "x"),
    ],
)
def test_link_configuration_rejects_non_exact_or_invalid_values(make):
    with pytest.raises((TypeError, ValueError)):
        make()


def test_per_channel_constructor_retains_and_symmetrically_derives_operands():
    edge = LinkEdgeConfig.from_per_channel_transaction(
        propagation_latency_ticks=us(1.0),
        per_channel_capacity_bits_per_us=2500.0,
        per_channel_payload_bits=50_000,
        channel_count=100,
        capacity_source="published per-channel capacity",
        payload_source="published per-channel payload",
        configuration_source="paper profile",
    )

    assert edge.channel.capacity.input_bits_per_us == 2500.0
    assert edge.channel.capacity.aggregate_bits_per_us == 250_000.0
    assert edge.default_payload.input_bits == 50_000
    assert edge.default_payload.aggregate_bits == 5_000_000
    assert edge.channel.capacity.channel_count == edge.default_payload.channel_count == 100


def test_exact_aliases_contend_but_equal_distinct_configs_do_not():
    capacity = LinkCapacityConfig(
        1000.0, LinkQuantityBasis.DIRECT_AGGREGATE, None, "hand oracle"
    )
    shared = LinkConfig(0, capacity, "shared")
    equal_but_distinct = LinkConfig(0, capacity, "shared")
    default = PayloadSizeConfig(
        1000, LinkQuantityBasis.DIRECT_AGGREGATE, None, "default"
    )
    profile = LinkModelConfig.reference_fixed_latency_profile()
    config = replace(
        profile,
        cwd=LinkEdgeConfig(shared, default, None),
        csd=LinkEdgeConfig(shared, default, None),
        dd=LinkEdgeConfig(equal_but_distinct, default, None),
    )
    first_run = config.resolve()
    second_run = config.resolve()

    cwd = first_run.reserve(
        LinkPath.CWD,
        payload_bits=None,
        now_ticks=0,
        attribution=_round_attribution(),
    )
    csd = first_run.reserve(
        LinkPath.CSD, payload_bits=None, now_ticks=0,
        attribution=_attribution(REQUEST_RELATION)
    )
    dd = first_run.reserve(
        LinkPath.DD, payload_bits=None, now_ticks=0,
        attribution=_attribution(BOUNDARY_RELATION)
    )
    fresh = second_run.reserve(
        LinkPath.CSD, payload_bits=None, now_ticks=0,
        attribution=_attribution(REQUEST_RELATION)
    )

    assert cwd.total_delay_ticks == us(1.0)
    assert csd.queue_wait_ticks == us(1.0)
    assert dd.queue_wait_ticks == 0
    assert fresh.queue_wait_ticks == 0


def test_default_only_actual_override_is_rejected_before_all_mutation():
    links = LinkModelConfig.reference_fixed_latency_profile().resolve()
    before = links.traffic_json_value()

    with pytest.raises(ValueError, match="does not declare"):
        links.reserve(
            LinkPath.DD,
            payload_bits=100,
            now_ticks=0,
            attribution=_attribution(BOUNDARY_RELATION),
        )

    assert links.traffic_json_value() == before


def test_actual_default_and_unresolved_provenance_are_distinct_and_reconcile():
    links = LinkModelConfig.reference_fixed_latency_profile().resolve()
    actual = links.reserve(
        LinkPath.QC,
        payload_bits=17,
        now_ticks=0,
        attribution=_round_attribution(),
    )
    default = links.reserve(
        LinkPath.DD,
        payload_bits=None,
        now_ticks=0,
        attribution=_attribution(BOUNDARY_RELATION),
    )
    unresolved = links.reserve(
        LinkPath.WSD,
        payload_bits=None,
        now_ticks=0,
        attribution=_attribution(REQUEST_RELATION),
    )
    traffic = links.traffic_json_value()

    assert (actual.payload_bits, default.payload_bits, unresolved.payload_bits) == (
        17, 100, None
    )
    assert [row["payload_selection"] for row in traffic["transfers"]] == [
        PayloadSelectionSource.ACTUAL.value,
        PayloadSelectionSource.CONFIGURED_DEFAULT.value,
        PayloadSelectionSource.UNRESOLVED.value,
    ]
    assert all(row["payload_source"] for row in traffic["transfers"])
    assert all(row["reconciles"] for row in traffic["reconciliation"])


@pytest.mark.parametrize(
    ("path", "attribution", "error"),
    [
        (LinkPath.QC, _attribution(), "requires"),
        (LinkPath.CWD, _operation_attribution(), "requires"),
        (LinkPath.CSD, _round_attribution(), "requires"),
        (LinkPath.DO, _operation_attribution(), "requires"),
        (LinkPath.OC, _round_attribution(), "requires"),
        (LinkPath.CQ, _attribution(), "requires"),
        (LinkPath.CSD, _attribution(WRONG_RELATION), "does not match"),
        (LinkPath.WDO, _attribution(REQUEST_RELATION), "tier"),
    ],
)
def test_invalid_path_attribution_fails_before_mutation(path, attribution, error):
    links = LinkModelConfig.reference_fixed_latency_profile().resolve()
    before = links.traffic_json_value()

    with pytest.raises(ValueError, match=error):
        links.reserve(path, payload_bits=None, now_ticks=0,
                      attribution=attribution)

    assert links.traffic_json_value() == before


def test_wiring_threads_one_link_model_to_controller_and_cluster():
    ops = [Operation(0, "M(q0)", (0,), clifford=True)]
    r = simulate(RunSpec(
            ops=ops,
            num_units=1,
            d=3,
            rounds_policy=FixedRounds(11),
            decoder=PerRoundDecoder(tau_us=1.0),
        ), verbose=False)
    assert r.controller.links is r.window_manager.links


def test_finite_aggregate_link_reserves_one_fifo_by_hand():
    config = LinkConfig(
        propagation_latency_ticks=us(1.0),
        capacity=LinkCapacityConfig(
            input_bits_per_us=1000.0,
            basis=LinkQuantityBasis.DIRECT_AGGREGATE,
            channel_count=None,
            source="hand oracle",
        ),
        configuration_source="unit test",
    )
    link = Link(config)

    first = link.reserve(payload_bits=1000, now_ticks=0)
    second = link.reserve(payload_bits=1000, now_ticks=0)

    assert (
        first.queue_wait_ticks,
        first.serialization_ticks,
        first.propagation_ticks,
        first.total_delay_ticks,
        first.physical_sequence,
    ) == (0, us(1.0), us(1.0), us(2.0), 0)
    assert (
        second.queue_wait_ticks,
        second.serialization_ticks,
        second.propagation_ticks,
        second.total_delay_ticks,
        second.physical_sequence,
    ) == (us(1.0), us(1.0), us(1.0), us(3.0), 1)


def _switching_run(*, low_confidence_probability, run_both_at_once=False,
                   double_window=False, rounds=3, links=None,
                   probability_for=None, record_switching_windows=False):
    weak = SampledConfidenceDecoder(
        PerRoundDecoder(0.0),
        low_confidence_probability,
        probability_for=probability_for,
    )
    return RunSpec(
        ops=[Operation(0, "memory", (0,))],
        d=3,
        rounds_policy=FixedRounds(rounds),
        scheme=SlidingWindowScheme(),
        strategy=Switching(
            expected_source=SAMPLED_CONFIDENCE_SOURCE,
            confidence_threshold=0.5,
            run_both_at_once=run_both_at_once,
            double_window=double_window,
        ),
        router=SwitchingRouter(weak, PerRoundDecoder(0.0)),
        unit_pools={"default": 1, "strong": 1},
        links=links,
        record_switching_windows=record_switching_windows,
    ).build()


def _path_counts(completed) -> dict:
    traffic = completed.result.link_traffic
    return {
        edge["path"]: edge["counters"]["transfer_count"]
        for edge in traffic["semantic_edges"]
    }


def test_selected_result_lifecycles_are_mutually_exclusive_end_to_end():
    baseline = RunSpec(
        ops=[Operation(0, "memory", (0,))],
        d=3,
        rounds_policy=FixedRounds(3),
        decoder=PerRoundDecoder(0.0),
    ).build()
    serial = _switching_run(low_confidence_probability=1.0)
    parallel_confident = _switching_run(
        low_confidence_probability=0.0,
        run_both_at_once=True,
    )

    assert {
        path: _path_counts(baseline)[path]
        for path in ("wsd", "csd", "wdo", "do")
    } == {"wsd": 0, "csd": 0, "wdo": 1, "do": 0}
    assert {
        path: _path_counts(serial)[path]
        for path in ("wsd", "csd", "wdo", "do")
    } == {"wsd": 1, "csd": 1, "wdo": 0, "do": 1}
    assert {
        path: _path_counts(parallel_confident)[path]
        for path in ("wsd", "csd", "wdo", "do")
    } == {"wsd": 0, "csd": 1, "wdo": 1, "do": 0}
    assert parallel_confident.decoder_manager.strong_cancelled == 1


def test_capture_rows_cover_serial_parallel_and_aligned_absorption():
    serial = _switching_run(low_confidence_probability=1.0,
                            record_switching_windows=True)
    parallel = _switching_run(low_confidence_probability=0.0,
                              run_both_at_once=True, record_switching_windows=True)
    for completed, selected_tier in ((serial, "strong"), (parallel, "weak")):
        records = completed.result.metric_values()["window_switching_records"]
        assert len(records["windows"]) == 1
        assert records["windows"][0]["selected_request_key"]["tier"] == selected_tier
        key_text = lambda key: json.dumps(key, sort_keys=True)
        request_keys = {key_text(row["request_key"]) for row in records["requests"]}
        service_members = {key_text(key) for service in records["services"]
            for key in service["original_request_keys"]}
        predispatch = {key_text(row["request_key"])
                       for row in records["requests"] if row["terminal_processing_outcome"] ==
                       "strong_cancelled_before_dispatch"}
        transfer_keys = {key_text(row["attribution"]["relation"]["request_key"])
                         for row in completed.result.link_traffic["transfers"]
                         if row["attribution"]["relation"] is not None}
        assert request_keys == service_members | predispatch
        assert transfer_keys <= request_keys
        assert not service_members & predispatch
        assert all(row["service_key"] is None for row in records["requests"]
                   if key_text(row["request_key"]) in predispatch)

    aligned = _switching_run(
        low_confidence_probability=0.0, double_window=True, rounds=30,
        probability_for=lambda job: 1.0 if job.window_id == 2 else 0.0,
        record_switching_windows=True)
    rows = aligned.result.metric_values()["window_switching_records"]["windows"]
    absorbed = [row for row in rows if row["window_disposition"] == "absorbed"]
    assert len(rows) == 10 and len(absorbed) == 2 and all(
        row["absorbed_into"] is not None and row["selected_request_key"] is None
        for row in absorbed)


def test_double_window_reserves_wsd_at_decision_and_csd_when_slab_exists():
    completed = _switching_run(
        low_confidence_probability=0.0,
        double_window=True,
        rounds=14,
        probability_for=lambda job: 1.0 if job.window_id == 2 else 0.0,
    )
    transfers = completed.result.link_traffic["transfers"]
    selected = [
        transfer
        for transfer in transfers
        if transfer["path"] in ("wsd", "csd", "do")
    ]

    assert [transfer["path"] for transfer in selected] == ["wsd", "csd", "do"]
    assert selected[0]["send_ticks"] < selected[1]["send_ticks"]
    assert (selected[0]["attribution"]["relation"]["request_key"]
            == selected[2]["attribution"]["relation"]["request_key"])
    assert (
        selected[0]["attribution"]["round_lo"],
        selected[0]["attribution"]["round_hi"],
    ) == (7, 12)
    assert (
        selected[1]["attribution"]["round_lo"],
        selected[1]["attribution"]["round_hi"],
    ) == (4, 14)


def test_serial_wsd_and_csd_contend_when_bound_to_one_physical_fifo():
    reference = LinkModelConfig.reference_fixed_latency_profile()
    shared = LinkConfig(
        0,
        LinkCapacityConfig(
            1000.0,
            LinkQuantityBasis.DIRECT_AGGREGATE,
            None,
            "hand shared-bus oracle",
        ),
        "hand shared-bus oracle",
    )
    payload = PayloadSizeConfig(
        1000,
        LinkQuantityBasis.DIRECT_AGGREGATE,
        None,
        "hand 1000-bit transaction",
    )
    config = replace(
        reference,
        wsd=LinkEdgeConfig(shared, payload, "decision metadata"),
        csd=LinkEdgeConfig(shared, payload, "retained syndrome data"),
        profile_name="test_shared_strong_input_fifo",
    )
    completed = _switching_run(
        low_confidence_probability=1.0,
        links=config,
    )
    transfers = completed.result.link_traffic["transfers"]
    wsd, csd = [
        transfer
        for transfer in transfers
        if transfer["path"] in ("wsd", "csd")
    ]

    assert wsd["send_ticks"] == csd["send_ticks"]
    assert wsd["queue_wait_ticks"] == 0
    assert csd["queue_wait_ticks"] == us(1.0)
    assert csd["delivery_ticks"] == wsd["delivery_ticks"] + us(1.0)
