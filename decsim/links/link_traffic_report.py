"""JSON reports of a run's links, from LinkModel.snapshot(): the topology
(which path rides which channel, with the cards) and the traffic (counters
per path and per channel, which must reconcile, and every transfer with its
attribution and timing). This is ``result.link_traffic``."""

from __future__ import annotations

from ..message import stable_identity_json
from .links import (BoundaryTransferRelation, LinkFabricSnapshot, SemanticTransferRecord,
                    TrafficCounters)


def topology_json_value(fabric: LinkFabricSnapshot) -> dict:
    """The wiring: one row per path with its channel alias and payload policy,
    one row per channel with its member paths and its card."""
    edges = [{
        "path": edge.path.value,
        "physical_alias": edge.physical_alias,
        "actual_payload_source": edge.edge.actual_payload_source,
        "default_payload": (None if edge.edge.default_payload is None
                            else edge.edge.default_payload.to_json_value()),
    } for edge in fabric.edges]
    channels = [{
        "physical_alias": channel.alias,
        "member_paths": [path.value for path in channel.member_paths],
        "propagation_latency_ticks": channel.config.propagation_latency_ticks,
        "capacity": (None if channel.config.capacity is None
                     else channel.config.capacity.to_json_value()),
        "configuration_source": channel.config.configuration_source,
        "service_scope": "aggregate_fifo",
    } for channel in fabric.channels]
    return {
        "schema_version": 1,
        "profile_name": fabric.profile_name,
        "path_order": [path.value for path in fabric.paths],
        "edges": edges,
        "physical_channels": channels,
        "cancellation_semantics": "non_preemptive_irrevocable",
    }


def traffic_json_value(fabric: LinkFabricSnapshot) -> dict:
    """What the links carried: counters per path, counters per channel (the sum
    of its paths' counters must equal the channel's own), and every transfer."""
    counters_by_path = {edge.path: edge.counters for edge in fabric.edges}
    semantic_edges = [{
        "path": edge.path.value,
        "physical_alias": edge.physical_alias,
        "counters": edge.counters.to_json_value(),
    } for edge in fabric.edges]
    physical_channels = []
    reconciliation = []
    for channel in fabric.channels:
        semantic_sum = TrafficCounters()
        for path in channel.member_paths:
            semantic_sum = semantic_sum.plus(counters_by_path[path])
        if semantic_sum != channel.counters:
            raise RuntimeError(f"traffic counters do not reconcile for {channel.alias}")
        member_paths = [path.value for path in channel.member_paths]
        physical_channels.append({
            "physical_alias": channel.alias,
            "member_paths": member_paths,
            "counters": channel.counters.to_json_value(),
        })
        reconciliation.append({
            "physical_alias": channel.alias,
            "member_paths": member_paths,
            "semantic_counter_sum": semantic_sum.to_json_value(),
            "physical_counters": channel.counters.to_json_value(),
            "reconciles": True,
        })
    return {
        "schema_version": 1,
        "path_order": [path.value for path in fabric.paths],
        "semantic_edges": semantic_edges,
        "physical_channels": physical_channels,
        "transfers": [_transfer_json(record) for record in fabric.transfers],
        "reconciliation": reconciliation,
    }


def _transfer_json(record: SemanticTransferRecord) -> dict:
    reservation = record.reservation
    attribution = record.attribution
    return {
        "path": record.path.value,
        "physical_alias": record.physical_alias,
        "attribution": {
            "operation_id": stable_identity_json(attribution.operation_id),
            "patch_ids": [stable_identity_json(patch_id) for patch_id in attribution.patch_ids],
            "window_id": attribution.window_id,
            "round_lo": attribution.round_lo,
            "round_hi": attribution.round_hi,
            "relation": _relation_json(attribution.relation),
        },
        "payload_bits": reservation.payload_bits,
        "payload_selection": record.payload_selection.value,
        "payload_source": record.payload_source,
        "send_ticks": reservation.send_ticks,
        "serializer_start_ticks": reservation.serializer_start_ticks,
        "serializer_end_ticks": reservation.serializer_end_ticks,
        "delivery_ticks": reservation.send_ticks + reservation.total_delay_ticks,
        "queue_wait_ticks": reservation.queue_wait_ticks,
        "serialization_ticks": reservation.serialization_ticks,
        "propagation_ticks": reservation.propagation_ticks,
        "total_delay_ticks": reservation.total_delay_ticks,
        "physical_sequence": reservation.physical_sequence,
    }


def _relation_json(relation):
    if relation is None:
        return None
    is_boundary = type(relation) is BoundaryTransferRelation
    key = relation.source_request_key if is_boundary else relation.request_key
    value = {"request_key": {
        "operation_id": stable_identity_json(key.operation_id),
        "window_id": key.window_id,
        "tier": key.tier.value,
        "run_sequence": key.run_sequence,
    }}
    if is_boundary:
        value.update({
            "source_window_key": stable_identity_json(relation.source_window_key),
            "destination_window_key": stable_identity_json(relation.destination_window_key),
            "source_revision": relation.source_revision,
            "delivery_revision": relation.delivery_revision,
        })
    return value
