"""JSON reports of a run's link fabric, from its frozen snapshot: the
topology (which semantic paths share which physical channels, with their
cards) and the traffic (counters per path and per channel, and every
transfer with its attribution and timing)."""

from __future__ import annotations

from ..message import stable_identity_json
from .links import (BoundaryTransferRelation, LinkFabricSnapshot, RequestTransferRelation,
                    SemanticTransferRecord, TrafficCounters)


def topology_json_value(fabric: LinkFabricSnapshot, *,
                        controller_link_integration_assurance: str) -> dict:
    """The wiring of a fabric: semantic edges, physical channels, and how the
    controller side of the links was verified."""
    if controller_link_integration_assurance not in (
        "shipped_controller",
        "custom_controller_unverified",
    ):
        raise ValueError("unknown controller link integration assurance")
    edges = []
    for edge in fabric.edges:
        edges.append({
            "path": edge.path.value,
            "physical_alias": edge.physical_alias,
            "actual_payload_source": edge.edge.actual_payload_source,
            "default_payload": (
                None
                if edge.edge.default_payload is None
                else edge.edge.default_payload.to_json_value()
            ),
        })
    physical_channels = []
    for channel in fabric.channels:
        capacity = channel.config.capacity
        physical_channels.append({
            "physical_alias": channel.alias,
            "member_paths": [path.value for path in channel.member_paths],
            "propagation_latency_ticks": channel.config.propagation_latency_ticks,
            "capacity": (
                None if capacity is None else capacity.to_json_value()
            ),
            "configuration_source": channel.config.configuration_source,
            "service_scope": "aggregate_fifo",
        })
    return {
        "schema_version": 1,
        "profile_name": fabric.profile_name,
        "path_order": [path.value for path in fabric.paths],
        "edges": edges,
        "physical_channels": physical_channels,
        "cancellation_semantics": "non_preemptive_irrevocable",
        "controller_link_integration_assurance": (
            controller_link_integration_assurance
        ),
    }


def traffic_json_value(fabric: LinkFabricSnapshot) -> dict:
    """What a fabric carried: per-path counters, per-channel counters (which
    reconcile with the sum of their paths), and every transfer."""
    semantic_edges = []
    counters_by_path = {}
    for edge in fabric.edges:
        counters_by_path[edge.path] = edge.counters
        semantic_edges.append({
            "path": edge.path.value,
            "physical_alias": edge.physical_alias,
            "counters": edge.counters.to_json_value(),
        })
    physical_channels = []
    reconciliation = []
    for channel in fabric.channels:
        semantic_sum = TrafficCounters()
        for path in channel.member_paths:
            semantic_sum = semantic_sum.plus(counters_by_path[path])
        if semantic_sum != channel.counters:
            raise RuntimeError(f"traffic counters do not reconcile for {channel.alias}")
        paths_json = [path.value for path in channel.member_paths]
        physical_channels.append({
            "physical_alias": channel.alias,
            "member_paths": paths_json,
            "counters": channel.counters.to_json_value(),
        })
        reconciliation.append({
            "physical_alias": channel.alias,
            "member_paths": paths_json,
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
    relation = attribution.relation
    relation_json = None
    if relation is not None:
        key = (relation.request_key if type(relation) is RequestTransferRelation
               else relation.source_request_key)
        relation_json = {
            "request_key": {
                "operation_id": stable_identity_json(key.operation_id),
                "window_id": key.window_id, "tier": key.tier.value,
                "run_sequence": key.run_sequence},
        }
        if type(relation) is BoundaryTransferRelation:
            relation_json.update({
                "source_window_key": stable_identity_json(
                    relation.source_window_key
                ),
                "destination_window_key": stable_identity_json(
                    relation.destination_window_key
                ),
                "source_revision": relation.source_revision,
                "delivery_revision": relation.delivery_revision})
    return {
        "path": record.path.value,
        "physical_alias": record.physical_alias,
        "attribution": {
            "operation_id": stable_identity_json(attribution.operation_id),
            "patch_ids": [
                stable_identity_json(patch_id)
                for patch_id in attribution.patch_ids
            ],
            "window_id": attribution.window_id,
            "round_lo": attribution.round_lo,
            "round_hi": attribution.round_hi,
            "relation": relation_json,
        },
        "payload_bits": reservation.payload_bits,
        "payload_selection": record.payload_selection.value,
        "payload_source": record.payload_source,
        "send_ticks": reservation.send_ticks,
        "serializer_start_ticks": reservation.serializer_start_ticks,
        "serializer_end_ticks": reservation.serializer_end_ticks,
        "delivery_ticks": (
            reservation.send_ticks + reservation.total_delay_ticks
        ),
        "queue_wait_ticks": reservation.queue_wait_ticks,
        "serialization_ticks": reservation.serialization_ticks,
        "propagation_ticks": reservation.propagation_ticks,
        "total_delay_ticks": reservation.total_delay_ticks,
        "physical_sequence": reservation.physical_sequence,
    }
