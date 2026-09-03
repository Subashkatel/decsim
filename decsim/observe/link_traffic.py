"""The traffic ledger: what the links carried, and the JSON the run exports.

The ledger listens to the fabric (LinkFabric hands it every finished
transfer once) and keeps counters per path and per channel and the list
of every transfer in request order; a channel's counters are the sum of
its paths' counters by construction. traffic_json_value is
result.link_traffic, and every key and value in it is pinned by the gate.
The fabric runs without a ledger; nothing in the machine reads one.
"""

import dataclasses
from typing import Optional

import decsim.links.settings as link_settings
import decsim.message as message


@dataclasses.dataclass(frozen=True)
class TrafficCounters:
    """Additive counters kept per path and per channel."""

    transfer_count: int = 0
    known_payload_bits: int = 0
    unknown_payload_transfer_count: int = 0
    serialization_ticks: int = 0
    propagation_ticks: int = 0
    queue_wait_ticks: int = 0

    def plus_transfer(self, transfer: message.Transfer) -> "TrafficCounters":
        """These counters with one more transfer added."""
        known_payload_bits = self.known_payload_bits
        unknown_payload_transfer_count = self.unknown_payload_transfer_count
        if transfer.payload_bits is None:
            unknown_payload_transfer_count += 1
        else:
            known_payload_bits += transfer.payload_bits
        transfer_count = self.transfer_count + 1
        serialization_ticks = (
            self.serialization_ticks + transfer.serialization_ticks
        )
        propagation_ticks = self.propagation_ticks + transfer.propagation_ticks
        queue_wait_ticks = self.queue_wait_ticks + transfer.queue_wait_ticks
        return TrafficCounters(
            transfer_count=transfer_count,
            known_payload_bits=known_payload_bits,
            unknown_payload_transfer_count=unknown_payload_transfer_count,
            serialization_ticks=serialization_ticks,
            propagation_ticks=propagation_ticks,
            queue_wait_ticks=queue_wait_ticks,
        )

    def plus(self, other: "TrafficCounters") -> "TrafficCounters":
        """The sum of these counters and another's."""
        transfer_count = self.transfer_count + other.transfer_count
        known_payload_bits = self.known_payload_bits + other.known_payload_bits
        unknown_payload_transfer_count = (
            self.unknown_payload_transfer_count
            + other.unknown_payload_transfer_count
        )
        serialization_ticks = (
            self.serialization_ticks + other.serialization_ticks
        )
        propagation_ticks = self.propagation_ticks + other.propagation_ticks
        queue_wait_ticks = self.queue_wait_ticks + other.queue_wait_ticks
        return TrafficCounters(
            transfer_count=transfer_count,
            known_payload_bits=known_payload_bits,
            unknown_payload_transfer_count=unknown_payload_transfer_count,
            serialization_ticks=serialization_ticks,
            propagation_ticks=propagation_ticks,
            queue_wait_ticks=queue_wait_ticks,
        )

    def to_json_value(self) -> dict:
        """The counters as the traffic report writes them."""
        unknown_payload_transfer_count = self.unknown_payload_transfer_count
        return {
            "transfer_count": self.transfer_count,
            "known_payload_bits": self.known_payload_bits,
            "unknown_payload_transfer_count": unknown_payload_transfer_count,
            "serialization_ticks": self.serialization_ticks,
            "propagation_ticks": self.propagation_ticks,
            "queue_wait_ticks": self.queue_wait_ticks,
        }


@dataclasses.dataclass(frozen=True)
class PathSnapshot:
    """One wired path, its channel's alias, its counters."""

    path: message.LinkPath
    physical_alias: str
    counters: TrafficCounters


@dataclasses.dataclass(frozen=True)
class ChannelSnapshot:
    """One channel, the paths that ride it, its settings, its counters."""

    alias: str
    member_paths: tuple
    settings: link_settings.ChannelSettings
    counters: TrafficCounters


@dataclasses.dataclass(frozen=True)
class FabricSnapshot:
    """What a run's fabric looked like and carried, frozen for reports."""

    paths: tuple
    channels: tuple
    transfers: tuple


class TrafficLedger:
    """The fabric's observer: every transfer counted per path and per channel.

    A channel's alias is "channel-<n>", numbered in the order the wired
    paths first meet it; the reports name channels by alias.
    """

    def __init__(self, fabric_settings: link_settings.FabricSettings):
        self._settings = fabric_settings
        self._alias_by_channel: dict[str, str] = {}
        self._counters_by_path: dict[message.LinkPath, TrafficCounters] = {}
        self._counters_by_channel: dict[str, TrafficCounters] = {}
        self._records: list[message.TransferRecord] = []
        for path in fabric_settings.wired_paths():
            path_settings = fabric_settings.path_settings(path)
            channel_name = path_settings.channel.name
            self._counters_by_path[path] = TrafficCounters()
            if channel_name in self._alias_by_channel:
                continue
            alias = f"channel-{len(self._alias_by_channel)}"
            self._alias_by_channel[channel_name] = alias
            self._counters_by_channel[channel_name] = TrafficCounters()

    def on_transfer(self, record: message.TransferRecord) -> None:
        """Count one delivered transfer for its path and its channel."""
        transfer = record.transfer
        path_counters = self._counters_by_path[record.path]
        self._counters_by_path[record.path] = path_counters.plus_transfer(
            transfer
        )
        channel_counters = self._counters_by_channel[record.channel]
        self._counters_by_channel[record.channel] = (
            channel_counters.plus_transfer(transfer)
        )
        self._records.append(record)

    def snapshot(self) -> FabricSnapshot:
        """A frozen view: the wiring, every counter, every transfer."""
        paths = []
        for path in self._counters_by_path:
            path_snapshot = self._path_snapshot(path)
            paths.append(path_snapshot)
        channels = []
        for channel_name in self._alias_by_channel:
            channel_snapshot = self._channel_snapshot(channel_name)
            channels.append(channel_snapshot)
        transfers = sorted(self._records, key=_request_sequence)
        return FabricSnapshot(
            paths=tuple(paths),
            channels=tuple(channels),
            transfers=tuple(transfers),
        )

    def traffic_json_value(self) -> dict:
        """What the links carried, as result.link_traffic.

        Counters per path with the setup ticks itemized (setup is
        engine-side work before the wire and never enters a channel's
        counters), counters per channel, every transfer, and the
        reconciliation of each channel against its member paths.
        """
        snapshot = self.snapshot()
        setup_ticks_by_path = {path.path: 0 for path in snapshot.paths}
        for record in snapshot.transfers:
            setup_ticks_by_path[record.path] += record.transfer.setup_ticks
        semantic_edges = []
        for path_snapshot in snapshot.paths:
            semantic_edge = _semantic_edge_json(
                path_snapshot, setup_ticks_by_path
            )
            semantic_edges.append(semantic_edge)
        physical_channels = []
        reconciliation = []
        for channel_snapshot in snapshot.channels:
            channel_json, reconciliation_json = self._channel_json(
                channel_snapshot
            )
            physical_channels.append(channel_json)
            reconciliation.append(reconciliation_json)
        transfers = []
        for record in snapshot.transfers:
            transfer_json = _transfer_json(record, self._alias_by_channel)
            transfers.append(transfer_json)
        path_order = [path.path.value for path in snapshot.paths]
        return {
            "schema_version": 1,
            "path_order": path_order,
            "semantic_edges": semantic_edges,
            "physical_channels": physical_channels,
            "transfers": transfers,
            "reconciliation": reconciliation,
        }

    def _path_snapshot(self, path: message.LinkPath) -> PathSnapshot:
        path_settings = self._settings.path_settings(path)
        alias = self._alias_by_channel[path_settings.channel.name]
        return PathSnapshot(
            path=path,
            physical_alias=alias,
            counters=self._counters_by_path[path],
        )

    def _channel_snapshot(self, channel_name: str) -> ChannelSnapshot:
        member_paths = []
        channel_settings = None
        for path in self._counters_by_path:
            path_settings = self._settings.path_settings(path)
            if path_settings.channel.name == channel_name:
                member_paths.append(path)
                channel_settings = path_settings.channel
        return ChannelSnapshot(
            alias=self._alias_by_channel[channel_name],
            member_paths=tuple(member_paths),
            settings=channel_settings,
            counters=self._counters_by_channel[channel_name],
        )

    def _channel_json(self, channel_snapshot: ChannelSnapshot) -> tuple:
        semantic_sum = TrafficCounters()
        for path in channel_snapshot.member_paths:
            semantic_sum = semantic_sum.plus(self._counters_by_path[path])
        assert semantic_sum == channel_snapshot.counters, (
            "a channel's counters are the sum of its paths' counters"
        )
        member_paths = [path.value for path in channel_snapshot.member_paths]
        channel_json = {
            "physical_alias": channel_snapshot.alias,
            "member_paths": member_paths,
            "counters": channel_snapshot.counters.to_json_value(),
        }
        reconciliation_json = {
            "physical_alias": channel_snapshot.alias,
            "member_paths": member_paths,
            "semantic_counter_sum": semantic_sum.to_json_value(),
            "physical_counters": channel_snapshot.counters.to_json_value(),
            "reconciles": True,
        }
        return channel_json, reconciliation_json


def _request_sequence(record: message.TransferRecord) -> int:
    return record.request_sequence


def _semantic_edge_json(
    path_snapshot: PathSnapshot, setup_ticks_by_path: dict
) -> dict:
    return {
        "path": path_snapshot.path.value,
        "physical_alias": path_snapshot.physical_alias,
        "counters": path_snapshot.counters.to_json_value(),
        "setup_ticks": setup_ticks_by_path[path_snapshot.path],
    }


def _transfer_json(record: message.TransferRecord, alias_by_channel) -> dict:
    transfer = record.transfer
    attribution = record.attribution
    patch_ids = []
    for patch_id in attribution.patch_ids:
        patch_json = message.stable_identity_json(patch_id)
        patch_ids.append(patch_json)
    return {
        "path": record.path.value,
        "physical_alias": alias_by_channel[record.channel],
        "attribution": {
            "operation_id": message.stable_identity_json(
                attribution.operation_id
            ),
            "patch_ids": patch_ids,
            "window_id": attribution.window_id,
            "round_lo": attribution.first_round,
            "round_hi": attribution.last_round,
            "relation": _relation_json(attribution.relation),
        },
        "payload_bits": transfer.payload_bits,
        "payload_selection": record.payload_selection.value,
        "payload_source": record.payload_source,
        "setup_ticks": transfer.setup_ticks,
        "send_ticks": transfer.send_ticks,
        "serializer_start_ticks": transfer.serializer_start_ticks,
        "serializer_end_ticks": transfer.serializer_end_ticks,
        "delivery_ticks": transfer.delivery_ticks,
        "queue_wait_ticks": transfer.queue_wait_ticks,
        "serialization_ticks": transfer.serialization_ticks,
        "propagation_ticks": transfer.propagation_ticks,
        "total_delay_ticks": transfer.total_delay_ticks,
        "physical_sequence": transfer.physical_sequence,
    }


def _relation_json(relation) -> Optional[dict]:
    if relation is None:
        return None
    is_boundary = type(relation) is message.BoundaryTransferRelation
    if is_boundary:
        key = relation.source_request_key
    else:
        key = relation.request_key
    value = {
        "request_key": {
            "operation_id": message.stable_identity_json(key.operation_id),
            "window_id": key.window_id,
            "tier": key.tier.value,
            "run_sequence": key.run_sequence,
        }
    }
    if is_boundary:
        value["source_window_key"] = message.stable_identity_json(
            relation.source_window_key
        )
        value["destination_window_key"] = message.stable_identity_json(
            relation.destination_window_key
        )
        value["source_revision"] = relation.source_revision
        value["delivery_revision"] = relation.delivery_revision
    return value
