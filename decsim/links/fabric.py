"""The link fabric: every hop of the reaction path wired to its channel.

LinkFabric is the root of the links and the one object the other
components hold; it implements the Link port. send(path, ...) selects the
payload the path is priced with, sends it on the path's channel, and at
delivery fires the finished transfer's record on transfer_delivered (the
traffic ledger and the trace writer listen) before the caller's
continuation runs. Two paths whose settings name the same channel share
its wire and its setup engine. The shape is gem5's: a SimObject owns its
parameters and its children and is reached through its ports
(src/sim/sim_object.hh, src/mem/port.hh); the trace source is ns-3's
TracedCallback fired at the device's transition
(point-to-point-net-device.cc TransmitComplete).
"""

import dataclasses
from typing import Optional

import decsim.engine
import decsim.links.channel as channel_module
import decsim.links.settings as link_settings
import decsim.records.transfers as transfer_records
import decsim.trace_source as trace_source


class LinkFabric:
    """One run's fabric: the wired paths on their channels.

    Trace source: transfer_delivered(record), one TransferRecord per
    delivered transfer, carrying the path, the attribution and the
    Transfer with its send, serializer and delivery ticks.
    """

    def __init__(
        self,
        fabric_settings: link_settings.FabricSettings,
        engine: decsim.engine.Engine,
    ):
        self.trace = _TraceSources()
        self._channel_by_name: dict[str, channel_module.Channel] = {}
        self._binding_by_path: dict[
            transfer_records.LinkPath, _PathBinding
        ] = {}
        self._send_count = 0
        for path in transfer_records.LinkPath:
            path_settings = fabric_settings.path_settings(path)
            channel = self._channel_for(path_settings.channel, engine)
            self._binding_by_path[path] = _PathBinding(path_settings, channel)

    def expected_delay_ticks(
        self,
        path: transfer_records.LinkPath,
        payload_bits: Optional[int],
        now_ticks: int,
    ) -> int:
        """What a send now would pay if nothing else reached its channel."""
        binding = self._binding_by_path[path]
        selected_bits, _selection, _source = _select_payload(
            path, binding.settings, payload_bits
        )
        return binding.channel.expected_delay_ticks(
            selected_bits, now_ticks, binding.settings.setup_ticks
        )

    def send(
        self,
        path: transfer_records.LinkPath,
        payload_bits: Optional[int],
        now_ticks: int,
        attribution: transfer_records.TransferAttribution,
        on_delivered: channel_module.OnDelivered,
    ) -> None:
        """Send one transfer on a path; on_delivered(transfer) runs at delivery.

        The payload is selected before the channel moves.
        """
        binding = self._binding_by_path[path]
        selected_bits, selection, payload_source = _select_payload(
            path, binding.settings, payload_bits
        )
        outgoing = _Outgoing(
            request_sequence=self._send_count,
            path=path,
            channel=binding.settings.channel.name,
            attribution=attribution,
            payload_selection=selection,
            payload_source=payload_source,
            on_delivered=on_delivered,
        )
        self._send_count += 1
        binding.channel.send(
            selected_bits,
            now_ticks,
            binding.settings.setup_ticks,
            lambda transfer: self._finish(outgoing, transfer),
        )

    def _channel_for(
        self,
        channel_settings: link_settings.ChannelSettings,
        engine: decsim.engine.Engine,
    ) -> channel_module.Channel:
        channel = self._channel_by_name.get(channel_settings.name)
        if channel is None:
            channel = channel_module.Channel(channel_settings, engine)
            self._channel_by_name[channel_settings.name] = channel
        return channel

    def _finish(
        self, outgoing: "_Outgoing", transfer: transfer_records.Transfer
    ) -> None:
        """At delivery: the listeners hear the transfer, then the caller."""
        if self.trace.transfer_delivered.has_listeners:
            record = transfer_records.TransferRecord(
                request_sequence=outgoing.request_sequence,
                path=outgoing.path,
                channel=outgoing.channel,
                attribution=outgoing.attribution,
                payload_selection=outgoing.payload_selection,
                payload_source=outgoing.payload_source,
                transfer=transfer,
            )
            self.trace.transfer_delivered.fire(record)
        outgoing.on_delivered(transfer)


@dataclasses.dataclass(frozen=True)
class _PathBinding:
    """One wired path: its settings and the channel it rides."""

    settings: link_settings.PathSettings
    channel: channel_module.Channel


@dataclasses.dataclass(frozen=True)
class _Outgoing:
    """What the ledger's record needs beyond the channel's timing."""

    request_sequence: int
    path: transfer_records.LinkPath
    channel: str
    attribution: transfer_records.TransferAttribution
    payload_selection: transfer_records.PayloadSelection
    payload_source: Optional[str]
    on_delivered: channel_module.OnDelivered


def _select_payload(
    path: transfer_records.LinkPath,
    path_settings: link_settings.PathSettings,
    payload_bits: Optional[int],
) -> tuple:
    """The bits a transfer is priced with, and where they came from.

    The actual payload when the caller supplied one (the path must name
    its source), else the card's default, else unresolved. An unresolved
    payload rides an unbounded channel for its latency alone; a bounded
    wire needs a size to serialize, so it refuses the transfer.
    """
    if payload_bits is not None:
        if path_settings.actual_payload_source is None:
            raise RuntimeError(
                f"{path.value} does not declare an actual payload source"
            )
        return (
            payload_bits,
            transfer_records.PayloadSelection.ACTUAL,
            path_settings.actual_payload_source,
        )
    default_payload = path_settings.default_payload
    if default_payload is not None:
        return (
            default_payload.aggregate_bits,
            transfer_records.PayloadSelection.CONFIGURED_DEFAULT,
            default_payload.source,
        )
    if path_settings.channel.capacity is not None:
        raise RuntimeError(
            f"{path.value} has no payload size and its channel is bounded; "
            f"a bounded wire needs a size to serialize"
        )
    return (
        None,
        transfer_records.PayloadSelection.UNRESOLVED,
        path_settings.actual_payload_source,
    )


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the link fabric reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (tmp/resources/gem5/src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    transfer_delivered: trace_source.TraceSource = trace_source.new_source()
