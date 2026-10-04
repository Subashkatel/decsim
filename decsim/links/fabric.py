"""The link fabric: every hop of the reaction path wired to its channel.

LinkFabric implements the Link port and is the one link object other
components hold. send selects the payload the path is priced with,
frames it with the path's header, sends it on the path's channel, and at
delivery fires transfer_delivered before the caller's continuation
runs. The shape is gem5's: a SimObject owns its parameters and children
and is reached through ports (src/sim/sim_object.hh, src/mem/port.hh);
the trace source is ns-3's TracedCallback at TransmitComplete. One
channel per name; the fabric is their seed composite, so a channel that
draws (the reliable row's losses) is seeded under its name.
"""

import dataclasses
from collections.abc import Callable
from typing import Optional

import decsim.engine
import decsim.links.channel as channel_module
import decsim.links.settings as link_settings
import decsim.ports as ports
import decsim.records.seeds as seed_records
import decsim.records.transfers as transfer_records
import decsim.trace_source as trace_source


def protocol_channel(
    channel_settings: link_settings.ChannelSettings,
    engine: decsim.engine.Engine,
) -> ports.Channel:
    """The channel the settings' protocol builds; the ideal wire for none."""
    protocol = channel_settings.protocol
    if protocol is None:
        return channel_module.Channel(channel_settings, engine)
    return protocol.build(channel_settings, engine)


class LinkFabric:
    """One run's fabric: the wired paths on their channels."""

    def __init__(
        self,
        fabric_settings: link_settings.FabricSettings,
        engine: decsim.engine.Engine,
    ) -> None:
        self._channel_by_name: dict[str, ports.Channel] = {}
        self._binding_by_path: dict[
            transfer_records.LinkPath, _PathBinding
        ] = {}
        self._readout_by_footprint: dict[tuple, _PathBinding] = {}
        self._send_count = 0
        for path in transfer_records.LinkPath:
            path_settings = fabric_settings.path_settings(path)
            channel = self._channel_for(path_settings.channel, engine)
            self._binding_by_path[path] = _PathBinding(path_settings, channel)
        for route in fabric_settings.readout_routes:
            channel = self._channel_for(route.settings.channel, engine)
            binding = _PathBinding(route.settings, channel)
            self._readout_by_footprint[route.patch_ids] = binding
        built_channels = self._channel_by_name.values()
        channels = tuple(built_channels)
        frame_landed = _EveryChannelsFrames(channels)
        self.trace = _TraceSources(frame_landed=frame_landed)

    def expected_delay_ticks(
        self,
        path: transfer_records.LinkPath,
        payload_bits: Optional[int],
        now_ticks: int,
    ) -> int:
        """What a send now would pay if nothing else reached its channel.

        A routed readout is priced by its footprint, which an estimate lacks, so
        asking for one is a caller's bug.
        """
        is_readout = path is transfer_records.LinkPath.QPU_TO_CONTROLLER
        if is_readout and self._readout_by_footprint:
            raise RuntimeError("a routed readout delay requires a footprint")
        binding = self._binding_by_path[path]
        selected_bits, _selection, _source = _select_payload(
            path, binding.settings, payload_bits
        )
        framed = channel_module.FramedPayload(
            selected_bits, binding.settings.header_bits_per_transfer
        )
        return binding.channel.expected_delay_ticks(
            framed, now_ticks, binding.settings.setup_ticks
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
        binding = self._binding_for(path, attribution)
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
        framed = channel_module.FramedPayload(
            selected_bits, binding.settings.header_bits_per_transfer
        )
        binding.channel.send(
            framed,
            now_ticks,
            binding.settings.setup_ticks,
            lambda transfer: self._finish(outgoing, transfer),
        )

    def run_seed_children(self) -> tuple:
        """Every channel, under its name; only a channel that draws binds."""
        children = []
        for name, channel in self._channel_by_name.items():
            segment = seed_records.RunSeedPathSegment("string_key", name)
            child = seed_records.RunSeedChild((segment,), channel)
            children.append(child)
        return tuple(children)

    def _binding_for(
        self,
        path: transfer_records.LinkPath,
        attribution: transfer_records.TransferAttribution,
    ) -> "_PathBinding":
        """Route the complete footprint, as gem5's xbar routes an address.

        src/mem/xbar.cc findPort keeps the default when no explicit route
        matches. Channel names, rather than footprint names, own contention.
        """
        default = self._binding_by_path[path]
        if path is not transfer_records.LinkPath.QPU_TO_CONTROLLER:
            return default
        return self._readout_by_footprint.get(attribution.patch_ids, default)

    def _channel_for(
        self,
        channel_settings: link_settings.ChannelSettings,
        engine: decsim.engine.Engine,
    ) -> ports.Channel:
        """The named channel, built by its protocol on its first path."""
        channel = self._channel_by_name.get(channel_settings.name)
        if channel is None:
            channel = protocol_channel(channel_settings, engine)
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
    channel: ports.Channel


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

    The caller's actual payload (the path must name its source), else the
    card's default, else unresolved: that rides an unbounded channel for its
    latency alone, and stops on a bounded wire.
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
            default_payload.input_bits,
            transfer_records.PayloadSelection.CONFIGURED_DEFAULT,
            default_payload.source,
        )
    return (
        None,
        transfer_records.PayloadSelection.UNRESOLVED,
        path_settings.actual_payload_source,
    )


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the link fabric reports, as one member."""

    frame_landed: "_EveryChannelsFrames"
    transfer_delivered: trace_source.TraceSource = trace_source.new_source()


class _EveryChannelsFrames:
    """The frame_landed sources of every channel, heard as one.

    A listener is connected to each channel's own source, so a channel
    with no listener builds no frame record.
    """

    def __init__(self, channels: tuple) -> None:
        self._channels = channels

    def connect(self, listener: Callable) -> None:
        """Hear every frame of every channel from now on."""
        for channel in self._channels:
            channel.trace.frame_landed.connect(listener)
