"""The link fabric: every hop of the reaction path wired to its channel.

LinkFabric is the root of the links and the one object the other
components hold; it implements the Link port. send(path, ...) selects the
payload the path is priced with, frames it with the path's header, sends
it on the path's channel, and at
delivery fires the finished transfer's record on transfer_delivered (the
traffic ledger and the trace writer listen) before the caller's
continuation runs. Two paths whose settings name the same channel share
its wire and its setup engine. The shape is gem5's: a SimObject owns its
parameters and its children and is reached through its ports
(src/sim/sim_object.hh, src/mem/port.hh); the trace source is ns-3's
TracedCallback fired at the device's transition
(point-to-point-net-device.cc TransmitComplete). The channels are what
the row's build names (the Channel port in decsim/ports.py), one per
channel name; the shipped rows build the PROTOCOLS row each card's
protocol names. The fabric is the seed composite of its channels, so a
channel that draws (the reliable row's losses) is seeded under its name.
"""

import dataclasses
from collections.abc import Callable, Mapping
from typing import Optional

import decsim.config as config
import decsim.engine
import decsim.links.channel as channel_module
import decsim.links.credit_channel as credit_channel
import decsim.links.reliable_channel as reliable_channel
import decsim.links.settings as link_settings
import decsim.ports as ports
import decsim.records.seeds as seed_records
import decsim.records.transfers as transfer_records
import decsim.tables as tables
import decsim.trace_source as trace_source

# What a Link row's build hands the fabric: called once per channel name
# with that channel's settings and the run's engine.
ChannelClass = Callable[
    [link_settings.ChannelSettings, decsim.engine.Engine], ports.Channel
]

# links.<path>.protocol.kind names one of these rows: how the path's
# channel moves a message. ideal is the whole transfer on an unbounded
# buffer with nothing lost; credit cuts it into frames that wait for
# receive-buffer credits; reliable adds loss and go-back-N recovery.
PROTOCOLS = {
    "ideal": channel_module.Channel,
    "credit": credit_channel.CreditChannel,
    "reliable": reliable_channel.ReliableChannel,
}


def protocol_channel(
    channel_settings: link_settings.ChannelSettings,
    engine: decsim.engine.Engine,
) -> ports.Channel:
    """The channel of the PROTOCOLS row the channel's settings name."""
    row = PROTOCOLS[channel_settings.protocol.kind]
    return row(channel_settings, engine)


def protocol_settings_from_yaml(
    section, clock: config.Clock, path_name: str
) -> link_settings.ProtocolSettings:
    """A card's protocol mapping: a kind, and the keys its row declares."""
    section_name = f"links.{path_name}.protocol"
    if not isinstance(section, Mapping):
        raise ValueError(
            f"{section_name} holds {section!r}; it is a mapping with a kind, "
            f"one of {sorted(PROTOCOLS)}"
        )
    kind = section.get("kind", "ideal")
    row = tables.row(PROTOCOLS, f"links.{path_name}.protocol.kind", kind)
    row_settings = tables.row_settings(
        row, section_name, section, ("kind",), path_name
    )
    return link_settings.ProtocolSettings(
        kind=kind, row_settings=row_settings, clock=clock
    )


class LinkFabric:
    """One run's fabric: the wired paths on their channels.

    Trace sources: transfer_delivered(record), one TransferRecord per
    delivered transfer, carrying the path, the attribution and the
    Transfer with its send, serializer and delivery ticks; and
    frame_landed(record), one FrameRecord per frame a channel moves,
    the transfer's inside.
    """

    def __init__(
        self,
        fabric_settings: link_settings.FabricSettings,
        engine: decsim.engine.Engine,
        channel_class: ChannelClass,
    ) -> None:
        self._channel_by_name: dict[str, ports.Channel] = {}
        self._binding_by_path: dict[
            transfer_records.LinkPath, _PathBinding
        ] = {}
        self._readout_by_footprint: dict[tuple, _PathBinding] = {}
        self._send_count = 0
        for path in transfer_records.LinkPath:
            path_settings = fabric_settings.path_settings(path)
            channel = self._channel_for(
                path_settings.channel, engine, channel_class
            )
            self._binding_by_path[path] = _PathBinding(path_settings, channel)
        for route in fabric_settings.readout_routes:
            channel = self._channel_for(
                route.settings.channel, engine, channel_class
            )
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

        A routed readout is priced by its footprint, which an estimate is
        not given, so asking for one is a caller's bug.
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
        channel_class: ChannelClass,
    ) -> ports.Channel:
        """The named channel, built of the row's class on its first path."""
        channel = self._channel_by_name.get(channel_settings.name)
        if channel is None:
            channel = channel_class(channel_settings, engine)
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
            default_payload.input_bits,
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
    (gem5 src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

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

    @property
    def has_listeners(self) -> bool:
        """Whether any channel's frames are heard."""
        for channel in self._channels:
            if channel.trace.frame_landed.has_listeners:
                return True
        return False
