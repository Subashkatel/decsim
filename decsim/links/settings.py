"""What the yaml and the number cards build for the links: settings only.

A channel is one wire with a propagation latency and, optionally, a
finite bandwidth: ns-3's point-to-point device, a DataRate and a delay
(point-to-point-net-device.cc). A path is one named hop between two
components; it rides one channel, carries a payload rule, and may pay a
per-transfer setup cost, the descriptor and doorbell work a processor
does before the data mover starts (Shao et al., MICRO 2016, section
III.C), and a per-transfer header, the framing the wire serializes with
the payload (ns-3's device adds a header to every packet it sends,
point-to-point-net-device.cc:528, and times the packet with it, :243).
A fabric is one path setting per hop plus a profile name; two paths
whose channels carry the same name share one wire and one setup engine.
The values are checked here, once, because the yaml and the number cards
are where they enter decsim.
"""

import dataclasses
import fractions
import math
from collections.abc import Mapping
from typing import Optional, Protocol, Union

import decsim.config as config
import decsim.engine
import decsim.ports as ports
import decsim.records.identity as identity_records
import decsim.records.transfers as transfer_records


@dataclasses.dataclass(frozen=True)
class CapacitySettings:
    """Bandwidth of one whole channel in bits per microsecond.

    A link of several parallel bit lanes is one wire with aggregate
    bandwidth (a PCIe x4 link stripes one transfer over four lanes), so
    a yaml card folds its `channels` count into this rate before the
    setting is built. The rate is kept as written, a decimal from a
    Python card or an exact Fraction from the yaml and the bandwidth
    card, and the serialization arithmetic reads it exactly
    (exact_aggregate_bits_per_microsecond).
    """

    input_bits_per_microsecond: Union[float, fractions.Fraction]
    # where the rate was written, a label: no part of a point's id
    source: str = dataclasses.field(compare=False)

    def __post_init__(self) -> None:
        _require_finite_number(
            self.input_bits_per_microsecond, "input_bits_per_microsecond"
        )
        if self.input_bits_per_microsecond <= 0:
            raise ValueError("input_bits_per_microsecond must be positive")

    def exact_aggregate_bits_per_microsecond(self) -> fractions.Fraction:
        """The whole channel's rate as an exact Fraction of the card's text.

        ns-3's DataRate does the same arithmetic in integers, so a
        whole-tick duration is never inflated by a binary float's hidden
        expansion.
        """
        rate_text = str(self.input_bits_per_microsecond)
        return fractions.Fraction(rate_text)


@dataclasses.dataclass(frozen=True)
class PayloadSettings:
    """Default payload of one path: the whole transfer's bits."""

    input_bits: int
    source: str

    def __post_init__(self) -> None:
        input_bits = _as_whole_number(self.input_bits, "input_bits")
        object.__setattr__(self, "input_bits", input_bits)
        if self.input_bits < 0:
            raise ValueError("input_bits must be nonnegative")


class PacketProtocolSettings(Protocol):
    """A packet protocol's settings record, which builds its channel.

    CreditChannel.Settings and ReliableChannel.Settings
    (credit_channel.py, reliable_channel.py) are the two.
    """

    def build(
        self,
        channel_settings: "ChannelSettings",
        engine: decsim.engine.Engine,
    ) -> ports.Channel:
        """A fresh channel whose protocol is these settings."""


@dataclasses.dataclass(frozen=True)
class ChannelSettings:
    """One physical channel: its name, a propagation latency, a bandwidth.

    No capacity means an unbounded wire that charges its latency only.
    Two paths whose channels carry the same name share one channel, so
    the name is the identity a fabric wires by. protocol is a packet
    protocol's settings record, which builds the channel
    (credit_channel.py, reliable_channel.py); None is the ideal wire,
    the whole transfer on an unbounded buffer with nothing lost. A
    packet protocol cuts every message into frames of known size, so it
    needs a bounded wire.
    """

    name: str
    propagation_latency_ticks: int
    capacity: Optional[CapacitySettings]
    # where the card was written, a label: no part of a point's id
    configuration_source: str = dataclasses.field(compare=False)
    protocol: Optional["PacketProtocolSettings"] = None

    def __post_init__(self) -> None:
        propagation_latency_ticks = _as_whole_number(
            self.propagation_latency_ticks, "propagation_latency_ticks"
        )
        object.__setattr__(
            self, "propagation_latency_ticks", propagation_latency_ticks
        )
        if self.propagation_latency_ticks < 0:
            raise ValueError("propagation_latency_ticks must be nonnegative")
        if self.protocol is not None and self.capacity is None:
            raise ValueError(
                f"channel {self.name!r} runs a packet protocol, which cuts "
                f"every message into frames of known size; give it a "
                f"bounded wire (bits_per_cycle)"
            )


@dataclasses.dataclass(frozen=True)
class PathSettings:
    """One path: its channel, its payload rule, and its setup cost.

    At least one of the default payload and the actual payload source is
    given. setup_ticks is paid on the channel's setup engine before every
    transfer of the path; zero means the path programs nothing.
    header_bits_per_transfer is the framing every transfer of the path
    carries beside its payload, serialized with it and counted apart
    from it; zero means the path frames nothing. CUDA-Q's real-time
    messages are the worked case: a 24 byte RPCHeader in front of every
    request and a 24 byte RPCResponse in front of every reply
    (cudaqx decoder_rpc_wire_format.h lines 41-43), and 32 bytes of
    fields in front of the syndromes of an enqueue (lines 62-69). Those
    bytes hold CUDA-Q's own ids, an int64 decoder id among the enqueue's
    fields and a 32-bit request id in each header, and the three kinds
    of message of a strong request already carry decsim's 64-bit name
    (records/windows.py REQUEST_KEY_WIRE_BITS), so a run that prices
    this framing leaves out the id the name stands for.
    excludes_receiver_processing says what the card's latency covers: a
    reference number measured end to end includes the receiver turning
    the arrival into bits, and a card the run's own yaml wrote times the
    wire alone, so only the second lets that processing be priced again
    on the receiving component. It has no default, so every card says
    which it is.
    """

    channel: ChannelSettings
    default_payload: Optional[PayloadSettings]
    actual_payload_source: Optional[str]
    setup_ticks: int = 0
    header_bits_per_transfer: int = 0
    # the card times the wire alone, so the receiving component's own
    # processing of what arrives is priced somewhere else
    excludes_receiver_processing: bool = dataclasses.field(kw_only=True)

    def __post_init__(self) -> None:
        has_default = self.default_payload is not None
        has_actual_source = self.actual_payload_source is not None
        if not has_default and not has_actual_source:
            raise ValueError(
                "a path needs a default payload or an actual payload source"
            )
        setup_ticks = _as_count(self.setup_ticks, "setup_ticks")
        object.__setattr__(self, "setup_ticks", setup_ticks)
        header_bits = _as_count(
            self.header_bits_per_transfer, "header_bits_per_transfer"
        )
        object.__setattr__(self, "header_bits_per_transfer", header_bits)


@dataclasses.dataclass(frozen=True)
class ReadoutRoute:
    """The readout path setting for one complete physical patch footprint.

    A joint acquisition stays one transfer. A route never splits its bits
    or selects a channel from only one of its contributing patches.
    """

    patch_ids: tuple
    settings: PathSettings

    def __post_init__(self) -> None:
        patches = tuple(self.patch_ids)
        if not identity_records.is_stable_identity(patches):
            raise ValueError("readout route patches must be stable identities")
        if not patches or len(set(patches)) != len(patches):
            raise ValueError("a readout route needs nonempty unique patches")
        ordered = sorted(patches, key=identity_records.stable_identity_bytes)
        object.__setattr__(self, "patch_ids", tuple(ordered))


@dataclasses.dataclass(frozen=True)
class FabricSettings:
    """A fabric card: one path setting per hop and a profile name.

    Every hop of the reaction path is priced, so a card names all eleven
    and a caller that leaves one out is refused where it constructs the
    card. A card whose QPU-to-controller latency leaves out the
    controller's readout processing says so, because the timing card
    prices that processing on its own line. profile_name is the preset's
    name, or the yaml's own file name for a card read from a yaml, which
    the run's description prints. Python
    callers can set readout_routes to choose a path card by the complete
    contributing patch footprint.
    Unmatched footprints use qpu_to_controller. Equal channel names share
    the same setup engine and serializer, including across routed cards.
    """

    qpu_to_controller: PathSettings
    controller_to_weak_buffer: PathSettings
    weak_buffer_to_weak_decoder: PathSettings
    weak_decoder_to_strong_decoder: PathSettings
    strong_buffer_to_strong_decoder: PathSettings
    weak_decoder_to_frame: PathSettings
    decoder_to_decoder: PathSettings
    strong_decoder_to_frame: PathSettings
    frame_to_controller: PathSettings
    controller_to_qpu: PathSettings
    controller_to_strong_buffer: PathSettings
    # the card's name for the run's description, a label: no part of a
    # point's id
    profile_name: str = dataclasses.field(compare=False)
    readout_routes: tuple[ReadoutRoute, ...] = ()

    def __post_init__(self) -> None:
        routes = tuple(self.readout_routes)
        object.__setattr__(self, "readout_routes", routes)
        footprints = [route.patch_ids for route in routes]
        if len(set(footprints)) != len(footprints):
            raise ValueError("readout routes must have distinct footprints")
        channel_by_name = {}
        bindings = self.path_bindings()
        for _path, path_settings in bindings:
            channel = path_settings.channel
            known = channel_by_name.setdefault(channel.name, channel)
            if known != channel:
                raise ValueError(
                    f"channel {channel.name!r} is declared with different "
                    f"settings by two paths"
                )

    def path_settings(self, path: transfer_records.LinkPath) -> PathSettings:
        """The setting of one path."""
        return getattr(self, path.value)

    def path_bindings(self) -> tuple:
        """Every semantic path's channel bindings, including readout routes."""
        bindings = []
        for path in transfer_records.LinkPath:
            settings = self.path_settings(path)
            bindings.append((path, settings))
        for route in self.readout_routes:
            bindings.append(
                (transfer_records.LinkPath.QPU_TO_CONTROLLER, route.settings)
            )
        return tuple(bindings)


def required_key(section: Mapping, key: str, section_name: str) -> object:
    """A key a link card needs, refused by name when missing."""
    if key not in section:
        raise ValueError(f"{section_name} needs {key}")
    return section[key]


def check_positive_count(name: str, value) -> None:
    """A positive whole number, never a boolean, or a refusal naming it."""
    if config.is_whole_count(value):
        return
    raise ValueError(f"{name} is {value!r}; it is a positive whole number")


def _as_whole_number(value, name: str) -> int:
    """A value as an exact int, or a ValueError naming the field.

    3.0 is fine; 3.5, NaN and None are not, nor a boolean, which Python
    would read as 0 or 1 and a yaml writes as a flag.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite whole number")
    try:
        whole = int(value)
    except (OverflowError, ValueError, TypeError):
        raise ValueError(f"{name} must be a finite whole number") from None
    if whole != value:
        raise ValueError(f"{name} must be a finite whole number")
    return whole


def _as_count(value, name: str) -> int:
    """A whole number that is not negative, or a ValueError naming the field."""
    whole = _as_whole_number(value, name)
    if whole < 0:
        raise ValueError(f"{name} must be nonnegative")
    return whole


def _require_finite_number(value, name: str) -> None:
    """Refuse NaN, infinity and a boolean.

    A Python int of any size is finite: math.isfinite overflows converting
    it to float, and that overflow reads as finite.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        is_finite = math.isfinite(value)
    except OverflowError:
        is_finite = True
    if not is_finite:
        raise ValueError(f"{name} must be a finite number")
