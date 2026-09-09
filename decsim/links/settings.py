"""What the yaml and the number cards build for the links: settings only.

A channel is one wire with a propagation latency and, optionally, a
finite bandwidth: ns-3's point-to-point device, a DataRate and a delay
(point-to-point-net-device.cc). A path is one named hop between two
components; it rides one channel, carries a payload rule, and may pay a
per-transfer setup cost, the descriptor and doorbell work a processor
does before the data mover starts (Shao et al., MICRO 2016, section
III.C). A fabric is one path setting per hop plus a profile name; two
paths whose channels carry the same name share one wire and one setup
engine. The values are checked here, once, because the yaml and the
number cards are where they enter decsim.
"""

import dataclasses
import enum
import fractions
import math
from typing import Optional, Union

import decsim.records.transfers as transfer_records


class QuantityBasis(str, enum.Enum):
    """Whether a rate or a payload is stated for the whole link or per lane.

    A link made of several parallel bit lanes is one wire with aggregate
    bandwidth (a PCIe x4 link stripes one transfer over four lanes). The
    values are the words the traffic report writes.
    """

    AGGREGATE = "direct_aggregate"
    PER_LANE = "per_channel"


@dataclasses.dataclass(frozen=True)
class CapacitySettings:
    """Bandwidth of one channel in bits per microsecond, aggregate or per lane.

    The rate is kept as written. The serialization arithmetic reads it as
    the decimal on the card and multiplies by the lane count exactly
    (exact_aggregate_bits_per_microsecond); the reported aggregate is the
    input times the lane count in the input's own arithmetic.
    """

    input_bits_per_microsecond: float
    basis: QuantityBasis
    lane_count: Optional[int]
    source: str

    def __post_init__(self) -> None:
        _require_finite_number(
            self.input_bits_per_microsecond, "input_bits_per_microsecond"
        )
        if self.input_bits_per_microsecond <= 0:
            raise ValueError("input_bits_per_microsecond must be positive")
        lane_count = _lane_count_for(self.basis, self.lane_count, "capacity")
        object.__setattr__(self, "lane_count", lane_count)

    @property
    def aggregate_bits_per_microsecond(
        self,
    ) -> Union[int, float, fractions.Fraction]:
        """The whole channel's rate: the input times the lane count."""
        if self.basis is QuantityBasis.AGGREGATE:
            return self.input_bits_per_microsecond
        return self.input_bits_per_microsecond * self.lane_count

    def exact_aggregate_bits_per_microsecond(self) -> fractions.Fraction:
        """The whole channel's rate as an exact Fraction of the card's text.

        ns-3's DataRate does the same arithmetic in integers, so a
        whole-tick duration is never inflated by a binary float's hidden
        expansion.
        """
        rate_text = str(self.input_bits_per_microsecond)
        rate = fractions.Fraction(rate_text)
        if self.basis is QuantityBasis.AGGREGATE:
            return rate
        return rate * self.lane_count


@dataclasses.dataclass(frozen=True)
class PayloadSettings:
    """Default payload of one path in bits, aggregate or per lane."""

    input_bits: int
    basis: QuantityBasis
    lane_count: Optional[int]
    source: str

    def __post_init__(self) -> None:
        input_bits = _as_whole_number(self.input_bits, "input_bits")
        object.__setattr__(self, "input_bits", input_bits)
        if self.input_bits < 0:
            raise ValueError("input_bits must be nonnegative")
        lane_count = _lane_count_for(self.basis, self.lane_count, "payload")
        object.__setattr__(self, "lane_count", lane_count)

    @property
    def aggregate_bits(self) -> int:
        """The whole transfer's bits: the input bits times the lane count."""
        if self.basis is QuantityBasis.AGGREGATE:
            return self.input_bits
        return self.input_bits * self.lane_count


@dataclasses.dataclass(frozen=True)
class ChannelSettings:
    """One physical channel: its name, a propagation latency, a bandwidth.

    No capacity means an unbounded wire that charges its latency only.
    Two paths whose channels carry the same name share one channel, so
    the name is the identity a fabric wires by.
    """

    name: str
    propagation_latency_ticks: int
    capacity: Optional[CapacitySettings]
    configuration_source: str

    def __post_init__(self) -> None:
        propagation_latency_ticks = _as_whole_number(
            self.propagation_latency_ticks, "propagation_latency_ticks"
        )
        object.__setattr__(
            self, "propagation_latency_ticks", propagation_latency_ticks
        )
        if self.propagation_latency_ticks < 0:
            raise ValueError("propagation_latency_ticks must be nonnegative")


@dataclasses.dataclass(frozen=True)
class PathSettings:
    """One path: its channel, its payload rule, and its setup cost.

    At least one of the default payload and the actual payload source is
    given. A default payload on a bounded channel is stated on the same
    basis as the channel's capacity, so the two describe the same lanes.
    setup_ticks is paid on the channel's setup engine before every
    transfer of the path; zero means the path programs nothing.
    """

    channel: ChannelSettings
    default_payload: Optional[PayloadSettings]
    actual_payload_source: Optional[str]
    setup_ticks: int = 0

    def __post_init__(self) -> None:
        has_default = self.default_payload is not None
        has_actual_source = self.actual_payload_source is not None
        if not has_default and not has_actual_source:
            raise ValueError(
                "a path needs a default payload or an actual payload source"
            )
        setup_ticks = _as_whole_number(self.setup_ticks, "setup_ticks")
        object.__setattr__(self, "setup_ticks", setup_ticks)
        if self.setup_ticks < 0:
            raise ValueError("setup_ticks must be nonnegative")
        capacity = self.channel.capacity
        default_payload = self.default_payload
        if capacity is None or default_payload is None:
            return
        if capacity.basis is not default_payload.basis:
            raise ValueError("capacity and payload bases must match")
        if capacity.lane_count != default_payload.lane_count:
            raise ValueError("capacity and payload lane counts must match")


@dataclasses.dataclass(frozen=True)
class FabricSettings:
    """A fabric card: one path setting per hop, a profile name and a kind.

    Every hop of the reaction path is priced, so a card names all eleven
    and a caller that leaves one out is refused where it constructs the
    card. A card whose QPU-to-controller latency leaves out the
    controller's readout processing says so, because the timing card
    prices that processing on its own line. kind names the row of
    LINK_FABRICS (link_profiles.py) that supplied these numbers and
    builds the run's fabric from them; profile_name is the same row's
    name with the yaml's own file appended, for the traffic report.
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
    profile_name: str
    kind: str = "logical_reference"
    is_controller_processing_outside_qpu_to_controller: bool = False

    def __post_init__(self) -> None:
        channel_by_name = {}
        for path in transfer_records.LinkPath:
            path_settings = self.path_settings(path)
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


def _as_whole_number(value, name: str) -> int:
    """A count as an exact int, or a ValueError naming the field.

    3.0 is fine; 3.5, NaN and None are not.
    """
    try:
        whole = int(value)
    except (OverflowError, ValueError, TypeError):
        raise ValueError(f"{name} must be a finite whole number") from None
    if whole != value:
        raise ValueError(f"{name} must be a finite whole number")
    return whole


def _require_finite_number(value, name: str) -> None:
    """Refuse NaN and infinity.

    A Python int of any size is finite: math.isfinite overflows converting
    it to float, and that overflow reads as finite.
    """
    try:
        is_finite = math.isfinite(value)
    except OverflowError:
        is_finite = True
    if not is_finite:
        raise ValueError(f"{name} must be a finite number")


def _lane_count_for(
    basis: QuantityBasis, lane_count, name: str
) -> Optional[int]:
    """The lane count a quantity's basis allows.

    An aggregate quantity has no lane count; a per-lane one needs a
    positive count.
    """
    if basis is QuantityBasis.AGGREGATE:
        if lane_count is not None:
            raise ValueError(f"an aggregate {name} has no lane count")
        return None
    whole_count = _as_whole_number(lane_count, f"per-lane {name} count")
    if whole_count <= 0:
        raise ValueError(f"per-lane {name} count must be positive")
    return whole_count
