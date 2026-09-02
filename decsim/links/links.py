"""The links between the components on the reaction path.

A link is one physical channel: a wire with a propagation latency and,
optionally, a finite bandwidth. A transfer on a bounded channel waits for
the previous transfer to leave the wire, is serialized at the channel's
rate, then propagates. Delivery is the send time plus the wait plus the
serialization plus the latency. That is the ns-3 point-to-point device
(PointToPointNetDevice: one packet on the wire at a time, the receiver
gets it one propagation delay after the last bit leaves). A fractional
tick of serialization rounds up, because a transfer can never end before
its exact time. An unbounded channel has no queue and no serialization;
it charges its latency only.

A path is one named hop between two components (LinkPath). A card
(LinkModelConfig; the number cards are in link_profiles.py) gives every
path an edge: the channel it rides, an optional default payload, and the
name of the runtime quantity that supplies the actual payload. Two edges
that hold the same LinkConfig object share one wire.

An edge may carry a per-transfer setup overhead: the descriptor and
doorbell work a processor does before the data mover starts, as in
gem5-Aladdin's DMA model (Shao et al., MICRO 2016). Setups serialize on
the one processor that programs the channel, while the wire keeps
streaming during a setup. Every request on a channel, a free edge
included, passes the channel's setup queue, so a channel sees its
requests in request order whichever paths share it.

LinkModel is one run's fabric. LinkModel.reserve is the one call the
runtime makes: it checks the attribution against the path's rule, selects
the payload, queues the setup, reserves the wire, and appends one transfer
record to the ledger. Every reservation is counted per path and per
channel; the traffic report refuses counts that do not reconcile.

The paths, one per pair of components on the reaction path:

- ``QC``, QPU to controller: syndrome readout leaving the QPU.
- ``CWB``, controller to syndrome buffer 0: a completed round published
  to the window-input route; optional, a card without it publishes for
  free.
- ``CSB``, controller to syndrome buffer 1: the packed round's second
  write, out of the fridge into the room-side store that feeds the strong
  tier; optional, a card without it stores for free.
- ``WBD``, weak buffer to weak decoder: syndrome data reaching the weak
  tier, as one window assembled from buffer 0 (window_manager) or as one
  feedback-memory round straight off packing (syndrome_packing). The
  feedback-memory sends are controller-sourced traffic on the shared
  channel: real-time stacks run one syndrome stream and classify
  downstream rather than wiring idle or feedback data separately
  (Battistel 2303.00054 Fig. 3; Google 2408.13687; 2605.30765).
- ``WSD``, weak decoder to strong decoder: the escalation that hands a
  window to the strong tier.
- ``SBD``, strong buffer to strong decoder: the strong window's syndrome
  input, assembled from syndrome buffer 1.
- ``WDO``, weak decoder to Pauli frame: the weak correction leaving the
  weak tier for the frame and the conditional release.
- ``DD``, decoder to decoder: a committed window boundary handed to a
  dependent window.
- ``DO``, strong decoder to Pauli frame: the strong correction leaving
  the strong tier.
- ``OC``, Pauli frame to controller: the conditional release returning to
  the controller.
- ``CQ``, controller to QPU: the instruction delivered back to the QPU.

Each path's meaning is declared once, in ``_RULE_BY_PATH``: what its
transfers are attributed to (an operation, a round, a window), which
provenance relation they carry (a decoder request, a boundary), and
whether every card must wire it. Adding a path: add the member to
``LinkPath``, its row to ``_RULE_BY_PATH``, and an
``Optional[LinkEdgeConfig]`` field to ``LinkModelConfig``; everything
else iterates the card's wired paths.
"""

import dataclasses
import enum
import fractions
import math
from typing import Optional, Union

import decsim.config as config
import decsim.message as message


class LinkQuantityBasis(str, enum.Enum):
    """Whether one configured quantity is aggregate or per active channel."""

    DIRECT_AGGREGATE = "direct_aggregate"
    PER_CHANNEL = "per_channel"


@dataclasses.dataclass(frozen=True)
class LinkCapacityConfig:
    """Bandwidth of one channel: bits per microsecond, aggregate or per channel.

    The rate may be any finite real number, a Fraction included. It is
    kept as given, and the aggregate reported to the topology is the
    input times the channel count in the input's own arithmetic. Only
    the serialization arithmetic is exact: it reads the rate as the
    decimal written on the card and multiplies by the count as a
    Fraction (exact_aggregate_bits_per_microsecond).
    """

    input_bits_per_microsecond: float
    basis: LinkQuantityBasis
    channel_count: Optional[int]
    source: str

    def __post_init__(self) -> None:
        _require_finite_number(
            self.input_bits_per_microsecond, "input_bits_per_microsecond"
        )
        if self.input_bits_per_microsecond <= 0:
            raise ValueError("input_bits_per_microsecond must be positive")
        channel_count = _normalized_channel_count(
            self.basis, self.channel_count, "capacity"
        )
        object.__setattr__(self, "channel_count", channel_count)
        if self.basis is LinkQuantityBasis.PER_CHANNEL:
            aggregate = self.aggregate_bits_per_microsecond
            _require_finite_number(aggregate, "aggregate_bits_per_microsecond")

    @property
    def aggregate_bits_per_microsecond(
        self,
    ) -> Union[int, float, fractions.Fraction]:
        """The whole channel's rate as reported: input times channel count.

        The result has the input's own type, a Fraction included.
        """
        if self.basis is LinkQuantityBasis.DIRECT_AGGREGATE:
            return self.input_bits_per_microsecond
        return self.input_bits_per_microsecond * self.channel_count

    def exact_aggregate_bits_per_microsecond(self) -> fractions.Fraction:
        """The whole channel's rate, exact, for the serialization arithmetic.

        The rate written on the card is read as the decimal it shows (the
        Fraction of its text), never as the binary float's hidden
        expansion, and multiplied by the channel count exactly. ns-3's
        DataRate does the same arithmetic in integers.
        """
        rate_text = str(self.input_bits_per_microsecond)
        rate = fractions.Fraction(rate_text)
        if self.basis is LinkQuantityBasis.DIRECT_AGGREGATE:
            return rate
        return rate * self.channel_count

    def to_json_value(self) -> dict:
        """The capacity as the topology report writes it."""
        return {
            "basis": self.basis.value,
            "input_bits_per_us": self.input_bits_per_microsecond,
            "channel_count": self.channel_count,
            "source": self.source,
            "aggregate_bits_per_us": self.aggregate_bits_per_microsecond,
        }


@dataclasses.dataclass(frozen=True)
class PayloadSizeConfig:
    """Default payload of one path: bits, aggregate or per channel."""

    input_bits: int
    basis: LinkQuantityBasis
    channel_count: Optional[int]
    source: str

    def __post_init__(self) -> None:
        input_bits = _as_whole_number(self.input_bits, "input_bits")
        object.__setattr__(self, "input_bits", input_bits)
        if self.input_bits < 0:
            raise ValueError("input_bits must be nonnegative")
        channel_count = _normalized_channel_count(
            self.basis, self.channel_count, "payload"
        )
        object.__setattr__(self, "channel_count", channel_count)

    @property
    def aggregate_bits(self) -> int:
        """The whole transfer's bits: the input bits times the channel count."""
        if self.basis is LinkQuantityBasis.DIRECT_AGGREGATE:
            return self.input_bits
        return self.input_bits * self.channel_count

    def to_json_value(self) -> dict:
        """The payload as the topology report writes it."""
        return {
            "basis": self.basis.value,
            "input_bits": self.input_bits,
            "channel_count": self.channel_count,
            "source": self.source,
            "aggregate_bits": self.aggregate_bits,
        }


@dataclasses.dataclass(frozen=True)
class LinkConfig:
    """One physical channel: a propagation latency and an optional bandwidth.

    No capacity means an unbounded wire. Two edges that hold the same
    LinkConfig object share the wire; build() tells channels apart by the
    object, so two equal cards on different objects are two wires.
    """

    propagation_latency_ticks: int
    capacity: Optional[LinkCapacityConfig]
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
class TransferOverheadConfig:
    """Fixed setup cost paid before each transfer on one path reaches the wire.

    This is the descriptor programming and doorbell work a processor does
    before the data mover starts. gem5-Aladdin charges it on the processor
    while issued transfers keep streaming, and successive setups serialize
    on the one processor doing them (Shao et al., MICRO 2016: initiating a
    DMA from the CPU costs 17 cycles plus housekeeping). The
    size-proportional cache flush and invalidate they measure are not
    modeled at this altitude.
    """

    overhead_ticks: int
    source: str

    def __post_init__(self) -> None:
        overhead_ticks = _as_whole_number(self.overhead_ticks, "overhead_ticks")
        object.__setattr__(self, "overhead_ticks", overhead_ticks)
        if self.overhead_ticks < 0:
            raise ValueError("overhead_ticks must be nonnegative")


@dataclasses.dataclass(frozen=True)
class LinkEdgeConfig:
    """One path on a card: its channel, its default payload, its payload source.

    At least one of the default payload and the actual payload source is
    given. A default payload on a bounded channel is stated on the same
    basis as the channel's capacity, so the two describe the same lanes.
    """

    channel: LinkConfig
    default_payload: Optional[PayloadSizeConfig]
    actual_payload_source: Optional[str]
    transfer_overhead: Optional[TransferOverheadConfig] = None

    def __post_init__(self) -> None:
        has_default = self.default_payload is not None
        has_actual_source = self.actual_payload_source is not None
        if not has_default and not has_actual_source:
            raise ValueError(
                "an edge requires a configured default or actual payload source"
            )
        capacity = self.channel.capacity
        default_payload = self.default_payload
        if capacity is None or default_payload is None:
            return
        if capacity.basis is not default_payload.basis:
            raise ValueError("capacity and payload bases must match")
        if capacity.channel_count != default_payload.channel_count:
            raise ValueError("capacity and payload channel counts must match")


class LinkPath(str, enum.Enum):
    """The hops of the reaction path; see the module docstring."""

    QC = "qc"
    CWB = "cwb"
    WBD = "wbd"
    WSD = "wsd"
    SBD = "sbd"
    WDO = "wdo"
    DD = "dd"
    DO = "do"
    OC = "oc"
    CQ = "cq"
    CSB = "csb"


class LinkAttributionScope(str, enum.Enum):
    """What one path's transfers are attributed to."""

    OPERATION_ONLY = "operation_only"
    ROUND = "round"
    ROUND_OR_WINDOW = "round_or_window"
    WINDOW = "window"


class LinkRelationRule(str, enum.Enum):
    """Which provenance record one path's transfers must carry."""

    NONE = "none"
    REQUEST = "request"
    REQUEST_WHEN_WINDOWED = "request_when_windowed"
    BOUNDARY = "boundary"


@dataclasses.dataclass(frozen=True)
class LinkPathRule:
    """The fixed meaning of one path.

    What its transfers are attributed to, which provenance they carry, and
    whether every card must wire it: the original nine paths are required,
    a path added later may be optional.
    """

    scope: LinkAttributionScope
    relation: LinkRelationRule
    required: bool


@dataclasses.dataclass(frozen=True)
class RequestTransferRelation:
    """Provenance tying one transfer to the decoder request it serves."""

    request_key: message.DecoderRequestKey


@dataclasses.dataclass(frozen=True)
class BoundaryTransferRelation:
    """Provenance tying one DD transfer to the boundary it delivers.

    From which window, produced by which request, to which window, and
    both revisions.
    """

    source_request_key: message.DecoderRequestKey
    source_window_key: tuple
    destination_window_key: tuple
    source_revision: int
    delivery_revision: int


@dataclasses.dataclass(frozen=True)
class TrafficAttribution:
    """Whose transfer this is.

    The operation, its patches, and the window or the inclusive round
    range the bits belong to, plus the relation the path's rule asks for.
    """

    # An operation id is whatever the front end chose; the links never
    # look inside it.
    operation_id: object
    patch_ids: tuple
    window_id: Optional[int]
    first_round: Optional[int]
    last_round: Optional[int]
    relation: Optional[
        Union[RequestTransferRelation, BoundaryTransferRelation]
    ] = None


class PayloadSelectionSource(str, enum.Enum):
    """How a transfer selected its aggregate payload size."""

    ACTUAL = "actual"
    CONFIGURED_DEFAULT = "configured_default"
    UNRESOLVED = "unresolved"


@dataclasses.dataclass(frozen=True)
class LinkReservation:
    """One interval on one wire and the setup that preceded it.

    send_ticks is the wire time. The transfer was requested setup_ticks
    earlier, and total_delay_ticks counts from that request: setup, then
    the queue wait, the serialization, and the propagation.
    """

    payload_bits: Optional[int]
    send_ticks: int
    queue_wait_ticks: int
    serialization_ticks: int
    propagation_ticks: int
    serializer_start_ticks: int
    serializer_end_ticks: int
    total_delay_ticks: int
    physical_sequence: int
    setup_ticks: int = 0


@dataclasses.dataclass(frozen=True)
class SemanticTransferRecord:
    """One entry of the ledger.

    Which path, on which channel, for whom, with which payload, and the
    reservation it got.
    """

    path: LinkPath
    physical_alias: str
    attribution: TrafficAttribution
    payload_selection: PayloadSelectionSource
    payload_source: str
    reservation: LinkReservation


@dataclasses.dataclass(frozen=True)
class TrafficCounters:
    """Additive counters kept per path and per channel."""

    transfer_count: int = 0
    known_payload_bits: int = 0
    unknown_payload_transfer_count: int = 0
    serialization_ticks: int = 0
    propagation_ticks: int = 0
    queue_wait_ticks: int = 0

    def plus_reservation(
        self, reservation: LinkReservation
    ) -> "TrafficCounters":
        """These counters with one more reservation added."""
        known_payload_bits = self.known_payload_bits
        unknown_payload_transfer_count = self.unknown_payload_transfer_count
        if reservation.payload_bits is None:
            unknown_payload_transfer_count += 1
        else:
            known_payload_bits += reservation.payload_bits
        transfer_count = self.transfer_count + 1
        serialization_ticks = (
            self.serialization_ticks + reservation.serialization_ticks
        )
        propagation_ticks = (
            self.propagation_ticks + reservation.propagation_ticks
        )
        queue_wait_ticks = self.queue_wait_ticks + reservation.queue_wait_ticks
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


class Link:
    """One physical channel at run time: a wire that sends in time order.

    A payload waits for the serializer to be free, is serialized at the
    channel's rate, then propagates. An unbounded channel never queues.
    """

    def __init__(self, channel_config: LinkConfig):
        self._channel_config = channel_config
        self._next_free_tick = 0
        self._last_send_tick: Optional[int] = None
        self._physical_sequence = 0
        self._counters = TrafficCounters()

    @property
    def channel_config(self) -> LinkConfig:
        """The channel's settings."""
        return self._channel_config

    def counters_snapshot(self) -> TrafficCounters:
        """What the channel has carried so far."""
        return self._counters

    def reserve(
        self, *, payload_bits: Optional[int], now_ticks: int
    ) -> LinkReservation:
        """Reserve one interval on the wire starting now and return its timing.

        A send before the previous send is refused: the wire is a queue,
        and requests reach it in time order.
        """
        payload_bits = _checked_payload_bits(payload_bits)
        now_ticks = self._checked_send_ticks(now_ticks)
        reservation = self._timing_from(payload_bits, now_ticks)
        if self._channel_config.capacity is not None:
            self._next_free_tick = reservation.serializer_end_ticks
        self._last_send_tick = now_ticks
        self._physical_sequence += 1
        self._counters = self._counters.plus_reservation(reservation)
        return reservation

    def _checked_send_ticks(self, now_ticks) -> int:
        now_ticks = _as_whole_number(now_ticks, "now_ticks")
        if now_ticks < 0:
            raise ValueError("now_ticks must be nonnegative")
        has_earlier_send = self._last_send_tick is not None
        if has_earlier_send and now_ticks < self._last_send_tick:
            raise ValueError("now_ticks must not precede the prior reservation")
        return now_ticks

    def _timing_from(
        self, payload_bits: Optional[int], now_ticks: int
    ) -> LinkReservation:
        serializer_start_ticks = now_ticks
        serialization_ticks = 0
        capacity = self._channel_config.capacity
        if capacity is not None:
            serializer_start_ticks = max(now_ticks, self._next_free_tick)
            serialization_ticks = _serialization_ticks(payload_bits, capacity)
        serializer_end_ticks = serializer_start_ticks + serialization_ticks
        queue_wait_ticks = serializer_start_ticks - now_ticks
        propagation_ticks = self._channel_config.propagation_latency_ticks
        channel_ticks = queue_wait_ticks + serialization_ticks
        total_delay_ticks = channel_ticks + propagation_ticks
        return LinkReservation(
            payload_bits=payload_bits,
            send_ticks=now_ticks,
            queue_wait_ticks=queue_wait_ticks,
            serialization_ticks=serialization_ticks,
            propagation_ticks=propagation_ticks,
            serializer_start_ticks=serializer_start_ticks,
            serializer_end_ticks=serializer_end_ticks,
            total_delay_ticks=total_delay_ticks,
            physical_sequence=self._physical_sequence,
        )


@dataclasses.dataclass(frozen=True)
class LinkModelConfig:
    """A fabric card: one edge per path plus a profile name.

    The nine original paths are required; cwb and csb are optional.
    """

    qc: LinkEdgeConfig
    wbd: LinkEdgeConfig
    wsd: LinkEdgeConfig
    sbd: LinkEdgeConfig
    wdo: LinkEdgeConfig
    dd: LinkEdgeConfig
    do: LinkEdgeConfig
    oc: LinkEdgeConfig
    cq: LinkEdgeConfig
    profile_name: str
    qc_excludes_controller_processing: bool = False
    cwb: Optional[LinkEdgeConfig] = None
    csb: Optional[LinkEdgeConfig] = None

    def wired_paths(self) -> tuple:
        """The paths this card wires, in vocabulary order.

        A card that leaves out a required path is refused.
        """
        wired_paths = []
        for path in LinkPath:
            edge = getattr(self, path.value)
            if edge is not None:
                wired_paths.append(path)
                continue
            rule = _RULE_BY_PATH[path]
            if rule.required:
                raise ValueError(f"{path.value} is a required link path")
        return tuple(wired_paths)

    def build(self) -> "LinkModel":
        """Build the run's fabric: one Link per distinct channel object."""
        channel_by_config_id = {}
        binding_by_path = {}
        for path in self.wired_paths():
            edge = getattr(self, path.value)
            channel_config = edge.channel
            channel_config_id = id(channel_config)
            channel = channel_by_config_id.get(channel_config_id)
            if channel is None:
                channel = Link(channel_config)
                channel_by_config_id[channel_config_id] = channel
            binding_by_path[path] = (edge, channel)
        return LinkModel(self, binding_by_path)


@dataclasses.dataclass(frozen=True)
class LinkEdgeSnapshot:
    """One wired path, its channel's alias, its edge, and its counters."""

    path: LinkPath
    physical_alias: str
    edge: LinkEdgeConfig
    counters: TrafficCounters


@dataclasses.dataclass(frozen=True)
class LinkChannelSnapshot:
    """One channel, the paths that ride it, its settings, and its counters."""

    alias: str
    member_paths: tuple
    config: LinkConfig
    counters: TrafficCounters


@dataclasses.dataclass(frozen=True)
class LinkFabricSnapshot:
    """What a run's link fabric looked like and carried, frozen for reports."""

    profile_name: str
    paths: tuple
    edges: tuple
    channels: tuple
    transfers: tuple


class LinkModel:
    """One run's fabric: the wired paths, their channels, and the ledger."""

    def __init__(self, model_config: LinkModelConfig, binding_by_path: dict):
        self._model_config = model_config
        # A binding is the pair (edge, channel) a path resolved to.
        self._binding_by_path = dict(binding_by_path)
        self._paths = tuple(self._binding_by_path)
        self._counters_by_path = {
            path: TrafficCounters() for path in self._paths
        }
        # Setups serialize per channel, not per path (module docstring).
        self._setup_queue_by_channel: dict[Link, _SetupQueue] = {}
        self._transfers: list[SemanticTransferRecord] = []
        self._alias_by_channel: dict[Link, str] = {}
        for path in self._paths:
            _edge, channel = self._binding_by_path[path]
            if channel not in self._alias_by_channel:
                alias = f"channel-{len(self._alias_by_channel)}"
                self._alias_by_channel[channel] = alias
                self._setup_queue_by_channel[channel] = _SetupQueue()

    @property
    def paths(self) -> tuple:
        """The paths this fabric wires, in vocabulary order."""
        return self._paths

    def reserve(
        self,
        path: LinkPath,
        *,
        payload_bits: Optional[int],
        now_ticks: int,
        attribution: TrafficAttribution,
    ) -> LinkReservation:
        """Send one transfer on a path and return its timing.

        The attribution is checked against the path's rule, the payload is
        selected, the setup is queued on the channel, the wire is reserved,
        and the transfer is recorded in the ledger.
        """
        if path not in self._binding_by_path:
            raise ValueError(f"{path.value} is not wired in this link fabric")
        _check_attribution(path, attribution)
        edge, channel = self._binding_by_path[path]
        selected_bits, selection, payload_source = _select_payload(
            path, edge, payload_bits
        )
        now_ticks, selected_bits = _checked_request(
            path, now_ticks, selected_bits
        )
        setup_ticks = self._queue_setup(path, channel, edge, now_ticks)
        wire_ticks = now_ticks + setup_ticks
        reservation = channel.reserve(
            payload_bits=selected_bits, now_ticks=wire_ticks
        )
        reservation = _with_setup(reservation, setup_ticks)
        counters = self._counters_by_path[path]
        self._counters_by_path[path] = counters.plus_reservation(reservation)
        self._record_transfer(
            path, channel, attribution, selection, payload_source, reservation
        )
        return reservation

    def snapshot(self) -> LinkFabricSnapshot:
        """A frozen view for reports.

        The wiring, every channel's counters, every path's counters, and
        the whole transfer ledger.
        """
        channels = []
        for channel, alias in self._alias_by_channel.items():
            channel_snapshot = self._channel_snapshot(channel, alias)
            channels.append(channel_snapshot)
        edges = []
        for path in self._paths:
            edge_snapshot = self._edge_snapshot(path)
            edges.append(edge_snapshot)
        return LinkFabricSnapshot(
            profile_name=self._model_config.profile_name,
            paths=self._paths,
            edges=tuple(edges),
            channels=tuple(channels),
            transfers=tuple(self._transfers),
        )

    def _queue_setup(
        self,
        path: LinkPath,
        channel: Link,
        edge: LinkEdgeConfig,
        now_ticks: int,
    ) -> int:
        """Ticks from the request to the wire.

        The setup waits for the channel's previous setup to finish, then
        costs the edge's overhead; the channel's setup queue moves.
        Requests on one channel arrive in nondecreasing order; a request
        earlier than the channel's previous one is a broken contract and
        is refused before the queue moves. Every other check on the
        request has already passed, so a refusal anywhere leaves the
        queue as it was.
        """
        queue = self._setup_queue_by_channel[channel]
        has_earlier_request = queue.last_request_ticks is not None
        if has_earlier_request and now_ticks < queue.last_request_ticks:
            raise RuntimeError(
                f"{path.value} requested at tick {now_ticks}, before the "
                f"channel's previous request at tick "
                f"{queue.last_request_ticks}; requests on one channel arrive "
                f"in nondecreasing order"
            )
        overhead_ticks = 0
        if edge.transfer_overhead is not None:
            overhead_ticks = edge.transfer_overhead.overhead_ticks
        setup_start_ticks = max(now_ticks, queue.free_ticks)
        wire_ticks = setup_start_ticks + overhead_ticks
        queue.free_ticks = wire_ticks
        queue.last_request_ticks = now_ticks
        return wire_ticks - now_ticks

    def _record_transfer(
        self,
        path: LinkPath,
        channel: Link,
        attribution: TrafficAttribution,
        selection: PayloadSelectionSource,
        payload_source: str,
        reservation: LinkReservation,
    ) -> None:
        record = SemanticTransferRecord(
            path=path,
            physical_alias=self._alias_by_channel[channel],
            attribution=attribution,
            payload_selection=selection,
            payload_source=payload_source,
            reservation=reservation,
        )
        self._transfers.append(record)

    def _channel_snapshot(
        self, channel: Link, alias: str
    ) -> LinkChannelSnapshot:
        member_paths = self._member_paths(channel)
        counters = channel.counters_snapshot()
        return LinkChannelSnapshot(
            alias=alias,
            member_paths=member_paths,
            config=channel.channel_config,
            counters=counters,
        )

    def _edge_snapshot(self, path: LinkPath) -> LinkEdgeSnapshot:
        edge, channel = self._binding_by_path[path]
        return LinkEdgeSnapshot(
            path=path,
            physical_alias=self._alias_by_channel[channel],
            edge=edge,
            counters=self._counters_by_path[path],
        )

    def _member_paths(self, channel: Link) -> tuple:
        member_paths = []
        for path in self._paths:
            _edge, member_channel = self._binding_by_path[path]
            if member_channel is channel:
                member_paths.append(path)
        return tuple(member_paths)


@dataclasses.dataclass
class _SetupQueue:
    """A channel's setup engine: its last setup's end and its last request tick.

    Setups serialize per channel, not per path, so the tick at which the
    queue frees belongs to the channel; the last request tick lets a
    refusal check the request order before the queue moves.
    """

    free_ticks: int = 0
    last_request_ticks: Optional[int] = None


_RULE_BY_PATH = {
    LinkPath.QC: LinkPathRule(
        LinkAttributionScope.ROUND, LinkRelationRule.NONE, required=True
    ),
    LinkPath.CWB: LinkPathRule(
        LinkAttributionScope.ROUND, LinkRelationRule.NONE, required=False
    ),
    LinkPath.WBD: LinkPathRule(
        LinkAttributionScope.ROUND_OR_WINDOW,
        LinkRelationRule.REQUEST_WHEN_WINDOWED,
        required=True,
    ),
    LinkPath.WSD: LinkPathRule(
        LinkAttributionScope.WINDOW, LinkRelationRule.REQUEST, required=True
    ),
    LinkPath.SBD: LinkPathRule(
        LinkAttributionScope.WINDOW, LinkRelationRule.REQUEST, required=True
    ),
    LinkPath.WDO: LinkPathRule(
        LinkAttributionScope.WINDOW, LinkRelationRule.REQUEST, required=True
    ),
    LinkPath.DD: LinkPathRule(
        LinkAttributionScope.WINDOW, LinkRelationRule.BOUNDARY, required=True
    ),
    LinkPath.DO: LinkPathRule(
        LinkAttributionScope.WINDOW, LinkRelationRule.REQUEST, required=True
    ),
    LinkPath.OC: LinkPathRule(
        LinkAttributionScope.OPERATION_ONLY,
        LinkRelationRule.NONE,
        required=True,
    ),
    LinkPath.CQ: LinkPathRule(
        LinkAttributionScope.OPERATION_ONLY,
        LinkRelationRule.NONE,
        required=True,
    ),
    LinkPath.CSB: LinkPathRule(
        LinkAttributionScope.ROUND, LinkRelationRule.NONE, required=False
    ),
}


def _as_whole_number(value, name: str) -> int:
    """A count or index as an exact int, or a ValueError naming the field.

    3.0 is fine; 3.5, NaN, None, a list and a bool are not.
    """
    whole = _whole_number_or_none(value)
    if whole is None:
        raise ValueError(f"{name} must be a finite whole number")
    return whole


def _whole_number_or_none(value) -> Optional[int]:
    """The value as an exact Python int, or None when it is not one.

    A bool (Python's or numpy's) is not a count, whatever int() makes
    of it.
    """
    value_type = type(value)
    if value_type.__name__ in ("bool", "bool_"):
        return None
    try:
        whole = int(value)
    except (OverflowError, ValueError, TypeError):
        return None
    if whole != value:
        return None
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


def _normalized_channel_count(basis, channel_count, name: str) -> Optional[int]:
    """The channel count a quantity's basis allows.

    An aggregate quantity has no channel count; a per-channel one needs a
    positive count.
    """
    is_aggregate = basis is LinkQuantityBasis.DIRECT_AGGREGATE
    is_per_channel = basis is LinkQuantityBasis.PER_CHANNEL
    if not is_aggregate and not is_per_channel:
        raise ValueError(f"unknown link quantity basis {basis!r}")
    if is_aggregate:
        if channel_count is not None:
            raise ValueError(
                f"direct aggregate {name} requires channel_count=None"
            )
        return None
    whole_count = _as_whole_number(channel_count, f"per-channel {name} count")
    if whole_count <= 0:
        raise ValueError(f"per-channel {name} count must be positive")
    return whole_count


def _checked_payload_bits(payload_bits) -> Optional[int]:
    if payload_bits is None:
        return None
    payload_bits = _as_whole_number(payload_bits, "payload_bits")
    if payload_bits < 0:
        raise ValueError("payload_bits must be nonnegative")
    return payload_bits


def _serialization_ticks(payload_bits, capacity: LinkCapacityConfig) -> int:
    """Whole ticks to put the payload on the wire at the channel's rate.

    A fractional tick rounds up: serialization never ends before the exact
    transmission time. Exact Fraction arithmetic keeps the card's rate
    exact, so a whole-tick duration is never inflated by float error.
    """
    rate = capacity.exact_aggregate_bits_per_microsecond()
    payload = fractions.Fraction(payload_bits)
    bits_times_ticks = payload * config.TICKS_PER_MICROSECOND
    exact_ticks = bits_times_ticks / rate
    return math.ceil(exact_ticks)


def _with_setup(
    reservation: LinkReservation, setup_ticks: int
) -> LinkReservation:
    """The wire reservation with the setup that preceded it counted in."""
    if setup_ticks == 0:
        return reservation
    total_delay_ticks = reservation.total_delay_ticks + setup_ticks
    return dataclasses.replace(
        reservation,
        setup_ticks=setup_ticks,
        total_delay_ticks=total_delay_ticks,
    )


def _checked_request(
    path: LinkPath, now_ticks, payload_bits
) -> tuple[int, Optional[int]]:
    """The request's tick and payload bits as Python ints, or a refusal.

    A wrong caller is a bug; the refusal comes before any state moves,
    so the setup queue, the wire, the counters and the ledger stay as
    they were. From here on every tick and size the fabric stores is a
    Python int, whatever numeric type the caller passed.
    """
    request_tick = _as_count(now_ticks)
    if request_tick is None:
        raise RuntimeError(
            f"{path.value} requested at tick {now_ticks!r}; a request tick "
            f"is a whole number, not negative"
        )
    if payload_bits is None:
        return request_tick, None
    payload = _as_count(payload_bits)
    if payload is None:
        raise RuntimeError(
            f"{path.value} requested with {payload_bits!r} bits; a payload "
            f"is a whole number of bits, not negative"
        )
    return request_tick, payload


def _as_count(value) -> Optional[int]:
    """The value as a Python int when it is whole and not negative, else None.

    None is the answer, not a refusal; the caller phrases the refusal.
    """
    whole = _whole_number_or_none(value)
    if whole is None or whole < 0:
        return None
    return whole


def _select_payload(path: LinkPath, edge: LinkEdgeConfig, payload_bits):
    """The bits a transfer is priced with, and where they came from.

    The actual payload when the caller supplied one (the edge must name
    its source), else the card's default, else unresolved. An unresolved
    payload rides an unbounded channel for its latency alone; a bounded
    wire needs a size to serialize, so it refuses the transfer.
    """
    if payload_bits is not None:
        if edge.actual_payload_source is None:
            raise ValueError(
                f"{path.value} does not declare an actual payload source"
            )
        return (
            payload_bits,
            PayloadSelectionSource.ACTUAL,
            edge.actual_payload_source,
        )
    default_payload = edge.default_payload
    if default_payload is not None:
        return (
            default_payload.aggregate_bits,
            PayloadSelectionSource.CONFIGURED_DEFAULT,
            default_payload.source,
        )
    if edge.channel.capacity is not None:
        raise RuntimeError(
            f"{path.value} has no payload size and its channel is bounded; "
            f"a bounded wire needs a size to serialize"
        )
    return (
        None,
        PayloadSelectionSource.UNRESOLVED,
        edge.actual_payload_source,
    )


def _check_attribution(path: LinkPath, attribution: TrafficAttribution) -> None:
    """Refuse an attribution whose shape is not what the path's rule declares.

    The rule fixes the scope (round, window, operation) and the relation
    kind; the relation must name the same operation and window as the
    attribution.
    """
    rule = _RULE_BY_PATH[path]
    has_window = attribution.window_id is not None
    has_rounds = attribution.first_round is not None
    if not _has_scope(rule.scope, has_window, has_rounds):
        raise ValueError(
            f"{path.value} requires {rule.scope.value} attribution"
        )
    needs_request = rule.relation is LinkRelationRule.REQUEST
    if rule.relation is LinkRelationRule.REQUEST_WHEN_WINDOWED:
        needs_request = has_window
    needs_boundary = rule.relation is LinkRelationRule.BOUNDARY
    _check_relation(path, attribution, needs_request, needs_boundary)


def _has_scope(
    scope: LinkAttributionScope, has_window: bool, has_rounds: bool
) -> bool:
    if scope is LinkAttributionScope.ROUND:
        return has_rounds and not has_window
    if scope is LinkAttributionScope.ROUND_OR_WINDOW:
        return has_rounds
    if scope is LinkAttributionScope.WINDOW:
        return has_window and has_rounds
    return not has_window and not has_rounds


def _check_relation(
    path: LinkPath,
    attribution: TrafficAttribution,
    needs_request: bool,
    needs_boundary: bool,
) -> None:
    relation = attribution.relation
    if needs_request and type(relation) is not RequestTransferRelation:
        raise ValueError(f"{path.value} requires a request relation")
    if needs_boundary and type(relation) is not BoundaryTransferRelation:
        raise ValueError(f"{path.value} requires a boundary relation")
    accepts_relation = needs_request or needs_boundary
    if not accepts_relation and relation is not None:
        raise ValueError(f"{path.value} does not accept a relation")
    if needs_request:
        request_key = relation.request_key
    elif needs_boundary:
        request_key = relation.source_request_key
    else:
        return
    is_same_operation = request_key.operation_id == attribution.operation_id
    is_same_window = request_key.window_id == attribution.window_id
    if not is_same_operation or not is_same_window:
        raise ValueError("transfer relation does not match attribution")
