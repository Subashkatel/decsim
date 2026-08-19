"""The reaction-path links: how long each hop takes and what evidence it leaves.

``LinkPath`` is the closed vocabulary of measured segments, one per pair of
components on the reaction path:

- ``QC``  QPU -> controller: syndrome readout leaving the QPU (t_qc).
- ``C2B`` controller -> syndrome buffer 0: a completed binary round published
  to the window-input route; optional, a card without it publishes for free.
- ``CWD`` controller -> weak decoder: syndrome data reaching the weak tier,
  either as one round (``syndrome_ingress``) or as one weak window
  (``window_manager``) (t_cwd).
- ``WSD`` weak decoder -> strong decoder: the escalation selection that hands a
  window to the strong tier (t_wsd).
- ``CSD`` controller -> strong decoder: the strong window's syndrome input
  (t_csd).
- ``WDO`` weak decoder -> Pauli frame: the weak correction leaving the weak
  tier for the frame and the conditional release; the weak counterpart of ``DO``.
- ``DD``  decoder -> decoder: a committed window boundary handed to a dependent
  window (t_dd).
- ``DO``  strong decoder -> Pauli frame: the strong correction leaving the
  strong tier (t_do).
- ``OC``  Pauli frame -> controller: the conditional release returning to the
  controller (t_oc).
- ``CQ``  controller -> QPU: the instruction delivered back to the QPU (t_cq).

Each path's meaning is declared once, in ``_PATH_RULES``: what its transfers
are attributed to (an operation, a round, a window), which provenance relation
they carry (a decoder request, a boundary), and whether every card must wire
it. Adding a segment: add the member to
``LinkPath``, its row to ``_PATH_RULES`` (optional while adopted), and an
``Optional[LinkEdgeConfig]`` field to ``LinkModelConfig``; nothing else changes
because everything iterates the card's wired paths.

A card (``LinkModelConfig``, see ``link_profiles.py`` for the number cards)
gives each path an edge: the physical channel it rides (propagation latency,
optional finite bandwidth), an optional default payload, and the name of the
runtime quantity that supplies the actual payload. Two paths given the same
``LinkConfig`` object share one physical FIFO. ``resolve()`` builds the
run-owned ``LinkModel``; ``LinkModel.reserve`` is the one call the runtime
makes: it checks the attribution against the path's rule, selects the payload,
reserves the FIFO interval, and appends one transfer record. Every
reservation is counted per path and per channel; the report refuses to emit
counts that do not reconcile.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Optional, Union

from ..config import us
from ..message import DecoderRequestKey


# ---- quantities on the cards -------------------------------------------------

def _whole(value, name: str) -> int:
    """A count or index as an exact int; 3.0 is fine, 3.5 and NaN are not."""
    try:
        normalized = int(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite whole number") from error
    if normalized != value:
        raise ValueError(f"{name} must be a finite whole number")
    return normalized


def _finite(value, name: str):
    """A rate; NaN and infinity are refused. A Python int of any size is finite:
    math.isfinite overflows converting it to float, which reads as finite."""
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = True
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    return value


class LinkQuantityBasis(str, Enum):
    """Whether one configured quantity is aggregate or per active channel."""

    DIRECT_AGGREGATE = "direct_aggregate"
    PER_CHANNEL = "per_channel"


def _channel_count_for(basis, channel_count, name: str) -> Optional[int]:
    """An aggregate quantity has no channel count; a per-channel one needs a
    positive count. Returns the normalized count."""
    if basis is not LinkQuantityBasis.DIRECT_AGGREGATE and basis is not LinkQuantityBasis.PER_CHANNEL:
        raise ValueError(f"unknown link quantity basis {basis!r}")
    if basis is LinkQuantityBasis.DIRECT_AGGREGATE:
        if channel_count is not None:
            raise ValueError(f"direct aggregate {name} requires channel_count=None")
        return None
    normalized = _whole(channel_count, f"per-channel {name} count")
    if normalized <= 0:
        raise ValueError(f"per-channel {name} count must be positive")
    return normalized


@dataclass(frozen=True, eq=False)
class LinkCapacityConfig:
    """Bandwidth of one channel: bits per microsecond, aggregate or per channel."""

    input_bits_per_us: float
    basis: LinkQuantityBasis
    channel_count: Optional[int]
    source: str

    def __post_init__(self) -> None:
        _finite(self.input_bits_per_us, "input_bits_per_us")
        if self.input_bits_per_us <= 0:
            raise ValueError("input_bits_per_us must be positive")
        object.__setattr__(self, "channel_count",
                           _channel_count_for(self.basis, self.channel_count, "capacity"))
        _finite(self.aggregate_bits_per_us, "aggregate_bits_per_us")

    @property
    def aggregate_bits_per_us(self) -> float:
        if self.basis is LinkQuantityBasis.DIRECT_AGGREGATE:
            return self.input_bits_per_us
        return self.input_bits_per_us * self.channel_count

    def to_json_value(self) -> dict:
        return {
            "basis": self.basis.value,
            "input_bits_per_us": self.input_bits_per_us,
            "channel_count": self.channel_count,
            "source": self.source,
            "aggregate_bits_per_us": self.aggregate_bits_per_us,
        }


@dataclass(frozen=True, eq=False)
class PayloadSizeConfig:
    """Default payload of one path: bits, aggregate or per channel."""

    input_bits: int
    basis: LinkQuantityBasis
    channel_count: Optional[int]
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_bits", _whole(self.input_bits, "input_bits"))
        if self.input_bits < 0:
            raise ValueError("input_bits must be nonnegative")
        object.__setattr__(self, "channel_count",
                           _channel_count_for(self.basis, self.channel_count, "payload"))

    @property
    def aggregate_bits(self) -> int:
        if self.basis is LinkQuantityBasis.DIRECT_AGGREGATE:
            return self.input_bits
        return self.input_bits * self.channel_count

    def to_json_value(self) -> dict:
        return {
            "basis": self.basis.value,
            "input_bits": self.input_bits,
            "channel_count": self.channel_count,
            "source": self.source,
            "aggregate_bits": self.aggregate_bits,
        }


@dataclass(frozen=True, eq=False)
class LinkConfig:
    """One physical channel: propagation latency and an optional finite bandwidth
    (an aggregate FIFO). Two paths that share this object share the FIFO."""

    propagation_latency_ticks: int
    capacity: Optional[LinkCapacityConfig]
    configuration_source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "propagation_latency_ticks",
                           _whole(self.propagation_latency_ticks, "propagation_latency_ticks"))
        if self.propagation_latency_ticks < 0:
            raise ValueError("propagation_latency_ticks must be nonnegative")


@dataclass(frozen=True, eq=False)
class LinkEdgeConfig:
    """One path on a card: its channel, its default payload, and the name of the
    runtime quantity that supplies the actual payload (at least one of the two)."""

    channel: LinkConfig
    default_payload: Optional[PayloadSizeConfig]
    actual_payload_source: Optional[str]

    def __post_init__(self) -> None:
        if self.default_payload is None and self.actual_payload_source is None:
            raise ValueError("an edge requires a configured default or actual payload source")
        capacity = self.channel.capacity
        default = self.default_payload
        if capacity is None or default is None:
            return
        if capacity.basis is not default.basis:
            raise ValueError("capacity and payload bases must match")
        if capacity.channel_count != default.channel_count:
            raise ValueError("capacity and payload channel counts must match")


# ---- the vocabulary and its rules ---------------------------------------------

class LinkPath(str, Enum):
    """The measured reaction-path segments; see the module docstring."""

    QC = "qc"
    C2B = "c2b"
    CWD = "cwd"
    WSD = "wsd"
    CSD = "csd"
    WDO = "wdo"
    DD = "dd"
    DO = "do"
    OC = "oc"
    CQ = "cq"


class LinkAttributionScope(str, Enum):
    """What one path's transfers are attributed to."""

    OPERATION_ONLY = "operation_only"
    ROUND = "round"
    ROUND_OR_WINDOW = "round_or_window"
    WINDOW = "window"


class LinkRelationRule(str, Enum):
    """Which provenance record one path's transfers must carry."""

    NONE = "none"
    REQUEST = "request"
    REQUEST_WHEN_WINDOWED = "request_when_windowed"
    BOUNDARY = "boundary"


@dataclass(frozen=True)
class LinkPathRule:
    """The fixed meaning of one segment: what its transfers are attributed
    to, which provenance they carry, and whether every card must wire it (the
    original nine are required; a segment added later may be optional)."""

    scope: LinkAttributionScope
    relation: LinkRelationRule
    required: bool


_PATH_RULES = MappingProxyType({
    LinkPath.QC: LinkPathRule(LinkAttributionScope.ROUND, LinkRelationRule.NONE, True),
    LinkPath.C2B: LinkPathRule(LinkAttributionScope.ROUND, LinkRelationRule.NONE, False),
    LinkPath.CWD: LinkPathRule(LinkAttributionScope.ROUND_OR_WINDOW,
                               LinkRelationRule.REQUEST_WHEN_WINDOWED, True),
    LinkPath.WSD: LinkPathRule(LinkAttributionScope.WINDOW, LinkRelationRule.REQUEST, True),
    LinkPath.CSD: LinkPathRule(LinkAttributionScope.WINDOW, LinkRelationRule.REQUEST, True),
    LinkPath.WDO: LinkPathRule(LinkAttributionScope.WINDOW, LinkRelationRule.REQUEST, True),
    LinkPath.DD: LinkPathRule(LinkAttributionScope.WINDOW, LinkRelationRule.BOUNDARY, True),
    LinkPath.DO: LinkPathRule(LinkAttributionScope.WINDOW, LinkRelationRule.REQUEST, True),
    LinkPath.OC: LinkPathRule(LinkAttributionScope.OPERATION_ONLY, LinkRelationRule.NONE, True),
    LinkPath.CQ: LinkPathRule(LinkAttributionScope.OPERATION_ONLY, LinkRelationRule.NONE, True),
})


# ---- what a transfer says about itself ---------------------------------------

@dataclass(frozen=True)
class RequestTransferRelation:
    """Provenance tying one transfer to the decoder request it serves."""

    request_key: DecoderRequestKey


@dataclass(frozen=True)
class BoundaryTransferRelation:
    """Provenance tying one DD transfer to the boundary it delivers: from which
    window (and which request produced it) to which, and both revisions."""

    source_request_key: DecoderRequestKey
    source_window_key: tuple
    destination_window_key: tuple
    source_revision: int
    delivery_revision: int


@dataclass(frozen=True)
class TrafficAttribution:
    """Whose transfer this is: the operation, its patches, and the window or the
    inclusive round range the bits belong to; plus the relation the path rule asks for."""

    operation_id: object
    patch_ids: tuple
    window_id: Optional[int]
    round_lo: Optional[int]
    round_hi: Optional[int]
    relation: Optional[Union[RequestTransferRelation, BoundaryTransferRelation]] = None


class PayloadSelectionSource(str, Enum):
    """How a transfer selected its aggregate payload size."""

    ACTUAL = "actual"
    CONFIGURED_DEFAULT = "configured_default"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class LinkReservation:
    """One FIFO interval on one channel: when it was sent, how long it waited,
    serialized and propagated, and its position in the channel's order."""

    payload_bits: Optional[int]
    send_ticks: int
    queue_wait_ticks: int
    serialization_ticks: int
    propagation_ticks: int
    serializer_start_ticks: int
    serializer_end_ticks: int
    total_delay_ticks: int
    physical_sequence: int


@dataclass(frozen=True)
class SemanticTransferRecord:
    """One entry of the ledger: which path, on which channel, for whom, with
    which payload, and the reservation it got."""

    path: LinkPath
    physical_alias: str
    attribution: TrafficAttribution
    payload_selection: PayloadSelectionSource
    payload_source: str
    reservation: LinkReservation


@dataclass(frozen=True)
class TrafficCounters:
    """Additive counters kept per path and per channel."""

    transfer_count: int = 0
    known_payload_bits: int = 0
    unknown_payload_transfer_count: int = 0
    serialization_ticks: int = 0
    propagation_ticks: int = 0
    queue_wait_ticks: int = 0

    def plus_reservation(self, reservation: LinkReservation) -> "TrafficCounters":
        payload_known = reservation.payload_bits is not None
        return TrafficCounters(
            transfer_count=self.transfer_count + 1,
            known_payload_bits=self.known_payload_bits + (reservation.payload_bits if payload_known else 0),
            unknown_payload_transfer_count=self.unknown_payload_transfer_count + (not payload_known),
            serialization_ticks=self.serialization_ticks + reservation.serialization_ticks,
            propagation_ticks=self.propagation_ticks + reservation.propagation_ticks,
            queue_wait_ticks=self.queue_wait_ticks + reservation.queue_wait_ticks,
        )

    def plus(self, other: "TrafficCounters") -> "TrafficCounters":
        return TrafficCounters(
            transfer_count=self.transfer_count + other.transfer_count,
            known_payload_bits=self.known_payload_bits + other.known_payload_bits,
            unknown_payload_transfer_count=(self.unknown_payload_transfer_count
                                            + other.unknown_payload_transfer_count),
            serialization_ticks=self.serialization_ticks + other.serialization_ticks,
            propagation_ticks=self.propagation_ticks + other.propagation_ticks,
            queue_wait_ticks=self.queue_wait_ticks + other.queue_wait_ticks,
        )

    def to_json_value(self) -> dict:
        return {
            "transfer_count": self.transfer_count,
            "known_payload_bits": self.known_payload_bits,
            "unknown_payload_transfer_count": self.unknown_payload_transfer_count,
            "serialization_ticks": self.serialization_ticks,
            "propagation_ticks": self.propagation_ticks,
            "queue_wait_ticks": self.queue_wait_ticks,
        }


# ---- the runtime -------------------------------------------------------------

class Link:
    """One resolved physical channel: an aggregate FIFO. Sends are in order of
    time; a payload waits for the serializer to be free, is serialized at the
    channel's bandwidth, then propagates."""

    def __init__(self, config: LinkConfig):
        self._config = config
        self._next_free_tick = 0
        self._last_send_tick = None
        self._physical_sequence = 0
        self._counters = TrafficCounters()

    @property
    def config(self) -> LinkConfig:
        return self._config

    def counters_snapshot(self) -> TrafficCounters:
        return self._counters

    def reserve(self, *, payload_bits: Optional[int], now_ticks: int) -> LinkReservation:
        """Reserve one FIFO interval starting now and return its timing."""
        if payload_bits is not None:
            payload_bits = _whole(payload_bits, "payload_bits")
            if payload_bits < 0:
                raise ValueError("payload_bits must be nonnegative")
        now_ticks = _whole(now_ticks, "now_ticks")
        if now_ticks < 0:
            raise ValueError("now_ticks must be nonnegative")
        if self._last_send_tick is not None and now_ticks < self._last_send_tick:
            raise ValueError("now_ticks must not precede the prior reservation")

        capacity = self._config.capacity
        if capacity is None:                     # unlimited bandwidth: no queue, no serialization
            serializer_start_ticks = now_ticks
            serialization_ticks = 0
        else:
            serializer_start_ticks = max(now_ticks, self._next_free_tick)
            serialization_ticks = us(payload_bits / capacity.aggregate_bits_per_us)
        serializer_end_ticks = serializer_start_ticks + serialization_ticks
        queue_wait_ticks = serializer_start_ticks - now_ticks
        propagation_ticks = self._config.propagation_latency_ticks
        reservation = LinkReservation(
            payload_bits=payload_bits,
            send_ticks=now_ticks,
            queue_wait_ticks=queue_wait_ticks,
            serialization_ticks=serialization_ticks,
            propagation_ticks=propagation_ticks,
            serializer_start_ticks=serializer_start_ticks,
            serializer_end_ticks=serializer_end_ticks,
            total_delay_ticks=queue_wait_ticks + serialization_ticks + propagation_ticks,
            physical_sequence=self._physical_sequence,
        )
        if capacity is not None:
            self._next_free_tick = serializer_end_ticks
        self._last_send_tick = now_ticks
        self._physical_sequence += 1
        self._counters = self._counters.plus_reservation(reservation)
        return reservation


@dataclass(frozen=True)
class LinkModelConfig:
    """A fabric card: one edge per path plus a profile name. The nine original
    paths are required; ``c2b`` is optional."""

    qc: LinkEdgeConfig
    cwd: LinkEdgeConfig
    wsd: LinkEdgeConfig
    csd: LinkEdgeConfig
    wdo: LinkEdgeConfig
    dd: LinkEdgeConfig
    do: LinkEdgeConfig
    oc: LinkEdgeConfig
    cq: LinkEdgeConfig
    profile_name: str
    qc_excludes_controller_processing: bool = False
    c2b: Optional[LinkEdgeConfig] = None

    def wired_paths(self) -> tuple:
        """The paths this card wires, in vocabulary order; a missing required path refuses."""
        wired = []
        for path in LinkPath:
            if getattr(self, path.value, None) is not None:
                wired.append(path)
            elif _PATH_RULES[path].required:
                raise ValueError(f"{path.value} is a required link path")
        return tuple(wired)

    def resolve(self) -> "LinkModel":
        """Build the run-owned fabric: one Link per distinct channel object."""
        channel_by_config_id = {}
        bindings = {}
        for path in self.wired_paths():
            edge = getattr(self, path.value)
            channel = channel_by_config_id.get(id(edge.channel))
            if channel is None:
                channel = Link(edge.channel)
                channel_by_config_id[id(edge.channel)] = channel
            bindings[path] = (edge, channel)
        return LinkModel(self, bindings)


@dataclass(frozen=True)
class LinkEdgeSnapshot:
    path: LinkPath
    physical_alias: str
    edge: LinkEdgeConfig
    counters: TrafficCounters


@dataclass(frozen=True)
class LinkChannelSnapshot:
    alias: str
    member_paths: tuple
    config: LinkConfig
    counters: TrafficCounters


@dataclass(frozen=True)
class LinkFabricSnapshot:
    """What a run's link fabric looked like and carried, frozen for reports."""

    profile_name: str
    paths: tuple
    edges: tuple
    channels: tuple
    transfers: tuple


class LinkModel:
    """One run's fabric: the wired paths, their channels, and the transfer ledger."""

    def __init__(self, config: LinkModelConfig, bindings: dict):
        self._config = config
        self._bindings = dict(bindings)                # path -> (edge, channel)
        self._paths = tuple(self._bindings)
        self._semantic_counters = {path: TrafficCounters() for path in self._paths}
        self._transfers = []
        self._alias_by_channel = {}
        for path in self._paths:
            _edge, channel = self._bindings[path]
            if channel not in self._alias_by_channel:
                self._alias_by_channel[channel] = f"channel-{len(self._alias_by_channel)}"

    @property
    def paths(self) -> tuple:
        """The paths this fabric wires, in vocabulary order."""
        return self._paths

    def reserve(self, path: LinkPath, *, payload_bits: Optional[int], now_ticks: int,
                attribution: TrafficAttribution) -> LinkReservation:
        """Send one transfer on a path: check the attribution against the path's
        rule, select the payload, reserve the channel, record the transfer."""
        if path not in self._bindings:
            raise ValueError(f"{path.value} is not wired in this link fabric")
        _check_attribution(path, attribution)
        edge, channel = self._bindings[path]
        selected_bits, selection, payload_source = _select_payload(path, edge, payload_bits)
        reservation = channel.reserve(payload_bits=selected_bits, now_ticks=now_ticks)
        self._semantic_counters[path] = self._semantic_counters[path].plus_reservation(reservation)
        self._transfers.append(SemanticTransferRecord(
            path=path,
            physical_alias=self._alias_by_channel[channel],
            attribution=attribution,
            payload_selection=selection,
            payload_source=payload_source,
            reservation=reservation,
        ))
        return reservation

    def _member_paths(self, channel: Link) -> tuple:
        return tuple(path for path in self._paths if self._bindings[path][1] is channel)

    def snapshot(self) -> LinkFabricSnapshot:
        """Frozen view for reports: the wiring, every channel's counters and
        the whole transfer ledger."""
        channels = []
        for channel, alias in self._alias_by_channel.items():
            channels.append(LinkChannelSnapshot(
                alias=alias, member_paths=self._member_paths(channel),
                config=channel.config, counters=channel.counters_snapshot()))
        edges = []
        for path in self._paths:
            edge, channel = self._bindings[path]
            edges.append(LinkEdgeSnapshot(
                path=path, physical_alias=self._alias_by_channel[channel],
                edge=edge, counters=self._semantic_counters[path]))
        return LinkFabricSnapshot(
            profile_name=self._config.profile_name, paths=self._paths,
            edges=tuple(edges), channels=tuple(channels), transfers=tuple(self._transfers))


def _select_payload(path: LinkPath, edge: LinkEdgeConfig, payload_bits):
    """The bits a transfer is priced with: the actual payload when the caller
    supplied one (the edge must name its source), else the card's default,
    else unresolved (priced as zero bits)."""
    if payload_bits is not None:
        if edge.actual_payload_source is None:
            raise ValueError(f"{path.value} does not declare an actual payload source")
        return payload_bits, PayloadSelectionSource.ACTUAL, edge.actual_payload_source
    if edge.default_payload is not None:
        return (edge.default_payload.aggregate_bits, PayloadSelectionSource.CONFIGURED_DEFAULT,
                edge.default_payload.source)
    return None, PayloadSelectionSource.UNRESOLVED, edge.actual_payload_source


def _check_attribution(path: LinkPath, attribution: TrafficAttribution) -> None:
    """The attribution has the shape the path's rule declares: the right scope
    (round, window, operation), the right relation kind, and a relation that
    names the same operation and window as the attribution."""
    rule = _PATH_RULES[path]
    has_window = attribution.window_id is not None
    has_rounds = attribution.round_lo is not None
    scope_ok = {
        LinkAttributionScope.ROUND: not has_window and has_rounds,
        LinkAttributionScope.ROUND_OR_WINDOW: has_rounds,
        LinkAttributionScope.WINDOW: has_window and has_rounds,
        LinkAttributionScope.OPERATION_ONLY: not has_window and not has_rounds,
    }[rule.scope]
    if not scope_ok:
        raise ValueError(f"{path.value} requires {rule.scope.value} attribution")

    relation = attribution.relation
    needs_request = (rule.relation is LinkRelationRule.REQUEST
                     or (rule.relation is LinkRelationRule.REQUEST_WHEN_WINDOWED and has_window))
    needs_boundary = rule.relation is LinkRelationRule.BOUNDARY
    if needs_request and type(relation) is not RequestTransferRelation:
        raise ValueError(f"{path.value} requires a request relation")
    if needs_boundary and type(relation) is not BoundaryTransferRelation:
        raise ValueError(f"{path.value} requires a boundary relation")
    if not needs_request and not needs_boundary and relation is not None:
        raise ValueError(f"{path.value} does not accept a relation")

    if needs_request:
        request_key = relation.request_key
    elif needs_boundary:
        request_key = relation.source_request_key
    else:
        return
    if (request_key.operation_id != attribution.operation_id
            or request_key.window_id != attribution.window_id):
        raise ValueError("transfer relation does not match attribution")
