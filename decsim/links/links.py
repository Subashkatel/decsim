"""The reaction-path link segments, their channels, and the traffic ledger.

decsim prices TIME on the path from syndrome generation to corrected feedback.
This module owns the transport part of that price: how long each hop takes and
what evidence each hop leaves behind. It holds mechanism only. The number cards
that fill it in live in ``decsim/link_profiles.py``.

THE PATHS. ``LinkPath`` is the closed reaction-path vocabulary. Each member
is one measured segment, named for the pair of runtime components it connects:

- ``QC``  QPU -> controller: syndrome readout leaving the QPU (t_qc).
- ``C2B`` controller -> syndrome buffer 0: a completed binary round published
  to the window-input route; a profile without this edge publishes for free.
- ``CWD`` controller -> weak decoder: syndrome data reaching the weak tier,
  either as one round (``syndrome_ingress``) or as one weak window
  (``window_manager``) (t_cwd).
- ``WSD`` weak decoder -> strong decoder: the escalation selection that hands a
  window to the strong tier (t_wsd).
- ``CSD`` controller -> strong decoder: the strong window's syndrome input
  (t_csd).
- ``WDO`` weak decoder -> orchestrator: the weak correction leaving the weak
  tier. This is the weak-tier counterpart of ``DO``; the research program in
  ORIENTATION.md names eight t segments and does not name this one separately.
- ``DD``  decoder -> decoder: a committed window boundary handed to a dependent
  window (t_dd).
- ``DO``  strong decoder -> orchestrator: the strong correction leaving the
  strong tier (t_do).
- ``OC``  orchestrator -> controller: the resolved decision returning to the
  controller (t_oc).
- ``CQ``  controller -> QPU: the instruction delivered back to the QPU (t_cq).

The vocabulary is CLOSED and each member's semantics are declared once, in
``_PATH_RULES``: the attribution scope its transfers must use, the provenance
relation they must carry, and the decoder tier they belong to. It is the
measurement decomposition, not a free-form axis, so a run cannot invent a
segment and change what a latency report means.

Closed does not mean frozen. The original nine segments are REQUIRED: their
rules say so and a card that omits one is refused. ``C2B`` is OPTIONAL so
existing cards remain inert until
they opt into the priced controller-to-buffer hop.
The wired set of a card is therefore always a declared subset of the closed
vocabulary, never an arbitrary set of names a run made up.

Adding a new measured segment is a three-line extension and no surgery:

1. add the member to ``LinkPath``;
2. add its row to ``_PATH_RULES``, marked optional while it is being adopted;
3. add an ``Optional[LinkEdgeConfig]`` field for it to ``LinkModelConfig``,
   defaulting to ``None`` so every existing fabric card keeps working.

Then any card that wants to price it supplies an edge. Nothing in ``resolve``,
``reserve``, the counters or the reports changes, because they all iterate the
card's wired paths rather than the enum. Reserving on a segment this fabric did
not wire is refused at the top of ``reserve``, before any attribution work.

THE CONFIGURATION. Each path carries a ``LinkEdgeConfig``: the physical
``LinkConfig`` channel it rides (propagation latency and an optional finite
``LinkCapacityConfig`` bandwidth), an optional ``PayloadSizeConfig`` default
payload, and the name of the runtime quantity that supplies an actual payload
size. ``LinkModelConfig`` is the whole fabric card: one edge per path plus a
profile name. Two paths that are given the SAME ``LinkConfig`` object share one
physical FIFO, so the number of physical channels behind the wired paths is
itself a configuration choice.

QUANTITY RULES. Bandwidth is a real rate in bits per microsecond and only has to
be finite and positive. Everything that counts something - bits, channels,
ticks, window and round indices, request ordinals, boundary revisions - is
integral by meaning: a value is accepted when it is finite and equal to its own
integer conversion, is refused otherwise, and is normalised to an exact ``int``
at the boundary where it becomes trusted. No quantity check asserts a Python
type. Two type-discriminating predicates DO remain, both semantic rather than
pedantic: the stable-identity rule imported from ``message``, which defines what
may serve as an identity in the ledger, and the closed two-kind relation
dispatch, which after the snapshot ranges only over values this module built.

THE RUNTIME. ``LinkModelConfig.resolve()`` builds a run-owned ``LinkModel``:
one ``Link`` per distinct channel object, each owning that channel's mutable
FIFO state. ``LinkModel.reserve`` is the only semantic boundary production sends
use. It snapshots the caller's attribution into links-owned immutable values,
checks the attribution against the path's meaning, selects the payload size,
reserves the physical interval, and appends one immutable transfer record.

THE EVIDENCE. Every reservation is counted twice, once per semantic path and
once per physical channel, and ``traffic_json_value`` refuses to emit a report
whose two counts disagree.

CUSTOMISING. An outside scientist builds their own ``LinkModelConfig`` (or
starts from a card in ``decsim/link_profiles.py``) and passes it as
``RunSpec.links``. No core module is edited to change any latency, bandwidth,
payload, channel sharing, or profile name.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Optional, Union

from ..config import us
from ..message import DecoderRequestKey, DecoderTier


def _whole(value, name: str) -> int:
    """Return one semantically integral quantity as an exact int."""
    try:
        normalized = int(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite whole number") from error
    if normalized != value:
        raise ValueError(f"{name} must be a finite whole number")
    return normalized

def _finite(value, name: str):
    """Return one real quantity after refusing NaN and infinity.

    A Python integer is finite at any magnitude; ``math.isfinite`` only raises
    ``OverflowError`` because it converts to float first, so that outcome is
    read as finite rather than as a failure.
    """
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = True
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    return value


@dataclass(frozen=True)
class RequestTransferRelation:
    """Provenance tying one transfer to the decoder request it serves."""

    request_key: DecoderRequestKey


@dataclass(frozen=True)
class BoundaryTransferRelation:
    source_request_key: DecoderRequestKey
    source_window_key: tuple
    destination_window_key: tuple
    source_revision: int
    delivery_revision: int


class LinkQuantityBasis(str, Enum):
    """Whether one configured quantity is aggregate or per active channel."""

    DIRECT_AGGREGATE = "direct_aggregate"
    PER_CHANNEL = "per_channel"


def _require_known_basis(basis) -> None:
    """Reject a basis outside the closed pair that decides the aggregate rule."""
    if (basis is not LinkQuantityBasis.DIRECT_AGGREGATE
            and basis is not LinkQuantityBasis.PER_CHANNEL):
        raise ValueError(f"unknown link quantity basis {basis!r}")


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


class PayloadSelectionSource(str, Enum):
    """How a transfer selected its aggregate payload size."""

    ACTUAL = "actual"
    CONFIGURED_DEFAULT = "configured_default"
    UNRESOLVED = "unresolved"

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
    """The fixed semantics of one reaction-path segment.

    ``required`` says whether every fabric card must wire this segment. All nine
    segments of the original reaction path are required, so no card can quietly
    stop pricing one. A segment added later may be declared optional, which is
    what makes adding one a config-or-small-extension step; optionality is a
    property of the declared vocabulary, never of the caller's configuration.
    """

    scope: LinkAttributionScope
    scope_description: str
    relation: LinkRelationRule
    tier: Optional[DecoderTier]
    required: bool


_PATH_RULES = MappingProxyType({
    LinkPath.QC: LinkPathRule(
        LinkAttributionScope.ROUND,
        "syndrome-round attribution without a window",
        LinkRelationRule.NONE, None, True),
    LinkPath.C2B: LinkPathRule(
        LinkAttributionScope.ROUND,
        "syndrome-round attribution without a window",
        LinkRelationRule.NONE, None, False),
    LinkPath.CWD: LinkPathRule(
        LinkAttributionScope.ROUND_OR_WINDOW,
        "syndrome-round or window-region attribution",
        LinkRelationRule.REQUEST_WHEN_WINDOWED, DecoderTier.WEAK, True),
    LinkPath.WSD: LinkPathRule(
        LinkAttributionScope.WINDOW, "window-region attribution",
        LinkRelationRule.REQUEST, DecoderTier.STRONG, True),
    LinkPath.CSD: LinkPathRule(
        LinkAttributionScope.WINDOW, "window-region attribution",
        LinkRelationRule.REQUEST, DecoderTier.STRONG, True),
    LinkPath.WDO: LinkPathRule(
        LinkAttributionScope.WINDOW, "window-region attribution",
        LinkRelationRule.REQUEST, DecoderTier.WEAK, True),
    LinkPath.DD: LinkPathRule(
        LinkAttributionScope.WINDOW, "window-region attribution",
        LinkRelationRule.BOUNDARY, None, True),
    LinkPath.DO: LinkPathRule(
        LinkAttributionScope.WINDOW, "window-region attribution",
        LinkRelationRule.REQUEST, DecoderTier.STRONG, True),
    LinkPath.OC: LinkPathRule(
        LinkAttributionScope.OPERATION_ONLY, "operation-only attribution",
        LinkRelationRule.NONE, None, True),
    LinkPath.CQ: LinkPathRule(
        LinkAttributionScope.OPERATION_ONLY, "operation-only attribution",
        LinkRelationRule.NONE, None, True),
})


@dataclass(frozen=True, eq=False)
class LinkCapacityConfig:
    """Reusable capacity input retaining its source basis and raw operands."""

    input_bits_per_us: float
    basis: LinkQuantityBasis
    channel_count: Optional[int]
    source: str

    def __post_init__(self) -> None:
        _finite(self.input_bits_per_us, "input_bits_per_us")
        if self.input_bits_per_us <= 0:
            raise ValueError("input_bits_per_us must be positive")
        _require_known_basis(self.basis)
        self._validate_channel_count()
        _finite(self.aggregate_bits_per_us, "aggregate_bits_per_us")

    def _validate_channel_count(self) -> None:
        if self.basis is LinkQuantityBasis.DIRECT_AGGREGATE:
            if self.channel_count is not None:
                raise ValueError(
                    "direct aggregate capacity requires channel_count=None"
                )
            return
        object.__setattr__(
            self, "channel_count",
            _whole(self.channel_count, "per-channel capacity count"))
        if self.channel_count <= 0:
            raise ValueError("per-channel capacity count must be positive")

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
    """Reusable payload input retaining raw, basis, count, and source."""

    input_bits: int
    basis: LinkQuantityBasis
    channel_count: Optional[int]
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "input_bits", _whole(self.input_bits, "input_bits"))
        if self.input_bits < 0:
            raise ValueError("input_bits must be nonnegative")
        _require_known_basis(self.basis)
        self._validate_channel_count()

    def _validate_channel_count(self) -> None:
        if self.basis is LinkQuantityBasis.DIRECT_AGGREGATE:
            if self.channel_count is not None:
                raise ValueError(
                    "direct aggregate payload requires channel_count=None"
                )
            return
        object.__setattr__(
            self, "channel_count",
            _whole(self.channel_count, "per-channel payload count"))
        if self.channel_count <= 0:
            raise ValueError("per-channel payload count must be positive")

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
    """Immutable physical propagation and aggregate-FIFO configuration."""

    propagation_latency_ticks: int
    capacity: Optional[LinkCapacityConfig]
    configuration_source: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "propagation_latency_ticks",
            _whole(self.propagation_latency_ticks,
                   "propagation_latency_ticks"))
        if self.propagation_latency_ticks < 0:
            raise ValueError("propagation_latency_ticks must be nonnegative")


@dataclass(frozen=True, eq=False)
class LinkEdgeConfig:
    """One semantic path's payload policy bound to a physical channel."""

    channel: LinkConfig
    default_payload: Optional[PayloadSizeConfig]
    actual_payload_source: Optional[str]

    def __post_init__(self) -> None:
        if self.default_payload is None and self.actual_payload_source is None:
            raise ValueError(
                "an edge requires a configured default or actual payload source"
            )
        capacity = self.channel.capacity
        default = self.default_payload
        if capacity is not None and default is not None:
            if capacity.basis is not default.basis:
                raise ValueError("capacity and payload bases must match")
            if capacity.channel_count != default.channel_count:
                raise ValueError("capacity and payload channel counts must match")


@dataclass(frozen=True)
class TrafficAttribution:
    """Stable operation, patch, window, and inclusive round attribution."""

    operation_id: object
    patch_ids: tuple
    window_id: Optional[int]
    round_lo: Optional[int]
    round_hi: Optional[int]
    relation: Optional[
        Union[RequestTransferRelation, BoundaryTransferRelation]
    ] = None


def _request_key_snapshot(request_key) -> DecoderRequestKey:
    """Return a fresh links-owned copy of one decoder request key."""
    return DecoderRequestKey(
        operation_id=request_key.operation_id,
        window_id=_whole(request_key.window_id, "request window_id"),
        tier=request_key.tier,
        run_sequence=_whole(request_key.run_sequence, "request run_sequence"),
    )


def _relation_snapshot(relation):
    """Return a fresh links-owned copy of one transfer relation."""
    if relation is None:
        return None
    if isinstance(relation, RequestTransferRelation):
        return RequestTransferRelation(
            _request_key_snapshot(relation.request_key)
        )
    if isinstance(relation, BoundaryTransferRelation):
        return BoundaryTransferRelation(
            _request_key_snapshot(relation.source_request_key),
            tuple(relation.source_window_key),
            tuple(relation.destination_window_key),
            relation.source_revision,
            relation.delivery_revision,
        )
    raise TypeError(
        "a transfer relation is a request relation or a boundary relation"
    )


def _attribution_snapshot(attribution) -> TrafficAttribution:
    """Return the fresh links-owned attribution the ledger actually stores."""
    return TrafficAttribution(
        operation_id=attribution.operation_id,
        patch_ids=tuple(attribution.patch_ids),
        window_id=attribution.window_id,
        round_lo=attribution.round_lo,
        round_hi=attribution.round_hi,
        relation=_relation_snapshot(attribution.relation),
    )


@dataclass(frozen=True)
class LinkReservation:
    """One immutable physical FIFO timing decision."""

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
class TrafficCounters:
    """Exact additive counters shared by semantic and physical snapshots."""

    transfer_count: int = 0
    known_payload_bits: int = 0
    unknown_payload_transfer_count: int = 0
    serialization_ticks: int = 0
    propagation_ticks: int = 0
    queue_wait_ticks: int = 0

    def plus_reservation(self, reservation: LinkReservation) -> "TrafficCounters":
        return TrafficCounters(
            transfer_count=self.transfer_count + 1,
            known_payload_bits=(
                self.known_payload_bits
                + (reservation.payload_bits if reservation.payload_bits is not None else 0)
            ),
            unknown_payload_transfer_count=(
                self.unknown_payload_transfer_count
                + (reservation.payload_bits is None)
            ),
            serialization_ticks=(
                self.serialization_ticks + reservation.serialization_ticks
            ),
            propagation_ticks=(
                self.propagation_ticks + reservation.propagation_ticks
            ),
            queue_wait_ticks=(
                self.queue_wait_ticks + reservation.queue_wait_ticks
            ),
        )

    def plus(self, other: "TrafficCounters") -> "TrafficCounters":
        return TrafficCounters(
            transfer_count=self.transfer_count + other.transfer_count,
            known_payload_bits=self.known_payload_bits + other.known_payload_bits,
            unknown_payload_transfer_count=(
                self.unknown_payload_transfer_count
                + other.unknown_payload_transfer_count
            ),
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


class Link:
    """One resolved aggregate FIFO physical channel."""

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

    def reserve(
        self,
        *,
        payload_bits: Optional[int],
        now_ticks: int,
    ) -> LinkReservation:
        """Reserve one FIFO interval and return its exact timing."""
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

        serialization_ticks = (
            0
            if capacity is None
            else us(payload_bits / capacity.aggregate_bits_per_us)
        )
        serializer_start_ticks = (
            now_ticks
            if capacity is None
            else max(now_ticks, self._next_free_tick)
        )
        serializer_end_ticks = serializer_start_ticks + serialization_ticks
        queue_wait_ticks = serializer_start_ticks - now_ticks
        total_delay_ticks = (
            queue_wait_ticks
            + serialization_ticks
            + self._config.propagation_latency_ticks
        )
        reservation = LinkReservation(
            payload_bits=payload_bits,
            send_ticks=now_ticks,
            queue_wait_ticks=queue_wait_ticks,
            serialization_ticks=serialization_ticks,
            propagation_ticks=self._config.propagation_latency_ticks,
            serializer_start_ticks=serializer_start_ticks,
            serializer_end_ticks=serializer_end_ticks,
            total_delay_ticks=total_delay_ticks,
            physical_sequence=self._physical_sequence,
        )
        if capacity is not None:
            self._next_free_tick = serializer_end_ticks
        self._last_send_tick = now_ticks
        self._physical_sequence += 1
        self._counters = self._counters.plus_reservation(reservation)
        return reservation


@dataclass(frozen=True)
class SemanticTransferRecord:
    path: LinkPath
    physical_alias: str
    attribution: TrafficAttribution
    payload_selection: PayloadSelectionSource
    payload_source: str
    reservation: LinkReservation


@dataclass(frozen=True)
class LinkModelConfig:
    """Reusable semantic fabric configuration."""

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
        """Return the paths this fabric card wires, in vocabulary order.

        A required path must be wired. Only a path the vocabulary declares
        optional may be left out, so the wired set is always a declared subset,
        never an arbitrary one.
        """
        wired = []
        for path in LinkPath:
            if getattr(self, path.value, None) is not None:
                wired.append(path)
            elif _PATH_RULES[path].required:
                raise ValueError(f"{path.value} is a required link path")
        return tuple(wired)

    def resolve(self) -> "LinkModel":
        physical_by_config_id = {}
        bindings = {}
        for path in self.wired_paths():
            edge = getattr(self, path.value)
            config_id = id(edge.channel)
            physical = physical_by_config_id.get(config_id)
            if physical is None:
                physical = Link(edge.channel)
                physical_by_config_id[config_id] = physical
            bindings[path] = (edge, physical)
        return LinkModel(self, bindings)


class LinkModel:
    """One run-owned semantic fabric and its immutable traffic ledger."""

    def __init__(self, config: LinkModelConfig, bindings: dict):
        self._config = config
        self._bindings = dict(bindings)
        self._paths = tuple(self._bindings)
        self._semantic_counters = {
            path: TrafficCounters()
            for path in self._paths
        }
        self._transfers = []
        self._alias_by_link = {}
        for path in self._paths:
            _edge, physical = self._bindings[path]
            if physical not in self._alias_by_link:
                self._alias_by_link[physical] = (
                    f"channel-{len(self._alias_by_link)}"
                )

    @property
    def paths(self) -> tuple:
        """Return the semantic paths this fabric wires, in vocabulary order."""
        return self._paths

    def reserve(
        self,
        path: LinkPath,
        *,
        payload_bits: Optional[int],
        now_ticks: int,
        attribution: TrafficAttribution,
    ) -> LinkReservation:
        if path not in self._bindings:
            raise ValueError(f"{path.value} is not wired in this link fabric")
        attribution = _attribution_snapshot(attribution)
        self._validate_attribution_shape(path, attribution)
        edge, physical = self._bindings[path]
        if payload_bits is not None and edge.actual_payload_source is None:
            raise ValueError(
                f"{path.value} does not declare an actual payload source"
            )
        if payload_bits is not None:
            selected_bits = payload_bits
            selection = PayloadSelectionSource.ACTUAL
            payload_source = edge.actual_payload_source
        elif edge.default_payload is not None:
            selected_bits = edge.default_payload.aggregate_bits
            selection = PayloadSelectionSource.CONFIGURED_DEFAULT
            payload_source = edge.default_payload.source
        else:
            selected_bits = None
            selection = PayloadSelectionSource.UNRESOLVED
            payload_source = edge.actual_payload_source

        reservation = physical.reserve(
            payload_bits=selected_bits,
            now_ticks=now_ticks,
        )
        self._semantic_counters[path] = (
            self._semantic_counters[path].plus_reservation(reservation)
        )
        self._transfers.append(SemanticTransferRecord(
            path=path,
            physical_alias=self._alias_by_link[physical],
            attribution=attribution,
            payload_selection=selection,
            payload_source=payload_source,
            reservation=reservation,
        ))
        return reservation

    @staticmethod
    def _validate_attribution_shape(
        path: LinkPath,
        attribution: TrafficAttribution,
    ) -> None:
        rule = _PATH_RULES[path]
        has_window = attribution.window_id is not None
        has_rounds = attribution.round_lo is not None
        if rule.scope is LinkAttributionScope.ROUND:
            valid = not has_window and has_rounds
        elif rule.scope is LinkAttributionScope.ROUND_OR_WINDOW:
            valid = has_rounds
        elif rule.scope is LinkAttributionScope.WINDOW:
            valid = has_window and has_rounds
        else:
            valid = not has_window and not has_rounds
        if not valid:
            raise ValueError(f"{path.value} requires {rule.scope_description}")
        relation = attribution.relation
        needs_request = (
            rule.relation is LinkRelationRule.REQUEST
            or (rule.relation is LinkRelationRule.REQUEST_WHEN_WINDOWED
                and has_window)
        )
        needs_boundary = rule.relation is LinkRelationRule.BOUNDARY
        if needs_request and type(relation) is not RequestTransferRelation:
            raise ValueError(f"{path.value} requires a request relation")
        if needs_boundary and type(relation) is not BoundaryTransferRelation:
            raise ValueError(f"{path.value} requires a boundary relation")
        if not needs_request and not needs_boundary and relation is not None:
            raise ValueError(f"{path.value} does not accept a relation")
        request_key = (relation.request_key if type(relation) is RequestTransferRelation
                       else relation.source_request_key
                       if type(relation) is BoundaryTransferRelation else None)
        if request_key is not None:
            if request_key.window_id < 0:
                raise ValueError(
                    "request window_id must be a nonnegative window index"
                )
            if request_key.run_sequence < 0:
                raise ValueError(
                    "request run_sequence must be a nonnegative request ordinal"
                )
        if needs_request and request_key.tier is not rule.tier:
            raise ValueError(f"{path.value} requires the {rule.tier.value} tier")
        if request_key is not None and (
                request_key.operation_id != attribution.operation_id
                or request_key.window_id != attribution.window_id):
            raise ValueError("transfer relation does not match attribution")

    def _member_paths(self, physical: Link) -> tuple:
        return tuple(
            path
            for path in self._paths
            if self._bindings[path][1] is physical
        )

    def snapshot(self) -> "LinkFabricSnapshot":
        """Frozen view for reports: the wiring, every channel's counters and
        the whole transfer ledger."""
        channels = tuple(
            LinkChannelSnapshot(
                alias=alias,
                member_paths=self._member_paths(physical),
                config=physical.config,
                counters=physical.counters_snapshot())
            for physical, alias in self._alias_by_link.items())
        edges = tuple(
            LinkEdgeSnapshot(
                path=path, physical_alias=self._alias_by_link[self._bindings[path][1]],
                edge=self._bindings[path][0], counters=self._semantic_counters[path])
            for path in self._paths)
        return LinkFabricSnapshot(
            profile_name=self._config.profile_name, paths=self._paths,
            edges=edges, channels=channels, transfers=tuple(self._transfers))


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
