"""Typed reaction-path link configuration, reservation, and traffic evidence.

Immutable configuration is reusable across runs. ``Link`` owns the mutable
FIFO state of one resolved physical channel, and ``LinkModel`` is the only
semantic routing/accounting boundary used by production sends.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional

from .config import us
from .message import (
    is_stable_identity,
    is_stable_string,
    stable_identity_order_key,
)


def _require_nonempty_stable_string(value, field_name: str) -> None:
    if not is_stable_string(value):
        raise TypeError(f"{field_name} must be an exact Unicode-scalar string")
    if not value:
        raise ValueError(f"{field_name} must be nonempty")


class LinkQuantityBasis(str, Enum):
    """Whether one configured quantity is aggregate or per active channel."""

    DIRECT_AGGREGATE = "direct_aggregate"
    PER_CHANNEL = "per_channel"


class LinkPath(str, Enum):
    """Closed semantic reaction-path vocabulary."""

    QC = "qc"
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


@dataclass(frozen=True, eq=False)
class LinkCapacityConfig:
    """Reusable capacity input retaining its source basis and raw operands."""

    input_bits_per_us: float
    basis: LinkQuantityBasis
    channel_count: Optional[int]
    source: str

    def __post_init__(self) -> None:
        if type(self.input_bits_per_us) is not float:
            raise TypeError("input_bits_per_us must be an exact built-in float")
        if not math.isfinite(self.input_bits_per_us):
            raise ValueError("input_bits_per_us must be finite")
        if self.input_bits_per_us <= 0:
            raise ValueError("input_bits_per_us must be positive")
        if type(self.basis) is not LinkQuantityBasis:
            raise TypeError("basis must be an exact LinkQuantityBasis")
        self._validate_channel_count()
        _require_nonempty_stable_string(self.source, "capacity source")
        if not math.isfinite(self.aggregate_bits_per_us):
            raise ValueError("aggregate_bits_per_us must be finite")

    def _validate_channel_count(self) -> None:
        if self.basis is LinkQuantityBasis.DIRECT_AGGREGATE:
            if self.channel_count is not None:
                raise ValueError(
                    "direct aggregate capacity requires channel_count=None"
                )
            return
        if type(self.channel_count) is not int:
            raise TypeError("per-channel capacity count must be an exact int")
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
        if type(self.input_bits) is not int:
            raise TypeError("input_bits must be an exact built-in int")
        if self.input_bits < 0:
            raise ValueError("input_bits must be nonnegative")
        if type(self.basis) is not LinkQuantityBasis:
            raise TypeError("basis must be an exact LinkQuantityBasis")
        self._validate_channel_count()
        _require_nonempty_stable_string(self.source, "payload source")

    def _validate_channel_count(self) -> None:
        if self.basis is LinkQuantityBasis.DIRECT_AGGREGATE:
            if self.channel_count is not None:
                raise ValueError(
                    "direct aggregate payload requires channel_count=None"
                )
            return
        if type(self.channel_count) is not int:
            raise TypeError("per-channel payload count must be an exact int")
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
        if type(self.propagation_latency_ticks) is not int:
            raise TypeError("propagation_latency_ticks must be an exact int")
        if self.propagation_latency_ticks < 0:
            raise ValueError("propagation_latency_ticks must be nonnegative")
        if self.capacity is not None and type(self.capacity) is not LinkCapacityConfig:
            raise TypeError("capacity must be an exact LinkCapacityConfig or None")
        _require_nonempty_stable_string(
            self.configuration_source,
            "configuration_source",
        )


@dataclass(frozen=True, eq=False)
class LinkEdgeConfig:
    """One semantic path's payload policy bound to a physical channel."""

    channel: LinkConfig
    default_payload: Optional[PayloadSizeConfig]
    actual_payload_source: Optional[str]

    def __post_init__(self) -> None:
        if type(self.channel) is not LinkConfig:
            raise TypeError("channel must be an exact LinkConfig")
        if (
            self.default_payload is not None
            and type(self.default_payload) is not PayloadSizeConfig
        ):
            raise TypeError(
                "default_payload must be an exact PayloadSizeConfig or None"
            )
        if self.actual_payload_source is not None:
            _require_nonempty_stable_string(
                self.actual_payload_source,
                "actual_payload_source",
            )
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

    @classmethod
    def from_per_channel_transaction(
        cls,
        *,
        propagation_latency_ticks: int,
        per_channel_capacity_bits_per_us: float,
        per_channel_payload_bits: int,
        channel_count: int,
        capacity_source: str,
        payload_source: str,
        configuration_source: str,
        actual_payload_source: Optional[str] = None,
    ) -> "LinkEdgeConfig":
        capacity = LinkCapacityConfig(
            input_bits_per_us=per_channel_capacity_bits_per_us,
            basis=LinkQuantityBasis.PER_CHANNEL,
            channel_count=channel_count,
            source=capacity_source,
        )
        payload = PayloadSizeConfig(
            input_bits=per_channel_payload_bits,
            basis=LinkQuantityBasis.PER_CHANNEL,
            channel_count=channel_count,
            source=payload_source,
        )
        return cls(
            channel=LinkConfig(
                propagation_latency_ticks=propagation_latency_ticks,
                capacity=capacity,
                configuration_source=configuration_source,
            ),
            default_payload=payload,
            actual_payload_source=actual_payload_source,
        )


@dataclass(frozen=True)
class TrafficAttribution:
    """Stable operation, patch, window, and inclusive round attribution."""

    operation_id: object
    patch_ids: tuple
    window_id: Optional[int]
    round_lo: Optional[int]
    round_hi: Optional[int]

    def __post_init__(self) -> None:
        if not is_stable_identity(self.operation_id):
            raise TypeError("operation_id must be a stable identity")
        if type(self.patch_ids) is not tuple:
            raise TypeError("patch_ids must be a tuple")
        if not all(is_stable_identity(patch_id) for patch_id in self.patch_ids):
            raise TypeError("patch_ids must contain stable identities")
        ordered = tuple(sorted(self.patch_ids, key=stable_identity_order_key))
        if tuple(map(stable_identity_order_key, self.patch_ids)) != tuple(
            map(stable_identity_order_key, ordered)
        ):
            raise ValueError("patch_ids must use stable structural order")
        if (self.round_lo is None) != (self.round_hi is None):
            raise ValueError("round endpoints are present together")
        if self.window_id is not None:
            if type(self.window_id) is not int or self.window_id < 0:
                raise ValueError("window_id must be a nonnegative exact int")
            if self.round_lo is None:
                raise ValueError("window attribution requires a round range")
        if self.round_lo is None:
            return
        if type(self.round_lo) is not int or self.round_lo < 1:
            raise ValueError("round_lo must be a positive exact int")
        if type(self.round_hi) is not int or self.round_hi < self.round_lo:
            raise ValueError("round_hi must be an exact int at least round_lo")


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
        if type(config) is not LinkConfig:
            raise TypeError("Link requires an exact LinkConfig")
        self._config = config
        self._next_free_tick = 0
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
        """Validate, reserve one FIFO interval, and return its exact timing."""
        if payload_bits is not None:
            if type(payload_bits) is not int:
                raise TypeError("payload_bits must be an exact int or None")
            if payload_bits < 0:
                raise ValueError("payload_bits must be nonnegative")
        if type(now_ticks) is not int:
            raise TypeError("now_ticks must be an exact int")
        if now_ticks < 0:
            raise ValueError("now_ticks must be nonnegative")
        capacity = self._config.capacity
        if capacity is not None and payload_bits is None:
            raise ValueError("finite bandwidth requires a resolved payload size")

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
        self._physical_sequence += 1
        self._counters = self._counters.plus_reservation(reservation)
        return reservation


@dataclass(frozen=True)
class SemanticTransferRecord:
    path: LinkPath
    physical_alias: str
    attribution: TrafficAttribution
    payload_bits: Optional[int]
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

    def __post_init__(self) -> None:
        for path in LinkPath:
            if type(getattr(self, path.value)) is not LinkEdgeConfig:
                raise TypeError(f"{path.value} must be an exact LinkEdgeConfig")
        _require_nonempty_stable_string(self.profile_name, "profile_name")

    def resolve(self) -> "LinkModel":
        physical_by_config_id = {}
        bindings = {}
        for path in LinkPath:
            edge = getattr(self, path.value)
            config_id = id(edge.channel)
            physical = physical_by_config_id.get(config_id)
            if physical is None:
                physical = Link(edge.channel)
                physical_by_config_id[config_id] = physical
            bindings[path] = (edge, physical)
        return LinkModel(self, bindings)

    @classmethod
    def reference_fixed_latency_profile(cls) -> "LinkModelConfig":
        def channel(latency_us: float, source: str) -> LinkConfig:
            return LinkConfig(us(latency_us), None, source)

        def actual_edge(latency_us: float, source: str, actual: str):
            return LinkEdgeConfig(channel(latency_us, source), None, actual)

        def direct_default(latency_us: float, bits: int, source: str):
            return LinkEdgeConfig(
                channel(latency_us, source),
                PayloadSizeConfig(
                    bits,
                    LinkQuantityBasis.DIRECT_AGGREGATE,
                    None,
                    source,
                ),
                None,
            )

        def per_channel_default(
            latency_us: float,
            bits: int,
            count: int,
            source: str,
        ):
            return LinkEdgeConfig(
                channel(latency_us, source),
                PayloadSizeConfig(
                    bits,
                    LinkQuantityBasis.PER_CHANNEL,
                    count,
                    source,
                ),
                None,
            )

        return cls(
            qc=actual_edge(0.15, "Khalid qc effective time", "SyndromePayload.size_bits"),
            cwd=actual_edge(
                2.0,
                "Khalid cd mapped to controller-to-weak",
                "SyndromeRoundPacket.fragment_size_sum",
            ),
            wsd=actual_edge(
                0.5,
                "repository weak-to-strong model choice",
                "switching decision payload_bits",
            ),
            csd=actual_edge(
                2.0,
                "Khalid cd mapped to controller-to-strong",
                "DecodeJob.retained_payload_size_bits",
            ),
            wdo=per_channel_default(
                1.0,
                50_000,
                100,
                "Khalid do mapped to weak output",
            ),
            dd=direct_default(
                0.5,
                100,
                "Khalid dd representative aggregate transaction",
            ),
            do=per_channel_default(1.0, 50_000, 100, "Khalid do"),
            oc=per_channel_default(4.0, 20_000, 1000, "Khalid oc"),
            cq=per_channel_default(0.15, 1, 5_000_000, "Khalid cq"),
            profile_name="reference_fixed_latency",
        )


class LinkModel:
    """One run-owned semantic fabric and its immutable traffic ledger."""

    def __init__(self, config: LinkModelConfig, bindings: dict):
        if type(config) is not LinkModelConfig:
            raise TypeError("LinkModel requires an exact LinkModelConfig")
        self._config = config
        self._bindings = dict(bindings)
        self._semantic_counters = {
            path: TrafficCounters()
            for path in LinkPath
        }
        self._transfers = []
        self._alias_by_link = {}
        for path in LinkPath:
            _edge, physical = self._bindings[path]
            if physical not in self._alias_by_link:
                self._alias_by_link[physical] = (
                    f"channel-{len(self._alias_by_link)}"
                )

    @property
    def config(self) -> LinkModelConfig:
        return self._config

    def reserve(
        self,
        path: LinkPath,
        *,
        payload_bits: Optional[int],
        now_ticks: int,
        attribution: TrafficAttribution,
    ) -> LinkReservation:
        if type(path) is not LinkPath:
            raise TypeError("path must be an exact LinkPath")
        if type(attribution) is not TrafficAttribution:
            raise TypeError("attribution must be an exact TrafficAttribution")
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
            payload_bits=selected_bits,
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
        has_window = attribution.window_id is not None
        has_rounds = attribution.round_lo is not None
        if path in (LinkPath.QC, LinkPath.CWD):
            valid = not has_window and has_rounds
            expected = "syndrome-round attribution without a window"
        elif path in (
            LinkPath.WSD,
            LinkPath.CSD,
            LinkPath.WDO,
            LinkPath.DD,
            LinkPath.DO,
        ):
            valid = has_window and has_rounds
            expected = "window-region attribution"
        else:
            valid = not has_window and not has_rounds
            expected = "operation-only attribution"
        if not valid:
            raise ValueError(f"{path.value} requires {expected}")

    def _member_paths(self, physical: Link) -> tuple:
        return tuple(
            path
            for path in LinkPath
            if self._bindings[path][1] is physical
        )

    def topology_json_value(
        self,
        *,
        controller_link_integration_assurance: str,
    ) -> dict:
        if controller_link_integration_assurance not in (
            "shipped_controller",
            "custom_controller_unverified",
        ):
            raise ValueError("unknown controller link integration assurance")
        edges = []
        for path in LinkPath:
            edge, physical = self._bindings[path]
            edges.append({
                "path": path.value,
                "physical_alias": self._alias_by_link[physical],
                "actual_payload_source": edge.actual_payload_source,
                "default_payload": (
                    None
                    if edge.default_payload is None
                    else edge.default_payload.to_json_value()
                ),
            })
        physical_channels = []
        for physical, alias in self._alias_by_link.items():
            capacity = physical.config.capacity
            physical_channels.append({
                "physical_alias": alias,
                "member_paths": [path.value for path in self._member_paths(physical)],
                "propagation_latency_ticks": (
                    physical.config.propagation_latency_ticks
                ),
                "capacity": (
                    None if capacity is None else capacity.to_json_value()
                ),
                "configuration_source": physical.config.configuration_source,
                "service_scope": "aggregate_fifo",
            })
        return {
            "schema_version": 1,
            "profile_name": self._config.profile_name,
            "path_order": [path.value for path in LinkPath],
            "edges": edges,
            "physical_channels": physical_channels,
            "cancellation_semantics": "non_preemptive_irrevocable",
            "controller_link_integration_assurance": (
                controller_link_integration_assurance
            ),
        }

    def traffic_json_value(self) -> dict:
        semantic_edges = []
        for path in LinkPath:
            _edge, physical = self._bindings[path]
            semantic_edges.append({
                "path": path.value,
                "physical_alias": self._alias_by_link[physical],
                "counters": self._semantic_counters[path].to_json_value(),
            })
        physical_channels = []
        reconciliation = []
        for physical, alias in self._alias_by_link.items():
            member_paths = self._member_paths(physical)
            physical_counters = physical.counters_snapshot()
            semantic_sum = TrafficCounters()
            for path in member_paths:
                semantic_sum = semantic_sum.plus(self._semantic_counters[path])
            if semantic_sum != physical_counters:
                raise RuntimeError(f"traffic counters do not reconcile for {alias}")
            paths_json = [path.value for path in member_paths]
            physical_channels.append({
                "physical_alias": alias,
                "member_paths": paths_json,
                "counters": physical_counters.to_json_value(),
            })
            reconciliation.append({
                "physical_alias": alias,
                "member_paths": paths_json,
                "semantic_counter_sum": semantic_sum.to_json_value(),
                "physical_counters": physical_counters.to_json_value(),
                "reconciles": True,
            })
        return {
            "schema_version": 1,
            "path_order": [path.value for path in LinkPath],
            "semantic_edges": semantic_edges,
            "physical_channels": physical_channels,
            "transfers": [self._transfer_json(record) for record in self._transfers],
            "reconciliation": reconciliation,
        }

    @staticmethod
    def _identity_json(identity) -> dict:
        if type(identity) is int:
            return {"kind": "integer", "value": str(identity), "items": None}
        if type(identity) is str:
            return {"kind": "string", "value": identity, "items": None}
        return {
            "kind": "tuple",
            "value": None,
            "items": [LinkModel._identity_json(item) for item in identity],
        }

    @classmethod
    def _transfer_json(cls, record: SemanticTransferRecord) -> dict:
        reservation = record.reservation
        attribution = record.attribution
        return {
            "path": record.path.value,
            "physical_alias": record.physical_alias,
            "attribution": {
                "operation_id": cls._identity_json(attribution.operation_id),
                "patch_ids": [
                    cls._identity_json(patch_id)
                    for patch_id in attribution.patch_ids
                ],
                "window_id": attribution.window_id,
                "round_lo": attribution.round_lo,
                "round_hi": attribution.round_hi,
            },
            "payload_bits": record.payload_bits,
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


def link_compression_decision(raw_bits_per_msg: float,
                              packed_bits_per_msg: float,
                              msgs_per_us: float,
                              bandwidth_bits_per_us: float,
                              headroom: float = 0.9,
                              buffer_bound: bool = False) -> dict:
    """The deck's row-22 rule: compress ON THE LINK only when
    BANDWIDTH is the binding constraint (Gate 7 P18).

    util_* = offered bits/us over bandwidth. When the buffer is the
    binding constraint instead, compression belongs in the STORE
    (V23 packed retention), not the wire — the rule returns
    compress_link=False with binding="buffer" so callers route the
    effort to the right place. sufficient=False flags bandwidth-
    binding cases packing alone cannot relieve.
    """
    if bandwidth_bits_per_us <= 0 or msgs_per_us < 0:
        raise ValueError("need bandwidth > 0 and msgs_per_us >= 0")
    util_raw = raw_bits_per_msg * msgs_per_us / bandwidth_bits_per_us
    util_packed = packed_bits_per_msg * msgs_per_us / bandwidth_bits_per_us
    bandwidth_binding = util_raw > headroom
    if bandwidth_binding:
        binding = "bandwidth"
    elif buffer_bound:
        binding = "buffer"
    else:
        binding = "none"
    compress_link = bandwidth_binding and util_packed <= headroom
    sufficient = (not bandwidth_binding) or util_packed <= headroom
    return {"util_raw": util_raw, "util_packed": util_packed,
            "binding": binding, "compress_link": compress_link,
            "sufficient": sufficient}
