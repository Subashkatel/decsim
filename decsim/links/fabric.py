"""The link fabric: every hop of the reaction path wired to its channel.

LinkFabric is the root of the links and the one object the other
components hold; it implements the Link port. send(path, ...) checks the
attribution against the path's rule, selects the payload the path is
priced with, sends it on the path's channel, and at delivery hands the
finished transfer first to the listener (the traffic ledger, when one is
given) and then to the caller's continuation. The rules live in
_RULE_BY_PATH, one row per path. Two paths whose settings name the same
channel share its wire and its setup engine. The shape is gem5's: a
SimObject owns its parameters and its children and is reached through
its ports (src/sim/sim_object.hh, src/mem/port.hh).
"""

import dataclasses
from typing import Callable, Optional, Protocol

import decsim.engine
import decsim.links.channel as channel_module
import decsim.links.settings as link_settings
import decsim.message as message

OnTransfer = Callable[[message.TransferRecord], None]


@dataclasses.dataclass(frozen=True)
class PathRule:
    """The fixed meaning of one path.

    What its transfers are attributed to (a round range, a window, or
    the operation alone), which provenance they carry (the decoder
    request, or the boundary they deliver), and whether every card must
    wire it. A path that carries a request carries it whenever its
    attribution names a window.
    """

    names_rounds: bool
    names_window: bool
    window_is_optional: bool
    carries_request: bool
    carries_boundary: bool
    is_required: bool


class TransferListener(Protocol):
    """What a fabric tells its observer: every finished transfer, once."""

    def on_transfer(self, record: message.TransferRecord) -> None:
        """One transfer was delivered."""


class LinkFabric:
    """One run's fabric: the wired paths on their channels."""

    def __init__(
        self,
        fabric_settings: link_settings.FabricSettings,
        engine: decsim.engine.Engine,
        listener: Optional[TransferListener] = None,
    ):
        self._settings = fabric_settings
        self._listener = listener
        self._channel_by_name: dict[str, channel_module.Channel] = {}
        self._binding_by_path: dict[message.LinkPath, _PathBinding] = {}
        self._send_count = 0
        for path in fabric_settings.wired_paths():
            path_settings = fabric_settings.path_settings(path)
            channel = self._channel_for(path_settings.channel, engine)
            rule = _RULE_BY_PATH[path]
            self._binding_by_path[path] = _PathBinding(
                path_settings, rule, channel
            )

    @property
    def settings(self) -> link_settings.FabricSettings:
        """The card this fabric was built from."""
        return self._settings

    def is_wired(self, path: message.LinkPath) -> bool:
        """Whether the card prices this path; an unwired hop is free."""
        return path in self._binding_by_path

    def expected_delay_ticks(
        self,
        path: message.LinkPath,
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
        path: message.LinkPath,
        payload_bits: Optional[int],
        now_ticks: int,
        attribution: message.TransferAttribution,
        on_delivered: channel_module.OnDelivered,
    ) -> None:
        """Send one transfer on a path; on_delivered(transfer) runs at delivery.

        The attribution is checked against the path's rule and the
        payload is selected before the channel moves.
        """
        binding = self._binding_by_path[path]
        _check_attribution(path, binding.rule, attribution)
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
        self, outgoing: "_Outgoing", transfer: message.Transfer
    ) -> None:
        """At delivery: the ledger sees the transfer, then the caller."""
        if self._listener is not None:
            record = message.TransferRecord(
                request_sequence=outgoing.request_sequence,
                path=outgoing.path,
                channel=outgoing.channel,
                attribution=outgoing.attribution,
                payload_selection=outgoing.payload_selection,
                payload_source=outgoing.payload_source,
                transfer=transfer,
            )
            self._listener.on_transfer(record)
        outgoing.on_delivered(transfer)


@dataclasses.dataclass(frozen=True)
class _PathBinding:
    """One wired path: its settings, its rule and the channel it rides.

    channel_pool is the extension point for the pool component (several
    independent channels between the same two components, a transfer
    taking the first free one); no path names a pool yet.
    """

    settings: link_settings.PathSettings
    rule: PathRule
    channel: channel_module.Channel
    channel_pool: Optional[object] = None


@dataclasses.dataclass(frozen=True)
class _Outgoing:
    """What the ledger's record needs beyond the channel's timing."""

    request_sequence: int
    path: message.LinkPath
    channel: str
    attribution: message.TransferAttribution
    payload_selection: message.PayloadSelection
    payload_source: Optional[str]
    on_delivered: channel_module.OnDelivered


def _round_rule(is_required: bool) -> PathRule:
    return PathRule(
        names_rounds=True,
        names_window=False,
        window_is_optional=False,
        carries_request=False,
        carries_boundary=False,
        is_required=is_required,
    )


def _window_rule(carries_boundary: bool) -> PathRule:
    return PathRule(
        names_rounds=True,
        names_window=True,
        window_is_optional=False,
        carries_request=not carries_boundary,
        carries_boundary=carries_boundary,
        is_required=True,
    )


_OPERATION_RULE = PathRule(
    names_rounds=False,
    names_window=False,
    window_is_optional=False,
    carries_request=False,
    carries_boundary=False,
    is_required=True,
)

# The weak input path carries a window assembled from buffer 0 or one
# feedback-memory round straight off packing; it carries the decoder
# request only in the first case.
_ROUND_OR_WINDOW_RULE = PathRule(
    names_rounds=True,
    names_window=False,
    window_is_optional=True,
    carries_request=True,
    carries_boundary=False,
    is_required=True,
)

_RULE_BY_PATH = {
    message.LinkPath.QPU_TO_CONTROLLER: _round_rule(is_required=True),
    message.LinkPath.CONTROLLER_TO_WEAK_BUFFER: _round_rule(is_required=False),
    message.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER: _ROUND_OR_WINDOW_RULE,
    message.LinkPath.WEAK_DECODER_TO_STRONG_DECODER: _window_rule(
        carries_boundary=False
    ),
    message.LinkPath.STRONG_BUFFER_TO_STRONG_DECODER: _window_rule(
        carries_boundary=False
    ),
    message.LinkPath.WEAK_DECODER_TO_FRAME: _window_rule(
        carries_boundary=False
    ),
    message.LinkPath.DECODER_TO_DECODER: _window_rule(carries_boundary=True),
    message.LinkPath.STRONG_DECODER_TO_FRAME: _window_rule(
        carries_boundary=False
    ),
    message.LinkPath.FRAME_TO_CONTROLLER: _OPERATION_RULE,
    message.LinkPath.CONTROLLER_TO_QPU: _OPERATION_RULE,
    message.LinkPath.CONTROLLER_TO_STRONG_BUFFER: _round_rule(
        is_required=False
    ),
}


def _select_payload(
    path: message.LinkPath,
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
            message.PayloadSelection.ACTUAL,
            path_settings.actual_payload_source,
        )
    default_payload = path_settings.default_payload
    if default_payload is not None:
        return (
            default_payload.aggregate_bits,
            message.PayloadSelection.CONFIGURED_DEFAULT,
            default_payload.source,
        )
    if path_settings.channel.capacity is not None:
        raise RuntimeError(
            f"{path.value} has no payload size and its channel is bounded; "
            f"a bounded wire needs a size to serialize"
        )
    return (
        None,
        message.PayloadSelection.UNRESOLVED,
        path_settings.actual_payload_source,
    )


def _check_attribution(
    path: message.LinkPath,
    rule: PathRule,
    attribution: message.TransferAttribution,
) -> None:
    """Refuse an attribution whose shape is not what the path's rule says.

    The rule fixes what the transfer names (rounds, a window, the
    operation alone) and which relation it carries; the relation must
    name the same operation and window as the attribution.
    """
    has_window = attribution.window_id is not None
    has_rounds = attribution.first_round is not None
    rounds_are_wrong = has_rounds != rule.names_rounds
    window_differs = has_window != rule.names_window
    window_is_wrong = window_differs and not rule.window_is_optional
    if rounds_are_wrong or window_is_wrong:
        scope_words = _scope_words(rule)
        attribution_words = _attribution_words(attribution)
        raise RuntimeError(
            f"{path.value} transfers are attributed to {scope_words}, "
            f"not to {attribution_words}"
        )
    needs_request = rule.carries_request and has_window
    _check_relation(path, attribution, needs_request, rule.carries_boundary)


def _scope_words(rule: PathRule) -> str:
    if rule.names_window:
        return "a window"
    if rule.window_is_optional:
        return "a round range or a window"
    if rule.names_rounds:
        return "a round range"
    return "the operation alone"


def _attribution_words(attribution: message.TransferAttribution) -> str:
    has_window = attribution.window_id is not None
    has_rounds = attribution.first_round is not None
    if has_window and has_rounds:
        return "a window"
    if has_window:
        return "a window without rounds"
    if has_rounds:
        return "a round range"
    return "the operation alone"


def _check_relation(
    path: message.LinkPath,
    attribution: message.TransferAttribution,
    needs_request: bool,
    needs_boundary: bool,
) -> None:
    relation = attribution.relation
    is_request = type(relation) is message.RequestTransferRelation
    is_boundary = type(relation) is message.BoundaryTransferRelation
    if needs_request and not is_request:
        raise RuntimeError(f"{path.value} transfers carry their request")
    if needs_boundary and not is_boundary:
        raise RuntimeError(f"{path.value} transfers carry their boundary")
    accepts_relation = needs_request or needs_boundary
    if not accepts_relation and relation is not None:
        raise RuntimeError(f"{path.value} transfers carry no relation")
    if needs_request:
        request_key = relation.request_key
    elif needs_boundary:
        request_key = relation.source_request_key
    else:
        return
    is_same_operation = request_key.operation_id == attribution.operation_id
    is_same_window = request_key.window_id == attribution.window_id
    if not is_same_operation or not is_same_window:
        raise RuntimeError(
            f"the {path.value} transfer's relation names another operation "
            f"or window than its attribution"
        )
