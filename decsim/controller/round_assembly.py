"""The assembler: raw measurement fragments become one packed round.

The controller's packing stage keeps a bounded workspace of rounds in
flight (controller.packing_rounds_in_flight), merges the fragments of a
round in fragment order, charges the packing time once per complete
round, forms the detection events with the device's formation table when
it has one, and hands the finished round on (on_packed). Caune et al.
2410.05202 measure 250 to 370 FPGA cycles for packetization, bus
transfer, result return and the conditional together, an upper bound
for the packing time. A full workspace stops the run: the QPU never
pauses and nothing upstream can hold a fragment.
"""

import dataclasses
import functools
from typing import Callable, Optional

import decsim.controller.settings as controller_settings
import decsim.message as message


class RoundAssembler:
    """Fragments in, one packed round out, after the packing time."""

    def __init__(
        self,
        engine,
        settings: controller_settings.ControllerSettings,
        *,
        form_round: Optional[Callable],
        on_packed: Callable[[message.PackedRound], None],
        recorder,
    ) -> None:
        self.engine = engine
        self.settings = settings
        # the device's formation table, called once per complete round;
        # None for a timing-only or synthetic source
        self.form_round = form_round
        self.workspace = _Workspace(
            settings.packing_rounds_in_flight, settings.packing_overflow
        )
        self.on_packed = on_packed
        self.recorder = recorder

    def add(
        self,
        fragment: message.RetainedSyndromeFragment,
        fragment_count: int,
        route: message.SyndromePacketRoute,
    ) -> None:
        """Take one fragment; the round is packed when its last one arrives."""
        self.recorder.record(
            "BINARY_AVAILABLE",
            fragment.operation_id,
            fragment.round_index,
            route,
            patch_id=fragment.patch_id,
        )
        round_key = (fragment.operation_id, fragment.round_index)
        if self.workspace.is_dropped(round_key):
            return
        context = self._context_for(fragment, fragment_count, route)
        if context is None:
            return
        context.fragments.append(fragment)
        if len(context.fragments) != context.fragment_count:
            return
        packing_ticks = self.settings.packing_ticks()
        if packing_ticks > 0:
            finish = functools.partial(self._finish_packing, context)
            self.engine.schedule(packing_ticks, finish, label="controller pack")
            return
        self._finish_packing(context)

    def check_settled(self) -> None:
        """At the end of a run no round may still be in assembly."""
        partial = self.workspace.partial_identities()
        if partial:
            raise RuntimeError(
                f"run ended with incomplete syndrome packing contexts: "
                f"{partial}"
            )

    def _context_for(self, fragment, fragment_count, route):
        """The live context of the fragment's round, opened on its first.

        None when the round was dropped for want of a context.
        """
        round_key = (fragment.operation_id, fragment.round_index)
        identity = (
            route.kind.name,
            route.source_operation_id,
            fragment.operation_id,
            fragment.round_index,
        )
        context = self.workspace.context_by_identity.get(identity)
        if context is not None:
            return context
        if self.workspace.has_room():
            return self._open_context(
                identity, round_key, route, fragment_count
            )
        dropped = self.workspace.refuse(round_key, self.engine.now)
        if dropped:
            self.recorder.round_dropped(
                fragment.operation_id,
                fragment.round_index,
                route,
                patch_id=fragment.patch_id,
            )
        return None

    def _open_context(self, identity, round_key, route, fragment_count):
        context = _PackingContext(identity, round_key, route, fragment_count)
        self.workspace.context_by_identity[identity] = context
        return context

    def _finish_packing(self, context) -> None:
        """The round is complete: merge, form detection events, hand on."""
        operation_id, round_index = context.round_key
        self.recorder.record("PACKED", operation_id, round_index, context.route)
        raw_fragments = _merge_fragments_by_patch(context.fragments)
        # the links carry the raw measurement bits; detection events exist
        # only from the decoder input (Buffer 0) onward
        wire_bits = _fragment_bits(raw_fragments)
        formed_fragments = self._form_detection_events(raw_fragments)
        packet = message.SyndromeRoundPacket(
            operation_id=operation_id,
            round_index=round_index,
            fragments=formed_fragments,
        )
        self.workspace.forget(context)
        packed = message.PackedRound(packet, context.route, wire_bits)
        self.on_packed(packed)

    def _form_detection_events(self, raw_fragments) -> tuple:
        """The round's detection events; raw when no table forms them."""
        if self.form_round is None:
            return raw_fragments
        has_bits = all(fragment.bits is not None for fragment in raw_fragments)
        if not has_bits:
            return raw_fragments
        assert len(raw_fragments) == 1, (
            "detector formation expects one merged raw fragment per round"
        )
        (raw,) = raw_fragments
        events = self.form_round(raw.operation_id, raw.round_index, raw.bits)
        formed = tuple(events)
        fragment = dataclasses.replace(raw, bits=formed, size_bits=len(formed))
        return (fragment,)


@dataclasses.dataclass
class _PackingContext:
    """One round in assembly: its route and the fragments so far."""

    identity: tuple
    round_key: tuple
    route: message.SyndromePacketRoute
    fragment_count: int
    fragments: list = dataclasses.field(default_factory=list)


class _Workspace:
    """The rounds in flight through the assembler, bounded by the settings."""

    def __init__(
        self,
        capacity: Optional[int],
        overflow: controller_settings.PackingOverflowPolicy,
    ) -> None:
        self.capacity = capacity
        self.overflow = overflow
        self.context_by_identity: dict = {}
        # rounds refused for want of a context; their later fragments
        # are ignored
        self.dropped_round_keys: set = set()

    def has_room(self) -> bool:
        if self.capacity is None:
            return True
        return len(self.context_by_identity) < self.capacity

    def is_dropped(self, round_key) -> bool:
        return round_key in self.dropped_round_keys

    def refuse(self, round_key, tick: int) -> bool:
        """A round that found no context: dropped, or the run stops."""
        drop = controller_settings.PackingOverflowPolicy.DROP_ROUND
        if self.overflow is drop:
            self.dropped_round_keys.add(round_key)
            return True
        in_flight = sorted(self.context_by_identity, key=repr)
        raise RuntimeError(
            f"the packing workspace is full at tick {tick}: round "
            f"{round_key!r} arrived while {self.capacity} rounds were in "
            f"flight through the controller (controller."
            f"packing_rounds_in_flight is {self.capacity}; in flight: "
            f"{in_flight})"
        )

    def forget(self, context: _PackingContext) -> None:
        self.context_by_identity.pop(context.identity, None)

    def partial_identities(self) -> tuple:
        return tuple(self.context_by_identity)


def _fragment_bits(fragments) -> Optional[int]:
    """The fragments' wire size, None when any fragment has no known size."""
    fragment_sizes = [fragment.size_bits for fragment in fragments]
    if None in fragment_sizes:
        return None
    return sum(fragment_sizes)


def _fragment_index(fragment: message.RetainedSyndromeFragment) -> int:
    return fragment.fragment_index


def _merge_fragments_by_patch(fragments) -> tuple:
    """Order fragments by index, merging parts from the same patch.

    SyndromeRoundPacket requires distinct patch identities, so parts of
    one patch concatenate bits and sizes in fragment-index order. Distinct
    patches keep their own immutable fragments untouched.
    """
    merged: list = []
    for fragment in sorted(fragments, key=_fragment_index):
        prior_index = _index_of_patch(merged, fragment.patch_id)
        if prior_index is None:
            merged.append(fragment)
            continue
        prior = merged[prior_index]
        merged[prior_index] = _concatenated(prior, fragment)
    return tuple(merged)


def _index_of_patch(fragments: list, patch_id) -> Optional[int]:
    for index, fragment in enumerate(fragments):
        if message.same_stable_identity(fragment.patch_id, patch_id):
            return index
    return None


def _concatenated(
    prior: message.RetainedSyndromeFragment,
    fragment: message.RetainedSyndromeFragment,
) -> message.RetainedSyndromeFragment:
    """The two parts of one patch as one fragment; unknown sizes stay so."""
    bits = None
    if prior.bits is not None and fragment.bits is not None:
        bits = prior.bits + fragment.bits
    size_bits = None
    if prior.size_bits is not None and fragment.size_bits is not None:
        size_bits = prior.size_bits + fragment.size_bits
    return dataclasses.replace(prior, bits=bits, size_bits=size_bits)
