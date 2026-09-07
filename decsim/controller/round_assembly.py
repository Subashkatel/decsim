"""The assembler: raw measurement fragments become one packed round.

The controller's packing stage merges the fragments of a round in
fragment order, charges the packing time once per complete round, forms
the detection events with the device's formation table when it has one,
and hands the finished round on (on_packed). Caune et al. 2410.05202
measure 250 to 370 FPGA cycles for packetization, bus transfer, result
return and the conditional together, an upper bound for the packing
time. The stage admits a bounded number of rounds at once
(controller.packing_rounds_in_flight, RoundsInFlight): a round counts
from its first fragment until the windows hear of it, whether it is
still in assembly, held for store room or on its route. A full stage
stops the run: the QPU never pauses and nothing upstream can hold a
fragment.
"""

import dataclasses
import functools
from typing import Callable, Optional

import decsim.controller.settings as controller_settings
import decsim.observe.trace_source as trace_source
import decsim.records.identity as identity_records
import decsim.records.rounds as round_records


class RoundAssembler:
    """Fragments in, one packed round out, after the packing time.

    Trace sources: round_event(RoundEvent) with kinds BINARY_AVAILABLE,
    PACKED and DROPPED; copy_made(round_key, bits, "controller intake",
    "controller assembler") for the merged round (data_path.md hop 2).
    """

    def __init__(
        self,
        engine,
        settings: controller_settings.ControllerSettings,
        *,
        form_round: Optional[Callable],
        on_packed: Callable[[round_records.PackedRound], None],
        rounds_in_flight: "RoundsInFlight",
    ) -> None:
        self.engine = engine
        self.settings = settings
        # the device's formation table, called once per complete round;
        # None for a timing-only or synthetic source
        self.form_round = form_round
        self.workspace = _Workspace(rounds_in_flight, settings.packing_overflow)
        self.on_packed = on_packed
        self.round_event = trace_source.TraceSource()
        self.copy_made = trace_source.TraceSource()

    def add(
        self,
        fragment: round_records.RetainedSyndromeFragment,
        fragment_count: int,
        route: round_records.SyndromePacketRoute,
    ) -> None:
        """Take one fragment; the round is packed when its last one arrives."""
        available = round_records.RoundEvent.of(
            "BINARY_AVAILABLE",
            self.engine.now,
            fragment.operation_id,
            fragment.round_index,
            route,
            fragment.patch_id,
        )
        self.round_event.fire(available)
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
            drop = round_records.RoundEvent.of(
                "DROPPED",
                self.engine.now,
                fragment.operation_id,
                fragment.round_index,
                route,
                fragment.patch_id,
            )
            self.round_event.fire(drop)
        return None

    def _open_context(self, identity, round_key, route, fragment_count):
        context = _PackingContext(identity, round_key, route, fragment_count)
        self.workspace.context_by_identity[identity] = context
        return context

    def _finish_packing(self, context) -> None:
        """The round is complete: merge, form detection events, hand on."""
        operation_id, round_index = context.round_key
        raw_fragments = _merge_fragments_by_patch(context.fragments)
        # the links carry the raw measurement bits; detection events exist
        # only from the decoder input (Buffer 0) onward
        wire_bits = _fragment_bits(raw_fragments)
        # the merge is what packs the round, so it is reported before the
        # round is packed, and the workspace's residence knows its bits
        self.copy_made.fire(
            context.round_key,
            wire_bits,
            "controller intake",
            "controller assembler",
        )
        packed_event = round_records.RoundEvent.of(
            "PACKED", self.engine.now, operation_id, round_index, context.route
        )
        self.round_event.fire(packed_event)
        formed_fragments = self._form_detection_events(raw_fragments)
        packet = round_records.SyndromeRoundPacket(
            operation_id=operation_id,
            round_index=round_index,
            fragments=formed_fragments,
        )
        self.workspace.forget(context)
        packed = round_records.PackedRound(packet, context.route, wire_bits)
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
    route: round_records.SyndromePacketRoute
    fragment_count: int
    fragments: list = dataclasses.field(default_factory=list)


class RoundsInFlight:
    """The rounds in flight through the packing stage, against the bound.

    A round is in flight from its first fragment until the windows hear
    of it: its publication on the window route, its delivery on the
    memory route. Until then it is in one of three places, in assembly
    here, held for store room (HeldRounds) or on its route
    (RoundTransmitter), and each place keeps its own count for its own
    settlement check; the stage's count is their sum, read when a first
    fragment asks for room, so no exit can forget a release. A round the
    strong writer carries alone leaves at its write and a dropped round
    at its drop: neither is held nor sent.
    """

    def __init__(self, capacity: Optional[int], held_rounds, transmitter):
        self.capacity = capacity
        self.held_rounds = held_rounds
        self.transmitter = transmitter

    def has_room(self, in_assembly: int) -> bool:
        """One more round may enter, given how many are in assembly."""
        if self.capacity is None:
            return True
        return self.count(in_assembly) < self.capacity

    def count(self, in_assembly: int) -> int:
        """The rounds in flight now, given how many are in assembly."""
        return in_assembly + self.held_rounds.count + self.transmitter.in_flight

    def downstream_text(self) -> str:
        """The rounds in flight past assembly, for the refusal sentence."""
        return (
            f"held for store room: {self.held_rounds.count}, on their "
            f"route: {self.transmitter.in_flight}"
        )


class _Workspace:
    """The rounds in assembly, admitted against the stage's bound."""

    def __init__(
        self,
        rounds_in_flight: RoundsInFlight,
        overflow: controller_settings.PackingOverflowPolicy,
    ) -> None:
        self.rounds_in_flight = rounds_in_flight
        self.overflow = overflow
        self.context_by_identity: dict = {}
        # rounds refused for want of a context; their later fragments
        # are ignored
        self.dropped_round_keys: set = set()

    def has_room(self) -> bool:
        in_assembly = len(self.context_by_identity)
        return self.rounds_in_flight.has_room(in_assembly)

    def is_dropped(self, round_key) -> bool:
        return round_key in self.dropped_round_keys

    def refuse(self, round_key, tick: int) -> bool:
        """A round that found no context: dropped, or the run stops."""
        drop = controller_settings.PackingOverflowPolicy.DROP_ROUND
        if self.overflow is drop:
            self.dropped_round_keys.add(round_key)
            return True
        capacity = self.rounds_in_flight.capacity
        in_assembly = sorted(self.context_by_identity, key=repr)
        downstream = self.rounds_in_flight.downstream_text()
        raise RuntimeError(
            f"the packing workspace is full at tick {tick}: round "
            f"{round_key!r} arrived while {capacity} rounds were in "
            f"flight through the controller (controller."
            f"packing_rounds_in_flight is {capacity}; in assembly: "
            f"{in_assembly}, {downstream})"
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


def _fragment_index(fragment: round_records.RetainedSyndromeFragment) -> int:
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
        if identity_records.same_stable_identity(fragment.patch_id, patch_id):
            return index
    return None


def _concatenated(
    prior: round_records.RetainedSyndromeFragment,
    fragment: round_records.RetainedSyndromeFragment,
) -> round_records.RetainedSyndromeFragment:
    """The two parts of one patch as one fragment; unknown sizes stay so."""
    bits = None
    if prior.bits is not None and fragment.bits is not None:
        bits = prior.bits + fragment.bits
    size_bits = None
    if prior.size_bits is not None and fragment.size_bits is not None:
        size_bits = prior.size_bits + fragment.size_bits
    return dataclasses.replace(prior, bits=bits, size_bits=size_bits)
