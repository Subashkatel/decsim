"""The assembler: raw measurement fragments become one packed round.

The controller's packing stage merges the fragments of a round in
fragment order, charges the packing time once per complete round, asks
the run's detection event placement for the round that leaves, and hands
it on (on_packed) once that placement's own time is charged
(detector_error_model/detection_event_formation.py, named by
controller.detection_events_formed_at). Caune et al. 2410.05202 measure
250 to 370 FPGA cycles for packetization, bus transfer, result return
and the conditional together, an upper bound for the packing time. The stage
admits a bounded number of rounds at once
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
import decsim.ports as ports
import decsim.records.identity as identity_records
import decsim.records.rounds as round_records
import decsim.trace_source as trace_source


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
        detection_events: ports.DetectionEventPlacement,
        on_packed: Callable[[round_records.PackedRound], None],
        rounds_in_flight: "RoundsInFlight",
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.detection_events = detection_events
        self.workspace = _Workspace(rounds_in_flight, settings.packing_overflow)
        self.on_packed = on_packed
        self.trace = _TraceSources()

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
        self.trace.round_event.fire(available)
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
            self.trace.round_event.fire(drop)
        return None

    def _open_context(self, identity, round_key, route, fragment_count):
        context = _PackingContext(identity, round_key, route, fragment_count)
        self.workspace.context_by_identity[identity] = context
        return context

    def _finish_packing(self, context) -> None:
        """The round is complete: merge, form detection events, hand on."""
        operation_id, round_index = context.round_key
        raw_fragments = _merge_fragments_by_patch(context.fragments)
        # the workspace holds the raw measurement bits the fragments
        # arrived with, whatever width leaves afterwards
        raw_bits = _fragment_bits(raw_fragments)
        # the merge is what packs the round, so it is reported before the
        # round is packed, and the workspace's residence knows its bits
        self.trace.copy_made.fire(
            context.round_key,
            raw_bits,
            "controller intake",
            "controller assembler",
        )
        packed_event = round_records.RoundEvent.of(
            "PACKED", self.engine.now, operation_id, round_index, context.route
        )
        self.trace.round_event.fire(packed_event)
        leaving = self.detection_events.form_before_departure(raw_fragments)
        # the link out of the controller carries what leaves it: the
        # detection events when this row forms them here, the raw
        # outcomes when the decoder forms them
        wire_bits = _fragment_bits(leaving)
        packet = round_records.SyndromeRoundPacket(
            operation_id=operation_id,
            round_index=round_index,
            fragments=leaving,
        )
        self.workspace.forget(context)
        packed = round_records.PackedRound(packet, context.route, wire_bits)
        self._depart(packed)

    def _depart(self, packed: round_records.PackedRound) -> None:
        """Hand the round on, after the placement's own time is charged.

        The controller row pays for the conversion it does here
        (controller.detection_event_cycles_per_round); the decoder row
        pays nothing at the controller, so the round leaves at once.
        """
        departure_ticks = self.detection_events.departure_ticks
        if departure_ticks == 0:
            self.on_packed(packed)
            return
        hand_on = functools.partial(self.on_packed, packed)
        self.engine.schedule(
            departure_ticks, hand_on, label="controller form detection events"
        )


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


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the round assembler reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (tmp/resources/gem5/src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    round_event: trace_source.TraceSource = trace_source.new_source()
    copy_made: trace_source.TraceSource = trace_source.new_source()
