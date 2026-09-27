"""The assembler: raw measurement fragments become one packed round.

The controller's packing stage merges the fragments of a round in
fragment order, charges the packing time once per complete round, asks
the run's detection event placement for the round that leaves, and hands
it to the writer once that placement's own time is charged
(detector_error_model/detection_event_formation.py, seated by
detection_events.formed_at). Caune et al. 2410.05202 measure
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
from typing import Optional

import decsim.controller.round_transmission as round_transmission
import decsim.controller.settings as controller_settings
import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.identity as identity_records
import decsim.records.rounds as round_records
import decsim.trace_source as trace_source

# the seat the assembler is on the path, as detection_events.formed_at
# names it
_SEAT = "controller"


class RoundsInFlight:
    """The rounds in flight through the packing stage, against the bound.

    A round is in flight from its first fragment until the windows hear
    of it: its publication on the window route, its delivery on the
    memory route, its landing in the strong syndrome buffer on a
    strong-primary run. Until then it is in one of four places, in
    assembly here, held for store room (HeldRounds), on its route
    (RoundTransmitter) or crossing to the strong syndrome buffer
    (SyndromeRoundSender), and each place keeps its own count; the
    stage's count is their sum, read when a first fragment asks for
    room, so no exit can forget a release. A dropped round leaves at
    its drop.
    """

    held_rounds = ports.Port(ports.HeldRounds)
    transmitter = ports.Port(round_transmission.RoundTransmitter)
    syndrome_round_sender = ports.Port(ports.SyndromeRoundSender)

    def __init__(self, capacity: Optional[int]) -> None:
        self.capacity = capacity

    def has_room(self, in_assembly: int) -> bool:
        """One more round may enter, given how many are in assembly."""
        if self.capacity is None:
            return True
        return self.count(in_assembly) < self.capacity

    def count(self, in_assembly: int) -> int:
        """The rounds in flight now, given how many are in assembly."""
        on_route = self._on_route()
        return in_assembly + self.held_rounds.count + on_route

    def downstream_text(self) -> str:
        """The rounds in flight past assembly, for the refusal sentence."""
        on_route = self._on_route()
        return (
            f"held for store room: {self.held_rounds.count}, on their "
            f"route: {on_route}"
        )

    def _on_route(self) -> int:
        """The rounds sent and not yet heard of by the windows."""
        strong_crossing_count = self.syndrome_round_sender.strong_crossing_count
        return self.transmitter.in_flight + strong_crossing_count


class RoundAssembler:
    """Fragments in, one packed round out, after the packing time.

    Trace sources: round_event(RoundEvent) with kinds BINARY_AVAILABLE,
    PACKED and DROPPED; copy_made(round_key, bits, "controller intake",
    "controller assembler") for the merged round (data_path.md hop 2).
    """

    syndrome_round_sender = ports.Port(ports.SyndromeRoundSender)
    detection_events = ports.Port(ports.DetectionEventPlacement)
    rounds_in_flight = ports.Port(RoundsInFlight)

    def __init__(
        self,
        engine: engine_module.Engine,
        settings: controller_settings.ControllerSettings,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.workspace = _Workspace(settings.packing_overflow)
        self.trace = _TraceSources()

    def expect_round(
        self,
        fragment: round_records.RetainedSyndromeFragment,
        route: round_records.SyndromePacketRoute,
    ) -> None:
        """Declare emission order before transport can reorder arrivals.

        This reserves identity only. Packing capacity starts at arrival,
        and a ready round retires after its predecessors, as gem5's
        src/cpu/o3/commit.cc commitInsts retires the ready head per thread.
        """
        self.workspace.expect_round(fragment, route)

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
            fragment.patch_ids,
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
        packing_cycles = self.settings.packing_cycles_per_round
        if packing_cycles > 0:
            now = self.engine.now
            delay = self.settings.clock.ticks_to_edge(packing_cycles, now)
            finish = functools.partial(self._finish_packing, context)
            self.engine.schedule(delay, finish, label="controller pack")
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
        identity = _round_identity(fragment, route)
        context = self.workspace.context_by_identity.get(identity)
        if context is not None:
            return context
        if self.workspace.has_room(self.rounds_in_flight):
            context = _PackingContext(
                identity, round_key, route, fragment_count
            )
            self.workspace.context_by_identity[identity] = context
            return context
        dropped = self.workspace.refuse(
            self.rounds_in_flight, round_key, self.engine.now
        )
        if dropped:
            drop = round_records.RoundEvent.of(
                "DROPPED",
                self.engine.now,
                fragment.operation_id,
                fragment.round_index,
                route,
                fragment.patch_ids,
            )
            self.trace.round_event.fire(drop)
            self.workspace.retire(identity)
            self._release_ready_rounds(identity)
        return None

    def _finish_packing(self, context) -> None:
        context.is_ready = True
        self._release_ready_rounds(context.identity)

    def _release_ready_rounds(self, identity) -> None:
        first = self.workspace.first_for(identity)
        while first is not None and first.is_ready:
            self.workspace.retire(first.identity)
            self._form_round(first)
            first = self.workspace.first_for(identity)

    def _form_round(self, context) -> None:
        """The round is complete: merge, form detection events, hand on."""
        operation_id, round_index = context.round_key
        raw_fragments = _merge_adjacent_fragments(context.fragments)
        # the workspace holds the raw measurement bits the fragments
        # arrived with, whatever width leaves afterwards
        raw_bits = round_records.fragment_wire_bits(raw_fragments)
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
        leaving = self.detection_events.form_at(_SEAT, raw_fragments)
        # the link out of the controller carries what leaves it: the
        # detection events when this seat forms them, the raw outcomes
        # when a seat after it does
        wire_bits = round_records.fragment_wire_bits(leaving)
        packet = round_records.SyndromeRoundPacket(
            operation_id=operation_id,
            round_index=round_index,
            fragments=leaving,
        )
        packed = round_records.PackedRound(packet, context.route, wire_bits)
        self._depart(packed, context)

    def _depart(self, packed: round_records.PackedRound, context) -> None:
        """Hand the round on, after the placement's own time is charged.

        The controller pays for a conversion it does here, one round's
        latency on the former's clock (detection_events); a seat after
        it pays its own, so the round leaves at once.
        """
        formation_cycles = self.detection_events.cycles_at(_SEAT, 1)
        if formation_cycles == 0:
            self._hand_on(packed, context)
            return
        clock = self.detection_events.clock
        now = self.engine.now
        edge = clock.edge(formation_cycles, now)
        delay = edge - now
        hand_on = functools.partial(self._hand_on, packed, context)
        self.engine.schedule(
            delay, hand_on, label="controller form detection events"
        )

    def _hand_on(self, packed, context) -> None:
        self.workspace.forget(context)
        self.syndrome_round_sender.admit(packed)


@dataclasses.dataclass
class _PackingContext:
    """One round in assembly: its route and the fragments so far."""

    identity: tuple
    round_key: tuple
    route: round_records.SyndromePacketRoute
    fragment_count: int
    fragments: list = dataclasses.field(default_factory=list)
    is_ready: bool = False


class _Workspace:
    """The rounds in assembly, admitted against the stage's bound."""

    def __init__(
        self,
        overflow: controller_settings.PackingOverflowPolicy,
    ) -> None:
        self.overflow = overflow
        self.context_by_identity: dict = {}
        self.expected_by_stream: dict[tuple, dict] = {}
        # rounds refused for want of a context; their later fragments
        # are ignored
        self.dropped_round_keys: set = set()

    def expect_round(
        self,
        fragment: round_records.RetainedSyndromeFragment,
        route: round_records.SyndromePacketRoute,
    ) -> None:
        round_key = (fragment.operation_id, fragment.round_index)
        if self.is_dropped(round_key):
            return
        identity = _round_identity(fragment, route)
        stream = identity[:-1]
        pending = self.expected_by_stream.setdefault(stream, {})
        pending.setdefault(identity, None)

    def first_for(self, identity: tuple) -> Optional[_PackingContext]:
        stream = identity[:-1]
        pending = self.expected_by_stream.get(stream, {})
        first_identity = next(iter(pending), None)
        return self.context_by_identity.get(first_identity)

    def retire(self, identity: tuple) -> None:
        stream = identity[:-1]
        pending = self.expected_by_stream[stream]
        del pending[identity]
        if not pending:
            del self.expected_by_stream[stream]

    def has_room(self, rounds_in_flight: RoundsInFlight) -> bool:
        in_assembly = len(self.context_by_identity)
        return rounds_in_flight.has_room(in_assembly)

    def is_dropped(self, round_key) -> bool:
        return round_key in self.dropped_round_keys

    def refuse(
        self, rounds_in_flight: RoundsInFlight, round_key, tick: int
    ) -> bool:
        """A round that found no context: dropped, or the run stops."""
        drop = controller_settings.PackingOverflowPolicy.DROP_ROUND
        if self.overflow is drop:
            self.dropped_round_keys.add(round_key)
            return True
        capacity = rounds_in_flight.capacity
        in_assembly = sorted(self.context_by_identity, key=repr)
        downstream = rounds_in_flight.downstream_text()
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
        outstanding = dict.fromkeys(self.context_by_identity)
        for pending in self.expected_by_stream.values():
            outstanding.update(pending)
        return tuple(outstanding)


def _round_identity(fragment, route) -> tuple:
    return (
        route.kind.name,
        route.source_operation_id,
        fragment.operation_id,
        fragment.round_index,
    )


def _fragment_index(fragment: round_records.RetainedSyndromeFragment) -> int:
    return fragment.fragment_index


def _merge_adjacent_fragments(fragments) -> tuple:
    """Coalesce adjacent acquisitions without permuting measurement records.

    Stim record targets address the declared measurement order. Two parts
    with another acquisition between them cannot be concatenated here.
    """
    merged = []
    ordered = sorted(fragments, key=_fragment_index)
    for fragment in ordered:
        if not merged:
            merged.append(fragment)
            continue
        prior = merged[-1]
        same_patches = identity_records.same_stable_identity(
            prior.patch_ids, fragment.patch_ids
        )
        if same_patches:
            merged[-1] = _concatenated(prior, fragment)
            continue
        merged.append(fragment)
    return tuple(merged)


def _concatenated(
    prior: round_records.RetainedSyndromeFragment,
    fragment: round_records.RetainedSyndromeFragment,
) -> round_records.RetainedSyndromeFragment:
    """The two parts of one patch as one fragment; unknown sizes stay so."""
    bits = None
    if prior.bits is not None and fragment.bits is not None:
        bits = prior.bits + fragment.bits
    size_bits = _joined_width(prior.size_bits, fragment.size_bits)
    event_bits = _joined_width(prior.event_bits, fragment.event_bits)
    return dataclasses.replace(
        prior, bits=bits, size_bits=size_bits, event_bits=event_bits
    )


def _joined_width(first: Optional[int], second: Optional[int]) -> Optional[int]:
    if first is None or second is None:
        return None
    return first + second


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the round assembler reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (gem5 src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    round_event: trace_source.TraceSource = trace_source.new_source()
    copy_made: trace_source.TraceSource = trace_source.new_source()
