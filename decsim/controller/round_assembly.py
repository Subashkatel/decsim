"""The assembler: raw measurement fragments become one packed round.

The packing stage merges a round's fragments, charges the packing time
once per round, and hands the round on after the detection event
placement's own time. Caune et al. 2410.05202 measure 250 to 370 FPGA
cycles for packetization, bus transfer, result return and the
conditional together, an upper bound for the packing time.

The stage admits a bounded number of rounds
(controller.packing_rounds_in_flight), taken in emission order. A round
that finds it full waits whole in front of it while the QPU keeps
measuring, so the wait is the controller's. A refused transfer waits at
its sender and is offered again on the retry (gem5 src/mem/port.hh;
garnet NetworkInterface.cc; Helios pu_arbitration.sv), the same line
the sender keeps in front of the stores (HeldRounds).
"""

import dataclasses
import functools
import operator
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

    A round is in flight from its emission until the windows hear of it: in
    assembly, held for store room, on its route, or crossing to the strong
    syndrome buffer. Each place keeps its own count and the stage reads
    their sum, so no exit can forget a release. A round waiting in front of
    the stage is not in it.
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

    def _on_route(self) -> int:
        """The rounds sent and not yet heard of by the windows."""
        strong_crossing_count = self.syndrome_round_sender.strong_crossing_count
        return self.transmitter.in_flight + strong_crossing_count


class RoundAssembler:
    """Fragments in, one packed round out, after the packing time.

    copy_made fires for the merged round (data_path.md hop 2).
    """

    syndrome_round_sender = ports.Port(ports.SyndromeRoundSender)
    detection_events = ports.Port(ports.DetectionEventPlacement)
    rounds_in_flight = ports.Port(RoundsInFlight)
    # the line a round waits in while the stage is full; the ends a
    # round leaves the stage by retry it
    packing_line = ports.Port(ports.HeldRounds)

    def __init__(
        self,
        engine: engine_module.Engine,
        settings: controller_settings.ControllerSettings,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.workspace = _Workspace()
        self.trace = _TraceSources()

    def expect_round(
        self,
        fragment: round_records.RetainedSyndromeFragment,
        fragment_count: int,
        route: round_records.SyndromePacketRoute,
    ) -> None:
        """The QPU emitted the round: it takes its place, or waits for one.

        Places are taken in emission order, before transport can reorder the
        arrivals, and a ready round retires after its predecessors, as gem5's O3
        rename and commit do (src/cpu/o3/rename.cc:556-584,
        commit.cc:1295-1316). An early round never holds its predecessor's
        place.
        """
        identity = _round_identity(fragment, route)
        if self.workspace.context_of(identity) is not None:
            return
        self.workspace.expect_round(identity)
        round_key = (fragment.operation_id, fragment.round_index)
        context = _PackingContext(identity, round_key, route, fragment_count)
        self.workspace.waiting_by_identity[identity] = context
        if self.packing_line.count == 0 and self._enter(context):
            return
        self.packing_line.refuse(context, self._enter)

    def add(
        self,
        fragment: round_records.RetainedSyndromeFragment,
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
        identity = _round_identity(fragment, route)
        context = self.workspace.context_of(identity)
        context.fragments.append(fragment)
        if not self.workspace.is_in_assembly(context):
            return
        if len(context.fragments) != context.fragment_count:
            return
        self._pack(context)

    def check_settled(self) -> None:
        """At the end of a run no round may still be in assembly."""
        partial = self.workspace.partial_identities()
        if partial:
            raise RuntimeError(
                f"run ended with incomplete syndrome packing contexts: "
                f"{partial}"
            )

    def _enter(self, context) -> bool:
        """A waiting round takes a free place; packs now if complete."""
        if not self.workspace.has_room(self.rounds_in_flight):
            return False
        self.workspace.enter(context)
        if len(context.fragments) == context.fragment_count:
            pack = functools.partial(self._pack, context)
            self.engine.schedule(0, pack, label="controller pack")
        return True

    def _pack(self, context) -> None:
        """Charge the packing time once for the complete round."""
        packing_cycles = self.settings.packing_cycles_per_round
        if packing_cycles > 0:
            now = self.engine.now
            delay = self.settings.clock.ticks_to_edge(packing_cycles, now)
            finish = functools.partial(self._finish_packing, context)
            self.engine.schedule(delay, finish, label="controller pack")
            return
        self._finish_packing(context)

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
        """Hand the round on after the controller's detection event time.

        A seat after the controller pays its own time, so the round leaves at
        once.
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
    """One round being packed from its fragments."""

    identity: tuple
    round_key: tuple
    route: round_records.SyndromePacketRoute
    fragment_count: int
    fragments: list = dataclasses.field(default_factory=list)
    is_ready: bool = False


class _Workspace:
    """The rounds in assembly, admitted against the stage's bound."""

    def __init__(self) -> None:
        self.context_by_identity: dict = {}
        self.expected_by_stream: dict[tuple, dict] = {}
        # rounds in the line in front of the stage, collecting fragments
        self.waiting_by_identity: dict = {}

    def expect_round(self, identity: tuple) -> None:
        """The round is next in its stream's retirement order."""
        stream = identity[:-1]
        pending = self.expected_by_stream.setdefault(stream, {})
        pending[identity] = None

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

    def context_of(self, identity: tuple) -> Optional[_PackingContext]:
        """The round's context, in assembly or waiting for a place."""
        context = self.context_by_identity.get(identity)
        if context is not None:
            return context
        return self.waiting_by_identity.get(identity)

    def is_in_assembly(self, context: _PackingContext) -> bool:
        return context.identity in self.context_by_identity

    def enter(self, context: _PackingContext) -> None:
        """The waiting round moves into assembly."""
        del self.waiting_by_identity[context.identity]
        self.context_by_identity[context.identity] = context

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


def _merge_adjacent_fragments(fragments) -> tuple:
    """Coalesce adjacent acquisitions without permuting measurement records.

    Stim record targets address the declared measurement order, so two parts
    with another acquisition between them stay apart.
    """
    merged = []
    by_fragment_index = operator.attrgetter("fragment_index")
    ordered = sorted(fragments, key=by_fragment_index)
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
    """Every event the round assembler reports, as one member."""

    round_event: trace_source.TraceSource = trace_source.new_source()
    copy_made: trace_source.TraceSource = trace_source.new_source()
