"""The controller's syndrome packing stage: raw measurement fragments of one
round are assembled in this stage's own workspace, merged, formed into
detection events (the device's formation table), and the finished round is
written into the stores (Buffer 0, and syndrome buffer 1 when wired) before
being arbitrated onto its route, CWB to the window input or WBD as a
feedback-memory round. The stores hold finished rounds only; a context is
PARTIAL while fragments are missing, PACKED_WAIT while it waits for its
route, and DRAINING once transmission has started."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Optional

from ..links.link_profiles import logical_reference_profile
from ..links.links import LinkModel, LinkPath, TrafficAttribution
from ..syndrome_buffer.syndrome_buffer import SyndromeBuffer
from ..message import (
    RetainedSyndromeFragment,
    SyndromePacketRoute,
    SyndromePacketRouteKind,
    SyndromeRoundPacket,
    same_stable_identity,
    stable_identity_order_key,
)


class ReassemblyQueueAdmission(Enum):
    """When a round starts competing for its route: as soon as its first
    fragment allocates a context, or once the round is complete."""

    ON_ALLOCATION = "on_allocation"
    ON_COMPLETION = "on_completion"


class PackingOverflowPolicy(Enum):
    """What a full Buffer 0 does with a round that cannot fit."""

    FAIL_STOP = "fail_stop"
    DROP_ROUND = "drop_round"


@dataclass(frozen=True)
class SyndromePackingPolicy:
    queue_admission: ReassemblyQueueAdmission = ReassemblyQueueAdmission.ON_COMPLETION
    overflow: PackingOverflowPolicy = PackingOverflowPolicy.FAIL_STOP
    reassembly_timeout_ticks: Optional[int] = None


@dataclass(frozen=True)
class SyndromePackingSnapshot:
    """Live packing contexts by state."""
    packing_context_capacity: Optional[int]
    packing_contexts: int
    partial_identities: tuple[tuple, ...]
    packed_wait_identities: tuple[tuple, ...]
    draining_identities: tuple[tuple, ...]


class SyndromeReassemblyTimeout(RuntimeError):
    """An incomplete round exceeded the configured reassembly timeout."""

    status = "syndrome_reassembly_timeout"

    def __init__(self, *, tick, identity, received_fragments, expected_fragments):
        self.tick = tick
        self.identity = identity
        self.received_fragments = received_fragments
        self.expected_fragments = expected_fragments
        super().__init__(
            f"syndrome reassembly {identity!r} timed out at tick {tick}: "
            f"received {received_fragments}/{expected_fragments} fragments")


class SyndromePackingOverflow(RuntimeError):
    """A new round found no free assembly context, or no free slot in
    Buffer 0."""

    status = "controller_packing_overflow"

    def __init__(self, *, tick, route, incoming_identity, capacity, snapshot):
        self.tick = tick
        self.route = route
        self.incoming_packet_or_fragment_identity = incoming_identity
        self.packing_context_capacity = capacity
        self.packing_contexts = snapshot.packing_contexts
        self.partial_identities = snapshot.partial_identities
        self.packed_wait_identities = snapshot.packed_wait_identities
        super().__init__(f"controller packing capacity {capacity} is full at tick {tick}")


class _PackingSlotState(Enum):
    PARTIAL = auto()
    PACKED_WAIT = auto()
    DRAINING = auto()


@dataclass
class _PackingContext:
    """One round on its way through the controller: its route, how far its
    assembly has come, and its packed packet once complete."""

    identity: tuple
    round_key: tuple
    route: SyndromePacketRoute
    fragment_count: int
    fragments: list = field(default_factory=list)
    received_fragments: int = 0
    state: _PackingSlotState = _PackingSlotState.PARTIAL
    packet: Optional[SyndromeRoundPacket] = None
    packet_bits: Optional[int] = None
    cwb_reserved: bool = False
    cwb_delivered: bool = False


class SyndromePacking:
    def __init__(
        self, engine, links: Optional[LinkModel] = None, t_pack: int = 0,
        log_syndromes: bool = True, *, packing_context_capacity: Optional[int],
        window_input_receiver, feedback_memory_receiver,
        syndrome_buffer: Optional[SyndromeBuffer] = None,
        syndrome_buffer_1=None,
        policy: SyndromePackingPolicy = SyndromePackingPolicy(),
        detector_formation=None,
    ):
        self.policy = policy
        # the source that forms a complete round's detection events from its
        # raw packet (the device, which holds the formation table); None for
        # timing-only or synthetic sources
        self.detector_formation = detector_formation
        self.engine = engine
        self.links = links if links is not None else logical_reference_profile().resolve()
        self.t_pack = t_pack
        self.log_syndromes = log_syndromes
        # assembly workspace: how many rounds may be in flight through this
        # stage at once; retention capacity is the stores' own knob
        self.packing_context_capacity = packing_context_capacity
        self.window_input_receiver = window_input_receiver
        self.feedback_memory_receiver = feedback_memory_receiver
        if syndrome_buffer is None:
            syndrome_buffer = SyndromeBuffer(capacity=packing_context_capacity)
        self.syndrome_buffer = syndrome_buffer
        self.syndrome_buffer_1 = syndrome_buffer_1
        self._contexts: dict[tuple, _PackingContext] = {}
        self._packed_rounds: set = set()
        self._route_queues = {kind: [] for kind in SyndromePacketRouteKind}
        self._next_route_index = 0
        self._arbitration_pending = False
        self._dropped_rounds: set = set()
        self.reassembly_timeouts = 0
        self.packing_drops = 0
        connect_ready = getattr(window_input_receiver, "connect_window_input_ready_receiver", None)
        if callable(connect_ready):
            connect_ready(self.notify_window_input_ready)

    # ---- arrival: QC delivery, controller processing, reassembly

    def relay_qpu_readout(self, payload, route: SyndromePacketRoute, *,
                          processing_ticks: int) -> None:
        """Carry one readout over QC, then receive it after controller processing."""
        fragment = RetainedSyndromeFragment.from_payload(payload)
        fragment_count = payload.n_fragments
        attribution = self._round_attribution(
            fragment.operation_id, (fragment.patch_id,), fragment.round_index)
        qc_delay = self._reserve(LinkPath.QC, payload_bits=payload.size_bits,
                                 attribution=attribution)

        def receive():
            self._receive_fragment(fragment, fragment_count, route)

        def at_controller():
            if processing_ticks == 0:
                receive()
            else:
                self.engine.schedule(processing_ticks, receive,
                                     label="controller-binary-availability")

        self.engine.schedule(qc_delay, at_controller, label="qpu->controller-readout")

    def _receive_fragment(self, fragment: RetainedSyndromeFragment, fragment_count: int,
                          route: SyndromePacketRoute) -> None:
        round_key = (fragment.operation_id, fragment.round_index)
        if round_key in self._dropped_rounds:
            return
        context = self._context_for(fragment, fragment_count, route)
        if context is None:
            return
        self._assemble_fragment(fragment, context)
        if context.received_fragments != context.fragment_count:
            return
        packing_takes_time = context.fragment_count > 1 and self.t_pack
        if packing_takes_time:
            self.engine.schedule(self.t_pack, lambda: self._finish_packing(context),
                                 label="controller pack")
        else:
            self._finish_packing(context)

    def _context_for(self, fragment, fragment_count, route) -> _PackingContext:
        """The live context of the fragment's round, opened on its first fragment."""
        round_key = (fragment.operation_id, fragment.round_index)
        identity = (route.kind.name, route.source_operation_id,
                    fragment.operation_id, fragment.round_index)
        context = self._contexts.get(identity)
        if context is not None:
            if fragment_count != context.fragment_count:
                raise ValueError("all fragments must declare the same count")
            return context
        if round_key in self._packed_rounds:
            raise ValueError(
                f"late fragment: round {round_key!r} was already packed")
        same_round_other_route = any(live.round_key == round_key
                                     for live in self._contexts.values())
        if same_round_other_route:
            raise ValueError("all fragments must share one typed route")
        capacity = self.packing_context_capacity
        if capacity is not None and len(self._contexts) >= capacity:
            if self.policy.overflow is PackingOverflowPolicy.DROP_ROUND:
                self.packing_drops += 1
                self._dropped_rounds.add(round_key)
                return None
            raise SyndromePackingOverflow(
                tick=self.engine.now, route=route, incoming_identity=identity,
                capacity=capacity, snapshot=self.packing_snapshot())
        return self._open_context(identity, round_key, route, fragment_count)

    def _assemble_fragment(self, fragment, context: _PackingContext) -> None:
        """Add one fragment to the round's assembly workspace."""
        if fragment.fragment_index >= context.fragment_count:
            raise ValueError("fragment index exceeds the declared count")
        if any(fragment.fragment_index == held.fragment_index
               for held in context.fragments):
            raise ValueError("duplicate syndrome fragment index")
        context.fragments.append(fragment)
        context.received_fragments = len(context.fragments)

    def _open_context(self, identity, round_key, route, fragment_count: int) -> _PackingContext:
        context = _PackingContext(identity, round_key, route, fragment_count)
        self._contexts[identity] = context
        if self.policy.queue_admission is ReassemblyQueueAdmission.ON_ALLOCATION:
            self._route_queues[route.kind].append(identity)
        timeout_ticks = self.policy.reassembly_timeout_ticks
        if timeout_ticks is not None:
            self.engine.schedule(timeout_ticks, lambda: self._expire_reassembly(identity),
                                 label="syndrome reassembly timeout")
        return context

    def _forget_context(self, context: _PackingContext) -> None:
        self._contexts.pop(context.identity, None)
        route_queue = self._route_queues[context.route.kind]
        if context.identity in route_queue:
            route_queue.remove(context.identity)

    def _finish_packing(self, context: _PackingContext) -> None:
        """The round is complete: merge, form, store, and let it compete for
        its route. Its publication tick is set now unless a CWB hop is
        priced later."""
        cwb_is_priced = LinkPath.CWB in self.links.paths
        publication_tick = None if cwb_is_priced else self.engine.now
        raw_fragments = _merge_fragments_by_patch(context.fragments)
        # CWB carries the raw measurement bits; detection events exist
        # only from the decoder input (Buffer 0) onward
        context.packet_bits = _fragment_bits(raw_fragments)
        packet = SyndromeRoundPacket(
            operation_id=context.round_key[0],
            round_index=context.round_key[1],
            fragments=self._form_detection_events(raw_fragments))
        if not self.syndrome_buffer.has_operation(packet.operation_id):
            self.syndrome_buffer.open_operation(packet.operation_id)
        admission = self.syndrome_buffer.accept_packed_round(
            packet, publication_tick=publication_tick)
        if admission.refused:
            self._forget_context(context)
            if self.policy.overflow is PackingOverflowPolicy.DROP_ROUND:
                self.packing_drops += 1
                self._dropped_rounds.add(context.round_key)
                return
            raise SyndromePackingOverflow(
                tick=self.engine.now, route=context.route,
                incoming_identity=context.identity,
                capacity=self.syndrome_buffer.capacity,
                snapshot=self.packing_snapshot())
        self._packed_rounds.add(context.round_key)
        self.engine.log_io(
            "Buffer 0",
            lambda: f"received round {packet.round_index} of "
                    f"op {packet.operation_id} from packing; "
                    f"{packet.defects_text()}; holds "
                    f"{self.syndrome_buffer.held_rounds_description()}")
        context.fragments = []
        if self.syndrome_buffer_1 is not None:
            # the dual write: the same packed round leaves for the room-side
            # store in parallel with its Buffer 0 publication
            self.syndrome_buffer_1.write(
                packet, packet_bits=context.packet_bits,
                attribution=self._packet_attribution(packet))
        context.packet = packet
        context.state = _PackingSlotState.PACKED_WAIT
        if self.policy.queue_admission is ReassemblyQueueAdmission.ON_COMPLETION:
            self._route_queues[context.route.kind].append(context.identity)
        self._schedule_arbitration()

    def _form_detection_events(self, raw_fragments):
        """Form the complete round's detection events from its raw packet.
        Sources without a formation table (timing-only or synthetic bits)
        are retained as they arrived."""
        form_round = getattr(getattr(self, "detector_formation", None), "form_round", None)
        if form_round is None or any(fragment.bits is None for fragment in raw_fragments):
            return raw_fragments
        if len(raw_fragments) != 1:
            raise ValueError("detector formation expects one merged raw fragment per round")
        (raw,) = raw_fragments
        formed = tuple(form_round(raw.operation_id, raw.round_index, raw.bits))
        return (replace(raw, bits=formed, size_bits=len(formed)),)

    # ---- arbitration onto the routes

    def _schedule_arbitration(self) -> None:
        if self._arbitration_pending:
            return
        self._arbitration_pending = True
        self.engine.schedule(0, self._arbitrate, label="syndrome packing arbitration")

    def _arbitrate(self) -> None:
        """Try each route once, round robin: the route after the last one that
        progressed goes first next time."""
        self._arbitration_pending = False
        kinds = tuple(SyndromePacketRouteKind)
        ordered_kinds = kinds[self._next_route_index:] + kinds[:self._next_route_index]
        for kind in ordered_kinds:
            route_queue = self._route_queues[kind]
            if not route_queue:
                continue
            if kind is SyndromePacketRouteKind.WINDOW_INPUT:
                progressed = self._drain_window_input_queue(route_queue)
            else:
                progressed = self._attempt_head(self._contexts[route_queue[0]])
            if progressed:
                self._next_route_index = (kinds.index(kind) + 1) % len(kinds)

    def _drain_window_input_queue(self, route_queue) -> bool:
        """Rounds pipeline onto CWB: every waiting round behind in-flight ones
        is sent in order; a refused round stops the walk."""
        progressed = False
        for identity in list(route_queue):
            context = self._contexts[identity]
            if context.state is not _PackingSlotState.PACKED_WAIT:
                continue
            if not self._attempt_head(context):
                break
            progressed = True
        return progressed

    def _attempt_head(self, context: _PackingContext) -> bool:
        if context.state is not _PackingSlotState.PACKED_WAIT:
            return False
        if context.route.kind is SyndromePacketRouteKind.WINDOW_INPUT:
            return self._transmit_window_input_round(context)
        context.state = _PackingSlotState.DRAINING
        self._transmit_feedback_memory_round(context)
        return True

    # ---- window input: CWB to Buffer 0

    def _transmit_window_input_round(self, context: _PackingContext) -> bool:
        """Reserve CWB once; a packet delivered but backpressured is retried
        without a second reservation, as is every packet on a fabric without CWB."""
        cwb_is_priced = LinkPath.CWB in self.links.paths
        if context.cwb_reserved or not cwb_is_priced:
            return self._deliver_window_input_round(context)
        packet = context.packet
        delay_ticks = self._reserve(LinkPath.CWB, payload_bits=context.packet_bits,
                                    attribution=self._packet_attribution(packet))
        context.cwb_reserved = True
        context.state = _PackingSlotState.DRAINING
        self.engine.schedule(delay_ticks, lambda: self._deliver_window_input_round(context),
                             label="controller->syndrome buffer 0")
        return True

    def _deliver_window_input_round(self, context: _PackingContext) -> bool:
        cwb_is_priced = LinkPath.CWB in self.links.paths
        if cwb_is_priced and not context.cwb_delivered:
            self.syndrome_buffer.mark_publication_tick(context.round_key, self.engine.now)
            context.cwb_delivered = True
        if self._refused_round_ahead_of(context):
            context.state = _PackingSlotState.PACKED_WAIT   # keep round order
            return False
        accepted = self.window_input_receiver.accept_window_input(context.packet)
        if not accepted:
            context.state = _PackingSlotState.PACKED_WAIT
            return False
        context.state = _PackingSlotState.DRAINING
        self.engine.schedule(0, lambda: self._release_context(context),
                             label="window input publication complete")
        return True

    def _refused_round_ahead_of(self, context: _PackingContext) -> bool:
        route_queue = self._route_queues[context.route.kind]
        ahead = route_queue[:route_queue.index(context.identity)]
        return any(self._contexts[identity].state is _PackingSlotState.PACKED_WAIT
                   for identity in ahead)

    def notify_window_input_ready(self) -> None:
        """Buffer 0 has room again: retry backpressured packets."""
        self._schedule_arbitration()

    # ---- feedback memory: WBD

    def _transmit_feedback_memory_round(self, context: _PackingContext) -> None:
        packet = context.packet
        source_operation_id = context.route.source_operation_id
        delay_ticks = self._reserve(LinkPath.WBD, payload_bits=context.packet_bits,
                                    attribution=self._packet_attribution(packet))
        self.engine.schedule(
            delay_ticks,
            lambda: self._deliver_feedback_memory_round(context, source_operation_id),
            label="controller->feedback memory")

    def _deliver_feedback_memory_round(self, context: _PackingContext,
                                       source_operation_id) -> None:
        self.feedback_memory_receiver.accept_feedback_memory_round(source_operation_id)
        self.syndrome_buffer.release_round(context.round_key)
        self._release_context(context)

    # ---- context end

    def _expire_reassembly(self, identity) -> None:
        context = self._contexts.get(identity)
        if context is None or context.state is not _PackingSlotState.PARTIAL:
            return
        self.reassembly_timeouts += 1
        self._forget_context(context)
        raise SyndromeReassemblyTimeout(
            tick=self.engine.now, identity=identity,
            received_fragments=context.received_fragments,
            expected_fragments=context.fragment_count)

    def _release_context(self, context: _PackingContext) -> None:
        route_queue = self._route_queues[context.route.kind]
        if not route_queue or route_queue.pop(0) != context.identity:
            raise RuntimeError("controller released a non-head route slot")
        del self._contexts[context.identity]
        if any(self._route_queues.values()):
            self._schedule_arbitration()

    def check_work_settled(self) -> None:
        """Fail if the run ended with a partial or blocked packing context."""
        snapshot = self.packing_snapshot()
        if snapshot.packing_contexts:
            live = (snapshot.partial_identities + snapshot.packed_wait_identities
                    + snapshot.draining_identities)
            raise RuntimeError(f"run ended with incomplete syndrome packing contexts: {live}")

    def packing_snapshot(self) -> SyndromePackingSnapshot:
        identities_by_state = {
            state: tuple(context.identity for context in self._contexts.values()
                         if context.state is state)
            for state in _PackingSlotState}
        return SyndromePackingSnapshot(
            packing_context_capacity=self.packing_context_capacity,
            packing_contexts=len(self._contexts),
            partial_identities=identities_by_state[_PackingSlotState.PARTIAL],
            packed_wait_identities=identities_by_state[_PackingSlotState.PACKED_WAIT],
            draining_identities=identities_by_state[_PackingSlotState.DRAINING])

    # ---- links

    def _reserve(self, path: LinkPath, *, payload_bits, attribution: TrafficAttribution) -> int:
        reservation = self.links.reserve(path, payload_bits=payload_bits,
                                         now_ticks=self.engine.now, attribution=attribution)
        return reservation.total_delay_ticks

    def _packet_attribution(self, packet: SyndromeRoundPacket) -> TrafficAttribution:
        patch_ids = tuple(fragment.patch_id for fragment in packet.fragments)
        return self._round_attribution(packet.operation_id, patch_ids, packet.round_index)

    @staticmethod
    def _round_attribution(operation_id, patch_ids: tuple, round_index: int):
        return TrafficAttribution(
            operation_id=operation_id,
            patch_ids=tuple(sorted(patch_ids, key=stable_identity_order_key)),
            window_id=None, round_lo=round_index, round_hi=round_index)


def _fragment_bits(fragments) -> Optional[int]:
    """The fragments' wire size, None when any fragment has no known size."""
    fragment_sizes = [fragment.size_bits for fragment in fragments]
    if any(size is None for size in fragment_sizes):
        return None
    return sum(fragment_sizes)


def _merge_fragments_by_patch(fragments) -> tuple:
    """Order fragments by index, merging parts from the same patch.

    ``SyndromeRoundPacket`` requires distinct patch identities, so parts of
    one patch concatenate bits and sizes in fragment-index order. Distinct
    patches keep their own immutable fragments untouched.
    """
    merged: list = []
    for fragment in sorted(fragments, key=lambda item: item.fragment_index):
        prior_index = next(
            (
                index
                for index, prior in enumerate(merged)
                if same_stable_identity(prior.patch_id, fragment.patch_id)
            ),
            None,
        )
        if prior_index is None:
            merged.append(fragment)
            continue
        prior = merged[prior_index]
        bits = (
            prior.bits + fragment.bits
            if prior.bits is not None and fragment.bits is not None
            else None
        )
        size_bits = (
            prior.size_bits + fragment.size_bits
            if prior.size_bits is not None and fragment.size_bits is not None
            else None
        )
        merged[prior_index] = replace(prior, bits=bits, size_bits=size_bits)
    return tuple(merged)
