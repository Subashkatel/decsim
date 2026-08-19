"""Controller-side arrival of syndrome data: fragments of one round are
reassembled, the complete round is packed, and packed rounds are arbitrated
onto their route, C2B to Buffer 0 (window input) or CWD as a feedback-memory
round. ``SyndromeBuffer`` owns the round slots; a context is PARTIAL while
fragments are missing, PACKED_WAIT while it waits for its route, and
DRAINING once transmission has started."""

from __future__ import annotations

from dataclasses import dataclass
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
    stable_identity_order_key,
)


class ReassemblyQueueAdmission(Enum):
    """When a round starts competing for its route: as soon as its first
    fragment allocates a context, or once the round is complete."""

    ON_ALLOCATION = "on_allocation"
    ON_COMPLETION = "on_completion"


class IngressOverflowPolicy(Enum):
    """What a full Buffer 0 does with a round that cannot fit."""

    FAIL_STOP = "fail_stop"
    DROP_ROUND = "drop_round"


@dataclass(frozen=True)
class SyndromeIngressPolicy:
    queue_admission: ReassemblyQueueAdmission = ReassemblyQueueAdmission.ON_COMPLETION
    overflow: IngressOverflowPolicy = IngressOverflowPolicy.FAIL_STOP
    reassembly_timeout_ticks: Optional[int] = None


@dataclass(frozen=True)
class SyndromeIngressSnapshot:
    """Live ingress contexts by state."""
    ingress_context_capacity: Optional[int]
    ingress_contexts: int
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


class SyndromeIngressOverflow(RuntimeError):
    """A new round found no free slot in Buffer 0."""

    status = "controller_ingress_overflow"

    def __init__(self, *, tick, route, incoming_identity, capacity, snapshot):
        self.tick = tick
        self.route = route
        self.incoming_packet_or_fragment_identity = incoming_identity
        self.ingress_context_capacity = capacity
        self.ingress_contexts = snapshot.ingress_contexts
        self.partial_identities = snapshot.partial_identities
        self.packed_wait_identities = snapshot.packed_wait_identities
        super().__init__(f"controller ingress capacity {capacity} is full at tick {tick}")


class _IngressSlotState(Enum):
    PARTIAL = auto()
    PACKED_WAIT = auto()
    DRAINING = auto()


@dataclass
class _IngressContext:
    """One round on its way through the controller: its route, how far its
    assembly has come, and its packed packet once complete."""

    identity: tuple
    round_key: tuple
    route: SyndromePacketRoute
    fragment_count: int
    received_fragments: int = 0
    state: _IngressSlotState = _IngressSlotState.PARTIAL
    packet: Optional[SyndromeRoundPacket] = None
    packet_bits: Optional[int] = None
    c2b_reserved: bool = False
    c2b_delivered: bool = False


class SyndromeIngress:
    def __init__(
        self, engine, links: Optional[LinkModel] = None, t_pack: int = 0,
        log_syndromes: bool = True, *, ingress_context_capacity: Optional[int],
        window_input_receiver, feedback_memory_receiver,
        syndrome_buffer: Optional[SyndromeBuffer] = None,
        policy: SyndromeIngressPolicy = SyndromeIngressPolicy(),
    ):
        self.policy = policy
        self.engine = engine
        self.links = links if links is not None else logical_reference_profile().resolve()
        self.t_pack = t_pack
        self.log_syndromes = log_syndromes
        self.ingress_context_capacity = ingress_context_capacity
        self.window_input_receiver = window_input_receiver
        self.feedback_memory_receiver = feedback_memory_receiver
        if syndrome_buffer is None:
            syndrome_buffer = SyndromeBuffer(capacity=ingress_context_capacity)
        self.syndrome_buffer = syndrome_buffer
        self._contexts: dict[tuple, _IngressContext] = {}
        self._route_queues = {kind: [] for kind in SyndromePacketRouteKind}
        self._next_route_index = 0
        self._arbitration_pending = False
        self._dropped_rounds: set = set()
        self.reassembly_timeouts = 0
        self.ingress_drops = 0
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
        admission = self._admit_to_buffer(fragment, context)
        if admission is None:
            return
        context.received_fragments = admission.received_fragments
        if not admission.round_complete:
            return
        packing_takes_time = context.fragment_count > 1 and self.t_pack
        if packing_takes_time:
            self.engine.schedule(self.t_pack, lambda: self._finish_packing(context),
                                 label="controller pack")
        else:
            self._finish_packing(context)

    def _context_for(self, fragment, fragment_count, route) -> _IngressContext:
        """The live context of the fragment's round, opened on its first fragment."""
        round_key = (fragment.operation_id, fragment.round_index)
        identity = (route.kind.name, route.source_operation_id,
                    fragment.operation_id, fragment.round_index)
        context = self._contexts.get(identity)
        if context is not None:
            return context
        same_round_other_route = any(live.round_key == round_key
                                     for live in self._contexts.values())
        if same_round_other_route:
            raise ValueError("all fragments must share one typed route")
        return self._open_context(identity, round_key, route, fragment_count)

    def _admit_to_buffer(self, fragment, context: _IngressContext):
        """Hand the fragment to Buffer 0; None when the round was refused
        (dropped under DROP_ROUND, otherwise the run fails)."""
        if not self.syndrome_buffer.has_operation(fragment.operation_id):
            self.syndrome_buffer.open_operation(fragment.operation_id)
        admission = self.syndrome_buffer.accept_fragment(
            fragment, expected_fragments=context.fragment_count)
        if not admission.refused:
            return admission
        self._forget_context(context)
        if self.policy.overflow is IngressOverflowPolicy.DROP_ROUND:
            self.ingress_drops += 1
            self._dropped_rounds.add(context.round_key)
            return None
        raise SyndromeIngressOverflow(
            tick=self.engine.now, route=context.route, incoming_identity=context.identity,
            capacity=self.ingress_context_capacity, snapshot=self.ingress_snapshot())

    def _open_context(self, identity, round_key, route, fragment_count: int) -> _IngressContext:
        context = _IngressContext(identity, round_key, route, fragment_count)
        self._contexts[identity] = context
        if self.policy.queue_admission is ReassemblyQueueAdmission.ON_ALLOCATION:
            self._route_queues[route.kind].append(identity)
        timeout_ticks = self.policy.reassembly_timeout_ticks
        if timeout_ticks is not None:
            self.engine.schedule(timeout_ticks, lambda: self._expire_reassembly(identity),
                                 label="syndrome reassembly timeout")
        return context

    def _forget_context(self, context: _IngressContext) -> None:
        self._contexts.pop(context.identity, None)
        route_queue = self._route_queues[context.route.kind]
        if context.identity in route_queue:
            route_queue.remove(context.identity)

    def _finish_packing(self, context: _IngressContext) -> None:
        """The round is complete: pack it and let it compete for its route.
        Its publication tick is set now unless a C2B hop is priced later."""
        c2b_is_priced = LinkPath.C2B in self.links.paths
        publication_tick = None if c2b_is_priced else self.engine.now
        packet = self.syndrome_buffer.finish_packing(context.round_key,
                                                     publication_tick=publication_tick)
        context.packet = packet
        context.packet_bits = _packet_bits(packet)
        context.state = _IngressSlotState.PACKED_WAIT
        if self.policy.queue_admission is ReassemblyQueueAdmission.ON_COMPLETION:
            self._route_queues[context.route.kind].append(context.identity)
        self._schedule_arbitration()

    # ---- arbitration onto the routes

    def _schedule_arbitration(self) -> None:
        if self._arbitration_pending:
            return
        self._arbitration_pending = True
        self.engine.schedule(0, self._arbitrate, label="syndrome ingress arbitration")

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
        """Rounds pipeline onto C2B: every waiting round behind in-flight ones
        is sent in order; a refused round stops the walk."""
        progressed = False
        for identity in list(route_queue):
            context = self._contexts[identity]
            if context.state is not _IngressSlotState.PACKED_WAIT:
                continue
            if not self._attempt_head(context):
                break
            progressed = True
        return progressed

    def _attempt_head(self, context: _IngressContext) -> bool:
        if context.state is not _IngressSlotState.PACKED_WAIT:
            return False
        if context.route.kind is SyndromePacketRouteKind.WINDOW_INPUT:
            return self._transmit_window_input_round(context)
        context.state = _IngressSlotState.DRAINING
        self._transmit_feedback_memory_round(context)
        return True

    # ---- window input: C2B to Buffer 0

    def _transmit_window_input_round(self, context: _IngressContext) -> bool:
        """Reserve C2B once; a packet delivered but backpressured is retried
        without a second reservation, as is every packet on a fabric without C2B."""
        c2b_is_priced = LinkPath.C2B in self.links.paths
        if context.c2b_reserved or not c2b_is_priced:
            return self._deliver_window_input_round(context)
        packet = context.packet
        delay_ticks = self._reserve(LinkPath.C2B, payload_bits=context.packet_bits,
                                    attribution=self._packet_attribution(packet))
        context.c2b_reserved = True
        context.state = _IngressSlotState.DRAINING
        self.engine.schedule(delay_ticks, lambda: self._deliver_window_input_round(context),
                             label="controller->syndrome buffer 0")
        return True

    def _deliver_window_input_round(self, context: _IngressContext) -> bool:
        c2b_is_priced = LinkPath.C2B in self.links.paths
        if c2b_is_priced and not context.c2b_delivered:
            self.syndrome_buffer.mark_publication_tick(context.round_key, self.engine.now)
            context.c2b_delivered = True
        if self._refused_round_ahead_of(context):
            context.state = _IngressSlotState.PACKED_WAIT   # keep round order
            return False
        accepted = self.window_input_receiver.accept_window_input(context.packet)
        if not accepted:
            context.state = _IngressSlotState.PACKED_WAIT
            return False
        context.state = _IngressSlotState.DRAINING
        self.engine.schedule(0, lambda: self._release_context(context),
                             label="window input publication complete")
        return True

    def _refused_round_ahead_of(self, context: _IngressContext) -> bool:
        route_queue = self._route_queues[context.route.kind]
        ahead = route_queue[:route_queue.index(context.identity)]
        return any(self._contexts[identity].state is _IngressSlotState.PACKED_WAIT
                   for identity in ahead)

    def notify_window_input_ready(self) -> None:
        """Buffer 0 has room again: retry backpressured packets."""
        self._schedule_arbitration()

    # ---- feedback memory: CWD

    def _transmit_feedback_memory_round(self, context: _IngressContext) -> None:
        packet = context.packet
        source_operation_id = context.route.source_operation_id
        delay_ticks = self._reserve(LinkPath.CWD, payload_bits=context.packet_bits,
                                    attribution=self._packet_attribution(packet))
        self.engine.schedule(
            delay_ticks,
            lambda: self._deliver_feedback_memory_round(context, source_operation_id),
            label="controller->feedback memory")

    def _deliver_feedback_memory_round(self, context: _IngressContext,
                                       source_operation_id) -> None:
        self.feedback_memory_receiver.accept_feedback_memory_round(source_operation_id)
        self.syndrome_buffer.release_round(context.round_key)
        self._release_context(context)

    # ---- context end

    def _expire_reassembly(self, identity) -> None:
        context = self._contexts.get(identity)
        if context is None or context.state is not _IngressSlotState.PARTIAL:
            return
        self.reassembly_timeouts += 1
        self._forget_context(context)
        self.syndrome_buffer.release_round(context.round_key)
        raise SyndromeReassemblyTimeout(
            tick=self.engine.now, identity=identity,
            received_fragments=context.received_fragments,
            expected_fragments=context.fragment_count)

    def _release_context(self, context: _IngressContext) -> None:
        route_queue = self._route_queues[context.route.kind]
        if not route_queue or route_queue.pop(0) != context.identity:
            raise RuntimeError("controller released a non-head route slot")
        del self._contexts[context.identity]
        if any(self._route_queues.values()):
            self._schedule_arbitration()

    def check_work_settled(self) -> None:
        """Fail if the run ended with a partial or blocked ingress context."""
        snapshot = self.ingress_snapshot()
        if snapshot.ingress_contexts:
            live = (snapshot.partial_identities + snapshot.packed_wait_identities
                    + snapshot.draining_identities)
            raise RuntimeError(f"run ended with incomplete syndrome ingress contexts: {live}")

    def ingress_snapshot(self) -> SyndromeIngressSnapshot:
        identities_by_state = {
            state: tuple(context.identity for context in self._contexts.values()
                         if context.state is state)
            for state in _IngressSlotState}
        return SyndromeIngressSnapshot(
            ingress_context_capacity=self.ingress_context_capacity,
            ingress_contexts=len(self._contexts),
            partial_identities=identities_by_state[_IngressSlotState.PARTIAL],
            packed_wait_identities=identities_by_state[_IngressSlotState.PACKED_WAIT],
            draining_identities=identities_by_state[_IngressSlotState.DRAINING])

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


def _packet_bits(packet: SyndromeRoundPacket) -> Optional[int]:
    """The packed round's size, None when any fragment has no known size."""
    fragment_sizes = [fragment.size_bits for fragment in packet.fragments]
    if any(size is None for size in fragment_sizes):
        return None
    return sum(fragment_sizes)
