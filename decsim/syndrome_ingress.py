"""Receive syndrome fragments and publish complete upstream rounds.

This module begins after QPU-to-controller delivery and models fragment
reassembly, packing delay, and route arbitration. ``SyndromeBuffer`` owns the round allocation and its lifetime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from .link_profiles import logical_reference_profile
from .links import LinkModel, LinkPath, TrafficAttribution
from .syndrome_buffer import SyndromeBuffer, SyndromeBufferCapacityExhaustion
from .message import (
    RetainedSyndromeFragment,
    SyndromePacketRoute,
    SyndromePacketRouteKind,
    SyndromePayload,
    SyndromeRoundPacket,
    stable_identity_order_key,
)


class ReassemblyQueueAdmission(Enum):
    """When a reassembly context begins competing for an output."""

    ON_ALLOCATION = "on_allocation"
    ON_COMPLETION = "on_completion"


class IngressOverflowPolicy(Enum):
    """What finite ingress storage does when a new context cannot fit."""

    FAIL_STOP = "fail_stop"
    DROP_ROUND = "drop_round"


@dataclass(frozen=True)
class SyndromeIngressPolicy:
    """Configure reassembly queueing, overflow, and timeout behavior."""

    queue_admission: ReassemblyQueueAdmission = ReassemblyQueueAdmission.ON_COMPLETION
    overflow: IngressOverflowPolicy = IngressOverflowPolicy.FAIL_STOP
    reassembly_timeout_ticks: Optional[int] = None


@dataclass(frozen=True)
class SyndromeIngressSnapshot:
    """Snapshot of live ingress reassembly and route contexts."""
    ingress_context_capacity: Optional[int]
    ingress_contexts: int
    free_slot_indices: tuple[int, ...]
    identity_to_slot: tuple[tuple[tuple, int], ...]
    partial_identities: tuple[tuple, ...]
    packed_wait_identities: tuple[tuple, ...]
    draining_identities: tuple[tuple, ...]


class SyndromeReassemblyTimeout(RuntimeError):
    """An incomplete round exceeded an explicitly configured timeout."""

    status = "syndrome_reassembly_timeout"

    def __init__(self, *, tick, identity, received_fragments,
                 expected_fragments):
        self.tick = tick
        self.identity = identity
        self.received_fragments = received_fragments
        self.expected_fragments = expected_fragments
        super().__init__(
            f"syndrome reassembly {identity!r} timed out at tick {tick}: "
            f"received {received_fragments}/{expected_fragments} fragments"
        )


class SyndromeIngressOverflow(RuntimeError):
    """A new packet could not claim a physical syndrome ingress slot."""

    status = "controller_ingress_overflow"

    def __init__(self, *, tick, route, incoming_identity, capacity, snapshot):
        self.tick = tick
        self.route = route
        self.incoming_packet_or_fragment_identity = incoming_identity
        self.ingress_context_capacity = capacity
        self.ingress_contexts = snapshot.ingress_contexts
        self.partial_identities = snapshot.partial_identities
        self.packed_wait_identities = snapshot.packed_wait_identities
        message = f"controller ingress capacity {capacity} is full at tick {tick}"
        super().__init__(message)


class _IngressSlotState(Enum):
    PARTIAL = auto()
    PACKED_WAIT = auto()
    DRAINING = auto()


@dataclass
class _IngressContext:
    identity: tuple
    route: SyndromePacketRoute
    fragment_count: int
    state: _IngressSlotState = _IngressSlotState.PARTIAL
    packet: Optional[SyndromeRoundPacket] = None
    packet_bits: Optional[int] = None
    c2b_reserved: bool = False
    c2b_delivered: bool = False


class SyndromeIngress:
    """Assemble fragments and send complete rounds to their route.

    A slot is partial while fragments are missing, packed while waiting for its
    route, and draining after transmission starts.
    """

    def __init__(
        self, engine, links: Optional[LinkModel] = None, t_pack: int = 0,
        log_syndromes: bool = True, *, ingress_context_capacity: Optional[int],
        window_input_receiver, feedback_memory_receiver,
        syndrome_buffer: Optional[SyndromeBuffer] = None,
        policy: SyndromeIngressPolicy = SyndromeIngressPolicy(),
    ):
        self.policy = policy
        self.engine = engine
        default_links = logical_reference_profile().resolve()
        self.links = links if links is not None else default_links
        self.t_pack = t_pack
        self.log_syndromes = log_syndromes
        self.ingress_context_capacity = ingress_context_capacity
        self.window_input_receiver = window_input_receiver
        self.feedback_memory_receiver = feedback_memory_receiver
        self.syndrome_buffer = (
            SyndromeBuffer(capacity=ingress_context_capacity)
            if syndrome_buffer is None else syndrome_buffer
        )
        finite_slots = [None] * ingress_context_capacity \
            if ingress_context_capacity is not None else []
        self._slots: list[Optional[_IngressContext]] = finite_slots
        self._free_slot_indices = (
            set(range(ingress_context_capacity))
            if ingress_context_capacity is not None else set()
        )
        self._identity_to_slot: dict[tuple, int] = {}
        self._route_queues = {kind: [] for kind in SyndromePacketRouteKind}
        self._next_route_index = 0
        self._arbitration_pending = False
        self._completed_rounds: set = set()
        self._dropped_rounds: set = set()
        self.reassembly_timeouts = 0
        self.ingress_drops = 0
        connect_ready = getattr(
            window_input_receiver, "connect_window_input_ready_receiver", None)
        if callable(connect_ready):
            connect_ready(self.notify_window_input_ready)

    # ------------------------------------------------------- syndrome path

    def relay_qpu_readout(
        self, payload, route: SyndromePacketRoute, *, processing_ticks: int,
    ) -> None:
        """Deliver a QPU result, then expose it after controller processing."""
        fragment_count = payload.n_fragments
        fragment = RetainedSyndromeFragment.from_payload(payload)
        attribution = self._round_attribution(
            fragment.operation_id, (fragment.patch_id,), fragment.round_index)
        qc_delay = self._reserve(
            LinkPath.QC, payload_bits=payload.size_bits, attribution=attribution)

        def at_controller() -> None:
            receive = lambda: self._receive_fragment(
                fragment, fragment_count, route)
            if processing_ticks == 0:
                receive()
            else:
                self.engine.schedule(
                    processing_ticks, receive,
                    label="controller-binary-availability")

        self.engine.schedule(
            qc_delay, at_controller, label="qpu->controller-readout")

    def relay_syndrome(self, payload, route: SyndromePacketRoute) -> None:
        """Accept one binary fragment after QPU-to-controller delivery."""
        if payload.n_fragments < 1:
            raise ValueError("n_fragments must be at least one")
        fragment_count = payload.n_fragments
        fragment = RetainedSyndromeFragment.from_payload(payload)
        attribution = self._round_attribution(
            fragment.operation_id, (fragment.patch_id,), fragment.round_index)
        delay = self._reserve(
            LinkPath.QC, payload_bits=payload.size_bits, attribution=attribution)
        receive = lambda: self._receive_fragment(fragment, fragment_count, route)
        self.engine.schedule(delay, receive, label="qpu->controller-ingress")

    def _receive_fragment(
        self,
        fragment: RetainedSyndromeFragment,
        fragment_count: int,
        route: SyndromePacketRoute,
    ) -> None:
        round_key = (fragment.operation_id, fragment.round_index)
        if round_key in self._dropped_rounds:
            return
        identity = (
            route.kind.name, route.source_operation_id,
            fragment.operation_id, fragment.round_index,
        )
        slot_index = self._identity_to_slot.get(identity)
        if slot_index is None:
            if any(live_identity[-2:] == round_key
                   for live_identity in self._identity_to_slot):
                raise ValueError("all fragments must share one typed route")
            slot_index = self._allocate_slot(identity, route, fragment_count)
            if slot_index is None:
                self._dropped_rounds.add(round_key)
                return
        pending = self._slots[slot_index]
        if not self.syndrome_buffer.has_operation(fragment.operation_id):
            self.syndrome_buffer.open_operation(fragment.operation_id)
        try:
            admission = self.syndrome_buffer.accept_fragment(
                fragment, expected_fragments=fragment_count)
        except SyndromeBufferCapacityExhaustion:
            if self._identity_to_slot.get(identity) == slot_index:
                del self._identity_to_slot[identity]
                if slot_index == len(self._slots) - 1 and (
                        self.ingress_context_capacity is not None
                        and slot_index >= self.ingress_context_capacity):
                    self._slots.pop()
                else:
                    self._slots[slot_index] = None
                    self._free_slot_indices.add(slot_index)
                route_queue = self._route_queues[route.kind]
                if slot_index in route_queue:
                    route_queue.remove(slot_index)
            if self.policy.overflow is IngressOverflowPolicy.DROP_ROUND:
                self.ingress_drops += 1
                self._dropped_rounds.add(round_key)
                return
            raise SyndromeIngressOverflow(
                tick=self.engine.now, route=route,
                incoming_identity=identity,
                capacity=self.ingress_context_capacity,
                snapshot=self.ingress_snapshot(),
            )
        if not admission.round_complete:
            return
        if pending.fragment_count > 1 and self.t_pack:
            self.engine.schedule(
                self.t_pack,
                lambda: self._finish_packing(slot_index),
                label="controller pack",
            )
        else:
            self._finish_packing(slot_index)

    def _allocate_slot(self, identity, route, fragment_count: int) -> int:
        if self._free_slot_indices:
            slot_index = min(self._free_slot_indices)
            self._free_slot_indices.remove(slot_index)
        else:
            slot_index = len(self._slots)
        slot = _IngressContext(identity, route, fragment_count)
        if slot_index == len(self._slots):
            self._slots.append(slot)
        else:
            self._slots[slot_index] = slot
        self._identity_to_slot[identity] = slot_index
        if self.policy.queue_admission is ReassemblyQueueAdmission.ON_ALLOCATION:
            self._route_queues[route.kind].append(slot_index)
        if self.policy.reassembly_timeout_ticks is not None:
            self.engine.schedule(
                self.policy.reassembly_timeout_ticks,
                lambda context=identity: self._expire_reassembly(context),
                label="syndrome reassembly timeout",
            )
        return slot_index

    def _finish_packing(self, slot_index) -> None:
        slot = self._slots[slot_index]
        round_key = (slot.identity[-2], slot.identity[-1])
        publication_tick = (
            None if LinkPath.C2B in self.links.paths else self.engine.now)
        packet = self.syndrome_buffer.finish_packing(
            round_key, publication_tick=publication_tick)
        slot.packet = packet
        fragment_sizes = [item.size_bits for item in packet.fragments]
        slot.packet_bits = (
            sum(fragment_sizes)
            if all(size is not None for size in fragment_sizes) else None
        )
        slot.state = _IngressSlotState.PACKED_WAIT
        if self.policy.queue_admission is ReassemblyQueueAdmission.ON_COMPLETION:
            self._route_queues[slot.route.kind].append(slot_index)
        self._schedule_arbitration()

    def _schedule_arbitration(self) -> None:
        if self._arbitration_pending:
            return
        self._arbitration_pending = True
        self.engine.schedule(0, self._arbitrate,
                             label="syndrome ingress arbitration")

    def _arbitrate(self) -> None:
        """Try each ready route once, rotating which route goes first."""
        self._arbitration_pending = False
        kinds = tuple(SyndromePacketRouteKind)
        ordered_kinds = kinds[self._next_route_index:] + kinds[:self._next_route_index]
        for kind in ordered_kinds:
            route_queue = self._route_queues[kind]
            if not route_queue:
                continue
            if kind is SyndromePacketRouteKind.WINDOW_INPUT:
                # Rounds pipeline onto the C2B link: every waiting round behind
                # in-flight ones is sent in order; a refused round stops the walk.
                progressed = False
                for slot_index in list(route_queue):
                    slot = self._slots[slot_index]
                    if slot.state is _IngressSlotState.PACKED_WAIT:
                        if not self._attempt_head(slot_index):
                            break
                        progressed = True
                if not progressed:
                    continue
            elif not self._attempt_head(route_queue[0]):
                continue
            self._next_route_index = (kinds.index(kind) + 1) % len(kinds)

    def _attempt_head(self, slot_index: int) -> bool:
        slot = self._slots[slot_index]
        if slot.state is not _IngressSlotState.PACKED_WAIT:
            return False
        if slot.route.kind is SyndromePacketRouteKind.WINDOW_INPUT:
            return self._transmit_window_input_round(slot_index, slot)
        else:
            slot.state = _IngressSlotState.DRAINING
            self._transmit_feedback_memory_round(slot_index, slot)
        return True

    def _transmit_window_input_round(
        self, slot_index: int, slot: _IngressContext,
    ) -> bool:
        """Reserve C2B once, then retain that reservation across backpressure."""
        if not slot.c2b_reserved and LinkPath.C2B in self.links.paths:
            packet = slot.packet
            attribution = self._round_attribution(
                packet.operation_id,
                tuple(fragment.patch_id for fragment in packet.fragments),
                packet.round_index,
            )
            delay_ticks = self._reserve(
                LinkPath.C2B,
                payload_bits=slot.packet_bits,
                attribution=attribution,
            )
            slot.c2b_reserved = True
            slot.state = _IngressSlotState.DRAINING
            self.engine.schedule(
                delay_ticks,
                lambda: self._deliver_window_input_round(slot_index),
                label="controller->syndrome buffer 0",
            )
            return True

        # No C2B edge on this fabric, or a delivered-but-backpressured packet:
        # deliver directly, without a second reservation.
        return self._deliver_window_input_round(slot_index)

    def _deliver_window_input_round(self, slot_index: int) -> bool:
        slot = self._slots[slot_index]
        if LinkPath.C2B in self.links.paths and not slot.c2b_delivered:
            round_identity = (slot.packet.operation_id, slot.packet.round_index)
            self.syndrome_buffer.mark_publication_tick(
                round_identity, self.engine.now)
            slot.c2b_delivered = True
        route_queue = self._route_queues[slot.route.kind]
        if any(self._slots[ahead].state is _IngressSlotState.PACKED_WAIT
               for ahead in route_queue[:route_queue.index(slot_index)]):
            slot.state = _IngressSlotState.PACKED_WAIT   # a refused round is ahead; keep order
            return False
        accepted = self.window_input_receiver.accept_window_input(slot.packet)
        if not accepted:
            slot.state = _IngressSlotState.PACKED_WAIT
            return False
        slot.state = _IngressSlotState.DRAINING
        self.engine.schedule(
            0,
            lambda: self._release_slot(slot_index),
            label="window input publication complete",
        )
        return True

    def notify_window_input_ready(self) -> None:
        """Retry a backpressured delivered packet without reserving C2B again."""
        self._schedule_arbitration()

    def _transmit_feedback_memory_round(
        self, slot_index: int, slot: _IngressContext,
    ) -> None:
        packet = slot.packet
        attribution = self._round_attribution(
            packet.operation_id,
            tuple(fragment.patch_id for fragment in packet.fragments),
            packet.round_index,
        )
        source_operation_id = slot.route.source_operation_id
        self.engine.schedule(
            self._reserve(
                LinkPath.CWD,
                payload_bits=slot.packet_bits,
                attribution=attribution,
            ),
            lambda: self._deliver_feedback_memory_round(
                slot_index, source_operation_id
            ),
            label="controller->feedback memory",
        )

    def _deliver_feedback_memory_round(
        self, slot_index: int, source_operation_id,
    ) -> None:
        self.feedback_memory_receiver.accept_feedback_memory_round(
            source_operation_id
        )
        slot = self._slots[slot_index]
        self.syndrome_buffer.release_round(
            (slot.packet.operation_id, slot.packet.round_index)
        )
        self._release_slot(slot_index)

    def _expire_reassembly(self, identity) -> None:
        """Raise an error when an incomplete round exceeds its timeout."""
        slot_index = self._identity_to_slot.get(identity)
        if slot_index is None:
            return
        slot = self._slots[slot_index]
        from .syndrome_buffer import SyndromeBufferRoundState
        round_key = (identity[-2], identity[-1])
        state = self.syndrome_buffer.round_state(round_key)
        if state is not SyndromeBufferRoundState.ASSEMBLING:
            return
        received = len(self.syndrome_buffer._rounds[round_key].fragments)
        self.reassembly_timeouts += 1
        route_queue = self._route_queues[slot.route.kind]
        if slot_index in route_queue:
            route_queue.remove(slot_index)
        del self._identity_to_slot[identity]
        self.syndrome_buffer.release_round((identity[-2], identity[-1]))
        self._slots[slot_index] = None
        self._free_slot_indices.add(slot_index)
        raise SyndromeReassemblyTimeout(
            tick=self.engine.now,
            identity=identity,
            received_fragments=received,
            expected_fragments=slot.fragment_count,
        )

    def _release_slot(self, slot_index: int) -> None:
        slot = self._slots[slot_index]
        route_queue = self._route_queues[slot.route.kind]
        if not route_queue or route_queue.pop(0) != slot_index:
            raise RuntimeError("controller released a non-head route slot")
        del self._identity_to_slot[slot.identity]
        round_key = (slot.packet.operation_id, slot.packet.round_index)
        self._completed_rounds.add(round_key)
        self._slots[slot_index] = None
        self._free_slot_indices.add(slot_index)
        if any(self._route_queues.values()):
            self._schedule_arbitration()

    def check_work_settled(self) -> None:
        """Fail if the run ended with a partial or blocked ingress context."""
        snapshot = self.ingress_snapshot()
        if snapshot.ingress_contexts:
            raise RuntimeError(
                "run ended with incomplete syndrome ingress contexts: "
                f"{snapshot.partial_identities + snapshot.packed_wait_identities + snapshot.draining_identities}"
            )

    def ingress_snapshot(self) -> SyndromeIngressSnapshot:
        identities_by_state = {
            state: tuple(slot.identity for slot in self._slots
                         if slot is not None and slot.state is state)
            for state in _IngressSlotState}
        return SyndromeIngressSnapshot(
            ingress_context_capacity=self.ingress_context_capacity,
            ingress_contexts=len(self._identity_to_slot),
            free_slot_indices=(
                tuple(sorted(self._free_slot_indices))
                if self.ingress_context_capacity is not None else ()
            ),
            identity_to_slot=tuple(sorted(
                self._identity_to_slot.items(), key=lambda item: item[1])),
            partial_identities=identities_by_state[_IngressSlotState.PARTIAL],
            packed_wait_identities=identities_by_state[
                _IngressSlotState.PACKED_WAIT],
            draining_identities=identities_by_state[_IngressSlotState.DRAINING],
        )

    def _reserve(
        self,
        path: LinkPath,
        *,
        payload_bits,
        attribution: TrafficAttribution,
    ) -> int:
        reservation = self.links.reserve(
            path,
            payload_bits=payload_bits,
            now_ticks=self.engine.now,
            attribution=attribution,
        )
        return reservation.total_delay_ticks

    @staticmethod
    def _round_attribution(operation_id, patch_ids: tuple, round_index: int):
        return TrafficAttribution(
            operation_id=operation_id,
            patch_ids=tuple(sorted(patch_ids, key=stable_identity_order_key)),
            window_id=None,
            round_lo=round_index,
            round_hi=round_index,
        )
