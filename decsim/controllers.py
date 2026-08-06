"""Typed controller staging and the classical reaction-path link relays."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Callable, Optional

from .links import LinkModel, LinkModelConfig, LinkPath, TrafficAttribution
from .message import (
    RetainedSyndromeFragment,
    SyndromePacketRoute,
    SyndromePacketRouteKind,
    SyndromeRoundPacket,
    same_stable_identity,
    stable_identity_order_key,
)


@dataclass(frozen=True)
class ControllerStagingSnapshot:
    """Immutable observation of controller-owned capacity state."""
    controller_capacity: Optional[int]
    controller_occupancy: int
    free_slot_indices: tuple[int, ...]
    identity_to_slot: tuple[tuple[tuple, int], ...]
    partial_identities: tuple[tuple, ...]
    packed_wait_identities: tuple[tuple, ...]
    draining_identities: tuple[tuple, ...]


class ControllerIngressOverflow(RuntimeError):
    """A new packet could not claim a physical controller staging slot."""

    status = "controller_ingress_overflow"

    def __init__(self, *, tick, route, incoming_identity, capacity, snapshot):
        self.tick = tick
        self.route = route
        self.incoming_packet_or_fragment_identity = incoming_identity
        self.controller_capacity = capacity
        self.controller_occupancy = snapshot.controller_occupancy
        self.partial_identities = snapshot.partial_identities
        self.packed_wait_identities = snapshot.packed_wait_identities
        message = f"controller ingress capacity {capacity} is full at tick {tick}"
        super().__init__(message)


class _ControllerSlotState(Enum):
    PARTIAL = auto()
    PACKED_WAIT = auto()
    DRAINING = auto()


@dataclass
class _StagingSlot:
    identity: tuple
    route: SyndromePacketRoute
    fragment_count: int
    fragments: list[RetainedSyndromeFragment] = field(default_factory=list)
    state: _ControllerSlotState = _ControllerSlotState.PARTIAL
    packet: Optional[SyndromeRoundPacket] = None
    packet_bits: Optional[int] = None


class ModularController:
    """Relays payloads and decisions across the named classical links."""

    def __init__(
        self, engine, links: Optional[LinkModel] = None, t_pack: int = 0,
        log_syndromes: bool = True, *, controller_capacity: Optional[int],
        window_input_receiver, feedback_memory_receiver,
    ):
        if controller_capacity is not None and (
            type(controller_capacity) is not int or controller_capacity < 1
        ):
            raise TypeError("controller_capacity must be a positive int or None")
        self.engine = engine
        default_links = LinkModelConfig.reference_fixed_latency_profile().resolve()
        self.links = links if links is not None else default_links
        self.t_pack = t_pack
        self.log_syndromes = log_syndromes
        self.controller_capacity = controller_capacity
        self.window_input_receiver = window_input_receiver
        self.feedback_memory_receiver = feedback_memory_receiver
        finite_slots = [None] * controller_capacity \
            if controller_capacity is not None else []
        self._slots: list[Optional[_StagingSlot]] = finite_slots
        self._free_slot_indices = (
            set(range(controller_capacity))
            if controller_capacity is not None else set()
        )
        self._next_unbounded_slot_index = 0
        self._identity_to_slot: dict[tuple, int] = {}
        self._route_queues = {kind: [] for kind in SyndromePacketRouteKind}
        self._next_route_index = 0
        self._arbitration_pending = False
        self._completed_rounds: set = set()

    # ------------------------------------------------------- syndrome path

    def relay_syndrome(self, payload, route: SyndromePacketRoute) -> None:
        """Chip -> controller (t_qc) -> decoder (t_cd), buffering fragments."""
        if type(route) is not SyndromePacketRoute:
            raise TypeError("relay_syndrome requires a typed packet route")
        if type(payload.n_fragments) is not int:
            raise TypeError("n_fragments must be an exact built-in int")
        if payload.n_fragments < 1:
            raise ValueError("n_fragments must be at least one")
        fragment_count = payload.n_fragments
        fragment = RetainedSyndromeFragment.from_payload(payload)
        attribution = self._round_attribution(
            fragment.operation_id, (fragment.patch_id,), fragment.round_index)
        delay = self._reserve(
            LinkPath.QC, payload_bits=payload.size_bits, attribution=attribution)
        receive = lambda: self._receive_fragment(fragment, fragment_count, route)
        self.engine.schedule(delay, receive, label="chip->controller")

    def _receive_fragment(
        self,
        fragment: RetainedSyndromeFragment,
        fragment_count: int,
        route: SyndromePacketRoute,
    ) -> None:
        round_key = (fragment.operation_id, fragment.round_index)
        if round_key in self._completed_rounds:
            raise ValueError(f"syndrome round {round_key!r} already completed")
        identity = (
            route.kind.name, route.source_operation_id,
            fragment.operation_id, fragment.round_index,
        )
        slot_index = self._identity_to_slot.get(identity)
        if slot_index is None:
            if any(
                live_identity[-2:] == round_key
                for live_identity in self._identity_to_slot
            ):
                raise ValueError("all fragments must share one typed route")
            slot_index = self._allocate_slot(identity, route, fragment_count)
        pending = self._slots[slot_index]
        if fragment_count != pending.fragment_count:
            raise ValueError("all fragments must declare the same count")
        duplicate = any(fragment.fragment_index == candidate.fragment_index
                        for candidate in pending.fragments)
        if duplicate:
            raise ValueError("duplicate syndrome fragment index")
        if fragment.fragment_index >= pending.fragment_count:
            raise ValueError("syndrome fragment index exceeds declared count")
        if len(pending.fragments) >= pending.fragment_count:
            raise ValueError("too many distinct syndrome fragments")

        pending.fragments.append(fragment)
        if len(pending.fragments) < pending.fragment_count:
            if self.log_syndromes:
                progress = f"{len(pending.fragments)}/{pending.fragment_count}"
                self.engine.log(
                    "Controller", f"buffered fragment {progress} of round "
                    f"{fragment.round_index} of op#{fragment.operation_id}")
            return
        packet = SyndromeRoundPacket(
            operation_id=fragment.operation_id,
            round_index=fragment.round_index,
            fragments=self._collapse_fragments(pending.fragments),
        )
        if self.log_syndromes:
            self.engine.log(
                "Controller", f"round {fragment.round_index} of "
                f"op#{fragment.operation_id} complete; packet staged")
        fragment_sizes = [item.size_bits for item in packet.fragments]
        packet_bits = (
            sum(fragment_sizes)
            if all(size is not None for size in fragment_sizes) else None
        )
        if pending.fragment_count > 1 and self.t_pack:
            self.engine.schedule(
                self.t_pack,
                lambda: self._finish_packing(slot_index, packet, packet_bits),
                label="controller pack")
        else:
            self._finish_packing(slot_index, packet, packet_bits)

    def _allocate_slot(self, identity, route, fragment_count: int) -> int:
        if self.controller_capacity is not None:
            if not self._free_slot_indices:
                raise ControllerIngressOverflow(
                    tick=self.engine.now,
                    route=route,
                    incoming_identity=identity,
                    capacity=self.controller_capacity,
                    snapshot=self.staging_snapshot(),
                )
            slot_index = min(self._free_slot_indices)
            self._free_slot_indices.remove(slot_index)
        elif self._free_slot_indices:
            slot_index = min(self._free_slot_indices)
            self._free_slot_indices.remove(slot_index)
        else:
            slot_index = self._next_unbounded_slot_index
            self._next_unbounded_slot_index += 1
        slot = _StagingSlot(identity, route, fragment_count)
        if slot_index == len(self._slots):
            self._slots.append(slot)
        else:
            self._slots[slot_index] = slot
        self._identity_to_slot[identity] = slot_index
        self._route_queues[route.kind].append(slot_index)
        return slot_index

    def _finish_packing(self, slot_index, packet, packet_bits) -> None:
        slot = self._slots[slot_index]
        slot.packet = packet
        slot.packet_bits = packet_bits
        slot.state = _ControllerSlotState.PACKED_WAIT
        self._schedule_arbitration()

    def _schedule_arbitration(self) -> None:
        if self._arbitration_pending:
            return
        self._arbitration_pending = True
        self.engine.schedule(0, self._arbitrate,
                             label="controller staging arbitration")

    def _arbitrate(self) -> None:
        self._arbitration_pending = False
        kinds = tuple(SyndromePacketRouteKind)
        ordered_kinds = kinds[self._next_route_index:] + kinds[:self._next_route_index]
        progressed = False
        for kind in ordered_kinds:
            route_queue = self._route_queues[kind]
            if not route_queue:
                continue
            slot_index = route_queue[0]
            if self._attempt_head(slot_index):
                progressed = True
                self._next_route_index = (kinds.index(kind) + 1) % len(kinds)

    def _attempt_head(self, slot_index: int) -> bool:
        slot = self._slots[slot_index]
        if slot.state is not _ControllerSlotState.PACKED_WAIT:
            return False
        if slot.route.kind is SyndromePacketRouteKind.WINDOW_INPUT:
            accepted = self.window_input_receiver.accept_window_input(slot.packet)
            if type(accepted) is not bool:
                raise TypeError("window input receiver must return an exact bool")
            if not accepted:
                return False
            slot.state = _ControllerSlotState.DRAINING
            release = lambda: self._release_slot(slot_index)
            self.engine.schedule(
                0, release, label="window input publication complete")
        else:
            slot.state = _ControllerSlotState.DRAINING
            self._transmit_feedback_memory_round(slot_index, slot)
        return True

    def _transmit_feedback_memory_round(
        self, slot_index: int, slot: _StagingSlot,
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
        self._release_slot(slot_index)

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

    def on_endpoint_capacity_changed(self) -> None:
        self._schedule_arbitration()
    def staging_snapshot(self) -> ControllerStagingSnapshot:
        identities_by_state = {
            state: tuple(slot.identity for slot in self._slots
                         if slot is not None and slot.state is state)
            for state in _ControllerSlotState}
        return ControllerStagingSnapshot(
            controller_capacity=self.controller_capacity,
            controller_occupancy=len(self._identity_to_slot),
            free_slot_indices=(
                tuple(sorted(self._free_slot_indices))
                if self.controller_capacity is not None else ()
            ),
            identity_to_slot=tuple(sorted(
                self._identity_to_slot.items(), key=lambda item: item[1])),
            partial_identities=identities_by_state[_ControllerSlotState.PARTIAL],
            packed_wait_identities=identities_by_state[
                _ControllerSlotState.PACKED_WAIT],
            draining_identities=identities_by_state[_ControllerSlotState.DRAINING],
        )

    @staticmethod
    def _collapse_fragments(fragments) -> tuple:
        """Order source fragments, then merge parts from the same patch."""
        collapsed = []
        for fragment in sorted(fragments, key=lambda item: item.fragment_index):
            prior_index = next((
                index for index, prior in enumerate(collapsed)
                if same_stable_identity(prior.patch_id, fragment.patch_id)
            ), None)
            if prior_index is None:
                collapsed.append(fragment)
                continue
            prior = collapsed[prior_index]
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
            collapsed[prior_index] = replace(
                prior,
                bits=bits,
                size_bits=size_bits,
            )
        return tuple(collapsed)

    def relay_instruction(self, decision, deliver: Callable) -> None:
        """Orchestrator -> controller (t_oc) -> chip (t_cq)."""
        attribution = TrafficAttribution(
            operation_id=decision.target_operation_id,
            patch_ids=(),
            window_id=None,
            round_lo=None,
            round_hi=None,
        )

        def at_controller():
            instruction = "release instruction" if decision.releases_operation \
                else "result return instruction"
            self.engine.log("Controller",
                            f"received {instruction} for "
                            f"op#{decision.target_operation_id} from "
                            f"orchestrator (t_oc); forwarding to chip (t_cq)")
            delay = self._reserve(
                LinkPath.CQ, payload_bits=None, attribution=attribution)
            self.engine.schedule(
                delay, lambda: deliver(decision), label="controller->chip")
        delay = self._reserve(
            LinkPath.OC, payload_bits=None, attribution=attribution)
        self.engine.schedule(
            delay, at_controller, label="orchestrator->controller")

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
