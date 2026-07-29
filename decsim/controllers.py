"""Controller relay (port 14): the classical hop chains between components.

Part module: ModularController delivery chains — qc->cd inbound
with fragment buffering + t_pack; oc->cq decision path. The named links
live in links.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .links import (
    LinkModel,
    LinkModelConfig,
    LinkPath,
    TrafficAttribution,
)
from .message import (
    RetainedSyndromeFragment,
    SyndromeRoundPacket,
    same_stable_identity,
    stable_identity_order_key,
)


def _delivery_sink_identity(deliver: Callable) -> tuple:
    if not callable(deliver):
        raise TypeError("syndrome delivery sink must be callable")
    receiver = getattr(deliver, "__self__", None)
    function = getattr(deliver, "__func__", None)
    if receiver is not None and function is not None:
        return ("bound", receiver, function)
    return ("callable", deliver)


def _same_delivery_sink(left: tuple, right: tuple) -> bool:
    return (
        len(left) == len(right)
        and left[0] == right[0]
        and all(left_item is right_item
                for left_item, right_item in zip(left[1:], right[1:]))
    )


@dataclass
class _PendingSyndromeRound:
    fragment_count: int
    sink_identity: tuple
    deliver: Callable
    fragments: list[RetainedSyndromeFragment] = field(default_factory=list)


class ModularController:
    """Relays payloads and decisions across the named classical links."""

    def __init__(self, engine, links: Optional[LinkModel] = None,
                 t_pack: int = 0, log_syndromes: bool = True):
        self.engine = engine
        self.links = (
            links
            if links is not None
            else LinkModelConfig.reference_fixed_latency_profile().resolve()
        )
        self.t_pack = t_pack
        self.log_syndromes = log_syndromes
        self._pending: dict = {}
        self._completed_rounds: set = set()

    # ------------------------------------------------------- syndrome path

    def relay_syndrome(self, payload, deliver: Callable) -> None:
        """Chip -> controller (t_qc) -> decoder (t_cd), buffering fragments."""
        if type(payload.n_fragments) is not int:
            raise TypeError("n_fragments must be an exact built-in int")
        if payload.n_fragments < 1:
            raise ValueError("n_fragments must be at least one")
        fragment_count = payload.n_fragments
        fragment = RetainedSyndromeFragment.from_payload(payload)
        sink_identity = _delivery_sink_identity(deliver)
        attribution = self._round_attribution(
            fragment.operation_id,
            (fragment.patch_id,),
            fragment.round_index,
        )
        self.engine.schedule(self._reserve(
                                 LinkPath.QC,
                                 payload_bits=payload.size_bits,
                                 attribution=attribution),
                             lambda: self._receive_fragment(
                                 fragment,
                                 fragment_count,
                                 sink_identity,
                                 deliver,
                             ),
                             label="chip->controller")

    def _receive_fragment(
        self,
        fragment: RetainedSyndromeFragment,
        fragment_count: int,
        sink_identity: tuple,
        deliver: Callable,
    ) -> None:
        round_key = (fragment.operation_id, fragment.round_index)
        if round_key in self._completed_rounds:
            raise ValueError(f"syndrome round {round_key!r} already completed")
        pending = self._pending.get(round_key)
        if pending is None:
            pending = _PendingSyndromeRound(
                fragment_count=fragment_count,
                sink_identity=sink_identity,
                deliver=deliver,
            )
        else:
            if fragment_count != pending.fragment_count:
                raise ValueError("all fragments must declare the same count")
            if not _same_delivery_sink(sink_identity, pending.sink_identity):
                raise ValueError("all fragments must share one delivery sink")
        if any(
            same_stable_identity(fragment.patch_id, candidate.patch_id)
            for candidate in pending.fragments
        ):
            raise ValueError("duplicate syndrome patch identity")
        if len(pending.fragments) >= pending.fragment_count:
            raise ValueError("too many distinct syndrome fragments")

        if round_key not in self._pending:
            self._pending[round_key] = pending
        pending.fragments.append(fragment)
        if len(pending.fragments) < pending.fragment_count:
            if self.log_syndromes:
                self.engine.log("Controller",
                                f"buffered fragment {len(pending.fragments)}/"
                                f"{pending.fragment_count} of round "
                                f"{fragment.round_index} of "
                                f"op#{fragment.operation_id} (waiting for the rest)")
            return
        packet = SyndromeRoundPacket(
            operation_id=fragment.operation_id,
            round_index=fragment.round_index,
            fragments=tuple(pending.fragments),
        )
        del self._pending[round_key]
        self._completed_rounds.add(round_key)
        if self.log_syndromes:
            self.engine.log("Controller",
                            f"round {fragment.round_index} of "
                            f"op#{fragment.operation_id} complete "
                            f"({pending.fragment_count} fragments); forwarding "
                            f"one packet to decoder")
        fragment_sizes = [item.size_bits for item in packet.fragments]
        packet_bits = sum(fragment_sizes) \
            if all(size is not None for size in fragment_sizes) else None
        if pending.fragment_count > 1 and self.t_pack:
            # Packing happens off the wire: elapse t_pack as its own event, then
            # arbitrate the cd bus at the real transmit time. Pricing the wire at
            # now+t_pack while a single next_free_tick tracks the bus cannot model
            # a future reservation -- it wrongly blocks (and reorders) traffic
            # that is ready during the [now, now+t_pack] packing gap.
            self.engine.schedule(
                self.t_pack,
                lambda: self._transmit_round_packet(
                    packet,
                    packet_bits,
                    pending.deliver,
                ),
                label="controller pack")
        else:
            self._transmit_round_packet(packet, packet_bits, pending.deliver)

    def _transmit_round_packet(
        self,
        packet: SyndromeRoundPacket,
        packet_bits,
        deliver: Callable,
    ) -> None:
        """Send a packed round over the cd wire, arbitrating the bus at now."""
        attribution = self._round_attribution(
            packet.operation_id,
            tuple(fragment.patch_id for fragment in packet.fragments),
            packet.round_index,
        )
        self.engine.schedule(self._reserve(
                                 LinkPath.CWD,
                                 payload_bits=packet_bits,
                                 attribution=attribution),
                             lambda: deliver(packet),
                             label="controller->decoder packet")

    # ------------------------------------------------------- decision path

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
            self.engine.schedule(self._reserve(
                                     LinkPath.CQ,
                                     payload_bits=None,
                                     attribution=attribution),
                                 lambda: deliver(decision),
                                 label="controller->chip")
        self.engine.schedule(self._reserve(
                                 LinkPath.OC,
                                 payload_bits=None,
                                 attribution=attribution), at_controller,
                             label="orchestrator->controller")

    # -------------------------------------------------- generic port surface

    def send(
        self,
        path: LinkPath,
        payload,
        deliver: Callable,
        now: int,
        attribution: TrafficAttribution,
    ) -> None:
        """Generic Transport.send over a named edge (port 14)."""
        if now != self.engine.now:
            raise ValueError("controller sends reserve at the current engine tick")
        self.engine.schedule(
            self._reserve(
                path,
                payload_bits=getattr(payload, "size_bits", None),
                attribution=attribution,
            ),
            lambda: deliver(payload),
            label=f"send:{path.value}",
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
    def _round_attribution(
        operation_id,
        patch_ids: tuple,
        round_index: int,
    ) -> TrafficAttribution:
        return TrafficAttribution(
            operation_id=operation_id,
            patch_ids=tuple(sorted(patch_ids, key=stable_identity_order_key)),
            window_id=None,
            round_lo=round_index,
            round_hi=round_index,
        )
