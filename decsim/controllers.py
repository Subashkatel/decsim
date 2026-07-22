"""Controller relay (port 14): the classical hop chains between components.

Part module: ModularController delivery chains — qc->cd inbound
with fragment buffering + t_pack; oc->cq decision path. The named links
live in links.py.
"""

from __future__ import annotations

from typing import Callable, Optional

from .links import LinkModel


class ModularController:
    """Relays payloads and decisions across the named classical links."""

    def __init__(self, engine, links: Optional[LinkModel] = None,
                 t_pack: int = 0, log_syndromes: bool = True):
        self.engine = engine
        self.links = links if links is not None else LinkModel()
        self.t_pack = t_pack
        self.log_syndromes = log_syndromes
        self._pending: dict = {}

    # ------------------------------------------------------- syndrome path

    def relay_syndrome(self, payload, deliver: Callable) -> None:
        """Chip -> controller (t_qc) -> decoder (t_cd), buffering fragments."""
        if payload.n_fragments == 1:
            self.engine.schedule(
                self._cost("qc", bits=payload.size_bits),
                lambda: self._forward_whole_round(payload, deliver),
                label="chip->controller")
            return
        self.engine.schedule(self._cost("qc", bits=payload.size_bits),
                             lambda: self._receive_fragment(payload, deliver),
                             label="chip->controller")

    def _forward_whole_round(self, payload, deliver: Callable) -> None:
        if self.log_syndromes:
            self.engine.log("Controller",
                            f"received round {payload.round_index} of "
                            f"op#{payload.operation_id} from chip (t_qc); "
                            f"forwarding to decoder (t_cd)")
        self.engine.schedule(self._cost("cd", bits=payload.size_bits),
                             lambda: deliver(payload),
                             label="controller->decoder")

    def _receive_fragment(self, payload, deliver: Callable) -> None:
        round_key = (payload.operation_id, payload.round_index)
        fragments = self._pending.setdefault(round_key, [])
        fragments.append((payload, deliver))
        if len(fragments) < payload.n_fragments:
            if self.log_syndromes:
                self.engine.log("Controller",
                                f"buffered fragment {len(fragments)}/"
                                f"{payload.n_fragments} of round "
                                f"{payload.round_index} of "
                                f"op#{payload.operation_id} (waiting for the rest)")
            return
        del self._pending[round_key]
        if self.log_syndromes:
            self.engine.log("Controller",
                            f"round {payload.round_index} of "
                            f"op#{payload.operation_id} complete "
                            f"({payload.n_fragments} fragments); packaging and "
                            f"forwarding one packet to decoder (t_pack + t_cd)")
        packet = tuple(fragments)
        fragment_sizes = [fragment.size_bits for fragment, _ in packet]
        packet_bits = sum(fragment_sizes) \
            if all(size is not None for size in fragment_sizes) else None
        if self.t_pack:
            # Packing happens off the wire: elapse t_pack as its own event, then
            # arbitrate the cd bus at the real transmit time. Pricing the wire at
            # now+t_pack while a single next_free_tick tracks the bus cannot model
            # a future reservation -- it wrongly blocks (and reorders) traffic
            # that is ready during the [now, now+t_pack] packing gap.
            self.engine.schedule(
                self.t_pack,
                lambda: self._transmit_fragment_packet(packet, packet_bits),
                label="controller pack")
        else:
            self._transmit_fragment_packet(packet, packet_bits)

    def _transmit_fragment_packet(self, packet: tuple, packet_bits) -> None:
        """Send a packed round over the cd wire, arbitrating the bus at now."""
        self.engine.schedule(self._cost("cd", bits=packet_bits),
                             lambda: self._deliver_fragment_packet(packet),
                             label="controller->decoder packet")

    @staticmethod
    def _deliver_fragment_packet(fragments: tuple) -> None:
        for payload, deliver in fragments:
            deliver(payload)

    # ------------------------------------------------------- decision path

    def relay_instruction(self, decision, deliver: Callable) -> None:
        """Orchestrator -> controller (t_oc) -> chip (t_cq)."""
        def at_controller():
            instruction = "release instruction" if decision.releases_operation \
                else "result return instruction"
            self.engine.log("Controller",
                            f"received {instruction} for "
                            f"op#{decision.target_operation_id} from "
                            f"orchestrator (t_oc); forwarding to chip (t_cq)")
            self.engine.schedule(self._cost("cq"),
                                 lambda: deliver(decision),
                                 label="controller->chip")
        self.engine.schedule(self._cost("oc"), at_controller,
                             label="orchestrator->controller")

    # -------------------------------------------------- generic port surface

    def send(self, edge: str, payload, deliver: Callable, now: int) -> None:
        """Generic Transport.send over a named edge (port 14)."""
        self.engine.schedule(self._cost(
                                 edge, bits=getattr(payload, "size_bits", None),
                                 now=now),
                             lambda: deliver(payload), label=f"send:{edge}")

    def _cost(self, edge: str, *, bits=None, now=None) -> int:
        """Price one transmission without dropping bandwidth/queueing inputs."""
        link = getattr(self.links, edge)
        return link.cost(bits=bits, now=self.engine.now if now is None else now)
