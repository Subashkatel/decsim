"""Controller-side transport for syndrome rounds and feedback decisions."""


from __future__ import annotations

from typing import Callable, Optional

from .config import us
from .engine import Engine
from .links import LinkModel
from .message import Decision, SyndromePayload


class ModularController:
    """Default controller implementation with optional round-fragment buffering."""

    def __init__(self, engine: Engine, t_qc=us(0.15), t_cd=us(2.0), t_dd=us(0.5),
                 t_do=us(1.0), t_oc=us(4.0), t_cq=us(0.15), log_syndromes=True,
                 t_pack=0, links: Optional[LinkModel] = None):
        self.engine = engine
        self.links = links if links is not None \
            else LinkModel(qc=t_qc, cd=t_cd, dd=t_dd, do=t_do, oc=t_oc, cq=t_cq)
        self.t_pack = t_pack
        self.log_syndromes = log_syndromes
        self._pending: dict = {}

    def relay_syndrome(self, payload: SyndromePayload,
                       deliver: Callable[[SyndromePayload], None]) -> None:
        """Send syndrome data from the chip to the decoder cluster."""
        if payload.n_fragments == 1:
            self._schedule_whole_round(payload, deliver)
            return

        self.engine.schedule(self.links.qc.cost(),
                             lambda: self._receive_fragment(payload, deliver),
                             label="chip->controller")

    def _schedule_whole_round(self, payload: SyndromePayload,
                              deliver: Callable[[SyndromePayload], None]) -> None:
        """Forward a complete round after chip-controller and controller-decoder hops."""
        self.engine.schedule(
            self.links.qc.cost(),
            lambda: self._forward_whole_round(payload, deliver),
            label="chip->controller")

    def _forward_whole_round(self, payload: SyndromePayload,
                             deliver: Callable[[SyndromePayload], None]) -> None:
        """Forward a whole-round payload from the controller to the decoder."""
        if self.log_syndromes:
            self.engine.log("Controller",
                            f"received round {payload.round_index} of "
                            f"op#{payload.operation_id} from chip (t_qc); "
                            f"forwarding to decoder (t_cd)")
        self.engine.schedule(self.links.cd.cost(),
                             lambda: deliver(payload),
                             label="controller->decoder")

    def _receive_fragment(self, payload: SyndromePayload,
                          deliver: Callable[[SyndromePayload], None]) -> None:
        """Buffer a fragment and ship the complete round when all fragments arrive."""
        round_key = (payload.operation_id, payload.round_index)
        fragments = self._pending.setdefault(round_key, [])
        fragments.append((payload, deliver))

        if len(fragments) < payload.n_fragments:
            self._log_waiting_for_fragments(payload, len(fragments))
            return

        del self._pending[round_key]
        self._schedule_fragment_packet(payload, tuple(fragments))

    def _log_waiting_for_fragments(self, payload: SyndromePayload, count: int) -> None:
        """Log that a fragmented round is not complete yet."""
        if not self.log_syndromes:
            return
        self.engine.log("Controller",
                        f"buffered fragment {count}/{payload.n_fragments} "
                        f"of round {payload.round_index} of "
                        f"op#{payload.operation_id} (waiting for the rest)")

    def _schedule_fragment_packet(self, payload: SyndromePayload,
                                  fragments: tuple) -> None:
        """Ship all fragments as one controller-decoder packet."""
        if self.log_syndromes:
            self.engine.log("Controller",
                            f"round {payload.round_index} of op#{payload.operation_id} "
                            f"complete ({payload.n_fragments} fragments); packaging "
                            f"and forwarding one packet to decoder (t_pack + t_cd)")
        self.engine.schedule(
            self.t_pack + self.links.cd.cost(),
            lambda: self._deliver_fragment_packet(fragments),
            label="controller->decoder packet")

    @staticmethod
    def _deliver_fragment_packet(fragments: tuple) -> None:
        """Deliver every payload in a completed fragmented round."""
        for payload, deliver in fragments:
            deliver(payload)

    def relay_instruction(self, decision: "Decision",
                          deliver: Callable[["Decision"], None]) -> None:
        """Send a correction orchestrator->controller->chip, delivering after the hop delays."""
        def at_controller():
            self.engine.log("Controller",
                            f"received instruction for op#{decision.gadget_id} from "
                            f"orchestrator (t_oc); forwarding to chip (t_cq)")
            self.engine.schedule(self.links.cq.cost(), lambda: deliver(decision),
                                 label="controller->chip")
        self.engine.schedule(self.links.oc.cost(), at_controller,
                             label="orchestrator->controller")
