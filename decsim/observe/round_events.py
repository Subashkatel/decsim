"""The flight recorder of the readout path: what happened to every round.

A listener the controller, the assembler, the writer, the transmitter
and the strong writer call; they never read it. The rows are append-only
and passive; recording never schedules or decides. A finished run's
terminal states are PUBLISHED, DROPPED or FEEDBACK_MEMORY_DELIVERED;
the run ledger (observe/run_views.py) is built from these rows, the
controller's output events and the strong store's landings. It also
narrates Buffer 0's intake on the I/O trace, the line the engine prints
when the observation section asks for component I/O. A component built
with NoRoundEvents records nothing.
"""

import decsim.message as message


class RoundEventRecorder:
    """The rows: round events, controller outputs, strong-store landings."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.events: list = []
        self.output_events: list = []
        # (tick, operation_id, round_index) per strong-store landing
        self.stored_rounds: list = []
        self.packing_drops = 0
        # counted by the assembler's reassembly timeout; the results
        # projection reads it until structural D retires both
        self.reassembly_timeouts = 0

    def record(
        self,
        kind: str,
        operation_id,
        round_index: int,
        route: message.SyndromePacketRoute,
        *,
        patch_id=None,
        tick=None,
    ) -> None:
        """One transition of one round, now unless the tick is given."""
        if tick is None:
            tick = self.engine.now
        event = message.RoundEvent(
            kind, tick, operation_id, round_index, patch_id, route.kind.name
        )
        self.events.append(event)

    def round_dropped(
        self,
        operation_id,
        round_index: int,
        route: message.SyndromePacketRoute,
        *,
        patch_id=None,
    ) -> None:
        """A round left the controller unstored; DROPPED is its terminal."""
        self.packing_drops += 1
        self.record(
            "DROPPED", operation_id, round_index, route, patch_id=patch_id
        )

    def round_stored(self, operation_id, round_index: int) -> None:
        """A round landed in the strong store."""
        self.stored_rounds.append((self.engine.now, operation_id, round_index))

    def reassembly_timed_out(self) -> None:
        """An incomplete round exceeded the reassembly timeout."""
        self.reassembly_timeouts += 1

    def output(self, kind: str, operation_id, payload) -> None:
        """One transition on the controller's digital-to-QPU path."""
        event = message.ControllerOutputEvent(
            kind, self.engine.now, operation_id, payload
        )
        self.output_events.append(event)

    def weak_store_received(
        self, packet: message.SyndromeRoundPacket, weak_store
    ) -> None:
        """Narrate Buffer 0's intake on the I/O trace."""
        self.engine.log_io(
            "Buffer 0", lambda: _received_text(packet, weak_store)
        )


class NoRoundEvents:
    """A recorder that records nothing, for a component run alone."""

    packing_drops = 0
    reassembly_timeouts = 0

    def record(self, kind, operation_id, round_index, route, **fields) -> None:
        """Nothing to keep."""
        del kind, operation_id, round_index, route, fields

    def round_dropped(self, operation_id, round_index, route, **fields) -> None:
        """Nothing to keep."""
        del operation_id, round_index, route, fields

    def round_stored(self, operation_id, round_index) -> None:
        """Nothing to keep."""
        del operation_id, round_index

    def reassembly_timed_out(self) -> None:
        """Nothing to keep."""

    def output(self, kind, operation_id, payload) -> None:
        """Nothing to keep."""
        del kind, operation_id, payload

    def weak_store_received(self, packet, weak_store) -> None:
        """Nothing to narrate."""
        del packet, weak_store


def _received_text(packet: message.SyndromeRoundPacket, weak_store) -> str:
    defects = packet.defects_text()
    holds = weak_store.held_rounds_description()
    return (
        f"received round {packet.round_index} of "
        f"op {packet.operation_id} from packing; {defects}; holds {holds}"
    )
