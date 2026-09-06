"""The flight recorder of the readout path: what happened to every round.

A listener on the round_event source of the controller's intake, the
assembler, the held rounds, the writer and the transmitter, on the
instruction output's output_event, and on the strong store's
round_stored; the components never read it. The rows are append-only
and passive; recording never schedules or decides. A finished run's
terminal states are PUBLISHED, DROPPED or FEEDBACK_MEMORY_DELIVERED;
the run ledger (observe/run_views.py) is built from these rows, the
controller's output events and the strong store's landings.
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

    def record(self, event: message.RoundEvent) -> None:
        """One transition of one round; a DROPPED one is counted."""
        self.events.append(event)
        if event.kind == "DROPPED":
            self.packing_drops += 1

    def output(self, event: message.ControllerOutputEvent) -> None:
        """One transition on the controller's digital-to-QPU path."""
        self.output_events.append(event)

    def round_stored(self, round_key) -> None:
        """A round landed in the strong store."""
        operation_id, round_index = round_key
        self.stored_rounds.append((self.engine.now, operation_id, round_index))
