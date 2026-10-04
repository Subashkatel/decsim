"""The flight recorder of the readout path: what happened to every round.

Append-only and passive; the components never read it. A finished run's
terminal states are PUBLISHED or FEEDBACK_MEMORY_DELIVERED. The
controller's output events are read only by the tests' run ledger
(tests/observe/run_ledger.py), and are kept here because output_event
has no other listener.
"""

import decsim.records.rounds as round_records


class RoundEventRecorder:
    """The rows: round events, controller outputs, strong-store landings."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.events: list = []
        self.output_events: list = []
        # (tick, operation_id, round_index) per strong-store landing
        self.stored_rounds: list = []

    def record(self, event: round_records.RoundEvent) -> None:
        """One transition of one round."""
        self.events.append(event)

    def output(self, event: round_records.ControllerOutputEvent) -> None:
        """One transition on the controller's digital-to-QPU path."""
        self.output_events.append(event)

    def round_stored(self, round_key, _packet) -> None:
        """A round landed in the strong store."""
        operation_id, round_index = round_key
        self.stored_rounds.append((self.engine.now, operation_id, round_index))
