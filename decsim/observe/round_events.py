"""The flight recorder of the readout path: what happened to every round.

Append-only and passive; the components never read it. A finished run's
terminal states are PUBLISHED or FEEDBACK_MEMORY_DELIVERED. The
controller's output events are read only by the tests' run ledger
(tests/observe/run_ledger.py), and are kept here because output_event
has no other listener.
"""

from typing import Any

import decsim.engine as engine_module
import decsim.records.rounds as round_records


class RoundEventRecorder:
    """The rows: round events, controller outputs, strong-store landings."""

    def __init__(self, engine: engine_module.Engine) -> None:
        self.engine = engine
        self.events: list = []
        self.output_events: list = []
        # (tick, operation_id, round_index) per strong-store landing
        self.stored_rounds: list = []
        # the tick of every round the QPU read out, an idle patch's
        # included, before any departure delay the source states
        self.readout_ticks: list = []
        # the last round the QPU read out of each operation, its rounds
        # generated so far, since it reads an operation's rounds in order
        self.rounds_read_out_by_operation: dict = {}

    def record(self, event: round_records.RoundEvent) -> None:
        """One transition of one round."""
        self.events.append(event)

    def output(self, event: round_records.ControllerOutputEvent) -> None:
        """One transition on the controller's digital-to-QPU path."""
        self.output_events.append(event)

    def round_read_out(self, readout: round_records.QPUReadout) -> None:
        """The QPU read out one round, or one fragment of it."""
        self.readout_ticks.append(self.engine.now)
        by_operation = self.rounds_read_out_by_operation
        latest = by_operation.get(readout.operation_id, 0)
        by_operation[readout.operation_id] = max(latest, readout.round_index)

    def idle_round_read_out(
        self,
        operation_id: Any,  # an opaque identity
        patch: Any,  # an opaque identity
        round_index: int,
    ) -> None:
        """An idle patch's round: generated work, though no operation's.

        Its decodes, when the idle policy charges any, are load-only, so
        it stretches the span the rounds were generated over and is no
        operation's backlog.
        """
        del operation_id, patch, round_index
        self.readout_ticks.append(self.engine.now)

    def round_stored(
        self, round_key: tuple, _packet: round_records.SyndromeRoundPacket
    ) -> None:
        """A round landed in the strong store."""
        operation_id, round_index = round_key
        self.stored_rounds.append((self.engine.now, operation_id, round_index))
