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
        # (tick, stream_id, round_index) per strong-store landing
        self.stored_rounds: list = []
        # the tick of every round the QPU read out, an idle patch's
        # included, before any departure delay the source states
        self.readout_ticks: list = []
        # the last round the QPU read out of each operation, its rounds
        # generated so far, since it reads an operation's rounds in order
        self.rounds_read_out_by_stream: dict = {}

    def record(self, event: round_records.RoundEvent) -> None:
        """One transition of one round."""
        self.events.append(event)

    def output(self, event: round_records.ControllerOutputEvent) -> None:
        """One transition on the controller's digital-to-QPU path."""
        self.output_events.append(event)

    def round_read_out(self, readout: round_records.QPUReadout) -> None:
        """The QPU read out one round, or one fragment of it."""
        self.note_readout_tick()
        by_stream = self.rounds_read_out_by_stream
        latest = by_stream.get(readout.stream_id, 0)
        by_stream[readout.stream_id] = max(latest, readout.round_index)

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
        self.note_readout_tick()

    def round_stored(
        self, round_key: tuple, _packet: round_records.SyndromeRoundPacket
    ) -> None:
        """A round landed in the strong store."""
        stream_id, round_index = round_key
        self.stored_rounds.append((self.engine.now, stream_id, round_index))

    def note_readout_tick(self) -> None:
        """The tick the QPU read a round out, an idle patch's included."""
        self.readout_ticks.append(self.engine.now)


class FoldingRoundEventRecorder(RoundEventRecorder):
    """A stream's recorder: no row kept, the rounds read out counted.

    A stream measures as it goes (observation.folds_windows,
    experiments/stream_measures.py), so a row kept per round would only
    grow; the rounds read out of each operation are what the backlog
    sampler reads (observe/metrics.py).
    """

    def record(self, event: round_records.RoundEvent) -> None:
        """A round's transition, kept by nobody in a stream."""
        del event

    def output(self, event: round_records.ControllerOutputEvent) -> None:
        """An output transition, kept by nobody in a stream."""
        del event

    def round_stored(
        self, round_key: tuple, _packet: round_records.SyndromeRoundPacket
    ) -> None:
        """A strong-store landing, kept by nobody in a stream."""
        del round_key

    def note_readout_tick(self) -> None:
        """The stream's measures keep the generation span's two ends."""
