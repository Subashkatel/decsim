"""The shots the syndrome source sampled, by the operation that asked.

A sampled shot is the run's input: the detection events the shot's
digest hashes and the circuit a reference decode reads. They are heard
at sampling, since a streaming source forms its circuit only then.
"""

import dataclasses

import stim

import decsim.records.program as program_records


@dataclasses.dataclass(frozen=True)
class SampledShot:
    """One sampled shot: the circuit, and the events it produced."""

    operation_id: int
    circuit: stim.Circuit  # the listener never reads inside it
    detection_events: tuple


class SampledShots:
    """Every shot the source sampled, by the operation it was sampled for."""

    def __init__(self) -> None:
        self.shots_by_operation: dict = {}

    def shot_sampled(
        self, operation: program_records.Operation, detection_events: tuple
    ) -> None:
        """One operation's shot was drawn, with its whole-circuit events."""
        events = tuple(bool(bit) for bit in detection_events)
        shot = SampledShot(
            operation_id=operation.id,
            circuit=operation.circuit,
            detection_events=events,
        )
        self.shots_by_operation[operation.id] = shot
