"""The shots the syndrome source sampled, by the operation that asked.

A listener on the SyndromeSource port's shot_sampled source. A sampled
shot is the run's input, not its output: the detection events the device
drew and the circuit they came from are what a whole-circuit reference
decode reads to check the loop's answer (front/measure.py), so they are
heard once at sampling time instead of read back off the device.
"""

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class SampledShot:
    """One sampled shot: the circuit, and the events it produced."""

    operation_id: Any
    circuit: Any  # a stim.Circuit; the listener never reads inside it
    detection_events: tuple


class SampledShots:
    """Every shot the source sampled, by the operation it was sampled for."""

    def __init__(self) -> None:
        self.shots_by_operation: dict = {}

    def shot_sampled(self, operation, detection_events) -> None:
        """One operation's shot was drawn, with its whole-circuit events."""
        events = tuple(bool(bit) for bit in detection_events)
        shot = SampledShot(
            operation_id=operation.id,
            circuit=operation.circuit,
            detection_events=events,
        )
        self.shots_by_operation[operation.id] = shot
