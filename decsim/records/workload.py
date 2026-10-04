"""What a workload maker hands decsim: an operation list and its circuit.

As gem5 separates a workload's artifacts from the board that runs them
(src/python/gem5/components/boards/se_binary_workload.py:225-250).
"""

import dataclasses
from typing import Optional, Union

import stim

import decsim.records.circuits as circuit_records


@dataclasses.dataclass(frozen=True)
class FiniteCircuit:
    """One finite physical history and the round each measurement lands in.

    measurement_rounds is the packet schedule a front end declares when its
    detectors carry no round coordinate, held as pairs so the record hashes.
    """

    # Stim's circuit has no hash; equal records hold equal circuits.
    circuit: stim.Circuit = dataclasses.field(hash=False)
    measurement_rounds: tuple

    def __post_init__(self) -> None:
        pairs = _held_pairs(self.measurement_rounds)
        object.__setattr__(self, "measurement_rounds", pairs)


@dataclasses.dataclass(frozen=True)
class Workload:
    """The operations a run issues, their rounds, and their physical circuit.

    An operation runs its own circuit or is a segment of the workload's one
    physical circuit; decsim never builds a merged circuit, so a shared
    circuit needs every emitting operation's round range. An operation left
    out of round_counts takes GateRounds' counts. physical is a finite
    circuit, four live fragments, or None. physical_error_probability is the
    noise the maker used, None when unstated: a fact a threshold reads.
    """

    operations: tuple
    round_counts: tuple = ()
    physical: Optional[
        Union[FiniteCircuit, circuit_records.RepeatedStimCircuit]
    ] = None
    physical_error_probability: Optional[float] = None

    def __post_init__(self) -> None:
        pairs = _held_pairs(self.round_counts)
        object.__setattr__(self, "round_counts", pairs)


def _held_pairs(keyed_values) -> tuple:
    """A mapping, or pairs, as a tuple of (key, value) pairs in their order."""
    by_key = dict(keyed_values)
    items = by_key.items()
    return tuple(items)
