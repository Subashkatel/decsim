"""What a workload maker hands decsim: an operation list and its circuit.

Every maker, Stim's generator, Deltakit or a producer outside decsim,
returns these two things and nothing else, as gem5 separates a
workload's artifacts from the board that runs them
(gem5 src/python/gem5/components/boards/se_binary_workload.py:225-250):
the physical circuit the QPU executes, and the operations the
controller issues over patches.
"""

import dataclasses
from typing import Optional, Union

import stim

import decsim.records.circuits as circuit_records


@dataclasses.dataclass(frozen=True)
class FiniteCircuit:
    """One finite physical history and the round each measurement lands in.

    measurement_rounds pairs every absolute measurement index with its
    one-based round, the packet schedule a front end declares when its
    detectors carry no round coordinate (detector_formation.py
    build_formation_table). A mapping given is held as its pairs, in its
    order, so the record hashes.
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

    operations are program_records.Operation records in program order.
    An operation runs its own circuit, or is a segment (stream_id,
    stream_offset) of the one physical circuit the workload carries;
    decsim derives the stream's owner from the segments and never builds
    a merged circuit, so a circuit shared by several operations needs
    every emitting operation's round range. round_counts pairs an
    operation's id with its rounds; one left out takes the lattice-surgery
    counts of round_policies.GateRounds, and a mapping given is held as
    its pairs, in its order. physical is a finite circuit, the four live
    fragments of a repeated memory, or None when every operation carries
    its own circuit or the run is timing only.
    """

    operations: tuple
    round_counts: tuple = ()
    physical: Optional[
        Union[FiniteCircuit, circuit_records.RepeatedStimCircuit]
    ] = None

    def __post_init__(self) -> None:
        pairs = _held_pairs(self.round_counts)
        object.__setattr__(self, "round_counts", pairs)


def _held_pairs(keyed_values) -> tuple:
    """A mapping, or pairs, as a tuple of (key, value) pairs in their order."""
    by_key = dict(keyed_values)
    items = by_key.items()
    return tuple(items)
