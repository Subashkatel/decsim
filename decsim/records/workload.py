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
    """One finite physical history.

    measurement_rounds is the round each measurement lands in: the packet
    schedule a front end declares when its detectors carry no round
    coordinate, held as pairs so the record hashes. Left empty, each
    measurement takes the round its DETECTOR group announces, its time
    coordinate plus one (detector_error_model/detector_formation.py), as
    a tqec lattice-surgery circuit's detectors carry it.
    """

    # Stim's circuit has no hash; equal records hold equal circuits.
    circuit: stim.Circuit = dataclasses.field(hash=False)
    measurement_rounds: tuple = ()

    def __post_init__(self) -> None:
        pairs = _held_pairs(self.measurement_rounds)
        object.__setattr__(self, "measurement_rounds", pairs)
        if pairs:
            return
        _refuse_a_detector_with_no_time(self.circuit)

    def round_count(self) -> int:
        """The history's rounds: its schedule's last, or its last DETECTOR's.

        Read off the circuit, the count is one past the last time
        coordinate, so a post-readout detector layer, as Stim's generated
        memory writes one, is a round of its own; a front end that folds
        it into the last round declares the schedule.
        """
        if self.measurement_rounds:
            schedule = dict(self.measurement_rounds)
            rounds = schedule.values()
            return max(rounds)
        coordinates = self.circuit.get_detector_coordinates()
        times = [coordinate[-1] for coordinate in coordinates.values()]
        last_time = max(times)
        return int(last_time) + 1


@dataclasses.dataclass(frozen=True)
class Workload:
    """What a workload maker hands one run.

    An operation runs its own circuit or is a segment of the workload's one
    physical circuit; decsim never builds a merged circuit, so a shared
    circuit needs every emitting operation's round range. An operation left
    out of round_counts takes GateRounds' counts. physical is a finite
    circuit, four live fragments, or None. physical_error_probability is the
    noise the maker used, None when unstated: a fact a threshold reads.
    patch_streams decodes each patch's life as one stream
    (frontends/patch_lives.py), a lattice-surgery program's decoding.
    """

    operations: tuple
    round_counts: tuple = ()
    physical: Optional[
        Union[FiniteCircuit, circuit_records.RepeatedStimCircuit]
    ] = None
    physical_error_probability: Optional[float] = None
    patch_streams: bool = False

    def __post_init__(self) -> None:
        pairs = _held_pairs(self.round_counts)
        object.__setattr__(self, "round_counts", pairs)


def _refuse_a_detector_with_no_time(circuit: stim.Circuit) -> None:
    """Rounds read off a circuit need a time coordinate on every detector."""
    coordinates = circuit.get_detector_coordinates()
    untimed = []
    for detector, coordinate in coordinates.items():
        if not coordinate:
            untimed.append(detector)
    if coordinates and not untimed:
        return
    raise ValueError(
        "a finite circuit with no measurement_rounds has its rounds read off "
        "its DETECTOR time coordinates, and this one has "
        f"{len(coordinates)} detectors, {len(untimed)} with no coordinate; "
        "declare measurement_rounds"
    )


def _held_pairs(keyed_values) -> tuple:
    """A mapping, or pairs, as a tuple of (key, value) pairs in their order."""
    by_key = dict(keyed_values)
    items = by_key.items()
    return tuple(items)
