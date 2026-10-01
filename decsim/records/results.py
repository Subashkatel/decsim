"""The records one run returns: the run and each logical operation.

And what a collect unit's runs return together, its rows and memory.
"""

import dataclasses
from typing import Optional, Protocol


@dataclasses.dataclass(frozen=True)
class LogicalOperationResult:
    """One operation's prediction and, when sampled, its truth.

    logical_failure is true when any predicted bit differs from truth.
    """

    operation_id: int
    result_status: str
    logical_observables: Optional[tuple]
    stream_offset: Optional[int]
    observable_truth: Optional[tuple] = None
    logical_failure: Optional[bool] = None


@dataclasses.dataclass(frozen=True)
class RunResult:
    """What one completed run computed; the gate hashes these fields."""

    terminal_status: str
    event_queue_empty: bool
    decode_work_settled: bool
    execution_workload_complete: bool
    execution_done_ticks: int
    fully_done_ticks: int
    # the ticks operations waited for a magic state, summed: the
    # factories' supply stall
    magic_state_stall_ticks: int
    operation_results: tuple
    link_traffic: dict
    # the copies, references and moves of the data path; None unless the
    # observation section asked for them
    data_movement: Optional[dict]


class UnitTask(Protocol):
    """The task a unit ran, as an outcome carries it: the point it is.

    decsim.collect.Task is one. Records import no component, so the
    task is typed by what it offers and not by its class.
    """

    def strong_id(self) -> str:
        """The point's id, a hash of the settings it resolved to."""


@dataclasses.dataclass(frozen=True)
class UnitOutcome:
    """What one unit ran: its rows, its task, its memory, its packages.

    task is the task as the unit left it, its online calibrator after
    the unit's shots. peak_memory_mb is the peak resident memory of the
    process that ran the unit, read when the unit ended. A worker runs
    units one after another, so it bounds the unit's own peak from
    above, which is the side a memory request needs. module_versions
    maps each third-party top-level module that process had imported
    when the unit ended to its version, read there because a decoder's
    package loads in the process that decodes.
    """

    rows: list
    task: UnitTask
    peak_memory_mb: float
    module_versions: dict
