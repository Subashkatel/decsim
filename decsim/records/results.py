"""The records one run returns: the run and each logical operation.

And what a collect unit's runs return together, its rows and memory.
"""

import dataclasses
from typing import Any, Optional


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


@dataclasses.dataclass(frozen=True)
class UnitOutcome:
    """What one unit ran: its measured rows, its task, its memory.

    task is the decsim.collect.Task the unit ran. peak_memory_mb is the
    peak resident memory of the process that ran the unit, read when the
    unit ended. A worker runs units one after another, so it bounds the
    unit's own peak from above, which is the side a memory request needs.
    """

    rows: list
    task: Any
    peak_memory_mb: float
