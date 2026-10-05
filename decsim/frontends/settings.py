"""The workload settings: what the machine runs, and for how many rounds.

A Python caller hands a maker's records.workload.Workload to
WorkloadSettings.running, which lowers it into the operations the
machine issues (circuit_frontend.lowered), or hands the lowered fields
in directly.
"""

import dataclasses
from typing import Optional

import stim

import decsim.frontends.circuit_frontend as circuit_frontend
import decsim.ports as ports
import decsim.records.workload as workload_records

FEEDBACK_BOUNDARY_MODES = ("trailing_buffer", "measurement_closed")


@dataclasses.dataclass(frozen=True)
class WorkloadSettings:
    """The program the machine runs: a maker's workload, lowered."""

    operations: tuple = ()
    decode_operations: Optional[tuple] = None
    dynamic_streams: tuple = ()
    protected_regions: tuple = ()
    rounds_policy: Optional[ports.RoundsPolicy] = None
    physical_circuits: tuple = ()
    feedback_boundary_mode: str = "trailing_buffer"
    # the record the fields above were lowered from, which a run folder
    # writes to its inputs (experiments/run_folder.py record_task)
    workload_record: Optional[workload_records.Workload] = None

    def __post_init__(self) -> None:
        if self.feedback_boundary_mode not in FEEDBACK_BOUNDARY_MODES:
            raise ValueError(
                "workload.feedback_boundary_mode must be one of "
                f"{FEEDBACK_BOUNDARY_MODES}, got "
                f"{self.feedback_boundary_mode!r}"
            )

    @classmethod
    def running(cls, workload: workload_records.Workload) -> "WorkloadSettings":
        """The settings that run a maker's workload, lowered for the machine.

        A Python caller states its workload here:
        WorkloadSettings.running(producers.memory_circuit(...)).
        """
        lowered = _lowered_fields(workload)
        return cls(**lowered)


def memory_circuit(
    code_task: str, rounds: int, distance: int, probability: float
) -> stim.Circuit:
    """Stim's generated memory circuit.

    One probability on all four noise channels, as Stim's guide does.
    """
    return stim.Circuit.generated(
        code_task,
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=probability,
        before_round_data_depolarization=probability,
        before_measure_flip_probability=probability,
        after_reset_flip_probability=probability,
    )


def _lowered_fields(workload: workload_records.Workload) -> dict:
    """The fields a maker's workload fills, lowered for the machine."""
    program = circuit_frontend.lowered(workload)
    return {
        "operations": program.operations,
        "dynamic_streams": program.dynamic_streams,
        "protected_regions": program.protected_regions,
        "rounds_policy": program.rounds_policy,
        "physical_circuits": program.physical_circuits,
        "workload_record": workload,
    }
