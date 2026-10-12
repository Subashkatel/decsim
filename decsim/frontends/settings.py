"""The workload settings: what the machine runs, and for how many rounds.

A Python caller hands a maker's records.workload.Workload to
WorkloadSettings.running, which lowers it into the operations the
machine issues (circuit_frontend.lowered), or hands the lowered fields
in directly. A workload lowered here was made at the distance the run
file gave its maker, so it fixes the qpu's distance; a maker row
(decsim/producers.py) is lowered at the task instead and follows it.
"""

import dataclasses
from typing import ClassVar, Optional

import stim

import decsim.frontends.circuit_frontend as circuit_frontend
import decsim.ports as ports
import decsim.records.workload as workload_records

FEEDBACK_BOUNDARY_MODES = ("trailing_buffer", "measurement_closed")


@dataclasses.dataclass(frozen=True)
class WorkloadSettings:
    """The program the machine runs: a maker's workload, lowered.

    fixes_the_distance and fields_set_in_the_run_file name what an
    assignment by path may not change (decsim/settings.py): the circuit
    was made at the qpu's distance and lowered to these fields.
    """

    fixes_the_distance: ClassVar[bool] = True
    fields_set_in_the_run_file: ClassVar[tuple] = (
        "operations",
        "decode_operations",
        "dynamic_streams",
        "protected_regions",
        "physical_circuits",
        "rounds_policy",
        "workload_record",
    )

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

    def lowered_for(self, qpu: object) -> "WorkloadSettings":
        """These settings: they were lowered when they were made."""
        del qpu
        return self


@dataclasses.dataclass(frozen=True)
class StimNoise:
    """Stim's four generated noise channels, each its own probability.

    The arguments of stim.Circuit.generated (Stim 1.16 docstring): the
    depolarizing after each Clifford gate, on each data qubit before a
    round, the flip before each measurement and after each reset. Each
    is 0 by default, and a channel at 0 is left out of the circuit.
    """

    after_clifford_depolarization: float = 0.0
    before_round_data_depolarization: float = 0.0
    before_measure_flip_probability: float = 0.0
    after_reset_flip_probability: float = 0.0


def memory_circuit(
    code_task: str,
    rounds: int,
    distance: int,
    probability: float,
    noise: Optional[StimNoise] = None,
) -> stim.Circuit:
    """Stim's generated memory circuit.

    One probability on all four noise channels, as Stim's guide does,
    or noise's four channels when it is given.
    """
    if noise is None:
        noise = StimNoise(probability, probability, probability, probability)
    channels = dataclasses.asdict(noise)
    return stim.Circuit.generated(
        code_task, rounds=rounds, distance=distance, **channels
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
